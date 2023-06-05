# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The primary workflows for xcp_d."""
import os
import sys
from copy import deepcopy

import bids
import matplotlib
import nibabel as nb
import nilearn
import numpy as np
import scipy
import templateflow
from nipype import __version__ as nipype_ver
from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d.__about__ import __version__
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.report import AboutSummary, SubjectSummary
from xcp_d.utils.bids import (
    _get_tr,
    collect_data,
    collect_run_data,
    collect_surface_data,
    get_entity,
    get_preproc_pipeline_info,
    group_across_runs,
    write_dataset_description,
)
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.modified_data import flag_bad_run
from xcp_d.utils.utils import estimate_brain_radius
from xcp_d.workflows.anatomical import (
    init_postprocess_anat_wf,
    init_postprocess_surfaces_wf,
)
from xcp_d.workflows.bold import init_postprocess_nifti_wf
from xcp_d.workflows.cifti import init_postprocess_cifti_wf
from xcp_d.workflows.concatenation import init_concatenate_data_wf

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_xcpd_ukb_wf(
    fmri_dir,
    output_dir,
    work_dir,
    subject_list,
    analysis_level,
    bids_filters,
    omp_nthreads,
    layout=None,
    min_coverage=0.5,
    name="xcpd_ukb_wf",
):
    """Build and organize execution of xcp_d pipeline.

    It also connects the subworkflows under the xcp_d workflow.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            import os
            import tempfile

            from xcp_d.workflows.base import init_xcpd_wf
            from xcp_d.utils.doc import download_example_data

            fmri_dir = download_example_data()
            out_dir = tempfile.mkdtemp()

            # Create xcp_d derivatives folder.
            os.mkdir(os.path.join(out_dir, "xcp_d"))

            wf = init_xcpd_wf(
                fmri_dir=fmri_dir,
                output_dir=out_dir,
                work_dir=".",
                subject_list=["01"],
                analysis_level="participant",
                task_id="imagery",
                bids_filters=None,
                bandpass_filter=True,
                high_pass=0.01,
                low_pass=0.08,
                bpf_order=2,
                fd_thresh=0.3,
                motion_filter_type=None,
                motion_filter_order=4,
                band_stop_min=12,
                band_stop_max=20,
                despike=True,
                head_radius=50.,
                params="36P",
                smoothing=6,
                custom_confounds_folder=None,
                dummy_scans=0,
                cifti=False,
                omp_nthreads=1,
                layout=None,
                process_surfaces=False,
                dcan_qc=False,
                input_type="fmriprep",
                min_coverage=0.5,
                min_time=100,
                combineruns=False,
                name="xcpd_wf",
            )

    Parameters
    ----------
    %(layout)s
    %(bandpass_filter)s
    %(high_pass)s
    %(low_pass)s
    %(despike)s
    %(bpf_order)s
    %(analysis_level)s
    %(motion_filter_type)s
    %(motion_filter_order)s
    %(band_stop_min)s
    %(band_stop_max)s
    %(omp_nthreads)s
    %(cifti)s
    task_id : :obj:`str` or None
        Task ID of BOLD  series to be selected for postprocess , or ``None`` to postprocess all
    bids_filters : dict or None
    %(output_dir)s
    %(fd_thresh)s
    run_uuid : :obj:`str`
        Unique identifier for execution instance
    subject_list : list
        List of subject labels
    %(work_dir)s
    %(head_radius)s
    %(params)s
    %(smoothing)s
    %(custom_confounds_folder)s
    %(dummy_scans)s
    %(process_surfaces)s
    %(dcan_qc)s
    %(input_type)s
    %(min_coverage)s
    %(min_time)s
    combineruns
    %(name)s

    References
    ----------
    .. footbibliography::
    """
    xcpd_wf = Workflow(name="xcpd_wf")
    xcpd_wf.base_dir = work_dir
    LOGGER.info(f"Beginning the {name} workflow")

    write_dataset_description(fmri_dir, os.path.join(output_dir, "xcp_d"))

    for subject_id in subject_list:
        single_subj_wf = init_subject_wf(
            layout=layout,
            fmri_dir=fmri_dir,
            omp_nthreads=omp_nthreads,
            subject_id=subject_id,
            bids_filters=bids_filters,
            output_dir=output_dir,
            min_coverage=min_coverage,
            name=f"single_subject_{subject_id}_wf",
        )

        single_subj_wf.config["execution"]["crashdump_dir"] = os.path.join(
            output_dir,
            "xcp_d",
            f"sub-{subject_id}",
            "log",
        )
        for node in single_subj_wf._get_all_nodes():
            node.config = deepcopy(single_subj_wf.config)
        print(f"Analyzing data at the {analysis_level} level")
        xcpd_wf.add_nodes([single_subj_wf])

    return xcpd_wf


@fill_doc
def init_subject_wf(
    fmri_dir,
    subject_id,
    bids_filters,
    output_dir,
    min_coverage,
    omp_nthreads,
    layout,
    name,
):
    """Organize the postprocessing pipeline for a single subject.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.base import init_subject_wf
            from xcp_d.utils.doc import download_example_data

            fmri_dir = download_example_data()

            wf = init_subject_wf(
                fmri_dir=fmri_dir,
                subject_id="01",
                input_type="fmriprep",
                process_surfaces=False,
                combineruns=False,
                cifti=False,
                task_id="imagery",
                bids_filters=None,
                bandpass_filter=True,
                high_pass=0.01,
                low_pass=0.08,
                bpf_order=2,
                motion_filter_type=None,
                motion_filter_order=4,
                band_stop_min=12,
                band_stop_max=20,
                smoothing=6.,
                head_radius=50,
                params="36P",
                output_dir=".",
                custom_confounds_folder=None,
                dummy_scans=0,
                fd_thresh=0.3,
                despike=True,
                dcan_qc=False,
                min_coverage=0.5,
                min_time=100,
                omp_nthreads=1,
                layout=None,
                name="single_subject_sub-01_wf",
            )

    Parameters
    ----------
    %(fmri_dir)s
    %(subject_id)s
    %(input_type)s
    %(process_surfaces)s
    combineruns
    %(cifti)s
    task_id : :obj:`str` or None
        Task ID of BOLD  series to be selected for postprocess , or ``None`` to postprocess all
    bids_filters : dict or None
    %(bandpass_filter)s
    %(high_pass)s
    %(low_pass)s
    %(bpf_order)s
    %(motion_filter_type)s
    %(motion_filter_order)s
    %(band_stop_min)s
    %(band_stop_max)s
    %(smoothing)s
    %(head_radius)s
    %(params)s
    %(output_dir)s
    %(custom_confounds_folder)s
    %(dummy_scans)s
    %(fd_thresh)s
    %(despike)s
    %(dcan_qc)s
    %(min_coverage)s
    %(min_time)s
    %(omp_nthreads)s
    %(layout)s
    %(name)s

    References
    ----------
    .. footbibliography::
    """
    subj_data = collect_ukb_data(
        bids_dir=fmri_dir,
        participant_label=subject_id,
        bids_filters=bids_filters,
        bids_validate=False,
    )

    # determine the appropriate post-processing workflow
    init_postprocess_bold_wf = init_postprocess_cifti_wf if cifti else init_postprocess_nifti_wf
    preproc_files = subj_data["bold"]

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "subj_data",  # not currently used, but will be in future
                "t1w",
                "t2w",  # optional
                "anat_brainmask",  # not used by cifti workflow
                "anat_dseg",
                "template_to_anat_xfm",  # not used by cifti workflow
                "anat_to_template_xfm",
                # mesh files
                "lh_pial_surf",
                "rh_pial_surf",
                "lh_wm_surf",
                "rh_wm_surf",
                # shape files
                "lh_sulcal_depth",
                "rh_sulcal_depth",
                "lh_sulcal_curv",
                "rh_sulcal_curv",
                "lh_cortical_thickness",
                "rh_cortical_thickness",
            ],
        ),
        name="inputnode",
    )
    inputnode.inputs.subj_data = subj_data
    inputnode.inputs.t1w = subj_data["t1w"]
    inputnode.inputs.t2w = subj_data["t2w"]
    inputnode.inputs.anat_brainmask = subj_data["anat_brainmask"]
    inputnode.inputs.anat_dseg = subj_data["anat_dseg"]
    inputnode.inputs.template_to_anat_xfm = subj_data["template_to_anat_xfm"]
    inputnode.inputs.anat_to_template_xfm = subj_data["anat_to_template_xfm"]

    workflow = Workflow(name=name)

    workflow.__desc__ = f"""
### Post-processing of UK Biobank outputs
The eXtensible Connectivity Pipeline- DCAN (XCP-D) [@mitigating_2018;@satterthwaite_2013]
was used to post-process the outputs of UK Biobank.
XCP-D was built with *Nipype* version {nipype_ver} [@nipype1, RRID:SCR_002502].
"""

    workflow.__postdesc__ = f"""

Many internal operations of *XCP-D* use
*AFNI* [@cox1996afni;@cox1997software],
*ANTS* [@avants2009advanced],
*TemplateFlow* version {templateflow.__version__} [@ciric2022templateflow],
*matplotlib* version {matplotlib.__version__} [@hunter2007matplotlib],
*Nibabel* version {nb.__version__} [@brett_matthew_2022_6658382],
*Nilearn* version {nilearn.__version__} [@abraham2014machine],
*numpy* version {np.__version__} [@harris2020array],
*pybids* version {bids.__version__} [@yarkoni2019pybids],
and *scipy* version {scipy.__version__} [@2020SciPy-NMeth].
For more details, see the *XCP-D* website (https://xcp-d.readthedocs.io).


#### Copyright Waiver

The above methods description text was automatically generated by *XCP-D*
with the express intention that users should copy and paste this
text into their manuscripts *unchanged*.
It is released under the [CC0](https://creativecommons.org/publicdomain/zero/1.0/) license.

#### References

"""

    summary = pe.Node(
        SubjectSummary(subject_id=subject_id, bold=preproc_files),
        name="summary",
    )

    about = pe.Node(
        AboutSummary(version=__version__, command=" ".join(sys.argv)),
        name="about",
    )

    ds_report_summary = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=preproc_files[0],
            datatype="figures",
            desc="summary",
        ),
        name="ds_report_summary",
    )

    ds_report_about = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=preproc_files[0],
            desc="about",
            datatype="figures",
        ),
        name="ds_report_about",
        run_without_submitting=True,
    )

    # Extract target volumetric space for T1w image
    target_space = get_entity(subj_data["anat_to_template_xfm"], "to")

    # Extract global signal from BOLD file

    # Run connectivity workflow

    n_runs = len(preproc_files)
    for i_run, bold_file in enumerate(preproc_files):
        run_data = collect_run_data(fmri_dir, bold_file)

        postprocess_bold_wf = init_postprocess_ukbiobank_wf(
            bold_file=bold_file,
            output_dir=output_dir,
            run_data=run_data,
            n_runs=n_runs,
            min_coverage=min_coverage,
            omp_nthreads=omp_nthreads,
            layout=layout,
            name=f"postprocess_ukbiobank_{i_run}_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, postprocess_bold_wf, [("bold_file", "inputnode.bold_file")]),
        ])
        # fmt:on

    # fmt:off
    workflow.connect([
        (summary, ds_report_summary, [("out_report", "in_file")]),
        (about, ds_report_about, [("out_report", "in_file")]),
    ])
    # fmt:on

    for node in workflow.list_node_names():
        if node.split(".")[-1].startswith("ds_"):
            workflow.get_node(node).interface.out_path_base = "xcp_d"

    return workflow
