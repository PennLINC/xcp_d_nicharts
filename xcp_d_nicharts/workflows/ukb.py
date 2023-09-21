# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for postprocessing UK Biobank data."""
import os
from copy import deepcopy

from nipype import Function, logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.confounds import NormalizeMotionParams
from niworkflows.interfaces.utility import AddTSVHeader
from xcp_d.interfaces.censoring import GenerateConfounds
from xcp_d.utils.bids import write_dataset_description
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import estimate_brain_radius
from xcp_d.workflows.connectivity import (
    init_functional_connectivity_nifti_wf,
    init_load_atlases_wf,
)
from xcp_d.workflows.postprocessing import init_denoise_bold_wf

from xcp_d_nicharts.utils.ukb import (
    collect_ukb_data,
    collect_ukb_run_data,
    create_motion_confounds,
    create_regression_confounds,
)

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

    # Extract target volumetric space for T1w image
    name_source = f"sub-{subject_id}_task-rest_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz"

    # Load the atlases, warping to the same space as the BOLD data if necessary.
    load_atlases_wf = init_load_atlases_wf(
        output_dir=output_dir,
        cifti=False,
        mem_gb=1,
        omp_nthreads=omp_nthreads,
        name="load_atlases_wf",
    )
    load_atlases_wf.inputs.inputnode.name_source = name_source
    load_atlases_wf.inputs.inputnode.bold_file = preproc_files[0]

    # Process each run
    for i_run, bold_file in enumerate(preproc_files):
        run_data = collect_ukb_run_data(fmri_dir, bold_file)

        postprocess_bold_wf = init_postprocess_ukbiobank_wf(
            bold_file=bold_file,
            output_dir=output_dir,
            run_data=run_data,
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

    for node in workflow.list_node_names():
        if node.split(".")[-1].startswith("ds_"):
            workflow.get_node(node).interface.out_path_base = "xcp_d"

    return workflow


def init_postprocess_ukbiobank_wf(
    bold_file,
    output_dir,
    run_data,
    min_coverage,
    fd_thresh,
    head_radius,
    name,
):
    """Postprocess UK Biobank BOLD data."""
    workflow = Workflow(name=name)

    t_r = run_data["bold_metadata"]["RepetitionTime"]

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "boldref",
                "bold_mask",
                "custom_confounds_file",
                "template_to_anat_xfm",
                "t1w",
                "t2w",
                "anat_dseg",
                "anat_brainmask",
                "atlas_names",
                "atlas_files",
                "atlas_labels_files",
            ],
        ),
        name="inputnode",
    )

    inputnode.inputs.bold_file = bold_file
    inputnode.inputs.boldref = run_data["boldref"]
    inputnode.inputs.bold_mask = run_data["boldmask"]
    inputnode.inputs.par_file = run_data["motion"]

    normalize_motion = pe.Node(
        NormalizeMotionParams(format="FSL"),
        name="normalize_motion",
    )
    workflow.connect([(inputnode, normalize_motion, [("par_file", "in_file")])])

    add_motion_headers = pe.Node(
        AddTSVHeader(columns=["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]),
        name="add_motion_headers",
        mem_gb=0.01,
        run_without_submitting=True,
    )
    workflow.connect([(normalize_motion, add_motion_headers, [("out_file", "in_file")])])

    make_motion_confounds = pe.Node(
        Function(
            input_names=["motion"],
            output_names=["confounds_file", "confounds_json"],
            function=create_motion_confounds,
        ),
        name="make_motion_confounds",
    )

    # fmt:off
    workflow.connect([
        (add_motion_headers, make_motion_confounds, [("out_file", "motion")]),
    ])
    # fmt:on

    # NOTE: Do we need to calculate FD and censor? Yes.
    # temporal_mask
    flag_motion_outliers = pe.Node(
        GenerateConfounds(
            in_file="",
            params="none",
            TR=t_r,
            fd_thresh=fd_thresh,
            head_radius=head_radius,
            custom_confounds_file=None,
            motion_filter_type="notch",
            motion_filter_order=4,
            band_stop_min=8,
            band_stop_max=12,
        ),
        name="flag_motion_outliers",
    )

    # fmt:off
    workflow.connect([
        (make_motion_confounds, flag_motion_outliers, [
            ("confounds_file", "fmriprep_confounds_file"),
            ("confounds_json", "fmriprep_confounds_json"),
        ]),
    ])
    # fmt:on

    make_regression_confounds = pe.Node(
        Function(
            input_names=["bold_file", "mask_file", "motion"],
            output_names=["confounds_file"],
            function=create_regression_confounds,
        ),
        name="make_regression_confounds",
    )

    # fmt:off
    workflow.connect([
        (inputnode, make_regression_confounds, [
            ("bold_file", "bold_file"),
            ("bold_mask", "mask_file"),
        ]),
    ])
    # fmt:on

    # Denoise BOLD file with global signal
    denoise_bold_wf = init_denoise_bold_wf(
        TR=t_r,
        low_pass=0,
        high_pass=0,
        bpf_order=0,
        bandpass_filter=False,
        smoothing=0,
        cifti=False,
        mem_gb=1,
        omp_nthreads=1,
        name="denoise_bold_wf",
    )
    denoise_bold_wf.inputs.inputnode.preprocessed_bold = bold_file
    denoise_bold_wf.inputs.inputnode.mask = run_data["bold_brainmask"]

    # fmt:off
    workflow.connect([
        (make_regression_confounds, denoise_bold_wf, [
            ("confounds_file", "inputnode.confounds_file"),
        ]),
        (flag_motion_outliers, denoise_bold_wf, [("temporal_mask", "inputnode.temporal_mask")]),
    ])
    # fmt:on

    # Run connectivity workflow
    connectivity_wf = init_functional_connectivity_nifti_wf(
        output_dir=output_dir,
        alff_available=False,
        min_coverage=min_coverage,
        mem_gb=1,
        name="connectivity_wf",
    )

    """name_source
    denoised_bold
    temporal_mask
    alff
    reho
    atlas_names
    atlas_files
    atlas_labels_files"""

    # fmt:off
    workflow.connect([
        (denoise_bold_wf, connectivity_wf, [
            ("outputnode.censored_denoised_bold", "inputnode.denoised_bold"),
        ]),
        (flag_motion_outliers, connectivity_wf, [("temporal_mask", "inputnode.temporal_mask")]),
    ])
    # fmt:on

    # Write out derivatives

    return workflow
