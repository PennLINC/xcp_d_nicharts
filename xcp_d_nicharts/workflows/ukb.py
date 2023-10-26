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
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.censoring import GenerateConfounds
from xcp_d.utils.bids import write_dataset_description
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import estimate_brain_radius
from xcp_d.workflows.connectivity import (
    init_functional_connectivity_nifti_wf,
    init_load_atlases_wf,
)
from xcp_d.workflows.postprocessing import init_denoise_bold_wf, init_despike_wf
from xcp_d.workflows.restingstate import init_alff_wf, init_reho_nifti_wf

from xcp_d_nicharts.utils.ukb import (
    collect_ukb_data,
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
    omp_nthreads,
    motion_filter_type,
    motion_filter_order,
    band_stop_min,
    band_stop_max,
    despike,
    min_coverage,
    fd_thresh,
    low_pass,
    high_pass,
    bpf_order,
    smoothing,
    name="xcpd_ukb_wf",
):
    """Build and organize execution of xcp_d pipeline.

    It also connects the subworkflows under the xcp_d workflow.

    Parameters
    ----------
    fmri_dir
    output_dir
    work_dir
    subject_list
    analysis_level
    omp_nthreads
    motion_filter_type
    motion_filter_order
    band_stop_min
    band_stop_max
    despike
    min_coverage
    fd_thresh
    low_pass
    high_pass
    smoothing
    name
    """
    xcpd_wf = Workflow(name="xcpd_wf")
    xcpd_wf.base_dir = work_dir
    LOGGER.info(f"Beginning the {name} workflow")

    write_dataset_description(fmri_dir, os.path.join(output_dir, "xcp_d"))

    for subject_id in subject_list:
        single_subj_wf = init_subject_wf(
            fmri_dir=fmri_dir,
            subject_id=subject_id,
            output_dir=output_dir,
            motion_filter_type=motion_filter_type,
            motion_filter_order=motion_filter_order,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            despike=despike,
            min_coverage=min_coverage,
            fd_thresh=fd_thresh,
            low_pass=low_pass,
            high_pass=high_pass,
            bpf_order=bpf_order,
            smoothing=smoothing,
            omp_nthreads=omp_nthreads,
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


def init_subject_wf(
    fmri_dir,
    subject_id,
    output_dir,
    motion_filter_type,
    motion_filter_order,
    band_stop_min,
    band_stop_max,
    despike,
    min_coverage,
    fd_thresh,
    low_pass,
    high_pass,
    bpf_order,
    smoothing,
    omp_nthreads,
    name,
):
    """Organize the postprocessing pipeline for a single subject.

    Parameters
    ----------
    fmri_dir
    subject_id
    output_dir
    motion_filter_type
    motion_filter_order
    band_stop_min
    band_stop_max
    despike
    min_coverage
    fd_thresh
    low_pass
    high_pass
    bpf_order
    smoothing
    omp_nthreads
    name
    """
    name_source = (
        f"/dset/sub-{subject_id}/func/"
        f"sub-{subject_id}_task-rest_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz"
    )
    t_r = 0.8  # TODO: FIND CORRECT TR

    workflow = Workflow(name=name)

    subj_data = collect_ukb_data(ukb_dir=os.path.join(fmri_dir, subject_id))

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "bold_file",
                "t1w",
                "brainmask",
                "par_file",
            ],
        ),
        name="inputnode",
    )
    inputnode.inputs.name_source = name_source
    inputnode.inputs.bold_file = subj_data["bold"]
    inputnode.inputs.t1w = subj_data["t1w"]
    inputnode.inputs.brainmask = subj_data["brainmask"]
    inputnode.inputs.par_file = subj_data["motion"]

    # Load the atlases, warping to the same space as the BOLD data if necessary.
    load_atlases_wf = init_load_atlases_wf(
        output_dir=output_dir,
        cifti=False,
        mem_gb=1,
        omp_nthreads=omp_nthreads,
        name="load_atlases_wf",
    )

    # fmt:off
    workflow.connect([
        (inputnode, load_atlases_wf, [
            ("name_source", "inputnode.name_source"),
            ("bold_file", "inputnode.bold_file"),
        ]),
    ])
    # fmt:on

    # Prepare motion parameters to produce temporal mask from filtered motion.
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
    workflow.connect([(add_motion_headers, make_motion_confounds, [("out_file", "motion")])])

    get_brain_radius = pe.Node(
        Function(
            input_names=["mask_file"],
            output_names=["head_radius"],
            function=estimate_brain_radius,
        ),
        name="get_brain_radius",
    )
    workflow.connect([(inputnode, get_brain_radius, [("brainmask", "mask_file")])])

    flag_motion_outliers = pe.Node(
        GenerateConfounds(
            in_file="",
            params="none",
            TR=t_r,
            fd_thresh=fd_thresh,
            custom_confounds_file=None,
            motion_filter_type=motion_filter_type,
            motion_filter_order=motion_filter_order,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
        ),
        name="flag_motion_outliers",
    )

    # fmt:off
    workflow.connect([
        (make_motion_confounds, flag_motion_outliers, [
            ("confounds_file", "fmriprep_confounds_file"),
            ("confounds_json", "fmriprep_confounds_json"),
        ]),
        (get_brain_radius, flag_motion_outliers, [("brain_radius", "head_radius")]),
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
            ("brainmask", "mask_file"),
        ]),
    ])
    # fmt:on

    # Denoise BOLD file with global signal
    denoise_bold_wf = init_denoise_bold_wf(
        TR=t_r,
        low_pass=low_pass,
        high_pass=high_pass,
        bpf_order=bpf_order,
        bandpass_filter=((low_pass != 0) or (high_pass != 0)),
        smoothing=smoothing,
        cifti=False,
        mem_gb=1,
        omp_nthreads=1,
        name="denoise_bold_wf",
    )

    if despike:
        despike_wf = init_despike_wf(
            TR=t_r,
            cifti=False,
            omp_nthreads=omp_nthreads,
            name="despike_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, despike_wf, [("bold_file", "inputnode.bold_file")]),
            (despike_wf, denoise_bold_wf, [
                ("outputnode.bold_file", "inputnode.preprocessed_bold"),
            ]),
        ])
        # fmt:on

    else:
        # fmt:off
        workflow.connect([
            (inputnode, denoise_bold_wf, [("bold_file", "inputnode.preprocessed_bold")]),
        ])
        # fmt:on

    # fmt:off
    workflow.connect([
        (inputnode, denoise_bold_wf, [("brainmask", "inputnode.mask")]),
        (make_regression_confounds, denoise_bold_wf, [
            ("confounds_file", "inputnode.confounds_file"),
        ]),
        (flag_motion_outliers, denoise_bold_wf, [("temporal_mask", "inputnode.temporal_mask")]),
    ])
    # fmt:on

    # Calculate ALFF and ReHo
    alff_wf = init_alff_wf(
        name_source=name_source,
        output_dir=output_dir,
        TR=t_r,
        low_pass=low_pass,
        high_pass=high_pass,
        smoothing=smoothing,
        cifti=False,
        name="alff_wf",
    )

    # fmt:off
    workflow.connect([
        (inputnode, alff_wf, [("brainmask", "inputnode.bold_mask")]),
        (denoise_bold_wf, alff_wf, [
            ("outputnode.censored_denoised_bold", "inputnode.denoised_bold"),
        ]),
    ])
    # fmt:on

    reho_wf = init_reho_nifti_wf(
        name_source=name_source,
        output_dir=output_dir,
        name="reho_wf",
    )

    # fmt:off
    workflow.connect([
        (inputnode, reho_wf, [("brainmask", "inputnode.bold_mask")]),
        (denoise_bold_wf, reho_wf, [
            ("outputnode.censored_denoised_bold", "inputnode.denoised_bold"),
        ]),
    ])
    # fmt:on

    # Run connectivity workflow
    connectivity_wf = init_functional_connectivity_nifti_wf(
        output_dir=output_dir,
        alff_available=True,
        min_coverage=min_coverage,
        mem_gb=1,
        name="connectivity_wf",
    )

    # fmt:off
    workflow.connect([
        (inputnode, connectivity_wf, [
            ("name_source", "inputnode.name_source"),
        ]),
        (load_atlases_wf, connectivity_wf, [
            ("outputnode.atlas_names", "inputnode.atlas_names"),
            ("outputnode.atlas_files", "inputnode.atlas_files"),
            ("outputnode.atlas_labels_files", "inputnode.atlas_labels_files"),
        ]),
        (denoise_bold_wf, connectivity_wf, [
            ("outputnode.censored_denoised_bold", "inputnode.denoised_bold"),
        ]),
        (flag_motion_outliers, connectivity_wf, [("temporal_mask", "inputnode.temporal_mask")]),
        (alff_wf, connectivity_wf, [("outputnode.alff", "inputnode.alff")]),
        (reho_wf, connectivity_wf, [("outputnode.reho", "inputnode.reho")]),
    ])
    # fmt:on

    # Write out derivatives
    ds_temporal_mask = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["atlas", "den", "res", "space", "cohort", "desc"],
            suffix="outliers",
            extension=".tsv",
            source_file=name_source,
        ),
        name="ds_temporal_mask",
        run_without_submitting=True,
        mem_gb=1,
    )

    # fmt:off
    workflow.connect([
        (flag_motion_outliers, ds_temporal_mask, [
            ("temporal_mask_metadata", "meta_dict"),
            ("temporal_mask", "in_file"),
        ]),
    ])
    # fmt:on

    ds_filtered_motion = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=name_source,
            dismiss_entities=["atlas", "den", "res", "space", "cohort", "desc"],
            desc="filtered" if motion_filter_type else None,
            suffix="motion",
            extension=".tsv",
        ),
        name="ds_filtered_motion",
        run_without_submitting=True,
        mem_gb=1,
    )

    # fmt:off
    workflow.connect([
        (flag_motion_outliers, ds_filtered_motion, [
            ("motion_metadata", "meta_dict"),
            ("filtered_motion", "in_file"),
        ]),
    ])
    # fmt:on

    ds_coverage_files = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=name_source,
            dismiss_entities=["desc"],
            suffix="coverage",
            extension=".tsv",
        ),
        name="ds_coverage_files",
        run_without_submitting=True,
        mem_gb=1,
        iterfield=["atlas", "in_file"],
    )
    ds_timeseries = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=name_source,
            dismiss_entities=["desc"],
            suffix="timeseries",
            extension=".tsv",
        ),
        name="ds_timeseries",
        run_without_submitting=True,
        mem_gb=1,
        iterfield=["atlas", "in_file"],
    )
    ds_correlations = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=name_source,
            dismiss_entities=["desc"],
            measure="pearsoncorrelation",
            suffix="conmat",
            extension=".tsv",
        ),
        name="ds_correlations",
        run_without_submitting=True,
        mem_gb=1,
        iterfield=["atlas", "in_file"],
    )
    ds_parcellated_reho = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=name_source,
            dismiss_entities=["desc"],
            suffix="reho",
            extension=".tsv",
        ),
        name="ds_parcellated_reho",
        run_without_submitting=True,
        mem_gb=1,
        iterfield=["atlas", "in_file"],
    )

    # fmt:off
    workflow.connect([
        (load_atlases_wf, ds_coverage_files, [("outputnode.atlas_names", "atlas")]),
        (connectivity_wf, ds_coverage_files, [("coverage", "in_file")]),
        (load_atlases_wf, ds_timeseries, [("outputnode.atlas_names", "atlas")]),
        (connectivity_wf, ds_timeseries, [("timeseries", "in_file")]),
        (load_atlases_wf, ds_correlations, [("outputnode.atlas_names", "atlas")]),
        (connectivity_wf, ds_correlations, [("correlations", "in_file")]),
        (load_atlases_wf, ds_parcellated_reho, [("outputnode.atlas_names", "atlas")]),
        (connectivity_wf, ds_parcellated_reho, [("parcellated_reho", "in_file")]),
    ])
    # fmt:on

    ds_denoised_bold = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=name_source,
            desc="denoised",
            extension=".nii.gz",
            compression=True,
        ),
        name="ds_denoised_bold",
        run_without_submitting=True,
        mem_gb=2,
    )

    ds_reho = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=name_source,
            dismiss_entities=["desc"],
            suffix="reho",
            extension=".nii.gz",
            compression=True,
        ),
        name="ds_reho",
        run_without_submitting=True,
        mem_gb=1,
    )

    # fmt:off
    workflow.connect([
        (denoise_bold_wf, ds_denoised_bold, [("outputnode.censored_denoised_bold", "in_file")]),
        (reho_wf, ds_reho, [("outputnode.reho", "in_file")]),
    ])
    # fmt:on

    for node in workflow.list_node_names():
        if node.split(".")[-1].startswith("ds_"):
            workflow.get_node(node).interface.out_path_base = "xcp_d"

    return workflow
