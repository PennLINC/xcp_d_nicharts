"""Workflows for applying individualized parcellations to postprocessed data."""
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d.workflows.connectivity import (
    init_functional_connectivity_cifti_wf,
    init_functional_connectivity_nifti_wf,
)


def init_apply_custom_parcellations_wf(
    xcpd_dir,
    output_dir,
    min_coverage,
    mem_gb,
    omp_nthreads,
    name="apply_custom_parcellations_wf",
):
    """Apply custom parcellations to XCP-D postprocessed data.

    1. Collect postprocessed files from XCP-D.
        -   Denoised BOLD.
        -   Brain mask.
    2.  Load the requested atlases.
    3.  Warp the atlases to the same space as the BOLD data.
        -   I can pull the warping code from init_load_atlases_wf.
    4.  Run the connectivity workflow.
        -   I might need to make reho optional as well.
    5.  Write out the derivatives.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "bold_mask",
                "denoised_bold",
                "alff",  # may be Undefined
                "reho",
                "atlas_names",
                "atlas_files",
                "atlas_labels_files",
            ],
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "coverage_ciftis",
                "timeseries_ciftis",
                "correlation_ciftis",
                "coverage",
                "timeseries",
                "correlations",
                "connectplot",
                "parcellated_alff",
                "parcellated_reho",
            ],
        ),
        name="outputnode",
    )

    connectivity_wf = init_functional_connectivity_nifti_wf(
        output_dir=output_dir,
        alff_available=False,
        min_coverage=min_coverage,
        mem_gb=mem_gb,
        omp_nthreads=omp_nthreads,
        name="connectivity_wf",
    )

    # fmt:off
    workflow.connect([
        (inputnode, connectivity_wf, [
            ("name_source", "name_source"),
            ("bold_mask", "bold_mask"),
            ("denoised_bold", "denoised_bold"),
            ("atlas_names", "atlas_names"),
            ("atlas_files", "atlas_files"),
            ("atlas_labels_files", "atlas_labels_files"),
        ]),
        (connectivity_wf, outputnode, [
            ("coverage_ciftis", "coverage_ciftis"),
            ("timeseries_ciftis", "timeseries_ciftis"),
            ("correlation_ciftis", "correlation_ciftis"),
            ("coverage", "coverage"),
            ("timeseries", "timeseries"),
            ("correlations", "correlations"),
            ("connectplot", "connectplot"),
            ("parcellated_alff", "parcellated_alff"),
            ("parcellated_reho", "parcellated_reho"),
        ]),
    ])
    # fmt:on

    return workflow
