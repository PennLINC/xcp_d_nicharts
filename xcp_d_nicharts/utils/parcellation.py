"""Parcellate and correlate BOLD data."""
from xcp_d.workflows.connectivity import (
    init_functional_connectivity_nifti_wf,
    init_load_atlases_wf,
)


def parcellate_with_custom_atlases(xcpd_dir, atlas_paths):
    """Parcellate denoised XCP-D derivatives with custom atlases.

    1.  Find the denoised BOLD files.
    2.  Find the atlases and associated label files.
    3.  Check that the atlases are in the same space as the BOLD files.
    4.  Run the functional connectivity workflow to extract coverage, time series, and correlation
        matrices.
    5.  Output everything to the XCP-D directory?
    """
    ...


def parcellate_with_xcpd_atlases(bold_file):
    """Parcellate data with XCP-D's atlases.

    1.  Load XCP-D's atlases.
    2.  Run the functional connectivity workflow to extract coverage, time series, and correlation
        matrices.
    """
    ...
