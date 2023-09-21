"""Functions for working with UKB data.

UK Biobank data are already denoised and in standard space,
but we want to additionally run global signal regression on these data,
before running XCP-D's parcellation code on them.
"""
import os

import numpy as np
from nilearn import masking


def collect_ukb_data(ukb_dir):
    """Collect necessary files from a UK Biobank dataset."""
    subj_data = {
        "bold": os.path.join(ukb_dir, "fMRI", "rfMRI.ica", "filtered_func_data_clean.nii.gz"),
        "brainmask": os.path.join(ukb_dir, "fMRI", "rfMRI.ica", "mask.nii.gz"),
        "t1w": os.path.join(ukb_dir, "T1", "T1_brain.nii.gz"),
        "motion": os.path.join(
            ukb_dir,
            "fMRI",
            "rfMRI.ica",
            "mc",
            "prefiltered_func_data_mcf.par",
        ),
    }
    return subj_data


def create_regression_confounds(bold_file, mask_file):
    """Create global signal confounds file."""
    import os

    import pandas as pd
    from xcp_d.utils.ingestion import extract_mean_signal

    confounds_file = os.path.abspath("confounds.tsv")

    # Extract global signal from BOLD file
    mean_gs = extract_mean_signal(
        mask=mask_file,
        nifti=bold_file,
        work_dir=os.getcwd(),
    )

    confounds_df = pd.DataFrame(columns=["global_signal"], data=mean_gs)

    # Write out the confounds file
    confounds_df.to_csv(confounds_file, sep="\t", index=False)
    return confounds_file


def create_motion_confounds(motion):
    """Create motion confounds file."""
    import json
    import os

    import pandas as pd

    confounds_file = os.path.abspath("motion.tsv")
    confounds_json = os.path.abspath("motion.json")

    motion_confounds_df = pd.read_table(motion)
    columns = motion_confounds_df.columns.tolist()
    for col in columns:
        new_col = f"{col}_derivative1"
        motion_confounds_df[new_col] = motion_confounds_df[col].diff()

    columns = motion_confounds_df.columns.tolist()
    for col in columns:
        new_col = f"{col}_power2"
        motion_confounds_df[new_col] = motion_confounds_df[col] ** 2

    # Add empty FD column
    motion_confounds_df["framewise_displacement"] = 0
    confounds_dict = {col: {"Description": ""} for col in motion_confounds_df.columns}

    motion_confounds_df.to_csv(confounds_file, sep="\t", index=False)
    with open(confounds_json, "w") as fo:
        json.dump(confounds_dict, fo)

    return confounds_file, confounds_json


def global_signal_regression(bold_file, mask_file, out_file):
    """Perform global signal regression.

    TODO: Check against fMRIPrep's method for extracting global signal.
    """
    bold_data = masking.apply_mask(bold_file, mask_file)
    global_signal = np.mean(bold_data, axis=1)
    global_signal_mean_centered = global_signal - np.mean(global_signal)

    # Estimate betas using only the censored data
    betas = np.linalg.lstsq(global_signal_mean_centered, bold_data, rcond=None)[0]

    # Apply the betas to denoise the *full* (uncensored) BOLD data
    denoised_bold = bold_data - np.dot(global_signal_mean_centered, betas)

    denoised_img = masking.unmask(denoised_bold, mask_file)
    denoised_img.to_filename(out_file)
    return out_file
