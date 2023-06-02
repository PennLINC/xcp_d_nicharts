"""Functions for working with UKB data.

UK Biobank data are already denoised and in standard space,
but we want to additionally run global signal regression on these data,
before running XCP-D's parcellation code on them.
"""
import numpy as np
from nilearn import masking


def collect_ukb_data(ukb_dir, participant_label):
    """Collect necessary files from a UK Biobank dataset."""
    subj_data = {
        "bold": "",
        "t1w": "",
        "confounds": "",
    }
    return subj_data


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
