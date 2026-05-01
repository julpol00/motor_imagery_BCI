import os
import glob
import mne
import numpy as np
from mne.time_frequency import psd_array_welch
import src.config as cfg

FREQ_BANDS = {
    'mu_low':       (8,  10),
    'mu_high':      (10, 12),
    'beta_rebound': (18, 22),
}

SELECTED_CHANNELS = ['C3', 'C4', 'CP1', 'CP2', 'FC1', 'FC2']

CHANNEL_PAIRS = [
    (SELECTED_CHANNELS.index('C3'),  SELECTED_CHANNELS.index('C4')),
    (SELECTED_CHANNELS.index('CP1'), SELECTED_CHANNELS.index('CP2')),
    (SELECTED_CHANNELS.index('FC1'), SELECTED_CHANNELS.index('FC2')),
]


def compute_band_power(X, sfreq, bands=FREQ_BANDS):

    #  X: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, _ = X.shape
    band_powers = np.zeros((n_epochs, n_channels, len(bands)))

    for i, (fmin, fmax) in enumerate(bands.values()):
        psds, _ = psd_array_welch(X, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
        band_powers[:, :, i] = psds.mean(axis=-1)

    return band_powers


def compute_lateralization(band_powers, channel_pairs=CHANNEL_PAIRS):

    lat_list = []
    for left_idx, right_idx in channel_pairs:
        left  = band_powers[:, left_idx,  :]
        right = band_powers[:, right_idx, :]
        lat_list.append((right - left) / (right + left + 1e-10))
    return np.hstack(lat_list)


def extract_erd_ers_from_epochs(epochs, picks=SELECTED_CHANNELS, t_min=0.2, t_max=2.0):

    sfreq = epochs.info['sfreq']

    data_t1 = epochs['T1'].get_data(picks=picks, tmin=t_min, tmax=t_max)
    data_t2 = epochs['T2'].get_data(picks=picks, tmin=t_min, tmax=t_max)

    X_raw = np.concatenate([data_t1, data_t2], axis=0)
    y = np.concatenate([np.zeros(len(data_t1)), np.ones(len(data_t2))])

    band_powers = compute_band_power(X_raw, sfreq)
    lateralization = compute_lateralization(band_powers)

    bp_log_flat = np.log(band_powers + 1e-10).reshape(len(X_raw), -1)
    X = np.hstack([bp_log_flat, lateralization])

    return X, y


def save_erd_ers_features(input_dir, output_dir, file_pattern=cfg.FILE_PATTERN,
                          picks=SELECTED_CHANNELS, t_min=0.2, t_max=2.0):

    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
    print(f"Found {len(file_list)} files in {input_dir}")

    for file_path in file_list:
        fname = os.path.basename(file_path)
        subject_run = fname.replace('.edf-epo.fif', '')
        output_path = os.path.join(output_dir, f"{subject_run}_erd_ers.npz")

        epochs = mne.read_epochs(file_path, preload=True, verbose=False)
        X, y = extract_erd_ers_from_epochs(epochs, picks, t_min, t_max)

        np.savez(output_path, X=X, y=y)
        print(f"  Saved: {os.path.basename(output_path)}  shape: {X.shape}")
