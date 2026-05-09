import os
import glob
import mne
import numpy as np
from mne.decoding import CSP
from mne.time_frequency import psd_array_welch
from scipy.signal import butter, filtfilt
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

MRP_WINDOWS = [
    (-2.0, -1.5),
    (-1.5, -1.0),
    (-1.0, -0.5),
    (-0.5,  0.0),
    ( 0.0,  0.5),
    ( 0.5,  1.0),
    ( 1.0,  1.5),
    ( 1.5,  2.0),
]

MRP_SLOPE_WINDOW = (-0.5, 0.5)

def compute_band_power(X, sfreq, bands=FREQ_BANDS):

    #  X (epochs, channels, times)
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

def _window_mean(data, times, tmin, tmax):
    mask = (times >= tmin) & (times <= tmax)
    return data[:, :, mask].mean(axis=-1)  # (n_epochs, n_channels)

def extract_mrp_features(epochs, picks=SELECTED_CHANNELS):
    data_t1 = epochs['T1'].get_data(picks=picks)
    data_t2 = epochs['T2'].get_data(picks=picks)
    X_raw = np.concatenate([data_t1, data_t2], axis=0)  # (n_epochs, n_channels, n_times)
    y = np.concatenate([np.zeros(len(data_t1)), np.ones(len(data_t2))])
    times = epochs.times

    features = []

    # 1. Mean amplitude
    window_means = []
    for tmin, tmax in MRP_WINDOWS:
        wm = _window_mean(X_raw, times, tmin, tmax)
        window_means.append(wm)
        features.append(wm)

    # 2. Lateralization per window (C4-C3, CP2-CP1, FC2-FC1)
    left_indices  = [left  for left, right in CHANNEL_PAIRS]
    right_indices = [right for left, right in CHANNEL_PAIRS]
    for wm in window_means:
        lat = wm[:, right_indices] - wm[:, left_indices]
        features.append(lat)

    # 3. Slope
    slope_mask = (times >= MRP_SLOPE_WINDOW[0]) & (times <= MRP_SLOPE_WINDOW[1])
    t_slope = times[slope_mask]
    t_c = t_slope - t_slope.mean()
    t_var = (t_c ** 2).sum()
    data_slope = X_raw[:, :, slope_mask]  # (epochs, channels, times)
    x_c = data_slope - data_slope.mean(axis=-1, keepdims=True)
    slopes = (x_c * t_c).sum(axis=-1) / t_var
    features.append(slopes)

    X = np.hstack(features)
    return X, y


def save_mrp_features(input_dir, output_dir, file_pattern=cfg.FILE_PATTERN,
                      picks=SELECTED_CHANNELS):

    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
    print(f"Found {len(file_list)} files in {input_dir}")

    for file_path in file_list:
        fname = os.path.basename(file_path)
        subject_run = fname.replace('.edf-epo.fif', '')
        output_path = os.path.join(output_dir, f"{subject_run}_mrp.npz")

        epochs = mne.read_epochs(file_path, preload=True, verbose=False)
        X, y = extract_mrp_features(epochs, picks)

        np.savez(output_path, X=X, y=y)
        print(f"  Saved: {os.path.basename(output_path)}  shape: {X.shape}")


# =============================================================================
# ERD/ERS v2 — CSP + time-bins + log-ratio
# =============================================================================

_BASELINE_WINDOW = (-2.0,  0.0)
_TASK_WINDOW     = ( 0.0,  2.0)
_TIME_BINS       = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
_N_CSP           = 4


def _bandpass(X, sfreq, fmin, fmax, order=4):
    nyq = sfreq / 2.0
    b, a = butter(order, [fmin / nyq, fmax / nyq], btype='band')
    return filtfilt(b, a, X, axis=-1)


def _apply_csp_spatial(X, csp):
    filters = csp.filters_[:csp.n_components]   # (n_components, n_channels)
    return np.einsum('fc,ect->eft', filters, X)


def _slice_time(X, times, tmin, tmax):
    mask = (times >= tmin) & (times <= tmax)
    return X[:, :, mask]


def _mean_power(X):
    return np.mean(X ** 2, axis=-1)


def _epochs_to_raw_array(epochs, picks):
    d1 = epochs['T1'].get_data(picks=picks)
    d2 = epochs['T2'].get_data(picks=picks)
    X  = np.concatenate([d1, d2], axis=0)
    y  = np.concatenate([np.zeros(len(d1)), np.ones(len(d2))])
    return X, y


def fit_csp_filters(epochs_train, picks=SELECTED_CHANNELS,
                    bands=FREQ_BANDS, n_components=_N_CSP):

    sfreq = epochs_train.info['sfreq']
    times = epochs_train.times
    X, y  = _epochs_to_raw_array(epochs_train, picks)

    task_mask = (times >= _TASK_WINDOW[0]) & (times <= _TASK_WINDOW[1])
    X_task    = X[:, :, task_mask]

    csp_filters = {}
    for band_name, (fmin, fmax) in bands.items():
        X_bp = _bandpass(X_task, sfreq, fmin, fmax)
        csp  = CSP(n_components=n_components, reg='ledoit_wolf',
                   log=False, norm_trace=False)
        csp.fit(X_bp, y)
        csp_filters[band_name] = csp
        print(f"  CSP fitted: {band_name} ({fmin}–{fmax} Hz)")

    return csp_filters


def extract_erd_ers_v2(epochs, csp_filters, picks=SELECTED_CHANNELS,
                        bands=FREQ_BANDS):

    sfreq    = epochs.info['sfreq']
    times    = epochs.times
    X_raw, y = _epochs_to_raw_array(epochs, picks)

    band_blocks = []

    for band_name, (fmin, fmax) in bands.items():
        csp     = csp_filters[band_name]
        x_bp    = _bandpass(X_raw, sfreq, fmin, fmax)
        x_csp   = _apply_csp_spatial(x_bp, csp)         # (epochs, comp, times)

        x_base  = _slice_time(x_csp, times, *_BASELINE_WINDOW)
        p_base  = _mean_power(x_base)                    # (epochs, comp)

        bin_blocks = []
        for t_start, t_end in _TIME_BINS:
            x_bin     = _slice_time(x_csp, times, t_start, t_end)
            p_bin     = _mean_power(x_bin)               # (epochs, comp)
            log_ratio = np.log(p_bin + 1e-10) - np.log(p_base + 1e-10)
            bin_blocks.append(log_ratio)

        band_blocks.append(np.hstack(bin_blocks))        # (epochs, comp*4)

    return np.hstack(band_blocks), y                     # (epochs, 48)


def save_erd_ers_features_v2(input_dir, output_dir, file_pattern=cfg.FILE_PATTERN,
                               picks=None, bands=None):
    if picks is None:
        picks = SELECTED_CHANNELS
    if bands is None:
        bands = FREQ_BANDS

    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
    print(f"Found {len(file_list)} files in {input_dir}")

    all_epochs = [mne.read_epochs(f, preload=True, verbose=False) for f in file_list]

    sfreqs       = [ep.info['sfreq'] for ep in all_epochs]
    target_sfreq = max(set(sfreqs), key=sfreqs.count)
    all_epochs   = [
        ep.resample(target_sfreq) if ep.info['sfreq'] != target_sfreq else ep
        for ep in all_epochs
    ]

    epochs_all  = mne.concatenate_epochs(all_epochs)
    csp_filters = fit_csp_filters(epochs_all, picks=picks, bands=bands)

    # extract and save per file
    for file_path, epochs in zip(file_list, all_epochs):
        fname       = os.path.basename(file_path)
        subject_run = fname.replace('.edf-epo.fif', '')
        output_path = os.path.join(output_dir, f"{subject_run}_erd_ers.npz")

        features, y = extract_erd_ers_v2(epochs, csp_filters, picks=picks, bands=bands)
        np.savez(output_path, X=features, y=y)
        print(f"  Saved: {os.path.basename(output_path)}  shape: {features.shape}")
