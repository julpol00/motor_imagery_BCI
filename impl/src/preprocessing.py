import mne as mne
import numpy as np
import os, glob
from autoreject import AutoReject
import config as cfg

def preprocess_file(file_path, n_jobs=cfg.N_JOBS, random_state=cfg.RANDOM_STATE):
  raw = mne.io.read_raw_edf(
        file_path,
        preload=True
  )

  # 1. rename channels
  raw.rename_channels(cfg.CHANNELS_NEW_NAMES_DICT)

  # 2. set montage
  raw.set_montage('standard_1020')

  # 3. re-reference: to average
  raw.set_eeg_reference(ref_channels='average')

  # 4. Notch filter
  nyquist_freq = raw.info['sfreq'] / 2

  raw.notch_filter(
      picks=['eeg'],
      freqs=np.arange(cfg.POWER_FREQ, nyquist_freq, cfg.POWER_FREQ),
      n_jobs=n_jobs,
  )

  # 5. Filters epochs
  raw_filtered = raw.copy().filter(
      picks=['eeg'],
      l_freq=0.05,
      h_freq=30.0,
      n_jobs=n_jobs,
      method='iir',
      iir_params=None
      )

  # 6. Segmentation
  event_dict = {'T1': 2, 'T2': 3} # T1 - left, T2 - right
  events, event_ids = mne.events_from_annotations(raw, event_id=event_dict)

  epochs = mne.Epochs(
      raw = raw_filtered,
      events=events,
      event_id=event_dict,
      tmin=-0.2,
      tmax=2.0,
      baseline=(-0.2, 0),
      reject_by_annotation=False,
      preload=True,
      reject=None,
      picks=['eeg'],
      on_missing = 'warn',
  )

  # 7. Artifact correction with autoreject
  ar = AutoReject(random_state=random_state, n_jobs=n_jobs, verbose=0)
  epochs_ar = ar.fit_transform(epochs, return_log=False)

  return epochs_ar

def saving_preprocessed_data(data_dir, output_dir):

    edf_files = [f for f in os.listdir(data_dir) if f.endswith(".edf")]

    for file_name in edf_files:
        edf_path = os.path.join(data_dir, file_name)

        pre_processed_epochs = preprocess_file(edf_path)
        output_file_path = os.path.join(output_dir, f"{file_name}-epo.fif")
        pre_processed_epochs.save(output_file_path)
