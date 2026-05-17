"""
Online BCI — EEG acquisition and classification.
Listens for LSL markers from paradigm.py and classifies each trial.

Usage:
  cd src
  python BCI.py

Start BCI.py first. After EEG is ready, open paradigm.py in PsychoPy.
"""
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import mne
import joblib
from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager
import config as cfg

try:
    from pylsl import StreamInlet, resolve_byprop
    LSL_AVAILABLE = True
except Exception as e:
    LSL_AVAILABLE = False
    print(f"[LSL] Could not load pylsl ({e}) — running in manual mode")

# ── MOTION TYPE SELECTION ─────────────────────────────────────────────────────
while True:
    motion = input("Motion type — enter 'imagery' or 'real': ").strip().lower()
    if motion in ('imagery', 'real'):
        break
    print("  Invalid input. Type 'imagery' or 'real'.")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATHS = {
    'imagery': r'D:\inżynierka\motor imagery BCI\impl\models\bci_model_svc_base_imagery_motion.pkl',
    'real':    r'D:\inżynierka\motor imagery BCI\impl\models\bci_model_svc_base_real_motion.pkl',
}
RESULTS_DIR  = Path(__file__).parent.parent / 'results'
DEVICE_NAME  = "BA MINI 039"
CAP          = {2: "C3", 3: "C4", 4: "CP1", 5: "CP2", 0: "FC1", 1: "FC2"}
SFREQ_IN     = 250    # Hz — device sampling rate
SFREQ_TARGET = 160    # Hz — resampled (matches training)
NUM_TRIALS   = 50
EPOCH_DUR    = 2.2    # s of EEG captured from imagery onset; model uses 0.0-2.0 s

MARKER_LEFT  = 1      # marker from paradigm.py -> label 0
MARKER_RIGHT = 2      # marker from paradigm.py -> label 1

# ── MODEL ─────────────────────────────────────────────────────────────────────
model = joblib.load(MODEL_PATHS[motion])
print(f"[MODEL] Loaded: {MODEL_PATHS[motion]}")
print(f"[CONFIG] Motion type: {motion.upper()}\n")


# ── PREPROCESSING ─────────────────────────────────────────────────────────────
def preprocess_online(raw: mne.io.Raw) -> np.ndarray:
    raw.set_eeg_reference(ref_channels='average', verbose=False)
    raw.notch_filter(freqs=50.0, picks=['eeg'], verbose=False)
    raw.filter(
        picks=['eeg'], l_freq=0.05, h_freq=30.0,
        n_jobs=cfg.N_JOBS, method='iir', verbose=False,
    )
    raw.pick(["C3", "C4", "CP1", "CP2", "FC1", "FC2"])
    raw.resample(SFREQ_TARGET, verbose=False)
    data = raw.get_data(tmin=0.0, tmax=2.0)
    return data[np.newaxis, :, :]


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
eeg = acquisition.EEG()
trial_results = []

with EEGManager() as mgr:
    eeg.setup(mgr, device_name=DEVICE_NAME, cap=CAP, sfreq=SFREQ_IN)
    eeg.start_acquisition()
    print("--- EEG ACQUISITION ACTIVE ---")
    print("Signal stabilization (2 s)...")
    time.sleep(2)

    # ── LSL CONNECTION (after EEG is ready) ───────────────────────────────────
    inlet = None
    if LSL_AVAILABLE:
        print("[LSL] Waiting for paradigm.py — open it in PsychoPy now...")
        streams = resolve_byprop('name', 'BCI_Markers')
        inlet = StreamInlet(streams[0])
        print("[LSL] Stream connected — press SPACE in PsychoPy to begin.\n")
    else:
        print("Ready (manual mode). Waiting for trials...\n")

    try:
        trial_num = 0
        while trial_num < NUM_TRIALS:

            if inlet is not None:
                sample, _ = inlet.pull_sample(timeout=120.0)
                if sample is None:
                    print("Marker timeout — stopping.")
                    break
                marker = int(sample[0])
            else:
                raw_input = input(
                    f"[MANUAL] Trial {trial_num + 1} — press Enter for LEFT, type 'r'+Enter for RIGHT: "
                )
                marker = MARKER_RIGHT if raw_input.strip().lower() == 'r' else MARKER_LEFT

            true_label = 0 if marker == MARKER_LEFT else 1
            direction  = "LEFT" if marker == MARKER_LEFT else "RIGHT"

            mgr.annotate(str(true_label))
            print(f"Trial {trial_num + 1:>2}/{NUM_TRIALS} [{direction}] — recording...")

            time.sleep(EPOCH_DUR + 0.2)

            eeg.get_mne()
            raw_all = eeg.data.mne_raw
            events, _ = mne.events_from_annotations(raw_all, verbose=False)

            if len(events) == 0:
                print("  No annotation found in stream — skipping trial.")
                continue

            last_sample = events[-1][0]
            t_start = last_sample / raw_all.info['sfreq']
            t_end   = t_start + EPOCH_DUR

            if raw_all.times[-1] < t_end:
                extra = t_end - raw_all.times[-1] + 0.1
                print(f"  Waiting extra {extra:.2f} s for data...")
                time.sleep(extra)
                eeg.get_mne()
                raw_all = eeg.data.mne_raw

            trial_raw = raw_all.copy().crop(tmin=t_start, tmax=t_end)

            try:
                X = preprocess_online(trial_raw)
                pred = model.predict(X)[0]
                is_correct = (pred == true_label)

                pred_name = "LEFT" if pred == 0 else "RIGHT"
                status    = "OK" if is_correct else "WRONG"
                print(f"  -> Predicted: {pred_name}  [{status}]")

                trial_results.append({'true_label': true_label, 'pred_label': int(pred)})

            except Exception as e:
                print(f"  Processing error: {e}")

            trial_num += 1

    except KeyboardInterrupt:
        print("\nSession interrupted manually.")

    finally:
        eeg.stop_acquisition()
        mgr.disconnect()
        eeg.close()

# ── RESULTS ───────────────────────────────────────────────────────────────────
if not trial_results:
    print("No trials completed.")
else:
    done          = len(trial_results)
    total_correct = sum(r['pred_label'] == r['true_label'] for r in trial_results)

    left_trials   = [r for r in trial_results if r['true_label'] == 0]
    right_trials  = [r for r in trial_results if r['true_label'] == 1]
    left_correct  = sum(r['pred_label'] == r['true_label'] for r in left_trials)
    right_correct = sum(r['pred_label'] == r['true_label'] for r in right_trials)

    summary_lines = [
        f"Motion type : {motion}",
        f"Date        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total : {total_correct}/{done}",
        f"LEFT  : {left_correct}/{len(left_trials)}",
        f"RIGHT : {right_correct}/{len(right_trials)}",
    ]

    print(f"\n{'=' * 32}")
    for line in summary_lines:
        print(line)
    print(f"{'=' * 32}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS_DIR / f'results_online_{motion}_{timestamp}.txt'
    out_path.write_text('\n'.join(summary_lines), encoding='utf-8')
    print(f"\nResults saved -> {out_path}")
