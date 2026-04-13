import time
import numpy as np
import mne
import joblib
from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager
import config as cfg

# 1. LOAD MODEL
model_path = r'D:\inżynierka\motor imagery BCI\impl\models\bci_model_lda_real_motion.pkl'
model = joblib.load(model_path)

cap = {2: "C3", 3: "C4", 4: "CP1", 5: "CP2", 0: "FC1", 1: "FC2"}
device_name = "BA MINI 039"
sfreq_original = 250
sfreq_target = 160
NUM_TRIALS = 10


def preprocess_online(raw):
    # Re-reference
    raw.set_eeg_reference(ref_channels='average', verbose=False)

    #Notch filter
    # nyquist_freq = raw.info['sfreq'] / 2
    # raw.notch_filter(
    #     picks=['eeg'],
    #     freqs=np.arange(cfg.POWER_FREQ, nyquist_freq, cfg.POWER_FREQ),
    #     n_jobs=cfg.N_JOBS,
    #     verbose=False
    # )

    # Bandpass filter
    raw.filter(
        picks=['eeg'],
        l_freq=0.05,
        h_freq=30.0,
        n_jobs=cfg.N_JOBS,
        method='iir',
        verbose=False
    )

    # 4. Baseline (0.0 - 0.2s)
    # data = raw.get_data()
    # times = raw.times
    # baseline_idx = (times >= 0) & (times <= 0.2)
    # if np.any(baseline_idx):
    #     b_mean = np.mean(data[:, baseline_idx], axis=1, keepdims=True)
    #     data -= b_mean
    #     raw._data = data

    target_channels = ["FC1", "FC2", "C3", "C4", "CP1", "CP2"]

    try:
        raw.pick(target_channels)
    except ValueError as e:
        print(f"Error: Channels not found. Available: {raw.ch_names}")
        raise e

    raw.resample(sfreq_target, verbose=False)

    data = raw.get_data(tmin=0.2, tmax=2.0)

    print(data.shape)
    # Reshape (1, n_channels, n_times)
    X = data[np.newaxis, :, :]
    return X


# 2. MAIN LOOP
eeg = acquisition.EEG()
correct_predictions = 0
results_log = []

with EEGManager() as mgr:
    eeg.setup(mgr, device_name=device_name, cap=cap, sfreq=sfreq_original)
    eeg.start_acquisition()
    print("--- SYSTEM STARTED ---")
    print("Acquisition active. Waiting 2 seconds for stabilization....")
    time.sleep(2)

    try:
        for i in range(1, NUM_TRIALS + 1):
            print(f"\n>>> TRIAL {i} FROM {NUM_TRIALS} <<<")

            print("[ CONCENTRATION ] - look at the point")
            time.sleep(3)

            direction = np.random.choice(['LEFT', 'RIGHT'])
            true_label = 0 if direction == 'LEFT' else 1

            # send a marker to EEG stream
            mgr.annotate(str(true_label))

            print(f"!!! MOVE: {direction} !!!")

            # waiting for move
            time.sleep(2.2)

            eeg.get_mne()
            raw_all = eeg.data.mne_raw
            events, event_id = mne.events_from_annotations(raw_all, verbose=False)

            if len(events) > 0:
                last_event_sample = events[-1][0]
                t_start = last_event_sample / raw_all.info['sfreq']
                t_max_needed = t_start + 2.2  # This is what the model needs

                current_max_time = raw_all.times[-1]

                if current_max_time < t_max_needed:
                    wait_time = t_max_needed - current_max_time + 0.2  # missing time + bufor
                    print(f"Waiting for a data... ({wait_time:.2f}s)")
                    time.sleep(wait_time)

                    eeg.get_mne()
                    raw_all = eeg.data.mne_raw

                trial_raw = raw_all.copy().crop(tmin=t_start, tmax=t_start + 2.2)

            try:
                X_online = preprocess_online(trial_raw)
                prediction = model.predict(X_online)[0]

                is_correct = (prediction == true_label)
                if is_correct:
                    correct_predictions += 1

                pred_name = "LEFT" if prediction == 0 else "RIGHT"
                status = "GOOD" if is_correct else "FAULT"

                print(f"RESULT: Detected {pred_name} | Status: {status}")
                results_log.append(is_correct)

            except Exception as e:
                print(f"Processing error in trial {i}: {e}")

            # Przerwa między próbami
            if i < NUM_TRIALS:
                print("Resting...")
                time.sleep(3)

        # --- PODSUMOWANIE ---
        print("\n" + "=" * 30)
        print("END OF SESSIONS - STATISTICS")
        print("=" * 30)
        accuracy = (correct_predictions / NUM_TRIALS) * 100
        print(f"Number of trials:      {NUM_TRIALS}")
        print(f"Correct:         {correct_predictions}")
        print(f"Fault:           {NUM_TRIALS - correct_predictions}")
        print(f"Accuracy (ACC): {accuracy:.2f}%")
        print("=" * 30)

    except KeyboardInterrupt:
        print("\nManually interrupted.")
    finally:
        eeg.stop_acquisition()
        mgr.disconnect()
        eeg.close()
        print("Device disconnected.")