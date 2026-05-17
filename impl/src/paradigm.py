"""
Motor Imagery BCI Paradigm — visual stimulus presentation.

Trial sequence:
  Fixation cross (2 s) -> Blank (0.5 s) -> Arrow cue (0.25 s)
  -> Motor imagery (4 s) -> Inter-trial interval (3 s)

LSL marker is sent at the START of the imagery phase (after the arrow disappears),
matching the PhysioNet T1/T2 annotation convention used during model training.

Usage:
  1. Start paradigm.py first (opens PsychoPy window)
  2. Start BCI.py in a second terminal (listens for LSL markers)
"""
import random
import csv
import time
from pathlib import Path

from psychopy import visual, core, event, monitors
import os
import sys
import logging

os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

ERROR_LOG = Path.home() / 'bci_results' / 'paradigm_errors.log'
ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(ERROR_LOG),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    encoding='utf-8',
)
sys.excepthook = lambda exc_type, exc_value, exc_tb: logging.error(
    'Uncaught exception', exc_info=(exc_type, exc_value, exc_tb)
)

# ── LSL ───────────────────────────────────────────────────────────────────────
try:
    from pylsl import StreamInfo, StreamOutlet
    _lsl_outlet = StreamOutlet(
        StreamInfo('BCI_Markers', 'Markers', 1, 0, 'int32', 'bci_mi_paradigm')
    )
    logging.info("[LSL] Outlet active.")
    print("[LSL] Outlet active.")
except ImportError:
    _lsl_outlet = None
    logging.info("[LSL] pylsl not installed — no LSL markers will be sent.")
    print("[LSL] pylsl not installed — no LSL markers will be sent.")

# ── CONFIG ────────────────────────────────────────────────────────────────────
NUM_TRIALS = 50    # must be even (equal number of left/right trials)
T_FIXATION = 2.0    # s
T_BLANK    = 0.5    # s
T_CUE      = 0.25   # s
T_IMAGERY  = 4.0    # s — model uses 0.0-2.0 s after imagery onset
T_ITI_MIN  = 3.0    # s
T_ITI_MAX  = 3.0    # s

MARKER_LEFT  = 1    # maps to model label 0 (left)
MARKER_RIGHT = 2    # maps to model label 1 (right)

LOG_PATH = Path.home() / 'bci_results' / 'paradigm_log.csv'

# ── WINDOW ────────────────────────────────────────────────────────────────────
mon = monitors.Monitor('myMonitor', width=53, distance=60)
mon.setSizePix([2048, 1152])

win = visual.Window(
    size=[2048, 1152],
    fullscr=True,
    monitor=mon,
    color='black',
    colorSpace='rgb',
    units='height',
    allowGUI=False,
)

# ── STIMULI ───────────────────────────────────────────────────────────────────
fixation     = visual.TextStim(win, text='+',  color='white', height=0.08)
arrow_left   = visual.TextStim(win, text='<',  color='white', height=0.15, pos=(-0.4, 0))
arrow_right  = visual.TextStim(win, text='>',  color='white', height=0.15, pos=( 0.4, 0))
dim_cross    = visual.TextStim(win, text='+',  color='grey',  height=0.05)

instruction = visual.TextStim(
    win,
    text=(
        'Press SPACE to begin.\n\n'
        'When an arrow appears — imagine moving that hand.\n'
        'Keep imagining for the entire duration of the grey cross.'
    ),
    color='white',
    height=0.045,
    wrapWidth=1.2,
)

end_screen = visual.TextStim(win, text='Session complete. Thank you!', color='white', height=0.07)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def push_marker(value: int):
    if _lsl_outlet:
        _lsl_outlet.push_sample([value])


def run_trial(trial_num: int, condition: str) -> dict:
    """Run one trial and return a timing log dict."""

    # 1. Fixation cross
    fixation.draw()
    win.flip()
    core.wait(T_FIXATION)

    # 2. Blank screen
    win.flip()
    core.wait(T_BLANK)

    # 3. Arrow cue
    (arrow_left if condition == 'left' else arrow_right).draw()
    win.flip()
    core.wait(T_CUE)

    # 4. Imagery phase — dim cross + LSL marker sent immediately on flip
    dim_cross.draw()
    win.flip()
    push_marker(MARKER_LEFT if condition == 'left' else MARKER_RIGHT)
    imagery_onset = time.perf_counter()
    core.wait(T_IMAGERY)

    # 5. ITI (blank screen)
    win.flip()
    iti = random.uniform(T_ITI_MIN, T_ITI_MAX)
    core.wait(iti)

    return {
        'trial':         trial_num + 1,
        'condition':     condition,
        'imagery_onset': round(imagery_onset, 4),
        'iti_s':         round(iti, 3),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
assert NUM_TRIALS % 2 == 0, "NUM_TRIALS must be even"

conditions = ['left'] * (NUM_TRIALS // 2) + ['right'] * (NUM_TRIALS // 2)
random.shuffle(conditions)

instruction.draw()
win.flip()
event.waitKeys(keyList=['space'])

logs = []
for i, cond in enumerate(conditions):
    if event.getKeys(['escape']):
        print("Session aborted by user.")
        break
    log = run_trial(i, cond)
    logs.append(log)
    print(f"  Trial {i + 1:>2}/{NUM_TRIALS}: {cond.upper()}")

end_screen.draw()
win.flip()
core.wait(2.0)

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(LOG_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['trial', 'condition', 'imagery_onset', 'iti_s'])
    writer.writeheader()
    writer.writerows(logs)

print(f"\nLog saved -> {LOG_PATH}")


try:
    win.close()
except Exception:
    pass

try:
    core.quit()
except SystemExit:
    pass
except Exception:
    pass