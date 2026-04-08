
N_JOBS = 10
RANDOM_STATE = 42
POWER_FREQ = 60
CHANNELS_NEW_NAMES_DICT = {
    'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCz', 'Fc2.': 'FC2', 'Fc4.': 'FC4', 'Fc6.': 'FC6',
    'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1', 'Cz..': 'Cz', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6',
    'Cp5.': 'CP5', 'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPz', 'Cp2.': 'CP2', 'Cp4.': 'CP4', 'Cp6.': 'CP6',
    'Fp1.': 'Fp1', 'Fpz.': 'Fpz', 'Fp2.': 'Fp2',
    'Af7.': 'AF7', 'Af3.': 'AF3', 'Afz.': 'AFz', 'Af4.': 'AF4', 'Af8.': 'AF8',
    'F7..': 'F7', 'F5..': 'F5', 'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'Fz', 'F2..': 'F2', 'F4..': 'F4', 'F6..': 'F6',
    'F8..': 'F8',
    'Ft7.': 'FT7', 'Ft8.': 'FT8',
    'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10',
    'Tp7.': 'TP7', 'Tp8.': 'TP8',
    'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1', 'Pz..': 'Pz', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6',
    'P8..': 'P8',
    'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POz', 'Po4.': 'PO4', 'Po8.': 'PO8',
    'O1..': 'O1', 'Oz..': 'Oz', 'O2..': 'O2', 'Iz..': 'Iz'
}

### DIR PATH
REAL_MOTION_PREPROCESSED_PATH = "D:\inżynierka\motor imagery BCI\data\\real_motion\preprocessed"
IMAGERY_MOTION_PREPROCESSED_PATH = "D:\inżynierka\motor imagery BCI\data\imagine_motion\preprocessed"
REAL_MOTION_RAW_PATH = "D:\inżynierka\motor imagery BCI\data\\real_motion\\raw"
IMAGERY_MOTION_RAW_PATH = "D:\inżynierka\motor imagery BCI\data\\imagery_motion\\raw"
FILE_PATTERN = 's*.edf-epo.fif'

### PARAM GRIDS
PARAM_GRID_LDA = {
    "lda__solver": ["lsqr", "eigen"],
    "lda__shrinkage": ["auto", 0.1, 0.5, 0.9]
}

PARAM_GRID_SVC = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__kernel': ['rbf'],
    'svc__gamma': ['scale', 'auto', 0.001, 0.01]
}

