import os
import glob
import mne
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC

# This function divides the dataset into training and test sections while maintaining
# group consistency. This prevents data from the same patient from being sent to both
# sets simultaneously, which could result in inflated model performance results.
def split_train_test_path_list(data_path, file_pattern, train_ratio):

    file_list = sorted(glob.glob(os.path.join(data_path, file_pattern)))
    subject_files = defaultdict(list)

    for f in file_list:
        fname = os.path.basename(f)
        subject = fname.split("R")[0]
        subject_files[subject].append(f)

    subjects = list(subject_files.keys())
    np.random.shuffle(subjects)

    split_id = int(len(subjects) * train_ratio)

    train_subjects = subjects[split_id:]
    test_subjects = subjects[:split_id]

    train_list = []
    test_list = []

    for s in train_subjects:
        train_list.extend(subject_files[s])

    for s in test_subjects:
        test_list.extend(subject_files[s])

    return train_list, test_list

def read_eeg_epochs(train_list, test_list):
    epochs_train_list = []
    epochs_test_list = []

    for file_path in train_list:
            epoch_train = mne.read_epochs(file_path, preload=True)
            epochs_train_list.append(epoch_train)

    for file_path in test_list:
            epoch_test = mne.read_epochs(file_path, preload=True)
            epochs_test_list.append(epoch_test)

    epochs_train = mne.concatenate_epochs(epochs_train_list)
    epochs_test = mne.concatenate_epochs(epochs_test_list)

    return epochs_train, epochs_test

def prepare_data_from_epochs(epochs, events, picks, t_min, t_max):
    data_ev1 = epochs[events[0]].get_data(picks=picks, tmin=t_min, tmax=t_max)
    data_ev2 = epochs[events[1]].get_data(picks=picks, tmin=t_min, tmax=t_max)

    X = np.concatenate((data_ev1, data_ev2), axis=0)

    y = np.concatenate([
        np.zeros(len(data_ev1)),
        np.ones(len(data_ev2))
    ])

    return X, y

def get_X_and_Y_from_epochs(train_list, test_list, events, picks=None, t_min=0.0, t_max=0.5):
    epochs_train, epochs_test = read_eeg_epochs(train_list, test_list)

    X_train, y_train = prepare_data_from_epochs(epochs_train, events, picks, t_min, t_max)
    X_test, y_test = prepare_data_from_epochs(epochs_test, events, picks, t_min, t_max)

    return X_train, X_test, y_train, y_test


def eval_split(name, X, y, model):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    print(f"\n== {name.upper()} ==")
    print(f"AUC      : {roc_auc_score(y, model.predict_proba(X)[:, 1]):.4f}")
    print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
    print(f"F1       : {f1_score(y, y_pred, pos_label=1):.4f}")
    print(f"Precision: {precision_score(y, y_pred, pos_label=1):.4f}")
    print(f"Recall   : {recall_score(y, y_pred, pos_label=1):.4f}")
    cm_df = pd.DataFrame(cm, index=["Actual left", "Actual right"], columns=["Predicted left", "Predicted right"])
    print(f"\nConfusion matrix:")
    print(cm_df)

def train_and_test_model(X_train, X_test, y_train, y_test, model, gridSearch = False):

    model.fit(X_train, y_train)

    eval_split("train", X_train, y_train, model)
    eval_split("test",  X_test,  y_test, model)

    if gridSearch == True:
        print("\n== GridSearchCV ==")
        print(f"Best params: {model.best_params_}")

    return model

def reshape_eeg(X):
    return X.reshape(X.shape[0], -1)


### PIPELINES

MODEL_LDA = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis())
])

MODEL_SVC = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, cache_size=1000))
])

MODEL_LDA_BEST = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.9))
])

MODEL_SVC_BEST = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True))
])
