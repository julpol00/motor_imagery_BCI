import os
import glob
import mne
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split, GroupKFold

METRICS = ['AUC', 'Accuracy', 'F1', 'Precision', 'Recall']

def split_train_test_path_list(data_path, file_pattern, train_ratio):
    file_list = sorted(glob.glob(os.path.join(data_path, file_pattern)))
    subject_files = defaultdict(list)

    for f in file_list:
        subject = os.path.basename(f).split("R")[0]
        subject_files[subject].append(f)

    subjects = list(subject_files.keys())
    np.random.shuffle(subjects)

    split_id = int(len(subjects) * train_ratio)
    train_list = [f for s in subjects[split_id:] for f in subject_files[s]]
    test_list  = [f for s in subjects[:split_id]  for f in subject_files[s]]

    return train_list, test_list

def _load_epochs_with_groups(file_list, events, picks, t_min, t_max):
    X_parts, y_parts, groups = [], [], []
    for f in file_list:
        epochs = mne.read_epochs(f, preload=True)
        data_ev1 = epochs[events[0]].get_data(picks=picks, tmin=t_min, tmax=t_max)
        data_ev2 = epochs[events[1]].get_data(picks=picks, tmin=t_min, tmax=t_max)
        X = np.concatenate([data_ev1, data_ev2], axis=0)
        y = np.concatenate([np.zeros(len(data_ev1)), np.ones(len(data_ev2))])
        subject = os.path.basename(f).split("R")[0]
        X_parts.append(X)
        y_parts.append(y)
        groups.extend([subject] * len(X))
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts), np.array(groups)


def _load_features_with_groups(file_list):
    X_parts, y_parts, groups = [], [], []
    for f in file_list:
        data = np.load(f)
        X, y = data['X'], data['y']
        subject = os.path.basename(f).split("R")[0]
        X_parts.append(X)
        y_parts.append(y)
        groups.extend([subject] * len(X))
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts), np.array(groups)


# --- public data loaders (return groups alongside X, y) ---

def get_X_and_Y_from_epochs(train_list, test_list, events, picks=None, t_min=0.0, t_max=0.5):
    """Returns X_train, X_test, y_train, y_test, groups_train, groups_test."""
    X_train, y_train, g_train = _load_epochs_with_groups(train_list, events, picks, t_min, t_max)
    X_test,  y_test,  g_test  = _load_epochs_with_groups(test_list,  events, picks, t_min, t_max)
    return X_train, X_test, y_train, y_test, g_train, g_test


def get_X_and_Y_from_features(train_list, test_list):
    """Returns X_train, X_test, y_train, y_test, groups_train, groups_test."""
    X_train, y_train, g_train = _load_features_with_groups(train_list)
    X_test,  y_test,  g_test  = _load_features_with_groups(test_list)
    return X_train, X_test, y_train, y_test, g_train, g_test


def get_X_and_Y_random_split(data_path, file_pattern, test_ratio=0.2, random_state=42):
    file_list = sorted(glob.glob(os.path.join(data_path, file_pattern)))
    X_list, y_list = [], []
    for f in file_list:
        data = np.load(f)
        X_list.append(data['X'])
        y_list.append(data['y'])
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return train_test_split(X, y, test_size=test_ratio, random_state=random_state, stratify=y)


def reshape_eeg(X):
    return X.reshape(X.shape[0], -1)


# --- metrics helpers ---

def _compute_metrics(y_true, y_pred, y_proba):
    return {
        'AUC':       roc_auc_score(y_true, y_proba[:, 1]),
        'Accuracy':  accuracy_score(y_true, y_pred),
        'F1':        f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def eval_split(mode, X, y, model):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)
    metrics = _compute_metrics(y, y_pred, y_proba)
    cm = confusion_matrix(y, y_pred)

    print(f"\n== {mode.upper()} ==")
    for name, val in metrics.items():
        print(f"  {name:12s}: {val:.4f}")
    cm_df = pd.DataFrame(cm, index=["Actual left", "Actual right"],
                         columns=["Predicted left", "Predicted right"])
    print(f"\n  Confusion matrix:\n{cm_df}")


# --- training ---

def train_and_test_model(X_train, X_test, y_train, y_test, model, model_name,
                          groups=None, n_splits=5, results_path=None):
    cv_stats = None
    if groups is not None:
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test])

        fold_scores = {split: {m: [] for m in METRICS} for split in ('train', 'test')}

        for tr_idx, te_idx in GroupKFold(n_splits=n_splits).split(X_all, y_all, groups):
            fold_model = clone(model)
            fold_model.fit(X_all[tr_idx], y_all[tr_idx])
            for split, idx in (('train', tr_idx), ('test', te_idx)):
                y_pred  = fold_model.predict(X_all[idx])
                y_proba = fold_model.predict_proba(X_all[idx])
                for m, v in _compute_metrics(y_all[idx], y_pred, y_proba).items():
                    fold_scores[split][m].append(v)

        cv_stats = {
            split: {m: (np.mean(vals), np.std(vals)) for m, vals in scores.items()}
            for split, scores in fold_scores.items()
        }

        print(f"\n{'='*60}")
        print(f"CV  {model_name}  ({n_splits}-fold GroupKFold)")
        print(f"{'='*60}")
        for split in ('train', 'test'):
            print(f"  [{split.upper()}]")
            for m, (mean, std) in cv_stats[split].items():
                print(f"    {m:12s}: {mean:.4f} ± {std:.4f}")

    # Train final model on X_train and evaluate both splits
    model.fit(X_train, y_train)
    print(f"\n==== {model_name} — Final Model ====")
    eval_split("train", X_train, y_train, model)
    eval_split("test",  X_test,  y_test,  model)

    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)
    final_metrics = {
        'train': _compute_metrics(y_train, y_pred_tr, model.predict_proba(X_train)),
        'test':  _compute_metrics(y_test,  y_pred_te, model.predict_proba(X_test)),
    }
    final_cm = {
        'train': confusion_matrix(y_train, y_pred_tr),
        'test':  confusion_matrix(y_test,  y_pred_te),
    }

    if results_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
        with open(results_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"MODEL: {model_name}\n")
            f.write(f"{'='*60}\n")

            if cv_stats is not None:
                f.write(f"\n--- {n_splits}-fold GroupKFold CV ---\n")
                for split in ('train', 'test'):
                    f.write(f"  [{split.upper()}]\n")
                    for m, (mean, std) in cv_stats[split].items():
                        f.write(f"    {m:12s}: {mean:.4f} ± {std:.4f}\n")

            f.write("\n--- Final Model ---\n")
            for split in ('train', 'test'):
                f.write(f"  [{split.upper()}]\n")
                for m, val in final_metrics[split].items():
                    f.write(f"    {m:12s}: {val:.4f}\n")
                cm = final_cm[split]
                f.write(f"    Confusion matrix:\n")
                f.write(f"                   Pred left  Pred right\n")
                f.write(f"    Actual left  :  {cm[0,0]:>9}  {cm[0,1]:>9}\n")
                f.write(f"    Actual right :  {cm[1,0]:>9}  {cm[1,1]:>9}\n")

    return model
