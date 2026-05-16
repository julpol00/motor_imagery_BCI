import json
import os

import matplotlib.pyplot as plt
import optuna
from lightgbm import LGBMClassifier
from optuna.visualization.matplotlib import plot_param_importances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import config as cfg
from data_utils import (
    get_X_and_Y_from_epochs,
    get_X_and_Y_from_features,
    reshape_eeg,
    split_train_test_path_list,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN_RATIO = 0.2
SELECTED_CHANNELS = ['C3', 'C4', 'CP1', 'CP2', 'FC1', 'FC2']
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'best_hyperparameters.json'
)
BEST_RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'the_best_models_best_hyperparameters.json'
)
PLOTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'plots'
)

N_TRIALS = {
    'lda':  50,
    'svc':  50,
    'rf':   50,
    'xgb':  50,
    'lgbm': 50,
}

DATA_CONFIGS = {
    'base__imagery': {
        'path':    cfg.IMAGERY_MOTION_PREPROCESSED_PATH,
        'pattern': cfg.FILE_PATTERN,
        'mode':    'epochs',
    },
    'base__real': {
        'path':    cfg.REAL_MOTION_PREPROCESSED_PATH,
        'pattern': cfg.FILE_PATTERN,
        'mode':    'epochs',
    },
    'mrp__imagery': {
        'path':    cfg.IMAGERY_MOTION_MRP_PATH,
        'pattern': cfg.MRP_FILE_PATTERN,
        'mode':    'features',
    },
    'mrp__real': {
        'path':    cfg.REAL_MOTION_MRP_PATH,
        'pattern': cfg.MRP_FILE_PATTERN,
        'mode':    'features',
    },
    'erd_ers__imagery': {
        'path':    cfg.IMAGERY_MOTION_ERD_ERS_PATH,
        'pattern': cfg.ERD_ERS_FILE_PATTERN,
        'mode':    'features',
    },
    'erd_ers__real': {
        'path':    cfg.REAL_MOTION_ERD_ERS_PATH,
        'pattern': cfg.ERD_ERS_FILE_PATTERN,
        'mode':    'features',
    }
}

def load_results(path=RESULTS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, 'r') as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    return {}


def save_results(results, path=RESULTS_PATH):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def save_importances_plot(study, result_key):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    ax = plot_param_importances(study)
    fig = ax.figure
    ax.set_xlabel('Ważność')
    ax.set_ylabel('Hiperparametr')
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    fig.set_size_inches(8, 5)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f'{result_key}_importances.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"          plot saved → {path}")


def load_data(data_config):
    train_list, test_list = split_train_test_path_list(
        data_config['path'], data_config['pattern'], TRAIN_RATIO
    )
    if data_config['mode'] == 'epochs':
        return get_X_and_Y_from_epochs(
            train_list, test_list, ['T1', 'T2'],
            picks=SELECTED_CHANNELS, t_min=0.0, t_max=2.0,
        )
    return get_X_and_Y_from_features(train_list, test_list)


def _cv_score(pipeline, X, y):
    return cross_val_score(pipeline, X, y, cv=5, n_jobs=-1, scoring='roc_auc').mean()


def objective_lda(trial, X_train, y_train):
    solver = trial.suggest_categorical('solver', ['lsqr', 'eigen'])
    shrinkage_type = trial.suggest_categorical('shrinkage_type', ['auto', 'manual', None])

    if shrinkage_type == 'manual':
        shrinkage = trial.suggest_float('shrinkage', 0.0, 1.0)
    elif shrinkage_type == 'auto':
        shrinkage = 'auto'
    else:
        shrinkage = None

    pipeline = Pipeline([
        ('reshape', FunctionTransformer(reshape_eeg)),
        ('scaler', StandardScaler()),
        ('clf', LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)),
    ])
    try:
        return _cv_score(pipeline, X_train, y_train)
    except Exception:  # eigen + shrinkage=None raises ValueError
        return 0.0


def objective_svc(trial, X_train, y_train):
    C     = trial.suggest_float('C',     0.01, 100.0, log=True)
    gamma = trial.suggest_float('gamma', 1e-4,   1.0, log=True)

    pipeline = Pipeline([
        ('reshape', FunctionTransformer(reshape_eeg)),
        ('scaler', StandardScaler()),
        ('clf', SVC(C=C, kernel='rbf', gamma=gamma, probability=True, random_state=42)),
    ])
    return _cv_score(pipeline, X_train, y_train)


def objective_rf(trial, X_train, y_train):
    pipeline = Pipeline([
        ('reshape', FunctionTransformer(reshape_eeg)),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators      = trial.suggest_int('n_estimators',      50, 500),
            max_depth         = trial.suggest_int('max_depth',          3,  20),
            min_samples_split = trial.suggest_int('min_samples_split',  2,  20),
            min_samples_leaf  = trial.suggest_int('min_samples_leaf',   1,  10),
            random_state=42,
            n_jobs=-1,
        )),
    ])
    return _cv_score(pipeline, X_train, y_train)


def objective_xgb(trial, X_train, y_train):
    pipeline = Pipeline([
        ('reshape', FunctionTransformer(reshape_eeg)),
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(
            n_estimators     = trial.suggest_int('n_estimators',       50, 500),
            max_depth        = trial.suggest_int('max_depth',           2,   6),
            learning_rate    = trial.suggest_float('learning_rate',  0.01, 0.3,  log=True),
            subsample        = trial.suggest_float('subsample',       0.5,  1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree',0.5,  1.0),
            reg_alpha        = trial.suggest_float('reg_alpha',      1e-3, 10.0, log=True),
            reg_lambda       = trial.suggest_float('reg_lambda',     1e-3, 10.0, log=True),
            min_child_weight = trial.suggest_int('min_child_weight',   10, 100),
            random_state=42,
            verbosity=0,
        )),
    ])
    return _cv_score(pipeline, X_train, y_train)


def objective_lgbm(trial, X_train, y_train):
    pipeline = Pipeline([
        ('reshape', FunctionTransformer(reshape_eeg)),
        ('scaler', StandardScaler()),
        ('clf', LGBMClassifier(
            n_estimators      = trial.suggest_int('n_estimators',      50, 500),
            learning_rate     = trial.suggest_float('learning_rate',  0.01, 0.3, log=True),
            num_leaves        = trial.suggest_int('num_leaves',         15, 127),
            min_child_samples = trial.suggest_int('min_child_samples',   5, 100),
            reg_alpha         = trial.suggest_float('reg_alpha',      1e-3, 10.0, log=True),
            reg_lambda        = trial.suggest_float('reg_lambda',     1e-3, 10.0, log=True),
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )),
    ])
    return _cv_score(pipeline, X_train, y_train)


OBJECTIVES = {
    'lda':  objective_lda,
    'svc':  objective_svc,
    'rf':   objective_rf,
    'xgb':  objective_xgb,
    'lgbm': objective_lgbm,
}

if __name__ == '__main__':

    RUN_CONFIGS = []

    results = load_results(BEST_RESULTS_PATH)
    loaded_data_cache = {}

    for model_name, data_key in RUN_CONFIGS:
        result_key = f"{model_name}__{data_key}"

        if result_key in results:
            print(f"[{result_key}] already done  AUC={results[result_key]['best_auc']:.4f} — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {result_key}")

        if data_key not in loaded_data_cache:
            loaded_data_cache[data_key] = load_data(DATA_CONFIGS[data_key])
        X_train, X_test, y_train, y_test, *_ = loaded_data_cache[data_key]
        print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

        objective_fn = OBJECTIVES[model_name]
        print(f"  [{model_name:5s}] running {N_TRIALS[model_name]} trials ...", end='', flush=True)
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial, fn=objective_fn: fn(trial, X_train, y_train),
            n_trials=N_TRIALS[model_name],
        )

        results[result_key] = {
            'best_params': study.best_params,
            'best_auc':    round(study.best_value, 6),
        }
        save_results(results, BEST_RESULTS_PATH)
        print(f"  AUC={study.best_value:.4f}")
        print(f"          params: {study.best_params}")
        save_importances_plot(study, result_key)

    print(f"\nDone. Results saved to: {BEST_RESULTS_PATH}")
