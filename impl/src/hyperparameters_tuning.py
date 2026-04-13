import optuna
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from models import reshape_eeg, split_train_test_path_list, get_X_and_Y_from_epochs
import config as cfg


def objective_lda(trial, X_train, y_train):

    solver = trial.suggest_categorical('solver', ['lsqr', 'eigen'])
    shrinkage_type = trial.suggest_categorical('shrinkage_type', ['auto', 'manual', None])

    if shrinkage_type == 'manual':
        shrinkage = trial.suggest_float('shrinkage', 0.0, 1.0)
    elif shrinkage_type == 'auto':
        shrinkage = 'auto'
    else:
        shrinkage = None

    # Tworzymy pipeline
    pipeline = Pipeline(steps=[
        ('reshape', FunctionTransformer(reshape_eeg)),
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage))
    ])

    try:
        score = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1)
        return score.mean()
    except Exception: # to avoid exception when solver eigen is used with shrinkage = None
        return 0.0

train_ratio = 0.2
selected_channels = ['C3', 'C4', 'CP1', 'CP2', 'FC1', 'FC2']

train_list, test_list = split_train_test_path_list(cfg.IMAGERY_MOTION_PREPROCESSED_PATH, cfg.FILE_PATTERN, train_ratio)
X_train, X_test, y_train, y_test = get_X_and_Y_from_epochs(train_list, test_list, ['T1','T2'], picks=selected_channels, t_min = 0.2, t_max = 2.0)

study_lda = optuna.create_study(direction='maximize')
study_lda.optimize(lambda trial: objective_lda(trial, X_train, y_train), n_trials=30)


print(f"Best parameters LDA: {study_lda.best_params}")