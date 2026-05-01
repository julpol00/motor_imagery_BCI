import optuna
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier
from data_utils import reshape_eeg, split_train_test_path_list, get_X_and_Y_from_epochs, get_X_and_Y_from_features
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
        score = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1, scoring='roc_auc')
        return score.mean()
    except Exception: # to avoid exception when solver eigen is used with shrinkage = None
        return 0.0

def objective_xgb(trial, X_train, y_train):

    n_estimators      = trial.suggest_int('n_estimators', 50, 500)
    max_depth         = trial.suggest_int('max_depth', 2, 6)
    learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    subsample         = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    reg_alpha         = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
    reg_lambda        = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
    min_child_weight  = trial.suggest_int('min_child_weight', 10, 100)

    pipeline = Pipeline(steps=[
        ('reshape', FunctionTransformer(reshape_eeg)),
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            random_state=42,
            verbosity=0,
        ))
    ])

    score = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1, scoring='roc_auc')
    return score.mean()


train_ratio = 0.2
selected_channels = ['C3', 'C4', 'CP1', 'CP2', 'FC1', 'FC2']

# ----------- IMAGERY MOTION - cross-subject split ------------------
train_list, test_list = split_train_test_path_list(cfg.IMAGERY_MOTION_PREPROCESSED_PATH, cfg.FILE_PATTERN, train_ratio)
X_train, X_test, y_train, y_test = get_X_and_Y_from_epochs(train_list, test_list, ['T1','T2'], picks=selected_channels, t_min=0.2, t_max=2.0)

study_xgb_img = optuna.create_study(direction='maximize')
study_xgb_img.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=50)

print(f"Best parameters XGB imagery: {study_xgb_img.best_params}")
print(f"Best CV AUC imagery:         {study_xgb_img.best_value:.4f}")

study_lda_img = optuna.create_study(direction='maximize')
study_lda_img.optimize(lambda trial: objective_lda(trial, X_train, y_train), n_trials=30)

print(f"Best parameters LDA imagery: {study_lda_img.best_params}")
print(f"Best CV AUC imagery:         {study_lda_img.best_value:.4f}")

# # ----------- REAL MOTION - cross-subject split ---------------------
# train_list, test_list = split_train_test_path_list(cfg.REAL_MOTION_ERD_ERS_PATH, cfg.ERD_ERS_FILE_PATTERN, train_ratio)
# X_train, X_test, y_train, y_test = get_X_and_Y_from_features(train_list, test_list)
#
# study_xgb_real = optuna.create_study(direction='maximize')
# study_xgb_real.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=50)
#
# print(f"Best parameters XGB real: {study_xgb_real.best_params}")
# print(f"Best CV AUC real:         {study_xgb_real.best_value:.4f}")