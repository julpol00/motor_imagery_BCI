import os
import joblib
import numpy as np
import src.config as cfg
from src.models import (
    MODEL_LDA_BASE_IMAGERY, MODEL_SVC_BASE_IMAGERY, MODEL_RF_BASE_IMAGERY,
    MODEL_XGB_BASE_IMAGERY, MODEL_LGBM_BASE_IMAGERY,
    MODEL_LDA_BASE_REAL, MODEL_SVC_BASE_REAL, MODEL_RF_BASE_REAL,
    MODEL_XGB_BASE_REAL, MODEL_LGBM_BASE_REAL,
    MODEL_LDA_MRP_IMAGERY, MODEL_SVC_MRP_IMAGERY, MODEL_RF_MRP_IMAGERY,
    MODEL_XGB_MRP_IMAGERY, MODEL_LGBM_MRP_IMAGERY,
    MODEL_LDA_MRP_REAL, MODEL_SVC_MRP_REAL, MODEL_RF_MRP_REAL,
    MODEL_XGB_MRP_REAL, MODEL_LGBM_MRP_REAL,
    MODEL_LDA_ERD_ERS_IMAGERY, MODEL_SVC_ERD_ERS_IMAGERY, MODEL_RF_ERD_ERS_IMAGERY,
    MODEL_XGB_ERD_ERS_IMAGERY, MODEL_LGBM_ERD_ERS_IMAGERY,
    MODEL_LDA_ERD_ERS_REAL, MODEL_SVC_ERD_ERS_REAL, MODEL_RF_ERD_ERS_REAL,
    MODEL_XGB_ERD_ERS_REAL, MODEL_LGBM_ERD_ERS_REAL,
)
from src.data_utils import (
    split_train_test_path_list, get_X_and_Y_from_epochs,
    get_X_and_Y_from_features, train_and_test_model,
)

MODELS_DIR   = r'D:\inżynierka\motor imagery BCI\impl\models'
RESULTS_PATH = r'results/results.txt'
TRAIN_RATIO      = 0.2
SELECTED_CHANNELS = ['C3', 'C4', 'CP1', 'CP2', 'FC1', 'FC2']


def save(model, name):
    joblib.dump(model, os.path.join(MODELS_DIR, f'bci_model_{name}.pkl'))


def run(xtr, xte, ytr, yte, gtr, gte, model, label, save_name):
    groups  = np.concatenate([gtr, gte])
    trained = train_and_test_model(xtr, xte, ytr, yte, model, label, groups=groups, results_path=RESULTS_PATH)
    save(trained, save_name)


# ============ BASE features (raw EEG epochs) ============

# --- IMAGERY ---
train_list, test_list = split_train_test_path_list(
    cfg.IMAGERY_MOTION_PREPROCESSED_PATH, cfg.FILE_PATTERN, TRAIN_RATIO)
X_train, X_test, y_train, y_test, g_train, g_test = get_X_and_Y_from_epochs(
    train_list, test_list, ['T1', 'T2'], picks=SELECTED_CHANNELS, t_min=0.0, t_max=2.0)

run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_LDA_BASE_IMAGERY,  "LDA base imagery",  "lda_base_imagery_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_SVC_BASE_IMAGERY,  "SVC base imagery",  "svc_base_imagery_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_RF_BASE_IMAGERY,   "RF base imagery",   "rf_base_imagery_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_XGB_BASE_IMAGERY,  "XGB base imagery",  "xgb_base_imagery_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_LGBM_BASE_IMAGERY, "LGBM base imagery", "lgbm_base_imagery_motion")

# --- REAL ---
train_list, test_list = split_train_test_path_list(
    cfg.REAL_MOTION_PREPROCESSED_PATH, cfg.FILE_PATTERN, TRAIN_RATIO)
X_train, X_test, y_train, y_test, g_train, g_test = get_X_and_Y_from_epochs(
    train_list, test_list, ['T1', 'T2'], picks=SELECTED_CHANNELS, t_min=0.0, t_max=2.0)

run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_LDA_BASE_REAL,  "LDA base real",  "lda_base_real_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_SVC_BASE_REAL,  "SVC base real",  "svc_base_real_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_RF_BASE_REAL,   "RF base real",   "rf_base_real_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_XGB_BASE_REAL,  "XGB base real",  "xgb_base_real_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_LGBM_BASE_REAL, "LGBM base real", "lgbm_base_real_motion")


# ============ MRP features ============

# --- IMAGERY ---
train_list, test_list = split_train_test_path_list(
    cfg.IMAGERY_MOTION_MRP_PATH, cfg.MRP_FILE_PATTERN, TRAIN_RATIO)
X_train, X_test, y_train, y_test, g_train, g_test = get_X_and_Y_from_features(train_list, test_list)

run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_LDA_MRP_IMAGERY,  "LDA MRP imagery",  "lda_mrp_imagery_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_SVC_MRP_IMAGERY,  "SVC MRP imagery",  "svc_mrp_imagery_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_RF_MRP_IMAGERY,   "RF MRP imagery",   "rf_mrp_imagery_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_XGB_MRP_IMAGERY,  "XGB MRP imagery",  "xgb_mrp_imagery_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_LGBM_MRP_IMAGERY, "LGBM MRP imagery", "lgbm_mrp_imagery_motion")

# --- REAL ---
train_list, test_list = split_train_test_path_list(
    cfg.REAL_MOTION_MRP_PATH, cfg.MRP_FILE_PATTERN, TRAIN_RATIO)
X_train, X_test, y_train, y_test, g_train, g_test = get_X_and_Y_from_features(train_list, test_list)

run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_LDA_MRP_REAL,  "LDA MRP real",  "lda_mrp_real_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_SVC_MRP_REAL,  "SVC MRP real",  "svc_mrp_real_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_RF_MRP_REAL,   "RF MRP real",   "rf_mrp_real_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_XGB_MRP_REAL,  "XGB MRP real",  "xgb_mrp_real_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_LGBM_MRP_REAL, "LGBM MRP real", "lgbm_mrp_real_motion")


# ============ ERD/ERS features ============

# --- IMAGERY ---
train_list, test_list = split_train_test_path_list(
    cfg.IMAGERY_MOTION_ERD_ERS_PATH, cfg.ERD_ERS_FILE_PATTERN, TRAIN_RATIO)
X_train, X_test, y_train, y_test, g_train, g_test = get_X_and_Y_from_features(train_list, test_list)

run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_LDA_ERD_ERS_IMAGERY,  "LDA ERD/ERS imagery",  "lda_erd_ers_imagery_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_SVC_ERD_ERS_IMAGERY,  "SVC ERD/ERS imagery",  "svc_erd_ers_imagery_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_RF_ERD_ERS_IMAGERY,   "RF ERD/ERS imagery",   "rf_erd_ers_imagery_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_XGB_ERD_ERS_IMAGERY,  "XGB ERD/ERS imagery",  "xgb_erd_ers_imagery_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_LGBM_ERD_ERS_IMAGERY, "LGBM ERD/ERS imagery", "lgbm_erd_ers_imagery_motion")

# --- REAL ---
train_list, test_list = split_train_test_path_list(
    cfg.REAL_MOTION_ERD_ERS_PATH, cfg.ERD_ERS_FILE_PATTERN, TRAIN_RATIO)
X_train, X_test, y_train, y_test, g_train, g_test = get_X_and_Y_from_features(train_list, test_list)

run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_LDA_ERD_ERS_REAL,  "LDA ERD/ERS real",  "lda_erd_ers_real_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_SVC_ERD_ERS_REAL,  "SVC ERD/ERS real",  "svc_erd_ers_real_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_RF_ERD_ERS_REAL,   "RF ERD/ERS real",   "rf_erd_ers_real_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_XGB_ERD_ERS_REAL,  "XGB ERD/ERS real",  "xgb_erd_ers_real_motion")
run(X_train, X_test, y_train, y_test, g_train, g_test, MODEL_LGBM_ERD_ERS_REAL, "LGBM ERD/ERS real", "lgbm_erd_ers_real_motion")
