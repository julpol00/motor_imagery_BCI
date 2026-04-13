import src.config as cfg
from src.models import *
import joblib

train_ratio = 0.2
selected_channels = ['C3', 'C4', 'CP1', 'CP2', 'FC1', 'FC2']


# ------------IMAGERY MOTION ---------------------
train_list, test_list = split_train_test_path_list(cfg.IMAGERY_MOTION_PREPROCESSED_PATH, cfg.FILE_PATTERN, train_ratio)
X_train, X_test, y_train, y_test = get_X_and_Y_from_epochs(train_list, test_list, ['T1','T2'], picks=selected_channels, t_min = 0.2, t_max = 2.0)

# SVC
model_SVC_img = train_and_test_model(X_train, X_test, y_train, y_test,  MODEL_SVC_BEST, "SVC img")
joblib.dump(model_SVC_img, r'D:\inżynierka\motor imagery BCI\impl\models\bci_model_svc_img_motion.pkl')

# LDA
model_LDA_img = train_and_test_model(X_train, X_test, y_train, y_test, MODEL_LDA_BEST, "LDA img")
joblib.dump(model_SVC_img, r'D:\inżynierka\motor imagery BCI\impl\models\bci_model_lda_img_motion.pkl')

# RANDOM FOREST
model_RF_img = train_and_test_model(X_train, X_test, y_train, y_test, MODEL_RF, "Random Forest img")
joblib.dump(model_RF_img, r'D:\inżynierka\motor imagery BCI\impl\models\bci_model_rf_img_motion.pkl')

# XGBoost
model_XGB_img = train_and_test_model(X_train, X_test, y_train, y_test, MODEL_XGB, "XGB img")
joblib.dump(model_XGB_img, r'D:\inżynierka\motor imagery BCI\impl\models\bci_model_xgb_img_motion.pkl')

# lightGBM
model_LGBM_img = train_and_test_model(X_train, X_test, y_train, y_test, MODEL_LGBM, "LGBM img")
joblib.dump(model_LGBM_img, r'D:\inżynierka\motor imagery BCI\impl\models\bci_model_lgbm_img_motion.pkl')



# ----------- REAL MOTION ------------------
train_list, test_list = split_train_test_path_list(cfg.REAL_MOTION_PREPROCESSED_PATH, cfg.FILE_PATTERN, train_ratio)
X_train, X_test, y_train, y_test = get_X_and_Y_from_epochs(train_list, test_list, ['T1','T2'], picks=selected_channels, t_min = 0.2, t_max = 2.0)

# SVC
model_SVC_real = train_and_test_model(X_train, X_test, y_train, y_test, MODEL_SVC_BEST, "SVC real")
joblib.dump(model_SVC_real, r'D:\inżynierka\motor imagery BCI\impl\models\bci_model_svc_real_motion.pkl')

# LDA
model_LDA_real = train_and_test_model(X_train, X_test, y_train, y_test, MODEL_LDA_BEST, "LDA real")
joblib.dump(model_SVC_real, r'D:\inżynierka\motor imagery BCI\impl\models\bci_model_lda_real_motion.pkl')

# RANDOM FOREST
model_RF_real = train_and_test_model(X_train, X_test, y_train, y_test, MODEL_RF, "Random Forest real")
joblib.dump(model_RF_real, r'D:\inżynierka\motor imagery BCI\impl\models\bci_model_rf_real_motion.pkl')

# XGBoost
model_XGB_real = train_and_test_model(X_train, X_test, y_train, y_test, MODEL_XGB, "XGB real")
joblib.dump(model_XGB_real, r'D:\inżynierka\motor imagery BCI\impl\models\bci_model_xgb_real_motion.pkl')

# lightGBM
model_LGBM_real = train_and_test_model(X_train, X_test, y_train, y_test, MODEL_LGBM, "LGBM real")
joblib.dump(model_LGBM_real, r'D:\inżynierka\motor imagery BCI\impl\models\bci_model_lgbm_real_motion.pkl')


# model5 = joblib.load('D:\inżynierka\motor imagery BCI\impl\models\\bci_model_svc_real_motion.pkl')
#
# eval_split("SVC", X_test, y_test, model5)

