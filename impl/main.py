from sklearn.model_selection import GridSearchCV
import src.config as cfg
from src.models import *

train_ratio = 0.2
selected_channels = ['C3', 'C4', 'CP1', 'CP2', 'FC1', 'FC2']

train_list, test_list = split_train_test_path_list(cfg.IMAGERY_MOTION_PREPROCESSED_PATH, cfg.FILE_PATTERN, train_ratio)
X_train, X_test, y_train, y_test = get_X_and_Y_from_epochs(train_list, test_list, ['T1','T2'], picks=selected_channels, t_min = 0.2, t_max = 2.0)
model = GridSearchCV(MODEL_SVC, cfg.PARAM_GRID_SVC, scoring="roc_auc", cv=5, verbose=3)
model = train_and_test_model(X_train, X_test, y_train, y_test, model)

train_list, test_list = split_train_test_path_list(cfg.REAL_MOTION_PREPROCESSED_PATH, cfg.FILE_PATTERN, train_ratio)
X_train, X_test, y_train, y_test = get_X_and_Y_from_epochs(train_list, test_list, ['T1','T2'], picks=selected_channels, t_min = 0.2, t_max = 2.0)
model2 = GridSearchCV(MODEL_SVC, cfg.PARAM_GRID_SVC, scoring="roc_auc", cv=5, verbose=3)
model2 = train_and_test_model(X_train, X_test, y_train, y_test, model2)

train_list, test_list = split_train_test_path_list(cfg.REAL_MOTION_PREPROCESSED_PATH, cfg.FILE_PATTERN, train_ratio)
X_train, X_test, y_train, y_test = get_X_and_Y_from_epochs(train_list, test_list, ['T1','T2'], picks=selected_channels, t_min = 0.2, t_max = 2.0)
model_3 = GridSearchCV(MODEL_LDA, cfg.PARAM_GRID_LDA, scoring="roc_auc", cv=5, verbose=3)
model3 = train_and_test_model(X_train, X_test, y_train, y_test, model_3)

train_list, test_list = split_train_test_path_list(cfg.IMAGERY_MOTION_PREPROCESSED_PATH, cfg.FILE_PATTERN, train_ratio)
X_train, X_test, y_train, y_test = get_X_and_Y_from_epochs(train_list, test_list, ['T1','T2'], picks=selected_channels, t_min = 0.2, t_max = 2.0)
model_4 = GridSearchCV(MODEL_LDA, cfg.PARAM_GRID_LDA, scoring="roc_auc", cv=5, verbose=3)
model4 = train_and_test_model(X_train, X_test, y_train, y_test, model_4)
