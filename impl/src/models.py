from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.data_utils import reshape_eeg

# -----------------------------------------------------------------
# Tuned pipelines — best hyperparameters from Optuna
# -----------------------------------------------------------------

# === BASE features — IMAGERY ===

MODEL_LDA_BASE_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.8631092341187571)),
])

MODEL_SVC_BASE_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, cache_size=1000, C=0.9073869074375944, gamma=0.00022140824330647477)),
])

MODEL_RF_BASE_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=493, max_depth=14, min_samples_split=14,
                                   min_samples_leaf=10, random_state=42)),
])

MODEL_XGB_BASE_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(n_estimators=409, max_depth=5, learning_rate=0.023743688252326763,
                           subsample=0.6069038872655155, colsample_bytree=0.6787615838398218,
                           reg_alpha=0.3766794456528051, reg_lambda=9.970740254624836,
                           min_child_weight=33, random_state=42)),
])

MODEL_LGBM_BASE_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lgbm', LGBMClassifier(n_estimators=432, learning_rate=0.04537624012525428, num_leaves=81,
                             min_child_samples=71, reg_alpha=0.004904431769207067,
                             reg_lambda=0.03649049389290538, random_state=42)),
])

# === BASE features — REAL ===

MODEL_LDA_BASE_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.8359246313986887)),
])

MODEL_SVC_BASE_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, cache_size=1000, C=1.2845684066929972, gamma=0.00027804699361002575)),
])

MODEL_RF_BASE_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=427, max_depth=16, min_samples_split=14,
                                   min_samples_leaf=8, random_state=42)),
])

MODEL_XGB_BASE_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(n_estimators=453, max_depth=6, learning_rate=0.02530328160496185,
                           subsample=0.503692880793686, colsample_bytree=0.9526245715751707,
                           reg_alpha=0.07358429550536165, reg_lambda=4.554861522093897,
                           min_child_weight=10, random_state=42)),
])

MODEL_LGBM_BASE_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lgbm', LGBMClassifier(n_estimators=453, learning_rate=0.07120496138139351, num_leaves=24,
                             min_child_samples=46, reg_alpha=0.07474783166757452,
                             reg_lambda=0.06522980622596367, random_state=42)),
])

# === MRP features — IMAGERY ===

MODEL_LDA_MRP_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis(solver='lsqr')),
])

MODEL_SVC_MRP_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, cache_size=1000, C=1.8430319304935165, gamma=0.002248228903524372)),
])

MODEL_RF_MRP_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=301, max_depth=20, min_samples_split=7,
                                   min_samples_leaf=5, random_state=42)),
])

MODEL_XGB_MRP_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(n_estimators=440, max_depth=4, learning_rate=0.03742861357353767,
                           subsample=0.6682865645654519, colsample_bytree=0.7665492580733263,
                           reg_alpha=0.46118667459483653, reg_lambda=0.06503417415850897,
                           min_child_weight=48, random_state=42)),
])

MODEL_LGBM_MRP_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lgbm', LGBMClassifier(n_estimators=164, learning_rate=0.03787406411656293, num_leaves=24,
                             min_child_samples=81, reg_alpha=0.010911719567715325,
                             reg_lambda=0.0054821511843596345, random_state=42)),
])

# === MRP features — REAL ===

MODEL_LDA_MRP_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.006167558441199635)),
])

MODEL_SVC_MRP_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, cache_size=1000, C=1.6009979172640463, gamma=0.0032294941272017633)),
])

MODEL_RF_MRP_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=499, max_depth=12, min_samples_split=2,
                                   min_samples_leaf=7, random_state=42)),
])

MODEL_XGB_MRP_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(n_estimators=430, max_depth=6, learning_rate=0.020249029180102622,
                           subsample=0.9546747249519072, colsample_bytree=0.5008964497339938,
                           reg_alpha=0.029013885290511986, reg_lambda=3.356313876819168,
                           min_child_weight=43, random_state=42)),
])

MODEL_LGBM_MRP_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lgbm', LGBMClassifier(n_estimators=290, learning_rate=0.020393656671124357, num_leaves=51,
                             min_child_samples=77, reg_alpha=0.17548962674105306,
                             reg_lambda=0.007699157079402251, random_state=42)),
])

# === ERD/ERS features — IMAGERY ===

MODEL_LDA_ERD_ERS_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.6406325413711079)),
])

MODEL_SVC_ERD_ERS_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, cache_size=1000, C=81.57235658878677, gamma=0.00038298538747146086)),
])

MODEL_RF_ERD_ERS_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=469, max_depth=4, min_samples_split=7,
                                   min_samples_leaf=2, random_state=42)),
])

MODEL_XGB_ERD_ERS_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(n_estimators=494, max_depth=2, learning_rate=0.010578806861496972,
                           subsample=0.5595419213097756, colsample_bytree=0.9986096771081043,
                           reg_alpha=0.03103237968412601, reg_lambda=0.0010830060571461902,
                           min_child_weight=32, random_state=42)),
])

MODEL_LGBM_ERD_ERS_IMAGERY = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lgbm', LGBMClassifier(n_estimators=107, learning_rate=0.011981468615529246, num_leaves=43,
                             min_child_samples=100, reg_alpha=0.001106021930907998,
                             reg_lambda=0.21281483592664555, random_state=42)),
])

# === ERD/ERS features — REAL ===

MODEL_LDA_ERD_ERS_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.562998472633145)),
])

MODEL_SVC_ERD_ERS_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, cache_size=1000, C=0.14863545045225227, gamma=0.0024688694858781947)),
])

MODEL_RF_ERD_ERS_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=370, max_depth=4, min_samples_split=11,
                                   min_samples_leaf=6, random_state=42)),
])

MODEL_XGB_ERD_ERS_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(n_estimators=225, max_depth=2, learning_rate=0.01744279822715601,
                           subsample=0.5633726763208353, colsample_bytree=0.9291769778777409,
                           reg_alpha=4.870635756580452, reg_lambda=0.005860514999570948,
                           min_child_weight=16, random_state=42)),
])

MODEL_LGBM_ERD_ERS_REAL = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lgbm', LGBMClassifier(n_estimators=432, learning_rate=0.025177538185616476, num_leaves=67,
                             min_child_samples=82, reg_alpha=1.974181578534989,
                             reg_lambda=1.241391272612945, random_state=42)),
])
