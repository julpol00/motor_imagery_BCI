from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.data_utils import reshape_eeg



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

MODEL_RF = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

MODEL_XGB = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
])

MODEL_LGBM = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lgbm', LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
])

# pipeline dla danych z feature extraction (ERD/ERS) - zredukowana złożoność modelu
MODEL_LGBM_REGULARIZED = Pipeline(steps=[
    ('reshape', FunctionTransformer(reshape_eeg)),
    ('scaler', StandardScaler()),
    ('lgbm', LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=8,
        max_depth=3,
        min_child_samples=40,
        reg_alpha=0.1,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ))
])

# -----------------------------------------------------------------

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
