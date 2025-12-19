import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Rebuild preprocessor inside this block to ensure it's fresh
lr_numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

lr_categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

lr_binary_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

lr_preprocessor = make_column_transformer(
    (lr_numeric_pipeline, numeric_cols),
    (lr_categorical_pipeline, categorical_cols),
    (lr_binary_pipeline, binary_cols),
    verbose_feature_names_out=False
)

# Create full pipeline with preprocessing and logistic regression
baseline_lr_pipeline = Pipeline([
    ('preprocessor', lr_preprocessor),
    ('classifier', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    ))
])

# Setup 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store CV results
cv_auc_scores = []
cv_gini_scores = []

print('Training Baseline Logistic Regression with 5-Fold Stratified CV...\n')

# Perform cross-validation
for _lr_fold_num, (_lr_train_idx, _lr_val_idx) in enumerate(skf.split(X_train, y_train), 1):
    # Split data
    _X_lr_fold_train = X_train.iloc[_lr_train_idx]
    _y_lr_fold_train = y_train.iloc[_lr_train_idx]
    _X_lr_fold_val = X_train.iloc[_lr_val_idx]
    _y_lr_fold_val = y_train.iloc[_lr_val_idx]
    
    # Fit pipeline (preprocessing + model)
    baseline_lr_pipeline.fit(_X_lr_fold_train, _y_lr_fold_train)
    
    # Predict probabilities
    _y_lr_pred_proba = baseline_lr_pipeline.predict_proba(_X_lr_fold_val)[:, 1]
    
    # Calculate ROC-AUC
    _lr_fold_auc = roc_auc_score(_y_lr_fold_val, _y_lr_pred_proba)
    
    # Calculate Gini coefficient: 2*AUC - 1
    _lr_fold_gini = 2 * _lr_fold_auc - 1
    
    cv_auc_scores.append(_lr_fold_auc)
    cv_gini_scores.append(_lr_fold_gini)
    
    print(f'Fold {_lr_fold_num}: AUC = {_lr_fold_auc:.4f}, Gini = {_lr_fold_gini:.4f}')

print(f'\n✓ Cross-validation completed successfully!')