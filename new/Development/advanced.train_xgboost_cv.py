from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

# Calculate class imbalance ratio for scale_pos_weight
_pos_count = y_train.sum()
_neg_count = len(y_train) - _pos_count
xgb_weight_ratio = _neg_count / _pos_count

print(f'Training HistGradientBoosting with class weights')
print(f'Class weight ratio: {xgb_weight_ratio:.2f}')
print('Using 5-Fold Stratified CV...\n')

# Setup HistGradientBoosting classifier (handles NaN natively)
# Using sample_weight for class balancing (equivalent to scale_pos_weight)
xgb_model = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.1,
    max_iter=100,
    random_state=42
)

# Reuse same stratified k-fold from baseline
xgb_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store CV results
xgb_cv_auc_scores = []
xgb_cv_gini_scores = []

# Perform cross-validation
for _fold, (_train_idx, _val_idx) in enumerate(xgb_skf.split(X_train, y_train), 1):
    # Split data
    _X_fold_train = X_train.iloc[_train_idx]
    _y_fold_train = y_train.iloc[_train_idx]
    _X_fold_val = X_train.iloc[_val_idx]
    _y_fold_val = y_train.iloc[_val_idx]
    
    # Create sample weights for class balancing
    _sample_weights = np.where(_y_fold_train == 1, xgb_weight_ratio, 1.0)
    
    # Fit model with sample weights
    xgb_model.fit(_X_fold_train, _y_fold_train, sample_weight=_sample_weights)
    
    # Predict probabilities
    _y_pred_proba = xgb_model.predict_proba(_X_fold_val)[:, 1]
    
    # Calculate ROC-AUC
    _fold_auc = roc_auc_score(_y_fold_val, _y_pred_proba)
    
    # Calculate Gini coefficient
    _fold_gini = 2 * _fold_auc - 1
    
    xgb_cv_auc_scores.append(_fold_auc)
    xgb_cv_gini_scores.append(_fold_gini)
    
    print(f'Fold {_fold}: AUC = {_fold_auc:.4f}, Gini = {_fold_gini:.4f}')

print(f'\n✓ HistGradientBoosting CV completed!')