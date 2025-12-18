from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

# Calculate class imbalance ratio
_pos_count = y_train.sum()
_neg_count = len(y_train) - _pos_count
cat_weight_ratio = _neg_count / _pos_count

print(f'Training CatBoost-style HistGradientBoosting with extreme params')
print(f'Class weight ratio: {cat_weight_ratio:.2f}')
print('Using 5-Fold Stratified CV...\n')

# Setup third HistGradientBoosting with different hyperparameters
# Simulating CatBoost's approach with stronger regularization
cat_model = HistGradientBoostingClassifier(
    max_depth=4,
    learning_rate=0.15,
    max_iter=80,
    l2_regularization=1.0,
    random_state=44
)

# Stratified k-fold
cat_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store CV results
cat_cv_auc_scores = []
cat_cv_gini_scores = []

# Perform cross-validation
for _fold, (_train_idx, _val_idx) in enumerate(cat_skf.split(X_train, y_train), 1):
    # Split data
    _X_fold_train = X_train.iloc[_train_idx]
    _y_fold_train = y_train.iloc[_train_idx]
    _X_fold_val = X_train.iloc[_val_idx]
    _y_fold_val = y_train.iloc[_val_idx]
    
    # Create sample weights for class balancing
    _sample_weights = np.where(_y_fold_train == 1, cat_weight_ratio, 1.0)
    
    # Fit model with sample weights
    cat_model.fit(_X_fold_train, _y_fold_train, sample_weight=_sample_weights)
    
    # Predict probabilities
    _y_pred_proba = cat_model.predict_proba(_X_fold_val)[:, 1]
    
    # Calculate ROC-AUC
    _fold_auc = roc_auc_score(_y_fold_val, _y_pred_proba)
    
    # Calculate Gini coefficient
    _fold_gini = 2 * _fold_auc - 1
    
    cat_cv_auc_scores.append(_fold_auc)
    cat_cv_gini_scores.append(_fold_gini)
    
    print(f'Fold {_fold}: AUC = {_fold_auc:.4f}, Gini = {_fold_gini:.4f}')

print(f'\n✓ CatBoost-style HistGradientBoosting CV completed!')