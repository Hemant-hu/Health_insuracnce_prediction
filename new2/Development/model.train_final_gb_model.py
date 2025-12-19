from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

# Engineer additional features to improve model performance
print('Creating additional features to improve Gini score...\n')

# Feature engineering: interaction and polynomial features for key variables
X_train_enhanced = X_train.copy()

# Add interaction features for highly correlated numeric features
for i, col1 in enumerate(numeric_cols[:5]):
    for col2 in numeric_cols[i+1:6]:
        X_train_enhanced[f'{col1}_x_{col2}'] = X_train[col1] * X_train[col2]

# Add squared features for numeric columns
for col in numeric_cols[:8]:
    X_train_enhanced[f'{col}_sq'] = X_train[col] ** 2

print(f'Enhanced feature set: {X_train_enhanced.shape[1]} features (from {X_train.shape[1]})')

# Calculate class imbalance ratio
_pos_count = y_train.sum()
_neg_count = len(y_train) - _pos_count
final_weight_ratio = _neg_count / _pos_count

print(f'Class weight ratio: {final_weight_ratio:.2f}')
print('Training final HistGradientBoosting model with 5-Fold CV...\n')

# Enhanced model with better hyperparameters
final_gb_model = HistGradientBoostingClassifier(
    max_depth=10,
    learning_rate=0.03,
    max_iter=500,
    min_samples_leaf=15,
    l2_regularization=0.5,
    max_leaf_nodes=50,
    random_state=42
)

# Cross-validation
final_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_cv_auc_scores = []
final_cv_gini_scores = []

for _fold, (_train_idx, _val_idx) in enumerate(final_skf.split(X_train_enhanced, y_train), 1):
    _X_fold_train = X_train_enhanced.iloc[_train_idx]
    _y_fold_train = y_train.iloc[_train_idx]
    _X_fold_val = X_train_enhanced.iloc[_val_idx]
    _y_fold_val = y_train.iloc[_val_idx]
    
    _sample_weights = np.where(_y_fold_train == 1, final_weight_ratio, 1.0)
    
    final_gb_model.fit(_X_fold_train, _y_fold_train, sample_weight=_sample_weights)
    _y_pred_proba = final_gb_model.predict_proba(_X_fold_val)[:, 1]
    
    _fold_auc = roc_auc_score(_y_fold_val, _y_pred_proba)
    _fold_gini = 2 * _fold_auc - 1
    
    final_cv_auc_scores.append(_fold_auc)
    final_cv_gini_scores.append(_fold_gini)
    
    print(f'Fold {_fold}: AUC = {_fold_auc:.4f}, Gini = {_fold_gini:.4f}')

final_mean_gini = np.mean(final_cv_gini_scores)
final_mean_auc = np.mean(final_cv_auc_scores)

print(f'\n✓ Final Model CV completed!')
print(f'Mean Gini: {final_mean_gini:.4f} | Mean AUC: {final_mean_auc:.4f}')

if final_mean_gini > 0.3:
    print(f'\n✅ SUCCESS: Gini {final_mean_gini:.4f} exceeds target threshold of 0.3')
else:
    print(f'\n⚠️  Gini {final_mean_gini:.4f} is below target 0.3, but close to data limits')