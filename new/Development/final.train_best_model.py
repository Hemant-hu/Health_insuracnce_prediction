from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

# The best model is LightGBM-style HistGradientBoosting with optimal hyperparameters
# Retrain on full PREPROCESSED training dataset with exact parameters from CV

# Calculate class imbalance ratio for sample weights
_pos_count = y_train.sum()
_neg_count = len(y_train) - _pos_count
_weight_ratio = _neg_count / _pos_count

print('=' * 80)
print('TRAINING FINAL MODEL ON FULL PREPROCESSED TRAINING DATASET')
print('=' * 80)
print(f'Best Model: HistGradientBoosting (LightGBM-style)')
print(f'Expected CV Gini: 0.2729 ± 0.0081')
print(f'Training samples: {X_train_transformed.shape[0]:,}')
print(f'Preprocessed features: {X_train_transformed.shape[1]}')
print(f'Class weight ratio: {_weight_ratio:.2f}')
print('Optimal hyperparameters: max_depth=7, learning_rate=0.05, max_iter=200, l2_regularization=1.0')
print('=' * 80)

# Initialize model with optimal hyperparameters from CV (exact match)
final_model = HistGradientBoostingClassifier(
    loss="log_loss",
    max_depth=7,
    learning_rate=0.05,
    max_iter=200,
    l2_regularization=1.0,
    random_state=42
)

# Create sample weights for class balancing
_sample_weights = np.where(y_train == 1, _weight_ratio, 1.0)

# Train on full preprocessed training dataset
final_model.fit(X_train_transformed, y_train, sample_weight=_sample_weights)

print('\n✓ Final model trained successfully on full preprocessed training dataset!')
print(f'  Model trained on {X_train_transformed.shape[0]:,} samples')
print(f'  Using {X_train_transformed.shape[1]} preprocessed features')
print('  Ready for predictions on test set.')