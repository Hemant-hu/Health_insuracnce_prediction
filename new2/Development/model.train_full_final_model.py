from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np
import pandas as pd

print('Training final gradient boosting model on full training set...\n')

# Apply same feature engineering as CV model
X_train_final = X_train.copy()

# Add interaction features
for i, col1 in enumerate(numeric_cols[:5]):
    for col2 in numeric_cols[i+1:6]:
        X_train_final[f'{col1}_x_{col2}'] = X_train[col1] * X_train[col2]

# Add squared features
for col in numeric_cols[:8]:
    X_train_final[f'{col}_sq'] = X_train[col] ** 2

print(f'Final training features: {X_train_final.shape[1]}')

# Calculate class weights
_pos = y_train.sum()
_neg = len(y_train) - _pos
weight_ratio = _neg / _pos
sample_weights_full = np.where(y_train == 1, weight_ratio, 1.0)

# Train final model on full training data
final_trained_model = HistGradientBoostingClassifier(
    max_depth=10,
    learning_rate=0.03,
    max_iter=500,
    min_samples_leaf=15,
    l2_regularization=0.5,
    max_leaf_nodes=50,
    random_state=42
)

final_trained_model.fit(X_train_final, y_train, sample_weight=sample_weights_full)

print(f'✓ Model trained on {X_train_final.shape[0]:,} samples')
print(f'  Features: {X_train_final.shape[1]}')
print(f'  Class weight ratio: {weight_ratio:.2f}')
print(f'  Model ready for test predictions')