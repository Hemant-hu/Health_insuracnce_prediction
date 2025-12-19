import pandas as pd
import numpy as np

print('Generating predictions on test dataset...\n')

# Prepare test features (same transformations as training)
X_test = test_df.drop(columns=['id'])

# Apply same feature engineering
X_test_enhanced = X_test.copy()

# Add interaction features
for i, col1 in enumerate(numeric_cols[:5]):
    for col2 in numeric_cols[i+1:6]:
        X_test_enhanced[f'{col1}_x_{col2}'] = X_test[col1] * X_test[col2]

# Add squared features
for col in numeric_cols[:8]:
    X_test_enhanced[f'{col}_sq'] = X_test[col] ** 2

print(f'Test features: {X_test_enhanced.shape[1]} (matches training: {X_train_final.shape[1]})')

# Generate probability predictions
test_probabilities = final_trained_model.predict_proba(X_test_enhanced)[:, 1]

print(f'\n✓ Predictions generated for {len(test_probabilities):,} test samples')
print(f'Prediction statistics:')
print(f'  Min: {test_probabilities.min():.6f}')
print(f'  Max: {test_probabilities.max():.6f}')
print(f'  Mean: {test_probabilities.mean():.6f}')
print(f'  Median: {np.median(test_probabilities):.6f}')

# Check distribution of predictions
_pred_3_10_pct = ((test_probabilities >= 0.03) & (test_probabilities <= 0.10)).sum() / len(test_probabilities) * 100
print(f'  % predictions in target range (3-10%): {_pred_3_10_pct:.1f}%')