import pandas as pd

# Generate predictions on test set using X_test_transformed (preprocessed)
print('Generating predictions on test set...')
print(f'Test set size: {X_test_transformed.shape[0]:,} samples')
print(f'Preprocessed features: {X_test_transformed.shape[1]}')

# Predict probabilities for positive class using transformed test data
test_predictions = final_model.predict_proba(X_test_transformed)[:, 1]

print(f'✓ Predictions generated for {len(test_predictions):,} samples')
print(f'Prediction range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]')
print(f'Mean prediction: {test_predictions.mean():.4f}')