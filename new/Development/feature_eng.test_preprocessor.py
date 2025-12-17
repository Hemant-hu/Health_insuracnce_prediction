import pandas as pd
import numpy as np

# Create a small test sample to verify the pipeline works
test_sample = X_train.head(100).copy()

# Fit and transform the test sample
preprocessor.fit(test_sample)
transformed_test = preprocessor.transform(test_sample)

# Get output shape and verify transformation
print(f'✓ Pipeline tested successfully!')
print(f'  Input shape: {test_sample.shape}')
print(f'  Output shape: {transformed_test.shape}')
print(f'  Output type: {type(transformed_test).__name__}')

# Count expected output features after one-hot encoding
# Note: actual count depends on unique values in categorical columns
print(f'\n✓ Expected features breakdown:')
print(f'  - Numeric features: {len(numeric_cols)}')
print(f'  - Binary features (passthrough): {len(binary_cols)}')
print(f'  - Categorical (after encoding): {transformed_test.shape[1] - len(numeric_cols) - len(binary_cols)}')