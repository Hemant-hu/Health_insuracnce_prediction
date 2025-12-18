import numpy as np

print('=' * 60)
print('PREPROCESSING PIPELINE SUMMARY')
print('=' * 60)
print()

# Pipeline details
print('Pipeline Configuration:')
print('  • Numeric features (16):')
print('    - SimpleImputer(strategy=median)')
print('    - StandardScaler()')
print()
print('  • Categorical features (11):')
print('    - SimpleImputer(strategy=most_frequent)')
print('    - OneHotEncoder(handle_unknown=ignore)')
print()
print('  • Binary features (23):')
print('    - Passthrough (no transformation)')
print()

# Transformation results
print('Transformation Results:')
print(f'  Training set:')
print(f'    - Input shape:  {X_train.shape}')
print(f'    - Output shape: {X_train_transformed.shape}')
print(f'    - Data type:    {type(X_train_transformed).__name__}')
print()
print(f'  Test set:')
print(f'    - Input shape:  {X_test.shape}')
print(f'    - Output shape: {X_test_transformed.shape}')
print(f'    - Data type:    {type(X_test_transformed).__name__}')
print()

print('Feature Breakdown:')
print(f'  Original features: 50')
print(f'  After transformation: {X_train_transformed.shape[1]}')
print(f'    - Numeric (scaled): 16')
print(f'    - Binary (unchanged): 23')
print(f'    - Categorical (one-hot): {X_train_transformed.shape[1] - 16 - 23}')
print()

print('✓ Pipeline ready for modeling!')
print('=' * 60)