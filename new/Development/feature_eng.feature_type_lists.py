import pandas as pd

# Create feature type lists ready for preprocessing pipeline
# These lists are already created in eda.feature_types, we'll use them directly
binary_cols = binary_features.copy()
categorical_cols = categorical_features.copy()
numeric_cols = numeric_features.copy()

print('=== FEATURE LISTS FOR PREPROCESSING ===')
print(f'Binary columns ({len(binary_cols)}): {binary_cols[:3]}...')
print(f'Categorical columns ({len(categorical_cols)}): {categorical_cols[:3]}...')
print(f'Numeric columns ({len(numeric_cols)}): {numeric_cols[:3]}...')
print(f'\nTotal features: {len(binary_cols) + len(categorical_cols) + len(numeric_cols)}')