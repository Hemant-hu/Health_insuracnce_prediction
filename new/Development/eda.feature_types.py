import pandas as pd
import numpy as np

# Analyze feature types (exclude 'id' column)
feature_cols = [col for col in training_df.columns if col not in ['id', 'target']]

binary_features = []
categorical_features = []
numeric_features = []

for col in feature_cols:
    unique_vals = training_df[col].nunique()
    
    if unique_vals == 2:
        binary_features.append(col)
    elif unique_vals < 20 and training_df[col].dtype in ['int64', 'object']:
        categorical_features.append(col)
    else:
        numeric_features.append(col)

print('=== FEATURE TYPE BREAKDOWN ===')
print(f'Binary features (2 unique values): {len(binary_features)}')
print(f'Categorical features (3-19 unique values): {len(categorical_features)}')
print(f'Numeric features (>=20 unique values or float): {len(numeric_features)}')
print(f'\nTotal features: {len(feature_cols)}')