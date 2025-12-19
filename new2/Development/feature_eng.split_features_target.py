import pandas as pd

# Separate features and target from training data
X_train = training_df.drop(columns=['id', 'target'])
y_train = training_df['target']

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'Target distribution: {y_train.value_counts().to_dict()}')