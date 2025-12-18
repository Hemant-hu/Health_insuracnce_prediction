# Extract features from test dataset (test_df has no target column)
X_test = test_df.drop(columns=['id'])

print(f'✓ Test features extracted')
print(f'  X_test shape: {X_test.shape}')