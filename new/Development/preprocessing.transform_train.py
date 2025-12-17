# Transform training data
X_train_transformed = final_preprocessor.transform(X_train)

print('✓ Training data transformed')
print(f'  Input shape: {X_train.shape}')
print(f'  Output shape: {X_train_transformed.shape}')