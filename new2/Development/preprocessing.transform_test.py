# Transform test data using the fitted pipeline
X_test_transformed = final_preprocessor.transform(X_test)

print('✓ Test data transformed')
print(f'  Input shape: {X_test.shape}')
print(f'  Output shape: {X_test_transformed.shape}')