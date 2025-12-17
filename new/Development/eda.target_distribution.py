# Analyze target variable distribution
target_counts = training_df['target'].value_counts().sort_index()
target_pcts = (training_df['target'].value_counts(normalize=True) * 100).sort_index()

print('=== TARGET DISTRIBUTION ===')
for val in target_counts.index:
    print(f'Class {val}: {target_counts[val]} ({target_pcts[val]:.2f}%)')

# Calculate imbalance ratio
imbalance_ratio = target_counts.max() / target_counts.min()
print(f'\nImbalance ratio: {imbalance_ratio:.2f}:1')