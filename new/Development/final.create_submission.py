import pandas as pd

# Create submission dataframe using actual test IDs from test_df
# test_df has an 'id' column that must be preserved in submission
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'target': test_predictions
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

print(f'✓ Submission file created: submission.csv')
print(f'Shape: {submission_df.shape}')
print(f'Columns: {list(submission_df.columns)}')
print(f'\nFirst 5 rows:')
print(submission_df.head().to_string(index=False))