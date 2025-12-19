import pandas as pd

print('Updating submission.csv with calibrated probabilities...\n')

# Create submission dataframe with calibrated probabilities
submission_calibrated = pd.DataFrame({
    'id': test_df['id'],
    'target': calibrated_probabilities
})

# Save to CSV
submission_calibrated.to_csv('submission.csv', index=False)

print(f'✅ Submission file updated with calibrated predictions')
print(f'   Records: {len(submission_calibrated):,}')
print(f'   Mean claim rate: {calibrated_probabilities.mean():.6f} ({calibrated_probabilities.mean()*100:.2f}%)')
print(f'   Target range (3-10%): {"✓ ACHIEVED" if 0.03 <= calibrated_probabilities.mean() <= 0.10 else "✗ FAILED"}')
print(f'\nFirst 10 rows:')
print(submission_calibrated.head(10).to_string(index=False))
print(f'\nLast 5 rows:')
print(submission_calibrated.tail(5).to_string(index=False))
print(f'\n📁 File saved: submission.csv')
