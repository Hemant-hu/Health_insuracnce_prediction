import pandas as pd

# Get summary statistics for numeric features only
numeric_cols = training_df.select_dtypes(include=['float64', 'int64']).columns
numeric_cols = [col for col in numeric_cols if col not in ['id', 'target']]

summary_stats = training_df[numeric_cols].describe()

print('=== SUMMARY STATISTICS (Numeric Features) ===')
print(f'Features analyzed: {len(numeric_cols)}')
print(f'\nKey statistics for first 5 features:')
print(summary_stats.iloc[:, :5].to_string())
print('\n...')
print(f'\nFull statistics available for all {len(numeric_cols)} numeric features')