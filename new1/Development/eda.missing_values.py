import pandas as pd

# Check missing values in training data
train_missing = training_df.isnull().sum()
train_missing_pct = (train_missing / len(training_df) * 100).round(2)

print('=== TRAINING DATA MISSING VALUES ===')
missing_cols = train_missing[train_missing > 0].sort_values(ascending=False)
if len(missing_cols) > 0:
    for col in missing_cols.index[:10]:  # Show top 10
        print(f'{col}: {train_missing[col]} ({train_missing_pct[col]}%)')
    if len(missing_cols) > 10:
        print(f'... and {len(missing_cols) - 10} more columns with missing values')
else:
    print('No missing values found')

# Check missing values in test data
test_missing = test_df.isnull().sum()
test_missing_pct = (test_missing / len(test_df) * 100).round(2)

print('\n=== TEST DATA MISSING VALUES ===')
missing_cols_test = test_missing[test_missing > 0].sort_values(ascending=False)
if len(missing_cols_test) > 0:
    for col in missing_cols_test.index[:10]:  # Show top 10
        print(f'{col}: {test_missing[col]} ({test_missing_pct[col]}%)')
    if len(missing_cols_test) > 10:
        print(f'... and {len(missing_cols_test) - 10} more columns with missing values')
else:
    print('No missing values found')