import pandas as pd

# Check if there are any NaN values in X_train that shouldn't be there
nan_counts = X_train.isnull().sum()
_cols_with_nans = nan_counts[nan_counts > 0]

print(f'Checking X_train for missing values...')
print(f'Shape: {X_train.shape}')
print(f'\nColumns with NaN values: {len(_cols_with_nans)}')

if len(_cols_with_nans) > 0:
    print(f'\nTop columns with NaN:')
    for _col, _count in _cols_with_nans.head(10).items():
        _pct = (_count / len(X_train)) * 100
        print(f'  {_col}: {_count:,} ({_pct:.2f}%)')
else:
    print('No NaN values found!')