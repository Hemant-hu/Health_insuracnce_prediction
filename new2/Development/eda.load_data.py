import pandas as pd

# Load training and test datasets
training_df = pd.read_csv('training_data.csv')
test_df = pd.read_csv('test_data.csv')

print(f'Training data loaded: {training_df.shape[0]} rows, {training_df.shape[1]} columns')
print(f'Test data loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns')