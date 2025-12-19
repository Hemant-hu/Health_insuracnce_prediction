from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Build fresh preprocessing pipeline
final_numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

final_categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

final_binary_pipeline = Pipeline([
    ('passthrough', 'passthrough')
])

# Combine all transformers
final_preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', final_numeric_pipeline, numeric_cols),
        ('categorical', final_categorical_pipeline, categorical_cols),
        ('binary', final_binary_pipeline, binary_cols)
    ],
    verbose_feature_names_out=False
)

# Fit pipeline on training data
final_preprocessor.fit(X_train)

print('✓ Preprocessing pipeline fitted on training data')
print(f'  Training samples: {X_train.shape[0]:,}')
print(f'  Features: {X_train.shape[1]}')