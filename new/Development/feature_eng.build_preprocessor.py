from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Build preprocessing pipeline for numeric features
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Build preprocessing pipeline for categorical features
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Build preprocessing pipeline for binary features (add imputation)
binary_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Combine all transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_pipeline, numeric_cols),
        ('categorical', categorical_pipeline, categorical_cols),
        ('binary', binary_pipeline, binary_cols)
    ],
    verbose_feature_names_out=False
)

print('✓ Preprocessing pipeline created successfully!')
print(f'  - Numeric ({len(numeric_cols)} cols): median imputation + scaling')
print(f'  - Categorical ({len(categorical_cols)} cols): most_frequent imputation + one-hot encoding')
print(f'  - Binary ({len(binary_cols)} cols): most_frequent imputation')