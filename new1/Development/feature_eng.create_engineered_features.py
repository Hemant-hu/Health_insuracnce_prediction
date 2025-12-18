import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

print("=" * 90)
print("ADVANCED FEATURE ENGINEERING - SECOND WAVE")
print("=" * 90)

# ====================================
# 1. BASELINE MUTUAL INFORMATION
# ====================================
print("\n1. Computing baseline mutual information scores...")

_X_filled = X_train.fillna(X_train.median())
mi_scores_baseline = mutual_info_classif(_X_filled, y_train, random_state=42, n_neighbors=5)
mi_df_baseline = pd.DataFrame({
    'feature': X_train.columns,
    'mi_score': mi_scores_baseline
}).sort_values('mi_score', ascending=False)

top_features_baseline = mi_df_baseline.head(10)['feature'].tolist()
print(f"   Top 10 features: {', '.join(top_features_baseline[:5])}...")

# ====================================
# 2. TARGET ENCODING WITH CROSS-VALIDATION
# ====================================
print("\n2. Creating target-encoded features with CV...")

eng_X_train_v2 = X_train.copy()
eng_X_test_v2 = X_test.copy()

_target_enc_count = 0
_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Target encode categorical features with CV to prevent leakage
for cat_feat in categorical_features[:5]:  # Top 5 categorical
    te_col_name = f'target_enc_{cat_feat}'
    
    # Initialize column with zeros
    eng_X_train_v2[te_col_name] = 0.0
    
    # CV encoding for train
    for _fold_idx, (_train_idx, _val_idx) in enumerate(_skf.split(X_train, y_train)):
        _X_fold_train = X_train.iloc[_train_idx]
        _y_fold_train = y_train.iloc[_train_idx]
        _X_fold_val = X_train.iloc[_val_idx]
        
        # Calculate target mean per category
        _target_sums = pd.Series(_y_fold_train.values, index=_X_fold_train.index).groupby(_X_fold_train[cat_feat]).sum()
        _target_counts = _X_fold_train.groupby(cat_feat)[cat_feat].count()
        _target_encoding = (_target_sums / _target_counts).to_dict()
        
        # Apply to validation fold
        eng_X_train_v2.loc[_val_idx, te_col_name] = _X_fold_val[cat_feat].map(_target_encoding).fillna(y_train.mean())
    
    # For test: use full training data
    _target_encoding_full = y_train.groupby(X_train[cat_feat]).mean().to_dict()
    eng_X_test_v2[te_col_name] = X_test[cat_feat].map(_target_encoding_full).fillna(y_train.mean())
    
    _target_enc_count += 1

print(f"   Created {_target_enc_count} target-encoded features with CV")

# ====================================
# 3. CLUSTERING-BASED FEATURES (K-MEANS DISTANCES)
# ====================================
print("\n3. Creating clustering-based features...")

# Use known numeric features directly
_known_numeric = ['feature_2', 'feature_9', 'feature_10', 'feature_12', 'feature_24', 'feature_25']
_X_cluster_train = eng_X_train_v2[_known_numeric].fillna(eng_X_train_v2[_known_numeric].median())
_X_cluster_test = eng_X_test_v2[_known_numeric].fillna(eng_X_test_v2[_known_numeric].median())

# Standardize for clustering
_scaler_cluster = StandardScaler()
_X_cluster_train_scaled = _scaler_cluster.fit_transform(_X_cluster_train)
_X_cluster_test_scaled = _scaler_cluster.transform(_X_cluster_test)

# Fit K-means with multiple cluster counts
_cluster_count = 0
for _n_clusters in [3, 5, 8]:
    _kmeans = KMeans(n_clusters=_n_clusters, random_state=42, n_init=10)
    _cluster_labels_train = _kmeans.fit_predict(_X_cluster_train_scaled)
    _cluster_labels_test = _kmeans.predict(_X_cluster_test_scaled)
    
    # Add cluster labels as features
    eng_X_train_v2[f'cluster_{_n_clusters}'] = _cluster_labels_train
    eng_X_test_v2[f'cluster_{_n_clusters}'] = _cluster_labels_test
    
    # Add distance to nearest cluster center
    _distances_train = _kmeans.transform(_X_cluster_train_scaled).min(axis=1)
    _distances_test = _kmeans.transform(_X_cluster_test_scaled).min(axis=1)
    
    eng_X_train_v2[f'cluster_dist_{_n_clusters}'] = _distances_train
    eng_X_test_v2[f'cluster_dist_{_n_clusters}'] = _distances_test
    
    _cluster_count += 2

print(f"   Created {_cluster_count} clustering features (labels + distances)")

# ====================================
# 4. FEATURE CROSSES (TOP-10 PAIRS)
# ====================================
print("\n4. Creating feature crosses for all top-10 pairs...")

_cross_count = 0
_top10_features = [f for f in top_features_baseline[:10]]

for _idx_i in range(len(_top10_features)):
    for _idx_j in range(_idx_i+1, len(_top10_features)):
        feat_i = _top10_features[_idx_i]
        feat_j = _top10_features[_idx_j]
        
        cross_name = f'cross_{feat_i}_{feat_j}'
        eng_X_train_v2[cross_name] = eng_X_train_v2[feat_i] * eng_X_train_v2[feat_j]
        eng_X_test_v2[cross_name] = eng_X_test_v2[feat_i] * eng_X_test_v2[feat_j]
        _cross_count += 1

print(f"   Created {_cross_count} feature crosses")

# ====================================
# 5. RATIO AND DIVISION FEATURES
# ====================================
print("\n5. Creating ratio/division features between numeric pairs...")

_ratio_count = 0
_top_numeric = ['feature_2', 'feature_9', 'feature_10', 'feature_12', 'feature_24', 'feature_25']

for _idx_i in range(len(_top_numeric)):
    for _idx_j in range(_idx_i+1, len(_top_numeric)):
        feat_i = _top_numeric[_idx_i]
        feat_j = _top_numeric[_idx_j]
        
        # Ratio feat_i / feat_j
        ratio_name = f'ratio_{feat_i}_{feat_j}'
        eng_X_train_v2[ratio_name] = eng_X_train_v2[feat_i] / (eng_X_train_v2[feat_j].abs() + 1e-5)
        eng_X_test_v2[ratio_name] = eng_X_test_v2[feat_i] / (eng_X_test_v2[feat_j].abs() + 1e-5)
        _ratio_count += 1

print(f"   Created {_ratio_count} ratio features")

# ====================================
# 6. LOG AND SQRT TRANSFORMATIONS
# ====================================
print("\n6. Creating log and sqrt transformations...")

_transform_count = 0
for feat in _top_numeric[:5]:
    # Log transform (handle zeros and negatives)
    log_name = f'log_{feat}'
    eng_X_train_v2[log_name] = np.log1p(np.abs(eng_X_train_v2[feat]))
    eng_X_test_v2[log_name] = np.log1p(np.abs(eng_X_test_v2[feat]))
    
    # Sqrt transform
    sqrt_name = f'sqrt_{feat}'
    eng_X_train_v2[sqrt_name] = np.sqrt(np.abs(eng_X_train_v2[feat]))
    eng_X_test_v2[sqrt_name] = np.sqrt(np.abs(eng_X_test_v2[feat]))
    
    _transform_count += 2

print(f"   Created {_transform_count} log/sqrt transformation features")

# ====================================
# 7. COUNT NEW FEATURES
# ====================================
_new_feature_cols_v2 = [c for c in eng_X_train_v2.columns if c not in X_train.columns]
print(f"\n7. Total new features created: {len(_new_feature_cols_v2)}")

# ====================================
# 8. FEATURE SELECTION - MUTUAL INFORMATION (TOP 80)
# ====================================
print("\n8. Performing feature selection to keep top 80 features...")

# Compute MI for ALL features (original + new)
_X_all_filled = eng_X_train_v2.fillna(eng_X_train_v2.median())
mi_scores_all = mutual_info_classif(_X_all_filled, y_train, random_state=42, n_neighbors=5)
mi_df_all = pd.DataFrame({
    'feature': eng_X_train_v2.columns,
    'mi_score': mi_scores_all
}).sort_values('mi_score', ascending=False)

# Select top 80 features
selected_top80_features = mi_df_all.head(80)['feature'].tolist()

# Create final datasets
eng_X_train_v2_final = eng_X_train_v2[selected_top80_features]
eng_X_test_v2_final = eng_X_test_v2[selected_top80_features]

print(f"   Selected top 80 features via mutual information")
print(f"   Top feature: {mi_df_all.iloc[0]['feature']} (MI: {mi_df_all.iloc[0]['mi_score']:.6f})")

# Count new features in top 80
_new_in_top80 = [f for f in selected_top80_features if f not in X_train.columns]
print(f"   New engineered features in top 80: {len(_new_in_top80)}")

# ====================================
# 9. ANALYZE TOP-20 FEATURES
# ====================================
print("\n9. Analyzing top-20 features...")

top20_features = mi_df_all.head(20)['feature'].tolist()
_new_in_top20 = [f for f in top20_features if f not in X_train.columns]

print(f"   New engineered features in top-20: {len(_new_in_top20)}")
if len(_new_in_top20) > 0:
    print(f"   Examples: {', '.join(_new_in_top20[:5])}")

# ====================================
# 10. SUMMARY
# ====================================
print("\n" + "=" * 90)
print("SECOND WAVE FEATURE ENGINEERING SUMMARY")
print("=" * 90)
print(f"✓ Target encoding (CV):          {_target_enc_count}")
print(f"✓ Clustering features:            {_cluster_count}")
print(f"✓ Feature crosses (top-10):       {_cross_count}")
print(f"✓ Ratio/division features:        {_ratio_count}")
print(f"✓ Log/sqrt transformations:       {_transform_count}")
print(f"✓ Total new features:             {len(_new_feature_cols_v2)}")
print(f"✓ Final feature set size:         {len(selected_top80_features)}")
print(f"✓ New features in top-80:         {len(_new_in_top80)}")
print(f"✓ New features in top-20:         {len(_new_in_top20)}") 
print(f"✓ Dataset shape:                  {eng_X_train_v2_final.shape}")
print("=" * 90)