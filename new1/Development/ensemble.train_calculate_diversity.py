import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("DIVERSE NON-TREE ENSEMBLE MODELS - CV SCORES & DIVERSITY")
print("=" * 90)

# Use X_train instead (50 features) - simpler, already available
_X_filled = X_train.fillna(X_train.median())

# Calculate class weight ratio
diverse_weight_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

print(f"\nUsing feature set: {X_train.shape}")
print(f"Class imbalance ratio: {diverse_weight_ratio:.2f}")

# ====================================
# TRAIN THREE DIVERSE MODELS WITH CV
# ====================================

# 1. Neural Network (MLP) with regularization
print("\n1. Training Neural Network (MLP)...")
mlp_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mlp_cv_gini_scores = []

for _fold, (_train_idx, _val_idx) in enumerate(mlp_skf.split(_X_filled, y_train), 1):
    _X_train, _X_val = _X_filled.iloc[_train_idx], _X_filled.iloc[_val_idx]
    _y_train, _y_val = y_train.iloc[_train_idx], y_train.iloc[_val_idx]
    
    _mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', 
                         alpha=0.001, batch_size=256, learning_rate='adaptive', 
                         learning_rate_init=0.001, max_iter=50, random_state=42, verbose=False)
    _mlp.fit(_X_train, _y_train)
    
    _preds = _mlp.predict_proba(_X_val)[:, 1]
    _gini = 2 * roc_auc_score(_y_val, _preds) - 1
    mlp_cv_gini_scores.append(_gini)
    print(f"  Fold {_fold}: Gini = {_gini:.6f}")

mlp_gini_mean = np.mean(mlp_cv_gini_scores)
mlp_gini_std = np.std(mlp_cv_gini_scores)
print(f"✓ MLP Mean Gini: {mlp_gini_mean:.6f} ± {mlp_gini_std:.6f}")

# 2. Ridge Logistic with Polynomial Features (degree=2, top 20 features)
print("\n2. Training Ridge+Polynomial...")
_top_20_features = list(X_train.columns[:20])
_X_top20 = _X_filled[_top_20_features]

ridge_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ridge_cv_gini_scores = []

for _fold, (_train_idx, _val_idx) in enumerate(ridge_skf.split(_X_top20, y_train), 1):
    _X_train, _X_val = _X_top20.iloc[_train_idx], _X_top20.iloc[_val_idx]
    _y_train, _y_val = y_train.iloc[_train_idx], y_train.iloc[_val_idx]
    
    _ridge = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
        ('scaler', StandardScaler()),
        ('ridge', LogisticRegression(penalty='l2', C=0.1, max_iter=500, solver='saga',
                                      random_state=42, class_weight={0: 1, 1: diverse_weight_ratio}))
    ])
    _ridge.fit(_X_train, _y_train)
    
    _preds = _ridge.predict_proba(_X_val)[:, 1]
    _gini = 2 * roc_auc_score(_y_val, _preds) - 1
    ridge_cv_gini_scores.append(_gini)
    print(f"  Fold {_fold}: Gini = {_gini:.6f}")

ridge_gini_mean = np.mean(ridge_cv_gini_scores)
ridge_gini_std = np.std(ridge_cv_gini_scores)
print(f"✓ Ridge+Poly Mean Gini: {ridge_gini_mean:.6f} ± {ridge_gini_std:.6f}")

# 3. Gaussian Naive Bayes
print("\n3. Training Gaussian Naive Bayes...")
nb_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nb_cv_gini_scores = []

for _fold, (_train_idx, _val_idx) in enumerate(nb_skf.split(_X_filled, y_train), 1):
    _X_train, _X_val = _X_filled.iloc[_train_idx], _X_filled.iloc[_val_idx]
    _y_train, _y_val = y_train.iloc[_train_idx], y_train.iloc[_val_idx]
    
    _nb = GaussianNB()
    _nb.fit(_X_train, _y_train)
    
    _preds = _nb.predict_proba(_X_val)[:, 1]
    _gini = 2 * roc_auc_score(_y_val, _preds) - 1
    nb_cv_gini_scores.append(_gini)
    print(f"  Fold {_fold}: Gini = {_gini:.6f}")

nb_gini_mean = np.mean(nb_cv_gini_scores)
nb_gini_std = np.std(nb_cv_gini_scores)
print(f"✓ Naive Bayes Mean Gini: {nb_gini_mean:.6f} ± {nb_gini_std:.6f}")

# ====================================
# DIVERSITY ANALYSIS: OOF PREDICTIONS AND CORRELATION
# ====================================
print("\n" + "=" * 90)
print("DIVERSITY ANALYSIS - CORRELATION WITH TREE MODELS")
print("=" * 90)

print("\nGenerating OOF predictions for correlation analysis...")

# Get tree model OOF predictions from the metadata extraction block
tree_oof_preds = analysis_oof_predictions  # Shape: (476.2k, 3) - LightGBM, XGBoost, CatBoost

# Generate OOF predictions for diverse models using cross_val_predict (faster)
mlp_oof = cross_val_predict(
    MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                  alpha=0.001, batch_size=256, max_iter=50, random_state=42, verbose=False),
    _X_filled, y_train, cv=mlp_skf, method='predict_proba', n_jobs=1
)[:, 1]

ridge_poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
    ('scaler', StandardScaler()),
    ('ridge', LogisticRegression(penalty='l2', C=0.1, max_iter=500, solver='saga',
                                  random_state=42, class_weight={0: 1, 1: diverse_weight_ratio}))
])

ridge_oof = cross_val_predict(
    ridge_poly_model, _X_top20, y_train, cv=ridge_skf, method='predict_proba', n_jobs=1
)[:, 1]

nb_oof = cross_val_predict(
    GaussianNB(), _X_filled, y_train, cv=nb_skf, method='predict_proba', n_jobs=1
)[:, 1]

# Create combined prediction DataFrame
all_predictions = pd.DataFrame({
    'LightGBM': tree_oof_preds[:, 0],
    'XGBoost': tree_oof_preds[:, 1],
    'CatBoost': tree_oof_preds[:, 2],
    'MLP': mlp_oof,
    'Ridge+Poly': ridge_oof,
    'NaiveBayes': nb_oof
})

# Calculate full correlation matrix
full_correlation_matrix = all_predictions.corr()

print("\nFull Prediction Correlation Matrix:")
print(full_correlation_matrix.to_string())

# Calculate correlations between diverse models and tree models
print("\n\nDiversity Check: Correlations with Tree Models")
print("-" * 90)

for diverse_model in ['MLP', 'Ridge+Poly', 'NaiveBayes']:
    tree_corrs = []
    for tree_model in ['LightGBM', 'XGBoost', 'CatBoost']:
        corr = full_correlation_matrix.loc[diverse_model, tree_model]
        tree_corrs.append(corr)
        print(f"  {diverse_model} vs {tree_model}: {corr:.6f}")
    
    avg_corr = np.mean(tree_corrs)
    max_corr = np.max(tree_corrs)
    print(f"  {diverse_model} - Avg correlation with trees: {avg_corr:.6f}, Max: {max_corr:.6f}")
    print(f"  {'✓ SUCCESS: Low correlation (<0.85)' if max_corr < 0.85 else '✗ High correlation (>=0.85)'}")
    print()

# ====================================
# FINAL SUMMARY
# ====================================
print("=" * 90)
print("FINAL SUMMARY")
print("=" * 90)
print(f"\n{'Model':<25} {'CV Gini':<20} {'Diversity Status':<30}")
print("-" * 90)

mlp_max_tree_corr = max([full_correlation_matrix.loc['MLP', t] for t in ['LightGBM', 'XGBoost', 'CatBoost']])
ridge_max_tree_corr = max([full_correlation_matrix.loc['Ridge+Poly', t] for t in ['LightGBM', 'XGBoost', 'CatBoost']])
nb_max_tree_corr = max([full_correlation_matrix.loc['NaiveBayes', t] for t in ['LightGBM', 'XGBoost', 'CatBoost']])

print(f"{'Neural Network (MLP)':<25} {mlp_gini_mean:.6f} ± {mlp_gini_std:.4f}   {'✓ Diverse' if mlp_max_tree_corr < 0.85 else '✗ Not diverse'} (max corr: {mlp_max_tree_corr:.3f})")
print(f"{'Ridge + Polynomial':<25} {ridge_gini_mean:.6f} ± {ridge_gini_std:.4f}   {'✓ Diverse' if ridge_max_tree_corr < 0.85 else '✗ Not diverse'} (max corr: {ridge_max_tree_corr:.3f})")
print(f"{'Gaussian Naive Bayes':<25} {nb_gini_mean:.6f} ± {nb_gini_std:.4f}   {'✓ Diverse' if nb_max_tree_corr < 0.85 else '✗ Not diverse'} (max corr: {nb_max_tree_corr:.3f})")
print("-" * 90)

success_count = sum([mlp_max_tree_corr < 0.85, ridge_max_tree_corr < 0.85, nb_max_tree_corr < 0.85])
print(f"\n{'✓ SUCCESS' if success_count >= 2 else '⚠ PARTIAL SUCCESS'}: {success_count}/3 models show true diversity from tree models (corr < 0.85)")
print("=" * 90)
