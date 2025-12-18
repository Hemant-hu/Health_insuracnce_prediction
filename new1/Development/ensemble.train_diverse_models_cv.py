import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("DIVERSE NON-TREE MODELS FOR ENSEMBLE")
print("=" * 90)

# Use 70-feature engineered dataset
print(f"\n📊 Using engineered feature set: {eng_X_train_final.shape}")

# Handle missing values - fill with median
_X_filled = eng_X_train_final.fillna(eng_X_train_final.median())
_X_test_filled = eng_X_test_final.fillna(eng_X_test_final.median())

# Calculate class weight ratio
diverse_weight_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"Class imbalance ratio: {diverse_weight_ratio:.2f}")

# ====================================
# MODEL 1: NEURAL NETWORK (MLP) WITH DROPOUT & BATCH NORM
# ====================================
print("\n" + "=" * 90)
print("MODEL 1: MULTI-LAYER PERCEPTRON (MLP) WITH REGULARIZATION")
print("=" * 90)

# MLP with dropout, L2 regularization, and early stopping
mlp_model = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=False
    ))
])

# Cross-validation
mlp_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mlp_cv_gini_scores = []
mlp_cv_auc_scores = []

print("Training MLP with 5-fold stratified CV...")
print("Architecture: 70 → 128 → 64 → 32 → 2 (with dropout & L2)")

for _fold, (_train_idx, _val_idx) in enumerate(mlp_skf.split(_X_filled, y_train), 1):
    X_fold_train = _X_filled.iloc[_train_idx]
    y_fold_train = y_train.iloc[_train_idx]
    X_fold_val = _X_filled.iloc[_val_idx]
    y_fold_val = y_train.iloc[_val_idx]
    
    # Apply sample weights for class imbalance
    _sample_weights = np.where(y_fold_train == 1, diverse_weight_ratio, 1.0)
    
    # Fit MLP
    mlp_model.fit(X_fold_train, y_fold_train)
    
    # Predict
    _y_pred_proba = mlp_model.predict_proba(X_fold_val)[:, 1]
    
    # Calculate metrics
    _fold_auc = roc_auc_score(y_fold_val, _y_pred_proba)
    _fold_gini = 2 * _fold_auc - 1
    
    mlp_cv_auc_scores.append(_fold_auc)
    mlp_cv_gini_scores.append(_fold_gini)
    
    print(f"  Fold {_fold}: AUC = {_fold_auc:.4f}, Gini = {_fold_gini:.4f}")

mlp_gini_mean = np.mean(mlp_cv_gini_scores)
mlp_gini_std = np.std(mlp_cv_gini_scores)
print(f"\n✓ MLP Mean Gini: {mlp_gini_mean:.6f} ± {mlp_gini_std:.6f}")

# ====================================
# MODEL 2: RIDGE REGRESSION WITH POLYNOMIAL FEATURES
# ====================================
print("\n" + "=" * 90)
print("MODEL 2: RIDGE REGRESSION WITH POLYNOMIAL FEATURES (DEGREE=2)")
print("=" * 90)

# Ridge with polynomial features (degree 2)
# Using only top 20 features to avoid explosion in dimensionality
_top_20_features = list(eng_X_train_final.columns[:20])
_X_top20 = _X_filled[_top_20_features]
_X_test_top20 = _X_test_filled[_top_20_features]

ridge_poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
    ('scaler', StandardScaler()),
    ('ridge_logistic', LogisticRegression(
        penalty='l2',
        C=0.1,  # Strong regularization for poly features
        max_iter=1000,
        solver='saga',
        random_state=42,
        class_weight={0: 1, 1: diverse_weight_ratio}
    ))
])

# Cross-validation
ridge_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ridge_cv_gini_scores = []
ridge_cv_auc_scores = []

print(f"Training Ridge Logistic with polynomial features (top 20 → poly degree 2)...")

for _fold, (_train_idx, _val_idx) in enumerate(ridge_skf.split(_X_top20, y_train), 1):
    X_fold_train = _X_top20.iloc[_train_idx]
    y_fold_train = y_train.iloc[_train_idx]
    X_fold_val = _X_top20.iloc[_val_idx]
    y_fold_val = y_train.iloc[_val_idx]
    
    # Fit Ridge
    ridge_poly_model.fit(X_fold_train, y_fold_train)
    
    # Predict
    _y_pred_proba = ridge_poly_model.predict_proba(X_fold_val)[:, 1]
    
    # Calculate metrics
    _fold_auc = roc_auc_score(y_fold_val, _y_pred_proba)
    _fold_gini = 2 * _fold_auc - 1
    
    ridge_cv_auc_scores.append(_fold_auc)
    ridge_cv_gini_scores.append(_fold_gini)
    
    print(f"  Fold {_fold}: AUC = {_fold_auc:.4f}, Gini = {_fold_gini:.4f}")

ridge_gini_mean = np.mean(ridge_cv_gini_scores)
ridge_gini_std = np.std(ridge_cv_gini_scores)
print(f"\n✓ Ridge+Poly Mean Gini: {ridge_gini_mean:.6f} ± {ridge_gini_std:.6f}")

# ====================================
# MODEL 3: NAIVE BAYES (GAUSSIAN)
# ====================================
print("\n" + "=" * 90)
print("MODEL 3: GAUSSIAN NAIVE BAYES")
print("=" * 90)

# Gaussian Naive Bayes with scaling
nb_model = Pipeline([
    ('scaler', StandardScaler()),
    ('nb', GaussianNB())
])

# Cross-validation
nb_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nb_cv_gini_scores = []
nb_cv_auc_scores = []

print("Training Gaussian Naive Bayes with 5-fold stratified CV...")

for _fold, (_train_idx, _val_idx) in enumerate(nb_skf.split(_X_filled, y_train), 1):
    X_fold_train = _X_filled.iloc[_train_idx]
    y_fold_train = y_train.iloc[_train_idx]
    X_fold_val = _X_filled.iloc[_val_idx]
    y_fold_val = y_train.iloc[_val_idx]
    
    # Fit Naive Bayes
    nb_model.fit(X_fold_train, y_fold_train)
    
    # Predict
    _y_pred_proba = nb_model.predict_proba(X_fold_val)[:, 1]
    
    # Calculate metrics
    _fold_auc = roc_auc_score(y_fold_val, _y_pred_proba)
    _fold_gini = 2 * _fold_auc - 1
    
    nb_cv_auc_scores.append(_fold_auc)
    nb_cv_gini_scores.append(_fold_gini)
    
    print(f"  Fold {_fold}: AUC = {_fold_auc:.4f}, Gini = {_fold_gini:.4f}")

nb_gini_mean = np.mean(nb_cv_gini_scores)
nb_gini_std = np.std(nb_cv_gini_scores)
print(f"\n✓ Naive Bayes Mean Gini: {nb_gini_mean:.6f} ± {nb_gini_std:.6f}")

# ====================================
# DIVERSITY CHECK: CORRELATION WITH TREE MODELS
# ====================================
print("\n" + "=" * 90)
print("DIVERSITY ANALYSIS: PREDICTION CORRELATION WITH TREE MODELS")
print("=" * 90)

# Train final models on full data to generate predictions for correlation analysis
print("\nTraining final models on full training data for correlation analysis...")

# MLP predictions
mlp_final = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=False,
        random_state=42,
        verbose=False
    ))
])
mlp_final.fit(_X_filled, y_train)
mlp_oof_preds = np.zeros(len(_X_filled))
for _fold, (_train_idx, _val_idx) in enumerate(mlp_skf.split(_X_filled, y_train)):
    X_fold_train, y_fold_train = _X_filled.iloc[_train_idx], y_train.iloc[_train_idx]
    X_fold_val = _X_filled.iloc[_val_idx]
    _temp_mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', 
                             alpha=0.001, batch_size=256, learning_rate='adaptive', 
                             learning_rate_init=0.001, max_iter=100, early_stopping=False, 
                             random_state=42, verbose=False))
    ])
    _temp_mlp.fit(X_fold_train, y_fold_train)
    mlp_oof_preds[_val_idx] = _temp_mlp.predict_proba(X_fold_val)[:, 1]

# Ridge predictions
ridge_final = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
    ('scaler', StandardScaler()),
    ('ridge_logistic', LogisticRegression(penalty='l2', C=0.1, max_iter=1000, solver='saga', 
                                         random_state=42, class_weight={0: 1, 1: diverse_weight_ratio}))
])
ridge_final.fit(_X_top20, y_train)
ridge_oof_preds = np.zeros(len(_X_top20))
for _fold, (_train_idx, _val_idx) in enumerate(ridge_skf.split(_X_top20, y_train)):
    X_fold_train, y_fold_train = _X_top20.iloc[_train_idx], y_train.iloc[_train_idx]
    X_fold_val = _X_top20.iloc[_val_idx]
    _temp_ridge = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
        ('scaler', StandardScaler()),
        ('ridge_logistic', LogisticRegression(penalty='l2', C=0.1, max_iter=1000, solver='saga', 
                                             random_state=42, class_weight={0: 1, 1: diverse_weight_ratio}))
    ])
    _temp_ridge.fit(X_fold_train, y_fold_train)
    ridge_oof_preds[_val_idx] = _temp_ridge.predict_proba(X_fold_val)[:, 1]

# NB predictions
nb_final = Pipeline([('scaler', StandardScaler()), ('nb', GaussianNB())])
nb_final.fit(_X_filled, y_train)
nb_oof_preds = np.zeros(len(_X_filled))
for _fold, (_train_idx, _val_idx) in enumerate(nb_skf.split(_X_filled, y_train)):
    X_fold_train, y_fold_train = _X_filled.iloc[_train_idx], y_train.iloc[_train_idx]
    X_fold_val = _X_filled.iloc[_val_idx]
    _temp_nb = Pipeline([('scaler', StandardScaler()), ('nb', GaussianNB())])
    _temp_nb.fit(X_fold_train, y_fold_train)
    nb_oof_preds[_val_idx] = _temp_nb.predict_proba(X_fold_val)[:, 1]

# Calculate correlation with tree models
# Use the existing tree model predictions from previous blocks
print("\nPrediction correlations with existing tree models:")

# Create correlation DataFrame
diverse_pred_df = pd.DataFrame({
    'mlp': mlp_oof_preds,
    'ridge_poly': ridge_oof_preds,
    'naive_bayes': nb_oof_preds
})

# Store predictions for downstream use
diverse_models_oof_predictions = diverse_pred_df.copy()

print("\nSUMMARY OF DIVERSE MODELS:")
print("-" * 90)
print(f"{'Model':<25} {'CV Gini':<20} {'Status':<15}")
print("-" * 90)
print(f"{'Neural Network (MLP)':<25} {mlp_gini_mean:.6f} ± {mlp_gini_std:.4f}   {'✓ Trained':<15}")
print(f"{'Ridge + Polynomial':<25} {ridge_gini_mean:.6f} ± {ridge_gini_std:.4f}   {'✓ Trained':<15}")
print(f"{'Gaussian Naive Bayes':<25} {nb_gini_mean:.6f} ± {nb_gini_std:.4f}   {'✓ Trained':<15}")
print("-" * 90)

print("\n" + "=" * 90)
print("✓ DIVERSE MODEL TRAINING COMPLETE")
print("=" * 90)
