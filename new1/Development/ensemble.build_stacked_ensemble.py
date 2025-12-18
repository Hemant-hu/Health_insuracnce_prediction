import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("STACKED ENSEMBLE MODELING")
print("=" * 90)

# Use engineered features
print(f"\n📊 Using engineered feature set: {eng_X_train_final.shape}")

# Handle missing values - fill with median
_X_train_filled = eng_X_train_final.fillna(eng_X_train_final.median())
_X_test_filled = eng_X_test_final.fillna(eng_X_test_final.median())

# ====================================
# STEP 1: TRAIN BASE MODELS AND GENERATE OOF PREDICTIONS
# ====================================
print("\n1. Training base models and generating out-of-fold predictions...")

ens_weight_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

# Initialize base models with tuned hyperparameters
base_models = {
    'lightgbm_style': HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=25,
        l2_regularization=0.1,
        max_bins=255,
        random_state=42,
        early_stopping=False,
        class_weight={0: 1, 1: ens_weight_ratio}
    ),
    'xgboost_style': HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=1.0,
        max_bins=255,
        random_state=42,
        early_stopping=False,
        class_weight={0: 1, 1: ens_weight_ratio}
    ),
    'catboost_style': HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.03,
        max_depth=7,
        min_samples_leaf=15,
        l2_regularization=3.0,
        max_bins=64,
        random_state=42,
        early_stopping=False,
        class_weight={0: 1, 1: ens_weight_ratio}
    )
}

# Generate out-of-fold predictions for meta-learner training
ens_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros((len(_X_train_filled), len(base_models)))

print(f"   Generating OOF predictions using {ens_skf.n_splits}-fold CV...")

for _ens_idx, (_ens_model_name, _ens_model) in enumerate(base_models.items()):
    print(f"   • {_ens_model_name}...", end=" ", flush=True)
    
    # Get out-of-fold predictions (probability of positive class)
    oof_preds = cross_val_predict(
        _ens_model, _X_train_filled, y_train,
        cv=ens_skf, method='predict_proba', n_jobs=1
    )[:, 1]
    
    oof_predictions[:, _ens_idx] = oof_preds
    
    # Calculate OOF AUC
    _oof_auc = roc_auc_score(y_train, oof_preds)
    _oof_gini = 2 * _oof_auc - 1
    print(f"OOF Gini: {_oof_gini:.6f}")

# ====================================
# STEP 2: TRAIN META-LEARNER
# ====================================
print("\n2. Training meta-learner (Logistic Regression) on OOF predictions...")

# Scale OOF predictions for logistic regression
ens_scaler = StandardScaler()
oof_predictions_scaled = ens_scaler.fit_transform(oof_predictions)

# Train meta-learner
meta_learner = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    class_weight={0: 1, 1: ens_weight_ratio}
)
meta_learner.fit(oof_predictions_scaled, y_train)

print(f"   Meta-learner weights: {dict(zip(base_models.keys(), meta_learner.coef_[0]))}")

# ====================================
# STEP 3: EVALUATE STACKED ENSEMBLE WITH CV
# ====================================
print("\n3. Evaluating stacked ensemble performance with cross-validation...")

stacked_cv_gini_scores = []

for _fold_idx, (_train_idx, _val_idx) in enumerate(ens_skf.split(_X_train_filled, y_train), 1):
    X_fold_train, X_fold_val = _X_train_filled.iloc[_train_idx], _X_train_filled.iloc[_val_idx]
    y_fold_train, y_fold_val = y_train.iloc[_train_idx], y_train.iloc[_val_idx]
    
    # Train base models on fold training data
    base_train_preds = np.zeros((len(X_fold_val), len(base_models)))
    
    for _ens_idx, (_ens_model_name, _ens_model) in enumerate(base_models.items()):
        _fold_model = _ens_model.__class__(**_ens_model.get_params())
        _fold_model.fit(X_fold_train, y_fold_train)
        base_train_preds[:, _ens_idx] = _fold_model.predict_proba(X_fold_val)[:, 1]
    
    # Scale and predict with meta-learner
    base_train_preds_scaled = ens_scaler.transform(base_train_preds)
    stacked_preds = meta_learner.predict_proba(base_train_preds_scaled)[:, 1]
    
    # Calculate metrics
    _fold_auc = roc_auc_score(y_fold_val, stacked_preds)
    _fold_gini = 2 * _fold_auc - 1
    stacked_cv_gini_scores.append(_fold_gini)
    
    print(f"   Fold {_fold_idx}: Gini = {_fold_gini:.6f}")

stacked_gini_mean = np.mean(stacked_cv_gini_scores)
stacked_gini_std = np.std(stacked_cv_gini_scores)

print(f"\n   Stacked Ensemble Mean Gini: {stacked_gini_mean:.6f} ± {stacked_gini_std:.6f}")

# ====================================
# STEP 4: COMPARE WITH BEST SINGLE MODEL
# ====================================
print("\n4. Comparing stacked ensemble with best single model...")

# Best single model is LightGBM-style with Gini 0.2729
best_single_gini = lgb_gini_mean
ensemble_improvement = ((stacked_gini_mean - best_single_gini) / best_single_gini) * 100

print(f"   Best Single Model (LightGBM): {best_single_gini:.6f}")
print(f"   Stacked Ensemble:             {stacked_gini_mean:.6f}")
print(f"   Improvement:                  {ensemble_improvement:+.2f}%")

target_gini = 0.28
if stacked_gini_mean >= target_gini:
    print(f"   ✓ SUCCESS: Ensemble Gini ({stacked_gini_mean:.6f}) >= target ({target_gini})")
else:
    print(f"   ⚠ Ensemble Gini ({stacked_gini_mean:.6f}) < target ({target_gini})")

# ====================================
# STEP 5: TRAIN FINAL ENSEMBLE ON FULL TRAINING DATA
# ====================================
print("\n5. Training final ensemble on full training data...")

# Train all base models on full data
final_base_models = {}
test_base_predictions = np.zeros((len(_X_test_filled), len(base_models)))

for _ens_idx, (_ens_model_name, _ens_model) in enumerate(base_models.items()):
    print(f"   • Training {_ens_model_name} on full data...")
    _final_model = _ens_model.__class__(**_ens_model.get_params())
    _final_model.fit(_X_train_filled, y_train)
    final_base_models[_ens_model_name] = _final_model
    
    # Generate test predictions
    test_base_predictions[:, _ens_idx] = _final_model.predict_proba(_X_test_filled)[:, 1]

# Generate final stacked predictions
test_base_predictions_scaled = ens_scaler.transform(test_base_predictions)
ensemble_test_predictions = meta_learner.predict_proba(test_base_predictions_scaled)[:, 1]

print("\n" + "=" * 90)
print("STACKED ENSEMBLE SUMMARY")
print("=" * 90)
print(f"✓ Base models: {len(base_models)} (LightGBM, XGBoost, CatBoost styles)")
print(f"✓ Meta-learner: Logistic Regression")
print(f"✓ Feature set: {_X_train_filled.shape[1]} engineered features")
print(f"✓ CV Gini Score: {stacked_gini_mean:.6f} ± {stacked_gini_std:.6f}")
print(f"✓ Improvement over best single: {ensemble_improvement:+.2f}%")
print(f"✓ Test predictions generated: {len(ensemble_test_predictions)} samples")
print("=" * 90)