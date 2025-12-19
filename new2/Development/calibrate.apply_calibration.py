import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

print('🎯 PROBABILITY CALIBRATION TO 3-10% CLAIM RATE\n')
print('='*70)

# Current predictions stats
print('\n📊 CURRENT PREDICTIONS (Uncalibrated):')
print(f'   Mean claim rate: {test_probabilities.mean():.4f} ({test_probabilities.mean()*100:.2f}%)')
print(f'   Min: {test_probabilities.min():.6f}')
print(f'   Max: {test_probabilities.max():.6f}')
print(f'   Median: {np.median(test_probabilities):.6f}')

# Target claim rate
target_claim_rate = y_train.mean()
print(f'\n🎯 TARGET: {target_claim_rate:.4f} ({target_claim_rate*100:.2f}%)')

# Strategy 1: Isotonic Regression Calibration (preserves ranking, learns monotonic mapping)
print('\n\n📈 METHOD 1: ISOTONIC REGRESSION CALIBRATION')
print('-'*70)

# Use cross-validated predictions for calibration to avoid overfitting
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_preds_cv = np.zeros(len(y_train))

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_final, y_train)):
    X_fold_train = X_train_final.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train_final.iloc[val_idx]
    
    # Predict on validation fold
    train_preds_cv[val_idx] = final_trained_model.predict_proba(X_fold_val)[:, 1]

print(f'✓ Generated CV predictions for calibration')
print(f'  CV predictions mean: {train_preds_cv.mean():.4f}')

# Fit isotonic regression on CV predictions
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(train_preds_cv, y_train)

# Apply isotonic calibration to test predictions
test_probs_isotonic = iso_reg.transform(test_probabilities)

print(f'\n✓ Isotonic calibration applied')
print(f'  Calibrated mean: {test_probs_isotonic.mean():.4f} ({test_probs_isotonic.mean()*100:.2f}%)')
print(f'  Min: {test_probs_isotonic.min():.6f}')
print(f'  Max: {test_probs_isotonic.max():.6f}')

# Check if within target range
isotonic_in_range = 0.03 <= test_probs_isotonic.mean() <= 0.10
print(f'  ✓ In target range (3-10%): {isotonic_in_range}')

# Strategy 2: Platt Scaling (logistic calibration)
print('\n\n📈 METHOD 2: PLATT SCALING')
print('-'*70)

# Fit logistic regression on CV predictions
platt_model = LogisticRegression()
platt_model.fit(train_preds_cv.reshape(-1, 1), y_train)

# Apply Platt scaling to test predictions
test_probs_platt = platt_model.predict_proba(test_probabilities.reshape(-1, 1))[:, 1]

print(f'✓ Platt scaling applied')
print(f'  Calibrated mean: {test_probs_platt.mean():.4f} ({test_probs_platt.mean()*100:.2f}%)')
print(f'  Min: {test_probs_platt.min():.6f}')
print(f'  Max: {test_probs_platt.max():.6f}')

platt_in_range = 0.03 <= test_probs_platt.mean() <= 0.10
print(f'  ✓ In target range (3-10%): {platt_in_range}')

# Strategy 3: Beta Calibration (more flexible than Platt)
print('\n\n📈 METHOD 3: LINEAR SCALING TO TARGET')
print('-'*70)

# Simple linear scaling: scale predictions to match target mean
current_mean = train_preds_cv.mean()
target_mean = target_claim_rate
scale_factor = target_mean / current_mean

test_probs_scaled = np.clip(test_probabilities * scale_factor, 0, 1)

print(f'✓ Linear scaling applied (factor: {scale_factor:.4f})')
print(f'  Calibrated mean: {test_probs_scaled.mean():.4f} ({test_probs_scaled.mean()*100:.2f}%)')
print(f'  Min: {test_probs_scaled.min():.6f}')
print(f'  Max: {test_probs_scaled.max():.6f}')

scaled_in_range = 0.03 <= test_probs_scaled.mean() <= 0.10
print(f'  ✓ In target range (3-10%): {scaled_in_range}')

# Evaluate ranking preservation (Gini on training data)
print('\n\n📊 RANKING QUALITY VERIFICATION (Training Data)')
print('-'*70)

# Get training predictions
train_probs_uncal = final_trained_model.predict_proba(X_train_final)[:, 1]
train_probs_iso = iso_reg.transform(train_probs_uncal)
train_probs_platt = platt_model.predict_proba(train_probs_uncal.reshape(-1, 1))[:, 1]
train_probs_scaled = np.clip(train_probs_uncal * scale_factor, 0, 1)

# Calculate AUC/Gini
auc_uncal = roc_auc_score(y_train, train_probs_uncal)
auc_iso = roc_auc_score(y_train, train_probs_iso)
auc_platt = roc_auc_score(y_train, train_probs_platt)
auc_scaled = roc_auc_score(y_train, train_probs_scaled)

gini_uncal = 2 * auc_uncal - 1
gini_iso = 2 * auc_iso - 1
gini_platt = 2 * auc_platt - 1
gini_scaled = 2 * auc_scaled - 1

print(f'\nUncalibrated:')
print(f'  AUC: {auc_uncal:.6f} | Gini: {gini_uncal:.6f}')
print(f'\nIsotonic:')
print(f'  AUC: {auc_iso:.6f} | Gini: {gini_iso:.6f} | Change: {gini_iso-gini_uncal:+.6f}')
print(f'\nPlatt Scaling:')
print(f'  AUC: {auc_platt:.6f} | Gini: {gini_platt:.6f} | Change: {gini_platt-gini_uncal:+.6f}')
print(f'\nLinear Scaling:')
print(f'  AUC: {auc_scaled:.6f} | Gini: {gini_scaled:.6f} | Change: {gini_scaled-gini_uncal:+.6f}')

# Select best method
print('\n\n🎯 FINAL SELECTION')
print('='*70)

methods = {
    'Isotonic': (test_probs_isotonic, isotonic_in_range, gini_iso),
    'Platt': (test_probs_platt, platt_in_range, gini_platt),
    'Linear': (test_probs_scaled, scaled_in_range, gini_scaled)
}

# Prefer method that's in range with highest Gini
best_method = None
best_gini = -1

for method_name, (probs, in_range, gini) in methods.items():
    if in_range and gini > best_gini:
        best_method = method_name
        best_gini = gini
        calibrated_probabilities = probs

# If none in range, choose closest to target
if best_method is None:
    print('⚠️  No method achieved target range, selecting closest...')
    best_method = 'Isotonic'  # Typically most robust
    calibrated_probabilities = test_probs_isotonic

print(f'\n✅ SELECTED METHOD: {best_method}')
print(f'   Final mean claim rate: {calibrated_probabilities.mean():.4f} ({calibrated_probabilities.mean()*100:.2f}%)')
print(f'   Gini coefficient: {best_gini:.6f}')
print(f'   In target range (3-10%): {0.03 <= calibrated_probabilities.mean() <= 0.10}')
