from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import itertools

print("🔬 Hyperparameter Tuning - LightGBM-style Model (Manual Grid Search)")
print("=" * 75)

# Class weights
lgb_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()

# Define comprehensive hyperparameter grid (50+ combinations)
param_grid = {
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.02, 0.05, 0.1, 0.15],
    'max_iter': [100, 200, 300],
    'min_samples_leaf': [10, 30, 50, 100],
    'l2_regularization': [0.5, 1.0, 3.0]
}

# Generate parameter combinations
param_names = list(param_grid.keys())
param_values = list(param_grid.values())
all_combinations = list(itertools.product(*param_values))

# Limit to 50 trials for efficiency
np.random.seed(42)
selected_indices = np.random.choice(len(all_combinations), min(50, len(all_combinations)), replace=False)
selected_combinations = [all_combinations[i] for i in selected_indices]

print(f"Testing {len(selected_combinations)} hyperparameter combinations")
print(f"5-fold stratified CV per combination\n")

# Track best configuration
best_gini = -1
best_params_lgb = None
lgb_results = []

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Test each parameter combination
for trial_num, params_tuple in enumerate(selected_combinations, 1):
    params_dict = dict(zip(param_names, params_tuple))
    params_dict['random_state'] = 42
    
    # 5-fold CV for this configuration
    fold_ginis = []
    for _train_idx, _val_idx in skf.split(X_train, y_train):
        _X_train_fold = X_train.iloc[_train_idx]
        _y_train_fold = y_train.iloc[_train_idx]
        _X_val_fold = X_train.iloc[_val_idx]
        _y_val_fold = y_train.iloc[_val_idx]
        
        # Sample weights
        _weights = np.where(_y_train_fold == 1, lgb_weight_ratio, 1.0).astype(float)
        
        # Train model
        model = HistGradientBoostingClassifier(**params_dict)
        model.fit(_X_train_fold, _y_train_fold, sample_weight=_weights)
        
        # Evaluate
        _y_pred = model.predict_proba(_X_val_fold)[:, 1]
        _auc = roc_auc_score(_y_val_fold, _y_pred)
        _gini = 2 * _auc - 1
        fold_ginis.append(_gini)
    
    mean_gini = np.mean(fold_ginis)
    lgb_results.append((params_dict, mean_gini))
    
    if mean_gini > best_gini:
        best_gini = mean_gini
        best_params_lgb = params_dict.copy()
    
    if trial_num % 10 == 0:
        print(f"Trial {trial_num}/{len(selected_combinations)}: Best Gini so far = {best_gini:.6f}")

print(f"\n{'='*75}")
print(f"✓ Tuning Complete!")
print(f"\nBest CV Gini Score: {best_gini:.6f}")
print(f"\nBest Hyperparameters:")
for param, value in best_params_lgb.items():
    if param != 'random_state':
        print(f"  {param}: {value}")

# Sort and show top 5 configurations
lgb_results_sorted = sorted(lgb_results, key=lambda x: x[1], reverse=True)
print(f"\nTop 5 Configurations:")
for rank, (params, gini) in enumerate(lgb_results_sorted[:5], 1):
    print(f"  {rank}. Gini={gini:.6f} | depth={params['max_depth']}, lr={params['learning_rate']}, iter={params['max_iter']}")