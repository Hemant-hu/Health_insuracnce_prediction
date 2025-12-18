import pandas as pd

# Create comparison DataFrame
model_comparison_results = pd.DataFrame({
    'Model': ['Baseline Logistic Regression', 'XGBoost (HistGradientBoosting)', 'CatBoost-style'],
    'Mean Gini': [baseline_lr_mean_gini, xgb_mean_gini, catboost_mean_gini],
    'Std Gini': [baseline_lr_std_gini, xgb_std_gini, catboost_std_gini]
})

# Sort by Mean Gini (descending)
model_comparison_results = model_comparison_results.sort_values('Mean Gini', ascending=False).reset_index(drop=True)

# Identify best model
best_model_selection = model_comparison_results.iloc[0]['Model']
best_gini_score = model_comparison_results.iloc[0]['Mean Gini']

print('=' * 70)
print('MODEL PERFORMANCE COMPARISON')
print('=' * 70)
print(model_comparison_results.to_string(index=False))
print()
print('=' * 70)
print('BEST MODEL SELECTION')
print('=' * 70)
print(f'Best Model: {best_model_selection}')
print(f'Mean Gini Score: {best_gini_score:.6f}')
print()
print('Justification:')
print(f'The {best_model_selection} achieves the highest mean Gini coefficient')
print(f'of {best_gini_score:.6f}, outperforming the baseline by')
print(f'{(best_gini_score - baseline_lr_mean_gini) / baseline_lr_mean_gini * 100:.2f}%.')
print(f'It also shows good stability with a standard deviation of {model_comparison_results.iloc[0]["Std Gini"]:.6f}.')
print('=' * 70)