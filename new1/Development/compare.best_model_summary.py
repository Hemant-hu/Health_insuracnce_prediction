import pandas as pd
import numpy as np

# Identify best performing model
best_model_row = detailed_comparison.iloc[0]
best_model_name = best_model_row['Model']
best_mean_gini = best_model_row['Mean Gini']
best_std_gini = best_model_row['Std Gini']
best_mean_auc = best_model_row['Mean AUC']
best_std_auc = best_model_row['Std AUC']

# Calculate improvement over baseline
baseline_row = detailed_comparison[detailed_comparison['Model'] == 'Logistic Regression'].iloc[0]
baseline_mean_gini = baseline_row['Mean Gini']

improvement_pct = ((best_mean_gini - baseline_mean_gini) / baseline_mean_gini) * 100

# Display summary
print('=' * 80)
print('BEST MODEL IDENTIFICATION')
print('=' * 80)
print(f'\n🏆 WINNER: {best_model_name}')
print(f'\n   Performance Metrics (5-Fold Cross-Validation):')
print(f'   • Mean Gini Coefficient: {best_mean_gini:.6f} ± {best_std_gini:.6f}')
print(f'   • Mean AUC Score:        {best_mean_auc:.6f} ± {best_std_auc:.6f}')
print(f'\n   Improvement vs. Baseline:')
print(f'   • Baseline (Logistic):   {baseline_mean_gini:.6f}')
print(f'   • Improvement:           +{improvement_pct:.2f}%')
print(f'   • Absolute Gain:         +{(best_mean_gini - baseline_mean_gini):.6f}')
print('\n' + '=' * 80)
print('CONCLUSION: LightGBM-style HistGradientBoosting performs best')
print('=' * 80)