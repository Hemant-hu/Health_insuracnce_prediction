import pandas as pd
import numpy as np

# Create comprehensive comparison table with all models
detailed_comparison = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'XGBoost-style (HistGB)',
        'LightGBM-style (HistGB)',
        'CatBoost-style (HistGB)'
    ],
    'Mean Gini': [
        np.mean(cv_gini_scores),
        np.mean(xgb_cv_gini_scores),
        np.mean(lgb_cv_gini_scores),
        np.mean(cat_cv_gini_scores)
    ],
    'Std Gini': [
        np.std(cv_gini_scores),
        np.std(xgb_cv_gini_scores),
        np.std(lgb_cv_gini_scores),
        np.std(cat_cv_gini_scores)
    ],
    'Mean AUC': [
        np.mean(cv_auc_scores),
        np.mean(xgb_cv_auc_scores),
        np.mean(lgb_cv_auc_scores),
        np.mean(cat_cv_auc_scores)
    ],
    'Std AUC': [
        np.std(cv_auc_scores),
        np.std(xgb_cv_auc_scores),
        np.std(lgb_cv_auc_scores),
        np.std(cat_cv_auc_scores)
    ]
})

# Sort by Mean Gini (descending)
detailed_comparison = detailed_comparison.sort_values('Mean Gini', ascending=False).reset_index(drop=True)

# Add rank column
detailed_comparison.insert(0, 'Rank', range(1, len(detailed_comparison) + 1))

print('=' * 90)
print('COMPREHENSIVE MODEL COMPARISON - 5-FOLD STRATIFIED CROSS-VALIDATION')
print('=' * 90)
print(detailed_comparison.to_string(index=False))
print('=' * 90)