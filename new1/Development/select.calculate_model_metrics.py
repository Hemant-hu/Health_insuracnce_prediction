import numpy as np

# Calculate mean and std Gini scores for each model
baseline_lr_mean_gini = np.mean(cv_gini_scores)
baseline_lr_std_gini = np.std(cv_gini_scores, ddof=1)

xgb_mean_gini = np.mean(xgb_cv_gini_scores)
xgb_std_gini = np.std(xgb_cv_gini_scores, ddof=1)

catboost_mean_gini = np.mean(cat_cv_gini_scores)
catboost_std_gini = np.std(cat_cv_gini_scores, ddof=1)

print('Mean Gini Scores (with standard deviation):')
print(f'Baseline Logistic Regression: {baseline_lr_mean_gini:.6f} ± {baseline_lr_std_gini:.6f}')
print(f'XGBoost (HistGradientBoosting): {xgb_mean_gini:.6f} ± {xgb_std_gini:.6f}')
print(f'CatBoost-style: {catboost_mean_gini:.6f} ± {catboost_std_gini:.6f}')