import numpy as np

# Calculate mean and std of CV Gini scores
mean_cv_gini = np.mean(cv_gini_scores)
std_cv_gini = np.std(cv_gini_scores)

mean_cv_auc = np.mean(cv_auc_scores)
std_cv_auc = np.std(cv_auc_scores)

print('=' * 60)
print('BASELINE LOGISTIC REGRESSION - CROSS-VALIDATION RESULTS')
print('=' * 60)
print(f'\nModel Configuration:')
print(f'  - Algorithm: Logistic Regression')
print(f'  - Class Weight: balanced')
print(f'  - Cross-Validation: 5-Fold Stratified')
print(f'\n{"-" * 60}')
print(f'ROC-AUC Scores:')
print(f'  Mean ± Std: {mean_cv_auc:.4f} ± {std_cv_auc:.4f}')
print(f'\nGini Coefficient Scores (2*AUC - 1):')
print(f'  Mean ± Std: {mean_cv_gini:.4f} ± {std_cv_gini:.4f}')
print(f'{"-" * 60}')
print(f'\n✓ Baseline model successfully trained and evaluated!')
print('=' * 60)