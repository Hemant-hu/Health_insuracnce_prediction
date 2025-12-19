import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====================================
# 1. COLLECT MODEL RESULTS
# ====================================
model_results = []

# Baseline Logistic Regression
baseline_gini_mean = np.mean(cv_gini_scores)
baseline_gini_std = np.std(cv_gini_scores)
model_results.append({
    "Model": "Baseline (Logistic Regression)",
    "Mean Gini": baseline_gini_mean,
    "Std Gini": baseline_gini_std,
    "Improvement (%)": 0.0
})

# XGBoost-style HistGradientBoosting
xgb_gini_mean = np.mean(xgb_cv_gini_scores)
xgb_gini_std = np.std(xgb_cv_gini_scores)
xgb_improvement = ((xgb_gini_mean - baseline_gini_mean) / baseline_gini_mean) * 100
model_results.append({
    "Model": "XGBoost-style",
    "Mean Gini": xgb_gini_mean,
    "Std Gini": xgb_gini_std,
    "Improvement (%)": xgb_improvement
})

# LightGBM-style HistGradientBoosting
lgb_gini_mean = np.mean(lgb_cv_gini_scores)
lgb_gini_std = np.std(lgb_cv_gini_scores)
lgb_improvement = ((lgb_gini_mean - baseline_gini_mean) / baseline_gini_mean) * 100
model_results.append({
    "Model": "LightGBM-style",
    "Mean Gini": lgb_gini_mean,
    "Std Gini": lgb_gini_std,
    "Improvement (%)": lgb_improvement
})

# CatBoost-style HistGradientBoosting
cat_gini_mean = np.mean(cat_cv_gini_scores)
cat_gini_std = np.std(cat_cv_gini_scores)
cat_improvement = ((cat_gini_mean - baseline_gini_mean) / baseline_gini_mean) * 100
model_results.append({
    "Model": "CatBoost-style",
    "Mean Gini": cat_gini_mean,
    "Std Gini": cat_gini_std,
    "Improvement (%)": cat_improvement
})

# ====================================
# 2. CREATE COMPARISON TABLE
# ====================================
model_comparison_table = pd.DataFrame(model_results)
model_comparison_table = model_comparison_table.sort_values(
    by="Mean Gini", ascending=False
).reset_index(drop=True)

# Add rank column
model_comparison_table.insert(0, "Rank", range(1, len(model_comparison_table) + 1))

# ====================================
# 3. DISPLAY RESULTS
# ====================================
print("=" * 90)
print("MODEL COMPARISON – 5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 90)
print(model_comparison_table.to_string(index=False))
print("=" * 90)
print(f"\n🏆 BEST MODEL: {model_comparison_table.iloc[0]['Model']}")
print(f"   Mean Gini: {model_comparison_table.iloc[0]['Mean Gini']:.6f} ± {model_comparison_table.iloc[0]['Std Gini']:.6f}")
print(f"   Improvement: {model_comparison_table.iloc[0]['Improvement (%)']:.2f}% over baseline")
print("=" * 90)

# ====================================
# 4. VISUALIZATION
# ====================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Set modern dark theme
fig.patch.set_facecolor('#1D1D20')
for ax in [ax1, ax2]:
    ax.set_facecolor('#1D1D20')
    ax.tick_params(colors='#fbfbff', which='both')
    ax.spines['bottom'].set_color('#909094')
    ax.spines['top'].set_color('#909094') 
    ax.spines['right'].set_color('#909094')
    ax.spines['left'].set_color('#909094')

# Plot 1: Mean Gini Scores with Error Bars
models = model_comparison_table['Model'].values
mean_ginis = model_comparison_table['Mean Gini'].values
std_ginis = model_comparison_table['Std Gini'].values

colors = ['#3262FF', '#9242DB', '#F79009', '#75E0A7']
bars1 = ax1.barh(models, mean_ginis, xerr=std_ginis, 
                 color=colors, capsize=5, alpha=0.9, edgecolor='#fbfbff', linewidth=1.5)

ax1.set_xlabel('Mean Gini Score', fontsize=13, color='#fbfbff', weight='bold')
ax1.set_title('Model Performance Comparison', fontsize=15, color='#fbfbff', weight='bold', pad=20)
ax1.grid(axis='x', alpha=0.2, color='#909094', linestyle='--')

# Add value labels
for i, (mean, std) in enumerate(zip(mean_ginis, std_ginis)):
    ax1.text(mean + std + 0.003, i, f'{mean:.4f}', 
             va='center', fontsize=11, color='#fbfbff', weight='bold')

# Plot 2: Percentage Improvement over Baseline
improvements = model_comparison_table['Improvement (%)'].values

# Color bars by improvement (green for positive, red for zero/negative)
bar_colors = ['#75E0A7' if imp > 0 else '#F97066' for imp in improvements]
bars2 = ax2.barh(models, improvements, color=bar_colors, 
                 alpha=0.9, edgecolor='#fbfbff', linewidth=1.5)

ax2.set_xlabel('Improvement over Baseline (%)', fontsize=13, color='#fbfbff', weight='bold')
ax2.set_title('Relative Performance Improvement', fontsize=15, color='#fbfbff', weight='bold', pad=20)
ax2.axvline(x=0, color='#909094', linestyle='-', linewidth=2)
ax2.grid(axis='x', alpha=0.2, color='#909094', linestyle='--')

# Add value labels
for i, imp in enumerate(improvements):
    ax2.text(imp + 0.5 if imp > 0 else imp - 0.5, i, f'{imp:.2f}%', 
             va='center', ha='left' if imp > 0 else 'right',
             fontsize=11, color='#fbfbff', weight='bold')

plt.tight_layout()
model_comparison_viz = fig
plt.close()

print("\n✓ Model comparison complete with visualization")