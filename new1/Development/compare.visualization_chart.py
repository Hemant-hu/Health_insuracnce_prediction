import matplotlib.pyplot as plt
import numpy as np

# Prepare data for visualization
models_viz = detailed_comparison['Model'].tolist()
mean_gini_viz = detailed_comparison['Mean Gini'].tolist()
std_gini_viz = detailed_comparison['Std Gini'].tolist()

# Create bar chart with error bars
_fig, _ax = plt.subplots(figsize=(12, 7))

# Color best model differently
colors_viz = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(models_viz))]

_bars = _ax.bar(models_viz, mean_gini_viz, yerr=std_gini_viz, capsize=8, 
               color=colors_viz, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for _bar, mean_val, std_val in zip(_bars, mean_gini_viz, std_gini_viz):
    height = _bar.get_height()
    _ax.text(_bar.get_x() + _bar.get_width()/2., height + std_val + 0.002,
            f'{mean_val:.4f}\n±{std_val:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

_ax.set_ylabel('Mean Gini Coefficient', fontsize=13, fontweight='bold')
_ax.set_xlabel('Model Type', fontsize=13, fontweight='bold')
_ax.set_title('Model Performance Comparison\n5-Fold Cross-Validation Gini Scores', 
             fontsize=15, fontweight='bold', pad=20)
_ax.grid(axis='y', alpha=0.3, linestyle='--')
_ax.set_ylim(0, max(mean_gini_viz) + max(std_gini_viz) + 0.05)

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()

print(f'Best model: {models_viz[0]} with Gini = {mean_gini_viz[0]:.4f}')