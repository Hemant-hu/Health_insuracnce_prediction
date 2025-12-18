import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Zerve design system colors
DARK_BG = '#1D1D20'
PRIMARY_TEXT = '#fbfbff'
SECONDARY_TEXT = '#909094'
ZERVE_COLORS = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']

# Set up matplotlib style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = DARK_BG
plt.rcParams['axes.facecolor'] = DARK_BG
plt.rcParams['text.color'] = PRIMARY_TEXT
plt.rcParams['axes.labelcolor'] = PRIMARY_TEXT
plt.rcParams['xtick.color'] = PRIMARY_TEXT
plt.rcParams['ytick.color'] = PRIMARY_TEXT
plt.rcParams['grid.color'] = SECONDARY_TEXT
plt.rcParams['grid.alpha'] = 0.2

# ====================================
# FIGURE 1: META-LEARNER COEFFICIENTS
# ====================================
meta_coef_chart = plt.figure(figsize=(12, 6))
ax1 = meta_coef_chart.add_subplot(111)

models = analysis_meta_coefficients['Base Model'].values
weights = analysis_meta_coefficients['Relative Weight (%)'].values

bars = ax1.barh(models, weights, color=ZERVE_COLORS[:3], alpha=0.9, edgecolor=PRIMARY_TEXT, linewidth=1.5)

ax1.set_xlabel('Relative Weight (%)', fontsize=13, fontweight='bold', color=PRIMARY_TEXT)
ax1.set_title('Meta-Learner Base Model Weights', fontsize=16, fontweight='bold', color=PRIMARY_TEXT, pad=20)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, weights)):
    ax1.text(val + 1, bar.get_y() + bar.get_height()/2, 
             f'{val:.1f}%', 
             va='center', fontsize=11, fontweight='bold', color=PRIMARY_TEXT)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color(SECONDARY_TEXT)
ax1.spines['bottom'].set_color(SECONDARY_TEXT)

plt.tight_layout()
print("✓ Meta-learner coefficient chart created")

# ====================================
# FIGURE 2: INDIVIDUAL MODEL CV GINI SCORES
# ====================================
gini_scores_chart = plt.figure(figsize=(12, 6))
ax2 = gini_scores_chart.add_subplot(111)

x_pos = np.arange(len(analysis_individual_scores))
mean_ginis = analysis_individual_scores['Mean Gini'].values
std_ginis = analysis_individual_scores['Std Gini'].values
model_names = analysis_individual_scores['Base Model'].values

bars = ax2.bar(x_pos, mean_ginis, yerr=std_ginis, 
               color=ZERVE_COLORS[:3], alpha=0.9, 
               edgecolor=PRIMARY_TEXT, linewidth=1.5,
               capsize=5, error_kw={'linewidth': 2, 'ecolor': PRIMARY_TEXT})

ax2.set_ylabel('Mean Gini Coefficient', fontsize=13, fontweight='bold', color=PRIMARY_TEXT)
ax2.set_title('Individual Base Model Performance (5-Fold CV)', fontsize=16, fontweight='bold', color=PRIMARY_TEXT, pad=20)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(model_names, fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, mean_ginis, std_ginis)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.002,
             f'{mean:.6f}', ha='center', va='bottom', 
             fontsize=10, fontweight='bold', color=PRIMARY_TEXT)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color(SECONDARY_TEXT)
ax2.spines['bottom'].set_color(SECONDARY_TEXT)

plt.tight_layout()
print("✓ Individual model Gini chart created")

# ====================================
# FIGURE 3: CORRELATION HEATMAP
# ====================================
correlation_heatmap = plt.figure(figsize=(10, 8))
ax3 = correlation_heatmap.add_subplot(111)

# Create heatmap with custom colormap
sns.heatmap(analysis_correlation_matrix, 
            annot=True, fmt='.4f', 
            cmap='RdYlGn_r', center=0.97,
            square=True, linewidths=2,
            cbar_kws={'label': 'Correlation'},
            ax=ax3,
            vmin=0.95, vmax=1.0,
            annot_kws={'fontsize': 12, 'fontweight': 'bold'})

ax3.set_title('Base Model Prediction Correlation (Diversity)', 
              fontsize=16, fontweight='bold', color=PRIMARY_TEXT, pad=20)
ax3.set_xlabel('')
ax3.set_ylabel('')

# Style the colorbar
cbar = ax3.collections[0].colorbar
cbar.ax.yaxis.label.set_color(PRIMARY_TEXT)
cbar.ax.tick_params(colors=PRIMARY_TEXT)

plt.tight_layout()
print("✓ Correlation heatmap created")

# ====================================
# FIGURE 4: FOLD-BY-FOLD COMPARISON
# ====================================
fold_comparison_chart = plt.figure(figsize=(14, 7))
ax4 = fold_comparison_chart.add_subplot(111)

fold_cols = [f'Fold {i+1}' for i in range(5)]
fold_data = analysis_individual_scores[fold_cols].values

x_positions = np.arange(5)
width = 0.25

for i, (model_name, scores) in enumerate(zip(model_names, fold_data)):
    offset = (i - 1) * width
    bars = ax4.bar(x_positions + offset, scores, width, 
                   label=model_name, color=ZERVE_COLORS[i], 
                   alpha=0.9, edgecolor=PRIMARY_TEXT, linewidth=1.5)

ax4.set_xlabel('Fold', fontsize=13, fontweight='bold', color=PRIMARY_TEXT)
ax4.set_ylabel('Gini Coefficient', fontsize=13, fontweight='bold', color=PRIMARY_TEXT)
ax4.set_title('Base Model Performance Across CV Folds', fontsize=16, fontweight='bold', color=PRIMARY_TEXT, pad=20)
ax4.set_xticks(x_positions)
ax4.set_xticklabels([f'Fold {i+1}' for i in range(5)], fontsize=11)
ax4.legend(loc='upper left', framealpha=0.9, edgecolor=SECONDARY_TEXT, 
           facecolor=DARK_BG, labelcolor=PRIMARY_TEXT, fontsize=11)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_color(SECONDARY_TEXT)
ax4.spines['bottom'].set_color(SECONDARY_TEXT)

plt.tight_layout()
print("✓ Fold-by-fold comparison chart created")

print("\n" + "=" * 90)
print("VISUALIZATION SUMMARY")
print("=" * 90)
print("✓ Meta-learner coefficient chart: Shows relative weights in ensemble")
print("✓ Individual model Gini chart: Compares base model CV performance")
print("✓ Correlation heatmap: Reveals model prediction diversity")
print("✓ Fold-by-fold chart: Displays consistency across CV folds")
print("=" * 90)