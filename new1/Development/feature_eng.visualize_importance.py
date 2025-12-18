import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 90)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 90)

# ====================================
# 1. TOP ORIGINAL FEATURES
# ====================================
print("\n📊 TOP 10 ORIGINAL FEATURES BY MUTUAL INFORMATION:")
print("-" * 90)
top_original = mi_df.head(10)
for idx, row in top_original.iterrows():
    print(f"   {row['feature']:20s} → MI Score: {row['mi_score']:.6f}")

# ====================================
# 2. TOP ENGINEERED FEATURES
# ====================================
print("\n✨ TOP 10 ENGINEERED FEATURES BY MUTUAL INFORMATION:")
print("-" * 90)
top_engineered = new_mi_df.head(10)
for idx, row in top_engineered.iterrows():
    print(f"   {row['feature']:40s} → MI Score: {row['mi_score']:.6f}")

# ====================================
# 3. COMPARISON: BEST ORIGINAL VS BEST ENGINEERED
# ====================================
best_original_mi = mi_df.iloc[0]['mi_score']
best_engineered_mi = new_mi_df.iloc[0]['mi_score']
improvement_pct = ((best_engineered_mi - best_original_mi) / best_original_mi) * 100

print(f"\n🎯 FEATURE QUALITY COMPARISON:")
print("-" * 90)
print(f"   Best original feature MI:    {best_original_mi:.6f}")
print(f"   Best engineered feature MI:  {best_engineered_mi:.6f}")
print(f"   Improvement:                 {improvement_pct:+.2f}%")

# ====================================
# 4. VISUALIZATION
# ====================================
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#1D1D20')

# Create 2 subplots
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# Style both axes
for ax in [ax1, ax2]:
    ax.set_facecolor('#1D1D20')
    ax.tick_params(colors='#fbfbff', which='both')
    for spine in ax.spines.values():
        spine.set_color('#909094')

# Plot 1: Top Original Features
_top_orig_features = mi_df.head(15)['feature'].values
_top_orig_scores = mi_df.head(15)['mi_score'].values

_bars1 = ax1.barh(_top_orig_features, _top_orig_scores, 
                  color='#A1C9F4', alpha=0.9, edgecolor='#fbfbff', linewidth=1.5)

ax1.set_xlabel('Mutual Information Score', fontsize=12, color='#fbfbff', weight='bold')
ax1.set_title('Top 15 Original Features by Mutual Information', 
              fontsize=14, color='#fbfbff', weight='bold', pad=15)
ax1.grid(axis='x', alpha=0.2, color='#909094', linestyle='--')

# Add value labels
for _i, (_feat, _score) in enumerate(zip(_top_orig_features, _top_orig_scores)):
    ax1.text(_score + 0.001, _i, f'{_score:.4f}', 
             va='center', fontsize=10, color='#fbfbff', weight='bold')

# Plot 2: Top Engineered Features
_top_eng_features = new_mi_df.head(15)['feature'].values
_top_eng_scores = new_mi_df.head(15)['mi_score'].values

_bars2 = ax2.barh(_top_eng_features, _top_eng_scores,
                  color='#8DE5A1', alpha=0.9, edgecolor='#fbfbff', linewidth=1.5)

ax2.set_xlabel('Mutual Information Score', fontsize=12, color='#fbfbff', weight='bold')
ax2.set_title('Top 15 Engineered Features by Mutual Information',
              fontsize=14, color='#fbfbff', weight='bold', pad=15)
ax2.grid(axis='x', alpha=0.2, color='#909094', linestyle='--')

# Add value labels
for _i, (_feat, _score) in enumerate(zip(_top_eng_features, _top_eng_scores)):
    ax2.text(_score + 0.001, _i, f'{_score:.4f}',
             va='center', fontsize=10, color='#fbfbff', weight='bold')

plt.tight_layout()
feature_importance_viz = fig

print("\n✓ Feature importance visualization created")
print("=" * 90)