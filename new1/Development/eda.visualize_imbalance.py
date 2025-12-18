import matplotlib.pyplot as plt

# Create bar chart for target class distribution
target_counts = training_df['target'].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(['Class 0', 'Class 1'], target_counts.values, color=['#3498db', '#e74c3c'])
plt.title('Target Class Distribution (Imbalance)', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Target Class', fontsize=12)

# Add count labels on bars
for i, count in enumerate(target_counts.values):
    plt.text(i, count + 5000, f'{count:,}', ha='center', fontsize=11)

plt.tight_layout()
plt.show()

print(f'Class imbalance clearly visible: {target_counts[0]:,} vs {target_counts[1]:,}')