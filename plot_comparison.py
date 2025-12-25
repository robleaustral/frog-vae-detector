import matplotlib.pyplot as plt
import numpy as np

categories = ['Recall', 'Precision', 'F1-Score']
v1 = [90.00, 36.73, 52.17]
v2 = [90.43, 84.57, 87.40]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, v1, width, label='v1 (40 chunks)', color='lightcoral', edgecolor='black')
bars2 = ax.bar(x + width/2, v2, width, label='v2 (328 chunks)', color='lightgreen', edgecolor='black')

ax.set_ylabel('Porcentaje (%)', fontsize=12, fontweight='bold')
ax.set_title('Comparación de Desempeño: v1 vs v2', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)

# Agregar valores sobre las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results_v2/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: results_v2/model_comparison.png")
plt.close()
