#!/usr/bin/env python3
"""
Generar todas las figuras para el paper en formato adecuado
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
from pathlib import Path

# Configuración para publicación
matplotlib.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 11,
})

# Crear directorio para figuras
output_dir = Path('paper_figures')
output_dir.mkdir(exist_ok=True)

print("Generando figuras para el paper...")

# ============================================================================
# FIGURA 1: Comparación completa de modelos (3 barras agrupadas)
# ============================================================================

with open('results/evaluation_metrics.json', 'r') as f:
    vae_v1 = json.load(f)

with open('results_v2/evaluation_metrics.json', 'r') as f:
    vae_v2 = json.load(f)

with open('results/classifier_metrics.json', 'r') as f:
    classifier = json.load(f)

models = ['VAE\n(40 muestras)', 'VAE\n(328 muestras)', 'VAE+Clasificador\n(328 muestras)']
recall = [vae_v1['recall']*100, vae_v2['recall']*100, classifier['recall']*100]
precision = [vae_v1['precision']*100, vae_v2['precision']*100, classifier['precision']*100]
f1 = [vae_v1['f1_score']*100, vae_v2['f1_score']*100, classifier['f1_score']*100]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, recall, width, label='Recall', color='#2E7D32', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x, precision, width, label='Precisión', color='#1976D2', edgecolor='black', linewidth=0.8)
bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#F57C00', edgecolor='black', linewidth=0.8)

ax.set_ylabel('Porcentaje (%)', fontweight='bold')
ax.set_ylim([0, 100])
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'fig1_model_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig1_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Figura 1: Comparación de modelos")
plt.close()

# ============================================================================
# FIGURA 2: Evolución del F1-Score
# ============================================================================

fig, ax = plt.subplots(figsize=(6, 3.5))

stages = ['Inicial\n(40)', 'Expandido\n(328)', 'Supervisado\n(328)']
f1_evolution = [52.17, 87.40, 93.65]

ax.plot(range(3), f1_evolution, marker='o', linewidth=2.5, markersize=10,
        color='#F57C00', markerfacecolor='white', markeredgewidth=2.5, 
        markeredgecolor='#F57C00')

# Áreas de mejora
ax.fill_between([0, 1], [52.17, 87.40], alpha=0.2, color='#4CAF50')
ax.fill_between([1, 2], [87.40, 93.65], alpha=0.2, color='#2196F3')

for i, (x, y) in enumerate(zip(range(3), f1_evolution)):
    ax.text(x, y + 3, f'{y:.1f}%', ha='center', fontsize=10, fontweight='bold')

# Anotar mejoras
ax.annotate('', xy=(1, 70), xytext=(0, 70),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(0.5, 73, '+35.2%', ha='center', fontsize=9, color='green', fontweight='bold')

ax.annotate('', xy=(2, 91), xytext=(1, 91),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.text(1.5, 93.5, '+6.3%', ha='center', fontsize=9, color='blue', fontweight='bold')

ax.set_ylabel('F1-Score (%)', fontweight='bold')
ax.set_ylim([0, 100])
ax.set_xticks(range(3))
ax.set_xticklabels(stages)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'fig2_f1_evolution.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig2_f1_evolution.png', dpi=300, bbox_inches='tight')
print("✓ Figura 2: Evolución del F1-Score")
plt.close()

# ============================================================================
# FIGURA 3: Comparación con estado del arte
# ============================================================================

fig, ax = plt.subplots(figsize=(7, 4))

studies = ['Xie et al.\n(2017)', 'Stowell &\nPlumbley\n(2014)', 
           'Bellinger\net al.\n(2019)', 'Este trabajo']
f1_scores = [87, 82, 85, 93.65]
datasets = [1200, 500, 800, 328]

colors = ['#9E9E9E', '#9E9E9E', '#9E9E9E', '#F57C00']
bars = ax.bar(studies, f1_scores, color=colors, edgecolor='black', linewidth=0.8)

for i, (bar, f1, n) in enumerate(zip(bars, f1_scores, datasets)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{f1:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2., 5,
            f'n={n}', ha='center', va='bottom', fontsize=8, style='italic')

ax.set_ylabel('F1-Score (%)', fontweight='bold')
ax.set_ylim([0, 100])
ax.axhline(y=90, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Umbral 90%')
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'fig3_state_of_art.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig3_state_of_art.png', dpi=300, bbox_inches='tight')
print("✓ Figura 3: Comparación con estado del arte")
plt.close()

# ============================================================================
# TABLA 1: Resultados detallados (guardar como CSV para LaTeX)
# ============================================================================

import csv

table_data = [
    ['Modelo', 'Tamaño Dataset', 'Recall (%)', 'Precisión (%)', 'F1-Score (%)'],
    ['VAE (baseline)', '40', '90.00', '36.73', '52.17'],
    ['VAE (expandido)', '328', '90.43', '84.57', '87.40'],
    ['VAE + Clasificador', '328', '92.41', '94.92', '93.65']
]

with open(output_dir / 'table1_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(table_data)

print("✓ Tabla 1: Resultados detallados")

print(f"\n{'='*60}")
print("FIGURAS GENERADAS EN: paper_figures/")
print(f"{'='*60}")
print("Archivos generados:")
print("  - fig1_model_comparison.pdf/png")
print("  - fig2_f1_evolution.pdf/png")
print("  - fig3_state_of_art.pdf/png")
print("  - table1_results.csv")
print(f"{'='*60}\n")
