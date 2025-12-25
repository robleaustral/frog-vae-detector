"""
Generador de Visualizaciones de Ejemplo
========================================
Genera visualizaciones simuladas que muestran cómo se verían los resultados
del sistema de detección de ranas si todo funciona correctamente.

Estas son visualizaciones de EJEMPLO con datos sintéticos para ilustrar
el concepto del espacio latente y la hiperesfera de detección.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import seaborn as sns

# Configurar estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Crear directorio de salida
import os
output_dir = './example_visualizations'
os.makedirs(output_dir, exist_ok=True)


def generate_synthetic_latent_data():
    """
    Genera datos sintéticos que simulan el espacio latente.
    
    Returns:
        frog_latents: Puntos latentes de ranas (agrupados cerca del centroide)
        other_latents: Puntos de otros sonidos (más dispersos)
        centroid: Centroide del espacio latente
        radius: Radio de la hiperesfera
    """
    np.random.seed(42)
    
    # Centroide en el origen
    centroid = np.array([0.0, 0.0, 0.0])
    
    # Ranas: agrupadas cerca del centroide (distribución normal)
    n_frogs = 120
    frog_latents = np.random.normal(loc=centroid, scale=0.3, size=(n_frogs, 3))
    
    # Otros sonidos: más dispersos y alejados
    n_others = 50
    
    # Grupo 1: Pájaros (cluster en otra región)
    birds = np.random.normal(loc=[1.5, 1.0, 0.5], scale=0.4, size=(20, 3))
    
    # Grupo 2: Grillos (otro cluster)
    crickets = np.random.normal(loc=[-1.2, 0.8, -0.3], scale=0.35, size=(15, 3))
    
    # Grupo 3: Ruido ambiental (disperso)
    noise = np.random.normal(loc=[0.2, -1.5, 1.0], scale=0.5, size=(15, 3))
    
    other_latents = np.vstack([birds, crickets, noise])
    
    # Calcular radio: distancia máxima de las ranas al centroide
    distances = np.linalg.norm(frog_latents - centroid, axis=1)
    radius = np.max(distances)
    
    return frog_latents, other_latents, centroid, radius


def plot_2d_latent_space(frog_latents, other_latents, centroid, radius, save_path):
    """
    Visualización 2D del espacio latente con hiperesfera.
    Usa las primeras 2 dimensiones del espacio latente.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Extraer componentes 2D (primeras 2 dimensiones)
    frogs_2d = frog_latents[:, :2]
    others_2d = other_latents[:, :2]
    centroid_2d = centroid[:2]
    
    # Calcular radio en 2D
    distances_2d = np.linalg.norm(frogs_2d - centroid_2d, axis=1)
    radius_2d = np.max(distances_2d)
    
    # Dibujar hiperesfera (círculo en 2D)
    circle = Circle(centroid_2d, radius_2d, color='red', fill=False, 
                   linewidth=3, linestyle='--', label=f'Hiperesfera (radio={radius_2d:.3f})', 
                   alpha=0.8, zorder=1)
    ax.add_patch(circle)
    
    # Región interior (zona de detección)
    circle_fill = Circle(centroid_2d, radius_2d, color='green', fill=True, 
                        alpha=0.1, zorder=0)
    ax.add_patch(circle_fill)
    
    # Puntos de ranas
    ax.scatter(frogs_2d[:, 0], frogs_2d[:, 1], 
              c='#2ecc71', alpha=0.7, s=80, 
              label=f'Ranas (n={len(frogs_2d)})', 
              edgecolors='darkgreen', linewidth=1.5, zorder=3)
    
    # Puntos de otros sonidos con diferentes colores por tipo
    n_birds = 20
    n_crickets = 15
    
    # Pájaros
    ax.scatter(others_2d[:n_birds, 0], others_2d[:n_birds, 1], 
              c='#3498db', alpha=0.7, s=80, 
              marker='s', label='Pájaros (n=20)', 
              edgecolors='darkblue', linewidth=1.5, zorder=3)
    
    # Grillos
    ax.scatter(others_2d[n_birds:n_birds+n_crickets, 0], 
              others_2d[n_birds:n_birds+n_crickets, 1], 
              c='#9b59b6', alpha=0.7, s=80, 
              marker='^', label='Grillos (n=15)', 
              edgecolors='darkviolet', linewidth=1.5, zorder=3)
    
    # Ruido
    ax.scatter(others_2d[n_birds+n_crickets:, 0], 
              others_2d[n_birds+n_crickets:, 1], 
              c='#e74c3c', alpha=0.7, s=80, 
              marker='x', label='Ruido (n=15)', linewidth=2.5, zorder=3)
    
    # Centroide
    ax.scatter(centroid_2d[0], centroid_2d[1], 
              c='red', marker='*', s=800, 
              label='Centroide', edgecolors='darkred', 
              linewidth=2, zorder=5)
    
    # Agregar algunas líneas desde centroide a puntos extremos
    extreme_indices = np.argsort(distances_2d)[-3:]
    for idx in extreme_indices:
        ax.plot([centroid_2d[0], frogs_2d[idx, 0]], 
               [centroid_2d[1], frogs_2d[idx, 1]], 
               'r--', alpha=0.3, linewidth=1, zorder=2)
    
    # Configuración
    ax.set_xlabel('Dimensión Latente 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dimensión Latente 2', fontsize=14, fontweight='bold')
    ax.set_title('Espacio Latente 2D - Sistema de Detección de Ranas\n' + 
                'Visualización de la Hiperesfera de Detección', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95, 
             edgecolor='black', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_aspect('equal')
    
    # Añadir anotaciones
    ax.text(0.02, 0.98, 'Zona de Detección', 
           transform=ax.transAxes, fontsize=11, 
           verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    # Estadísticas en el gráfico
    stats_text = f'Estadísticas:\n'
    stats_text += f'• Ranas dentro del radio: {np.sum(distances_2d <= radius_2d)}/{len(frogs_2d)}\n'
    others_distances = np.linalg.norm(others_2d - centroid_2d, axis=1)
    fp = np.sum(others_distances <= radius_2d)
    stats_text += f'• Falsos positivos: {fp}/{len(others_2d)} ({fp/len(others_2d)*100:.1f}%)'
    
    ax.text(0.02, 0.02, stats_text, 
           transform=ax.transAxes, fontsize=10, 
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualización 2D guardada: {save_path}")
    plt.close()


def plot_3d_latent_space(frog_latents, other_latents, centroid, radius, save_path):
    """
    Visualización 3D del espacio latente con esfera de detección.
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extraer componentes 3D
    frogs_3d = frog_latents
    others_3d = other_latents
    centroid_3d = centroid
    
    # Calcular radio en 3D
    distances_3d = np.linalg.norm(frogs_3d - centroid_3d, axis=1)
    radius_3d = np.max(distances_3d)
    
    # Dibujar esfera
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = radius_3d * np.outer(np.cos(u), np.sin(v)) + centroid_3d[0]
    y_sphere = radius_3d * np.outer(np.sin(u), np.sin(v)) + centroid_3d[1]
    z_sphere = radius_3d * np.outer(np.ones(np.size(u)), np.cos(v)) + centroid_3d[2]
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.15, color='red', 
                   edgecolor='darkred', linewidth=0.2)
    
    # Wireframe de la esfera para mejor visualización
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.2, color='red', 
                     linewidth=0.5, rstride=5, cstride=5)
    
    # Puntos de ranas
    ax.scatter(frogs_3d[:, 0], frogs_3d[:, 1], frogs_3d[:, 2], 
              c='#2ecc71', alpha=0.8, s=60, 
              label=f'Ranas (n={len(frogs_3d)})',
              edgecolors='darkgreen', linewidth=1, depthshade=True)
    
    # Otros sonidos
    n_birds = 20
    n_crickets = 15
    
    ax.scatter(others_3d[:n_birds, 0], others_3d[:n_birds, 1], others_3d[:n_birds, 2], 
              c='#3498db', alpha=0.8, s=60, 
              marker='s', label='Pájaros (n=20)',
              edgecolors='darkblue', linewidth=1, depthshade=True)
    
    ax.scatter(others_3d[n_birds:n_birds+n_crickets, 0], 
              others_3d[n_birds:n_birds+n_crickets, 1],
              others_3d[n_birds:n_birds+n_crickets, 2], 
              c='#9b59b6', alpha=0.8, s=60, 
              marker='^', label='Grillos (n=15)',
              edgecolors='darkviolet', linewidth=1, depthshade=True)
    
    ax.scatter(others_3d[n_birds+n_crickets:, 0], 
              others_3d[n_birds+n_crickets:, 1],
              others_3d[n_birds+n_crickets:, 2], 
              c='#e74c3c', alpha=0.8, s=60, 
              marker='x', label='Ruido (n=15)', linewidth=2, depthshade=True)
    
    # Centroide
    ax.scatter(centroid_3d[0], centroid_3d[1], centroid_3d[2], 
              c='red', marker='*', s=1000, 
              label='Centroide', edgecolors='darkred', linewidth=2, zorder=10)
    
    # Configuración
    ax.set_xlabel('Dimensión Latente 1', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Dimensión Latente 2', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Dimensión Latente 3', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Espacio Latente 3D - Hiperesfera de Detección\n' + 
                'Sistema de Detección de Ranas Chilenas', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    
    # Rotar para mejor vista
    ax.view_init(elev=20, azim=45)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualización 3D guardada: {save_path}")
    plt.close()


def plot_distance_distributions(frog_latents, other_latents, centroid, radius, save_path):
    """
    Visualización de distribuciones de distancias al centroide.
    """
    # Calcular distancias
    frog_distances = np.linalg.norm(frog_latents - centroid, axis=1)
    other_distances = np.linalg.norm(other_latents - centroid, axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Histograma combinado
    ax = axes[0, 0]
    bins = np.linspace(0, max(frog_distances.max(), other_distances.max()), 40)
    
    ax.hist(frog_distances, bins=bins, alpha=0.7, color='#2ecc71', 
           label=f'Ranas (n={len(frog_distances)})', edgecolor='darkgreen', linewidth=1.5)
    ax.hist(other_distances, bins=bins, alpha=0.7, color='#3498db', 
           label=f'Otros sonidos (n={len(other_distances)})', edgecolor='darkblue', linewidth=1.5)
    
    ax.axvline(radius, color='red', linestyle='--', linewidth=3, 
              label=f'Radio = {radius:.4f}', zorder=10)
    
    ax.set_xlabel('Distancia al Centroide', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Distancias al Centroide', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Box plot
    ax = axes[0, 1]
    data_to_plot = [frog_distances, other_distances]
    labels = ['Ranas', 'Otros Sonidos']
    colors = ['#2ecc71', '#3498db']
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                   notch=True, showmeans=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(2)
    
    # Estilizar
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], linewidth=2)
    
    ax.axhline(radius, color='red', linestyle='--', linewidth=3, 
              label=f'Radio = {radius:.4f}', zorder=10)
    ax.set_ylabel('Distancia al Centroide', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Distancias (Box Plot)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Violin plot
    ax = axes[1, 0]
    parts = ax.violinplot([frog_distances, other_distances], 
                          positions=[1, 2], showmeans=True, showmedians=True)
    
    # Colorear violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    ax.axhline(radius, color='red', linestyle='--', linewidth=3, 
              label=f'Radio = {radius:.4f}', zorder=10)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Distancia al Centroide', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Distancias (Violin Plot)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Estadísticas en tabla
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_data = [
        ['Métrica', 'Ranas', 'Otros Sonidos'],
        ['Media', f'{frog_distances.mean():.4f}', f'{other_distances.mean():.4f}'],
        ['Std', f'{frog_distances.std():.4f}', f'{other_distances.std():.4f}'],
        ['Min', f'{frog_distances.min():.4f}', f'{other_distances.min():.4f}'],
        ['Max', f'{frog_distances.max():.4f}', f'{other_distances.max():.4f}'],
        ['Radio', f'{radius:.4f}', f'{radius:.4f}'],
        ['% dentro', f'{(frog_distances <= radius).mean()*100:.1f}%', 
         f'{(other_distances <= radius).mean()*100:.1f}%'],
    ]
    
    table = ax.table(cellText=stats_data, cellLoc='center', loc='center',
                    colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Estilizar header
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternar colores de filas
    for i in range(1, len(stats_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    ax.set_title('Estadísticas de Detección', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Análisis de Distancias al Centroide - Sistema de Detección de Ranas', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Distribuciones guardadas: {save_path}")
    plt.close()


def plot_training_history(save_path):
    """
    Simula el historial de entrenamiento del VAE.
    """
    epochs = np.arange(1, 101)
    
    # Simular pérdidas decrecientes
    np.random.seed(42)
    total_loss = 1000 * np.exp(-epochs / 30) + 200 + np.random.normal(0, 20, 100)
    recon_loss = 800 * np.exp(-epochs / 30) + 150 + np.random.normal(0, 15, 100)
    kl_loss = 200 * np.exp(-epochs / 30) + 50 + np.random.normal(0, 5, 100)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Total Loss
    ax = axes[0]
    ax.plot(epochs, total_loss, linewidth=2, color='#e74c3c', label='Total Loss')
    ax.fill_between(epochs, total_loss - 20, total_loss + 20, alpha=0.2, color='#e74c3c')
    ax.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Reconstruction Loss
    ax = axes[1]
    ax.plot(epochs, recon_loss, linewidth=2, color='#3498db', label='Reconstruction Loss')
    ax.fill_between(epochs, recon_loss - 15, recon_loss + 15, alpha=0.2, color='#3498db')
    ax.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Reconstruction Loss (MSE)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # KL Divergence
    ax = axes[2]
    ax.plot(epochs, kl_loss, linewidth=2, color='#9b59b6', label='KL Divergence')
    ax.fill_between(epochs, kl_loss - 5, kl_loss + 5, alpha=0.2, color='#9b59b6')
    ax.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('KL Divergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Historial de Entrenamiento del VAE', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Historial de entrenamiento guardado: {save_path}")
    plt.close()


def plot_spectrogram_examples(save_path):
    """
    Simula ejemplos de espectrogramas de ranas.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    np.random.seed(42)
    
    titles = [
        'Rana Chilena 1 - Telmatobius',
        'Rana Chilena 2 - Alsodes', 
        'Rana Chilena 3 - Rhinella',
        'Pájaro - No Rana',
        'Grillo - No Rana',
        'Ruido Ambiental - No Rana'
    ]
    
    for idx, (ax, title) in enumerate(zip(axes.flat, titles)):
        # Generar espectrograma sintético
        if idx < 3:  # Ranas
            # Patrones más estructurados para ranas
            t = np.linspace(0, 5, 128)
            f = np.linspace(0, 128, 128)
            T, F = np.meshgrid(t, f)
            
            # Llamado de rana: pulsos periódicos
            spec = np.sin(2 * np.pi * (2 + idx) * T) * np.exp(-F/30)
            spec += 0.5 * np.sin(2 * np.pi * (4 + idx*0.5) * T) * np.exp(-F/40)
            spec += np.random.normal(0, 0.1, spec.shape)
        else:  # Otros sonidos
            # Patrones diferentes
            if idx == 3:  # Pájaro
                spec = np.random.normal(0, 0.3, (128, 128))
                spec[60:80, :] += 0.8  # Banda de frecuencias
            elif idx == 4:  # Grillo
                spec = np.random.normal(0, 0.2, (128, 128))
                spec[90:100, :] += 1.0  # Alta frecuencia
            else:  # Ruido
                spec = np.random.normal(0, 0.5, (128, 128))
        
        # Normalizar
        spec = (spec - spec.min()) / (spec.max() - spec.min())
        
        # Plotear
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis', 
                      interpolation='bilinear')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Tiempo (frames)', fontsize=10)
        ax.set_ylabel('Frecuencia Mel', fontsize=10)
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Borde verde para ranas, azul para otros
        color = 'green' if idx < 3 else 'blue'
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.suptitle('Ejemplos de Espectrogramas Mel - Ranas vs Otros Sonidos', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Ejemplos de espectrogramas guardados: {save_path}")
    plt.close()


def plot_reconstruction_comparison(save_path):
    """
    Simula la comparación entre espectrograma original y reconstruido.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    np.random.seed(42)
    
    for row in range(3):
        # Generar espectrograma "original"
        t = np.linspace(0, 5, 128)
        f = np.linspace(0, 128, 128)
        T, F = np.meshgrid(t, f)
        
        original = np.sin(2 * np.pi * (2 + row) * T) * np.exp(-F/30)
        original += 0.5 * np.sin(2 * np.pi * (4 + row*0.5) * T) * np.exp(-F/40)
        original += np.random.normal(0, 0.1, original.shape)
        original = (original - original.min()) / (original.max() - original.min())
        
        # Simular reconstrucción (ligeramente suavizada)
        from scipy.ndimage import gaussian_filter
        reconstructed = gaussian_filter(original, sigma=0.8)
        reconstructed += np.random.normal(0, 0.05, reconstructed.shape)
        reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())
        
        # Error
        error = np.abs(original - reconstructed)
        
        # Original
        im1 = axes[row, 0].imshow(original, aspect='auto', origin='lower', cmap='viridis')
        axes[row, 0].set_title(f'Original {row+1}', fontsize=12, fontweight='bold')
        axes[row, 0].set_ylabel('Freq. Mel', fontsize=10)
        plt.colorbar(im1, ax=axes[row, 0], fraction=0.046, pad=0.04)
        
        # Reconstruido
        im2 = axes[row, 1].imshow(reconstructed, aspect='auto', origin='lower', cmap='viridis')
        axes[row, 1].set_title(f'Reconstruido {row+1}', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[row, 1], fraction=0.046, pad=0.04)
        
        # Error
        im3 = axes[row, 2].imshow(error, aspect='auto', origin='lower', cmap='hot')
        axes[row, 2].set_title(f'Error Absoluto {row+1}', fontsize=12, fontweight='bold')
        plt.colorbar(im3, ax=axes[row, 2], fraction=0.046, pad=0.04)
        
        # MSE
        mse = np.mean(error**2)
        axes[row, 2].text(0.5, -0.15, f'MSE = {mse:.6f}', 
                         transform=axes[row, 2].transAxes, 
                         ha='center', fontsize=10, 
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if row == 2:
            axes[row, 0].set_xlabel('Tiempo', fontsize=10)
            axes[row, 1].set_xlabel('Tiempo', fontsize=10)
            axes[row, 2].set_xlabel('Tiempo', fontsize=10)
    
    plt.suptitle('Reconstrucción del VAE - Original vs Reconstruido', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparación de reconstrucción guardada: {save_path}")
    plt.close()


def main():
    """Genera todas las visualizaciones de ejemplo."""
    
    print("\n" + "="*70)
    print("Generando Visualizaciones de Ejemplo")
    print("="*70 + "\n")
    
    # Generar datos sintéticos
    print("Generando datos sintéticos del espacio latente...")
    frog_latents, other_latents, centroid, radius = generate_synthetic_latent_data()
    print(f"  ✓ Ranas: {len(frog_latents)} puntos")
    print(f"  ✓ Otros sonidos: {len(other_latents)} puntos")
    print(f"  ✓ Radio: {radius:.4f}\n")
    
    # Generar visualizaciones
    plot_2d_latent_space(
        frog_latents, other_latents, centroid, radius,
        f'{output_dir}/01_espacio_latente_2d.png'
    )
    
    plot_3d_latent_space(
        frog_latents, other_latents, centroid, radius,
        f'{output_dir}/02_espacio_latente_3d.png'
    )
    
    plot_distance_distributions(
        frog_latents, other_latents, centroid, radius,
        f'{output_dir}/03_distribucion_distancias.png'
    )
    
    plot_training_history(
        f'{output_dir}/04_historial_entrenamiento.png'
    )
    
    plot_spectrogram_examples(
        f'{output_dir}/05_ejemplos_espectrogramas.png'
    )
    
    plot_reconstruction_comparison(
        f'{output_dir}/06_reconstruccion_vae.png'
    )
    
    print("\n" + "="*70)
    print("✓ Todas las visualizaciones generadas exitosamente")
    print("="*70)
    print(f"\nArchivos guardados en: {output_dir}/")
    print("\nVisualizaciones generadas:")
    print("  1. Espacio Latente 2D con hiperesfera")
    print("  2. Espacio Latente 3D con esfera de detección")
    print("  3. Distribución de distancias al centroide")
    print("  4. Historial de entrenamiento del VAE")
    print("  5. Ejemplos de espectrogramas")
    print("  6. Comparación original vs reconstruido")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
