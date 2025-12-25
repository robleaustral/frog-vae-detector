"""
Visualizador del Espacio Latente
=================================
Genera visualizaciones del espacio latente del VAE incluyendo:
- Distribución de puntos latentes de ranas
- Hiperesfera de detección (centroide + radio)
- Comparación con otros sonidos (no-ranas)
- Proyecciones 2D y 3D del espacio latente

Autor: Sistema de Detección de Ranas
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from pathlib import Path
import argparse
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys

sys.path.append(str(Path(__file__).parent.parent / 'models'))
from vae_model import AudioVAE
from audio_processor import AudioProcessor


def load_model_and_config(model_path, config_path, device):
    """
    Carga el modelo VAE y su configuración.
    
    Args:
        model_path: Ruta al modelo .pth
        config_path: Ruta al config.json
        device: Dispositivo (cpu/cuda)
        
    Returns:
        model: Modelo cargado
        config: Configuración
    """
    # Cargar configuración
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Crear modelo
    model = AudioVAE(
        input_shape=tuple(config['model_config']['input_shape']),
        latent_dim=config['latent_dim']
    ).to(device)
    
    # Cargar pesos
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, config


def extract_latent_representations(model, audio_dir, processor, device, label="Audio"):
    """
    Extrae representaciones latentes de un directorio de audio.
    
    Args:
        model: Modelo VAE
        audio_dir: Directorio con archivos de audio
        processor: AudioProcessor
        device: Dispositivo
        label: Etiqueta para estos audios
        
    Returns:
        latents: Array de vectores latentes
        filenames: Lista de nombres de archivos
    """
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob('*.wav')) + list(audio_dir.glob('*.mp3'))
    
    if len(audio_files) == 0:
        print(f"⚠ No se encontraron archivos en {audio_dir}")
        return np.array([]), []
    
    latents = []
    filenames = []
    
    print(f"Extrayendo representaciones latentes de {label}...")
    
    with torch.no_grad():
        for audio_file in audio_files:
            try:
                # Procesar audio
                spec_tensor = processor.process_audio_to_tensor(str(audio_file))
                spec_tensor = spec_tensor.unsqueeze(0).to(device)  # Add batch dim
                
                # Obtener representación latente
                mu = model.get_latent_representation(spec_tensor)
                latents.append(mu.cpu().numpy().squeeze())
                filenames.append(audio_file.name)
                
            except Exception as e:
                print(f"  Error procesando {audio_file.name}: {e}")
    
    latents = np.array(latents)
    print(f"✓ Extraídas {len(latents)} representaciones de {label}")
    
    return latents, filenames


def reduce_dimensionality(latents, method='pca', n_components=2):
    """
    Reduce dimensionalidad del espacio latente para visualización.
    
    Args:
        latents: Vectores latentes (N, latent_dim)
        method: 'pca' o 'tsne'
        n_components: Número de componentes (2 o 3)
        
    Returns:
        reduced: Vectores reducidos (N, n_components)
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(latents)
        variance = reducer.explained_variance_ratio_
        print(f"  PCA - Varianza explicada: {variance.sum():.2%}")
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(latents)-1))
        reduced = reducer.fit_transform(latents)
    else:
        raise ValueError(f"Método no soportado: {method}")
    
    return reduced


def plot_latent_space_2d(frog_latents, other_latents, centroid, radius, 
                          method='pca', save_path=None):
    """
    Visualiza el espacio latente en 2D con hiperesfera.
    
    Args:
        frog_latents: Vectores latentes de ranas (N, latent_dim)
        other_latents: Vectores latentes de otros sonidos (M, latent_dim)
        centroid: Centroide del espacio latente
        radius: Radio de la hiperesfera
        method: Método de reducción ('pca' o 'tsne')
        save_path: Ruta para guardar figura
    """
    print(f"\nGenerando visualización 2D ({method.upper()})...")
    
    # Combinar todos los datos para reducir dimensionalidad consistentemente
    all_latents = np.vstack([frog_latents, other_latents, centroid.reshape(1, -1)])
    
    # Reducir dimensionalidad
    all_reduced = reduce_dimensionality(all_latents, method=method, n_components=2)
    
    # Separar de nuevo
    n_frogs = len(frog_latents)
    n_others = len(other_latents)
    
    frogs_2d = all_reduced[:n_frogs]
    others_2d = all_reduced[n_frogs:n_frogs+n_others]
    centroid_2d = all_reduced[-1]
    
    # Calcular radio en espacio reducido
    # Proyectar vectores en la dirección del radio máximo
    distances_2d = np.linalg.norm(frogs_2d - centroid_2d, axis=1)
    radius_2d = np.max(distances_2d)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Dibujar hiperesfera (círculo en 2D)
    circle = plt.Circle(centroid_2d, radius_2d, color='red', fill=False, 
                       linewidth=2, linestyle='--', label=f'Radio de detección ({radius_2d:.2f})')
    ax.add_patch(circle)
    
    # Dibujar puntos de ranas
    ax.scatter(frogs_2d[:, 0], frogs_2d[:, 1], c='green', alpha=0.6, 
              s=50, label=f'Ranas (n={len(frogs_2d)})', edgecolors='darkgreen')
    
    # Dibujar puntos de otros sonidos
    if len(others_2d) > 0:
        ax.scatter(others_2d[:, 0], others_2d[:, 1], c='blue', alpha=0.6, 
                  s=50, label=f'Otros sonidos (n={len(others_2d)})', 
                  marker='x', edgecolors='darkblue')
    
    # Dibujar centroide
    ax.scatter(centroid_2d[0], centroid_2d[1], c='red', marker='*', 
              s=500, label='Centroide', edgecolors='darkred', linewidth=2, zorder=5)
    
    # Configuración
    ax.set_xlabel(f'{method.upper()} Componente 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Componente 2', fontsize=12)
    ax.set_title(f'Espacio Latente (2D - {method.upper()})\n' + 
                f'Visualización de la Hiperesfera de Detección', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figura 2D guardada en: {save_path}")
    
    plt.show()


def plot_latent_space_3d(frog_latents, other_latents, centroid, radius, 
                          method='pca', save_path=None):
    """
    Visualiza el espacio latente en 3D con hiperesfera.
    
    Args:
        frog_latents: Vectores latentes de ranas
        other_latents: Vectores latentes de otros sonidos
        centroid: Centroide del espacio latente
        radius: Radio de la hiperesfera
        method: Método de reducción ('pca' o 'tsne')
        save_path: Ruta para guardar figura
    """
    print(f"\nGenerando visualización 3D ({method.upper()})...")
    
    # Combinar todos los datos
    all_latents = np.vstack([frog_latents, other_latents, centroid.reshape(1, -1)])
    
    # Reducir a 3D
    all_reduced = reduce_dimensionality(all_latents, method=method, n_components=3)
    
    # Separar
    n_frogs = len(frog_latents)
    n_others = len(other_latents)
    
    frogs_3d = all_reduced[:n_frogs]
    others_3d = all_reduced[n_frogs:n_frogs+n_others]
    centroid_3d = all_reduced[-1]
    
    # Calcular radio en 3D
    distances_3d = np.linalg.norm(frogs_3d - centroid_3d, axis=1)
    radius_3d = np.max(distances_3d)
    
    # Crear figura 3D
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibujar esfera
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = radius_3d * np.outer(np.cos(u), np.sin(v)) + centroid_3d[0]
    y_sphere = radius_3d * np.outer(np.sin(u), np.sin(v)) + centroid_3d[1]
    z_sphere = radius_3d * np.outer(np.ones(np.size(u)), np.cos(v)) + centroid_3d[2]
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='red')
    
    # Dibujar puntos de ranas
    ax.scatter(frogs_3d[:, 0], frogs_3d[:, 1], frogs_3d[:, 2], 
              c='green', alpha=0.7, s=50, label=f'Ranas (n={len(frogs_3d)})',
              edgecolors='darkgreen')
    
    # Dibujar puntos de otros sonidos
    if len(others_3d) > 0:
        ax.scatter(others_3d[:, 0], others_3d[:, 1], others_3d[:, 2], 
                  c='blue', alpha=0.7, s=50, label=f'Otros sonidos (n={len(others_3d)})',
                  marker='x', edgecolors='darkblue')
    
    # Dibujar centroide
    ax.scatter(centroid_3d[0], centroid_3d[1], centroid_3d[2], 
              c='red', marker='*', s=500, label='Centroide',
              edgecolors='darkred', linewidths=2)
    
    # Configuración
    ax.set_xlabel(f'{method.upper()} Comp. 1', fontsize=11)
    ax.set_ylabel(f'{method.upper()} Comp. 2', fontsize=11)
    ax.set_zlabel(f'{method.upper()} Comp. 3', fontsize=11)
    ax.set_title(f'Espacio Latente (3D - {method.upper()})\n' + 
                f'Hiperesfera de Detección', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    # Rotar para mejor visualización
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figura 3D guardada en: {save_path}")
    
    plt.show()


def plot_distance_distributions(frog_latents, other_latents, centroid, radius, save_path=None):
    """
    Visualiza la distribución de distancias al centroide.
    
    Args:
        frog_latents: Vectores latentes de ranas
        other_latents: Vectores latentes de otros sonidos
        centroid: Centroide
        radius: Radio de detección
        save_path: Ruta para guardar
    """
    print("\nGenerando distribución de distancias...")
    
    # Calcular distancias
    frog_distances = np.linalg.norm(frog_latents - centroid, axis=1)
    other_distances = np.linalg.norm(other_latents - centroid, axis=1) if len(other_latents) > 0 else np.array([])
    
    # Crear figura
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histograma
    ax = axes[0]
    bins = np.linspace(0, max(frog_distances.max(), 
                             other_distances.max() if len(other_distances) > 0 else frog_distances.max()), 
                       50)
    
    ax.hist(frog_distances, bins=bins, alpha=0.7, color='green', 
           label=f'Ranas (n={len(frog_distances)})', edgecolor='darkgreen')
    
    if len(other_distances) > 0:
        ax.hist(other_distances, bins=bins, alpha=0.7, color='blue', 
               label=f'Otros sonidos (n={len(other_distances)})', edgecolor='darkblue')
    
    ax.axvline(radius, color='red', linestyle='--', linewidth=2, label=f'Radio = {radius:.4f}')
    ax.set_xlabel('Distancia al Centroide', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title('Distribución de Distancias al Centroide', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Box plot
    ax = axes[1]
    data_to_plot = [frog_distances]
    labels = ['Ranas']
    colors = ['green']
    
    if len(other_distances) > 0:
        data_to_plot.append(other_distances)
        labels.append('Otros sonidos')
        colors.append('blue')
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                   notch=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(radius, color='red', linestyle='--', linewidth=2, label=f'Radio = {radius:.4f}')
    ax.set_ylabel('Distancia al Centroide', fontsize=12)
    ax.set_title('Comparación de Distancias (Box Plot)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figura de distribuciones guardada en: {save_path}")
    
    plt.show()
    
    # Imprimir estadísticas
    print(f"\n{'='*70}")
    print("Estadísticas de Distancias")
    print(f"{'='*70}")
    print(f"\nRanas:")
    print(f"  Media: {frog_distances.mean():.4f}")
    print(f"  Std: {frog_distances.std():.4f}")
    print(f"  Min: {frog_distances.min():.4f}")
    print(f"  Max: {frog_distances.max():.4f}")
    print(f"  % dentro del radio: {(frog_distances <= radius).mean() * 100:.2f}%")
    
    if len(other_distances) > 0:
        print(f"\nOtros sonidos:")
        print(f"  Media: {other_distances.mean():.4f}")
        print(f"  Std: {other_distances.std():.4f}")
        print(f"  Min: {other_distances.min():.4f}")
        print(f"  Max: {other_distances.max():.4f}")
        print(f"  % dentro del radio (falsos positivos): {(other_distances <= radius).mean() * 100:.2f}%")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualizar espacio latente del VAE')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Ruta al modelo entrenado (.pth)')
    parser.add_argument('--config-path', type=str, required=True,
                       help='Ruta al detector_config.json')
    parser.add_argument('--frog-data', type=str, required=True,
                       help='Directorio con audios de ranas')
    parser.add_argument('--other-data', type=str, default=None,
                       help='Directorio con otros sonidos (opcional)')
    parser.add_argument('--output-dir', type=str, default='./visualizations',
                       help='Directorio donde guardar visualizaciones')
    parser.add_argument('--method', type=str, choices=['pca', 'tsne'], default='pca',
                       help='Método de reducción de dimensionalidad')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}\n")
    
    # Cargar modelo y configuración
    print("Cargando modelo...")
    model, config = load_model_and_config(args.model_path, args.config_path, device)
    centroid = np.array(config['centroid'])
    radius = config['radius']
    
    print(f"✓ Modelo cargado")
    print(f"  Latent dim: {config['latent_dim']}")
    print(f"  Radio: {radius:.4f}\n")
    
    # Crear procesador
    model_config = config['model_config']
    processor = AudioProcessor(
        sample_rate=model_config['sample_rate'],
        n_mels=model_config['n_mels'],
        duration=model_config['duration'],
        target_shape=(model_config['n_mels'], model_config['n_mels'])
    )
    
    # Extraer representaciones latentes
    frog_latents, _ = extract_latent_representations(
        model, args.frog_data, processor, device, "Ranas"
    )
    
    if args.other_data:
        other_latents, _ = extract_latent_representations(
            model, args.other_data, processor, device, "Otros sonidos"
        )
    else:
        other_latents = np.array([])
        print("⚠ No se proporcionaron otros sonidos para comparar\n")
    
    if len(frog_latents) == 0:
        print("Error: No se pudieron extraer representaciones latentes de ranas")
        return
    
    # Generar visualizaciones
    print(f"\n{'='*70}")
    print("Generando visualizaciones")
    print(f"{'='*70}")
    
    # 2D
    plot_latent_space_2d(
        frog_latents, other_latents, centroid, radius, 
        method=args.method,
        save_path=output_dir / f'latent_space_2d_{args.method}.png'
    )
    
    # 3D
    plot_latent_space_3d(
        frog_latents, other_latents, centroid, radius, 
        method=args.method,
        save_path=output_dir / f'latent_space_3d_{args.method}.png'
    )
    
    # Distribuciones
    plot_distance_distributions(
        frog_latents, other_latents, centroid, radius,
        save_path=output_dir / 'distance_distributions.png'
    )
    
    print(f"\n{'='*70}")
    print("✓ Visualizaciones completadas")
    print(f"{'='*70}\n")
    print(f"Archivos guardados en: {output_dir}")


if __name__ == "__main__":
    main()
