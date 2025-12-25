#!/usr/bin/env python3
"""
Visualización del Espacio Latente del VAE
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
import json
from tqdm import tqdm

from models.vae_model import AudioVAE
from models.audio_processor import AudioProcessor


def load_model_and_config(model_path, config_path, device):
    """Cargar modelo y configuración"""
    
    # Cargar configuración
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Crear modelo
    model = AudioVAE(latent_dim=config['latent_dim']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Cargar centroide
    centroid = np.array(config['centroid'])
    radius = config['radius']
    
    return model, centroid, radius, config


def extract_latent_vectors(model, data_dir, processor, device):
    """Extraer vectores latentes de todos los audios"""
    
    audio_files = list(Path(data_dir).glob("*.wav"))
    latent_vectors = []
    
    print(f"Procesando {len(audio_files)} archivos...")
    
    for audio_path in tqdm(audio_files):
        # Cargar y procesar audio
        audio = processor.load_audio(str(audio_path))
        mel_spec = processor.audio_to_melspectrogram(audio)
        mel_spec_norm = processor.normalize_spectrogram(mel_spec)
        mel_spec_resized = processor.resize_spectrogram(mel_spec_norm)
        
        # Convertir a tensor
        spec_tensor = torch.FloatTensor(mel_spec_resized).unsqueeze(0).unsqueeze(0)
        spec_tensor = spec_tensor.to(device)
        
        # Extraer vector latente
        with torch.no_grad():
            mu, _ = model.encode(spec_tensor)
            latent_vectors.append(mu.cpu().numpy()[0])
    
    return np.array(latent_vectors)


def visualize_2d(latent_vectors, centroid, radius, output_path):
    """Visualización 2D con PCA"""
    
    # Reducir a 2D con PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    centroid_2d = pca.transform(centroid.reshape(1, -1))[0]
    
    # Calcular distancias
    distances = np.linalg.norm(latent_vectors - centroid, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Puntos coloreados por distancia
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                         c=distances, cmap='viridis', 
                         s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Centroide
    plt.scatter(centroid_2d[0], centroid_2d[1], 
               c='red', s=200, marker='*', 
               edgecolors='black', linewidth=2,
               label='Centroide', zorder=5)
    
    # Radio (círculo aproximado en espacio reducido)
    # Nota: esto es una aproximación, el radio real es en espacio latente completo
    circle = plt.Circle(centroid_2d, radius * 0.5, 
                       fill=False, color='red', 
                       linestyle='--', linewidth=2,
                       label=f'Radio aprox. ({radius:.2f})')
    plt.gca().add_patch(circle)
    
    plt.colorbar(scatter, label='Distancia al Centroide')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)')
    plt.title('Espacio Latente del VAE (2D - PCA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_path}")


def visualize_3d(latent_vectors, centroid, radius, output_path):
    """Visualización 3D con PCA"""
    
    from mpl_toolkits.mplot3d import Axes3D
    
    # Reducir a 3D con PCA
    pca = PCA(n_components=3)
    latent_3d = pca.fit_transform(latent_vectors)
    centroid_3d = pca.transform(centroid.reshape(1, -1))[0]
    
    # Calcular distancias
    distances = np.linalg.norm(latent_vectors - centroid, axis=1)
    
    # Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Puntos
    scatter = ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2],
                        c=distances, cmap='viridis',
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Centroide
    ax.scatter(centroid_3d[0], centroid_3d[1], centroid_3d[2],
              c='red', s=300, marker='*',
              edgecolors='black', linewidth=2, label='Centroide')
    
    fig.colorbar(scatter, label='Distancia al Centroide', shrink=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    ax.set_title('Espacio Latente del VAE (3D - PCA)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_path}")


def visualize_distances(latent_vectors, centroid, radius, output_path):
    """Histograma de distancias"""
    
    distances = np.linalg.norm(latent_vectors - centroid, axis=1)
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(radius, color='red', linestyle='--', linewidth=2, 
               label=f'Radio de Detección ({radius:.2f})')
    plt.axvline(np.mean(distances), color='green', linestyle='--', linewidth=2,
               label=f'Distancia Media ({np.mean(distances):.2f})')
    
    plt.xlabel('Distancia Euclidiana al Centroide')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Distancias al Centroide')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualizar espacio latente del VAE')
    parser.add_argument('--model-path', required=True, help='Ruta al modelo .pth')
    parser.add_argument('--config-path', required=True, help='Ruta al config .json')
    parser.add_argument('--frog-data', required=True, help='Directorio con audios de ranas')
    parser.add_argument('--output-dir', default='./visualizations', help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Cargar modelo
    print("\nCargando modelo...")
    model, centroid, radius, config = load_model_and_config(
        args.model_path, args.config_path, device
    )
    
    print(f"Modelo cargado:")
    print(f"  Latent dim: {config['latent_dim']}")
    print(f"  Radio: {radius:.4f}")
    print(f"  Samples entrenamiento: {config['training_samples']}")
    
    # Procesar audios
    processor = AudioProcessor()
    print("\nExtrayendo vectores latentes...")
    latent_vectors = extract_latent_vectors(model, args.frog_data, processor, device)
    
    print(f"\nVectores latentes extraídos: {latent_vectors.shape}")
    
    # Generar visualizaciones
    print("\nGenerando visualizaciones...")
    
    visualize_2d(latent_vectors, centroid, radius, 
                output_dir / 'latent_space_2d_pca.png')
    
    visualize_3d(latent_vectors, centroid, radius,
                output_dir / 'latent_space_3d_pca.png')
    
    visualize_distances(latent_vectors, centroid, radius,
                       output_dir / 'distance_distributions.png')
    
    print(f"\n{'='*70}")
    print("VISUALIZACIÓN COMPLETADA")
    print(f"{'='*70}")
    print(f"Visualizaciones guardadas en: {output_dir}")
    print(f"  - latent_space_2d_pca.png")
    print(f"  - latent_space_3d_pca.png")
    print(f"  - distance_distributions.png")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
