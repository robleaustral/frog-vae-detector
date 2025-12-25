#!/usr/bin/env python3
"""
Comparar diferentes métricas de distancia para detección
- Euclidiana (actual)
- Coseno
- Manhattan
- Mahalanobis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from models.vae_model import AudioVAE
from models.audio_processor import AudioProcessor


def extract_all_latent_vectors(model, frog_files, non_frog_files, processor, device):
    """Extraer todos los vectores latentes"""
    
    all_vectors = []
    all_labels = []
    
    print("Extrayendo vectores latentes...")
    
    # Ranas
    for audio_path in tqdm(frog_files, desc="Ranas"):
        audio = processor.load_audio(str(audio_path))
        mel_spec = processor.audio_to_melspectrogram(audio)
        mel_spec_norm = processor.normalize_spectrogram(mel_spec)
        mel_spec_resized = processor.resize_spectrogram(mel_spec_norm)
        spec_tensor = torch.FloatTensor(mel_spec_resized).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, _ = model.encode(spec_tensor)
            all_vectors.append(mu.cpu().numpy()[0])
            all_labels.append(1)
    
    # No-ranas
    for audio_path in tqdm(non_frog_files, desc="No-ranas"):
        audio = processor.load_audio(str(audio_path))
        mel_spec = processor.audio_to_melspectrogram(audio)
        mel_spec_norm = processor.normalize_spectrogram(mel_spec)
        mel_spec_resized = processor.resize_spectrogram(mel_spec_norm)
        spec_tensor = torch.FloatTensor(mel_spec_resized).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, _ = model.encode(spec_tensor)
            all_vectors.append(mu.cpu().numpy()[0])
            all_labels.append(0)
    
    return np.array(all_vectors), np.array(all_labels)


def compute_euclidean_distances(vectors, centroid):
    """Distancia Euclidiana (L2)"""
    return np.linalg.norm(vectors - centroid, axis=1)


def compute_cosine_distances(vectors, centroid):
    """Distancia de Coseno (1 - similitud)"""
    # Cosine similarity retorna valores en [-1, 1]
    # Convertimos a distancia: 1 - similarity
    similarities = cosine_similarity(vectors, centroid.reshape(1, -1)).flatten()
    distances = 1 - similarities  # Ahora en [0, 2], donde 0 = idéntico
    return distances


def compute_manhattan_distances(vectors, centroid):
    """Distancia Manhattan (L1)"""
    return np.sum(np.abs(vectors - centroid), axis=1)


def compute_chebyshev_distances(vectors, centroid):
    """Distancia Chebyshev (L-infinity)"""
    return np.max(np.abs(vectors - centroid), axis=1)


def optimize_threshold(distances, labels, metric_name):
    """Encontrar mejor umbral para una métrica"""
    
    # Probar diferentes umbrales
    min_dist = distances.min()
    max_dist = distances.max()
    thresholds = np.linspace(min_dist, max_dist, 100)
    
    best_f1 = 0
    best_threshold = 0
    best_metrics = {}
    
    for threshold in thresholds:
        predictions = (distances <= threshold).astype(int)
        
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'threshold': threshold
            }
    
    return best_metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Cargar configuración y modelo
    with open('./trained_models/detector_config.json', 'r') as f:
        config = json.load(f)
    
    model = AudioVAE(latent_dim=config['latent_dim']).to(device)
    model.load_state_dict(torch.load('./trained_models/best_model.pth', map_location=device))
    model.eval()
    
    centroid = np.array(config['centroid'])
    
    # Cargar datos
    processor = AudioProcessor()
    frog_files = list(Path('./data/processed').glob('*.wav'))
    non_frog_files = list(Path('./data/other_sounds_processed').glob('*.wav'))
    
    print(f"\nArchivos de ranas: {len(frog_files)}")
    print(f"Archivos de no-ranas: {len(non_frog_files)}")
    
    # Extraer vectores latentes
    vectors, labels = extract_all_latent_vectors(
        model, frog_files, non_frog_files, processor, device
    )
    
    print(f"\nVectores extraídos: {vectors.shape}")
    
    # Probar diferentes métricas
    metrics_to_test = {
        'Euclidiana (L2)': compute_euclidean_distances,
        'Coseno': compute_cosine_distances,
        'Manhattan (L1)': compute_manhattan_distances,
        'Chebyshev (L∞)': compute_chebyshev_distances
    }
    
    results = {}
    
    print("\n" + "="*70)
    print("COMPARACIÓN DE MÉTRICAS DE DISTANCIA")
    print("="*70)
    
    for metric_name, compute_func in metrics_to_test.items():
        print(f"\n{metric_name}:")
        
        # Calcular distancias
        distances = compute_func(vectors, centroid)
        
        # Optimizar umbral
        best = optimize_threshold(distances, labels, metric_name)
        
        print(f"  Umbral óptimo: {best['threshold']:.4f}")
        print(f"  Precision: {best['precision']:.4f}")
        print(f"  Recall: {best['recall']:.4f}")
        print(f"  F1-Score: {best['f1']:.4f}")
        
        results[metric_name] = best
    
    # Encontrar la mejor métrica
    best_metric = max(results.items(), key=lambda x: x[1]['f1'])
    
    print("\n" + "="*70)
    print("MEJOR MÉTRICA")
    print("="*70)
    print(f"Métrica: {best_metric[0]}")
    print(f"  Precision: {best_metric[1]['precision']:.4f}")
    print(f"  Recall: {best_metric[1]['recall']:.4f}")
    print(f"  F1-Score: {best_metric[1]['f1']:.4f}")
    print(f"  Umbral: {best_metric[1]['threshold']:.4f}")
    print("="*70)
    
    # Graficar comparación
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (metric_name, compute_func) in enumerate(metrics_to_test.items()):
        ax = axes[idx // 2, idx % 2]
        
        distances = compute_func(vectors, centroid)
        
        # Separar por clase
        frog_distances = distances[labels == 1]
        non_frog_distances = distances[labels == 0]
        
        # Histograma
        ax.hist(frog_distances, bins=30, alpha=0.7, label='Ranas', 
               color='green', edgecolor='black')
        ax.hist(non_frog_distances, bins=30, alpha=0.7, label='No-Ranas', 
               color='red', edgecolor='black')
        
        # Umbral óptimo
        best_threshold = results[metric_name]['threshold']
        ax.axvline(best_threshold, color='blue', linestyle='--', linewidth=2,
                  label=f'Umbral ({best_threshold:.2f})')
        
        ax.set_xlabel('Distancia')
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'{metric_name}\nF1={results[metric_name]["f1"]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/distance_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: results/distance_metrics_comparison.png")
    
    # Guardar resultados
    results_summary = {
        'best_metric': best_metric[0],
        'metrics': {k: v for k, v in results.items()}
    }
    
    with open('./results/distance_metrics_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"✓ Resultados guardados: results/distance_metrics_results.json")
    
    # Guardar configuración con mejor métrica
    config_optimized = config.copy()
    config_optimized['distance_metric'] = best_metric[0]
    config_optimized['threshold'] = float(best_metric[1]['threshold'])
    config_optimized['radius'] = float(best_metric[1]['threshold'])  # Para compatibilidad
    
    with open('./trained_models/detector_config_best_metric.json', 'w') as f:
        json.dump(config_optimized, f, indent=2)
    
    print(f"✓ Config optimizada: trained_models/detector_config_best_metric.json")
    
    # Comparación de métricas
    print("\n" + "="*70)
    print("TABLA COMPARATIVA")
    print("="*70)
    print(f"{'Métrica':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    for metric_name, metrics in results.items():
        print(f"{metric_name:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
