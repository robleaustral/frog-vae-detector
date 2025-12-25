#!/usr/bin/env python3
"""
Optimizar umbral del clasificador para mejor F1-Score
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.vae_model import AudioVAE
from models.audio_processor import AudioProcessor


class BinaryClassifier(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.classifier(z)


def get_all_predictions():
    """Obtener todas las probabilidades predichas"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar modelos
    import json
    with open('./trained_models/detector_config.json', 'r') as f:
        vae_config = json.load(f)
    
    vae = AudioVAE(latent_dim=vae_config['latent_dim']).to(device)
    vae.load_state_dict(torch.load('./trained_models/best_model.pth', map_location=device))
    vae.eval()
    
    classifier = BinaryClassifier(vae_config['latent_dim']).to(device)
    classifier.load_state_dict(torch.load('./trained_models/classifier.pth', map_location=device))
    classifier.eval()
    
    # Cargar datos
    processor = AudioProcessor()
    frog_files = list(Path('./data/processed').glob('*.wav'))
    non_frog_files = list(Path('./data/other_sounds_processed').glob('*.wav'))
    
    probabilities = []
    labels = []
    
    # Ranas
    for audio_path in tqdm(frog_files, desc="Procesando ranas"):
        audio = processor.load_audio(str(audio_path))
        mel_spec = processor.audio_to_melspectrogram(audio)
        mel_spec_norm = processor.normalize_spectrogram(mel_spec)
        mel_spec_resized = processor.resize_spectrogram(mel_spec_norm)
        spec_tensor = torch.FloatTensor(mel_spec_resized).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, _ = vae.encode(spec_tensor)
            prob = classifier(mu).item()
        
        probabilities.append(prob)
        labels.append(1)
    
    # No-ranas
    for audio_path in tqdm(non_frog_files, desc="Procesando no-ranas"):
        audio = processor.load_audio(str(audio_path))
        mel_spec = processor.audio_to_melspectrogram(audio)
        mel_spec_norm = processor.normalize_spectrogram(mel_spec)
        mel_spec_resized = processor.resize_spectrogram(mel_spec_norm)
        spec_tensor = torch.FloatTensor(mel_spec_resized).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, _ = vae.encode(spec_tensor)
            prob = classifier(mu).item()
        
        probabilities.append(prob)
        labels.append(0)
    
    return np.array(probabilities), np.array(labels)


def optimize_threshold(probabilities, labels):
    """Encontrar mejor umbral"""
    
    thresholds = np.linspace(0.01, 0.99, 100)
    results = []
    
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return results


def main():
    print("Obteniendo predicciones...")
    probabilities, labels = get_all_predictions()
    
    print(f"\nRango de probabilidades:")
    print(f"  Ranas: [{probabilities[labels==1].min():.4f}, {probabilities[labels==1].max():.4f}]")
    print(f"  No-ranas: [{probabilities[labels==0].min():.4f}, {probabilities[labels==0].max():.4f}]")
    print(f"  Media ranas: {probabilities[labels==1].mean():.4f}")
    print(f"  Media no-ranas: {probabilities[labels==0].mean():.4f}")
    
    print("\nOptimizando umbral...")
    results = optimize_threshold(probabilities, labels)
    
    best = max(results, key=lambda x: x['f1'])
    
    print("\n" + "="*70)
    print("OPTIMIZACIÓN DE UMBRAL")
    print("="*70)
    print(f"\nUmbral actual: 0.5000")
    current = [r for r in results if abs(r['threshold']-0.5)<0.01][0]
    print(f"  Precision: {current['precision']:.4f}")
    print(f"  Recall: {current['recall']:.4f}")
    print(f"  F1-Score: {current['f1']:.4f}")
    
    print(f"\nMejor umbral: {best['threshold']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall: {best['recall']:.4f}")
    print(f"  F1-Score: {best['f1']:.4f}")
    print("="*70)
    
    # Graficar
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot([r['threshold'] for r in results], [r['precision'] for r in results], 'b-', linewidth=2)
    plt.axvline(best['threshold'], color='r', linestyle='--', label='Óptimo')
    plt.axvline(0.5, color='gray', linestyle=':', label='Actual (0.5)')
    plt.xlabel('Umbral')
    plt.ylabel('Precision')
    plt.title('Precision vs Umbral')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot([r['threshold'] for r in results], [r['recall'] for r in results], 'g-', linewidth=2)
    plt.axvline(best['threshold'], color='r', linestyle='--', label='Óptimo')
    plt.axvline(0.5, color='gray', linestyle=':', label='Actual (0.5)')
    plt.xlabel('Umbral')
    plt.ylabel('Recall')
    plt.title('Recall vs Umbral')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot([r['threshold'] for r in results], [r['f1'] for r in results], 'purple', linewidth=2)
    plt.axvline(best['threshold'], color='r', linestyle='--', label='Óptimo')
    plt.axvline(0.5, color='gray', linestyle=':', label='Actual (0.5)')
    plt.xlabel('Umbral')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Umbral')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/classifier_threshold_optimization.png', dpi=300)
    print(f"\n✓ Gráfico guardado: results/classifier_threshold_optimization.png")
    
    # Histograma de probabilidades
    plt.figure(figsize=(10, 6))
    plt.hist(probabilities[labels==1], bins=30, alpha=0.7, label='Ranas', 
             color='green', edgecolor='black')
    plt.hist(probabilities[labels==0], bins=30, alpha=0.7, label='No-Ranas', 
             color='red', edgecolor='black')
    plt.axvline(best['threshold'], color='blue', linestyle='--', linewidth=2, 
                label=f'Umbral óptimo ({best["threshold"]:.3f})')
    plt.axvline(0.5, color='gray', linestyle=':', linewidth=2, label='Umbral actual (0.5)')
    plt.xlabel('Probabilidad Predicha')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Probabilidades')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./results/classifier_probability_distribution.png', dpi=300)
    print(f"✓ Histograma guardado: results/classifier_probability_distribution.png")
    
    print(f"\nPara evaluar con el umbral óptimo, modifica evaluate_classifier.py")
    print(f"y cambia 'output > 0.5' por 'output > {best['threshold']:.4f}'")


if __name__ == "__main__":
    main()
