#!/usr/bin/env python3
"""
Evaluar Clasificador Binario
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

from models.vae_model import AudioVAE
from models.audio_processor import AudioProcessor


class BinaryClassifier(nn.Module):
    """Clasificador sobre el espacio latente del VAE"""
    
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


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar VAE
    with open('./trained_models/detector_config.json', 'r') as f:
        vae_config = json.load(f)
    
    vae = AudioVAE(latent_dim=vae_config['latent_dim']).to(device)
    vae.load_state_dict(torch.load('./trained_models/best_model.pth', map_location=device))
    vae.eval()
    
    # Cargar clasificador
    classifier = BinaryClassifier(vae_config['latent_dim']).to(device)
    classifier.load_state_dict(torch.load('./trained_models/classifier.pth', map_location=device))
    classifier.eval()
    
    # Cargar datos
    processor = AudioProcessor()
    frog_files = list(Path('./data/processed').glob('*.wav'))
    non_frog_files = list(Path('./data/other_sounds_processed').glob('*.wav'))
    
    print(f"Evaluando clasificador...")
    print(f"Ranas: {len(frog_files)}")
    print(f"No-ranas: {len(non_frog_files)}")
    
    predictions = []
    labels = []
    confidences = []
    
    # Evaluar ranas
    for audio_path in tqdm(frog_files, desc="Ranas"):
        audio = processor.load_audio(str(audio_path))
        mel_spec = processor.audio_to_melspectrogram(audio)
        mel_spec_norm = processor.normalize_spectrogram(mel_spec)
        mel_spec_resized = processor.resize_spectrogram(mel_spec_norm)
        spec_tensor = torch.FloatTensor(mel_spec_resized).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, _ = vae.encode(spec_tensor)
            output = classifier(mu).item()
        
        predictions.append(1 if output > 0.5 else 0)
        labels.append(1)
        confidences.append(output)
    
    # Evaluar no-ranas
    for audio_path in tqdm(non_frog_files, desc="No-ranas"):
        audio = processor.load_audio(str(audio_path))
        mel_spec = processor.audio_to_melspectrogram(audio)
        mel_spec_norm = processor.normalize_spectrogram(mel_spec)
        mel_spec_resized = processor.resize_spectrogram(mel_spec_norm)
        spec_tensor = torch.FloatTensor(mel_spec_resized).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, _ = vae.encode(spec_tensor)
            output = classifier(mu).item()
        
        predictions.append(1 if output > 0.5 else 0)
        labels.append(0)
        confidences.append(output)
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    confidences = np.array(confidences)
    
    # Calcular métricas
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    print("\n" + "="*70)
    print("RESULTADOS DEL CLASIFICADOR")
    print("="*70)
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print("\n" + classification_report(labels, predictions, target_names=['No-Rana', 'Rana']))
    
    # Matriz de confusión
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No-Rana', 'Rana'],
                yticklabels=['No-Rana', 'Rana'])
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.title(f'Matriz de Confusión - Clasificador\nF1-Score: {f1:.3f}')
    plt.tight_layout()
    plt.savefig('./results/classifier_confusion_matrix.png', dpi=300)
    print(f"\n✓ Matriz guardada: results/classifier_confusion_matrix.png")
    
    # Distribución de confianzas
    plt.figure(figsize=(10, 6))
    frog_conf = confidences[labels == 1]
    non_frog_conf = confidences[labels == 0]
    
    plt.hist(frog_conf, bins=30, alpha=0.7, label='Ranas', color='green', edgecolor='black')
    plt.hist(non_frog_conf, bins=30, alpha=0.7, label='No-Ranas', color='red', edgecolor='black')
    plt.axvline(0.5, color='blue', linestyle='--', linewidth=2, label='Umbral (0.5)')
    plt.xlabel('Confianza (Probabilidad de ser Rana)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Confianzas del Clasificador')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./results/classifier_confidence_distribution.png', dpi=300)
    print(f"✓ Distribución guardada: results/classifier_confidence_distribution.png")
    
    # Guardar métricas
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    with open('./results/classifier_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Métricas guardadas: results/classifier_metrics.json")
    print("="*70)
    print("\nResumen para tu tesis:")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall: {recall*100:.2f}%")
    print(f"  F1-Score: {f1*100:.2f}%")
    print("="*70)


if __name__ == "__main__":
    evaluate()
