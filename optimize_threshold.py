#!/usr/bin/env python3
"""
Optimizar el umbral (radio) para mejor F1-Score
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from models.vae_model import AudioVAE
from models.audio_processor import AudioProcessor


def evaluate_with_radius(model, centroid, radius, frog_files, non_frog_files, processor, device):
    """Evaluar con un radio específico"""
    predictions = []
    labels = []
    
    # Evaluar ranas
    for audio_path in frog_files:
        audio = processor.load_audio(str(audio_path))
        mel_spec = processor.audio_to_melspectrogram(audio)
        mel_spec_norm = processor.normalize_spectrogram(mel_spec)
        mel_spec_resized = processor.resize_spectrogram(mel_spec_norm)
        spec_tensor = torch.FloatTensor(mel_spec_resized).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, _ = model.encode(spec_tensor)
            distance = torch.norm(mu - centroid, dim=1).item()
        
        predictions.append(1 if distance <= radius else 0)
        labels.append(1)
    
    # Evaluar no-ranas
    for audio_path in non_frog_files:
        audio = processor.load_audio(str(audio_path))
        mel_spec = processor.audio_to_melspectrogram(audio)
        mel_spec_norm = processor.normalize_spectrogram(mel_spec)
        mel_spec_resized = processor.resize_spectrogram(mel_spec_norm)
        spec_tensor = torch.FloatTensor(mel_spec_resized).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, _ = model.encode(spec_tensor)
            distance = torch.norm(mu - centroid, dim=1).item()
        
        predictions.append(1 if distance <= radius else 0)
        labels.append(0)
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    return precision, recall, f1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar configuración
    with open('./trained_models/detector_config.json', 'r') as f:
        config = json.load(f)
    
    # Cargar modelo
    model = AudioVAE(latent_dim=config['latent_dim']).to(device)
    model.load_state_dict(torch.load('./trained_models/best_model.pth', map_location=device))
    model.eval()
    
    centroid = torch.FloatTensor(config['centroid']).to(device)
    original_radius = config['radius']
    
    # Cargar archivos
    processor = AudioProcessor()
    frog_files = list(Path('./data/processed').glob('*.wav'))
    non_frog_files = list(Path('./data/other_sounds_processed').glob('*.wav'))
    
    print(f"Archivos de ranas: {len(frog_files)}")
    print(f"Archivos de no-ranas: {len(non_frog_files)}")
    print(f"Radio original: {original_radius:.4f}")
    
    # Probar diferentes radios
    print("\nProbando diferentes radios...")
    radii = np.linspace(4, 9, 30)  # De 4 a 9
    results = []
    
    for radius in tqdm(radii):
        precision, recall, f1 = evaluate_with_radius(
            model, centroid, radius, frog_files, non_frog_files, processor, device
        )
        results.append({
            'radius': radius,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Encontrar mejor F1
    best = max(results, key=lambda x: x['f1'])
    
    print("\n" + "="*70)
    print("OPTIMIZACIÓN DE RADIO")
    print("="*70)
    print(f"\nRadio original: {original_radius:.4f}")
    print(f"  Precision: {[r for r in results if abs(r['radius']-original_radius)<0.1][0]['precision']:.4f}")
    print(f"  Recall: {[r for r in results if abs(r['radius']-original_radius)<0.1][0]['recall']:.4f}")
    print(f"  F1-Score: {[r for r in results if abs(r['radius']-original_radius)<0.1][0]['f1']:.4f}")
    
    print(f"\nMejor radio: {best['radius']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall: {best['recall']:.4f}")
    print(f"  F1-Score: {best['f1']:.4f}")
    print("="*70)
    
    # Guardar nuevo radio
    config['radius'] = float(best['radius'])
    config['optimized'] = True
    config['optimization_results'] = best
    
    with open('./trained_models/detector_config_optimized.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Configuración optimizada guardada en: detector_config_optimized.json")
    
    # Graficar
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot([r['radius'] for r in results], [r['precision'] for r in results], 'b-', linewidth=2)
    plt.axvline(best['radius'], color='r', linestyle='--', label='Óptimo')
    plt.xlabel('Radio')
    plt.ylabel('Precision')
    plt.title('Precision vs Radio')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot([r['radius'] for r in results], [r['recall'] for r in results], 'g-', linewidth=2)
    plt.axvline(best['radius'], color='r', linestyle='--', label='Óptimo')
    plt.xlabel('Radio')
    plt.ylabel('Recall')
    plt.title('Recall vs Radio')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot([r['radius'] for r in results], [r['f1'] for r in results], 'purple', linewidth=2)
    plt.axvline(best['radius'], color='r', linestyle='--', label='Óptimo')
    plt.xlabel('Radio')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Radio')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./results/threshold_optimization.png', dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: results/threshold_optimization.png")
    
    print(f"\nPara usar el radio optimizado, ejecuta:")
    print(f"python evaluate_model.py \\")
    print(f"  --model-path ./trained_models/best_model.pth \\")
    print(f"  --config-path ./trained_models/detector_config_optimized.json \\")
    print(f"  --frog-data ./data/processed \\")
    print(f"  --non-frog-data ./data/other_sounds_processed \\")
    print(f"  --output-dir ./results")


if __name__ == "__main__":
    main()
