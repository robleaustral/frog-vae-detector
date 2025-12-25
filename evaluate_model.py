#!/usr/bin/env python3
"""
Script de Evaluación del Modelo VAE - Versión Simplificada
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import argparse
import json
from tqdm import tqdm

from models.vae_model import AudioVAE
from models.audio_processor import AudioProcessor


class SimpleFrogDetector:
    """Detector simplificado de ranas"""
    
    def __init__(self, model_path, config_path, device):
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Cargar modelo
        self.model = AudioVAE(latent_dim=self.config['latent_dim']).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # Cargar centroide y radio
        self.centroid = torch.FloatTensor(self.config['centroid']).to(device)
        self.radius = self.config['radius']
        self.device = device
    
    def detect(self, spectrogram):
        """
        Detectar si un espectrograma es de una rana
        Returns: (is_frog, distance, confidence)
        """
        with torch.no_grad():
            # Codificar
            mu, _ = self.model.encode(spectrogram)
            
            # Calcular distancia al centroide
            distance = torch.norm(mu - self.centroid, dim=1).item()
            
            # Determinar si es rana (dentro del radio)
            is_frog = distance <= self.radius
            
            # Calcular confianza (inverso normalizado de la distancia)
            confidence = max(0, 1 - (distance / self.radius))
            
            return is_frog, distance, confidence


def load_audio_files(data_dir):
    """Cargar archivos de audio"""
    audio_files = list(Path(data_dir).glob("*.wav"))
    return audio_files


def evaluate_detector(detector, frog_files, non_frog_files, processor, device):
    """Evaluar detector con datos de ranas y no-ranas"""
    
    print("\nEvaluando detector...")
    
    all_predictions = []
    all_labels = []
    all_distances = []
    all_confidences = []
    
    # Evaluar archivos de RANAS (label = 1)
    print(f"\nProcesando {len(frog_files)} archivos de ranas...")
    for audio_path in tqdm(frog_files):
        # Cargar y procesar
        audio = processor.load_audio(str(audio_path))
        mel_spec = processor.audio_to_melspectrogram(audio)
        mel_spec_norm = processor.normalize_spectrogram(mel_spec)
        mel_spec_resized = processor.resize_spectrogram(mel_spec_norm)
        
        # Convertir a tensor
        spec_tensor = torch.FloatTensor(mel_spec_resized).unsqueeze(0).unsqueeze(0)
        spec_tensor = spec_tensor.to(device)
        
        # Detectar
        is_frog, distance, confidence = detector.detect(spec_tensor)
        
        all_predictions.append(1 if is_frog else 0)
        all_labels.append(1)  # True label: es rana
        all_distances.append(distance)
        all_confidences.append(confidence)
    
    # Evaluar archivos de NO-RANAS (label = 0)
    if len(non_frog_files) > 0:
        print(f"\nProcesando {len(non_frog_files)} archivos de no-ranas...")
        for audio_path in tqdm(non_frog_files):
            # Cargar y procesar
            audio = processor.load_audio(str(audio_path))
            mel_spec = processor.audio_to_melspectrogram(audio)
            mel_spec_norm = processor.normalize_spectrogram(mel_spec)
            mel_spec_resized = processor.resize_spectrogram(mel_spec_norm)
            
            # Convertir a tensor
            spec_tensor = torch.FloatTensor(mel_spec_resized).unsqueeze(0).unsqueeze(0)
            spec_tensor = spec_tensor.to(device)
            
            # Detectar
            is_frog, distance, confidence = detector.detect(spec_tensor)
            
            all_predictions.append(1 if is_frog else 0)
            all_labels.append(0)  # True label: NO es rana
            all_distances.append(distance)
            all_confidences.append(confidence)
    
    return (np.array(all_predictions), np.array(all_labels), 
            np.array(all_distances), np.array(all_confidences))


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Graficar matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No-Rana', 'Rana'],
                yticklabels=['No-Rana', 'Rana'])
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_path}")


def plot_distance_distribution(distances, labels, radius, output_path):
    """Graficar distribución de distancias por clase"""
    frog_distances = distances[labels == 1]
    non_frog_distances = distances[labels == 0] if len(distances[labels == 0]) > 0 else []
    
    plt.figure(figsize=(10, 6))
    
    if len(frog_distances) > 0:
        plt.hist(frog_distances, bins=30, alpha=0.7, label='Ranas', 
                color='green', edgecolor='black')
    
    if len(non_frog_distances) > 0:
        plt.hist(non_frog_distances, bins=30, alpha=0.7, label='No-Ranas', 
                color='red', edgecolor='black')
    
    plt.axvline(radius, color='blue', linestyle='--', linewidth=2, 
               label=f'Umbral (radio={radius:.2f})')
    
    plt.xlabel('Distancia al Centroide')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Distancias por Clase')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluar modelo de detección de ranas')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--config-path', required=True)
    parser.add_argument('--frog-data', required=True)
    parser.add_argument('--non-frog-data', default=None)
    parser.add_argument('--output-dir', default='./results')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Crear detector
    print("\nCargando detector...")
    detector = SimpleFrogDetector(args.model_path, args.config_path, device)
    
    # Cargar archivos
    processor = AudioProcessor()
    frog_files = load_audio_files(args.frog_data)
    
    if args.non_frog_data:
        non_frog_files = load_audio_files(args.non_frog_data)
    else:
        non_frog_files = []
        print("\n⚠️  No se proporcionaron archivos de no-ranas.")
        print("   Solo se evaluará la detección de ranas (recall).")
    
    print(f"\nArchivos de ranas: {len(frog_files)}")
    print(f"Archivos de no-ranas: {len(non_frog_files)}")
    
    # Evaluar
    predictions, labels, distances, confidences = evaluate_detector(
        detector, frog_files, non_frog_files, processor, device
    )
    
    # Calcular métricas
    print("\n" + "="*70)
    print("RESULTADOS DE EVALUACIÓN")
    print("="*70)
    
    # Recall (siempre podemos calcularlo)
    recall = recall_score(labels, predictions)
    print(f"\nRecall (Detección de Ranas): {recall:.4f}")
    print(f"Ranas detectadas: {predictions[labels==1].sum()} / {len(frog_files)}")
    
    # Métricas completas si hay no-ranas
    if len(non_frog_files) > 0:
        precision = precision_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        
        print(f"Precision: {precision:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nReporte de Clasificación:")
        print(classification_report(labels, predictions, 
                                    target_names=['No-Rana', 'Rana'],
                                    digits=4))
    
    # Estadísticas de distancias
    print(f"\nEstadísticas de Distancias:")
    print(f"  Radio de detección: {detector.radius:.4f}")
    print(f"  Distancia media (ranas): {distances[labels==1].mean():.4f}")
    print(f"  Distancia std (ranas): {distances[labels==1].std():.4f}")
    print(f"  Distancia min (ranas): {distances[labels==1].min():.4f}")
    print(f"  Distancia max (ranas): {distances[labels==1].max():.4f}")
    
    if len(non_frog_files) > 0:
        print(f"  Distancia media (no-ranas): {distances[labels==0].mean():.4f}")
        print(f"  Distancia std (no-ranas): {distances[labels==0].std():.4f}")
    
    # Guardar métricas
    metrics = {
        'recall': float(recall),
        'total_frogs': int(len(frog_files)),
        'detected_frogs': int(predictions[labels==1].sum()),
        'missed_frogs': int(len(frog_files) - predictions[labels==1].sum()),
        'total_non_frogs': int(len(non_frog_files)),
        'radius': float(detector.radius),
        'mean_distance_frogs': float(distances[labels==1].mean()),
        'std_distance_frogs': float(distances[labels==1].std()),
        'min_distance_frogs': float(distances[labels==1].min()),
        'max_distance_frogs': float(distances[labels==1].max())
    }
    
    if len(non_frog_files) > 0:
        metrics.update({
            'precision': float(precision),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'detected_non_frogs_as_frogs': int(predictions[labels==0].sum()),
            'mean_distance_non_frogs': float(distances[labels==0].mean()),
            'std_distance_non_frogs': float(distances[labels==0].std())
        })
    
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Métricas guardadas: {output_dir / 'evaluation_metrics.json'}")
    
    # Generar gráficos
    print("\nGenerando gráficos...")
    
    if len(non_frog_files) > 0:
        plot_confusion_matrix(labels, predictions, output_dir / 'confusion_matrix.png')
    
    plot_distance_distribution(distances, labels, detector.radius, 
                               output_dir / 'distance_distribution.png')
    
    print(f"\n{'='*70}")
    print("EVALUACIÓN COMPLETADA")
    print(f"{'='*70}")
    print(f"Resultados guardados en: {output_dir}")
    print(f"  - evaluation_metrics.json")
    if len(non_frog_files) > 0:
        print(f"  - confusion_matrix.png")
    print(f"  - distance_distribution.png")
    print(f"{'='*70}\n")
    
    print("\nResumen para tu tesis:")
    print(f"  Recall: {recall*100:.2f}%")
    if len(non_frog_files) > 0:
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  F1-Score: {f1*100:.2f}%")


if __name__ == "__main__":
    main()
