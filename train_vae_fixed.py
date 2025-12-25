#!/usr/bin/env python3
"""
Script de Entrenamiento del VAE - Versión Corregida
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al PYTHONPATH
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm

# Importar módulos locales
from models.vae_model import AudioVAE, vae_loss_function
from models.audio_processor import AudioProcessor


class FrogAudioDataset(Dataset):
    """Dataset de audios de ranas preprocesados"""
    
    def __init__(self, audio_dir, processor):
        self.audio_dir = Path(audio_dir)
        self.processor = processor
        self.audio_files = list(self.audio_dir.glob("*.wav"))
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No se encontraron archivos WAV en {audio_dir}")
        
        print(f"Encontrados {len(self.audio_files)} archivos de audio")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Cargar audio usando AudioProcessor
        audio = self.processor.load_audio(str(audio_path))
        
        # Crear mel-spectrogram usando AudioProcessor
        mel_spec = self.processor.audio_to_melspectrogram(audio)
        
        # Normalizar
        mel_spec_norm = self.processor.normalize_spectrogram(mel_spec)
        
        # Redimensionar al tamaño esperado
        mel_spec_resized = self.processor.resize_spectrogram(mel_spec_norm)
        
        # Convertir a tensor [1, H, W]
        spectrogram = torch.FloatTensor(mel_spec_resized).unsqueeze(0)
        
        return spectrogram


def train_vae(args):
    """Función principal de entrenamiento"""
    
    # Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsando device: {device}")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inicializar procesador de audio
    processor = AudioProcessor()
    
    # Cargar dataset
    print(f"\nCargando dataset desde {args.data_dir}...")
    dataset = FrogAudioDataset(args.data_dir, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Crear modelo
    print(f"\nCreando modelo VAE (latent_dim={args.latent_dim})...")
    model = AudioVAE(latent_dim=args.latent_dim).to(device)
    
    # Optimizador
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Historia de entrenamiento
    history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': []
    }
    
    # Entrenamiento
    print(f"\nIniciando entrenamiento por {args.epochs} épocas...")
    print("="*70)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        pbar = tqdm(dataloader, desc=f"Época {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            batch = batch.to(device)
            
            # Forward pass
            recon_batch, mu, logvar = model(batch)
            
            # Calcular pérdida
            loss, recon_loss, kl_loss = vae_loss_function(
                recon_batch, batch, mu, logvar
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Acumular pérdidas
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            
            # Actualizar barra de progreso
            pbar.set_postfix({
                'loss': f'{loss.item():.2f}',
                'recon': f'{recon_loss.item():.2f}',
                'kl': f'{kl_loss.item():.2f}'
            })
        
        # Promediar pérdidas
        avg_loss = epoch_loss / len(dataloader)
        avg_recon = epoch_recon / len(dataloader)
        avg_kl = epoch_kl / len(dataloader)
        
        # Guardar en historia
        history['total_loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        
        # Imprimir progreso
        print(f"\nÉpoca [{epoch+1}/{args.epochs}]")
        print(f"  Loss Total: {avg_loss:.4f}")
        print(f"  Recon Loss: {avg_recon:.4f}")
        print(f"  KL Loss: {avg_kl:.4f}")
        
        # Guardar mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"  ✓ Mejor modelo guardado (loss: {best_loss:.4f})")
        
        print("="*70)
    
    # Guardar modelo final
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    
    # Calcular centroide y radio
    print("\nCalculando centroide y radio para detección...")
    model.eval()
    latent_vectors = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculando espacio latente"):
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            latent_vectors.append(mu.cpu().numpy())
    
    latent_vectors = np.vstack(latent_vectors)
    
    # Calcular centroide
    centroid = np.mean(latent_vectors, axis=0)
    
    # Calcular radio (95 percentil de distancias)
    distances = np.linalg.norm(latent_vectors - centroid, axis=1)
    radius = np.percentile(distances, 95)
    
    print(f"\nCentroide calculado: {centroid[:5]}... (dim={len(centroid)})")
    print(f"Radio calculado: {radius:.4f}")
    print(f"Distancia promedio: {np.mean(distances):.4f}")
    print(f"Distancia std: {np.std(distances):.4f}")
    
    # Guardar centroide y configuración
    np.save(output_dir / 'centroid.npy', centroid)
    
    config = {
        'latent_dim': args.latent_dim,
        'radius': float(radius),
        'centroid': centroid.tolist(),
        'training_samples': len(dataset),
        'epochs': args.epochs,
        'final_loss': float(best_loss)
    }
    
    with open(output_dir / 'detector_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Graficar historia de entrenamiento
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['total_loss'])
    plt.title('Total Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['recon_loss'])
    plt.title('Reconstruction Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['kl_loss'])
    plt.title('KL Divergence')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {output_dir / 'training_history.png'}")
    
    print(f"\n{'='*70}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"Modelo guardado en: {output_dir}")
    print(f"  - best_model.pth")
    print(f"  - detector_config.json")
    print(f"  - centroid.npy")
    print(f"  - training_history.png")
    print(f"\nPróximos pasos:")
    print(f"  1. Visualizar espacio latente")
    print(f"  2. Evaluar modelo")
    print(f"  3. Probar sistema completo")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Entrenar VAE para detección de ranas')
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directorio con audios preprocesados')
    parser.add_argument('--output-dir', type=str, default='./trained_models',
                       help='Directorio de salida para modelo')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Tamaño de batch (default: 16)')
    parser.add_argument('--latent-dim', type=int, default=16,
                       help='Dimensión del espacio latente (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO DE VAE PARA DETECCIÓN DE RANAS")
    print("="*70)
    print(f"Configuración:")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Épocas: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Learning rate: {args.learning_rate}")
    print("="*70)
    
    train_vae(args)


if __name__ == "__main__":
    main()
