"""
Script de Entrenamiento del VAE para Detección de Ranas
========================================================
Entrena el modelo VAE usando el dataset de chunks de audio procesado.

Funcionalidades:
- Carga dataset de chunks de 5 segundos
- Entrena modelo VAE
- Guarda checkpoints
- Genera visualizaciones de reconstrucción
- Calcula y guarda centroide + radio para detección

Autor: Sistema de Detección de Ranas
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import sys

# Importar módulos del proyecto
sys.path.append(str(Path(__file__).parent))
from models.vae_model import AudioVAE, vae_loss_function
from models.audio_processor import AudioProcessor


class AudioDataset(Dataset):
    """Dataset de espectrogramas de audio."""
    
    def __init__(self, audio_dir, processor):
        """
        Args:
            audio_dir: Directorio con archivos de audio
            processor: Instancia de AudioProcessor
        """
        self.audio_dir = Path(audio_dir)
        self.processor = processor
        
        # Obtener lista de archivos de audio
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        self.audio_files = []
        for ext in audio_extensions:
            self.audio_files.extend(self.audio_dir.glob(f'*{ext}'))
        
        self.audio_files = sorted(self.audio_files)
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No se encontraron archivos de audio en {audio_dir}")
        
        print(f"✓ Dataset creado con {len(self.audio_files)} archivos")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """
        Retorna un espectrograma procesado.
        
        Returns:
            tensor: Espectrograma como tensor (1, H, W)
            filename: Nombre del archivo
        """
        filepath = self.audio_files[idx]
        
        try:
            # Procesar audio a tensor
            spec_tensor = self.processor.process_audio_to_tensor(str(filepath))
            return spec_tensor, filepath.name
        except Exception as e:
            # Si hay error, retornar espectrograma vacío
            print(f"Error procesando {filepath.name}: {e}")
            return torch.zeros(1, *self.processor.target_shape), filepath.name


def train_vae(model, train_loader, optimizer, device, beta=1.0):
    """
    Entrena el VAE por una época.
    
    Args:
        model: Modelo VAE
        train_loader: DataLoader de entrenamiento
        optimizer: Optimizador
        device: Dispositivo (cpu/cuda)
        beta: Factor beta para β-VAE
        
    Returns:
        avg_loss: Pérdida promedio
        avg_recon: Pérdida de reconstrucción promedio
        avg_kl: Pérdida KL promedio
    """
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    pbar = tqdm(train_loader, desc="Entrenando")
    
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Calcular pérdida
        loss, recon_loss, kl_loss = vae_loss_function(
            recon_batch, data, mu, logvar, beta=beta
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Acumular pérdidas
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        
        # Actualizar barra de progreso
        pbar.set_postfix({
            'loss': loss.item() / len(data),
            'recon': recon_loss.item() / len(data),
            'kl': kl_loss.item() / len(data)
        })
    
    avg_loss = total_loss / len(train_loader.dataset)
    avg_recon = total_recon / len(train_loader.dataset)
    avg_kl = total_kl / len(train_loader.dataset)
    
    return avg_loss, avg_recon, avg_kl


def visualize_reconstruction(model, data_loader, device, save_path, num_samples=5):
    """
    Visualiza reconstrucciones del modelo.
    
    Args:
        model: Modelo VAE
        data_loader: DataLoader
        device: Dispositivo
        save_path: Ruta para guardar figura
        num_samples: Número de muestras a visualizar
    """
    model.eval()
    
    with torch.no_grad():
        # Obtener un batch
        data, filenames = next(iter(data_loader))
        data = data[:num_samples].to(device)
        
        # Reconstruir
        recon, _, _ = model(data)
        
        # Mover a CPU
        data = data.cpu().numpy()
        recon = recon.cpu().numpy()
        
        # Crear figura
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 2))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Original
            axes[i, 0].imshow(data[i, 0], aspect='auto', origin='lower', cmap='viridis')
            axes[i, 0].set_title(f'Original: {filenames[i][:20]}...')
            axes[i, 0].set_ylabel('Mel Bins')
            if i == num_samples - 1:
                axes[i, 0].set_xlabel('Time')
            else:
                axes[i, 0].set_xticks([])
            
            # Reconstruido
            axes[i, 1].imshow(recon[i, 0], aspect='auto', origin='lower', cmap='viridis')
            axes[i, 1].set_title('Reconstruido')
            axes[i, 1].set_ylabel('Mel Bins')
            if i == num_samples - 1:
                axes[i, 1].set_xlabel('Time')
            else:
                axes[i, 1].set_xticks([])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualización de reconstrucción guardada en: {save_path}")


def compute_latent_statistics(model, data_loader, device):
    """
    Calcula el centroide y radio del espacio latente.
    
    Args:
        model: Modelo VAE entrenado
        data_loader: DataLoader con datos de ranas
        device: Dispositivo
        
    Returns:
        centroid: Centroide del espacio latente
        radius: Radio (distancia máxima al centroide)
        all_latents: Todas las representaciones latentes
    """
    model.eval()
    all_latents = []
    
    print("\nCalculando estadísticas del espacio latente...")
    
    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Procesando"):
            data = data.to(device)
            mu = model.get_latent_representation(data)
            all_latents.append(mu.cpu().numpy())
    
    # Concatenar todos los vectores latentes
    all_latents = np.concatenate(all_latents, axis=0)
    
    # Calcular centroide (media)
    centroid = np.mean(all_latents, axis=0)
    
    # Calcular distancias al centroide
    distances = np.linalg.norm(all_latents - centroid, axis=1)
    
    # Radio = distancia máxima
    radius = np.max(distances)
    
    # Estadísticas adicionales
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    percentile_95 = np.percentile(distances, 95)
    percentile_99 = np.percentile(distances, 99)
    
    print(f"\n{'='*70}")
    print("Estadísticas del Espacio Latente")
    print(f"{'='*70}")
    print(f"Dimensión latente: {all_latents.shape[1]}")
    print(f"Total de muestras: {all_latents.shape[0]}")
    print(f"Centroide shape: {centroid.shape}")
    print(f"\nDistancias al centroide:")
    print(f"  Media: {mean_distance:.4f}")
    print(f"  Std: {std_distance:.4f}")
    print(f"  Mínima: {np.min(distances):.4f}")
    print(f"  Máxima (radio): {radius:.4f}")
    print(f"  Percentil 95: {percentile_95:.4f}")
    print(f"  Percentil 99: {percentile_99:.4f}")
    print(f"{'='*70}\n")
    
    return centroid, radius, all_latents


def save_detector_config(centroid, radius, model_config, save_path):
    """
    Guarda la configuración del detector (centroide + radio).
    
    Args:
        centroid: Centroide del espacio latente
        radius: Radio de detección
        model_config: Configuración del modelo
        save_path: Ruta donde guardar
    """
    config = {
        'centroid': centroid.tolist(),
        'radius': float(radius),
        'latent_dim': len(centroid),
        'model_config': model_config
    }
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Configuración del detector guardada en: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Entrenar VAE para detección de ranas')
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directorio con chunks de audio procesados (5 segundos)')
    parser.add_argument('--output-dir', type=str, default='./trained_models',
                       help='Directorio donde guardar modelos y resultados')
    parser.add_argument('--latent-dim', type=int, default=16,
                       help='Dimensión del espacio latente')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Tamaño del batch')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Factor beta para β-VAE')
    parser.add_argument('--spec-size', type=int, default=128,
                       help='Tamaño del espectrograma (altura y anchura)')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Dispositivo: {device}")
    print(f"{'='*70}\n")
    
    # Crear procesador de audio
    processor = AudioProcessor(
        sample_rate=22050,
        n_mels=args.spec_size,
        duration=5.0,
        target_shape=(args.spec_size, args.spec_size)
    )
    
    # Crear dataset y dataloader
    print(f"Cargando dataset desde: {args.data_dir}")
    dataset = AudioDataset(args.data_dir, processor)
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0  # Cambia a 4 si tienes múltiples CPUs
    )
    
    # Crear modelo
    model = AudioVAE(
        input_shape=(1, args.spec_size, args.spec_size),
        latent_dim=args.latent_dim
    ).to(device)
    
    print(f"\nModelo creado:")
    print(f"  Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Latent dim: {args.latent_dim}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Historial de entrenamiento
    history = {
        'loss': [],
        'recon_loss': [],
        'kl_loss': []
    }
    
    # Entrenamiento
    print(f"\n{'='*70}")
    print(f"Iniciando entrenamiento por {args.epochs} épocas")
    print(f"{'='*70}\n")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nÉpoca {epoch}/{args.epochs}")
        print("-" * 70)
        
        # Entrenar
        avg_loss, avg_recon, avg_kl = train_vae(
            model, data_loader, optimizer, device, beta=args.beta
        )
        
        # Guardar historial
        history['loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        
        print(f"Pérdida promedio: {avg_loss:.4f} (recon: {avg_recon:.4f}, kl: {avg_kl:.4f})")
        
        # Guardar mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"✓ Mejor modelo guardado (loss: {best_loss:.4f})")
        
        # Checkpoint cada 10 épocas
        if epoch % 10 == 0:
            torch.save(model.state_dict(), output_dir / f'model_epoch_{epoch}.pth')
            visualize_reconstruction(
                model, data_loader, device, 
                output_dir / f'reconstruction_epoch_{epoch}.png'
            )
    
    # Guardar modelo final
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    print(f"\n✓ Modelo final guardado")
    
    # Graficar historial de entrenamiento
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'])
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
    plt.savefig(output_dir / 'training_history.png', dpi=150)
    plt.close()
    print(f"✓ Historial de entrenamiento guardado")
    
    # Calcular estadísticas del espacio latente
    centroid, radius, all_latents = compute_latent_statistics(model, data_loader, device)
    
    # Guardar centroide y radio
    np.save(output_dir / 'centroid.npy', centroid)
    np.save(output_dir / 'all_latents.npy', all_latents)
    
    # Guardar configuración del detector
    model_config = {
        'input_shape': [1, args.spec_size, args.spec_size],
        'latent_dim': args.latent_dim,
        'sample_rate': 22050,
        'n_mels': args.spec_size,
        'duration': 5.0
    }
    
    save_detector_config(centroid, radius, model_config, output_dir / 'detector_config.json')
    
    print(f"\n{'='*70}")
    print("✓ Entrenamiento completado exitosamente")
    print(f"{'='*70}\n")
    print(f"Archivos generados en: {output_dir}")
    print(f"  - best_model.pth (mejor modelo)")
    print(f"  - final_model.pth (modelo final)")
    print(f"  - centroid.npy (centroide del espacio latente)")
    print(f"  - detector_config.json (configuración del detector)")
    print(f"  - training_history.png (gráficos de entrenamiento)")
    print(f"  - all_latents.npy (todos los vectores latentes)")


if __name__ == "__main__":
    main()
