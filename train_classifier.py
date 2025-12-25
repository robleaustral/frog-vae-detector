#!/usr/bin/env python3
"""
Entrenar un Clasificador Binario: Rana vs No-Rana
Usa el VAE como extractor de características + clasificador
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
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


class FrogDataset(Dataset):
    """Dataset con ranas y no-ranas"""
    
    def __init__(self, frog_dir, non_frog_dir, processor):
        self.processor = processor
        
        # Cargar archivos de ranas (label=1)
        self.frog_files = list(Path(frog_dir).glob("*.wav"))
        
        # Cargar archivos de no-ranas (label=0)
        self.non_frog_files = list(Path(non_frog_dir).glob("*.wav"))
        
        print(f"Ranas: {len(self.frog_files)}")
        print(f"No-ranas: {len(self.non_frog_files)}")
        
        # Combinar
        self.files = [(f, 1) for f in self.frog_files] + [(f, 0) for f in self.non_frog_files]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path, label = self.files[idx]
        
        # Cargar y procesar
        audio = self.processor.load_audio(str(audio_path))
        mel_spec = self.processor.audio_to_melspectrogram(audio)
        mel_spec_norm = self.processor.normalize_spectrogram(mel_spec)
        mel_spec_resized = self.processor.resize_spectrogram(mel_spec_norm)
        
        spectrogram = torch.FloatTensor(mel_spec_resized).unsqueeze(0)
        label_tensor = torch.FloatTensor([label])
        
        return spectrogram, label_tensor


def train_classifier(args):
    """Entrenar clasificador"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Cargar VAE pre-entrenado
    with open('./trained_models/detector_config.json', 'r') as f:
        vae_config = json.load(f)
    
    vae = AudioVAE(latent_dim=vae_config['latent_dim']).to(device)
    vae.load_state_dict(torch.load('./trained_models/best_model.pth', map_location=device))
    vae.eval()  # Congelar VAE
    
    # Crear clasificador
    classifier = BinaryClassifier(vae_config['latent_dim']).to(device)
    
    # Dataset
    processor = AudioProcessor()
    dataset = FrogDataset('./data/processed', './data/other_sounds_processed', processor)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Optimizer y loss
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Entrenar
    epochs = 50
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nEntrenando clasificador por {epochs} épocas...")
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Train
        classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for specs, labels in tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}"):
            specs, labels = specs.to(device), labels.to(device)
            
            # Extraer características con VAE (frozen)
            with torch.no_grad():
                mu, _ = vae.encode(specs)
            
            # Clasificar
            outputs = classifier(mu)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        # Validation
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                
                mu, _ = vae.encode(specs)
                outputs = classifier(mu)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        # Métricas
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        print(f"Época {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), './trained_models/classifier.pth')
    
    print(f"\n✓ Mejor accuracy de validación: {best_val_acc:.3f}")
    
    # Guardar configuración
    config = {
        'latent_dim': vae_config['latent_dim'],
        'type': 'classifier',
        'best_val_acc': float(best_val_acc)
    }
    
    with open('./trained_models/classifier_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Graficar
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./results/classifier_training.png', dpi=300)
    print(f"✓ Gráfico guardado: results/classifier_training.png")


if __name__ == "__main__":
    train_classifier(None)
