"""
Variational Autoencoder (VAE) para Audio de Ranas
==================================================
Modelo VAE diseñado para aprender representaciones latentes de cantos de ranas.

El modelo procesa espectrogramas mel de audio y los codifica en un espacio latente
de baja dimensionalidad que captura las características esenciales del canto.

Arquitectura:
- Encoder: Convoluciones 2D sobre espectrogramas → espacio latente (μ, σ)
- Decoder: Convoluciones transpuestas → reconstrucción del espectrograma

Autor: Sistema de Detección de Ranas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AudioVAE(nn.Module):
    """
    Variational Autoencoder para espectrogramas de audio.
    
    Args:
        input_shape: Tupla (channels, height, width) del espectrograma de entrada
        latent_dim: Dimensionalidad del espacio latente
    """
    
    def __init__(self, input_shape=(1, 128, 128), latent_dim=16):
        super(AudioVAE, self).__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.channels, self.height, self.width = input_shape
        
        # ============= ENCODER =============
        # Convierte espectrograma → representación latente
        
        self.encoder = nn.Sequential(
            # Conv1: (1, 128, 128) → (32, 64, 64)
            nn.Conv2d(self.channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Conv2: (32, 64, 64) → (64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Conv3: (64, 32, 32) → (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Conv4: (128, 16, 16) → (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Calcular dimensión aplanada después del encoder
        self.flatten_dim = 256 * (self.height // 16) * (self.width // 16)
        
        # Capas fully connected para μ y log(σ²)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # ============= DECODER =============
        # Convierte representación latente → espectrograma reconstruido
        
        # Capa FC para expandir desde latent_dim
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        self.decoder = nn.Sequential(
            # ConvTranspose1: (256, 8, 8) → (128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # ConvTranspose2: (128, 16, 16) → (64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # ConvTranspose3: (64, 32, 32) → (32, 64, 64)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # ConvTranspose4: (32, 64, 64) → (1, 128, 128)
            nn.ConvTranspose2d(32, self.channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Salida entre [0, 1]
        )
    
    def encode(self, x):
        """
        Codifica el input en el espacio latente.
        
        Args:
            x: Tensor de espectrogramas (batch_size, channels, height, width)
            
        Returns:
            mu: Media de la distribución latente
            logvar: Logaritmo de la varianza
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Aplanar
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Truco de reparametrización: z = μ + σ * ε, donde ε ~ N(0,1)
        
        Args:
            mu: Media
            logvar: Logaritmo de varianza
            
        Returns:
            z: Muestra del espacio latente
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decodifica desde el espacio latente a espectrograma.
        
        Args:
            z: Vector latente
            
        Returns:
            Espectrograma reconstruido
        """
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, self.height // 16, self.width // 16)
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x):
        """
        Forward pass completo del VAE.
        
        Args:
            x: Espectrograma de entrada
            
        Returns:
            reconstruction: Espectrograma reconstruido
            mu: Media del espacio latente
            logvar: Log-varianza del espacio latente
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def get_latent_representation(self, x):
        """
        Obtiene solo la representación latente (μ) sin reparametrización.
        Útil para detección de anomalías.
        
        Args:
            x: Espectrograma de entrada
            
        Returns:
            mu: Vector latente (representación determinística)
        """
        mu, _ = self.encode(x)
        return mu


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Función de pérdida del VAE: Reconstrucción + KL Divergence
    
    Loss = MSE(x, recon_x) + β * KL(q(z|x) || p(z))
    
    Args:
        recon_x: Espectrograma reconstruido
        x: Espectrograma original
        mu: Media del espacio latente
        logvar: Log-varianza del espacio latente
        beta: Factor de peso para KL divergence (β-VAE)
        
    Returns:
        total_loss: Pérdida total
        recon_loss: Pérdida de reconstrucción
        kl_loss: Pérdida KL
    """
    # Pérdida de reconstrucción (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Pérdida total
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


class AudioVAESmall(nn.Module):
    """
    Versión más pequeña del VAE para dispositivos con recursos limitados.
    Útil para deployment en celulares.
    """
    
    def __init__(self, input_shape=(1, 64, 64), latent_dim=8):
        super(AudioVAESmall, self).__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.channels, self.height, self.width = input_shape
        
        # Encoder más simple
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.flatten_dim = 64 * (self.height // 8) * (self.width // 8)
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder más simple
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, self.channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 64, self.height // 8, self.width // 8)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def get_latent_representation(self, x):
        mu, _ = self.encode(x)
        return mu


if __name__ == "__main__":
    # Test del modelo
    print("="*70)
    print("Test del modelo VAE")
    print("="*70)
    
    # Modelo estándar
    model = AudioVAE(input_shape=(1, 128, 128), latent_dim=16)
    print(f"\n✓ AudioVAE creado")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Latent dim: {model.latent_dim}")
    print(f"  Total parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 128)
    recon, mu, logvar = model(x)
    
    print(f"\n✓ Forward pass exitoso")
    print(f"  Input shape: {x.shape}")
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Latent (mu) shape: {mu.shape}")
    print(f"  Latent (logvar) shape: {logvar.shape}")
    
    # Test loss
    loss, recon_loss, kl_loss = vae_loss_function(recon, x, mu, logvar)
    print(f"\n✓ Loss calculation exitosa")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Reconstruction loss: {recon_loss.item():.4f}")
    print(f"  KL loss: {kl_loss.item():.4f}")
    
    # Test get_latent_representation
    latent = model.get_latent_representation(x)
    print(f"\n✓ Get latent representation exitoso")
    print(f"  Latent shape: {latent.shape}")
    
    # Modelo pequeño
    print("\n" + "="*70)
    model_small = AudioVAESmall(input_shape=(1, 64, 64), latent_dim=8)
    print(f"\n✓ AudioVAESmall creado")
    print(f"  Input shape: {model_small.input_shape}")
    print(f"  Latent dim: {model_small.latent_dim}")
    print(f"  Total parámetros: {sum(p.numel() for p in model_small.parameters()):,}")
    
    print("\n" + "="*70)
    print("✓ Todos los tests pasaron exitosamente")
    print("="*70)
