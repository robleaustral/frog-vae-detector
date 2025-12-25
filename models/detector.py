"""
Detector de Ranas basado en Espacio Latente
============================================
Detecta presencia de ranas usando la distancia al centroide en el espacio latente.

Método:
1. Procesa audio → espectrograma
2. Encode con VAE → vector latente
3. Calcula distancia al centroide
4. Si distancia ≤ radio → Rana detectada
5. Si distancia > radio → No es rana

Autor: Sistema de Detección de Ranas
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'models'))
from vae_model import AudioVAE
from audio_processor import AudioProcessor


class FrogDetector:
    """
    Detector de ranas basado en espacio latente del VAE.
    """
    
    def __init__(self, model_path, config_path, device='cpu'):
        """
        Inicializa el detector.
        
        Args:
            model_path: Ruta al modelo VAE entrenado (.pth)
            config_path: Ruta a detector_config.json
            device: Dispositivo ('cpu' o 'cuda')
        """
        self.device = torch.device(device)
        
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.centroid = np.array(self.config['centroid'])
        self.radius = self.config['radius']
        self.latent_dim = self.config['latent_dim']
        
        # Crear y cargar modelo
        model_config = self.config['model_config']
        self.model = AudioVAE(
            input_shape=tuple(model_config['input_shape']),
            latent_dim=self.latent_dim
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Crear procesador de audio
        self.processor = AudioProcessor(
            sample_rate=model_config['sample_rate'],
            n_mels=model_config['n_mels'],
            duration=model_config['duration'],
            target_shape=(model_config['n_mels'], model_config['n_mels'])
        )
        
        print(f"✓ Detector inicializado")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  Radio de detección: {self.radius:.4f}")
        print(f"  Dispositivo: {self.device}")
    
    def detect_from_file(self, audio_path):
        """
        Detecta si un archivo de audio contiene canto de rana.
        
        Args:
            audio_path: Ruta al archivo de audio
            
        Returns:
            dict con:
                - is_frog: bool indicando si es rana
                - distance: distancia al centroide
                - confidence: nivel de confianza (1 - distance/radius)
                - latent_vector: representación latente
        """
        try:
            # Procesar audio
            spec_tensor = self.processor.process_audio_to_tensor(audio_path)
            spec_tensor = spec_tensor.unsqueeze(0).to(self.device)  # Add batch dim
            
            # Obtener representación latente
            with torch.no_grad():
                latent = self.model.get_latent_representation(spec_tensor)
                latent_np = latent.cpu().numpy().squeeze()
            
            # Calcular distancia al centroide
            distance = np.linalg.norm(latent_np - self.centroid)
            
            # Determinar si es rana
            is_frog = distance <= self.radius
            
            # Calcular confianza
            # Confianza = 1 cuando está en el centroide
            # Confianza = 0 cuando está en el radio
            # Confianza < 0 cuando está fuera del radio
            confidence = 1.0 - (distance / self.radius)
            
            return {
                'is_frog': bool(is_frog),
                'distance': float(distance),
                'confidence': float(confidence),
                'radius': float(self.radius),
                'latent_vector': latent_np.tolist()
            }
            
        except Exception as e:
            return {
                'is_frog': False,
                'distance': None,
                'confidence': 0.0,
                'radius': float(self.radius),
                'error': str(e)
            }
    
    def detect_from_audio_array(self, audio_array):
        """
        Detecta desde un array de audio ya cargado.
        
        Args:
            audio_array: Array numpy con señal de audio
            
        Returns:
            dict con resultados de detección
        """
        try:
            # Convertir audio a espectrograma
            mel_spec = self.processor.audio_to_melspectrogram(audio_array)
            mel_spec_resized = self.processor.resize_spectrogram(mel_spec)
            spec_norm = self.processor.normalize_spectrogram(mel_spec_resized)
            
            # Convertir a tensor
            spec_tensor = torch.FloatTensor(spec_norm).unsqueeze(0).unsqueeze(0)
            spec_tensor = spec_tensor.to(self.device)
            
            # Obtener representación latente
            with torch.no_grad():
                latent = self.model.get_latent_representation(spec_tensor)
                latent_np = latent.cpu().numpy().squeeze()
            
            # Calcular distancia
            distance = np.linalg.norm(latent_np - self.centroid)
            is_frog = distance <= self.radius
            confidence = 1.0 - (distance / self.radius)
            
            return {
                'is_frog': bool(is_frog),
                'distance': float(distance),
                'confidence': float(confidence),
                'radius': float(self.radius),
                'latent_vector': latent_np.tolist()
            }
            
        except Exception as e:
            return {
                'is_frog': False,
                'distance': None,
                'confidence': 0.0,
                'radius': float(self.radius),
                'error': str(e)
            }
    
    def batch_detect(self, audio_dir):
        """
        Detecta en múltiples archivos de un directorio.
        
        Args:
            audio_dir: Directorio con archivos de audio
            
        Returns:
            Lista de resultados para cada archivo
        """
        audio_dir = Path(audio_dir)
        audio_files = list(audio_dir.glob('*.wav')) + list(audio_dir.glob('*.mp3'))
        
        results = []
        
        print(f"Procesando {len(audio_files)} archivos...")
        
        for audio_file in audio_files:
            result = self.detect_from_file(str(audio_file))
            result['filename'] = audio_file.name
            results.append(result)
            
            status = "✓ RANA" if result['is_frog'] else "✗ NO-RANA"
            print(f"  {status} | {audio_file.name} | " + 
                  f"Dist: {result['distance']:.4f} | Conf: {result['confidence']:.2f}")
        
        return results
    
    def get_statistics(self, results):
        """
        Calcula estadísticas de detección.
        
        Args:
            results: Lista de resultados de batch_detect
            
        Returns:
            dict con estadísticas
        """
        total = len(results)
        detected = sum(1 for r in results if r['is_frog'])
        not_detected = total - detected
        
        distances = [r['distance'] for r in results if r['distance'] is not None]
        confidences = [r['confidence'] for r in results if 'confidence' in r]
        
        stats = {
            'total_files': total,
            'frogs_detected': detected,
            'not_frogs': not_detected,
            'detection_rate': detected / total if total > 0 else 0,
            'avg_distance': np.mean(distances) if distances else None,
            'std_distance': np.std(distances) if distances else None,
            'avg_confidence': np.mean(confidences) if confidences else None,
            'radius': self.radius
        }
        
        return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detector de ranas basado en espacio latente')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Ruta al modelo VAE (.pth)')
    parser.add_argument('--config-path', type=str, required=True,
                       help='Ruta a detector_config.json')
    parser.add_argument('--audio-file', type=str, default=None,
                       help='Archivo de audio individual a detectar')
    parser.add_argument('--audio-dir', type=str, default=None,
                       help='Directorio con archivos de audio')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Dispositivo a usar')
    
    args = parser.parse_args()
    
    # Crear detector
    detector = FrogDetector(args.model_path, args.config_path, device=args.device)
    
    print(f"\n{'='*70}")
    
    # Detección individual
    if args.audio_file:
        print(f"Detectando en archivo: {args.audio_file}")
        print(f"{'='*70}\n")
        
        result = detector.detect_from_file(args.audio_file)
        
        print(f"Resultado:")
        print(f"  Es rana: {'✓ SÍ' if result['is_frog'] else '✗ NO'}")
        print(f"  Distancia al centroide: {result['distance']:.4f}")
        print(f"  Radio de detección: {result['radius']:.4f}")
        print(f"  Confianza: {result['confidence']:.2f}")
        
        if result['confidence'] >= 0:
            print(f"  Estado: Dentro del radio (RANA DETECTADA)")
        else:
            print(f"  Estado: Fuera del radio (NO ES RANA)")
    
    # Detección por lotes
    elif args.audio_dir:
        print(f"Detectando en directorio: {args.audio_dir}")
        print(f"{'='*70}\n")
        
        results = detector.batch_detect(args.audio_dir)
        
        print(f"\n{'='*70}")
        print("Estadísticas de Detección")
        print(f"{'='*70}")
        
        stats = detector.get_statistics(results)
        
        print(f"\nTotal de archivos: {stats['total_files']}")
        print(f"Ranas detectadas: {stats['frogs_detected']}")
        print(f"No-ranas: {stats['not_frogs']}")
        print(f"Tasa de detección: {stats['detection_rate']*100:.1f}%")
        print(f"\nDistancia promedio: {stats['avg_distance']:.4f}")
        print(f"Desviación estándar: {stats['std_distance']:.4f}")
        print(f"Confianza promedio: {stats['avg_confidence']:.2f}")
        print(f"Radio de detección: {stats['radius']:.4f}")
        
        print(f"\n{'='*70}\n")
    
    else:
        print("Error: Debes especificar --audio-file o --audio-dir")
        print("Uso:")
        print("  python detector.py --model-path modelo.pth --config-path config.json --audio-file audio.wav")
        print("  python detector.py --model-path modelo.pth --config-path config.json --audio-dir ./test_audio")
