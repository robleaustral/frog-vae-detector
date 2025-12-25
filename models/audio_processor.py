"""
Audio Processor - Extracción de características de audio
=========================================================
Procesa archivos de audio y los convierte en espectrogramas para el VAE.

Funcionalidades:
- Carga de archivos de audio
- Extracción de espectrogramas mel
- Normalización y preprocesamiento
- Conversión a tensores PyTorch

Autor: Sistema de Detección de Ranas
"""

import librosa
import librosa.display
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt


class AudioProcessor:
    """
    Procesador de audio para convertir señales en espectrogramas.
    """
    
    def __init__(self, 
                 sample_rate=22050,
                 n_mels=128,
                 n_fft=2048,
                 hop_length=512,
                 duration=5.0,
                 target_shape=(128, 128)):
        """
        Inicializa el procesador de audio.
        
        Args:
            sample_rate: Frecuencia de muestreo en Hz
            n_mels: Número de bandas mel
            n_fft: Tamaño de la FFT
            hop_length: Número de muestras entre frames
            duration: Duración esperada del audio en segundos
            target_shape: Forma objetivo del espectrograma (height, width)
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.target_shape = target_shape
        
        self.expected_samples = int(sample_rate * duration)
    
    def load_audio(self, filepath, offset=0.0):
        """
        Carga un archivo de audio.
        
        Args:
            filepath: Ruta al archivo de audio
            offset: Desplazamiento en segundos desde el inicio
            
        Returns:
            audio: Señal de audio normalizada
        """
        try:
            audio, sr = librosa.load(
                filepath, 
                sr=self.sample_rate, 
                duration=self.duration,
                offset=offset,
                mono=True
            )
            
            # Asegurar que tenga la duración correcta
            if len(audio) < self.expected_samples:
                # Pad con ceros si es muy corto
                audio = np.pad(audio, (0, self.expected_samples - len(audio)))
            elif len(audio) > self.expected_samples:
                # Truncar si es muy largo
                audio = audio[:self.expected_samples]
            
            return audio
            
        except Exception as e:
            raise Exception(f"Error cargando audio {filepath}: {e}")
    
    def audio_to_melspectrogram(self, audio):
        """
        Convierte señal de audio a espectrograma mel.
        
        Args:
            audio: Señal de audio (numpy array)
            
        Returns:
            mel_spec: Espectrograma mel en escala de potencia
        """
        # Calcular espectrograma mel
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=self.sample_rate // 2
        )
        
        # Convertir a escala de dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def resize_spectrogram(self, spec):
        """
        Redimensiona el espectrograma a la forma objetivo.
        
        Args:
            spec: Espectrograma original
            
        Returns:
            Espectrograma redimensionado
        """
        from scipy.ndimage import zoom
        
        h, w = spec.shape
        target_h, target_w = self.target_shape
        
        zoom_h = target_h / h
        zoom_w = target_w / w
        
        spec_resized = zoom(spec, (zoom_h, zoom_w), order=1)
        
        return spec_resized
    
    def normalize_spectrogram(self, spec):
        """
        Normaliza el espectrograma a rango [0, 1].
        
        Args:
            spec: Espectrograma en dB
            
        Returns:
            Espectrograma normalizado
        """
        spec_min = spec.min()
        spec_max = spec.max()
        
        if spec_max - spec_min > 0:
            spec_norm = (spec - spec_min) / (spec_max - spec_min)
        else:
            spec_norm = np.zeros_like(spec)
        
        return spec_norm
    
    def process_audio_file(self, filepath, offset=0.0):
        """
        Pipeline completo: audio → espectrograma normalizado.
        
        Args:
            filepath: Ruta al archivo de audio
            offset: Desplazamiento en segundos
            
        Returns:
            spec_norm: Espectrograma normalizado (numpy array)
        """
        # 1. Cargar audio
        audio = self.load_audio(filepath, offset=offset)
        
        # 2. Convertir a espectrograma mel
        mel_spec = self.audio_to_melspectrogram(audio)
        
        # 3. Redimensionar
        mel_spec_resized = self.resize_spectrogram(mel_spec)
        
        # 4. Normalizar
        spec_norm = self.normalize_spectrogram(mel_spec_resized)
        
        return spec_norm
    
    def process_audio_to_tensor(self, filepath, offset=0.0):
        """
        Procesa audio y lo convierte a tensor de PyTorch.
        
        Args:
            filepath: Ruta al archivo de audio
            offset: Desplazamiento en segundos
            
        Returns:
            tensor: Tensor (1, H, W) listo para el modelo
        """
        spec = self.process_audio_file(filepath, offset=offset)
        
        # Agregar dimensión de canal
        spec = spec[np.newaxis, :]  # (1, H, W)
        
        # Convertir a tensor
        tensor = torch.FloatTensor(spec)
        
        return tensor
    
    def batch_process_directory(self, directory, max_files=None):
        """
        Procesa múltiples archivos de audio en un directorio.
        
        Args:
            directory: Directorio con archivos de audio
            max_files: Máximo número de archivos a procesar (None = todos)
            
        Returns:
            spectrograms: Lista de espectrogramas normalizados
            filenames: Lista de nombres de archivos
        """
        directory = Path(directory)
        
        # Obtener lista de archivos de audio
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(directory.glob(f'*{ext}'))
        
        audio_files = sorted(audio_files)
        
        if max_files:
            audio_files = audio_files[:max_files]
        
        spectrograms = []
        filenames = []
        
        print(f"Procesando {len(audio_files)} archivos de audio...")
        
        for i, filepath in enumerate(audio_files):
            try:
                spec = self.process_audio_file(str(filepath))
                spectrograms.append(spec)
                filenames.append(filepath.name)
                
                if (i + 1) % 10 == 0:
                    print(f"  Procesados {i + 1}/{len(audio_files)} archivos")
                    
            except Exception as e:
                print(f"  Error procesando {filepath.name}: {e}")
        
        print(f"✓ Total procesados exitosamente: {len(spectrograms)}")
        
        return spectrograms, filenames
    
    def visualize_spectrogram(self, spec, title="Espectrograma Mel", save_path=None):
        """
        Visualiza un espectrograma.
        
        Args:
            spec: Espectrograma (2D array)
            title: Título del gráfico
            save_path: Ruta para guardar la figura (opcional)
        """
        plt.figure(figsize=(10, 4))
        
        if spec.max() <= 1.0:
            # Espectrograma normalizado
            plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(format='%.2f', label='Amplitude (normalized)')
        else:
            # Espectrograma en dB
            plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
        
        plt.title(title)
        plt.ylabel('Mel Frequency Bins')
        plt.xlabel('Time Frames')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")
        
        plt.show()
    
    def compare_spectrograms(self, spec1, spec2, title1="Original", 
                            title2="Reconstruido", save_path=None):
        """
        Compara dos espectrogramas lado a lado.
        
        Args:
            spec1: Primer espectrograma
            spec2: Segundo espectrograma
            title1: Título del primer espectrograma
            title2: Título del segundo espectrograma
            save_path: Ruta para guardar la figura (opcional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Espectrograma 1
        im1 = axes[0].imshow(spec1, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title(title1)
        axes[0].set_ylabel('Mel Frequency Bins')
        axes[0].set_xlabel('Time Frames')
        plt.colorbar(im1, ax=axes[0], format='%.2f')
        
        # Espectrograma 2
        im2 = axes[1].imshow(spec2, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title(title2)
        axes[1].set_ylabel('Mel Frequency Bins')
        axes[1].set_xlabel('Time Frames')
        plt.colorbar(im2, ax=axes[1], format='%.2f')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Test del procesador
    print("="*70)
    print("Test del Audio Processor")
    print("="*70)
    
    # Crear procesador
    processor = AudioProcessor(
        sample_rate=22050,
        n_mels=128,
        duration=5.0,
        target_shape=(128, 128)
    )
    
    print(f"\n✓ AudioProcessor creado")
    print(f"  Sample rate: {processor.sample_rate} Hz")
    print(f"  Mel bins: {processor.n_mels}")
    print(f"  Duration: {processor.duration}s")
    print(f"  Target shape: {processor.target_shape}")
    
    # Generar audio sintético para test
    print(f"\n✓ Generando audio sintético para test...")
    t = np.linspace(0, 5, int(22050 * 5))
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Test procesamiento
    mel_spec = processor.audio_to_melspectrogram(test_audio)
    print(f"\n✓ Espectrograma generado")
    print(f"  Shape original: {mel_spec.shape}")
    
    mel_spec_resized = processor.resize_spectrogram(mel_spec)
    print(f"  Shape redimensionado: {mel_spec_resized.shape}")
    
    mel_spec_norm = processor.normalize_spectrogram(mel_spec_resized)
    print(f"  Rango normalizado: [{mel_spec_norm.min():.3f}, {mel_spec_norm.max():.3f}]")
    
    print("\n" + "="*70)
    print("✓ Todos los tests pasaron exitosamente")
    print("="*70)
