"""
Audio Preprocessor para Dataset de Ranas
==========================================
Este script procesa un directorio de archivos de audio y los estandariza a chunks de 5 segundos.

Funcionalidad:
- Analiza todos los archivos de audio en un directorio
- Descarta audios menores a 5 segundos
- Divide audios mayores a 5 segundos en chunks de 5 segundos
- Descarta fragmentos residuales menores a 5 segundos
- Mantiene la estructura de nombres: archivo_original_chunk1.wav, archivo_original_chunk2.wav, etc.

Autor: Sistema de Detección de Ranas
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm


class AudioPreprocessor:
    """Preprocesador de audio para estandarizar duración a chunks fijos."""
    
    def __init__(self, chunk_duration: float = 5.0, sample_rate: int = 22050):
        """
        Inicializa el preprocesador.
        
        Args:
            chunk_duration: Duración de cada chunk en segundos (default: 5.0)
            sample_rate: Frecuencia de muestreo objetivo (default: 22050 Hz)
        """
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        
        # Estadísticas del procesamiento
        self.stats = {
            'total_files': 0,
            'discarded_short': 0,
            'processed_files': 0,
            'total_chunks': 0,
            'errors': 0
        }
    
    def get_audio_duration(self, filepath: str) -> float:
        """
        Obtiene la duración de un archivo de audio.
        
        Args:
            filepath: Ruta al archivo de audio
            
        Returns:
            Duración en segundos
        """
        try:
            duration = librosa.get_duration(path=filepath)
            return duration
        except Exception as e:
            print(f"Error al leer duración de {filepath}: {e}")
            return 0.0
    
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Carga un archivo de audio y lo resamplea si es necesario.
        
        Args:
            filepath: Ruta al archivo de audio
            
        Returns:
            Tuple de (señal de audio, sample rate)
        """
        try:
            audio, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            print(f"Error al cargar {filepath}: {e}")
            return None, None
    
    def create_chunks(self, audio: np.ndarray, original_filename: str) -> List[Tuple[np.ndarray, str]]:
        """
        Divide un audio en chunks de duración fija.
        
        Args:
            audio: Señal de audio como array numpy
            original_filename: Nombre del archivo original (sin extensión)
            
        Returns:
            Lista de tuplas (chunk_audio, chunk_name)
        """
        chunks = []
        total_samples = len(audio)
        num_complete_chunks = total_samples // self.chunk_samples
        
        for i in range(num_complete_chunks):
            start_sample = i * self.chunk_samples
            end_sample = start_sample + self.chunk_samples
            chunk = audio[start_sample:end_sample]
            
            chunk_name = f"{original_filename}_chunk{i+1}.wav"
            chunks.append((chunk, chunk_name))
        
        return chunks
    
    def save_audio(self, audio: np.ndarray, filepath: str):
        """
        Guarda un chunk de audio en disco.
        
        Args:
            audio: Señal de audio
            filepath: Ruta donde guardar el archivo
        """
        try:
            sf.write(filepath, audio, self.sample_rate)
        except Exception as e:
            print(f"Error al guardar {filepath}: {e}")
            raise
    
    def process_file(self, input_path: str, output_dir: str) -> int:
        """
        Procesa un archivo de audio individual.
        
        Args:
            input_path: Ruta al archivo de entrada
            output_dir: Directorio donde guardar los chunks
            
        Returns:
            Número de chunks generados
        """
        filename = Path(input_path).stem
        
        # Obtener duración
        duration = self.get_audio_duration(input_path)
        
        if duration < self.chunk_duration:
            self.stats['discarded_short'] += 1
            print(f"  ⊗ Descartado (duración {duration:.2f}s < {self.chunk_duration}s): {Path(input_path).name}")
            return 0
        
        # Cargar audio
        audio, sr = self.load_audio(input_path)
        if audio is None:
            self.stats['errors'] += 1
            return 0
        
        # Crear chunks
        chunks = self.create_chunks(audio, filename)
        
        # Guardar chunks
        for chunk_audio, chunk_name in chunks:
            output_path = os.path.join(output_dir, chunk_name)
            self.save_audio(chunk_audio, output_path)
        
        num_chunks = len(chunks)
        residual_duration = duration - (num_chunks * self.chunk_duration)
        
        print(f"  ✓ Procesado: {Path(input_path).name}")
        print(f"    - Duración original: {duration:.2f}s")
        print(f"    - Chunks generados: {num_chunks}")
        print(f"    - Residual descartado: {residual_duration:.2f}s")
        
        self.stats['processed_files'] += 1
        self.stats['total_chunks'] += num_chunks
        
        return num_chunks
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         audio_extensions: List[str] = None):
        """
        Procesa todos los archivos de audio en un directorio.
        
        Args:
            input_dir: Directorio con archivos de audio originales
            output_dir: Directorio donde guardar los chunks procesados
            audio_extensions: Lista de extensiones a procesar (default: wav, mp3, flac, ogg)
        """
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Obtener lista de archivos de audio
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(Path(input_dir).glob(f'*{ext}'))
            audio_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        audio_files = sorted(set(audio_files))
        self.stats['total_files'] = len(audio_files)
        
        if not audio_files:
            print(f"⚠ No se encontraron archivos de audio en {input_dir}")
            return
        
        print(f"\n{'='*70}")
        print(f"Iniciando preprocesamiento de audio")
        print(f"{'='*70}")
        print(f"Directorio de entrada: {input_dir}")
        print(f"Directorio de salida: {output_dir}")
        print(f"Duración de chunk: {self.chunk_duration}s")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Total de archivos encontrados: {len(audio_files)}")
        print(f"{'='*70}\n")
        
        # Procesar cada archivo
        for audio_file in tqdm(audio_files, desc="Procesando archivos"):
            try:
                self.process_file(str(audio_file), output_dir)
            except Exception as e:
                print(f"  ✗ Error procesando {audio_file.name}: {e}")
                self.stats['errors'] += 1
        
        # Mostrar estadísticas finales
        self.print_statistics()
    
    def print_statistics(self):
        """Imprime las estadísticas del procesamiento."""
        print(f"\n{'='*70}")
        print(f"Estadísticas del procesamiento")
        print(f"{'='*70}")
        print(f"Archivos totales analizados:    {self.stats['total_files']}")
        print(f"Archivos procesados exitosos:   {self.stats['processed_files']}")
        print(f"Archivos descartados (cortos):  {self.stats['discarded_short']}")
        print(f"Errores durante procesamiento:  {self.stats['errors']}")
        print(f"Total de chunks generados:      {self.stats['total_chunks']}")
        print(f"{'='*70}\n")
        
        if self.stats['total_chunks'] > 0:
            avg_chunks = self.stats['total_chunks'] / max(1, self.stats['processed_files'])
            print(f"✓ Procesamiento completado exitosamente")
            print(f"  Promedio de chunks por archivo: {avg_chunks:.1f}")
        else:
            print(f"⚠ No se generaron chunks. Verifica los archivos de entrada.")


def main():
    """Función principal para ejecutar desde línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Preprocesa archivos de audio dividiéndolos en chunks de duración fija.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python audio_preprocessor.py -i ./dataset_ranas -o ./dataset_procesado
  python audio_preprocessor.py -i ./raw_audio -o ./chunks -d 10.0 -sr 16000
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Directorio con archivos de audio originales'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Directorio donde guardar los chunks procesados'
    )
    
    parser.add_argument(
        '-d', '--duration',
        type=float,
        default=5.0,
        help='Duración de cada chunk en segundos (default: 5.0)'
    )
    
    parser.add_argument(
        '-sr', '--sample-rate',
        type=int,
        default=22050,
        help='Frecuencia de muestreo objetivo en Hz (default: 22050)'
    )
    
    parser.add_argument(
        '-e', '--extensions',
        type=str,
        nargs='+',
        default=None,
        help='Extensiones de audio a procesar (default: wav mp3 flac ogg m4a)'
    )
    
    args = parser.parse_args()
    
    # Validar que el directorio de entrada existe
    if not os.path.isdir(args.input):
        print(f"Error: El directorio de entrada no existe: {args.input}")
        return 1
    
    # Crear preprocesador
    preprocessor = AudioPreprocessor(
        chunk_duration=args.duration,
        sample_rate=args.sample_rate
    )
    
    # Procesar directorio
    try:
        preprocessor.process_directory(
            input_dir=args.input,
            output_dir=args.output,
            audio_extensions=args.extensions
        )
        return 0
    except Exception as e:
        print(f"\nError fatal durante el procesamiento: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
