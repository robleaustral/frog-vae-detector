"""
Script de prueba para el Audio Preprocessor
============================================
Genera audios sintéticos de diferentes duraciones para probar el preprocesador.

Este script crea un directorio de prueba con audios de ejemplo que simulan
diferentes escenarios:
- Audios muy cortos (< 5 segundos) → deberían descartarse
- Audios exactos de 5 segundos → 1 chunk
- Audios largos con residuales → múltiples chunks + descartar residual
"""

import numpy as np
import soundfile as sf
import os
from pathlib import Path


def generate_sine_wave(frequency, duration, sample_rate=22050):
    """
    Genera una onda sinusoidal simple.
    
    Args:
        frequency: Frecuencia en Hz
        duration: Duración en segundos
        sample_rate: Frecuencia de muestreo
        
    Returns:
        Array numpy con la señal
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = 0.5 * np.sin(2 * np.pi * frequency * t)
    return signal


def generate_chirp(duration, f0=200, f1=800, sample_rate=22050):
    """
    Genera un chirp (barrido de frecuencias) que simula un canto de rana.
    
    Args:
        duration: Duración en segundos
        f0: Frecuencia inicial en Hz
        f1: Frecuencia final en Hz
        sample_rate: Frecuencia de muestreo
        
    Returns:
        Array numpy con la señal
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Barrido logarítmico de frecuencias
    frequency = f0 * (f1/f0) ** (t/duration)
    phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
    signal = 0.5 * np.sin(phase)
    
    # Aplicar envolvente para simular un pulso
    envelope = np.exp(-3 * t / duration)
    signal = signal * envelope
    
    return signal


def create_test_dataset(output_dir='./test_audio_raw'):
    """
    Crea un dataset de prueba con audios sintéticos.
    
    Args:
        output_dir: Directorio donde crear los archivos de prueba
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Generando dataset de prueba")
    print(f"{'='*70}\n")
    
    test_cases = [
        # (nombre, duración, tipo, descripción)
        ("rana_muy_corta", 2.5, "chirp", "Audio corto - DEBE DESCARTARSE"),
        ("rana_corta", 4.8, "chirp", "Audio casi 5s - DEBE DESCARTARSE"),
        ("rana_exacta", 5.0, "chirp", "Audio exacto 5s - 1 chunk"),
        ("rana_media_01", 7.3, "chirp", "Audio 7.3s - 1 chunk + 2.3s descartados"),
        ("rana_media_02", 9.9, "chirp", "Audio 9.9s - 1 chunk + 4.9s descartados"),
        ("rana_larga_01", 12.5, "chirp", "Audio 12.5s - 2 chunks + 2.5s descartados"),
        ("rana_larga_02", 15.2, "chirp", "Audio 15.2s - 3 chunks + 0.2s descartados"),
        ("rana_larga_03", 18.7, "chirp", "Audio 18.7s - 3 chunks + 3.7s descartados"),
        ("rana_muy_larga", 27.3, "chirp", "Audio 27.3s - 5 chunks + 2.3s descartados"),
        ("ruido_corto", 3.0, "noise", "Ruido corto - DEBE DESCARTARSE"),
        ("tono_puro", 10.0, "sine", "Tono puro 10s - 2 chunks"),
    ]
    
    sample_rate = 22050
    
    for i, (name, duration, audio_type, description) in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Generando: {name}.wav")
        print(f"    Duración: {duration}s - {description}")
        
        # Generar señal según el tipo
        if audio_type == "chirp":
            # Variamos la frecuencia inicial para cada audio
            f0 = 200 + (i * 50) % 400
            f1 = 600 + (i * 100) % 800
            signal = generate_chirp(duration, f0, f1, sample_rate)
        elif audio_type == "sine":
            frequency = 440 + (i * 55)  # Diferentes tonos
            signal = generate_sine_wave(frequency, duration, sample_rate)
        elif audio_type == "noise":
            # Ruido blanco
            signal = np.random.normal(0, 0.1, int(sample_rate * duration))
        
        # Guardar archivo
        filepath = os.path.join(output_dir, f"{name}.wav")
        sf.write(filepath, signal, sample_rate)
    
    print(f"\n{'='*70}")
    print(f"✓ Dataset de prueba creado en: {output_dir}")
    print(f"  Total de archivos: {len(test_cases)}")
    print(f"{'='*70}\n")
    
    print("Ahora puedes probar el preprocesador con:")
    print(f"python audio_preprocessor.py -i {output_dir} -o ./test_audio_procesado\n")


def print_expected_results():
    """Imprime los resultados esperados del preprocesamiento."""
    print(f"\n{'='*70}")
    print("RESULTADOS ESPERADOS DEL PREPROCESAMIENTO")
    print(f"{'='*70}\n")
    
    expected = [
        ("rana_muy_corta.wav", 2.5, 0, "DESCARTADO"),
        ("rana_corta.wav", 4.8, 0, "DESCARTADO"),
        ("rana_exacta.wav", 5.0, 1, "1 chunk"),
        ("rana_media_01.wav", 7.3, 1, "1 chunk (2.3s descartados)"),
        ("rana_media_02.wav", 9.9, 1, "1 chunk (4.9s descartados)"),
        ("rana_larga_01.wav", 12.5, 2, "2 chunks (2.5s descartados)"),
        ("rana_larga_02.wav", 15.2, 3, "3 chunks (0.2s descartados)"),
        ("rana_larga_03.wav", 18.7, 3, "3 chunks (3.7s descartados)"),
        ("rana_muy_larga.wav", 27.3, 5, "5 chunks (2.3s descartados)"),
        ("ruido_corto.wav", 3.0, 0, "DESCARTADO"),
        ("tono_puro.wav", 10.0, 2, "2 chunks"),
    ]
    
    total_chunks = 0
    total_descartados = 0
    
    for name, duration, chunks, resultado in expected:
        status = "⊗" if chunks == 0 else "✓"
        print(f"{status} {name:25} ({duration:4.1f}s) → {resultado}")
        total_chunks += chunks
        if chunks == 0:
            total_descartados += 1
    
    print(f"\n{'-'*70}")
    print(f"Total chunks esperados: {total_chunks}")
    print(f"Total archivos descartados: {total_descartados}")
    print(f"Total archivos procesados: {len(expected) - total_descartados}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Genera un dataset de prueba para validar el preprocesador')
    parser.add_argument('-o', '--output', type=str, default='./test_audio_raw',
                       help='Directorio donde crear los archivos de prueba')
    
    args = parser.parse_args()
    
    # Generar dataset
    create_test_dataset(args.output)
    
    # Mostrar resultados esperados
    print_expected_results()
    
    print("📋 Pasos siguientes:")
    print("1. Ejecuta: python audio_preprocessor.py -i ./test_audio_raw -o ./test_audio_procesado")
    print("2. Verifica que los resultados coincidan con lo esperado")
    print("3. Si todo funciona correctamente, úsalo con tu dataset real de ranas\n")
