"""
Script para Descargar Audio de Ranas desde Xeno-canto
======================================================
Descarga automáticamente grabaciones de ranas chilenas desde Xeno-canto.org

Uso:
    python download_xenocanto.py --species "Telmatobius" --output ./data/raw

Autor: Sistema de Detección de Ranas
"""

import requests
import json
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import time


class XenoCantoDownloader:
    """Descargador de audio desde Xeno-canto API."""
    
    BASE_URL = "https://xeno-canto.org/api/2/recordings"
    
    def __init__(self, output_dir):
        """
        Args:
            output_dir: Directorio donde guardar las grabaciones
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def search_recordings(self, query, max_results=100):
        """
        Busca grabaciones en Xeno-canto.
        
        Args:
            query: Query de búsqueda (ej: "Telmatobius cnt:Chile")
            max_results: Máximo número de resultados
            
        Returns:
            Lista de metadatos de grabaciones
        """
        print(f"\n{'='*70}")
        print(f"Buscando en Xeno-canto: {query}")
        print(f"{'='*70}\n")
        
        params = {'query': query}
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            recordings = data.get('recordings', [])
            total_found = len(recordings)
            
            print(f"✓ Encontradas {total_found} grabaciones")
            
            if max_results and total_found > max_results:
                recordings = recordings[:max_results]
                print(f"  Limitando a {max_results} grabaciones")
            
            return recordings
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error en la búsqueda: {e}")
            return []
    
    def download_recording(self, recording_metadata):
        """
        Descarga una grabación individual.
        
        Args:
            recording_metadata: Metadata de la grabación de Xeno-canto
            
        Returns:
            bool: True si se descargó exitosamente
        """
        try:
            # Obtener URL del archivo
            file_url = recording_metadata.get('file')
            if not file_url:
                return False
            
            # Crear nombre de archivo
            recording_id = recording_metadata.get('id', 'unknown')
            species = recording_metadata.get('gen', '') + '_' + recording_metadata.get('sp', '')
            species = species.replace(' ', '_')
            filename = f"XC{recording_id}_{species}.mp3"
            filepath = self.output_dir / filename
            
            # Si ya existe, skip
            if filepath.exists():
                return True
            
            # Descargar
            response = requests.get(file_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error descargando {recording_id}: {e}")
            return False
    
    def download_batch(self, recordings, delay=1.0):
        """
        Descarga múltiples grabaciones.
        
        Args:
            recordings: Lista de metadatos de grabaciones
            delay: Delay entre descargas (segundos) para no sobrecargar el servidor
        """
        print(f"\nDescargando {len(recordings)} grabaciones...")
        print(f"Destino: {self.output_dir}\n")
        
        successful = 0
        failed = 0
        
        for recording in tqdm(recordings, desc="Descargando"):
            if self.download_recording(recording):
                successful += 1
            else:
                failed += 1
            
            # Delay para no sobrecargar el servidor
            time.sleep(delay)
        
        print(f"\n{'='*70}")
        print("Resumen de Descarga")
        print(f"{'='*70}")
        print(f"Exitosas: {successful}")
        print(f"Fallidas: {failed}")
        print(f"Total: {len(recordings)}")
        print(f"{'='*70}\n")
    
    def get_recording_info(self, recordings):
        """
        Muestra información sobre las grabaciones encontradas.
        
        Args:
            recordings: Lista de metadatos
        """
        if not recordings:
            print("No hay grabaciones para mostrar")
            return
        
        print(f"\n{'='*70}")
        print("Información de las Grabaciones")
        print(f"{'='*70}\n")
        
        # Especies únicas
        species = set()
        countries = set()
        qualities = {}
        
        for rec in recordings:
            sp = f"{rec.get('gen', '')} {rec.get('sp', '')}"
            species.add(sp)
            countries.add(rec.get('cnt', 'Unknown'))
            
            q = rec.get('q', 'Unknown')
            qualities[q] = qualities.get(q, 0) + 1
        
        print(f"Especies encontradas ({len(species)}):")
        for sp in sorted(species):
            count = sum(1 for r in recordings if f"{r.get('gen', '')} {r.get('sp', '')}" == sp)
            print(f"  • {sp}: {count} grabaciones")
        
        print(f"\nPaíses:")
        for country in sorted(countries):
            count = sum(1 for r in recordings if r.get('cnt') == country)
            print(f"  • {country}: {count} grabaciones")
        
        print(f"\nCalidad de las grabaciones:")
        for quality, count in sorted(qualities.items(), reverse=True):
            print(f"  • Calidad {quality}: {count} grabaciones")
        
        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Descarga grabaciones de ranas desde Xeno-canto',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Opción 1: Usar query personalizado (RECOMENDADO)
  python download_xenocanto.py --query "grp:frogs" --max 300 --output ./data/raw
  python download_xenocanto.py --query "grp:frogs area:south-america" --max 300 --output ./data/raw
  python download_xenocanto.py --query "gen:Telmatobius" --max 300 --output ./data/raw

  # Opción 2: Usar parámetros individuales
  python download_xenocanto.py --species "Telmatobius" --output ./data/raw
  python download_xenocanto.py --species "Alsodes" --country "Chile" --quality "A" --output ./data/raw
  python download_xenocanto.py --group "frogs" --max 500 --output ./data/raw

Filtros disponibles en Xeno-canto:
  - grp:frogs               → Todos los anfibios
  - grp:frogs area:south-america → Anfibios de Sudamérica
  - gen:Telmatobius         → Género específico
  - cnt:Chile               → País
  - q>C                     → Calidad C o mejor (A, B, C, D, E)
  - type:call               → Tipo de sonido

Para ver todos los filtros: https://xeno-canto.org/explore/taxonomy?grp=frogs
        """
    )
    
    # Grupo principal de argumentos
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--query', type=str, default=None,
                       help='Query personalizado de Xeno-canto (ej: "grp:frogs", "gen:Telmatobius area:south-america")')
    
    # Argumentos individuales (alternativa al query)
    parser.add_argument('--species', type=str, default=None,
                       help='Nombre del género o especie (ej: "Telmatobius", "Alsodes")')
    parser.add_argument('--country', type=str, default=None,
                       help='País (ej: "Chile", "Peru", "Argentina")')
    parser.add_argument('--quality', type=str, default=None,
                       help='Calidad mínima (A, B, C, D, E)')
    parser.add_argument('--type', type=str, default=None,
                       help='Tipo de grabación (ej: "call", "song")')
    parser.add_argument('--group', type=str, default=None,
                       help='Grupo taxonómico (ej: "frogs" para todos los anfibios)')
    parser.add_argument('--area', type=str, default=None,
                       help='Área geográfica (ej: "south-america", "africa")')
    
    # Argumentos generales
    parser.add_argument('--max', type=int, default=100,
                       help='Máximo número de grabaciones a descargar (default: 100)')
    parser.add_argument('--output', type=str, required=True,
                       help='Directorio de salida')
    parser.add_argument('--info-only', action='store_true',
                       help='Solo mostrar información, no descargar')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay entre descargas en segundos (default: 1.0)')
    
    args = parser.parse_args()
    
    # Construir query
    if args.query:
        # Usar query personalizado directamente
        query = args.query
    else:
        # Construir query desde parámetros individuales
        query_parts = []
        
        if args.group:
            query_parts.append(f"grp:{args.group}")
        
        if args.species:
            query_parts.append(f"gen:{args.species}")
        
        if args.country:
            query_parts.append(f"cnt:{args.country}")
        
        if args.area:
            query_parts.append(f"area:{args.area}")
        
        if args.quality:
            query_parts.append(f"q>={args.quality}")
        
        if args.type:
            query_parts.append(f"type:{args.type}")
        
        query = " ".join(query_parts)
        
        if not query:
            print("Error: Debes especificar --query o al menos un criterio de búsqueda")
            print("Ejemplos:")
            print('  python download_xenocanto.py --query "grp:frogs" --output ./data/raw')
            print('  python download_xenocanto.py --species "Telmatobius" --output ./data/raw')
            print('  python download_xenocanto.py --group "frogs" --area "south-america" --output ./data/raw')
            return 1
    
    # Crear downloader
    downloader = XenoCantoDownloader(args.output)
    
    # Buscar grabaciones
    recordings = downloader.search_recordings(query, max_results=args.max)
    
    if not recordings:
        print("No se encontraron grabaciones con los criterios especificados")
        return 1
    
    # Mostrar información
    downloader.get_recording_info(recordings)
    
    # Descargar o solo info
    if args.info_only:
        print("Modo info-only: No se descargaron archivos")
        print("Para descargar, ejecuta sin --info-only")
    else:
        downloader.download_batch(recordings, delay=args.delay)
        
        print(f"\n✓ Descarga completada")
        print(f"  Archivos guardados en: {args.output}")
        print(f"\nPróximos pasos:")
        print(f"  1. Revisar los archivos descargados")
        print(f"  2. Ejecutar preprocesador:")
        print(f"     python audio_preprocessor.py -i {args.output} -o ./data/processed")
    
    return 0


if __name__ == "__main__":
    exit(main())
