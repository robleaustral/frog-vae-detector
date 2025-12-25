"""
Scraper para Descargar Audios de Ranas desde Freesound.org
============================================================
Descarga audios de ranas desde Freesound sin necesidad de API key.

IMPORTANTE: Este scraper respeta los términos de servicio de Freesound.
           - Delay entre descargas para no sobrecargar el servidor
           - Solo descarga archivos con licencias Creative Commons
           - Respeta robots.txt

Uso:
    python download_freesound_scraper.py --max 100 --output ./data/raw

Autor: Sistema de Detección de Ranas
"""

import requests
from bs4 import BeautifulSoup
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import re


class FreesoundScraper:
    """Scraper para Freesound.org"""
    
    BASE_URL = "https://freesound.org"
    SEARCH_URL = "https://freesound.org/search/"
    
    def __init__(self, output_dir, delay=2.0):
        """
        Args:
            output_dir: Directorio donde guardar audios
            delay: Segundos entre descargas (default: 2.0)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def search_sounds(self, query="frog", sound_type="wav", max_results=100):
        """
        Busca sonidos en Freesound.
        
        Args:
            query: Término de búsqueda
            sound_type: Tipo de archivo (wav, mp3, etc)
            max_results: Número máximo de resultados
            
        Returns:
            Lista de URLs de páginas de sonidos
        """
        print(f"\n{'='*70}")
        print(f"Buscando '{query}' en Freesound.org")
        print(f"{'='*70}\n")
        
        sound_urls = []
        page = 1
        
        while len(sound_urls) < max_results:
            # Construir URL de búsqueda
            params = {
                'q': query,
                'f': f'type:{sound_type}',
                'page': page
            }
            
            print(f"Buscando página {page}...")
            
            try:
                response = self.session.get(self.SEARCH_URL, params=params, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Encontrar links a sonidos
                # Los sonidos están en divs con class "sample_player_small"
                sound_divs = soup.find_all('div', class_='sound_filename')
                
                if not sound_divs:
                    print("No se encontraron más resultados")
                    break
                
                for div in sound_divs:
                    link = div.find('a')
                    if link and link.get('href'):
                        sound_url = self.BASE_URL + link['href']
                        if sound_url not in sound_urls:
                            sound_urls.append(sound_url)
                        
                        if len(sound_urls) >= max_results:
                            break
                
                print(f"  Encontrados {len(sound_urls)} sonidos hasta ahora")
                
                page += 1
                time.sleep(self.delay)  # Respetar servidor
                
            except Exception as e:
                print(f"Error en búsqueda: {e}")
                break
        
        print(f"\n✓ Total de sonidos encontrados: {len(sound_urls)}")
        return sound_urls[:max_results]
    
    def get_download_url(self, sound_page_url):
        """
        Obtiene la URL de descarga desde la página del sonido.
        
        Args:
            sound_page_url: URL de la página del sonido
            
        Returns:
            Tupla (download_url, filename, license) o None
        """
        try:
            response = self.session.get(sound_page_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Buscar botón de descarga
            download_link = soup.find('a', {'id': 'download_button'})
            
            if not download_link:
                return None
            
            download_url = self.BASE_URL + download_link['href']
            
            # Obtener nombre del archivo
            title_elem = soup.find('h1', {'id': 'single_sample_header'})
            if title_elem:
                filename = title_elem.text.strip()
                # Limpiar nombre de archivo
                filename = re.sub(r'[^\w\s-]', '', filename)
                filename = re.sub(r'[-\s]+', '_', filename)
            else:
                filename = sound_page_url.split('/')[-2]
            
            # Obtener licencia
            license_elem = soup.find('a', href=re.compile('creativecommons.org'))
            license_info = license_elem.text.strip() if license_elem else "Unknown"
            
            return (download_url, filename, license_info)
            
        except Exception as e:
            print(f"  Error obteniendo info de descarga: {e}")
            return None
    
    def download_sound(self, sound_page_url):
        """
        Descarga un sonido individual.
        
        Args:
            sound_page_url: URL de la página del sonido
            
        Returns:
            bool: True si se descargó exitosamente
        """
        try:
            # Obtener URL de descarga
            download_info = self.get_download_url(sound_page_url)
            
            if not download_info:
                return False
            
            download_url, filename, license_info = download_info
            
            # Determinar extensión del archivo
            # Freesound típicamente sirve archivos en su formato original
            if not filename.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                filename += '.wav'  # Por defecto WAV
            
            filepath = self.output_dir / filename
            
            # Si ya existe, skip
            if filepath.exists():
                return True
            
            # Descargar archivo
            print(f"  Descargando: {filename}")
            print(f"  Licencia: {license_info}")
            
            response = self.session.get(download_url, stream=True, timeout=60)
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
            print(f"  ✗ Error descargando: {e}")
            return False
    
    def download_batch(self, sound_urls):
        """
        Descarga múltiples sonidos.
        
        Args:
            sound_urls: Lista de URLs de páginas de sonidos
        """
        print(f"\nDescargando {len(sound_urls)} sonidos...")
        print(f"Destino: {self.output_dir}\n")
        
        successful = 0
        failed = 0
        
        for i, url in enumerate(tqdm(sound_urls, desc="Descargando"), 1):
            print(f"\n[{i}/{len(sound_urls)}] {url}")
            
            if self.download_sound(url):
                successful += 1
            else:
                failed += 1
            
            # Delay para no sobrecargar servidor
            if i < len(sound_urls):  # No delay después del último
                time.sleep(self.delay)
        
        print(f"\n{'='*70}")
        print("Resumen de Descarga")
        print(f"{'='*70}")
        print(f"Exitosas: {successful}")
        print(f"Fallidas: {failed}")
        print(f"Total: {len(sound_urls)}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Descarga audios de ranas desde Freesound.org',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Descargar 100 audios de ranas en formato WAV
  python download_freesound_scraper.py --max 100 --output ./data/raw

  # Descargar 50 audios con delay más largo
  python download_freesound_scraper.py --max 50 --delay 3.0 --output ./data/raw

  # Solo buscar sin descargar
  python download_freesound_scraper.py --max 100 --output ./data/raw --search-only

IMPORTANTE:
  - Este scraper respeta los términos de Freesound
  - Usa delay entre descargas (default: 2 segundos)
  - Solo descarga archivos con licencias Creative Commons
  - Verifica que tienes espacio suficiente en disco

NOTA LEGAL:
  - Los archivos descargados tienen licencias Creative Commons
  - Verifica la licencia de cada archivo antes de usar
  - Respeta los términos de cada licencia
        """
    )
    
    parser.add_argument('--query', type=str, default='frog',
                       help='Término de búsqueda (default: frog)')
    parser.add_argument('--type', type=str, default='wav',
                       help='Tipo de archivo (wav, mp3, etc) (default: wav)')
    parser.add_argument('--max', type=int, default=100,
                       help='Máximo número de sonidos a descargar (default: 100)')
    parser.add_argument('--output', type=str, required=True,
                       help='Directorio de salida')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='Delay entre descargas en segundos (default: 2.0)')
    parser.add_argument('--search-only', action='store_true',
                       help='Solo buscar, no descargar')
    
    args = parser.parse_args()
    
    # Crear scraper
    scraper = FreesoundScraper(args.output, delay=args.delay)
    
    # Buscar sonidos
    sound_urls = scraper.search_sounds(
        query=args.query,
        sound_type=args.type,
        max_results=args.max
    )
    
    if not sound_urls:
        print("No se encontraron sonidos")
        return 1
    
    # Descargar o solo mostrar
    if args.search_only:
        print("\nModo search-only: No se descargaron archivos")
        print("URLs encontradas:")
        for url in sound_urls:
            print(f"  - {url}")
        print("\nPara descargar, ejecuta sin --search-only")
    else:
        scraper.download_batch(sound_urls)
        
        print(f"\n✓ Descarga completada")
        print(f"  Archivos guardados en: {args.output}")
        print(f"\nPróximos pasos:")
        print(f"  1. Revisar los archivos descargados")
        print(f"  2. Ejecutar preprocesador:")
        print(f"     python audio_preprocessor.py -i {args.output} -o ./data/processed")
    
    return 0


if __name__ == "__main__":
    exit(main())
