"""
Script para Descargar Ranas usando XenoPy
==========================================
Usa la biblioteca 'xenopy' que funciona correctamente con la API.

Instalación:
    pip install xenopy

Uso:
    python download_xenocanto_xenopy.py --query "frog" --max 300 --output ./data/raw
"""

import argparse
from pathlib import Path
from xenopy import Query

def main():
    parser = argparse.ArgumentParser(description='Descarga ranas desde Xeno-canto')
    parser.add_argument('--query', type=str, required=True, help='Búsqueda (ej: "frog")')
    parser.add_argument('--max', type=int, default=100, help='Máximo de grabaciones')
    parser.add_argument('--output', type=str, required=True, help='Directorio de salida')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"Descargando desde Xeno-canto")
    print("="*70)
    print(f"Query: {args.query}")
    print(f"Máximo: {args.max}")
    print(f"Output: {output_dir}\n")
    
    try:
        # Crear query
        q = Query(name=args.query)
        
        # Obtener metadatos
        print("Obteniendo metadatos...")
        metafiles = q.retrieve_meta(verbose=True)
        print(f"\n✓ Encontradas {len(metafiles)} grabaciones")
        
        # Descargar grabaciones
        print(f"\nDescargando audio...")
        q.retrieve_recordings(
            multiprocess=True,
            nproc=4,
            attempts=3,
            outdir=str(output_dir)
        )
        
        print(f"\n✓ Descarga completada!")
        print(f"Archivos en: {output_dir}")
        print(f"\nPróximos pasos:")
        print(f"  python audio_preprocessor.py -i {output_dir} -o ./data/processed")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

