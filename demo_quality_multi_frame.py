#!/usr/bin/env python3
"""
Demo per scansione multi-frame con focus sulla QUALITA' e VISUALIZZAZIONI
Salva tutte le immagini di debug per analisi visiva
"""

import sys
import subprocess
from pathlib import Path
import time

def run_quality_demo():
    """Esegue demo con multi-frame per massima qualità e visualizzazioni complete."""
    
    print("="*70)
    print("DEMO QUALITA' MULTI-FRAME CON VISUALIZZAZIONI COMPLETE")
    print("="*70)
    print("Focus: Qualità della superficie, non numero di punti")
    print("Salvataggio: Tutte le visualizzazioni di debug")
    print("")
    
    # Directory output
    output_dir = "demo_quality_multiframe"
    
    # Comando ottimizzato per qualità con multi-frame e visualizzazioni
    cmd = [
        ".venv/Scripts/python.exe", 
        "unlook/examples/scanning/process_offline.py",
        "--input", "unlook/examples/scanning/captured_data/test1_2k/20250603_201954",
        "--surface-reconstruction",
        "--disparity-fusion",        # USA IL NUOVO METODO DI FUSION
        "--all-optimizations",       # Tutte le ottimizzazioni
        "--use-cgal",               # Triangolazione professionale
        "--save-visualizations",     # SALVA TUTTE LE VISUALIZZAZIONI
        "--generate-mesh",          # Genera anche mesh
        "--mesh-method", "poisson",
        "--output", output_dir
    ]
    
    print("COMANDO OTTIMIZZATO:")
    print(" ".join(cmd))
    print("")
    
    print("Caratteristiche:")
    print("- Multi-frame disparity fusion (nuovo metodo)")
    print("- Tutte le ottimizzazioni attive")
    print("- Salvataggio visualizzazioni complete")
    print("- CGAL per triangolazione di qualità")
    print("- Mesh generation con Poisson")
    print("")
    
    start_time = time.time()
    
    try:
        # Crea directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Esegui comando
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print("")
            print("="*70)
            print("SUCCESSO!")
            print("="*70)
            print(f"Tempo: {duration:.1f} secondi")
            
            # Estrai metriche dall'output
            output = result.stdout
            for line in output.split('\n'):
                if "Points Generated:" in line or "Quality Score:" in line:
                    print(line.strip())
            
            # Lista file generati
            print("")
            print("FILE GENERATI:")
            print(f"- Point cloud: {output_dir}/surface_reconstruction.ply")
            print(f"- Mesh: {output_dir}/surface_mesh.ply") 
            print("")
            print("VISUALIZZAZIONI SALVATE:")
            
            viz_dir = Path(output_dir) / "debug_visualizations"
            if viz_dir.exists():
                viz_files = list(viz_dir.glob("*"))
                print(f"Trovate {len(viz_files)} visualizzazioni in {viz_dir}:")
                
                # Categorizza i file
                disparity_files = [f for f in viz_files if "disparity" in f.name.lower()]
                depth_files = [f for f in viz_files if "depth" in f.name.lower()]
                confidence_files = [f for f in viz_files if "confidence" in f.name.lower()]
                fusion_files = [f for f in viz_files if "fusion" in f.name.lower()]
                other_files = [f for f in viz_files if f not in disparity_files + depth_files + confidence_files + fusion_files]
                
                if disparity_files:
                    print("\nDISPARITY MAPS:")
                    for f in sorted(disparity_files):
                        print(f"  - {f.name}")
                
                if depth_files:
                    print("\nDEPTH MAPS:")
                    for f in sorted(depth_files):
                        print(f"  - {f.name}")
                        
                if confidence_files:
                    print("\nCONFIDENCE MAPS:")
                    for f in sorted(confidence_files):
                        print(f"  - {f.name}")
                        
                if fusion_files:
                    print("\nFUSION ANALYSIS:")
                    for f in sorted(fusion_files):
                        print(f"  - {f.name}")
                        
                if other_files:
                    print("\nALTRE VISUALIZZAZIONI:")
                    for f in sorted(other_files):
                        print(f"  - {f.name}")
            else:
                print("ATTENZIONE: Directory visualizzazioni non trovata")
            
            print("")
            print("="*70)
            print("DEMO COMPLETATA CON SUCCESSO!")
            print("Controlla le visualizzazioni per analisi dettagliata")
            print("="*70)
            
        else:
            print("ERRORE NELL'ESECUZIONE!")
            if "UnicodeEncodeError" in result.stderr:
                print("Problema con encoding caratteri (emoji)")
            print("STDERR:", result.stderr[-500:])  # Ultimi 500 caratteri
            
    except Exception as e:
        print(f"ERRORE: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = run_quality_demo()
    sys.exit(0 if success else 1)