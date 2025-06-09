#!/usr/bin/env python3
"""
Demo che usa SOLO StereoSGBM (NO StereoBM) per pattern proiettati
StereoSGBM è molto migliore per structured light patterns!
"""

import sys
import subprocess
from pathlib import Path
import time

def run_sgbm_demo():
    """Esegue demo con StereoSGBM per pattern proiettati."""
    
    print("="*70)
    print("DEMO STEREOSGBM PER PATTERN PROIETTATI")
    print("="*70)
    print("StereoBM = per texture naturali (NON adatto)")
    print("StereoSGBM = per pattern proiettati (OTTIMALE)")
    print("")
    
    # Directory output
    output_dir = "demo_sgbm_patterns"
    
    # Comando che forza l'uso di StereoSGBM con advanced-stereo flag
    cmd = [
        ".venv/Scripts/python.exe", 
        "unlook/examples/scanning/process_offline.py",
        "--input", "unlook/examples/scanning/captured_data/test1_2k/20250603_201954",
        "--surface-reconstruction",
        "--advanced-stereo",         # USA STEREOSGBM invece di StereoBM!
        "--disparity-fusion",        # Multi-frame fusion
        "--use-cgal",               # Triangolazione professionale
        "--all-optimizations",      # Attiva tutte le ottimizzazioni
        "--save-visualizations",     # Salva visualizzazioni
        "--generate-mesh",          # Genera mesh
        "--mesh-method", "poisson",
        "--output", output_dir
    ]
    
    print("COMANDO OTTIMIZZATO PER PATTERN PROIETTATI:")
    print(" ".join(cmd))
    print("")
    
    print("Caratteristiche:")
    print("- StereoSGBM: Algoritmo Semi-Global Matching")
    print("- Ottimizzato per structured light patterns")
    print("- Sub-pixel accuracy integrata")
    print("- Multi-frame disparity fusion")
    print("- Confidence filtering avanzato")
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
            
            # Estrai metriche
            output = result.stdout
            for line in output.split('\n'):
                if "Points Generated:" in line:
                    print(line.strip())
                elif "Quality Score:" in line:
                    print(line.strip())
                elif "Using ANTI-BACKGROUND ADVANCED STEREOSGBM" in line:
                    print("CONFERMATO: Usando StereoSGBM (non StereoBM)")
            
            # Lista file generati
            print("")
            print("FILE GENERATI:")
            print(f"- Point cloud: {output_dir}/surface_reconstruction.ply")
            print(f"- Mesh: {output_dir}/surface_mesh.ply") 
            
            # Visualizzazioni
            viz_dir = Path(output_dir) / "debug_visualizations"
            if viz_dir.exists():
                viz_files = list(viz_dir.glob("*"))
                print(f"\nVISUALIZZAZIONI: {len(viz_files)} file salvati")
                
                # Mostra solo alcune visualizzazioni chiave
                key_files = [
                    "fused_disparity.colored.png",
                    "fusion_coverage_analysis.png", 
                    "fusion_consistency_analysis.png",
                    "disparity_analysis.png",
                    "depth_analysis.png",
                    "confidence_analysis.png"
                ]
                
                print("\nVISUALIZZAZIONI PRINCIPALI:")
                for key_file in key_files:
                    full_path = viz_dir / key_file
                    if full_path.exists():
                        print(f"  - {key_file}")
            
            print("")
            print("="*70)
            print("DEMO COMPLETATA!")
            print("StereoSGBM dovrebbe dare MOLTI più punti per pattern proiettati")
            print("="*70)
            
        else:
            print("ERRORE NELL'ESECUZIONE!")
            print("STDERR:", result.stderr[-500:] if result.stderr else "No stderr")
            
    except Exception as e:
        print(f"ERRORE: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = run_sgbm_demo()
    sys.exit(0 if success else 1)