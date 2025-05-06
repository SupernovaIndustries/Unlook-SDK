"""
Esempio di elaborazione avanzata di mesh da scansioni 3D.

Questo script mostra come utilizzare le varie librerie di processamento mesh
con i dati acquisiti dallo scanner UnLook.
"""

import os
import time
import logging
import argparse
from typing import Optional

from unlook.client import (
    UnlookClient,
    ScanProcessor, PatternDirection,
    StereoCalibrationData, StereoProcessor
)

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Variabili globali
client = None
output_dir = "output"


# Funzione per verificare la disponibilità delle librerie
def check_libraries(library):
    if library == "pymeshlab":
        try:
            from unlook.client.mesh_processing import PyMeshLabProcessor
            processor = PyMeshLabProcessor()
            return processor.available
        except Exception:
            return False
    elif library == "trimesh":
        try:
            from unlook.client.mesh_processing import TrimeshProcessor
            processor = TrimeshProcessor()
            return processor.available
        except Exception:
            return False
    elif library == "open3d":
        try:
            import open3d as o3d
            from unlook.client.open3d_utils import Open3DWrapper
            wrapper = Open3DWrapper()
            return True
        except Exception:
            return False
    return False


# Funzione per connettersi allo scanner
def connect_to_scanner():
    global client

    # Crea il client
    client = UnlookClient(client_name="UnlookMeshProcessingDemo")

    # Avvia la discovery
    client.start_discovery()

    print("Ricerca scanner in corso (5 secondi)...")
    time.sleep(5)

    # Ottieni la lista degli scanner
    scanners = client.get_discovered_scanners()
    if not scanners:
        print("Nessuno scanner trovato. Uscita.")
        client.stop_discovery()
        return False

    # Connetti al primo scanner
    scanner = scanners[0]
    print(f"Connessione a: {scanner.name}...")

    if not client.connect(scanner):
        print("Impossibile connettersi allo scanner.")
        client.stop_discovery()
        return False

    print(f"Connesso a {scanner.name} ({scanner.uuid})")
    return True


# Funzione per creare la directory di output
def ensure_output_dir():
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory di output: {output_dir}")


# Funzione per acquisire una scansione 3D
def acquire_3d_scan(camera_id):
    """
    Acquisisce una scansione 3D usando la luce strutturata.

    Args:
        camera_id: ID della telecamera

    Returns:
        Risultato della scansione o None in caso di errore
    """
    # Crea processore di scansione
    scan_processor = ScanProcessor(client)

    print("Posiziona l'oggetto da scansionare.")
    input("Premi INVIO per avviare la scansione...")

    # Esegui la scansione
    print("Esecuzione scansione con luce strutturata...")
    success, result = scan_processor.capture_gray_code_scan(
        camera_id=camera_id,
        pattern_width=1280,
        pattern_height=800,
        direction=PatternDirection.BOTH,
        capture_texture=True,
        show_preview=True
    )

    if not success or not result.has_point_cloud():
        print("Errore durante la scansione.")
        return None

    print(f"Scansione completata: {result.num_points} punti generati.")
    return result


# Funzione per elaborare una nuvola di punti con PyMeshLab
def process_with_pymeshlab(result, output_file):
    """
    Elabora una nuvola di punti con PyMeshLab.

    Args:
        result: Risultato della scansione
        output_file: Percorso del file di output

    Returns:
        True se l'elaborazione ha successo, False altrimenti
    """
    try:
        from unlook.client.mesh_processing import PyMeshLabProcessor

        print("\n--- Elaborazione con PyMeshLab ---")
        processor = PyMeshLabProcessor()

        # Elabora la nuvola di punti
        print("Elaborazione della nuvola di punti...")
        meshset = processor.process_result(
            result,
            method="ball_pivoting",
            point_cloud_simplification=True,
            simplification_voxel=0.5,
            mesh_cleaning=True,
            mesh_smoothing=True
        )

        # Applica ulteriori filtri
        print("Applicazione filtri aggiuntivi...")

        # Riduci il numero di facce
        print("Riduzione facce...")
        meshset.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=10000)

        # Ottimizza la mesh
        print("Ottimizzazione mesh...")
        meshset.apply_filter('normalize_face_normals')
        meshset.apply_filter('remove_duplicate_vertices')

        # Calcola le normali per vertice
        print("Calcolo normali per vertice...")
        meshset.apply_filter('compute_normals_for_point_sets')

        # Salva la mesh
        print(f"Salvataggio mesh in {output_file}...")
        processor.save_mesh(meshset, output_file)

        print("Elaborazione con PyMeshLab completata.")
        return True

    except Exception as e:
        print(f"Errore durante l'elaborazione con PyMeshLab: {e}")
        return False


# Funzione per elaborare una nuvola di punti con trimesh
def process_with_trimesh(result, output_file):
    """
    Elabora una nuvola di punti con trimesh.

    Args:
        result: Risultato della scansione
        output_file: Percorso del file di output

    Returns:
        True se l'elaborazione ha successo, False altrimenti
    """
    try:
        from unlook.client.mesh_processing import TrimeshProcessor

        print("\n--- Elaborazione con trimesh ---")
        processor = TrimeshProcessor()

        # Elabora la nuvola di punti
        print("Elaborazione della nuvola di punti...")
        mesh = processor.process_result(
            result,
            method="alpha",
            sampling_factor=1.0,
            cleaning=True,
            smoothing=True
        )

        # Salva la mesh
        print(f"Salvataggio mesh in {output_file}...")
        processor.save_mesh(mesh, output_file)

        print("Elaborazione con trimesh completata.")
        return True

    except Exception as e:
        print(f"Errore durante l'elaborazione con trimesh: {e}")
        return False


# Funzione per elaborare una nuvola di punti con Open3D
def process_with_open3d(result, output_file):
    """
    Elabora una nuvola di punti con Open3D.

    Args:
        result: Risultato della scansione
        output_file: Percorso del file di output

    Returns:
        True se l'elaborazione ha successo, False altrimenti
    """
    try:
        import open3d as o3d
        from unlook.client.open3d_utils import Open3DWrapper

        print("\n--- Elaborazione con Open3D ---")
        wrapper = Open3DWrapper()

        # Converti il risultato in nuvola di punti Open3D
        print("Conversione in nuvola di punti Open3D...")
        pcd = wrapper.process_result_to_pointcloud(result)

        # Filtra i valori anomali
        print("Filtraggio valori anomali...")
        pcd = wrapper.filter_statistical_outliers(pcd)

        # Downsampling
        print("Downsampling...")
        pcd = wrapper.voxel_downsample(pcd, voxel_size=0.01)

        # Ricostruzione della superficie
        print("Ricostruzione della superficie...")
        mesh = wrapper.create_mesh_from_pointcloud(pcd, method="poisson")

        # Salva la mesh
        print(f"Salvataggio mesh in {output_file}...")
        wrapper.save_mesh(mesh, output_file)

        print("Elaborazione con Open3D completata.")
        return True

    except Exception as e:
        print(f"Errore durante l'elaborazione con Open3D: {e}")
        return False


# Funzione principale
def main():
    global output_dir

    # Parsing degli argomenti
    parser = argparse.ArgumentParser(description="Elaborazione avanzata di mesh da scansioni 3D")
    parser.add_argument("-o", "--output", default="output", help="Directory di output")
    parser.add_argument("-l", "--library", choices=["pymeshlab", "trimesh", "open3d", "all"], default="all",
                        help="Libreria da utilizzare per l'elaborazione")
    args = parser.parse_args()

    # Impostazioni
    output_dir = args.output
    ensure_output_dir()

    # Verifica la disponibilità delle librerie
    libraries_to_use = []

    if args.library == "all":
        for lib in ["pymeshlab", "trimesh", "open3d"]:
            if check_libraries(lib):
                libraries_to_use.append(lib)
                print(f"Libreria {lib} disponibile.")
            else:
                print(f"Libreria {lib} non disponibile. Installa con: pip install {lib}")
    else:
        if check_libraries(args.library):
            libraries_to_use.append(args.library)
            print(f"Libreria {args.library} disponibile.")
        else:
            print(f"Libreria {args.library} non disponibile. Installa con: pip install {args.library}")
            return 1

    if not libraries_to_use:
        print("Nessuna libreria di elaborazione mesh disponibile.")
        return 1

    try:
        # Connessione allo scanner
        if not connect_to_scanner():
            return 1

        # Ottieni le telecamere disponibili
        cameras = client.camera.get_cameras()
        if not cameras:
            print("Nessuna telecamera disponibile.")
            client.disconnect()
            client.stop_discovery()
            return 1

        # Usa la prima telecamera
        camera_id = cameras[0]["id"]
        print(f"Utilizzo telecamera: {cameras[0]['name']} ({camera_id})")

        # Acquisisci una scansione 3D
        result = acquire_3d_scan(camera_id)

        if result is None:
            client.disconnect()
            client.stop_discovery()
            return 1

        # Timestamp per i file
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Elabora la scansione con le librerie disponibili
        for library in libraries_to_use:
            output_file = os.path.join(output_dir, f"mesh_{library}_{timestamp}.obj")

            if library == "pymeshlab":
                process_with_pymeshlab(result, output_file)
            elif library == "trimesh":
                process_with_trimesh(result, output_file)
            elif library == "open3d":
                process_with_open3d(result, output_file)

        # Disconnetti
        client.disconnect()
        client.stop_discovery()

        print("Elaborazione completata con successo!")
        return 0

    except KeyboardInterrupt:
        print("\nInterruzione rilevata. Uscita.")
        if client:
            client.disconnect()
            client.stop_discovery()
        return 130
    except Exception as e:
        print(f"Errore non gestito: {e}")
        if client:
            client.disconnect()
            client.stop_discovery()
        return 1


if __name__ == "__main__":
    exit(main())