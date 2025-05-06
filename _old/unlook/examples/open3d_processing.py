"""
Esempio di processamento avanzato di nuvole di punti con Open3D.

Questo script mostra come utilizzare Open3D con UnLook per
migliorare, filtrare e generare mesh da nuvole di punti 3D.
"""

import os
import time
import logging
import argparse
import numpy as np
import cv2

from unlook.client import (
    UnlookClient, UnlookClientEvent,
    ScanProcessor, PatternDirection,
    StereoCalibrationData, StereoProcessor
)

# Import condizionale di Open3D
try:
    import open3d as o3d
    from unlook.client.open3d_utils import Open3DWrapper

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Variabili globali
client = None
o3d_wrapper = None
output_dir = "output"


# Funzione per verificare la disponibilità di Open3D
def check_open3d():
    if not OPEN3D_AVAILABLE:
        print("Open3D non è disponibile. Installalo con: pip install open3d")
        return False

    try:
        global o3d_wrapper
        o3d_wrapper = Open3DWrapper()
        return True
    except Exception as e:
        print(f"Errore nell'inizializzazione di Open3D: {e}")
        return False


# Funzione per trovare e connettersi a uno scanner
def connect_to_scanner():
    global client

    # Crea il client
    client = UnlookClient(client_name="UnlookOpen3DDemo")

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


# Funzione per acquisire una scansione
def acquire_scan(mode="stereo"):
    """
    Acquisisce una scansione 3D.

    Args:
        mode: Modalità di scansione ('stereo', 'structured_light')

    Returns:
        Nuvola di punti Open3D o None in caso di errore
    """
    if mode == "stereo":
        # Carica calibrazione stereo
        stereo_calib_file = "calibration/stereo_calibration.json"
        if not os.path.exists(stereo_calib_file):
            print(f"File di calibrazione stereo non trovato: {stereo_calib_file}")
            print("Eseguire prima stereo_calibration.py")
            return None

        stereo_calib = StereoCalibrationData.load(stereo_calib_file)
        if not stereo_calib or not stereo_calib.is_valid():
            print("Calibrazione stereo non valida.")
            return None

        # Crea processore stereo
        stereo_processor = StereoProcessor(stereo_calib)

        # Ottieni la coppia stereo
        left_camera_id, right_camera_id = client.camera.get_stereo_pair()
        if left_camera_id is None or right_camera_id is None:
            print("Impossibile trovare una coppia stereo valida.")
            return None

        print("Acquisizione immagini stereo...")
        left_image, right_image = client.camera.capture_stereo_pair()

        if left_image is None or right_image is None:
            print("Errore durante l'acquisizione delle immagini stereo.")
            return None

        print("Elaborazione immagini stereo...")
        points, colors, disparity = stereo_processor.compute_stereo_scan(left_image, right_image)

        if points is None:
            print("Errore durante l'elaborazione stereo.")
            return None

        print(f"Elaborazione completata: {len(points)} punti generati.")

        # Converti in nuvola di punti Open3D
        pcd = o3d_wrapper.points_to_pointcloud(points, colors)

    elif mode == "structured_light":
        # Crea processore di scansione
        scan_processor = ScanProcessor(client)

        # Ottieni la prima telecamera
        cameras = client.camera.get_cameras()
        if not cameras:
            print("Nessuna telecamera disponibile.")
            return None

        camera_id = cameras[0]["id"]

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

        # Converti in nuvola di punti Open3D
        pcd = o3d_wrapper.process_result_to_pointcloud(result)

    else:
        print(f"Modalità non supportata: {mode}")
        return None

    return pcd


# Funzione per processare una nuvola di punti
def process_pointcloud(pcd):
    """
    Processa una nuvola di punti con Open3D.

    Args:
        pcd: Nuvola di punti Open3D

    Returns:
        Nuvola di punti processata
    """
    if pcd is None or len(pcd.points) == 0:
        print("Nuvola di punti vuota o non valida.")
        return None

    # 1. Stima le normali se non presenti
    if len(pcd.normals) == 0:
        print("Calcolo delle normali...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

    # 2. Rimuovi i valori anomali
    print("Rimozione dei valori anomali statistici...")
    pcd_filtered = o3d_wrapper.filter_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0)

    # 3. Voxel downsampling per ridurre la complessità
    print("Esecuzione voxel downsampling...")
    pcd_downsampled = o3d_wrapper.voxel_downsample(pcd_filtered, voxel_size=0.005)

    # 4. Rimozione di valori anomali basati sul raggio
    print("Rimozione dei valori anomali basati sul raggio...")
    pcd_final = o3d_wrapper.filter_radius_outliers(pcd_downsampled, nb_points=16, radius=0.05)

    return pcd_final


# Funzione per creare una mesh
def create_mesh(pcd, method="poisson"):
    """
    Crea una mesh triangolare da una nuvola di punti.

    Args:
        pcd: Nuvola di punti Open3D
        method: Metodo di ricostruzione ('poisson' o 'ball_pivoting')

    Returns:
        Mesh triangolare
    """
    print(f"Creazione mesh con metodo {method}...")

    if method == "poisson":
        mesh = o3d_wrapper.create_mesh_from_pointcloud(pcd, method="poisson", depth=8)
    elif method == "ball_pivoting":
        mesh = o3d_wrapper.create_mesh_from_pointcloud(pcd, method="ball_pivoting")
    else:
        print(f"Metodo non supportato: {method}")
        return None

    return mesh


# Funzione principale
def main():
    global output_dir

    # Parsing degli argomenti
    parser = argparse.ArgumentParser(description="Processamento di nuvole di punti con Open3D")
    parser.add_argument("-o", "--output", default="output", help="Directory di output")
    parser.add_argument("-m", "--mode", choices=["stereo", "structured_light"], default="stereo",
                        help="Modalità di acquisizione")
    parser.add_argument("--mesh-method", choices=["poisson", "ball_pivoting"], default="poisson",
                        help="Metodo di ricostruzione mesh")
    parser.add_argument("--visualize", action="store_true", help="Visualizza i risultati")
    args = parser.parse_args()

    # Verifica la disponibilità di Open3D
    if not check_open3d():
        return 1

    # Impostazioni
    output_dir = args.output
    ensure_output_dir()

    try:
        # Connessione allo scanner
        if not connect_to_scanner():
            return 1

        # Acquisizione scansione
        pcd = acquire_scan(args.mode)
        if pcd is None:
            client.disconnect()
            client.stop_discovery()
            return 1

        # Timestamp per i file
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Salva nuvola di punti originale
        raw_pcd_file = os.path.join(output_dir, f"raw_pointcloud_{timestamp}.ply")
        o3d_wrapper.save_pointcloud(pcd, raw_pcd_file)
        print(f"Nuvola di punti originale salvata in: {raw_pcd_file}")

        # Visualizza la nuvola di punti originale
        if args.visualize:
            print("Visualizzazione nuvola di punti originale...")
            o3d_wrapper.visualize_pointcloud(pcd, "Nuvola di punti originale")

        # Processa la nuvola di punti
        print("\n=== Processamento della nuvola di punti ===")
        processed_pcd = process_pointcloud(pcd)

        if processed_pcd is None:
            print("Errore durante il processamento della nuvola di punti.")
            client.disconnect()
            client.stop_discovery()
            return 1

        # Salva nuvola di punti processata
        processed_pcd_file = os.path.join(output_dir, f"processed_pointcloud_{timestamp}.ply")
        o3d_wrapper.save_pointcloud(processed_pcd, processed_pcd_file)
        print(f"Nuvola di punti processata salvata in: {processed_pcd_file}")

        # Visualizza la nuvola di punti processata
        if args.visualize:
            print("Visualizzazione nuvola di punti processata...")
            o3d_wrapper.visualize_pointcloud(processed_pcd, "Nuvola di punti processata")

        # Crea mesh
        print("\n=== Creazione mesh 3D ===")
        mesh = create_mesh(processed_pcd, args.mesh_method)

        if mesh is None:
            print("Errore durante la creazione della mesh.")
            client.disconnect()
            client.stop_discovery()
            return 1

        # Salva mesh
        mesh_file = os.path.join(output_dir, f"mesh_{args.mesh_method}_{timestamp}.obj")
        o3d_wrapper.save_mesh(mesh, mesh_file)
        print(f"Mesh salvata in: {mesh_file}")

        # Visualizza la mesh
        if args.visualize:
            print("Visualizzazione mesh...")
            o3d.visualization.draw_geometries([mesh], window_name="Mesh 3D")

        # Disconnetti
        client.disconnect()
        client.stop_discovery()

        print("Processamento completato con successo!")
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