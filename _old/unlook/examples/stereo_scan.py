"""
Esempio di scansione 3D stereo con UnLook.

Questo script esegue una scansione 3D utilizzando le telecamere stereo
e i pattern di luce strutturata proiettati.
"""

import logging
import time
import os
import cv2
import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional

from unlook.client import (
    UnlookClient, UnlookClientEvent,
    StereoCalibrationData, StereoProcessor,
    ScanProcessor, PatternDirection, PatternType,
    ModelExporter
)

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Variabili globali
client = None
output_dir = "output"
stereo_calib = None
stereo_processor = None
scan_processor = None
exporter = None


# Callback per gli eventi del client
def on_connected(scanner):
    print(f"Connesso a: {scanner.name} ({scanner.uuid})")


def on_disconnected():
    print("Disconnesso dallo scanner")


def on_error(error):
    print(f"Errore: {error}")


# Funzione per creare la directory di output
def ensure_output_dir():
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory di output: {output_dir}")


# Funzione per trovare e connettersi allo scanner
def connect_to_scanner():
    global client

    # Crea il client
    client = UnlookClient(client_name="UnlookStereoScan")

    # Registra i callback
    client.on_event(UnlookClientEvent.CONNECTED, on_connected)
    client.on_event(UnlookClientEvent.DISCONNECTED, on_disconnected)
    client.on_event(UnlookClientEvent.ERROR, on_error)

    # Avvia la discovery
    client.start_discovery()

    print("Ricerca scanner in corso (5 secondi)...")
    time.sleep(5)

    # Ottieni la lista degli scanner trovati
    scanners = client.get_discovered_scanners()

    if not scanners:
        print("Nessuno scanner trovato. Uscita.")
        client.stop_discovery()
        return False

    # Seleziona il primo scanner
    scanner = scanners[0]
    print(f"Connessione a: {scanner.name} ({scanner.uuid})...")

    # Connetti allo scanner
    if not client.connect(scanner):
        print("Impossibile connettersi allo scanner. Uscita.")
        client.stop_discovery()
        return False

    # Ottieni informazioni sullo scanner
    info = client.get_info()
    capabilities = info.get("capabilities", {})

    # Verifica se lo scanner supporta stereo
    is_stereo_capable = capabilities.get("cameras", {}).get("stereo_capable", False)

    if not is_stereo_capable:
        print("Lo scanner non supporta la stereovisione. È necessario avere almeno 2 telecamere.")
        client.disconnect()
        client.stop_discovery()
        return False

    print("Scanner con supporto stereo rilevato!")
    return True


# Funzione per caricare la calibrazione stereo
def load_stereo_calibration(calib_file):
    global stereo_calib, stereo_processor

    print(f"Caricamento calibrazione stereo da {calib_file}...")

    stereo_calib = StereoCalibrationData.load(calib_file)

    if stereo_calib and stereo_calib.is_valid():
        print("Calibrazione stereo caricata con successo.")

        # Crea il processore stereo
        stereo_processor = StereoProcessor(stereo_calib)
        return True
    else:
        print("Errore durante il caricamento della calibrazione stereo.")
        return False


# Funzione per catturare ed elaborare una coppia stereo
def capture_and_process_stereo_pair():
    # Ottieni la coppia stereo
    left_camera_id, right_camera_id = client.camera.get_stereo_pair()

    if left_camera_id is None or right_camera_id is None:
        print("Impossibile trovare una coppia stereo valida.")
        return None, None, None

    print(f"Cattura immagini dalla coppia stereo: {left_camera_id} (sinistra), {right_camera_id} (destra)")

    # Cattura immagini
    left_image, right_image = client.camera.capture_stereo_pair()

    if left_image is None or right_image is None:
        print("Errore durante la cattura delle immagini stereo.")
        return None, None, None

    print("Immagini stereo catturate con successo.")

    # Elabora le immagini stereo
    print("Elaborazione stereo in corso...")

    # Verifica che il processore stereo sia inizializzato
    if stereo_processor is None:
        print("Processore stereo non inizializzato. Carica prima la calibrazione stereo.")
        return None, None, None

    # Calcola la mappa di disparità e la nuvola di punti
    points, colors, disparity = stereo_processor.compute_stereo_scan(left_image, right_image)

    if points is None or disparity is None:
        print("Errore durante l'elaborazione stereo.")
        return None, None, None

    print(f"Elaborazione stereo completata: {len(points)} punti 3D generati.")

    return points, colors, disparity


# Funzione per eseguire una scansione stereo con luce strutturata
def perform_structured_light_scan():
    global scan_processor

    # Ottieni la coppia stereo
    left_camera_id, right_camera_id = client.camera.get_stereo_pair()

    if left_camera_id is None or right_camera_id is None:
        print("Impossibile trovare una coppia stereo valida.")
        return None, None

    # Crea il processore di scansione se non esiste
    if scan_processor is None:
        scan_processor = ScanProcessor(client)

        # Imposta la calibrazione se disponibile
        if stereo_calib and stereo_calib.is_valid():
            # Nota: qui stiamo usando solo la calibrazione della telecamera sinistra
            # In un'implementazione completa, si dovrebbe adattare ScanProcessor per supportare stereo
            left_calib = CalibrationData()
            left_calib.camera_matrix = stereo_calib.left_camera_matrix
            left_calib.dist_coeffs = stereo_calib.left_dist_coeffs
            scan_processor.calibration = left_calib

    print("\n--- Scansione 3D con Luce Strutturata ---")
    print("Posizionare l'oggetto da scansionare.")
    input("Premere INVIO per iniziare la scansione...")

    # Esegui la scansione con Gray code usando la telecamera sinistra
    print("Esecuzione scansione con pattern Gray code...")
    success_left, result_left = scan_processor.capture_gray_code_scan(
        camera_id=left_camera_id,
        pattern_width=1280,
        pattern_height=800,
        direction=PatternDirection.BOTH,
        capture_texture=True,
        show_preview=True
    )

    if not success_left or not result_left.has_point_cloud():
        print("Scansione telecamera sinistra fallita.")
        return None, None

    print(f"Scansione telecamera sinistra completata: {result_left.num_points} punti.")

    # Esegui la scansione con Gray code usando la telecamera destra
    print("Esecuzione scansione con pattern Gray code (telecamera destra)...")
    success_right, result_right = scan_processor.capture_gray_code_scan(
        camera_id=right_camera_id,
        pattern_width=1280,
        pattern_height=800,
        direction=PatternDirection.BOTH,
        capture_texture=True,
        show_preview=True
    )

    if not success_right or not result_right.has_point_cloud():
        print("Scansione telecamera destra fallita.")
        return result_left, None

    print(f"Scansione telecamera destra completata: {result_right.num_points} punti.")

    return result_left, result_right


# Funzione per combinare i risultati di due scansioni
def combine_scan_results(result_left, result_right):
    if result_left is None:
        return None, None

    if result_right is None:
        # Se abbiamo solo il risultato sinistro, restituiscilo
        return result_left.point_cloud, result_left.colors

    # Combina i punti e i colori
    points = np.vstack((result_left.point_cloud, result_right.point_cloud))

    # Combina i colori se disponibili
    colors = None
    if result_left.colors is not None and result_right.colors is not None:
        colors = np.vstack((result_left.colors, result_right.colors))

    print(f"Nuvola di punti combinata: {len(points)} punti totali.")

    return points, colors


# Funzione per esportare la nuvola di punti
def export_point_cloud(points, colors, base_filename):
    global exporter

    if points is None:
        print("Nessuna nuvola di punti da esportare.")
        return False

    if exporter is None:
        exporter = ModelExporter()

    # Crea una struttura simile a ProcessingResult per l'esportazione
    from unlook.client.processing import ProcessingResult
    result = ProcessingResult()
    result.point_cloud = points
    result.colors = colors
    result.num_points = len(points)

    # Esporta in vari formati
    success = False

    # PLY
    ply_file = f"{base_filename}.ply"
    if exporter.export_ply(result, ply_file):
        print(f"Nuvola di punti esportata in: {ply_file}")
        success = True

    # OBJ
    obj_file = f"{base_filename}.obj"
    if exporter.export_obj(result, obj_file):
        print(f"Modello OBJ esportato in: {obj_file}")
        success = True

    return success


# Funzione per visualizzare la mappa di disparità
def display_disparity_map(disparity_map):
    if disparity_map is None:
        print("Nessuna mappa di disparità da visualizzare.")
        return

    # Normalizza per la visualizzazione
    min_disp = np.min(disparity_map[disparity_map > 0])
    max_disp = np.max(disparity_map)

    # Converti in uint8 per la visualizzazione
    disp_norm = np.zeros_like(disparity_map, dtype=np.uint8)
    disp_norm[disparity_map > 0] = (255 * (disparity_map[disparity_map > 0] - min_disp) / (max_disp - min_disp)).astype(
        np.uint8)

    # Applica una colormap
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

    # Visualizza
    cv2.imshow("Mappa di Disparità", disp_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Salva l'immagine
    disp_file = os.path.join(output_dir, "disparity_map.png")
    cv2.imwrite(disp_file, disp_color)
    print(f"Mappa di disparità salvata in: {disp_file}")


# Funzione principale
def main():
    global output_dir

    # Parsing degli argomenti
    parser = argparse.ArgumentParser(description="Scansione 3D stereo per UnLook")
    parser.add_argument("-o", "--output", default="output", help="Directory di output")
    parser.add_argument("-c", "--calibration", default="calibration/stereo_calibration.json",
                        help="File di calibrazione stereo")
    parser.add_argument("--mode", choices=["structured_light", "passive_stereo", "both"], default="both",
                        help="Modalità di scansione: structured_light, passive_stereo o both")
    args = parser.parse_args()

    output_dir = args.output
    ensure_output_dir()

    try:
        # Connessione allo scanner
        if not connect_to_scanner():
            return 1

        # Carica la calibrazione stereo
        if not load_stereo_calibration(args.calibration):
            client.disconnect()
            client.stop_discovery()
            return 1

        points = None
        colors = None
        disparity = None
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        try:
            # Esegui la scansione nella modalità richiesta
            if args.mode in ["passive_stereo", "both"]:
                print("\n=== Modalità Stereo Passiva ===")
                points, colors, disparity = capture_and_process_stereo_pair()

                if points is not None:
                    # Esporta i risultati
                    passive_filename = os.path.join(output_dir, f"passive_stereo_{timestamp}")
                    export_point_cloud(points, colors, passive_filename)

                    # Visualizza la mappa di disparità
                    if disparity is not None:
                        display_disparity_map(disparity)

            if args.mode in ["structured_light", "both"]:
                print("\n=== Modalità Luce Strutturata ===")
                result_left, result_right = perform_structured_light_scan()

                # Combina i risultati
                sl_points, sl_colors = combine_scan_results(result_left, result_right)

                if sl_points is not None:
                    # Esporta i risultati
                    sl_filename = os.path.join(output_dir, f"structured_light_{timestamp}")
                    export_point_cloud(sl_points, sl_colors, sl_filename)

                    # Se in modalità "both", combina con i risultati stereo passivi
                    if args.mode == "both" and points is not None:
                        combined_points = np.vstack((points, sl_points))

                        # Combina i colori se disponibili
                        combined_colors = None
                        if colors is not None and sl_colors is not None:
                            combined_colors = np.vstack((colors, sl_colors))

                        # Esporta i risultati combinati
                        combined_filename = os.path.join(output_dir, f"combined_{timestamp}")
                        export_point_cloud(combined_points, combined_colors, combined_filename)
                        print(f"Nuvola di punti combinata: {len(combined_points)} punti totali.")

        finally:
            # Metti il proiettore in standby
            client.projector.set_standby()

        # Disconnetti
        client.disconnect()
        client.stop_discovery()
        print("Scansione completata. Uscita.")
        return 0

    except KeyboardInterrupt:
        print("\nInterruzione rilevata. Uscita.")
        if client:
            client.projector.set_standby()
            client.disconnect()
            client.stop_discovery()
        return 130
    except Exception as e:
        print(f"Errore non gestito: {e}")
        if client:
            client.disconnect()
            client.stop_discovery()
        return 1
    finally:
        # Chiudi eventuali finestre OpenCV
        cv2.destroyAllWindows()


if __name__ == "__main__":
    exit(main())