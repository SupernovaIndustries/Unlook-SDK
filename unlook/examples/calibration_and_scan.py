#!/usr/bin/env python3
"""
Script per la calibrazione e scansione 3D con UnLook.

Questo script esegue la calibrazione del sistema e una scansione 3D
utilizzando pattern di luce strutturata.
"""

import os
import sys
import time
import logging
import argparse
import cv2
import numpy as np

# Aggiungi la directory corrente al percorso di ricerca dei moduli
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from unlook.client import (
        UnlookClient, UnlookClientEvent,
        Calibrator, CalibrationData,
        StereoCalibrator, StereoCalibrationData,
        ScanProcessor, PatternDirection,
        ModelExporter
    )
except ImportError as e:
    print(f"ERRORE: Impossibile importare i moduli UnLook: {e}")
    print("Assicurati che l'SDK UnLook sia installato correttamente")
    sys.exit(1)

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Variabili globali
client = None
output_dir = "output"
calibration_dir = "calibration"


# Callback per gli eventi del client
def on_connected(scanner):
    """Callback chiamato quando il client si connette a uno scanner."""
    print(f"Connesso a: {scanner.name} ({scanner.uuid})")


def on_disconnected():
    """Callback chiamato quando il client si disconnette da uno scanner."""
    print("Disconnesso dallo scanner")


def on_error(error):
    """Callback chiamato quando si verifica un errore."""
    print(f"Errore: {error}")


def on_scanner_found(scanner, added):
    """Callback chiamato quando viene trovato uno scanner."""
    if added:
        print(f"Scanner trovato: {scanner.name} ({scanner.uuid}) su {scanner.endpoint}")
    else:
        print(f"Scanner perso: {scanner.name} ({scanner.uuid})")


def ensure_directories():
    """Crea le directory necessarie."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory di output: {output_dir}")

    os.makedirs(calibration_dir, exist_ok=True)
    print(f"Directory di calibrazione: {calibration_dir}")


def discover_and_connect():
    """Scopre e si connette a uno scanner."""
    global client

    print("Ricerca scanner in corso...")
    client.start_discovery(on_scanner_found)

    # Attendi 5 secondi per la discovery
    time.sleep(5)

    # Ottieni la lista degli scanner
    scanners = client.get_discovered_scanners()

    if not scanners:
        print("Nessuno scanner trovato. Uscita.")
        client.stop_discovery()
        return None

    print("\nScanner disponibili:")
    for i, scanner in enumerate(scanners):
        print(f"{i + 1}. {scanner.name} ({scanner.uuid}) - {scanner.endpoint}")

    # Se c'è un solo scanner, selezionalo automaticamente
    if len(scanners) == 1:
        scanner = scanners[0]
        print(f"Selezionato automaticamente l'unico scanner disponibile: {scanner.name}")
    else:
        # Chiedi all'utente di scegliere
        while True:
            try:
                choice = int(input("\nSeleziona lo scanner (numero): "))
                if 1 <= choice <= len(scanners):
                    scanner = scanners[choice - 1]
                    break
                else:
                    print("Scelta non valida.")
            except ValueError:
                print("Inserire un numero valido.")

    # Connetti allo scanner
    print(f"Connessione a {scanner.name}...")
    if not client.connect(scanner):
        print("Impossibile connettersi allo scanner.")
        return None

    return scanner


def get_cameras():
    """Ottiene la lista delle telecamere e seleziona quella da usare."""
    global client

    if not client.connected:
        print("Client non connesso.")
        return None

    # Ottieni la lista delle telecamere
    cameras = client.camera.get_cameras()

    if not cameras:
        print("Nessuna telecamera disponibile.")
        return None

    print("\nTelecamere disponibili:")
    for i, camera in enumerate(cameras):
        print(f"{i + 1}. {camera['name']} - {camera['id']}")

    # Se c'è solo una telecamera, selezionala automaticamente
    if len(cameras) == 1:
        camera_id = cameras[0]["id"]
        print(f"Selezionata automaticamente l'unica telecamera disponibile: {cameras[0]['name']}")
        return camera_id

    # Altrimenti, chiedi all'utente
    while True:
        try:
            choice = int(input("\nSeleziona la telecamera (numero): "))
            if 1 <= choice <= len(cameras):
                camera_id = cameras[choice - 1]["id"]
                return camera_id
            else:
                print("Scelta non valida.")
        except ValueError:
            print("Inserire un numero valido.")


def get_stereo_pair():
    """Ottiene la coppia stereo."""
    global client

    if not client.connected:
        print("Client non connesso.")
        return None, None

    # Ottieni la coppia stereo
    left_camera_id, right_camera_id = client.camera.get_stereo_pair()

    if left_camera_id is None or right_camera_id is None:
        print("Nessuna coppia stereo disponibile.")
        return None, None

    print(f"\nCoppia stereo trovata:")
    print(f"Telecamera sinistra: {left_camera_id}")
    print(f"Telecamera destra: {right_camera_id}")

    return left_camera_id, right_camera_id


def calibrate_single_camera(camera_id):
    """Calibra una singola telecamera."""
    global client, calibration_dir

    print(f"\n=== Calibrazione telecamera {camera_id} ===")

    if not client.connected:
        print("Client non connesso.")
        return None

    # Crea il calibratore
    calibrator = Calibrator(client)

    print("Posiziona una scacchiera davanti alla telecamera.")
    print("La scacchiera deve essere completamente visibile e stabile.")
    print("Verranno acquisite 15 immagini da diverse angolazioni.")
    input("Premi INVIO quando sei pronto...")

    # Esegui la calibrazione
    success, message = calibrator.calibrate_camera(
        camera_id=camera_id,
        num_images=15,
        delay_between_captures=1.0,
        show_preview=True
    )

    if not success:
        print(f"Calibrazione fallita: {message}")
        return None

    print(f"Calibrazione completata: {message}")

    # Salva la calibrazione
    calib_file = os.path.join(calibration_dir, f"camera_{camera_id}_calibration.json")
    if calibrator.current_calibration.save(calib_file):
        print(f"Calibrazione salvata in: {calib_file}")
        return calibrator.current_calibration
    else:
        print("Errore durante il salvataggio della calibrazione.")
        return None


def calibrate_stereo_system():
    """Calibra il sistema stereo."""
    global client, calibration_dir

    print("\n=== Calibrazione stereo ===")

    if not client.connected:
        print("Client non connesso.")
        return None

    # Ottieni la coppia stereo
    left_camera_id, right_camera_id = get_stereo_pair()

    if left_camera_id is None or right_camera_id is None:
        print("Impossibile calibrare il sistema stereo: coppia stereo non disponibile.")
        return None

    # Prima calibra le singole telecamere
    print("\nCalibrando la telecamera sinistra...")
    left_calib = calibrate_single_camera(left_camera_id)

    if left_calib is None:
        print("Impossibile calibrare il sistema stereo: calibrazione sinistra fallita.")
        return None

    print("\nCalibrando la telecamera destra...")
    right_calib = calibrate_single_camera(right_camera_id)

    if right_calib is None:
        print("Impossibile calibrare il sistema stereo: calibrazione destra fallita.")
        return None

    # Crea il calibratore stereo
    stereo_calibrator = StereoCalibrator(left_calib, right_calib)

    print("\nOra calibriamo il sistema stereo.")
    print("Posiziona una scacchiera visibile da ENTRAMBE le telecamere.")
    print("La scacchiera deve essere completamente visibile in entrambe le immagini.")
    print("Verranno acquisite 15 coppie di immagini da diverse angolazioni.")
    input("Premi INVIO quando sei pronto...")

    # Acquisisci immagini stereo
    left_images = []
    right_images = []

    for i in range(15):
        print(f"Acquisizione coppia {i + 1}/15...")

        # Cattura le immagini
        left_image, right_image = client.camera.capture_stereo_pair()

        if left_image is None or right_image is None:
            print("Errore durante la cattura delle immagini stereo. Riprova.")
            continue

        # Cerca la scacchiera in entrambe le immagini
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        board_size = (9, 6)  # Numero di angoli interni

        left_found, left_corners = cv2.findChessboardCorners(
            left_gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )

        right_found, right_corners = cv2.findChessboardCorners(
            right_gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )

        if not (left_found and right_found):
            print("Scacchiera non rilevata in una o entrambe le immagini. Riprova.")
            continue

        # Visualizza le immagini con gli angoli
        left_corners_image = cv2.drawChessboardCorners(left_image.copy(), board_size, left_corners, left_found)
        right_corners_image = cv2.drawChessboardCorners(right_image.copy(), board_size, right_corners, right_found)

        # Visualizza le immagini affiancate
        corners_image = np.hstack((left_corners_image, right_corners_image))

        # Ridimensiona se l'immagine è troppo grande
        h, w = corners_image.shape[:2]
        if w > 1600:
            scale = 1600 / w
            corners_image = cv2.resize(corners_image, (1600, int(h * scale)))

        cv2.putText(
            corners_image,
            f"Coppia {i + 1}/15",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Calibrazione stereo", corners_image)
        key = cv2.waitKey(1000)

        if key == 27:  # ESC
            break

        # Aggiungi alle liste
        left_images.append(left_image)
        right_images.append(right_image)

        # Breve pausa tra le acquisizioni
        time.sleep(1.0)

    cv2.destroyAllWindows()

    # Verifica che ci siano abbastanza immagini
    if len(left_images) < 5 or len(right_images) < 5:
        print("Troppo poche immagini per la calibrazione stereo.")
        return None

    # Esegui la calibrazione stereo
    print(f"Calibrazione stereo in corso con {len(left_images)} coppie di immagini...")

    # Dimensioni dell'immagine
    image_size = (left_images[0].shape[1], left_images[0].shape[0])

    # Calibrazione stereo
    success, message = stereo_calibrator.calibrate_stereo(left_images, right_images, image_size)

    if not success:
        print(f"Calibrazione stereo fallita: {message}")
        return None

    print(f"Calibrazione stereo completata: {message}")

    # Salva la calibrazione stereo
    stereo_calib_file = os.path.join(calibration_dir, "stereo_calibration.json")
    if stereo_calibrator.save_calibration(stereo_calib_file):
        print(f"Calibrazione stereo salvata in: {stereo_calib_file}")
        return stereo_calibrator.stereo_calib
    else:
        print("Errore durante il salvataggio della calibrazione stereo.")
        return None


def perform_structured_light_scan(camera_id, calibration_data=None):
    """Esegue una scansione a luce strutturata."""
    global client, output_dir

    print("\n=== Scansione a luce strutturata ===")

    if not client.connected:
        print("Client non connesso.")
        return False

    # Crea il processore di scansione
    scan_processor = ScanProcessor(client)

    # Imposta la calibrazione se disponibile
    if calibration_data is not None:
        scan_processor.calibration = calibration_data
        print("Dati di calibrazione impostati")

    print("Posiziona l'oggetto da scansionare.")
    print("L'oggetto deve essere ben illuminato e non riflettente.")
    input("Premi INVIO quando sei pronto...")

    # Esegui la scansione
    print("Esecuzione scansione in corso...")
    success, result = scan_processor.capture_gray_code_scan(
        camera_id=camera_id,
        pattern_width=1280,
        pattern_height=800,
        direction=PatternDirection.BOTH,
        capture_texture=True,
        show_preview=True
    )

    if not success:
        print("Scansione fallita.")
        return False

    print(f"Scansione completata: {result.num_points} punti 3D")

    # Esporta i risultati
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.join(output_dir, f"scan_{timestamp}")

    exporter = ModelExporter()

    # Esporta in formati diversi
    if exporter.export_ply(result, f"{base_filename}.ply"):
        print(f"Nuvola di punti esportata in: {base_filename}.ply")

    if exporter.export_obj(result, f"{base_filename}.obj"):
        print(f"Modello OBJ esportato in: {base_filename}.obj")

    if exporter.export_xyz(result, f"{base_filename}.xyz"):
        print(f"File XYZ esportato in: {base_filename}.xyz")

    # Esporta i metadati
    if exporter.export_metadata(result, f"{base_filename}_meta.json"):
        print(f"Metadati esportati in: {base_filename}_meta.json")

    return True


def perform_stereo_scan():
    """Esegue una scansione stereo."""
    global client, calibration_dir, output_dir

    print("\n=== Scansione stereo ===")

    if not client.connected:
        print("Client non connesso.")
        return False

    # Carica la calibrazione stereo
    stereo_calib_file = os.path.join(calibration_dir, "stereo_calibration.json")

    if not os.path.exists(stereo_calib_file):
        print(f"File di calibrazione stereo non trovato: {stereo_calib_file}")
        print("Esegui prima la calibrazione stereo.")
        return False

    stereo_calib = StereoCalibrationData.load(stereo_calib_file)

    if not stereo_calib or not stereo_calib.is_valid():
        print("Calibrazione stereo non valida.")
        return False

    print("Calibrazione stereo caricata con successo.")

    # Ottieni la coppia stereo
    left_camera_id, right_camera_id = get_stereo_pair()

    if left_camera_id is None or right_camera_id is None:
        print("Impossibile eseguire la scansione stereo: coppia stereo non disponibile.")
        return False

    from unlook.client import StereoProcessor

    # Crea il processore stereo
    stereo_processor = StereoProcessor(stereo_calib)

    print("Posiziona l'oggetto da scansionare.")
    print("L'oggetto deve essere visibile da entrambe le telecamere.")
    input("Premi INVIO quando sei pronto...")

    # Cattura un'immagine stereo
    print("Acquisizione immagini stereo...")
    left_image, right_image = client.camera.capture_stereo_pair()

    if left_image is None or right_image is None:
        print("Errore durante la cattura delle immagini stereo.")
        return False

    # Visualizza le immagini stereo
    stereo_image = np.hstack((left_image, right_image))

    # Ridimensiona se l'immagine è troppo grande
    h, w = stereo_image.shape[:2]
    if w > 1600:
        scale = 1600 / w
        stereo_image = cv2.resize(stereo_image, (1600, int(h * scale)))

    cv2.imshow("Immagini stereo", stereo_image)
    cv2.waitKey(1000)

    # Elabora le immagini stereo
    print("Elaborazione immagini stereo in corso...")
    points, colors, disparity = stereo_processor.compute_stereo_scan(left_image, right_image)

    if points is None:
        print("Errore durante l'elaborazione stereo.")
        return False

    print(f"Elaborazione stereo completata: {len(points)} punti 3D")

    # Visualizza la mappa di disparità
    if disparity is not None:
        # Normalizza per la visualizzazione
        min_disp = np.min(disparity[disparity > 0])
        max_disp = np.max(disparity)

        # Converti in uint8 per la visualizzazione
        disp_norm = np.zeros_like(disparity, dtype=np.uint8)
        disp_norm[disparity > 0] = (255 * (disparity[disparity > 0] - min_disp) / (max_disp - min_disp)).astype(
            np.uint8)

        # Applica una colormap
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

        # Visualizza
        cv2.imshow("Mappa di disparità", disp_color)
        cv2.waitKey(2000)

        # Salva l'immagine
        disp_file = os.path.join(output_dir, "disparity_map.png")
        cv2.imwrite(disp_file, disp_color)
        print(f"Mappa di disparità salvata in: {disp_file}")

    # Esporta la nuvola di punti
    if points is not None:
        from unlook.client.processing import ProcessingResult

        # Crea un oggetto ProcessingResult
        result = ProcessingResult()
        result.point_cloud = points
        result.colors = colors
        result.num_points = len(points)

        # Esporta
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(output_dir, f"stereo_scan_{timestamp}")

        exporter = ModelExporter()

        if exporter.export_ply(result, f"{base_filename}.ply"):
            print(f"Nuvola di punti esportata in: {base_filename}.ply")

        if exporter.export_obj(result, f"{base_filename}.obj"):
            print(f"Modello OBJ esportato in: {base_filename}.obj")

    cv2.destroyAllWindows()
    return True


def menu():
    """Mostra il menu principale."""
    global client

    while True:
        print("\n=== UnLook Scanner 3D ===")
        print("1. Calibra singola telecamera")
        print("2. Calibra sistema stereo")
        print("3. Esegui scansione a luce strutturata")
        print("4. Esegui scansione stereo")
        print("5. Disconnetti")
        print("0. Esci")

        choice = input("Scelta: ")

        if choice == "1":
            camera_id = get_cameras()
            if camera_id:
                calibration_data = calibrate_single_camera(camera_id)
        elif choice == "2":
            stereo_calib = calibrate_stereo_system()
        elif choice == "3":
            camera_id = get_cameras()
            if camera_id:
                # Prova a caricare la calibrazione
                calib_file = os.path.join(calibration_dir, f"camera_{camera_id}_calibration.json")
                calibration_data = None

                if os.path.exists(calib_file):
                    calibration_data = CalibrationData.load(calib_file)
                    if calibration_data and calibration_data.is_valid():
                        print(f"Calibrazione caricata da: {calib_file}")
                    else:
                        print("Impossibile caricare la calibrazione.")

                perform_structured_light_scan(camera_id, calibration_data)
        elif choice == "4":
            perform_stereo_scan()
        elif choice == "5":
            if client.connected:
                client.disconnect()
                print("Disconnesso dallo scanner.")

                # Riconnetti
                scanner = discover_and_connect()
                if not scanner:
                    print("Impossibile riconnettersi a uno scanner.")
        elif choice == "0":
            break
        else:
            print("Scelta non valida.")


def main():
    """Funzione principale."""
    global client, output_dir, calibration_dir

    parser = argparse.ArgumentParser(description="Calibrazione e scansione 3D con UnLook")
    parser.add_argument("-o", "--output", default="output", help="Directory di output")
    parser.add_argument("-c", "--calibration", default="calibration", help="Directory per i dati di calibrazione")
    args = parser.parse_args()

    output_dir = args.output
    calibration_dir = args.calibration

    # Assicura che le directory esistano
    ensure_directories()

    # Crea il client
    client = UnlookClient(client_name="UnlookCalibrationAndScanClient")

    # Registra i callback
    client.on_event(UnlookClientEvent.CONNECTED, on_connected)
    client.on_event(UnlookClientEvent.DISCONNECTED, on_disconnected)
    client.on_event(UnlookClientEvent.ERROR, on_error)

    try:
        # Scopri e connetti a uno scanner
        scanner = discover_and_connect()

        if not scanner:
            print("Impossibile connettersi a uno scanner. Uscita.")
            client.stop_discovery()
            return

        # Mostra il menu principale
        menu()

    except KeyboardInterrupt:
        print("\nOperazione interrotta dall'utente.")
    except Exception as e:
        print(f"\nErrore imprevisto: {e}")
    finally:
        # Disconnetti e ferma la discovery
        if client:
            if client.connected:
                client.disconnect()
            client.stop_discovery()

        # Chiudi eventuali finestre OpenCV
        cv2.destroyAllWindows()

        print("Programma terminato.")


if __name__ == "__main__":
    main()