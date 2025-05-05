"""
Esempio completo di scansione 3D con UnLook.
"""

import logging
import time
import os
import cv2
import numpy as np
import argparse

from unlook.client import (
    UnlookClient, UnlookClientEvent,
    Calibrator, CalibrationData,
    ScanProcessor, PatternDirection,
    ModelExporter
)

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Variabili globali
client = None
calibrator = None
scan_processor = None
exporter = None
output_dir = "output"


# Callback per la scoperta degli scanner
def scanner_found_callback(scanner, added):
    if added:
        print(f"Scanner trovato: {scanner.name} ({scanner.uuid}) su {scanner.endpoint}")
    else:
        print(f"Scanner perso: {scanner.name} ({scanner.uuid})")


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


# Funzioni principali
def discover_scanners(timeout=5):
    """Scopre gli scanner disponibili."""
    print(f"Ricerca scanner in corso ({timeout} secondi)...")
    time.sleep(timeout)

    scanners = client.get_discovered_scanners()

    if not scanners:
        print("Nessuno scanner trovato.")
        return None

    print("\nScanner disponibili:")
    for i, scanner in enumerate(scanners):
        print(f"{i + 1}. {scanner.name} ({scanner.uuid}) - {scanner.endpoint}")

    # Se c'è un solo scanner, selezionalo automaticamente
    if len(scanners) == 1:
        print(f"Selezionato automaticamente l'unico scanner disponibile.")
        return scanners[0]

    # Altrimenti, chiedi all'utente
    while True:
        try:
            choice = int(input("\nSeleziona lo scanner (numero): "))
            if 1 <= choice <= len(scanners):
                return scanners[choice - 1]
            else:
                print("Scelta non valida.")
        except ValueError:
            print("Inserire un numero valido.")


def connect_to_scanner(scanner):
    """Connette a uno scanner."""
    print(f"Connessione a: {scanner.name} ({scanner.uuid})...")

    if client.connect(scanner):
        print("Connessione stabilita.")

        # Ottieni informazioni
        info = client.get_info()
        print("\nInformazioni scanner:")
        print(f"Nome: {info.get('scanner_name', 'N/A')}")
        print(f"UUID: {info.get('scanner_uuid', 'N/A')}")

        # Elenca le telecamere
        cameras = client.camera.get_cameras()
        print(f"\nTelecamere disponibili: {len(cameras)}")
        for i, camera in enumerate(cameras):
            print(f"{i + 1}. {camera['name']} - {camera['id']}")

        if cameras:
            return cameras[0]["id"]  # Restituisci l'ID della prima telecamera
        else:
            print("Nessuna telecamera disponibile.")
            return None
    else:
        print("Impossibile connettersi allo scanner.")
        return None


def calibrate_camera(camera_id):
    """Calibra la telecamera."""
    print("\n--- Calibrazione Telecamera ---")
    print("Tenere una scacchiera davanti alla telecamera.")
    print("Verranno catturate 15 immagini da diverse angolazioni.")

    input("Premere INVIO per iniziare...")

    # Crea calibratore
    global calibrator
    calibrator = Calibrator(client)

    # Esegui la calibrazione
    success, message = calibrator.calibrate_camera(
        camera_id=camera_id,
        num_images=15,
        delay_between_captures=1.0,
        show_preview=True
    )

    if success:
        print(f"Calibrazione telecamera completata: {message}")

        # Salva i dati di calibrazione
        calib_file = os.path.join(output_dir, "calibration.json")
        if calibrator.current_calibration.save(calib_file):
            print(f"Dati di calibrazione salvati in: {calib_file}")
            return True
        else:
            print("Errore durante il salvataggio dei dati di calibrazione.")
            return False
    else:
        print(f"Calibrazione fallita: {message}")
        return False


def load_calibration():
    """Carica la calibrazione da file."""
    calib_file = os.path.join(output_dir, "calibration.json")

    if not os.path.exists(calib_file):
        print(f"File di calibrazione non trovato: {calib_file}")
        return False

    # Crea calibratore se non esiste
    global calibrator
    if calibrator is None:
        calibrator = Calibrator(client)

    # Carica la calibrazione
    calib = CalibrationData.load(calib_file)
    if calib and calib.is_valid():
        calibrator.current_calibration = calib
        print(f"Calibrazione caricata da: {calib_file}")
        return True
    else:
        print("Errore durante il caricamento della calibrazione.")
        return False


def perform_scan(camera_id):
    """Esegue una scansione 3D."""
    print("\n--- Scansione 3D ---")

    # Verifica se è disponibile la calibrazione
    global calibrator
    if calibrator is None or not calibrator.current_calibration.is_valid():
        if not load_calibration():
            print("Calibrazione non disponibile. Procedere comunque?")
            proceed = input("Continuare? (s/n): ").lower() == 's'
            if not proceed:
                return False

    # Crea il processore di scansione
    global scan_processor
    scan_processor = ScanProcessor(client)

    # Imposta la calibrazione
    if calibrator is not None and calibrator.current_calibration.is_valid():
        scan_processor.calibration = calibrator.current_calibration

    print("Posizionare l'oggetto da scansionare.")
    input("Premere INVIO per iniziare la scansione...")

    # Esegui la scansione con Gray code
    success, result = scan_processor.capture_gray_code_scan(
        camera_id=camera_id,
        pattern_width=1280,
        pattern_height=800,
        direction=PatternDirection.BOTH,
        capture_texture=True,
        show_preview=True
    )

    if success and result.has_point_cloud():
        print(f"Scansione completata: {result.num_points} punti 3D")

        # Crea nome file unico con timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(output_dir, f"scan_{timestamp}")

        # Esporta i risultati
        global exporter
        if exporter is None:
            exporter = ModelExporter()

        # Esporta in vari formati
        ply_file = f"{base_filename}.ply"
        if exporter.export_ply(result, ply_file):
            print(f"Nuvola di punti esportata in: {ply_file}")

        obj_file = f"{base_filename}.obj"
        if exporter.export_obj(result, obj_file):
            print(f"Modello OBJ esportato in: {obj_file}")

        xyz_file = f"{base_filename}.xyz"
        if exporter.export_xyz(result, xyz_file):
            print(f"File XYZ esportato in: {xyz_file}")

        meta_file = f"{base_filename}_meta.json"
        if exporter.export_metadata(result, meta_file):
            print(f"Metadati esportati in: {meta_file}")

        return True
    else:
        print("Scansione fallita o nessun punto valido.")
        return False


def main():
    """Funzione principale."""
    global client, output_dir

    # Parsing degli argomenti
    parser = argparse.ArgumentParser(description="Esempio di scansione 3D con UnLook")
    parser.add_argument("-o", "--output", default="output", help="Directory di output")
    args = parser.parse_args()

    output_dir = args.output
    ensure_output_dir()

    # Crea il client
    client = UnlookClient(client_name="UnlookScannerDemo")

    # Registra i callback per gli eventi
    client.on_event(UnlookClientEvent.CONNECTED, on_connected)
    client.on_event(UnlookClientEvent.DISCONNECTED, on_disconnected)
    client.on_event(UnlookClientEvent.ERROR, on_error)

    try:
        # Avvia la discovery degli scanner
        client.start_discovery(scanner_found_callback)

        # Menu principale
        scanner = None
        camera_id = None
        calibrated = False

        while True:
            print("\n=== UnLook Scanner 3D ===")

            if scanner is None:
                print("1. Cerca scanner")
                print("0. Esci")

                choice = input("Scelta: ")
                if choice == "1":
                    scanner = discover_scanners()
                    if scanner:
                        camera_id = connect_to_scanner(scanner)
                elif choice == "0":
                    break
                else:
                    print("Scelta non valida.")
            else:
                print(f"Scanner connesso: {scanner.name}")
                print(f"Telecamera selezionata: {camera_id}")
                print("1. Calibra telecamera")
                print("2. Esegui scansione 3D")
                print("3. Disconnetti")
                print("0. Esci")

                choice = input("Scelta: ")
                if choice == "1":
                    calibrated = calibrate_camera(camera_id)
                elif choice == "2":
                    perform_scan(camera_id)
                elif choice == "3":
                    client.disconnect()
                    scanner = None
                    camera_id = None
                    calibrated = False
                    print("Disconnesso dallo scanner.")
                elif choice == "0":
                    break
                else:
                    print("Scelta non valida.")

    except KeyboardInterrupt:
        print("\nInterruzione rilevata.")

    finally:
        # Pulizia
        if client:
            client.disconnect()
            client.stop_discovery()

        # Chiudi le finestre OpenCV
        cv2.destroyAllWindows()

        print("Programma terminato.")


if __name__ == "__main__":
    main()