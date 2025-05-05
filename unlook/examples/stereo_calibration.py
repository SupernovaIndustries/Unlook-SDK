"""
Esempio di calibrazione stereo con UnLook.

Questo script esegue la calibrazione di un sistema stereo utilizzando
una scacchiera, e salva i parametri di calibrazione per un uso futuro.
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
    StereoCalibrator, StereoCalibrationData
)

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Variabili globali
client = None
output_dir = "calibration"


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
    client = UnlookClient(client_name="UnlookStereoCalibration")

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


# Funzione per calibrare le singole telecamere
def calibrate_single_cameras(num_images=15):
    # Ottieni la coppia stereo
    left_camera_id, right_camera_id = client.camera.get_stereo_pair()

    if left_camera_id is None or right_camera_id is None:
        print("Impossibile trovare una coppia stereo valida.")
        return None, None

    print(f"Coppia stereo trovata: {left_camera_id} (sinistra), {right_camera_id} (destra)")

    # Crea calibratori per le singole telecamere
    left_calibrator = Calibrator(client)
    right_calibrator = Calibrator(client)

    # Calibrazione telecamera sinistra
    print("\n--- Calibrazione Telecamera Sinistra ---")
    print("Tenere una scacchiera davanti alla telecamera sinistra.")
    print(f"Verranno catturate {num_images} immagini da diverse angolazioni.")

    input("Premere INVIO per iniziare...")

    success_left, message_left = left_calibrator.calibrate_camera(
        camera_id=left_camera_id,
        num_images=num_images,
        delay_between_captures=1.0,
        show_preview=True
    )

    if not success_left:
        print(f"Calibrazione telecamera sinistra fallita: {message_left}")
        return None, None

    print(f"Calibrazione telecamera sinistra completata: {message_left}")

    # Salva calibrazione sinistra
    left_calib_file = os.path.join(output_dir, "left_camera_calibration.json")
    if left_calibrator.current_calibration.save(left_calib_file):
        print(f"Calibrazione telecamera sinistra salvata in: {left_calib_file}")

    # Calibrazione telecamera destra
    print("\n--- Calibrazione Telecamera Destra ---")
    print("Tenere una scacchiera davanti alla telecamera destra.")
    print(f"Verranno catturate {num_images} immagini da diverse angolazioni.")

    input("Premere INVIO per iniziare...")

    success_right, message_right = right_calibrator.calibrate_camera(
        camera_id=right_camera_id,
        num_images=num_images,
        delay_between_captures=1.0,
        show_preview=True
    )

    if not success_right:
        print(f"Calibrazione telecamera destra fallita: {message_right}")
        return None, None

    print(f"Calibrazione telecamera destra completata: {message_right}")

    # Salva calibrazione destra
    right_calib_file = os.path.join(output_dir, "right_camera_calibration.json")
    if right_calibrator.current_calibration.save(right_calib_file):
        print(f"Calibrazione telecamera destra salvata in: {right_calib_file}")

    return left_calibrator.current_calibration, right_calibrator.current_calibration


# Funzione per calibrare il sistema stereo
def calibrate_stereo(left_camera_calib, right_camera_calib, num_images=15):
    # Ottieni la coppia stereo
    left_camera_id, right_camera_id = client.camera.get_stereo_pair()

    if left_camera_id is None or right_camera_id is None:
        print("Impossibile trovare una coppia stereo valida.")
        return False

    # Crea il calibratore stereo
    stereo_calibrator = StereoCalibrator(left_camera_calib, right_camera_calib)

    print("\n--- Calibrazione Stereo ---")
    print("Tenere una scacchiera visibile da ENTRAMBE le telecamere simultaneamente.")
    print(f"Verranno catturate {num_images} coppie di immagini da diverse angolazioni.")
    print("Assicurarsi che la scacchiera sia completamente visibile in entrambe le immagini.")

    input("Premere INVIO per iniziare...")

    # Raccogli coppie di immagini
    left_images = []
    right_images = []

    for i in range(num_images):
        print(f"Acquisizione coppia {i + 1}/{num_images}...")

        # Cattura immagini sincronizzate
        images = client.camera.capture_multi([left_camera_id, right_camera_id])

        if not images or left_camera_id not in images or right_camera_id not in images:
            print("Errore durante la cattura delle immagini. Riprova.")
            continue

        left_img = images[left_camera_id]
        right_img = images[right_camera_id]

        # Visualizza preview
        preview = np.hstack((left_img, right_img))
        # Ridimensiona se troppo grande
        h, w = preview.shape[:2]
        if w > 1600:
            preview = cv2.resize(preview, (1600, int(h * 1600 / w)))

        cv2.putText(
            preview,
            f"Coppia {i + 1}/{num_images}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.imshow("Calibrazione Stereo", preview)
        key = cv2.waitKey(1000)

        if key == 27:  # ESC
            break

        # Aggiungi alle liste
        left_images.append(left_img)
        right_images.append(right_img)

        # Pausa tra le acquisizioni
        time.sleep(1.0)

    # Chiudi la finestra
    cv2.destroyAllWindows()

    # Verifica di avere abbastanza immagini
    if len(left_images) < 3:
        print("Raccolte meno di 3 coppie di immagini. Impossibile calibrare.")
        return False

    print(f"Raccolte {len(left_images)} coppie di immagini. Avvio calibrazione stereo...")

    # Esegui la calibrazione stereo
    image_size = (left_images[0].shape[1], left_images[0].shape[0])
    success, message = stereo_calibrator.calibrate_stereo(left_images, right_images, image_size)

    if not success:
        print(f"Calibrazione stereo fallita: {message}")
        return False

    print(f"Calibrazione stereo completata: {message}")

    # Salva la calibrazione stereo
    stereo_calib_file = os.path.join(output_dir, "stereo_calibration.json")
    if stereo_calibrator.save_calibration(stereo_calib_file):
        print(f"Calibrazione stereo salvata in: {stereo_calib_file}")
        return True
    else:
        print("Errore durante il salvataggio della calibrazione stereo.")
        return False


# Funzione principale
def main():
    global output_dir

    # Parsing degli argomenti
    parser = argparse.ArgumentParser(description="Calibrazione stereo per UnLook")
    parser.add_argument("-o", "--output", default="calibration", help="Directory di output")
    parser.add_argument("-n", "--num-images", type=int, default=15, help="Numero di immagini per calibrazione")
    parser.add_argument("--load-camera-calib", action="store_true", help="Carica calibrazioni telecamere esistenti")
    args = parser.parse_args()

    output_dir = args.output
    ensure_output_dir()

    try:
        # Connessione allo scanner
        if not connect_to_scanner():
            return 1

        left_camera_calib = None
        right_camera_calib = None

        # Carica o esegui calibrazione delle singole telecamere
        if args.load_camera_calib:
            left_calib_file = os.path.join(output_dir, "left_camera_calibration.json")
            right_calib_file = os.path.join(output_dir, "right_camera_calibration.json")

            if os.path.exists(left_calib_file) and os.path.exists(right_calib_file):
                left_camera_calib = CalibrationData.load(left_calib_file)
                right_camera_calib = CalibrationData.load(right_calib_file)

                if left_camera_calib and right_camera_calib and left_camera_calib.is_valid() and right_camera_calib.is_valid():
                    print("Calibrazioni telecamere caricate con successo.")
                else:
                    print("Errore durante il caricamento delle calibrazioni telecamere.")
                    return 1
            else:
                print("File di calibrazione telecamere non trovati. Esecuzione calibrazione...")
                left_camera_calib, right_camera_calib = calibrate_single_cameras(args.num_images)
        else:
            # Esegui calibrazione delle singole telecamere
            left_camera_calib, right_camera_calib = calibrate_single_cameras(args.num_images)

        if left_camera_calib is None or right_camera_calib is None:
            print("Impossibile procedere senza calibrazioni telecamere valide.")
            client.disconnect()
            client.stop_discovery()
            return 1

        # Esegui calibrazione stereo
        if calibrate_stereo(left_camera_calib, right_camera_calib, args.num_images):
            print("Calibrazione stereo completata con successo!")
        else:
            print("Calibrazione stereo fallita.")
            return 1

        # Disconnessione
        client.disconnect()
        client.stop_discovery()
        print("Calibrazione completata. Uscita.")
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
    finally:
        # Chiudi eventuali finestre OpenCV
        cv2.destroyAllWindows()


if __name__ == "__main__":
    exit(main())