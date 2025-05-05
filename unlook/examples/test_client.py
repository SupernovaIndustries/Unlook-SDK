#!/usr/bin/env python3
"""
Script di test per la connessione al server UnLook.
Questo script verifica la connessione al server e testa le funzionalità
di base come il controllo del proiettore e la cattura di immagini.
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
    from unlook.client import UnlookClient, UnlookClientEvent
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


def ensure_output_dir():
    """Crea la directory di output."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory di output: {output_dir}")


def discover_scanners(timeout=5):
    """Scopre gli scanner disponibili."""
    global client

    print(f"Ricerca scanner in corso ({timeout} secondi)...")

    # Avvia la discovery
    client.start_discovery(on_scanner_found)

    # Attendi il timeout
    time.sleep(timeout)

    # Ottieni la lista degli scanner trovati
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


def test_connection(scanner):
    """Testa la connessione a uno scanner."""
    global client

    print(f"Test di connessione a: {scanner.name} ({scanner.uuid})...")

    try:
        # Connessione
        if not client.connect(scanner):
            print("Errore: impossibile connettersi allo scanner")
            return False

        # Verifica la connessione ottenendo le informazioni
        info = client.get_info()

        print("\nInformazioni sullo scanner:")
        print(f"Nome: {info.get('scanner_name', 'N/A')}")
        print(f"UUID: {info.get('scanner_uuid', 'N/A')}")

        # Verifica le capacità
        capabilities = info.get('capabilities', {})
        projector_available = capabilities.get('projector', {}).get('available', False)
        cameras_available = capabilities.get('cameras', {}).get('available', False)
        cameras_count = capabilities.get('cameras', {}).get('count', 0)

        print(f"Proiettore disponibile: {projector_available}")
        print(f"Telecamere disponibili: {cameras_available} (conteggio: {cameras_count})")

        return True
    except Exception as e:
        print(f"Errore durante il test di connessione: {e}")
        return False


def test_projector():
    """Testa il controllo del proiettore."""
    global client

    print("\n=== Test del proiettore ===")

    if not client.connected:
        print("Errore: client non connesso")
        return False

    try:
        # Metti il proiettore in modalità test pattern
        if not client.projector.set_test_pattern_mode():
            print("Errore: impossibile impostare la modalità test pattern")
            return False

        print("Modalità test pattern impostata")

        # Proietta una sequenza di pattern
        patterns = [
            ("bianco", lambda: client.projector.show_solid_field("White")),
            ("nero", lambda: client.projector.show_solid_field("Black")),
            ("rosso", lambda: client.projector.show_solid_field("Red")),
            ("verde", lambda: client.projector.show_solid_field("Green")),
            ("blu", lambda: client.projector.show_solid_field("Blue")),
            ("griglia", lambda: client.projector.show_grid()),
            ("scacchiera", lambda: client.projector.show_checkerboard()),
            ("barre orizzontali", lambda: client.projector.show_horizontal_lines()),
            ("barre verticali", lambda: client.projector.show_vertical_lines()),
            ("barre di colore", lambda: client.projector.show_colorbars()),
        ]

        for name, pattern_func in patterns:
            print(f"Proiezione pattern: {name}")

            # Proietta il pattern
            pattern_func()

            # Attendi che l'utente prema un tasto
            input("Premi INVIO per continuare...")

        # Metti il proiettore in standby
        client.projector.set_standby()
        print("Proiettore in standby")

        return True
    except Exception as e:
        print(f"Errore durante il test del proiettore: {e}")
        return False


def test_cameras():
    """Testa le telecamere."""
    global client, output_dir

    print("\n=== Test delle telecamere ===")

    if not client.connected:
        print("Errore: client non connesso")
        return False

    try:
        # Ottieni la lista delle telecamere
        cameras = client.camera.get_cameras()

        if not cameras:
            print("Nessuna telecamera disponibile")
            return False

        print(f"Telecamere disponibili: {len(cameras)}")
        for i, camera in enumerate(cameras):
            print(f"{i + 1}. {camera['name']} - {camera['id']}")

        # Test di cattura immagini
        for camera in cameras:
            camera_id = camera["id"]
            print(f"\nTest della telecamera {camera['name']} ({camera_id}):")

            # Cattura un'immagine
            print("Cattura immagine...")
            image = client.camera.capture(camera_id)

            if image is None:
                print(f"Errore: impossibile catturare un'immagine dalla telecamera {camera_id}")
                continue

            # Visualizza l'immagine
            print(f"Immagine catturata: {image.shape}")

            # Salva l'immagine
            filename = os.path.join(output_dir, f"test_camera_{camera_id}.jpg")
            cv2.imwrite(filename, image)
            print(f"Immagine salvata come: {filename}")

            # Visualizza l'immagine
            cv2.imshow(f"Telecamera {camera_id}", image)
            cv2.waitKey(1000)

        # Prova a catturare immagini da tutte le telecamere simultaneamente
        if len(cameras) > 1:
            print("\nTest di cattura simultanea da più telecamere:")

            camera_ids = [camera["id"] for camera in cameras]
            images = client.camera.capture_multi(camera_ids)

            if not images:
                print("Errore: impossibile catturare immagini simultaneamente")
            else:
                print(f"Catturate {len(images)} immagini simultaneamente")

                # Visualizza un mosaico delle immagini
                if len(images) > 0:
                    # Ridimensiona le immagini alla stessa dimensione
                    height = min(image.shape[0] for image in images.values())
                    width = min(image.shape[1] for image in images.values())

                    # Crea un mosaico orizzontale
                    mosaic = np.hstack([cv2.resize(img, (width, height)) for img in images.values()])

                    # Visualizza il mosaico
                    cv2.imshow("Cattura simultanea", mosaic)
                    cv2.waitKey(2000)

                    # Salva il mosaico
                    filename = os.path.join(output_dir, "test_multi_camera.jpg")
                    cv2.imwrite(filename, mosaic)
                    print(f"Mosaico salvato come: {filename}")

        cv2.destroyAllWindows()
        return True
    except Exception as e:
        print(f"Errore durante il test delle telecamere: {e}")
        return False


def test_stereo_pair():
    """Testa la coppia stereo."""
    global client, output_dir

    print("\n=== Test della coppia stereo ===")

    if not client.connected:
        print("Errore: client non connesso")
        return False

    try:
        # Ottieni la coppia stereo
        left_camera_id, right_camera_id = client.camera.get_stereo_pair()

        if left_camera_id is None or right_camera_id is None:
            print("Nessuna coppia stereo disponibile")
            return False

        print(f"Coppia stereo trovata:")
        print(f"Telecamera sinistra: {left_camera_id}")
        print(f"Telecamera destra: {right_camera_id}")

        # Cattura un'immagine stereo
        print("Cattura immagine stereo...")
        left_image, right_image = client.camera.capture_stereo_pair()

        if left_image is None or right_image is None:
            print("Errore: impossibile catturare l'immagine stereo")
            return False

        # Visualizza le immagini affiancate
        stereo_image = np.hstack((left_image, right_image))

        # Ridimensiona se l'immagine è troppo grande
        h, w = stereo_image.shape[:2]
        if w > 1600:
            scale = 1600 / w
            stereo_image = cv2.resize(stereo_image, (1600, int(h * scale)))

        cv2.imshow("Coppia stereo", stereo_image)
        cv2.waitKey(2000)

        # Salva l'immagine
        filename = os.path.join(output_dir, "test_stereo_pair.jpg")
        cv2.imwrite(filename, stereo_image)
        print(f"Immagine stereo salvata come: {filename}")

        cv2.destroyAllWindows()
        return True
    except Exception as e:
        print(f"Errore durante il test della coppia stereo: {e}")
        return False


def run_all_tests():
    """Esegue tutti i test."""
    global client

    # Crea il client
    client = UnlookClient(client_name="UnlookTestClient")

    # Registra i callback
    client.on_event(UnlookClientEvent.CONNECTED, on_connected)
    client.on_event(UnlookClientEvent.DISCONNECTED, on_disconnected)
    client.on_event(UnlookClientEvent.ERROR, on_error)

    try:
        # Assicura che la directory di output esista
        ensure_output_dir()

        # Scopri gli scanner
        scanner = discover_scanners()
        if not scanner:
            print("Impossibile trovare uno scanner. Uscita.")
            return

        # Testa la connessione
        if not test_connection(scanner):
            print("Test di connessione fallito. Uscita.")
            return

        # Testa il proiettore
        test_projector()

        # Testa le telecamere
        test_cameras()

        # Testa la coppia stereo
        test_stereo_pair()

        print("\nTutti i test completati.")

    except KeyboardInterrupt:
        print("\nTest interrotto dall'utente.")
    except Exception as e:
        print(f"\nErrore durante i test: {e}")
    finally:
        # Disconnetti e ferma la discovery
        if client:
            if client.connected:
                client.disconnect()
            client.stop_discovery()

        # Chiudi eventuali finestre OpenCV
        cv2.destroyAllWindows()


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(description="Test del client UnLook")
    parser.add_argument("-o", "--output", default="output", help="Directory di output")
    args = parser.parse_args()

    global output_dir
    output_dir = args.output

    run_all_tests()


if __name__ == "__main__":
    main()