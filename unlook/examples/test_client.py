#!/usr/bin/env python3
"""
Script di test per la connessione al server UnLook.
Questo script verifica la connessione al server e testa le funzionalità
di base come il controllo del proiettore e la cattura di immagini.
Versione aggiornata con supporto per streaming migliorato.
"""

import os
import sys
import time
import logging
import argparse
import cv2
import numpy as np
import threading

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
streaming_active = False
frame_counters = {}
display_windows = {}


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


def on_stream_started(camera_id):
    """Callback chiamato quando uno stream viene avviato."""
    global streaming_active
    streaming_active = True
    print(f"Stream avviato per la camera {camera_id}")


def on_stream_stopped(camera_id):
    """Callback chiamato quando uno stream viene fermato."""
    global streaming_active
    streaming_active = False
    print(f"Stream fermato per la camera {camera_id}")


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


def stream_callback(frame, metadata):
    """Callback per lo streaming da una singola telecamera."""
    global frame_counters, display_windows

    camera_id = metadata.get("camera_id", "unknown")

    # Aggiorna il contatore dei frame
    if camera_id not in frame_counters:
        frame_counters[camera_id] = 0
    frame_counters[camera_id] += 1

    # Visualizza info ogni 30 frame per non intasare il terminale
    if frame_counters[camera_id] % 30 == 0:
        print(f"Camera {camera_id}: Frame #{frame_counters[camera_id]}, dimensione: {frame.shape[1]}x{frame.shape[0]}")

    # Visualizza il frame
    if camera_id not in display_windows:
        display_windows[camera_id] = f"Stream - Camera {camera_id}"
        cv2.namedWindow(display_windows[camera_id], cv2.WINDOW_NORMAL)
        cv2.resizeWindow(display_windows[camera_id], 640, 480)

    # Ridimensiona per visualizzazione
    display_frame = cv2.resize(frame, (640, 480))
    cv2.imshow(display_windows[camera_id], display_frame)
    cv2.waitKey(1)


def stereo_stream_callback(left_frame, right_frame, metadata):
    """Callback per lo streaming stereo."""
    global frame_counters

    # Aggiorna il contatore dei frame
    if "stereo" not in frame_counters:
        frame_counters["stereo"] = 0
    frame_counters["stereo"] += 1

    # Ottieni informazioni sul tempo di sincronizzazione
    sync_time_ms = metadata.get("sync_time", 0) * 1000  # in millisecondi

    # Visualizza info ogni 30 frame
    if frame_counters["stereo"] % 30 == 0:
        print(f"Stereo: Frame #{frame_counters['stereo']}, sync: {sync_time_ms:.1f}ms")

    # Crea un'immagine combinata
    stereo_image = np.hstack((left_frame, right_frame))

    # Ridimensiona per visualizzazione
    h, w = stereo_image.shape[:2]
    if w > 1280:
        scale = 1280 / w
        stereo_image = cv2.resize(stereo_image, (1280, int(h * scale)))

    # Aggiungi info sulla sincronizzazione
    cv2.putText(stereo_image, f"Sync: {sync_time_ms:.1f}ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Visualizza
    cv2.imshow("Stereo Stream", stereo_image)
    cv2.waitKey(1)


def test_streaming():
    """Testa lo streaming video migliorato."""
    global client, streaming_active, frame_counters, display_windows

    print("\n=== Test dello streaming video ===")

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

        # Resetta i contatori
        frame_counters = {}
        display_windows = {}

        # Test modalità di streaming singola telecamera
        camera_id = cameras[0]["id"]
        print(f"\nTest streaming singola telecamera ({camera_id})...")

        # Avvia lo streaming
        streaming_active = False
        if not client.stream.start(camera_id, stream_callback, fps=15):
            print("Errore nell'avvio dello streaming")
            return False

        # Attendi che l'utente prema un tasto
        print("Streaming avviato. Premi INVIO per continuare...")
        input()

        # Mostra statistiche
        stats = client.stream.get_stats(camera_id)
        print(f"Statistiche streaming: {stats}")
        print(f"Frame ricevuti: {frame_counters.get(camera_id, 0)}")

        # Ferma lo streaming
        client.stream.stop_stream(camera_id)
        print("Streaming fermato")

        # Test streaming multiplo se ci sono più telecamere
        if len(cameras) > 1:
            print("\nTest streaming multiplo...")

            # Avvia lo streaming su tutte le telecamere
            active_streams = []
            for camera in cameras:
                if client.stream.start(camera["id"], stream_callback, fps=10):
                    print(f"Streaming avviato per la camera {camera['id']}")
                    active_streams.append(camera["id"])
                else:
                    print(f"Errore nell'avvio dello streaming per la camera {camera['id']}")

            # Attendi che l'utente prema un tasto
            print("Streaming multiplo avviato. Premi INVIO per continuare...")
            input()

            # Mostra statistiche
            for camera_id in active_streams:
                stats = client.stream.get_stats(camera_id)
                print(f"Statistiche per camera {camera_id}: {stats}")
                print(f"Frame ricevuti: {frame_counters.get(camera_id, 0)}")

            # Ferma tutti gli stream
            client.stream.stop()
            print("Tutti gli stream fermati")

        # Test streaming stereo se disponibile
        if len(cameras) >= 2:
            print("\nTest streaming stereo...")

            # Avvia lo streaming stereo
            frame_counters["stereo"] = 0
            if client.stream.start_stereo_stream(stereo_stream_callback, fps=15):
                print("Streaming stereo avviato")

                # Attendi che l'utente prema un tasto
                print("Streaming stereo avviato. Premi INVIO per continuare...")
                input()

                # Mostra statistiche
                stats = client.stream.get_stats()
                print(f"Statistiche streaming: {stats}")
                print(f"Frame stereo ricevuti: {frame_counters.get('stereo', 0)}")

                # Ferma lo streaming
                client.stream.stop()
                print("Streaming stereo fermato")
            else:
                print("Errore nell'avvio dello streaming stereo")

        cv2.destroyAllWindows()
        return True

    except Exception as e:
        print(f"Errore durante il test dello streaming: {e}")
        return False
    finally:
        # Assicurati di fermare tutti gli stream
        client.stream.stop()
        cv2.destroyAllWindows()


def run_all_tests():
    """Esegue tutti i test."""
    global client

    # Crea il client
    client = UnlookClient(client_name="UnlookTestClient")

    # Registra i callback
    client.on_event(UnlookClientEvent.CONNECTED, on_connected)
    client.on_event(UnlookClientEvent.DISCONNECTED, on_disconnected)
    client.on_event(UnlookClientEvent.ERROR, on_error)
    client.on_event(UnlookClientEvent.STREAM_STARTED, on_stream_started)
    client.on_event(UnlookClientEvent.STREAM_STOPPED, on_stream_stopped)

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

        # Menù di test
        while True:
            print("\n=== Menu di Test ===")
            print("1. Test proiettore")
            print("2. Test telecamere")
            print("3. Test coppia stereo")
            print("4. Test streaming video (NUOVO)")
            print("5. Esegui tutti i test")
            print("0. Esci")

            choice = input("\nSeleziona un test (numero): ")

            if choice == "1":
                test_projector()
            elif choice == "2":
                test_cameras()
            elif choice == "3":
                test_stereo_pair()
            elif choice == "4":
                test_streaming()
            elif choice == "5":
                test_projector()
                test_cameras()
                test_stereo_pair()
                test_streaming()
            elif choice == "0":
                break
            else:
                print("Scelta non valida.")

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