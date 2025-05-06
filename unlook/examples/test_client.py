#!/usr/bin/env python3
"""
Script di test per lo streaming diretto di UnLook.
Questo script dimostra le capacità di streaming a bassa latenza e la
sincronizzazione con i pattern del proiettore.
"""

import os
import sys
import time
import logging
import argparse
import cv2
import numpy as np
import threading
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Aggiungi la directory radice al percorso di ricerca dei moduli
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)

try:
    # Importa dalla nuova struttura
    from unlook.client import UnlookClient
    from unlook.core import EventType
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
direct_streaming_active = False
frame_counters = {}
latency_history = deque(maxlen=100)  # Ultimi 100 valori di latenza
fps_history = deque(maxlen=100)  # Ultimi 100 valori di FPS
last_pattern_info = None  # Ultima informazione sul pattern del proiettore
sync_diff_values = deque(maxlen=100)  # Valori di differenza di sincronizzazione per stream stereo


# Callback per gli eventi del client
def on_connected(scanner):
    """Callback chiamato quando il client si connette a uno scanner."""
    print(f"Connesso a: {scanner.name} ({scanner.uuid})")


def on_disconnected():
    """Callback chiamato quando il client si disconnette da uno scanner."""
    print("Disconnesso dallo scanner")
    global streaming_active, direct_streaming_active
    streaming_active = False
    direct_streaming_active = False


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


def on_direct_stream_started(camera_id):
    """Callback chiamato quando uno stream diretto viene avviato."""
    global direct_streaming_active
    direct_streaming_active = True
    print(f"Stream diretto avviato per la camera {camera_id}")


def on_direct_stream_stopped(camera_id):
    """Callback chiamato quando uno stream diretto viene fermato."""
    global direct_streaming_active
    direct_streaming_active = False
    print(f"Stream diretto fermato per la camera {camera_id}")


def on_pattern_changed(pattern_info):
    """Callback chiamato quando cambia il pattern del proiettore."""
    global last_pattern_info
    last_pattern_info = pattern_info
    print(f"Pattern proiettore cambiato: {pattern_info.get('pattern_type')}")


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

        # Verifica capacità di streaming diretto
        direct_streaming = capabilities.get('direct_streaming', {})
        direct_stream_available = direct_streaming.get('available', False)
        direct_stream_max_fps = direct_streaming.get('max_fps', 60)
        direct_stream_low_latency = direct_streaming.get('low_latency', False)
        direct_stream_sync = direct_streaming.get('sync_capabilities', False)

        print(f"Proiettore disponibile: {projector_available}")
        print(f"Telecamere disponibili: {cameras_available} (conteggio: {cameras_count})")
        print(f"Streaming diretto disponibile: {direct_stream_available}")
        if direct_stream_available:
            print(f"  Max FPS: {direct_stream_max_fps}")
            print(f"  Bassa latenza: {direct_stream_low_latency}")
            print(f"  Sincronizzazione proiettore: {direct_stream_sync}")

        return True
    except Exception as e:
        print(f"Errore durante il test di connessione: {e}")
        return False


def test_direct_streaming():
    """Testa lo streaming video diretto."""
    global client, direct_streaming_active, frame_counters, latency_history, fps_history, last_pattern_info, sync_diff_values

    print("\n=== Test dello streaming video diretto ===")

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

        # Reset contatori e statistiche
        frame_counters = {}
        latency_history.clear()
        fps_history.clear()
        last_pattern_info = None
        sync_diff_values = []  # Reset differenze sync

        # Crea la finestra di visualizzazione
        cv2.namedWindow("Streaming Diretto", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Streaming Diretto", 1280, 720)

        # Avvia il grafico per le statistiche in tempo reale
        stats_plot = start_stats_plot()

        # Test streaming singola telecamera
        camera_id = cameras[0]["id"]
        print(f"\nTest streaming diretto singola telecamera ({camera_id})...")

        # Parametri per lo streaming diretto
        fps = 60  # FPS più alto per lo streaming diretto
        jpeg_quality = 85  # Qualità leggermente superiore
        sync_with_projector = True  # Attiva sincronizzazione con il proiettore
        low_latency = True  # Modalità a bassa latenza

        # Callback per lo streaming diretto
        def direct_stream_callback(frame, metadata):
            global frame_counters, latency_history, fps_history, last_pattern_info

            # Estrai informazioni dal metadata
            camera_id = metadata.get("camera_id", "unknown")
            timestamp = metadata.get("timestamp", time.time())
            latency = metadata.get("latency_ms", 0)
            is_sync_frame = metadata.get("is_sync_frame", False)
            pattern_info = metadata.get("pattern_info", None)

            # Se c'è l'informazione sul pattern, aggiornala
            if pattern_info:
                last_pattern_info = pattern_info

            # Aggiorna contatore frames
            if camera_id not in frame_counters:
                frame_counters[camera_id] = 0
            frame_counters[camera_id] += 1

            # Aggiorna cronologia latenza
            latency_history.append(latency)

            # Calcola FPS
            if len(latency_history) > 1:
                current_fps = 1000 / max(1, np.mean(np.diff(latency_history)))
                fps_history.append(min(current_fps, 120))  # Cap a 120 FPS per visualizzazione

            # Visualizza info ogni 30 frame per non intasare il terminale
            if frame_counters[camera_id] % 30 == 0:
                print(f"Camera {camera_id}: Frame #{frame_counters[camera_id]}, " +
                      f"Latenza: {latency:.1f}ms, " +
                      f"Sync: {is_sync_frame}, " +
                      f"Pattern: {last_pattern_info['pattern_type'] if last_pattern_info else 'N/A'}")

            # Aggiungi overlay con informazioni sulla latenza e sul pattern
            frame_with_info = frame.copy()

            # Aggiungi informazioni su latenza e FPS
            cv2.putText(
                frame_with_info,
                f"Frame: {frame_counters[camera_id]} | Latenza: {latency:.1f}ms | FPS: {fps_history[-1]:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if latency < 50 else (0, 165, 255) if latency < 100 else (0, 0, 255),
                2
            )

            # Aggiungi informazioni sul pattern del proiettore
            if last_pattern_info:
                pattern_text = f"Pattern: {last_pattern_info.get('pattern_type', 'unknown')}"
                cv2.putText(
                    frame_with_info,
                    pattern_text,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2
                )

            # Se è un frame di sincronizzazione, aggiungi un indicatore visivo
            if is_sync_frame:
                cv2.rectangle(frame_with_info, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)

            # Visualizza il frame
            cv2.imshow("Streaming Diretto", frame_with_info)
            key = cv2.waitKey(1)

            # Se premuto ESC, interrompi lo streaming
            if key == 27:
                client.stream.stop_direct_stream(camera_id)

        # Avvia lo streaming diretto
        direct_streaming_active = False
        print("Avvio streaming diretto...")
        if not client.stream.start_direct_stream(
                camera_id,
                direct_stream_callback,
                fps=fps,
                jpeg_quality=jpeg_quality,
                sync_with_projector=sync_with_projector,
                synchronization_pattern_interval=5,
                low_latency=low_latency
        ):
            print("Errore nell'avvio dello streaming diretto")
            return False

        print(f"""
Streaming diretto avviato con i seguenti parametri:
- FPS target: {fps}
- Qualità JPEG: {jpeg_quality}
- Sincronizzazione proiettore: {'Attiva' if sync_with_projector else 'Disattiva'}
- Modalità bassa latenza: {'Attiva' if low_latency else 'Disattiva'}
""")

        # Attendi che l'utente prema un tasto
        print("Streaming diretto avviato. Premi INVIO per continuare...")
        input()

        # Mostra statistiche
        stats = client.stream.get_direct_stream_stats(camera_id)
        print(f"Statistiche streaming diretto: {stats}")
        print(f"Frame ricevuti: {frame_counters.get(camera_id, 0)}")
        print(f"Latenza media: {np.mean(latency_history):.1f}ms")
        print(f"FPS effettivo: {stats.get('fps', 0):.1f}")

        # Ferma lo streaming
        client.stream.stop_direct_stream(camera_id)
        print("Streaming diretto fermato")

        # Chiudi il grafico
        stop_stats_plot(stats_plot)

        # Test streaming stereo diretto se disponibile
        if len(cameras) >= 2:
            print("\nTest streaming stereo diretto...")

            # Reset contatori e statistiche
            frame_counters = {}
            latency_history.clear()
            fps_history.clear()
            last_pattern_info = None
            sync_diff_values = []  # Reset differenze sync

            # Crea la finestra di visualizzazione
            cv2.namedWindow("Streaming Stereo Diretto", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Streaming Stereo Diretto", 1600, 600)

            # Avvia il grafico per le statistiche in tempo reale
            stats_plot = start_stats_plot()

            # Callback per lo streaming stereo diretto
            def stereo_direct_callback(left_frame, right_frame, metadata):
                global frame_counters, latency_history, fps_history, last_pattern_info, sync_diff_values

                # Estrai informazioni dal metadata
                timestamp = metadata.get("timestamp", time.time())
                sync_diff_ms = metadata.get("sync_diff_ms", 0)
                pattern_info = metadata.get("pattern_info", None)
                
                # Salva la differenza di sincronizzazione per calcoli futuri
                sync_diff_values.append(sync_diff_ms)

                # Se c'è l'informazione sul pattern, aggiornala
                if pattern_info:
                    last_pattern_info = pattern_info

                # Aggiorna contatore frames
                if "stereo" not in frame_counters:
                    frame_counters["stereo"] = 0
                frame_counters["stereo"] += 1

                # Aggiorna cronologia latenza
                left_latency = metadata.get("left", {}).get("latency_ms", 0)
                right_latency = metadata.get("right", {}).get("latency_ms", 0)
                avg_latency = (left_latency + right_latency) / 2
                latency_history.append(avg_latency)

                # Calcola FPS
                if len(latency_history) > 1:
                    current_fps = 1000 / max(1, np.mean(np.diff(latency_history)))
                    fps_history.append(min(current_fps, 120))  # Cap a 120 FPS per visualizzazione

                # Visualizza info ogni 30 frame per non intasare il terminale
                if frame_counters["stereo"] % 30 == 0:
                    print(f"Stereo: Frame #{frame_counters['stereo']}, " +
                          f"Latenza: {avg_latency:.1f}ms, " +
                          f"Sync diff: {sync_diff_ms:.1f}ms, " +
                          f"Pattern: {last_pattern_info['pattern_type'] if last_pattern_info else 'N/A'}")

                # Crea un'immagine combinata
                stereo_image = np.hstack((left_frame, right_frame))

                # Ridimensiona per visualizzazione
                h, w = stereo_image.shape[:2]
                if w > 1600:
                    scale = 1600 / w
                    stereo_image = cv2.resize(stereo_image, (1600, int(h * scale)))

                # Aggiungi informazioni su latenza, sync e FPS
                cv2.putText(
                    stereo_image,
                    f"Frame: {frame_counters['stereo']} | Latenza: {avg_latency:.1f}ms | Sync: {sync_diff_ms:.1f}ms | FPS: {fps_history[-1]:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if avg_latency < 50 else (0, 165, 255) if avg_latency < 100 else (0, 0, 255),
                    2
                )

                # Aggiungi informazioni sul pattern del proiettore
                if last_pattern_info:
                    pattern_text = f"Pattern: {last_pattern_info.get('pattern_type', 'unknown')}"
                    cv2.putText(
                        stereo_image,
                        pattern_text,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        2
                    )

                # Visualizza
                cv2.imshow("Streaming Stereo Diretto", stereo_image)
                key = cv2.waitKey(1)

                # Se premuto ESC, interrompi lo streaming
                if key == 27:
                    client.stream.stop()

            # Avvia lo streaming stereo diretto
            print("Avvio streaming stereo diretto...")
            if client.stream.start_direct_stereo_stream(
                    stereo_direct_callback,
                    fps=fps,
                    jpeg_quality=jpeg_quality,
                    sync_with_projector=sync_with_projector,
                    synchronization_pattern_interval=5,
                    low_latency=low_latency
            ):
                print("Streaming stereo diretto avviato")

                # Attendi che l'utente prema un tasto
                print("Streaming stereo diretto avviato. Premi INVIO per continuare...")
                input()

                # Mostra statistiche
                stats = client.stream.get_direct_stream_stats()
                print(f"Statistiche streaming diretto: {stats}")
                print(f"Frame stereo ricevuti: {frame_counters.get('stereo', 0)}")
                print(f"Latenza media: {np.mean(latency_history):.1f}ms")
                
                # In stereo_direct_callback abbiamo accesso a sync_diff_ms dai metadati
                # Calcoliamo la media delle differenze di sincronizzazione
                avg_sync_diff = np.mean(sync_diff_values) if len(sync_diff_values) > 0 else 0
                print(f"Differenza sync media: {avg_sync_diff:.1f}ms")

                # Ferma lo streaming
                client.stream.stop()
                print("Streaming stereo diretto fermato")

                # Chiudi il grafico
                stop_stats_plot(stats_plot)
            else:
                print("Errore nell'avvio dello streaming stereo diretto")

        cv2.destroyAllWindows()
        return True

    except KeyboardInterrupt:
        print("\nTest interrotto dall'utente")
        return False
    except Exception as e:
        print(f"Errore durante il test dello streaming diretto: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Assicurati di fermare tutti gli stream
        if client:
            try:
                client.stream.stop()
                # Cancella eventuali riferimenti ai callback per aiutare il GC
                client.stream.frame_callbacks.clear()
            except:
                pass
        cv2.destroyAllWindows()


def start_stats_plot():
    """Avvia un grafico per visualizzare statistiche in tempo reale."""
    # Crea una figura con due subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Statistiche Streaming Diretto in Tempo Reale')

    # Inizializza i dati
    x = np.arange(100)
    latency_line, = ax1.plot(x, np.zeros(100), 'r-', label='Latenza (ms)')
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, 100)
    ax1.set_ylabel('Latenza (ms)')
    ax1.set_xlabel('Frame')
    ax1.grid(True)
    ax1.legend()

    fps_line, = ax2.plot(x, np.zeros(100), 'b-', label='FPS')
    ax2.set_ylim(0, 120)
    ax2.set_xlim(0, 100)
    ax2.set_ylabel('FPS')
    ax2.set_xlabel('Frame')
    ax2.grid(True)
    ax2.legend()

    # Funzione di aggiornamento
    def update(frame):
        global latency_history, fps_history

        if len(latency_history) > 0:
            latency_data = list(latency_history)
            if len(latency_data) < 100:
                latency_data = [0] * (100 - len(latency_data)) + latency_data
            else:
                latency_data = latency_data[-100:]
            latency_line.set_ydata(latency_data)

            # Aggiorna i limiti dell'asse y in base ai dati
            max_latency = max(latency_data) * 1.1
            ax1.set_ylim(0, max(100, max_latency))

        if len(fps_history) > 0:
            fps_data = list(fps_history)
            if len(fps_data) < 100:
                fps_data = [0] * (100 - len(fps_data)) + fps_data
            else:
                fps_data = fps_data[-100:]
            fps_line.set_ydata(fps_data)

            # Aggiorna i limiti dell'asse y in base ai dati
            max_fps = max(fps_data) * 1.1
            ax2.set_ylim(0, max(60, max_fps))

        return latency_line, fps_line

    # Avvia l'animazione
    ani = FuncAnimation(fig, update, interval=100, blit=True)
    plt.tight_layout()
    plt.ion()  # Abilita la modalità interattiva
    plt.show(block=False)

    return ani


def stop_stats_plot(animation):
    """Ferma il grafico delle statistiche."""
    if animation:
        animation.event_source.stop()
    plt.close('all')


def run_tests():
    """Esegue i test specifici per lo streaming diretto."""
    global client

    # Crea il client
    client = UnlookClient(client_name="UnlookTestClient")

    # Registra i callback
    client.on(EventType.CONNECTED, on_connected)
    client.on(EventType.DISCONNECTED, on_disconnected)
    client.on(EventType.ERROR, on_error)
    client.on(EventType.DIRECT_STREAM_STARTED, on_direct_stream_started)
    client.on(EventType.DIRECT_STREAM_STOPPED, on_direct_stream_stopped)
    client.on(EventType.PATTERN_CHANGED, on_pattern_changed)

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
            print("\n=== Menu di Test Streaming Diretto ===")
            print("1. Test streaming diretto")
            print("2. Valuta prestazioni streaming standard vs diretto")
            print("3. Test diversi parametri di qualità e framerate")
            print("4. Test sincronizzazione proiettore-telecamera")
            print("0. Esci")

            choice = input("\nSeleziona un test (numero): ")

            if choice == "1":
                test_direct_streaming()
            elif choice == "2":
                compare_streaming_performance()
            elif choice == "3":
                test_streaming_parameters()
            elif choice == "4":
                test_projector_sync()
            elif choice == "0":
                break
            else:
                print("Scelta non valida.")

    except KeyboardInterrupt:
        print("\nTest interrotto dall'utente.")
    except Exception as e:
        print(f"\nErrore durante i test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Disconnetti e ferma la discovery
        if client:
            if client.connected:
                client.disconnect()
            client.stop_discovery()

        # Chiudi eventuali finestre OpenCV
        cv2.destroyAllWindows()


def compare_streaming_performance():
    """Confronta le prestazioni di latenza e framerate tra streaming standard e diretto."""
    global client, streaming_active, direct_streaming_active, frame_counters, latency_history, fps_history

    print("\n=== Confronto prestazioni streaming standard vs diretto ===")

    if not client.connected:
        print("Errore: client non connesso")
        return False

    try:
        # Ottieni la lista delle telecamere
        cameras = client.camera.get_cameras()

        if not cameras:
            print("Nessuna telecamera disponibile")
            return False

        camera_id = cameras[0]["id"]
        print(f"Utilizzo della telecamera: {camera_id}")

        # Parametri di test
        test_duration = 10  # secondi per ogni test

        # Crea finestra per i grafici
        plt.figure(figsize=(12, 8))

        # Test streaming standard
        print("\nTest streaming standard...")

        # Reset contatori e statistiche
        frame_counters = {}
        latency_history.clear()
        fps_history.clear()

        # Callback per lo streaming standard
        def standard_stream_callback(frame, metadata):
            global frame_counters, latency_history

            # Calcola latenza
            timestamp = metadata.get("timestamp", 0)
            latency = (time.time() - timestamp) * 1000  # milliseconds

            # Aggiorna statistiche
            latency_history.append(latency)

            # Incrementa contatore
            if "standard" not in frame_counters:
                frame_counters["standard"] = 0
            frame_counters["standard"] += 1

        # Avvia streaming standard
        print(f"Avvio streaming standard per {test_duration} secondi...")
        client.stream.start(camera_id, standard_stream_callback, fps=30)

        # Attendi per la durata del test
        time.sleep(test_duration)

        # Raccogli risultati
        standard_frames = frame_counters.get("standard", 0)
        standard_fps = standard_frames / test_duration
        standard_latency = np.mean(latency_history) if latency_history else 0
        standard_latency_std = np.std(latency_history) if latency_history else 0

        # Ferma streaming
        client.stream.stop_stream(camera_id)
        print(
            f"Streaming standard: {standard_frames} frames, {standard_fps:.1f} FPS, {standard_latency:.1f}±{standard_latency_std:.1f}ms latenza")

        # Salva statistiche per il confronto
        standard_latencies = list(latency_history)

        # Breve pausa
        time.sleep(1)

        # Test streaming diretto
        print("\nTest streaming diretto...")

        # Reset contatori e statistiche
        frame_counters = {}
        latency_history.clear()
        fps_history.clear()

        # Callback per lo streaming diretto
        def direct_stream_callback(frame, metadata):
            global frame_counters, latency_history

            # La latenza è già calcolata nel metadata per lo streaming diretto
            latency = metadata.get("latency_ms", 0)

            # Aggiorna statistiche
            latency_history.append(latency)

            # Incrementa contatore
            if "direct" not in frame_counters:
                frame_counters["direct"] = 0
            frame_counters["direct"] += 1

        # Avvia streaming diretto
        print(f"Avvio streaming diretto per {test_duration} secondi...")
        client.stream.start_direct_stream(
            camera_id,
            direct_stream_callback,
            fps=60,
            jpeg_quality=85,
            low_latency=True
        )

        # Attendi per la durata del test
        time.sleep(test_duration)

        # Raccogli risultati
        direct_frames = frame_counters.get("direct", 0)
        direct_fps = direct_frames / test_duration
        direct_latency = np.mean(latency_history) if latency_history else 0
        direct_latency_std = np.std(latency_history) if latency_history else 0

        # Ferma streaming
        client.stream.stop_direct_stream(camera_id)
        print(
            f"Streaming diretto: {direct_frames} frames, {direct_fps:.1f} FPS, {direct_latency:.1f}±{direct_latency_std:.1f}ms latenza")

        # Salva statistiche per il confronto
        direct_latencies = list(latency_history)

        # Visualizza risultati in grafici
        plt.figure(figsize=(14, 10))

        # Grafico 1: Confronto FPS
        plt.subplot(2, 2, 1)
        plt.bar(['Standard', 'Diretto'], [standard_fps, direct_fps])
        plt.ylabel('Frames per second')
        plt.title('Confronto FPS')
        plt.grid(axis='y')

        # Grafico 2: Confronto latenza media
        plt.subplot(2, 2, 2)
        plt.bar(['Standard', 'Diretto'], [standard_latency, direct_latency])
        plt.ylabel('Latenza media (ms)')
        plt.title('Confronto latenza media')
        plt.grid(axis='y')

        # Grafico 3: Distribuzione latenza standard
        plt.subplot(2, 2, 3)
        plt.hist(standard_latencies, bins=30, alpha=0.7)
        plt.xlabel('Latenza (ms)')
        plt.ylabel('Frequenza')
        plt.title('Distribuzione latenza - Streaming Standard')
        plt.grid(True)

        # Grafico 4: Distribuzione latenza diretto
        plt.subplot(2, 2, 4)
        plt.hist(direct_latencies, bins=30, alpha=0.7)
        plt.xlabel('Latenza (ms)')
        plt.ylabel('Frequenza')
        plt.title('Distribuzione latenza - Streaming Diretto')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'streaming_comparison.png'))
        plt.show()

        # Stampa conclusioni
        fps_improvement = (direct_fps / standard_fps - 1) * 100
        latency_improvement = (1 - direct_latency / standard_latency) * 100 if standard_latency > 0 else 0

        print("\n=== Risultati del confronto ===")
        print(f"- FPS: Miglioramento del {fps_improvement:.1f}% con streaming diretto")
        print(f"- Latenza: Riduzione del {latency_improvement:.1f}% con streaming diretto")
        print(f"- File di output salvato in: {os.path.join(output_dir, 'streaming_comparison.png')}")

        return True

    except Exception as e:
        print(f"Errore durante il confronto: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Assicurati di fermare tutti gli stream
        if client:
            client.stream.stop()
            try:
                client.stream.stop_direct_stream(camera_id)
            except:
                pass


def test_streaming_parameters():
    """Testa l'effetto di diversi parametri sulla qualità e prestazioni dello streaming."""
    global client, frame_counters, latency_history, fps_history

    print("\n=== Test parametri streaming diretto ===")

    if not client.connected:
        print("Errore: client non connesso")
        return False

    try:
        # Ottieni la lista delle telecamere
        cameras = client.camera.get_cameras()

        if not cameras:
            print("Nessuna telecamera disponibile")
            return False

        camera_id = cameras[0]["id"]
        print(f"Utilizzo della telecamera: {camera_id}")

        # Parametri da testare
        qualities = [50, 75, 90]
        framerates = [30, 60, 90]
        low_latency_modes = [True, False]

        # Durata di ogni test
        test_duration = 5  # secondi

        # Dati per il confronto
        results = []

        # Crea finestra per visualizzazione
        cv2.namedWindow("Test Parametri", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Test Parametri", 960, 540)

        # Esegui i test
        for quality in qualities:
            for fps in framerates:
                for low_latency in low_latency_modes:
                    print(f"\nTest con: Qualità={quality}, FPS={fps}, Bassa latenza={low_latency}")

                    # Reset contatori e statistiche
                    frame_counters = {}
                    latency_history.clear()
                    fps_history.clear()

                    # Callback per lo streaming
                    def test_params_callback(frame, metadata):
                        global frame_counters, latency_history

                        # La latenza è già calcolata nel metadata per lo streaming diretto
                        latency = metadata.get("latency_ms", 0)

                        # Aggiorna statistiche
                        latency_history.append(latency)

                        # Incrementa contatore
                        if "test" not in frame_counters:
                            frame_counters["test"] = 0
                        frame_counters["test"] += 1

                        # Aggiungi overlay con informazioni
                        display_frame = frame.copy()
                        cv2.putText(
                            display_frame,
                            f"Qualità: {quality}, FPS: {fps}, Bassa latenza: {low_latency}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            display_frame,
                            f"Latenza: {latency:.1f}ms, Frame: {frame_counters['test']}",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                        # Visualizza
                        cv2.imshow("Test Parametri", display_frame)
                        cv2.waitKey(1)

                    # Avvia streaming
                    success = client.stream.start_direct_stream(
                        camera_id,
                        test_params_callback,
                        fps=fps,
                        jpeg_quality=quality,
                        low_latency=low_latency
                    )

                    if not success:
                        print(f"Errore nell'avvio dello streaming con i parametri specificati")
                        continue

                    # Attendi per la durata del test
                    start_time = time.time()
                    while time.time() - start_time < test_duration:
                        time.sleep(0.1)

                        # Controllo se ci sono key press
                        key = cv2.waitKey(1)
                        if key == 27:  # ESC
                            break

                    # Raccogli risultati
                    actual_frames = frame_counters.get("test", 0)
                    actual_fps = actual_frames / test_duration
                    actual_latency = np.mean(latency_history) if latency_history else 0
                    actual_jitter = np.std(latency_history) if len(latency_history) > 1 else 0

                    # Aggiungi ai risultati
                    results.append({
                        "quality": quality,
                        "target_fps": fps,
                        "low_latency": low_latency,
                        "actual_fps": actual_fps,
                        "latency_ms": actual_latency,
                        "jitter_ms": actual_jitter
                    })

                    # Ferma streaming
                    client.stream.stop_direct_stream(camera_id)

                    print(
                        f"Risultati: {actual_fps:.1f} FPS, {actual_latency:.1f}ms latenza, {actual_jitter:.1f}ms jitter")

        # Chiudi finestra
        cv2.destroyAllWindows()

        # Analizza risultati
        print("\n=== Risultati dei test ===")

        # Ordina per latenza (più bassa prima)
        results_by_latency = sorted(results, key=lambda x: x["latency_ms"])

        print("\nConfigurazione con latenza più bassa:")
        best_latency = results_by_latency[0]
        print(f"- Qualità: {best_latency['quality']}")
        print(f"- FPS target: {best_latency['target_fps']}")
        print(f"- Modalità bassa latenza: {best_latency['low_latency']}")
        print(f"- FPS effettivo: {best_latency['actual_fps']:.1f}")
        print(f"- Latenza: {best_latency['latency_ms']:.1f}ms")
        print(f"- Jitter: {best_latency['jitter_ms']:.1f}ms")

        # Ordina per FPS (più alto prima)
        results_by_fps = sorted(results, key=lambda x: x["actual_fps"], reverse=True)

        print("\nConfigurazione con FPS più alto:")
        best_fps = results_by_fps[0]
        print(f"- Qualità: {best_fps['quality']}")
        print(f"- FPS target: {best_fps['target_fps']}")
        print(f"- Modalità bassa latenza: {best_fps['low_latency']}")
        print(f"- FPS effettivo: {best_fps['actual_fps']:.1f}")
        print(f"- Latenza: {best_fps['latency_ms']:.1f}ms")
        print(f"- Jitter: {best_fps['jitter_ms']:.1f}ms")

        # Trova il miglior equilibrio
        balance_score = [(r["actual_fps"] / max(r["latency_ms"], 1)) for r in results]
        best_balance_idx = balance_score.index(max(balance_score))
        best_balance = results[best_balance_idx]

        print("\nMiglior equilibrio tra latenza e FPS:")
        print(f"- Qualità: {best_balance['quality']}")
        print(f"- FPS target: {best_balance['target_fps']}")
        print(f"- Modalità bassa latenza: {best_balance['low_latency']}")
        print(f"- FPS effettivo: {best_balance['actual_fps']:.1f}")
        print(f"- Latenza: {best_balance['latency_ms']:.1f}ms")
        print(f"- Jitter: {best_balance['jitter_ms']:.1f}ms")

        # Crea grafici di confronto
        plt.figure(figsize=(15, 10))

        # Prepara dati per grafici
        labels = [f"Q{r['quality']}-FPS{r['target_fps']}-LL{int(r['low_latency'])}" for r in results]
        latencies = [r["latency_ms"] for r in results]
        actual_fps = [r["actual_fps"] for r in results]
        jitters = [r["jitter_ms"] for r in results]

        # Grafico 1: Confronto latenza
        plt.subplot(2, 2, 1)
        plt.bar(range(len(labels)), latencies)
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.ylabel('Latenza (ms)')
        plt.title('Confronto latenza per configurazione')
        plt.grid(axis='y')

        # Grafico 2: Confronto FPS
        plt.subplot(2, 2, 2)
        plt.bar(range(len(labels)), actual_fps)
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.ylabel('FPS effettivo')
        plt.title('Confronto FPS per configurazione')
        plt.grid(axis='y')

        # Grafico 3: Confronto jitter
        plt.subplot(2, 2, 3)
        plt.bar(range(len(labels)), jitters)
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.ylabel('Jitter (ms)')
        plt.title('Confronto jitter per configurazione')
        plt.grid(axis='y')

        # Grafico 4: Rapporto FPS/latenza (più alto è meglio)
        plt.subplot(2, 2, 4)
        plt.bar(range(len(labels)), [fps / max(lat, 1) for fps, lat in zip(actual_fps, latencies)])
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.ylabel('FPS/Latenza (più alto è meglio)')
        plt.title('Rapporto FPS/Latenza per configurazione')
        plt.grid(axis='y')

        plt.tight_layout()
        output_file = os.path.join(output_dir, 'streaming_parameters_comparison.png')
        plt.savefig(output_file)
        plt.show()

        print(f"\nGrafico salvato in: {output_file}")

        return True

    except Exception as e:
        print(f"Errore durante il test dei parametri: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Assicurati di fermare tutti gli stream
        if client:
            client.stream.stop()
            try:
                client.stream.stop_direct_stream(camera_id)
            except:
                pass
            cv2.destroyAllWindows()


def test_projector_sync():
    """Testa la sincronizzazione tra proiettore e telecamera."""
    global client, output_dir

    print("\n=== Test sincronizzazione proiettore-telecamera ===")

    if not client.connected:
        print("Errore: client non connesso")
        return False

    try:
        # Ottieni la lista delle telecamere
        cameras = client.camera.get_cameras()
        if not cameras:
            print("Nessuna telecamera disponibile")
            return False

        camera_id = cameras[0]["id"]

        # Parametri di test
        num_patterns = 5
        sync_delay = 50  # ms tra cambio pattern e acquisizione
        pattern_types = ["solid_field", "checkerboard", "grid", "horizontal_lines", "vertical_lines"]
        pattern_colors = ["White", "Red", "Green", "Blue", "Cyan"]

        # Metti il proiettore in modalità test pattern
        if not client.projector.set_test_pattern_mode():
            print("Errore: impossibile impostare la modalità test pattern")
            return False

        print("Avvio test di sincronizzazione...")

        # Array per memorizzare i tempi di sincronizzazione
        sync_times = []

        # Avvia lo streaming diretto con sincronizzazione proiettore
        sync_callback = lambda frame, metadata: process_sync_frame(frame, metadata, sync_times)

        if hasattr(client.stream, 'start_direct_stream'):
            # Utilizzo streaming diretto se disponibile
            print("Utilizzo streaming diretto per il test di sincronizzazione")
            success = client.stream.start_direct_stream(
                camera_id,
                sync_callback,
                fps=30,
                sync_with_projector=True
            )
        else:
            # Fallback allo streaming normale
            print("Utilizzo streaming standard per il test di sincronizzazione")
            success = client.stream.start(
                camera_id,
                sync_callback,
                fps=30
            )

        if not success:
            print("Errore nell'avvio dello streaming")
            return False

        # Ciclo di test con diversi pattern
        for i in range(num_patterns):
            pattern_type = pattern_types[i % len(pattern_types)]
            color = pattern_colors[i % len(pattern_colors)]

            print(f"Test pattern {i + 1}/{num_patterns}: {pattern_type} {color}")

            # Proietta il pattern
            if pattern_type == "solid_field":
                client.projector.show_solid_field(color)
            elif pattern_type == "checkerboard":
                client.projector.show_checkerboard(foreground_color=color)
            elif pattern_type == "grid":
                client.projector.show_grid(foreground_color=color)
            elif pattern_type == "horizontal_lines":
                client.projector.show_horizontal_lines(foreground_color=color)
            elif pattern_type == "vertical_lines":
                client.projector.show_vertical_lines(foreground_color=color)

            # Attendi la sincronizzazione
            time.sleep(sync_delay / 1000.0)

            # Cattura un'immagine per verificare visivamente
            image = client.camera.capture(camera_id)
            if image is not None:
                filename = os.path.join(output_dir, f"sync_test_{i + 1}_pattern_{pattern_type}_{color}.jpg")
                cv2.imwrite(filename, image)
                print(f"Immagine salvata: {filename}")

            # Attendi un po' prima del pattern successivo
            time.sleep(0.5)

        # Ferma lo streaming
        if hasattr(client.stream, 'stop_direct_stream'):
            client.stream.stop_direct_stream(camera_id)
        else:
            client.stream.stop_stream(camera_id)

        # Calcola la differenza di sincronizzazione media
        if sync_times:
            sync_diff_ms = sum(sync_times) / len(sync_times)
            print(f"Differenza sync media: {sync_diff_ms:.1f}ms")
            print(f"Numero di campioni: {len(sync_times)}")

            # Valuta la qualità della sincronizzazione
            if sync_diff_ms < 10:
                print("Sincronizzazione eccellente (<10ms)")
            elif sync_diff_ms < 20:
                print("Sincronizzazione buona (<20ms)")
            elif sync_diff_ms < 50:
                print("Sincronizzazione accettabile (<50ms)")
            else:
                print("Sincronizzazione debole (>50ms)")
        else:
            sync_diff_ms = float('inf')
            print("Nessun dato di sincronizzazione raccolto")

        # Metti il proiettore in standby
        client.projector.set_standby()

        # Mostra grafici di sincronizzazione se ci sono dati
        if sync_times and len(sync_times) > 1:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(sync_times, 'b-', label='Differenza di sincronizzazione (ms)')
                plt.axhline(y=sync_diff_ms, color='r', linestyle='--', label=f'Media: {sync_diff_ms:.1f}ms')
                plt.title('Sincronizzazione Proiettore-Telecamera')
                plt.xlabel('Campione')
                plt.ylabel('Differenza (ms)')
                plt.legend()
                plt.grid(True)

                # Salva il grafico
                plot_filename = os.path.join(output_dir, "sync_performance.png")
                plt.savefig(plot_filename)
                print(f"Grafico salvato: {plot_filename}")
                plt.close()
            except Exception as e:
                print(f"Impossibile creare il grafico: {e}")

        return True

    except Exception as e:
        print(f"Errore durante il test di sincronizzazione: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    finally:
        # Assicurati di fermare lo streaming
        client.stream.stop()
        # Metti il proiettore in standby
        client.projector.set_standby()


def process_sync_frame(frame, metadata, sync_times):
    """Callback per processare un frame e calcolare la sincronizzazione."""
    # Estrai l'informazione di sincronizzazione dai metadati
    if metadata and 'projector_sync_time' in metadata:
        sync_time = metadata['projector_sync_time'] * 1000  # converti in ms
        sync_times.append(sync_time)

        # Mostra informazioni ogni 10 frame per non intasare il terminale
        if len(sync_times) % 10 == 0:
            avg = sum(sync_times[-10:]) / 10
            print(f"Sync #{len(sync_times)}: {sync_time:.1f}ms, Media ultimi 10: {avg:.1f}ms")


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(description="Test dello streaming diretto di UnLook")
    parser.add_argument("-o", "--output", default="output", help="Directory di output")
    args = parser.parse_args()

    global output_dir
    output_dir = args.output

    run_tests()


if __name__ == "__main__":
    main()