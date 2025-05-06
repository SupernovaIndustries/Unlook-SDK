#!/usr/bin/env python3
"""
Server UnLook per scanner 3D - Script di avvio automatico

Questo script avvia il server UnLook sul Raspberry Pi e gestisce
il controllo del proiettore DLP342X e delle telecamere PiCamera.
È progettato per essere eseguito all'avvio del sistema.

Uso:
    python3 unlook_server.py [--config CONFIG_FILE]
"""

import os
import sys
import time
import json
import signal
import logging
import argparse
import threading
from pathlib import Path
import RPi.GPIO as GPIO

# Aggiungi la directory corrente al percorso di ricerca dei moduli
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Prima importa i moduli di base senza dipendenze
try:
    # Importa solo ciò che è necessario, evitando importazioni circolari
    from unlook.server.scanner import UnlookServer
    from unlook.server.projector.dlp342x import OperatingMode, Color
except ImportError as e:
    print(f"ERRORE: Impossibile importare i moduli UnLook: {e}")
    print("Assicurati che l'SDK UnLook sia installato correttamente")
    sys.exit(1)
# Configurazione predefinita
DEFAULT_CONFIG = {
    "server": {
        "name": "UnLookScanner",
        "control_port": 5555,
        "stream_port": 5556,
        "scanner_uuid": None,  # Generato automaticamente se None
        "log_level": "INFO",
        "log_file": "logs/unlook_server.log"
    },
    "projector": {
        "i2c_bus": 3,
        "i2c_address": "0x1B",
        "standby_on_start": False,
        "test_on_start": False
    },
    "camera": {
        "default_resolution": [1920, 1080],
        "default_fps": 30
    },
    "gpio": {
        "status_led_pin": 25,
        "error_led_pin": 24,
        "reset_button_pin": 23,
        "use_gpio": True
    }
}

# Inizializzazione variabili globali
server = None
terminate_flag = False
config = DEFAULT_CONFIG.copy()
status_thread = None


def setup_logging(log_level, log_file=None):
    """Configura il sistema di logging."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configurazione base
    logging.basicConfig(
        level=numeric_level,
        format=log_format
    )

    # Se è specificato un file di log, aggiungi un FileHandler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception as e:
                print(f"AVVISO: Impossibile creare la directory di log {log_dir}: {e}")
                print(f"I log verranno scritti solo sulla console")
                return

        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger('').addHandler(file_handler)
        except Exception as e:
            print(f"AVVISO: Impossibile creare il file di log {log_file}: {e}")
            print(f"I log verranno scritti solo sulla console")


def load_config(config_file):
    """Carica la configurazione da file."""
    global config

    if not config_file or not os.path.exists(config_file):
        logging.info(f"File di configurazione non specificato o non esistente, utilizzo dei valori predefiniti")
        return

    try:
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)

        # Aggiorna la configurazione ricorsivamente
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        config = update_dict(config, loaded_config)
        logging.info(f"Configurazione caricata da {config_file}")
    except Exception as e:
        logging.error(f"Errore durante il caricamento della configurazione: {e}")
        logging.info("Utilizzo dei valori predefiniti")


def setup_gpio():
    """Configura i pin GPIO."""
    if not config["gpio"]["use_gpio"]:
        logging.info("GPIO disabilitato nella configurazione")
        return

    try:
        # Disabilita gli avvertimenti per i canali già in uso
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)

        # LED di stato
        status_pin = config["gpio"]["status_led_pin"]
        error_pin = config["gpio"]["error_led_pin"]
        reset_pin = config["gpio"]["reset_button_pin"]

        GPIO.setup(status_pin, GPIO.OUT)
        GPIO.setup(error_pin, GPIO.OUT)
        GPIO.setup(reset_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # Imposta LED iniziali
        GPIO.output(status_pin, GPIO.LOW)
        GPIO.output(error_pin, GPIO.LOW)

        # Gestisci gli eventi con try-except per essere resilienti
        try:
            # Aggiunge callback per il pulsante di reset
            GPIO.add_event_detect(reset_pin, GPIO.FALLING,
                                callback=reset_button_callback,
                                bouncetime=300)
        except Exception as e:
            logging.warning(f"Impossibile aggiungere il rilevamento di eventi GPIO: {e}")

        logging.info("GPIO configurato correttamente")
    except Exception as e:
        logging.error(f"Errore durante la configurazione GPIO: {e}")
        config["gpio"]["use_gpio"] = False

def reset_button_callback(channel):
    """Gestisce la pressione del pulsante di reset."""
    logging.info("Pulsante di reset premuto, riavvio del server in corso...")

    global server

    # Avvia il riavvio in un thread separato per non bloccare l'evento GPIO
    restart_thread = threading.Thread(target=restart_server)
    restart_thread.daemon = True
    restart_thread.start()


def restart_server():
    """Riavvia il server."""
    global server

    try:
        if server:
            logging.info("Arresto del server in corso...")
            server.stop()
            server = None

        time.sleep(1)  # Breve pausa per assicurarsi che tutto sia pulito

        logging.info("Avvio del nuovo server in corso...")
        start_server()
    except Exception as e:
        logging.error(f"Errore durante il riavvio del server: {e}")
        set_error_state(True)


def set_status_led(state):
    """Imposta il LED di stato."""
    if not config["gpio"]["use_gpio"]:
        return

    try:
        GPIO.output(config["gpio"]["status_led_pin"], GPIO.HIGH if state else GPIO.LOW)
    except Exception as e:
        logging.error(f"Errore durante l'impostazione del LED di stato: {e}")


def set_error_state(error_state):
    """Imposta il LED di errore."""
    if not config["gpio"]["use_gpio"]:
        return

    try:
        GPIO.output(config["gpio"]["error_led_pin"], GPIO.HIGH if error_state else GPIO.LOW)
    except Exception as e:
        logging.error(f"Errore durante l'impostazione del LED di errore: {e}")


def status_led_blink_thread():
    """Thread per far lampeggiare il LED di stato durante il funzionamento."""
    while not terminate_flag:
        if server and server.running:
            set_status_led(True)
            time.sleep(0.9)
            set_status_led(False)
            time.sleep(0.1)
        else:
            # Se il server non è in esecuzione, lampeggio rapido
            set_status_led(True)
            time.sleep(0.1)
            set_status_led(False)
            time.sleep(0.1)


def test_projector():
    """Testa il proiettore mostrando una sequenza di pattern."""
    global server

    if not server or not server.projector:
        logging.error("Impossibile testare il proiettore: server o proiettore non disponibile")
        return False

    try:
        logging.info("Test del proiettore in corso...")

        # Imposta modalità test pattern
        server.projector.set_operating_mode(OperatingMode.TestPatternGenerator)
        time.sleep(0.5)

        # Mostra una sequenza di pattern
        patterns = [
            (Color.White, "bianco"),
            (Color.Red, "rosso"),
            (Color.Green, "verde"),
            (Color.Blue, "blu")
        ]

        for color, name in patterns:
            logging.info(f"Proiezione pattern {name}")
            server.projector.generate_solid_field(color)
            time.sleep(1)

        # Mostra un pattern griglia
        server.projector.generate_grid(
            Color.Black, Color.White,
            4, 20, 4, 20
        )
        time.sleep(1)

        # Metti in standby
        server.projector.generate_solid_field(Color.Black)
        logging.info("Pattern nero proiettato come alternativa allo standby")

        logging.info("Test del proiettore completato con successo")
        return True
    except Exception as e:
        logging.error(f"Errore durante il test del proiettore: {e}")
        set_error_state(True)
        return False


def test_cameras():
    """Testa le telecamere cercando di ottenere la lista e catturare un'immagine."""
    global server

    if not server or not server.camera_manager:
        logging.error("Impossibile testare le telecamere: server non disponibile")
        return False

    try:
        logging.info("Test delle telecamere in corso...")

        # Ottieni la lista delle telecamere
        cameras = server.camera_manager.get_cameras()

        if not cameras:
            logging.warning("Nessuna telecamera rilevata")
            return False

        logging.info(f"Rilevate {len(cameras)} telecamere")

        # Tenta di catturare un'immagine dalla prima telecamera
        for camera_id in cameras:
            logging.info(f"Test della telecamera {camera_id}")

            # Apri la telecamera
            if not server.camera_manager.open_camera(camera_id):
                logging.error(f"Impossibile aprire la telecamera {camera_id}")
                continue

            # Cattura un'immagine
            image = server.camera_manager.capture_image(camera_id)

            if image is not None:
                logging.info(f"Immagine catturata correttamente dalla telecamera {camera_id}")

                # Chiudi la telecamera
                server.camera_manager.close_camera(camera_id)
            else:
                logging.error(f"Impossibile catturare un'immagine dalla telecamera {camera_id}")

        logging.info("Test delle telecamere completato")
        return True
    except Exception as e:
        logging.error(f"Errore durante il test delle telecamere: {e}")
        set_error_state(True)
        return False


def signal_handler(sig, frame):
    """Gestisce i segnali di terminazione."""
    global terminate_flag, server

    logging.info("Segnale di arresto ricevuto, terminazione del server in corso...")
    terminate_flag = True

    if server and server.projector:
        try:
            # Proietta nero invece di standby
            server.projector.set_operating_mode(OperatingMode.TestPatternGenerator)
            time.sleep(0.5)
            server.projector.generate_solid_field(Color.Black)
        except:
            pass

    if server:
        server.stop()

    sys.exit(0)


def start_server():
    """Avvia il server UnLook."""
    global server, config

    # Reset dello stato di errore
    set_error_state(False)

    try:
        server_config = config["server"]
        projector_config = config["projector"]

        # Converte l'indirizzo I2C da stringa a int
        i2c_address = int(projector_config["i2c_address"], 0)

        logging.info(f"Avvio del server UnLook (nome: {server_config['name']}, "
                     f"porta controllo: {server_config['control_port']}, "
                     f"porta streaming: {server_config['stream_port']})")

        # Crea il server
        server = UnlookServer(
            name=server_config["name"],
            control_port=server_config["control_port"],
            stream_port=server_config["stream_port"],
            scanner_uuid=server_config["scanner_uuid"],
            auto_start=True  # Avvia automaticamente il server
        )

        # Verifica che il server sia in esecuzione
        if not server.running:
            logging.error("Il server non è stato avviato correttamente")
            set_error_state(True)
            return False

        logging.info("Server UnLook avviato con successo")

        # Test del proiettore se richiesto
        if projector_config["test_on_start"]:

            time.sleep(1)  # Pausa prima del test
            test_projector()
        elif projector_config["standby_on_start"] and server.projector:
            time.sleep(1)  # Pausa prima di cambiare pattern
            logging.info("Proiezione pattern nero come alternativa allo standby")
            server.projector.set_operating_mode(OperatingMode.TestPatternGenerator)
            time.sleep(1)
            server.projector.generate_solid_field(Color.Black)

        # Test delle telecamere
        test_cameras()

        return True
    except Exception as e:
        logging.error(f"Errore durante l'avvio del server: {e}")
        set_error_state(True)
        return False


def cleanup():
    """Pulisce le risorse prima dell'uscita."""
    global server, terminate_flag

    terminate_flag = True

    if server:
        try:
            server.stop()
        except Exception as e:
            logging.error(f"Errore durante l'arresto del server: {e}")

    if server and server.projector:
        try:
            # Proietta nero invece di standby
            server.projector.set_operating_mode(OperatingMode.TestPatternGenerator)
            time.sleep(0.5)
            server.projector.generate_solid_field(Color.Black)
            logging.info("Proiettore impostato su nero durante la chiusura")
        except Exception as e:
            logging.error(f"Errore durante l'impostazione del pattern nero sul proiettore: {e}")

    if config["gpio"]["use_gpio"]:
        try:
            GPIO.cleanup()
        except Exception as e:
            logging.error(f"Errore durante la pulizia GPIO: {e}")


def main():
    """Funzione principale."""
    global status_thread

    # Analizza gli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Server UnLook per scanner 3D")
    parser.add_argument("--config", help="Percorso del file di configurazione JSON")
    args = parser.parse_args()

    try:
        # Carica la configurazione
        load_config(args.config)

        # Configura il logging
        setup_logging(
            config["server"]["log_level"],
            config["server"]["log_file"]
        )

        # Registra i gestori di segnale
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Configura GPIO
        setup_gpio()

        # Avvia il thread del LED di stato
        if config["gpio"]["use_gpio"]:
            status_thread = threading.Thread(target=status_led_blink_thread)
            status_thread.daemon = True
            status_thread.start()

        # Avvia il server
        success = start_server()

        # Se il server è stato avviato con successo, entra in un loop di controllo
        if success:
            logging.info("Server UnLook in esecuzione, premi Ctrl+C per terminare")

            # Loop principale che controlla lo stato del server
            while not terminate_flag:
                time.sleep(10)  # Controlla ogni 10 secondi

                if server and not server.running:
                    logging.error("Il server si è arrestato inaspettatamente, tentativo di riavvio...")
                    restart_server()
        else:
            logging.error("Impossibile avviare il server, uscita")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Errore fatale: {e}")
        set_error_state(True)
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()