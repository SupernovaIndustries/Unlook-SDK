#!/usr/bin/env python3
"""
Script di bootstrap per il server UnLook.
Questo script evita problemi di importazione circolare avviando il server
in un modo completamente indipendente.
"""

import os
import sys
import time
import json
import signal
import logging
import argparse
from pathlib import Path

# Configura il logging di base
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ServerBoot")

# Fix: Add the root directory (Unlook-SDK) to the Python path for proper import resolution
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

# Print for debugging
print(f"Adding to path: {root_dir}")
print(f"Full sys.path: {sys.path}")

# Set a flag in the global namespace to inform the unlook module
# that we're running in server-only mode and should not import client modules
import builtins
builtins._SERVER_ONLY_MODE = True


def main():
    """Funzione principale per l'avvio del server."""
    parser = argparse.ArgumentParser(description="Server UnLook per scanner 3D")
    parser.add_argument("--config", help="Percorso del file di configurazione JSON")
    args = parser.parse_args()

    try:
        # Using server-only mode (set earlier), we can directly import just the server module
        # without the risk of circular imports with client modules
        from unlook.server.scanner import UnlookServer

        # Carica la configurazione
        config_path = args.config or os.path.join(os.path.dirname(__file__), "unlook_config.json")

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configurazione caricata da {config_path}")
        else:
            logger.warning(f"File di configurazione {config_path} non trovato, utilizzo configurazione predefinita")
            config = {
                "server": {
                    "name": "UnLookScanner",
                    "control_port": 5555,
                    "stream_port": 5556,
                    "direct_stream_port": 5557  # Added default for direct streaming
                }
            }

        # Estrai le configurazioni del server
        server_config = config.get("server", {})

        # Crea e avvia il server
        server = UnlookServer(
            name=server_config.get("name", "UnLookScanner"),
            control_port=server_config.get("control_port", 5555),
            stream_port=server_config.get("stream_port", 5556),
            direct_stream_port=server_config.get("direct_stream_port", 5557),  # Added direct stream port
            scanner_uuid=server_config.get("scanner_uuid"),
            auto_start=True
        )

        logger.info("Server avviato con successo")

        # Configura il gestore di segnali per l'arresto pulito
        def signal_handler(sig, frame):
            logger.info("Segnale di arresto ricevuto, terminazione del server...")
            server.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Mantieni il processo in esecuzione
        while True:
            time.sleep(1)

    except Exception as e:
        logger.error(f"Errore durante l'avvio del server: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()