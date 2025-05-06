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

# Aggiungi la directory corrente al percorso di ricerca dei moduli
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Funzione principale per l'avvio del server."""
    parser = argparse.ArgumentParser(description="Server UnLook per scanner 3D")
    parser.add_argument("--config", help="Percorso del file di configurazione JSON")
    args = parser.parse_args()

    try:
        # Importa il modulo server - questo import è sicuro con la nuova struttura
        from unlook.server import UnlookServer

        # Carica la configurazione
        config_path = args.config or "unlook/unlook_config.json"

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
                    "stream_port": 5556
                }
            }

        # Estrai le configurazioni del server
        server_config = config.get("server", {})

        # Crea e avvia il server
        server = UnlookServer(
            name=server_config.get("name", "UnLookScanner"),
            control_port=server_config.get("control_port", 5555),
            stream_port=server_config.get("stream_port", 5556),
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