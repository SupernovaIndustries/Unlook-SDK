"""
Esempio di utilizzo programmatico dell'SDK UnLook.
Questo script mostra come utilizzare l'SDK in uno script personalizzato
per automatizzare il processo di scansione e analisi.
"""

import logging
import os
import time
import sys
import numpy as np
import cv2

from unlook.client import (
    UnlookClient, UnlookClientEvent,
    PatternDirection, PatternType,
    ScanProcessor, ModelExporter,
    CalibrationData
)

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Parametri di configurazione
CONFIG = {
    "scanner_name": "UnLookScanner",  # Nome dello scanner da cercare
    "calibration_file": "calibration.json",  # File di calibrazione
    "output_dir": "output",  # Directory di output
    "scan_count": 3,  # Numero di scansioni da eseguire
    "pause_between_scans": 5,  # Pausa tra le scansioni (secondi)
    "auto_process": True,  # Elabora automaticamente le scansioni
    "pattern_type": PatternType.GRAY_CODE,  # Tipo di pattern
    "pattern_direction": PatternDirection.BOTH,  # Direzione dei pattern
    "save_formats": ["ply", "obj", "xyz"],  # Formati di salvataggio
    "connection_timeout": 10000,  # Timeout di connessione (ms)
    "discovery_timeout": 5  # Timeout per la discovery (secondi)
}


class UnlookAutomatedScanner:
    """
    Classe per l'automazione della scansione con UnLook.
    """

    def __init__(self, config=None):
        """
        Inizializza lo scanner automatizzato.

        Args:
            config: Configurazione personalizzata
        """
        self.config = CONFIG.copy()
        if config:
            self.config.update(config)

        # Crea il client
        self.client = UnlookClient(client_name="UnlookAutomated")

        # Directory di output
        self.output_dir = self.config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # Altri componenti
        self.scan_processor = None
        self.exporter = ModelExporter()

        # Registra i callback
        self.client.on_event(UnlookClientEvent.CONNECTED, self._on_connected)
        self.client.on_event(UnlookClientEvent.DISCONNECTED, self._on_disconnected)
        self.client.on_event(UnlookClientEvent.ERROR, self._on_error)
        self.client.on_event(UnlookClientEvent.SCANNER_FOUND, self._on_scanner_found)

        # Stato
        self.target_scanner = None
        self.connected = False
        self.calibration = None
        self.camera_id = None
        self.scan_results = []

    def _on_connected(self, scanner):
        """Callback per la connessione."""
        self.connected = True
        logging.info(f"Connesso a {scanner.name} ({scanner.uuid})")

    def _on_disconnected(self):
        """Callback per la disconnessione."""
        self.connected = False
        logging.info("Disconnesso dallo scanner")

    def _on_error(self, error):
        """Callback per gli errori."""
        logging.error(f"Errore: {error}")

    def _on_scanner_found(self, scanner, added):
        """Callback per la scoperta degli scanner."""
        if added:
            logging.info(f"Scanner trovato: {scanner.name} ({scanner.uuid}) su {scanner.endpoint}")

            # Se il nome corrisponde al target, memorizzalo
            if scanner.name == self.config["scanner_name"]:
                self.target_scanner = scanner
                logging.info(f"Scanner target trovato: {scanner.name}")

    def _find_camera(self):
        """Trova la prima telecamera disponibile."""
        if not self.connected:
            return None

        cameras = self.client.camera.get_cameras()
        if cameras:
            camera_id = cameras[0]["id"]
            logging.info(f"Telecamera trovata: {cameras[0]['name']} ({camera_id})")
            return camera_id
        else:
            logging.error("Nessuna telecamera disponibile")
            return None

    def _load_calibration(self):
        """Carica i dati di calibrazione."""
        calib_file = os.path.join(self.output_dir, self.config["calibration_file"])

        if os.path.exists(calib_file):
            self.calibration = CalibrationData.load(calib_file)
            if self.calibration and self.calibration.is_valid():
                logging.info(f"Calibrazione caricata da {calib_file}")
                return True
            else:
                logging.warning(f"File di calibrazione non valido: {calib_file}")
        else:
            logging.warning(f"File di calibrazione non trovato: {calib_file}")

        return False

    def connect(self):
        """Connette allo scanner target."""
        # Avvia la discovery
        self.client.start_discovery()

        # Attendi la discovery
        timeout = self.config["discovery_timeout"]
        logging.info(f"Ricerca scanner in corso ({timeout} secondi)...")

        start_time = time.time()
        while time.time() - start_time < timeout and self.target_scanner is None:
            time.sleep(0.5)

        # Verifica se è stato trovato lo scanner target
        if self.target_scanner is None:
            # Prova a usare il primo scanner disponibile
            scanners = self.client.get_discovered_scanners()
            if scanners:
                self.target_scanner = scanners[0]
                logging.info(f"Usato primo scanner disponibile: {self.target_scanner.name}")
            else:
                logging.error("Nessuno scanner trovato")
                return False

        # Connetti allo scanner
        logging.info(f"Connessione a {self.target_scanner.name}...")
        if self.client.connect(self.target_scanner, timeout=self.config["connection_timeout"]):
            # Trova la telecamera
            self.camera_id = self._find_camera()
            if not self.camera_id:
                self.client.disconnect()
                return False

            # Carica la calibrazione
            self._load_calibration()

            return True
        else:
            logging.error("Impossibile connettersi allo scanner")
            return False

    def disconnect(self):
        """Disconnette dallo scanner."""
        if self.connected:
            self.client.disconnect()
        self.client.stop_discovery()

    def run_scans(self):
        """Esegue una serie di scansioni automatizzate."""
        if not self.connected or not self.camera_id:
            logging.error("Non connesso o telecamera non disponibile")
            return False

        # Inizializza il processore di scansione
        self.scan_processor = ScanProcessor(self.client)

        # Imposta la calibrazione se disponibile
        if self.calibration:
            self.scan_processor.calibration = self.calibration

        # Esegui le scansioni
        scan_count = self.config["scan_count"]
        logging.info(f"Avvio di {scan_count} scansioni automatizzate")

        for i in range(scan_count):
            logging.info(f"Scansione {i + 1}/{scan_count}")

            # Esegui scansione
            success, result = self._execute_scan()

            if success and result.has_point_cloud():
                self.scan_results.append(result)
                logging.info(f"Scansione {i + 1} completata con {result.num_points} punti")

                # Esporta i risultati
                if self.config["auto_process"]:
                    self._export_scan(result, f"scan_{i + 1}")
            else:
                logging.error(f"Scansione {i + 1} fallita")

            # Pausa tra le scansioni
            if i < scan_count - 1:
                pause = self.config["pause_between_scans"]
                logging.info(f"Pausa di {pause} secondi prima della prossima scansione...")
                time.sleep(pause)

        return len(self.scan_results) > 0

    def _execute_scan(self):
        """Esegue una singola scansione."""
        pattern_type = self.config["pattern_type"]
        pattern_direction = self.config["pattern_direction"]

        logging.info(f"Esecuzione scansione con pattern {pattern_type.value}, direzione {pattern_direction.value}")

        if pattern_type == PatternType.GRAY_CODE:
            return self.scan_processor.capture_gray_code_scan(
                camera_id=self.camera_id,
                pattern_width=1280,
                pattern_height=800,
                direction=pattern_direction,
                capture_texture=True,
                show_preview=False
            )
        elif pattern_type == PatternType.PHASE_SHIFT:
            return self.scan_processor.capture_phase_shift_scan(
                camera_id=self.camera_id,
                pattern_width=1280,
                pattern_height=800,
                direction=pattern_direction,
                num_shifts=4,
                capture_texture=True,
                show_preview=False
            )
        else:
            logging.error(f"Tipo di pattern non supportato: {pattern_type}")
            return False, None

    def _export_scan(self, result, base_name):
        """Esporta i risultati di una scansione."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(self.output_dir, f"{base_name}_{timestamp}")

        formats = self.config["save_formats"]

        if "ply" in formats:
            ply_file = f"{base_filename}.ply"
            if self.exporter.export_ply(result, ply_file):
                logging.info(f"Nuvola di punti esportata in: {ply_file}")

        if "obj" in formats:
            obj_file = f"{base_filename}.obj"
            if self.exporter.export_obj(result, obj_file):
                logging.info(f"Modello OBJ esportato in: {obj_file}")

        if "xyz" in formats:
            xyz_file = f"{base_filename}.xyz"
            if self.exporter.export_xyz(result, xyz_file):
                logging.info(f"File XYZ esportato in: {xyz_file}")

        # Esporta sempre i metadati
        meta_file = f"{base_filename}_meta.json"
        if self.exporter.export_metadata(result, meta_file):
            logging.info(f"Metadati esportati in: {meta_file}")

    def process_scans(self):
        """Elabora le scansioni."""
        if not self.scan_results:
            logging.error("Nessuna scansione da elaborare")
            return False

        # Qui puoi implementare l'elaborazione delle scansioni
        # Per esempio, allineamento, fusione, ecc.
        logging.info("Elaborazione scansioni (non implementata)")

        return True


def main():
    """Funzione principale."""
    try:
        # Crea lo scanner automatizzato
        scanner = UnlookAutomatedScanner()

        # Connetti allo scanner
        if not scanner.connect():
            logging.error("Impossibile connettersi a uno scanner. Uscita.")
            return 1

        # Esegui le scansioni
        if scanner.run_scans():
            logging.info(f"Completate {len(scanner.scan_results)} scansioni")

            # Elabora le scansioni
            scanner.process_scans()
        else:
            logging.error("Tutte le scansioni sono fallite")

        # Disconnetti
        scanner.disconnect()

        return 0

    except KeyboardInterrupt:
        logging.info("Interruzione rilevata")
        return 130
    except Exception as e:
        logging.exception(f"Errore non gestito: {e}")
        return 1
    finally:
        # Chiudi eventuali finestre OpenCV
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())