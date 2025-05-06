#!/usr/bin/env python3
"""
Script di test per il proiettore DLP342X.
Proietta una serie di pattern di linee con diverse configurazioni.
"""

import time
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from unlook.server.projector.dlp342x import (
    DLPC342XController,
    OperatingMode,
    Color,
    BorderEnable
)

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestProiettore")


def main():
    # Inizializza il controller (hardcoded su bus 3, indirizzo 0x1B)
    try:
        projector = DLPC342XController(bus=3, address=0x1b)
        logger.info("Controller proiettore inizializzato correttamente")
    except Exception as e:
        logger.error(f"Errore durante l'inizializzazione del proiettore: {e}")
        return 1

    try:
        # Imposta la modalità test pattern
        logger.info("Impostazione modalità test pattern...")
        if not projector.set_operating_mode(OperatingMode.TestPatternGenerator):
            logger.error("Impossibile impostare la modalità test pattern")
            return 1

        # Test 1: Campo bianco
        logger.info("Test 1: Proiezione campo bianco")
        projector.generate_solid_field(Color.White)
        time.sleep(2)

        # Test 2: Linee orizzontali
        logger.info("Test 2: Proiezione linee orizzontali")
        # Varie configurazioni di linee orizzontali
        configs = [
            (Color.White, Color.Black, 4, 20),  # Standard
            (Color.Red, Color.Black, 8, 40),  # Linee rosse spesse
            (Color.Green, Color.Black, 2, 10)  # Linee verdi sottili
        ]

        for fg_color, bg_color, fg_width, bg_width in configs:
            logger.info(f"Linee orizzontali: {fg_color.name} su {bg_color.name}, fg={fg_width}, bg={bg_width}")
            projector.generate_horizontal_lines(
                bg_color, fg_color, fg_width, bg_width
            )
            time.sleep(2)

        # Test 3: Linee verticali
        logger.info("Test 3: Proiezione linee verticali")
        # Varie configurazioni di linee verticali
        configs = [
            (Color.White, Color.Black, 4, 20),  # Standard
            (Color.Blue, Color.Black, 8, 40),  # Linee blu spesse
            (Color.Yellow, Color.Black, 2, 10)  # Linee gialle sottili
        ]

        for fg_color, bg_color, fg_width, bg_width in configs:
            logger.info(f"Linee verticali: {fg_color.name} su {bg_color.name}, fg={fg_width}, bg={bg_width}")
            projector.generate_vertical_lines(
                bg_color, fg_color, fg_width, bg_width
            )
            time.sleep(2)

        # Test 4: Griglia
        logger.info("Test 4: Proiezione griglia")
        projector.generate_grid(
            Color.Black, Color.White, 4, 20, 4, 20
        )
        time.sleep(2)

        # Test 5: Scacchiera
        logger.info("Test 5: Proiezione scacchiera")
        projector.generate_checkerboard(
            Color.White, Color.Black, 8, 6
        )
        time.sleep(2)

        # Test 6: Barre di colore
        logger.info("Test 6: Proiezione barre di colore")
        projector.generate_colorbars()
        time.sleep(2)

        # Fine: Imposta la modalità standby
        logger.info("Test completato. Impostazione modalità standby...")
        projector.set_operating_mode(OperatingMode.Standby)

        logger.info("Test del proiettore completato con successo")
        return 0

    except Exception as e:
        logger.error(f"Errore durante il test del proiettore: {e}")
        # Prova ad impostare la modalità standby
        try:
            projector.set_operating_mode(OperatingMode.Standby)
        except:
            pass
        return 1
    finally:
        # Chiudi il controller
        projector.close()


if __name__ == "__main__":
    sys.exit(main())