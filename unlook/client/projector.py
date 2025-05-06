"""
Client per il controllo del proiettore dello scanner UnLook.
"""

import logging
import time
from typing import Dict, Optional, Any, List

from ..common.protocol import MessageType

logger = logging.getLogger(__name__)


class ProjectorClient:
    """
    Client per il controllo del proiettore dello scanner UnLook.
    """

    def __init__(self, parent_client):
        """
        Inizializza il client proiettore.

        Args:
            parent_client: Client principale UnlookClient
        """
        self.client = parent_client

    def set_mode(self, mode: str) -> bool:
        """
        Imposta la modalità del proiettore.

        Args:
            mode: Modalità del proiettore (vedi OperatingMode nell'SDK server)

        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_MODE,
            {"mode": mode}
        )

        if success and response:
            logger.info(f"Modalità proiettore impostata: {mode}")
            return True
        else:
            logger.error(f"Errore nell'impostazione della modalità proiettore: {mode}")
            return False

    def show_solid_field(self, color: str = "White") -> bool:
        """
        Mostra un campo solido di un colore.

        Args:
            color: Colore (Black, Red, Green, Blue, Cyan, Magenta, Yellow, White)

        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN,
            {
                "pattern_type": "solid_field",
                "color": color
            }
        )

        if success and response:
            logger.info(f"Campo solido proiettato: {color}")
            return True
        else:
            logger.error(f"Errore nella proiezione del campo solido: {color}")
            return False

    def show_horizontal_lines(
            self,
            foreground_color: str = "White",
            background_color: str = "Black",
            foreground_width: int = 4,
            background_width: int = 20
    ) -> bool:
        """
        Mostra linee orizzontali.

        Args:
            foreground_color: Colore delle linee
            background_color: Colore dello sfondo
            foreground_width: Larghezza delle linee
            background_width: Larghezza degli spazi tra le linee

        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN,
            {
                "pattern_type": "horizontal_lines",
                "foreground_color": foreground_color,
                "background_color": background_color,
                "foreground_width": foreground_width,
                "background_width": background_width
            }
        )

        if success and response:
            logger.info("Linee orizzontali proiettate")
            return True
        else:
            logger.error("Errore nella proiezione delle linee orizzontali")
            return False

    def show_vertical_lines(
            self,
            foreground_color: str = "White",
            background_color: str = "Black",
            foreground_width: int = 4,
            background_width: int = 20
    ) -> bool:
        """
        Mostra linee verticali.

        Args:
            foreground_color: Colore delle linee
            background_color: Colore dello sfondo
            foreground_width: Larghezza delle linee
            background_width: Larghezza degli spazi tra le linee

        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN,
            {
                "pattern_type": "vertical_lines",
                "foreground_color": foreground_color,
                "background_color": background_color,
                "foreground_width": foreground_width,
                "background_width": background_width
            }
        )

        if success and response:
            logger.info("Linee verticali proiettate")
            return True
        else:
            logger.error("Errore nella proiezione delle linee verticali")
            return False

    def show_grid(
            self,
            foreground_color: str = "White",
            background_color: str = "Black",
            h_foreground_width: int = 4,
            h_background_width: int = 20,
            v_foreground_width: int = 4,
            v_background_width: int = 20
    ) -> bool:
        """
        Mostra una griglia.

        Args:
            foreground_color: Colore delle linee
            background_color: Colore dello sfondo
            h_foreground_width: Larghezza delle linee orizzontali
            h_background_width: Larghezza degli spazi tra le linee orizzontali
            v_foreground_width: Larghezza delle linee verticali
            v_background_width: Larghezza degli spazi tra le linee verticali

        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN,
            {
                "pattern_type": "grid",
                "foreground_color": foreground_color,
                "background_color": background_color,
                "h_foreground_width": h_foreground_width,
                "h_background_width": h_background_width,
                "v_foreground_width": v_foreground_width,
                "v_background_width": v_background_width
            }
        )

        if success and response:
            logger.info("Griglia proiettata")
            return True
        else:
            logger.error("Errore nella proiezione della griglia")
            return False

    def show_checkerboard(
            self,
            foreground_color: str = "White",
            background_color: str = "Black",
            horizontal_count: int = 8,
            vertical_count: int = 6
    ) -> bool:
        """
        Mostra una scacchiera.

        Args:
            foreground_color: Colore primo quadrato
            background_color: Colore secondo quadrato
            horizontal_count: Numero di quadrati orizzontali
            vertical_count: Numero di quadrati verticali

        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN,
            {
                "pattern_type": "checkerboard",
                "foreground_color": foreground_color,
                "background_color": background_color,
                "horizontal_count": horizontal_count,
                "vertical_count": vertical_count
            }
        )

        if success and response:
            logger.info("Scacchiera proiettata")
            return True
        else:
            logger.error("Errore nella proiezione della scacchiera")
            return False

    def show_colorbars(self) -> bool:
        """
        Mostra barre di colore.

        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        success, response, _ = self.client.send_message(
            MessageType.PROJECTOR_PATTERN,
            {"pattern_type": "colorbars"}
        )

        if success and response:
            logger.info("Barre di colore proiettate")
            return True
        else:
            logger.error("Errore nella proiezione delle barre di colore")
            return False

    def set_standby(self) -> bool:
        """
        Mette il proiettore in uno stato equivalente allo standby in modo sicuro.
        Usa un campo nero invece della modalità standby per evitare problemi di timeout I2C.

        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        try:
            # Impostiamo prima la modalità test pattern
            mode_success = self.set_test_pattern_mode()
            if not mode_success:
                logger.warning("Impossibile impostare la modalità test pattern, provo comunque a proiettare nero")

            # Proietta un campo nero (equivalente visivo dello standby)
            black_success = self.show_solid_field("Black")

            # Considerare l'operazione riuscita se riusciamo almeno a proiettare il nero
            return black_success

        except Exception as e:
            logger.error(f"Errore durante l'impostazione dello 'standby visivo': {e}")
            return False

    def set_test_pattern_mode(self) -> bool:
        """
        Imposta il proiettore in modalità test pattern.

        Returns:
            True se l'operazione ha successo, False altrimenti
        """
        return self.set_mode("TestPatternGenerator")