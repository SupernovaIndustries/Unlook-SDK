"""
Modulo per la calibrazione del sistema scanner 3D UnLook.
"""

import logging
import os
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class CalibrationData:
    """Classe per gestire i dati di calibrazione."""

    def __init__(self):
        """Inizializza i dati di calibrazione."""
        # Parametri intrinseci della telecamera
        self.camera_matrix = None  # Matrice della telecamera 3x3
        self.dist_coeffs = None  # Coefficienti di distorsione

        # Parametri del proiettore
        self.projector_matrix = None  # Matrice del proiettore 3x3
        self.projector_dist_coeffs = None  # Coefficienti di distorsione del proiettore

        # Parametri estrinseci
        self.cam_to_proj_rotation = None  # Matrice di rotazione camera-proiettore 3x3
        self.cam_to_proj_translation = None  # Vettore di traslazione camera-proiettore 3x1

        # Parametri aggiuntivi
        self.calibration_error = None  # Errore di riproiezione
        self.calibration_date = None  # Data della calibrazione
        self.scanner_uuid = None  # UUID dello scanner calibrato

    def is_valid(self) -> bool:
        """Verifica se i dati di calibrazione sono validi e completi."""
        return (
                self.camera_matrix is not None and
                self.dist_coeffs is not None and
                self.projector_matrix is not None and
                self.projector_dist_coeffs is not None and
                self.cam_to_proj_rotation is not None and
                self.cam_to_proj_translation is not None
        )

    def save(self, filepath: str) -> bool:
        """
        Salva i dati di calibrazione su file.

        Args:
            filepath: Percorso del file

        Returns:
            True se il salvataggio ha successo, False altrimenti
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            data = {
                "camera_matrix": self.camera_matrix.tolist() if self.camera_matrix is not None else None,
                "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
                "projector_matrix": self.projector_matrix.tolist() if self.projector_matrix is not None else None,
                "projector_dist_coeffs": self.projector_dist_coeffs.tolist() if self.projector_dist_coeffs is not None else None,
                "cam_to_proj_rotation": self.cam_to_proj_rotation.tolist() if self.cam_to_proj_rotation is not None else None,
                "cam_to_proj_translation": self.cam_to_proj_translation.tolist() if self.cam_to_proj_translation is not None else None,
                "calibration_error": self.calibration_error,
                "calibration_date": self.calibration_date or time.strftime("%Y-%m-%d %H:%M:%S"),
                "scanner_uuid": self.scanner_uuid
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Dati di calibrazione salvati in {filepath}")
            return True

        except Exception as e:
            logger.error(f"Errore durante il salvataggio dei dati di calibrazione: {e}")
            return False

    @classmethod
    def load(cls, filepath: str) -> 'CalibrationData':
        """
        Carica i dati di calibrazione da file.

        Args:
            filepath: Percorso del file

        Returns:
            Istanza di CalibrationData, None in caso di errore
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            calib = cls()

            # Converte le liste in array numpy
            if data.get("camera_matrix"):
                calib.camera_matrix = np.array(data["camera_matrix"])
            if data.get("dist_coeffs"):
                calib.dist_coeffs = np.array(data["dist_coeffs"])
            if data.get("projector_matrix"):
                calib.projector_matrix = np.array(data["projector_matrix"])
            if data.get("projector_dist_coeffs"):
                calib.projector_dist_coeffs = np.array(data["projector_dist_coeffs"])
            if data.get("cam_to_proj_rotation"):
                calib.cam_to_proj_rotation = np.array(data["cam_to_proj_rotation"])
            if data.get("cam_to_proj_translation"):
                calib.cam_to_proj_translation = np.array(data["cam_to_proj_translation"])

            calib.calibration_error = data.get("calibration_error")
            calib.calibration_date = data.get("calibration_date")
            calib.scanner_uuid = data.get("scanner_uuid")

            logger.info(f"Dati di calibrazione caricati da {filepath}")
            return calib

        except Exception as e:
            logger.error(f"Errore durante il caricamento dei dati di calibrazione: {e}")
            return None


class Calibrator:
    """Classe per la calibrazione del sistema scanner 3D."""

    def __init__(self, client):
        """
        Inizializza il calibratore.

        Args:
            client: Istanza di UnlookClient
        """
        self.client = client
        self.current_calibration = CalibrationData()

        # Parametri predefiniti della scacchiera per la calibrazione
        self.board_size = (9, 6)  # Numero di angoli interni
        self.square_size = 25.0  # Dimensione del quadrato in mm

        # Cache delle immagini
        self.camera_images = []
        self.projector_patterns = []
        self.corners_cache = []

    def calibrate_camera(
            self,
            camera_id: str,
            num_images: int = 15,
            delay_between_captures: float = 1.0,
            show_preview: bool = True
    ) -> Tuple[bool, str]:
        """
        Calibra la telecamera utilizzando una scacchiera.

        Args:
            camera_id: ID della telecamera da calibrare
            num_images: Numero di immagini da acquisire
            delay_between_captures: Ritardo tra le acquisizioni (secondi)
            show_preview: Mostra anteprima con angoli rilevati

        Returns:
            Tuple (successo, messaggio)
        """
        if not self.client.connected:
            return False, "Client non connesso"

        # Pulisci cache
        self.camera_images = []
        self.corners_cache = []

        # Punti 3D sulla scacchiera (Z=0)
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # Scala alle dimensioni reali

        # Array per memorizzare i punti 3D e 2D
        objpoints = []  # Punti 3D nello spazio reale
        imgpoints = []  # Punti 2D nel piano immagine

        captured = 0

        logger.info(f"Inizio calibrazione telecamera: richieste {num_images} immagini")

        while captured < num_images:
            # Cattura immagine
            image = self.client.camera.capture(camera_id)
            if image is None:
                continue

            # Converti in grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Trova gli angoli della scacchiera
            ret, corners = cv2.findChessboardCorners(
                gray, self.board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            )

            if ret:
                # Raffina la posizione degli angoli
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Aggiungi ai punti
                objpoints.append(objp)
                imgpoints.append(refined_corners)

                # Aggiungi immagine alla cache
                self.camera_images.append(image)
                self.corners_cache.append(refined_corners)

                # Incrementa contatore
                captured += 1

                # Visualizza anteprima
                if show_preview:
                    # Disegna gli angoli trovati
                    draw_img = cv2.drawChessboardCorners(image.copy(), self.board_size, refined_corners, ret)
                    cv2.putText(
                        draw_img,
                        f"Immagine {captured}/{num_images}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    cv2.imshow("Calibrazione Telecamera", draw_img)
                    cv2.waitKey(500)  # Mostra per 500 ms

                logger.info(f"Acquisita immagine {captured}/{num_images}")

                # Attendi prima della prossima acquisizione
                time.sleep(delay_between_captures)
            else:
                # Mostra l'immagine senza angoli se richiesto
                if show_preview:
                    cv2.putText(
                        image,
                        "Scacchiera non rilevata",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    cv2.imshow("Calibrazione Telecamera", image)
                    cv2.waitKey(500)

                logger.warning("Scacchiera non rilevata, riprova")
                time.sleep(0.5)

        # Chiudi finestra preview
        if show_preview:
            cv2.destroyWindow("Calibrazione Telecamera")

        # Esegui la calibrazione
        if captured > 0:
            try:
                # Ottieni dimensioni immagine
                h, w = self.camera_images[0].shape[:2]

                # Calibra la telecamera
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, (w, h), None, None
                )

                if ret:
                    # Calcola errore di riproiezione
                    total_error = 0
                    for i in range(len(objpoints)):
                        imgpoints_reprojected, _ = cv2.projectPoints(
                            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                        )
                        error = cv2.norm(imgpoints[i], imgpoints_reprojected, cv2.NORM_L2) / len(imgpoints_reprojected)
                        total_error += error

                    mean_error = total_error / len(objpoints)

                    # Memorizza i risultati
                    self.current_calibration.camera_matrix = camera_matrix
                    self.current_calibration.dist_coeffs = dist_coeffs
                    self.current_calibration.calibration_error = mean_error
                    self.current_calibration.calibration_date = time.strftime("%Y-%m-%d %H:%M:%S")
                    self.current_calibration.scanner_uuid = self.client.scanner.uuid

                    logger.info(f"Calibrazione telecamera completata con errore medio: {mean_error}")
                    return True, f"Calibrazione completata con errore medio: {mean_error:.6f}"
                else:
                    logger.error("Calibrazione telecamera fallita")
                    return False, "Calibrazione fallita"

            except Exception as e:
                logger.error(f"Errore durante la calibrazione della telecamera: {e}")
                return False, f"Errore durante la calibrazione: {e}"
        else:
            return False, "Nessuna immagine valida acquisita"

    def calibrate_system(
            self,
            camera_id: str,
            projector_width: int = 1280,
            projector_height: int = 800,
            pattern_rows: int = 10,
            pattern_cols: int = 18,
            show_preview: bool = True
    ) -> Tuple[bool, str]:
        """
        Calibra il sistema telecamera-proiettore.

        Args:
            camera_id: ID della telecamera
            projector_width: Larghezza del proiettore in pixel
            projector_height: Altezza del proiettore in pixel
            pattern_rows: Numero di righe del pattern proiettato
            pattern_cols: Numero di colonne del pattern proiettato
            show_preview: Mostra anteprima durante la calibrazione

        Returns:
            Tuple (successo, messaggio)
        """
        if not self.client.connected:
            return False, "Client non connesso"

        if self.current_calibration.camera_matrix is None:
            return False, "Calibra prima la telecamera"

        # TODO: Implementare la calibrazione effettiva del sistema
        # Per ora usiamo una calibrazione fittizia per dimostrare la struttura

        # Matrice del proiettore (esempio)
        self.current_calibration.projector_matrix = np.array([
            [1000.0, 0.0, projector_width / 2],
            [0.0, 1000.0, projector_height / 2],
            [0.0, 0.0, 1.0]
        ])

        # Coefficienti di distorsione (esempio)
        self.current_calibration.projector_dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # Rotazione e traslazione tra telecamera e proiettore (esempio)
        self.current_calibration.cam_to_proj_rotation = np.identity(3, dtype=np.float32)
        self.current_calibration.cam_to_proj_translation = np.array([[100.0], [0.0], [0.0]], dtype=np.float32)

        # Aggiorna data e UUID
        self.current_calibration.calibration_date = time.strftime("%Y-%m-%d %H:%M:%S")
        self.current_calibration.scanner_uuid = self.client.scanner.uuid

        logger.info("Calibrazione sistema completata (esempio)")
        return True, "Calibrazione sistema completata (esempio)"

    def generate_calibration_pattern(self, cols: int, rows: int, square_size: int = 80) -> np.ndarray:
        """
        Genera un pattern di calibrazione a scacchiera.

        Args:
            cols: Numero di colonne
            rows: Numero di righe
            square_size: Dimensione di ogni quadrato in pixel

        Returns:
            Immagine del pattern di calibrazione
        """
        width = cols * square_size
        height = rows * square_size
        pattern = np.zeros((height, width), dtype=np.uint8)

        # Crea pattern a scacchiera
        for i in range(rows):
            for j in range(cols):
                if (i + j) % 2 == 0:
                    pattern[i * square_size:(i + 1) * square_size, j * square_size:(j + 1) * square_size] = 255

        # Converti a colori
        pattern_color = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)

        return pattern_color