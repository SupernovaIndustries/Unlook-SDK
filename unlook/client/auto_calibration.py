"""
Modulo per la calibrazione automatica di sistemi stereo.

Questo modulo fornisce funzionalità per automatizzare il processo di calibrazione,
utilizzando rilevamento automatico della scacchiera e feedback in tempo reale.
"""

import logging
import time
import threading
import queue
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable

from .calibration import CalibrationData
from .stereo import StereoCalibrator, StereoCalibrationData

logger = logging.getLogger(__name__)


class CalibrationStatus:
    """Classe per mantenere lo stato della calibrazione."""

    def __init__(self):
        """Inizializza lo stato della calibrazione."""
        self.progress = 0.0
        self.message = "In attesa"
        self.status = "waiting"  # waiting, running, success, error
        self.left_images = []
        self.right_images = []
        self.detected_corners = []
        self.is_complete = False
        self.calibration_result = None
        self.error = None


class AutoCalibration:
    """
    Classe per la calibrazione automatica di sistemi stereo.
    Gestisce l'acquisizione automatica di immagini stereo per la calibrazione.
    """

    def __init__(self, client, board_size=(9, 6), square_size=25.0):
        """
        Inizializza la calibrazione automatica.

        Args:
            client: Client UnLook
            board_size: Dimensioni della scacchiera (angoli interni)
            square_size: Dimensione del quadrato in mm
        """
        self.client = client
        self.board_size = board_size
        self.square_size = square_size

        # Stato della calibrazione
        self.status = CalibrationStatus()

        # Thread di calibrazione
        self.calibration_thread = None
        self.stop_flag = False

        # Code di comunicazione
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Stato di esecuzione
        self.is_running = False

    def start_auto_calibration(
            self,
            num_images: int = 15,
            timeout: int = 120,
            min_movement: float = 10.0,
            callback: Optional[Callable[[CalibrationStatus], None]] = None
    ) -> bool:
        """
        Avvia la calibrazione automatica stereo.

        Args:
            num_images: Numero di immagini da acquisire
            timeout: Timeout in secondi
            min_movement: Movimento minimo tra immagini consecutive (pixel)
            callback: Funzione chiamata quando lo stato cambia

        Returns:
            True se l'avvio ha successo, False altrimenti
        """
        if self.is_running:
            logger.warning("La calibrazione è già in esecuzione")
            return False

        # Ottieni la coppia stereo
        left_camera_id, right_camera_id = self.client.camera.get_stereo_pair()
        if left_camera_id is None or right_camera_id is None:
            logger.error("Impossibile trovare una coppia stereo valida")
            return False

        # Resetta lo stato
        self.status = CalibrationStatus()
        self.stop_flag = False

        # Avvia il thread di calibrazione
        self.calibration_thread = threading.Thread(
            target=self._calibration_loop,
            args=(left_camera_id, right_camera_id, num_images, timeout, min_movement, callback),
            daemon=True
        )
        self.calibration_thread.start()
        self.is_running = True

        return True

    def stop_calibration(self):
        """Ferma la calibrazione automatica."""
        if not self.is_running:
            return

        self.stop_flag = True

        # Attendi la terminazione del thread
        if self.calibration_thread and self.calibration_thread.is_alive():
            self.calibration_thread.join(timeout=3.0)

        self.is_running = False
        logger.info("Calibrazione automatica fermata")

    def get_status(self) -> CalibrationStatus:
        """
        Ottiene lo stato corrente della calibrazione.

        Returns:
            Stato della calibrazione
        """
        return self.status

    def _calibration_loop(
            self,
            left_camera_id: str,
            right_camera_id: str,
            num_images: int,
            timeout: int,
            min_movement: float,
            callback: Optional[Callable[[CalibrationStatus], None]]
    ):
        """
        Loop principale di calibrazione.

        Args:
            left_camera_id: ID della telecamera sinistra
            right_camera_id: ID della telecamera destra
            num_images: Numero di immagini da acquisire
            timeout: Timeout in secondi
            min_movement: Movimento minimo tra immagini consecutive (pixel)
            callback: Funzione chiamata quando lo stato cambia
        """
        try:
            # Aggiorna lo stato
            self.status.status = "running"
            self.status.message = "Inizializzazione..."

            if callback:
                callback(self.status)

            # Calcola il criterio di ricerca degli angoli
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # Prepara i punti 3D della scacchiera
            objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
            objp *= self.square_size  # Scala alle dimensioni reali

            # Liste per memorizzare i punti
            object_points = []  # Punti 3D nel mondo reale
            left_img_points = []  # Punti 2D nella telecamera sinistra
            right_img_points = []  # Punti 2D nella telecamera destra

            # Memorizza l'ultima posizione degli angoli per verificare il movimento
            last_corners_left = None

            # Tempo di inizio
            start_time = time.time()

            # Acquisisci immagini fino a quando non raggiungiamo il numero desiderato o il timeout
            collected_pairs = 0

            while collected_pairs < num_images and not self.stop_flag:
                # Verifica timeout
                if time.time() - start_time > timeout:
                    self.status.status = "error"
                    self.status.message = f"Timeout: impossibile raccogliere {num_images} coppie valide in {timeout} secondi"
                    self.status.error = "Timeout"

                    if callback:
                        callback(self.status)

                    logger.warning(self.status.message)
                    return

                # Aggiorna stato
                self.status.message = f"Acquisizione coppia {collected_pairs + 1}/{num_images}..."
                self.status.progress = collected_pairs / num_images

                if callback:
                    callback(self.status)

                # Cattura immagini stereo
                stereo_images = self.client.camera.capture_multi([left_camera_id, right_camera_id])

                if not stereo_images or left_camera_id not in stereo_images or right_camera_id not in stereo_images:
                    logger.warning("Errore nella cattura delle immagini stereo")
                    time.sleep(0.5)
                    continue

                left_img = stereo_images[left_camera_id]
                right_img = stereo_images[right_camera_id]

                if left_img is None or right_img is None:
                    logger.warning("Immagini stereo non valide")
                    time.sleep(0.5)
                    continue

                # Converti in grayscale
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

                # Trova gli angoli della scacchiera nelle immagini
                left_found, left_corners = cv2.findChessboardCorners(
                    left_gray, self.board_size,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
                )

                right_found, right_corners = cv2.findChessboardCorners(
                    right_gray, self.board_size,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
                )

                # Se la scacchiera è trovata in entrambe le immagini
                if left_found and right_found:
                    # Raffina la posizione degli angoli
                    left_corners_refined = cv2.cornerSubPix(
                        left_gray, left_corners, (11, 11), (-1, -1), criteria
                    )

                    right_corners_refined = cv2.cornerSubPix(
                        right_gray, right_corners, (11, 11), (-1, -1), criteria
                    )

                    # Verifica se c'è stato un movimento sufficiente rispetto all'ultima immagine
                    if last_corners_left is not None:
                        # Calcola il movimento medio
                        movement = np.mean(np.abs(left_corners_refined - last_corners_left))

                        if movement < min_movement:
                            # Movimento insufficiente
                            self.status.message = f"Movimento insufficiente ({movement:.1f}px). Muovi la scacchiera."
                            if callback:
                                callback(self.status)
                            time.sleep(0.5)
                            continue

                    # Aggiorna l'ultima posizione
                    last_corners_left = left_corners_refined.copy()

                    # Disegna gli angoli sulle immagini
                    left_corners_img = cv2.drawChessboardCorners(
                        left_img.copy(), self.board_size, left_corners_refined, left_found
                    )

                    right_corners_img = cv2.drawChessboardCorners(
                        right_img.copy(), self.board_size, right_corners_refined, right_found
                    )

                    # Salva le immagini e i punti
                    self.status.left_images.append(left_img)
                    self.status.right_images.append(right_img)
                    self.status.detected_corners.append((left_corners_img, right_corners_img))

                    object_points.append(objp)
                    left_img_points.append(left_corners_refined)
                    right_img_points.append(right_corners_refined)

                    # Incrementa il contatore
                    collected_pairs += 1

                    # Aggiorna lo stato
                    self.status.message = f"Coppia {collected_pairs}/{num_images} acquisita con successo"
                    self.status.progress = collected_pairs / num_images

                    if callback:
                        callback(self.status)

                    # Attendi un po' prima di acquisire la prossima coppia
                    time.sleep(1.0)
                else:
                    # Scacchiera non trovata in una o entrambe le immagini
                    self.status.message = "Scacchiera non rilevata. Posiziona correttamente la scacchiera."
                    if callback:
                        callback(self.status)
                    time.sleep(0.5)

            # Se fermato manualmente
            if self.stop_flag:
                self.status.status = "error"
                self.status.message = "Calibrazione interrotta dall'utente"
                if callback:
                    callback(self.status)
                return

            # Verifica se abbiamo raccolto abbastanza immagini
            if collected_pairs < 3:
                self.status.status = "error"
                self.status.message = f"Raccolte solo {collected_pairs} coppie valide. Ne servono almeno 3."
                self.status.error = "Dati insufficienti"
                if callback:
                    callback(self.status)
                return

            # Esegui la calibrazione stereo
            self.status.message = "Esecuzione calibrazione stereo..."
            if callback:
                callback(self.status)

            # Dimensioni dell'immagine
            image_size = (left_gray.shape[1], left_gray.shape[0])

            # Calibra prima le singole telecamere
            self.status.message = "Calibrazione telecamera sinistra..."
            if callback:
                callback(self.status)

            ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
                object_points, left_img_points, image_size, None, None
            )

            self.status.message = "Calibrazione telecamera destra..."
            if callback:
                callback(self.status)

            ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
                object_points, right_img_points, image_size, None, None
            )

            # Calibrazione stereo
            self.status.message = "Calibrazione stereo..."
            if callback:
                callback(self.status)

            flags = cv2.CALIB_FIX_INTRINSIC
            ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
                object_points, left_img_points, right_img_points,
                mtx_left, dist_left, mtx_right, dist_right,
                image_size, flags=flags
            )

            # Calcola le matrici di rettificazione
            self.status.message = "Calcolo rettificazione stereo..."
            if callback:
                callback(self.status)

            R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
                mtx_left, dist_left, mtx_right, dist_right,
                image_size, R, T, alpha=0
            )

            # Crea l'oggetto di calibrazione
            stereo_calib = StereoCalibrationData()
            stereo_calib.left_camera_matrix = mtx_left
            stereo_calib.left_dist_coeffs = dist_left
            stereo_calib.right_camera_matrix = mtx_right
            stereo_calib.right_dist_coeffs = dist_right
            stereo_calib.R = R
            stereo_calib.T = T
            stereo_calib.E = E
            stereo_calib.F = F
            stereo_calib.R1 = R1
            stereo_calib.R2 = R2
            stereo_calib.P1 = P1
            stereo_calib.P2 = P2
            stereo_calib.Q = Q
            stereo_calib.image_size = image_size
            stereo_calib.calibration_error = ret
            stereo_calib.calibration_date = time.strftime("%Y-%m-%d %H:%M:%S")

            # Imposta lo stato finale
            self.status.status = "success"
            self.status.message = f"Calibrazione completata con successo (errore: {ret:.6f})"
            self.status.progress = 1.0
            self.status.is_complete = True
            self.status.calibration_result = stereo_calib

            if callback:
                callback(self.status)

            logger.info("Calibrazione stereo automatica completata con successo")

        except Exception as e:
            logger.error(f"Errore durante la calibrazione automatica: {e}")

            self.status.status = "error"
            self.status.message = f"Errore: {str(e)}"
            self.status.error = str(e)

            if callback:
                callback(self.status)

        finally:
            self.is_running = False

    def create_visualization(self) -> Optional[np.ndarray]:
        """
        Crea un'immagine di visualizzazione dello stato corrente.

        Returns:
            Immagine di visualizzazione o None
        """
        if not self.status.detected_corners:
            return None

        # Prendi l'ultima coppia di immagini con angoli rilevati
        left_corners_img, right_corners_img = self.status.detected_corners[-1]

        # Ridimensiona le immagini se necessario
        h, w = left_corners_img.shape[:2]
        max_width = 800
        if w > max_width:
            scale = max_width / w
            h_new = int(h * scale)
            left_corners_img = cv2.resize(left_corners_img, (max_width, h_new))
            right_corners_img = cv2.resize(right_corners_img, (max_width, h_new))

        # Aggiungi testo informativo
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            left_corners_img,
            f"Sinistra: {len(self.status.left_images)}/{self.status.progress * 100:.0f}%",
            (10, 30),
            font,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            right_corners_img,
            f"Destra: {self.status.message}",
            (10, 30),
            font,
            0.8,
            (0, 255, 0),
            2
        )

        # Combina le immagini
        combined = np.vstack((left_corners_img, right_corners_img))

        return combined