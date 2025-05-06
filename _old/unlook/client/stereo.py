"""
Modulo per la stereovisione e l'elaborazione di immagini stereo per la ricostruzione 3D.
"""

import logging
import time
import os
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import cv2

from .calibration import CalibrationData

logger = logging.getLogger(__name__)


class StereoCalibrationData:
    """Classe per gestire i dati di calibrazione stereo."""

    def __init__(self):
        """Inizializza i dati di calibrazione stereo."""
        # Calibrazione singole telecamere
        self.left_camera_matrix = None  # Matrice della telecamera sinistra 3x3
        self.left_dist_coeffs = None  # Coefficienti di distorsione sinistra
        self.right_camera_matrix = None  # Matrice della telecamera destra 3x3
        self.right_dist_coeffs = None  # Coefficienti di distorsione destra

        # Parametri estrinseci stereo
        self.R = None  # Matrice di rotazione sinistra-destra 3x3
        self.T = None  # Vettore di traslazione sinistra-destra 3x1
        self.E = None  # Matrice essenziale
        self.F = None  # Matrice fondamentale

        # Parametri di rettificazione
        self.R1 = None  # Matrice di rotazione per rettificare la telecamera sinistra
        self.R2 = None  # Matrice di rotazione per rettificare la telecamera destra
        self.P1 = None  # Matrice di proiezione per la telecamera sinistra rettificata
        self.P2 = None  # Matrice di proiezione per la telecamera destra rettificata
        self.Q = None  # Matrice di disparità-profondità

        # Parametri aggiuntivi
        self.calibration_error = None  # Errore di riproiezione
        self.calibration_date = None  # Data della calibrazione
        self.scanner_uuid = None  # UUID dello scanner calibrato
        self.image_size = None  # Dimensione delle immagini (width, height)

    def is_valid(self) -> bool:
        """Verifica se i dati di calibrazione sono validi e completi."""
        return (
                self.left_camera_matrix is not None and
                self.left_dist_coeffs is not None and
                self.right_camera_matrix is not None and
                self.right_dist_coeffs is not None and
                self.R is not None and
                self.T is not None and
                self.R1 is not None and
                self.R2 is not None and
                self.P1 is not None and
                self.P2 is not None and
                self.Q is not None and
                self.image_size is not None
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
                "left_camera_matrix": self.left_camera_matrix.tolist() if self.left_camera_matrix is not None else None,
                "left_dist_coeffs": self.left_dist_coeffs.tolist() if self.left_dist_coeffs is not None else None,
                "right_camera_matrix": self.right_camera_matrix.tolist() if self.right_camera_matrix is not None else None,
                "right_dist_coeffs": self.right_dist_coeffs.tolist() if self.right_dist_coeffs is not None else None,
                "R": self.R.tolist() if self.R is not None else None,
                "T": self.T.tolist() if self.T is not None else None,
                "E": self.E.tolist() if self.E is not None else None,
                "F": self.F.tolist() if self.F is not None else None,
                "R1": self.R1.tolist() if self.R1 is not None else None,
                "R2": self.R2.tolist() if self.R2 is not None else None,
                "P1": self.P1.tolist() if self.P1 is not None else None,
                "P2": self.P2.tolist() if self.P2 is not None else None,
                "Q": self.Q.tolist() if self.Q is not None else None,
                "calibration_error": self.calibration_error,
                "calibration_date": self.calibration_date or time.strftime("%Y-%m-%d %H:%M:%S"),
                "scanner_uuid": self.scanner_uuid,
                "image_size": self.image_size
            }

            import json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Dati di calibrazione stereo salvati in {filepath}")
            return True

        except Exception as e:
            logger.error(f"Errore durante il salvataggio dei dati di calibrazione stereo: {e}")
            return False

    @classmethod
    def load(cls, filepath: str) -> 'StereoCalibrationData':
        """
        Carica i dati di calibrazione da file.

        Args:
            filepath: Percorso del file

        Returns:
            Istanza di StereoCalibrationData, None in caso di errore
        """
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)

            calib = cls()

            # Converte le liste in array numpy
            if data.get("left_camera_matrix"):
                calib.left_camera_matrix = np.array(data["left_camera_matrix"])
            if data.get("left_dist_coeffs"):
                calib.left_dist_coeffs = np.array(data["left_dist_coeffs"])
            if data.get("right_camera_matrix"):
                calib.right_camera_matrix = np.array(data["right_camera_matrix"])
            if data.get("right_dist_coeffs"):
                calib.right_dist_coeffs = np.array(data["right_dist_coeffs"])
            if data.get("R"):
                calib.R = np.array(data["R"])
            if data.get("T"):
                calib.T = np.array(data["T"])
            if data.get("E"):
                calib.E = np.array(data["E"])
            if data.get("F"):
                calib.F = np.array(data["F"])
            if data.get("R1"):
                calib.R1 = np.array(data["R1"])
            if data.get("R2"):
                calib.R2 = np.array(data["R2"])
            if data.get("P1"):
                calib.P1 = np.array(data["P1"])
            if data.get("P2"):
                calib.P2 = np.array(data["P2"])
            if data.get("Q"):
                calib.Q = np.array(data["Q"])

            calib.calibration_error = data.get("calibration_error")
            calib.calibration_date = data.get("calibration_date")
            calib.scanner_uuid = data.get("scanner_uuid")
            calib.image_size = data.get("image_size")

            logger.info(f"Dati di calibrazione stereo caricati da {filepath}")
            return calib

        except Exception as e:
            logger.error(f"Errore durante il caricamento dei dati di calibrazione stereo: {e}")
            return None


class StereoProcessor:
    """Classe per l'elaborazione di immagini stereo e ricostruzione 3D."""

    def __init__(self, stereo_calibration: Optional[StereoCalibrationData] = None):
        """
        Inizializza il processore stereo.

        Args:
            stereo_calibration: Dati di calibrazione stereo
        """
        self.calibration = stereo_calibration

        # Parametri algoritmo di corrispondenza stereo
        self.min_disparity = 0
        self.num_disparities = 64  # Deve essere divisibile per 16
        self.block_size = 11
        self.uniqueness_ratio = 15
        self.speckle_window_size = 100
        self.speckle_range = 2
        self.disp12_max_diff = 1

        # Mappe di rettificazione (calcolate la prima volta che vengono usate)
        self.stereo_rectify_maps = None

        # Algoritmo di corrispondenza stereo
        self.stereo_matcher = None
        self._init_stereo_matcher()

    def _init_stereo_matcher(self):
        """Inizializza l'algoritmo di corrispondenza stereo."""
        # Utilizziamo StereoSGBM per risultati migliori
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2,
            P2=32 * 3 * self.block_size ** 2,
            disp12MaxDiff=self.disp12_max_diff,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def set_calibration(self, calibration: StereoCalibrationData):
        """
        Imposta i dati di calibrazione.

        Args:
            calibration: Dati di calibrazione stereo
        """
        self.calibration = calibration
        self.stereo_rectify_maps = None  # Reset delle mappe di rettificazione

    def compute_stereo_rectify_maps(self, image_size: Tuple[int, int] = None) -> bool:
        """
        Calcola le mappe di rettificazione stereo.

        Args:
            image_size: Dimensione delle immagini (width, height)

        Returns:
            True se il calcolo ha successo, False altrimenti
        """
        if not self.calibration or not self.calibration.is_valid():
            logger.error("Dati di calibrazione mancanti o non validi")
            return False

        # Usa le dimensioni dell'immagine dalla calibrazione se non specificate
        if image_size is None:
            if self.calibration.image_size is None:
                logger.error("Dimensione immagine non specificata e non presente nella calibrazione")
                return False
            image_size = self.calibration.image_size

        try:
            # Calcola le mappe di rettificazione
            left_map_x, left_map_y = cv2.initUndistortRectifyMap(
                self.calibration.left_camera_matrix,
                self.calibration.left_dist_coeffs,
                self.calibration.R1,
                self.calibration.P1,
                image_size,
                cv2.CV_32FC1
            )

            right_map_x, right_map_y = cv2.initUndistortRectifyMap(
                self.calibration.right_camera_matrix,
                self.calibration.right_dist_coeffs,
                self.calibration.R2,
                self.calibration.P2,
                image_size,
                cv2.CV_32FC1
            )

            self.stereo_rectify_maps = {
                "left_map_x": left_map_x,
                "left_map_y": left_map_y,
                "right_map_x": right_map_x,
                "right_map_y": right_map_y,
                "image_size": image_size
            }

            logger.info("Mappe di rettificazione stereo calcolate con successo")
            return True

        except Exception as e:
            logger.error(f"Errore durante il calcolo delle mappe di rettificazione: {e}")
            return False

    def rectify_stereo_images(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Rettifica una coppia di immagini stereo.

        Args:
            left_image: Immagine dalla telecamera sinistra
            right_image: Immagine dalla telecamera destra

        Returns:
            Tuple (left_rectified, right_rectified)
        """
        if not self.calibration or not self.calibration.is_valid():
            logger.error("Dati di calibrazione mancanti o non validi")
            return None, None

        # Verifica che le immagini abbiano le stesse dimensioni
        if left_image.shape[:2] != right_image.shape[:2]:
            logger.error(f"Le immagini hanno dimensioni diverse: {left_image.shape[:2]} vs {right_image.shape[:2]}")
            return None, None

        image_size = (left_image.shape[1], left_image.shape[0])  # (width, height)

        # Se le mappe di rettificazione non sono state calcolate o sono per dimensioni diverse
        if (self.stereo_rectify_maps is None or
                self.stereo_rectify_maps["image_size"] != image_size):
            if not self.compute_stereo_rectify_maps(image_size):
                return None, None

        try:
            # Applica la rettificazione
            left_rectified = cv2.remap(
                left_image,
                self.stereo_rectify_maps["left_map_x"],
                self.stereo_rectify_maps["left_map_y"],
                cv2.INTER_LINEAR
            )

            right_rectified = cv2.remap(
                right_image,
                self.stereo_rectify_maps["right_map_x"],
                self.stereo_rectify_maps["right_map_y"],
                cv2.INTER_LINEAR
            )

            return left_rectified, right_rectified

        except Exception as e:
            logger.error(f"Errore durante la rettificazione delle immagini: {e}")
            return None, None

    def compute_disparity_map(self, left_image: np.ndarray, right_image: np.ndarray, rectified: bool = False) -> \
    Optional[np.ndarray]:
        """
        Calcola la mappa di disparità da una coppia di immagini stereo.

        Args:
            left_image: Immagine dalla telecamera sinistra
            right_image: Immagine dalla telecamera destra
            rectified: Se True, le immagini sono già rettificate

        Returns:
            Mappa di disparità, None in caso di errore
        """
        try:
            # Converti in grayscale se necessario
            if len(left_image.shape) == 3:
                left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_image

            if len(right_image.shape) == 3:
                right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
            else:
                right_gray = right_image

            # Rettifica le immagini se necessario
            if not rectified and self.calibration and self.calibration.is_valid():
                left_rectified, right_rectified = self.rectify_stereo_images(left_gray, right_gray)
                if left_rectified is None or right_rectified is None:
                    return None
                left_gray = left_rectified
                right_gray = right_rectified

            # Calcola la mappa di disparità
            disparity = self.stereo_matcher.compute(left_gray, right_gray)

            # Normalizza per la visualizzazione
            # OpenCV restituisce la disparità in pixel * 16
            disparity = disparity.astype(np.float32) / 16.0

            return disparity

        except Exception as e:
            logger.error(f"Errore durante il calcolo della mappa di disparità: {e}")
            return None

    def disparity_to_depth(self, disparity: np.ndarray) -> Optional[np.ndarray]:
        """
        Converte una mappa di disparità in una mappa di profondità.

        Args:
            disparity: Mappa di disparità

        Returns:
            Mappa di profondità, None in caso di errore
        """
        if not self.calibration or self.calibration.Q is None:
            logger.error("Dati di calibrazione mancanti o matrice Q non disponibile")
            return None

        try:
            # Calcola la mappa di profondità (punto per punto)
            points_3d = cv2.reprojectImageTo3D(disparity, self.calibration.Q)

            # Estrai la componente Z (profondità)
            depth_map = points_3d[:, :, 2]

            # Applica una maschera per rimuovere i valori infiniti e negativi
            mask = (disparity > self.min_disparity) & (depth_map > 0) & np.isfinite(depth_map)
            depth_map[~mask] = 0

            return depth_map

        except Exception as e:
            logger.error(f"Errore durante la conversione da disparità a profondità: {e}")
            return None

    def compute_point_cloud(self, disparity: np.ndarray, left_image: np.ndarray = None) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calcola una nuvola di punti 3D da una mappa di disparità.

        Args:
            disparity: Mappa di disparità
            left_image: Immagine sinistra per i colori (opzionale)

        Returns:
            Tuple (points, colors), None in caso di errore
        """
        if not self.calibration or self.calibration.Q is None:
            logger.error("Dati di calibrazione mancanti o matrice Q non disponibile")
            return None, None

        try:
            # Riproietta la mappa di disparità nello spazio 3D
            points_3d = cv2.reprojectImageTo3D(disparity, self.calibration.Q)

            # Crea una maschera per i punti validi
            mask = (disparity > self.min_disparity) & np.isfinite(points_3d[:, :, 2])

            # Estrai i punti validi
            points = points_3d[mask]

            # Estrai i colori se disponibili
            colors = None
            if left_image is not None:
                if len(left_image.shape) == 3:
                    colors = left_image[mask]
                else:
                    # Se l'immagine è in grayscale, crea un array RGB replicando il valore
                    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
                    colors[:, 0] = left_image[mask]
                    colors[:, 1] = left_image[mask]
                    colors[:, 2] = left_image[mask]

            return points, colors

        except Exception as e:
            logger.error(f"Errore durante il calcolo della nuvola di punti: {e}")
            return None, None

    def compute_stereo_scan(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Esegue una scansione stereo completa.

        Args:
            left_image: Immagine dalla telecamera sinistra
            right_image: Immagine dalla telecamera destra

        Returns:
            Tuple (points, colors, disparity_map)
        """
        # Rettifica le immagini
        left_rect, right_rect = self.rectify_stereo_images(left_image, right_image)
        if left_rect is None or right_rect is None:
            return None, None, None

        # Calcola la mappa di disparità
        disparity = self.compute_disparity_map(left_rect, right_rect, rectified=True)
        if disparity is None:
            return None, None, None

        # Calcola la nuvola di punti
        points, colors = self.compute_point_cloud(disparity, left_rect)
        if points is None:
            return None, None, None

        return points, colors, disparity


class StereoCalibrator:
    """Classe per la calibrazione di sistemi stereo."""

    def __init__(self, left_camera_calib: Optional[CalibrationData] = None,
                 right_camera_calib: Optional[CalibrationData] = None):
        """
        Inizializza il calibratore stereo.

        Args:
            left_camera_calib: Calibrazione telecamera sinistra
            right_camera_calib: Calibrazione telecamera destra
        """
        self.left_calib = left_camera_calib
        self.right_calib = right_camera_calib
        self.stereo_calib = StereoCalibrationData()

        # Parametri della scacchiera
        self.board_size = (9, 6)  # Numero di angoli interni
        self.square_size = 25.0  # Dimensione del quadrato in mm

        # Cache delle immagini
        self.left_images = []
        self.right_images = []
        self.object_points = []
        self.left_corners = []
        self.right_corners = []

    def set_camera_calibrations(self, left_calib: CalibrationData, right_calib: CalibrationData):
        """
        Imposta le calibrazioni delle singole telecamere.

        Args:
            left_calib: Calibrazione telecamera sinistra
            right_calib: Calibrazione telecamera destra
        """
        self.left_calib = left_calib
        self.right_calib = right_calib

    def calibrate_stereo(self, left_images: List[np.ndarray], right_images: List[np.ndarray],
                         image_size: Tuple[int, int] = None) -> Tuple[bool, str]:
        """
        Esegue la calibrazione stereo.

        Args:
            left_images: Lista di immagini dalla telecamera sinistra
            right_images: Lista di immagini dalla telecamera destra
            image_size: Dimensione delle immagini (width, height)

        Returns:
            Tuple (successo, messaggio)
        """
        if len(left_images) != len(right_images) or len(left_images) < 3:
            return False, "Servono almeno 3 coppie di immagini stereo corrispondenti"

        if self.left_calib is None or self.right_calib is None:
            return False, "Calibrazioni delle singole telecamere mancanti"

        # Se non è specificata la dimensione dell'immagine, usa quella della prima immagine
        if image_size is None:
            if left_images and left_images[0] is not None:
                h, w = left_images[0].shape[:2]
                image_size = (w, h)
            else:
                return False, "Dimensioni dell'immagine non specificate e non ricavabili"

        # Resetta le cache
        self.left_images = []
        self.right_images = []
        self.object_points = []
        self.left_corners = []
        self.right_corners = []

        # Punti 3D sulla scacchiera (Z=0)
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # Scala alle dimensioni reali

        # Trova gli angoli della scacchiera in entrambe le immagini
        for i, (left_img, right_img) in enumerate(zip(left_images, right_images)):
            # Converti in grayscale
            if len(left_img.shape) == 3:
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_img

            if len(right_img.shape) == 3:
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            else:
                right_gray = right_img

            # Trova gli angoli nelle immagini sinistra e destra
            ret_left, corners_left = cv2.findChessboardCorners(
                left_gray, self.board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            )

            ret_right, corners_right = cv2.findChessboardCorners(
                right_gray, self.board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            )

            # Se trovati in entrambe le immagini, raffina gli angoli
            if ret_left and ret_right:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

                refined_corners_left = cv2.cornerSubPix(
                    left_gray, corners_left, (11, 11), (-1, -1), criteria
                )

                refined_corners_right = cv2.cornerSubPix(
                    right_gray, corners_right, (11, 11), (-1, -1), criteria
                )

                # Aggiungi alle cache
                self.object_points.append(objp)
                self.left_corners.append(refined_corners_left)
                self.right_corners.append(refined_corners_right)
                self.left_images.append(left_img)
                self.right_images.append(right_img)

                logger.info(f"Coppia di immagini {i + 1}: scacchiera trovata in entrambe le immagini")
            else:
                logger.warning(f"Coppia di immagini {i + 1}: scacchiera non trovata in una o entrambe le immagini")

        # Verifica che ci siano abbastanza coppie valide
        if len(self.object_points) < 3:
            return False, "Trovate meno di 3 coppie valide con scacchiera visibile"

        logger.info(f"Inizio calibrazione stereo con {len(self.object_points)} coppie valide")

        try:
            # Usa le matrici di calibrazione delle singole telecamere
            if not self.left_calib.is_valid() or not self.right_calib.is_valid():
                return False, "Calibrazione delle singole telecamere non valida"

            # Imposta le matrici delle telecamere
            self.stereo_calib.left_camera_matrix = self.left_calib.camera_matrix
            self.stereo_calib.left_dist_coeffs = self.left_calib.dist_coeffs
            self.stereo_calib.right_camera_matrix = self.right_calib.camera_matrix
            self.stereo_calib.right_dist_coeffs = self.right_calib.dist_coeffs

            # Esegui la calibrazione stereo
            flags = cv2.CALIB_FIX_INTRINSIC  # Usa le calibrazioni già disponibili

            (
                self.stereo_calib.calibration_error,
                self.stereo_calib.left_camera_matrix,
                self.stereo_calib.left_dist_coeffs,
                self.stereo_calib.right_camera_matrix,
                self.stereo_calib.right_dist_coeffs,
                self.stereo_calib.R,
                self.stereo_calib.T,
                self.stereo_calib.E,
                self.stereo_calib.F
            ) = cv2.stereoCalibrate(
                self.object_points,
                self.left_corners,
                self.right_corners,
                self.stereo_calib.left_camera_matrix,
                self.stereo_calib.left_dist_coeffs,
                self.stereo_calib.right_camera_matrix,
                self.stereo_calib.right_dist_coeffs,
                image_size,
                flags=flags
            )

            # Calcola le matrici di rettificazione
            (
                self.stereo_calib.R1,
                self.stereo_calib.R2,
                self.stereo_calib.P1,
                self.stereo_calib.P2,
                self.stereo_calib.Q,
                _,
                _
            ) = cv2.stereoRectify(
                self.stereo_calib.left_camera_matrix,
                self.stereo_calib.left_dist_coeffs,
                self.stereo_calib.right_camera_matrix,
                self.stereo_calib.right_dist_coeffs,
                image_size,
                self.stereo_calib.R,
                self.stereo_calib.T,
                alpha=0
            )

            # Memorizza le dimensioni dell'immagine e la data
            self.stereo_calib.image_size = image_size
            self.stereo_calib.calibration_date = time.strftime("%Y-%m-%d %H:%M:%S")

            logger.info(f"Calibrazione stereo completata con errore: {self.stereo_calib.calibration_error}")
            return True, f"Calibrazione stereo completata con errore: {self.stereo_calib.calibration_error}"

        except Exception as e:
            logger.error(f"Errore durante la calibrazione stereo: {e}")
            return False, f"Errore durante la calibrazione stereo: {e}"

    def save_calibration(self, filepath: str) -> bool:
        """
        Salva la calibrazione stereo su file.

        Args:
            filepath: Percorso del file

        Returns:
            True se il salvataggio ha successo, False altrimenti
        """
        if not self.stereo_calib.is_valid():
            logger.error("Calibrazione stereo non valida")
            return False

        return self.stereo_calib.save(filepath)