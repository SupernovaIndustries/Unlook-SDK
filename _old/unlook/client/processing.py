"""
Modulo per l'elaborazione 3D delle immagini di luce strutturata.
"""

import logging
import time
import os
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import cv2

from .calibration import CalibrationData

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Tipi di pattern di luce strutturata supportati."""
    GRAY_CODE = "gray_code"
    PHASE_SHIFT = "phase_shift"
    BINARY = "binary"
    COMBINED = "combined"


class PatternDirection(Enum):
    """Direzione dei pattern di luce strutturata."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    BOTH = "both"


class ProcessingResult:
    """Classe per i risultati dell'elaborazione 3D."""

    def __init__(self):
        """Inizializza i risultati."""
        self.point_cloud = None  # Nuvola di punti (N, 3)
        self.confidence = None  # Confidenza dei punti (N,)
        self.colors = None  # Colori dei punti (N, 3)
        self.normals = None  # Normali dei punti (N, 3)
        self.disparity_map = None  # Mappa di disparità
        self.texture = None  # Immagine di texture

        # Metadati
        self.timestamp = time.time()
        self.num_points = 0
        self.num_frames = 0
        self.scanner_uuid = None
        self.capture_params = {}

    def has_point_cloud(self) -> bool:
        """
        Verifica se è presente una nuvola di punti.

        Returns:
            True se presente, False altrimenti
        """
        return self.point_cloud is not None and self.num_points > 0

    def save_point_cloud(self, filepath: str) -> bool:
        """
        Salva la nuvola di punti in formato PLY.

        Args:
            filepath: Percorso del file PLY

        Returns:
            True se salvato con successo, False altrimenti
        """
        if not self.has_point_cloud():
            logger.error("Nessuna nuvola di punti da salvare")
            return False

        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            with open(filepath, 'w') as f:
                # Scrivi header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {self.num_points}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")

                if self.colors is not None:
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")

                if self.normals is not None:
                    f.write("property float nx\n")
                    f.write("property float ny\n")
                    f.write("property float nz\n")

                if self.confidence is not None:
                    f.write("property float confidence\n")

                f.write("end_header\n")

                # Scrivi vertici
                for i in range(self.num_points):
                    # Coordinate
                    f.write(f"{self.point_cloud[i, 0]} {self.point_cloud[i, 1]} {self.point_cloud[i, 2]}")

                    # Colori
                    if self.colors is not None:
                        r, g, b = self.colors[i]
                        f.write(f" {int(r)} {int(g)} {int(b)}")

                    # Normali
                    if self.normals is not None:
                        nx, ny, nz = self.normals[i]
                        f.write(f" {nx} {ny} {nz}")

                    # Confidenza
                    if self.confidence is not None:
                        f.write(f" {self.confidence[i]}")

                    f.write("\n")

            logger.info(f"Nuvola di punti salvata in {filepath}")
            return True

        except Exception as e:
            logger.error(f"Errore durante il salvataggio della nuvola di punti: {e}")
            return False


class StructuredLightProcessor:
    """
    Elaboratore di luce strutturata per la ricostruzione 3D.
    """

    def __init__(self, calibration_data: CalibrationData = None):
        """
        Inizializza l'elaboratore di luce strutturata.

        Args:
            calibration_data: Dati di calibrazione
        """
        self.calibration = calibration_data

        # Pipeline di elaborazione
        self.max_disparity = 128
        self.min_disparity = 0

        # Impostazioni
        self.decoding_threshold = 0.1
        self.noise_filtering = True

    def set_calibration(self, calibration_data: CalibrationData):
        """
        Imposta i dati di calibrazione.

        Args:
            calibration_data: Dati di calibrazione
        """
        self.calibration = calibration_data

    def generate_gray_code_patterns(
            self,
            width: int,
            height: int,
            direction: PatternDirection = PatternDirection.BOTH,
            num_bits: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Genera pattern di Gray code.

        Args:
            width: Larghezza del pattern
            height: Altezza del pattern
            direction: Direzione dei pattern
            num_bits: Numero di bit

        Returns:
            Lista di pattern di Gray code
        """
        patterns = []

        # Numero di bit
        h_bits = num_bits or int(np.ceil(np.log2(width)))
        v_bits = num_bits or int(np.ceil(np.log2(height)))

        # Genera pattern orizzontali
        if direction in [PatternDirection.HORIZONTAL, PatternDirection.BOTH]:
            for i in range(h_bits):
                pattern = np.zeros((height, width), dtype=np.uint8)

                # Gray code
                for x in range(width):
                    binary = x ^ (x >> 1)  # Converti in Gray code
                    if (binary >> i) & 1:
                        pattern[:, x] = 255

                patterns.append(pattern)

                # Aggiungi pattern complementare
                patterns.append(255 - pattern)

        # Genera pattern verticali
        if direction in [PatternDirection.VERTICAL, PatternDirection.BOTH]:
            for i in range(v_bits):
                pattern = np.zeros((height, width), dtype=np.uint8)

                # Gray code
                for y in range(height):
                    binary = y ^ (y >> 1)  # Converti in Gray code
                    if (binary >> i) & 1:
                        pattern[y, :] = 255

                patterns.append(pattern)

                # Aggiungi pattern complementare
                patterns.append(255 - pattern)

        # Converti in BGR
        color_patterns = []
        for pattern in patterns:
            color_pattern = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
            color_patterns.append(color_pattern)

        return color_patterns

    def generate_phase_shift_patterns(
            self,
            width: int,
            height: int,
            direction: PatternDirection = PatternDirection.BOTH,
            num_shifts: int = 4
    ) -> List[np.ndarray]:
        """
        Genera pattern di spostamento di fase.

        Args:
            width: Larghezza del pattern
            height: Altezza del pattern
            direction: Direzione dei pattern
            num_shifts: Numero di spostamenti di fase

        Returns:
            Lista di pattern di spostamento di fase
        """
        patterns = []

        # Genera pattern orizzontali
        if direction in [PatternDirection.HORIZONTAL, PatternDirection.BOTH]:
            for i in range(num_shifts):
                pattern = np.zeros((height, width), dtype=np.uint8)

                # Spostamento di fase
                phase = 2 * np.pi * i / num_shifts
                for x in range(width):
                    val = 127.5 + 127.5 * np.cos(2 * np.pi * x / 64 + phase)
                    pattern[:, x] = val

                patterns.append(pattern)

        # Genera pattern verticali
        if direction in [PatternDirection.VERTICAL, PatternDirection.BOTH]:
            for i in range(num_shifts):
                pattern = np.zeros((height, width), dtype=np.uint8)

                # Spostamento di fase
                phase = 2 * np.pi * i / num_shifts
                for y in range(height):
                    val = 127.5 + 127.5 * np.cos(2 * np.pi * y / 64 + phase)
                    pattern[y, :] = val

                patterns.append(pattern)

        # Aggiungi pattern all-white e all-black per normalizzazione
        white = np.ones((height, width), dtype=np.uint8) * 255
        black = np.zeros((height, width), dtype=np.uint8)
        patterns.append(white)
        patterns.append(black)

        # Converti in BGR
        color_patterns = []
        for pattern in patterns:
            color_pattern = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
            color_patterns.append(color_pattern)

        return color_patterns

    def decode_gray_code(
            self,
            images: List[np.ndarray],
            inverted_images: List[np.ndarray],
            direction: PatternDirection = PatternDirection.BOTH
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Decodifica pattern di Gray code.

        Args:
            images: Lista di immagini dei pattern
            inverted_images: Lista di immagini dei pattern invertiti
            direction: Direzione dei pattern

        Returns:
            Tupla (decoded_x, decoded_y, mask)
        """
        if len(images) == 0 or len(inverted_images) == 0:
            logger.error("Nessuna immagine da decodificare")
            return None, None, None

        height, width = images[0].shape[:2]

        # Inizializza i risultati
        decoded_x = None
        decoded_y = None

        # Combina le immagini per ottenere la maschera
        mask = np.ones((height, width), dtype=np.uint8)
        min_diff = 30  # Differenza minima tra pattern e invertito

        # Decodifica Gray code orizzontale
        if direction in [PatternDirection.HORIZONTAL, PatternDirection.BOTH]:
            h_bits = len(images) // 2 if direction == PatternDirection.BOTH else len(images)

            # Inizializza il risultato
            decoded_x = np.zeros((height, width), dtype=np.float32)

            for i in range(h_bits):
                # Converti in grayscale se necessario
                img = images[i] if len(images[i].shape) == 2 else cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
                inv_img = inverted_images[i] if len(inverted_images[i].shape) == 2 else cv2.cvtColor(inverted_images[i],
                                                                                                     cv2.COLOR_BGR2GRAY)

                # Calcola la differenza
                diff = cv2.absdiff(img, inv_img)

                # Aggiorna la maschera
                mask = cv2.bitwise_and(mask, (diff > min_diff).astype(np.uint8))

                # Decodifica bit
                bit = img > inv_img

                # Aggiorna il risultato
                decoded_x += bit.astype(np.float32) * (2 ** (h_bits - i - 1))

        # Decodifica Gray code verticale
        if direction in [PatternDirection.VERTICAL, PatternDirection.BOTH]:
            v_bits = len(images) // 2 if direction == PatternDirection.BOTH else len(images)

            # Offset delle immagini
            offset = len(images) // 2 if direction == PatternDirection.BOTH else 0

            # Inizializza il risultato
            decoded_y = np.zeros((height, width), dtype=np.float32)

            for i in range(v_bits):
                # Converti in grayscale se necessario
                img = images[offset + i] if len(images[offset + i].shape) == 2 else cv2.cvtColor(images[offset + i],
                                                                                                 cv2.COLOR_BGR2GRAY)
                inv_img = inverted_images[offset + i] if len(inverted_images[offset + i].shape) == 2 else cv2.cvtColor(
                    inverted_images[offset + i], cv2.COLOR_BGR2GRAY)

                # Calcola la differenza
                diff = cv2.absdiff(img, inv_img)

                # Aggiorna la maschera
                mask = cv2.bitwise_and(mask, (diff > min_diff).astype(np.uint8))

                # Decodifica bit
                bit = img > inv_img

                # Aggiorna il risultato
                decoded_y += bit.astype(np.float32) * (2 ** (v_bits - i - 1))

        return decoded_x, decoded_y, mask

    def decode_phase_shift(
            self,
            images: List[np.ndarray],
            direction: PatternDirection = PatternDirection.BOTH
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Decodifica pattern di spostamento di fase.

        Args:
            images: Lista di immagini dei pattern
            direction: Direzione dei pattern

        Returns:
            Tupla (decoded_x, decoded_y, mask)
        """
        if len(images) < 3:
            logger.error("Servono almeno 3 immagini per la decodifica dello spostamento di fase")
            return None, None, None

        height, width = images[0].shape[:2]

        # Converti in grayscale se necessario
        gray_images = []
        for img in images:
            if len(img.shape) == 3:
                gray_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            else:
                gray_images.append(img)

        # Calcola la maschera (ultimo frame è all-white, penultimo è all-black)
        white = gray_images[-2]
        black = gray_images[-1]
        mask = (white - black) > 30

        # Inizializza i risultati
        decoded_x = None
        decoded_y = None

        # Numero di spostamenti per direzione
        n_shifts = (len(gray_images) - 2) // 2 if direction == PatternDirection.BOTH else len(gray_images) - 2

        # Decodifica orizzontale
        if direction in [PatternDirection.HORIZONTAL, PatternDirection.BOTH]:
            # Estrai le immagini per la direzione orizzontale
            h_images = gray_images[:n_shifts]

            # Calcola la fase
            numerator = 0
            denominator = 0

            for i, img in enumerate(h_images):
                phase = 2 * np.pi * i / n_shifts
                numerator += img.astype(np.float32) * np.sin(phase)
                denominator += img.astype(np.float32) * np.cos(phase)

            phase_map = np.arctan2(numerator, denominator)

            # Normalizza a [0, width-1]
            decoded_x = (phase_map + np.pi) / (2 * np.pi) * (width - 1)

        # Decodifica verticale
        if direction in [PatternDirection.VERTICAL, PatternDirection.BOTH]:
            # Estrai le immagini per la direzione verticale
            v_images = gray_images[n_shifts:2 * n_shifts] if direction == PatternDirection.BOTH else gray_images[
                                                                                                     :n_shifts]

            # Calcola la fase
            numerator = 0
            denominator = 0

            for i, img in enumerate(v_images):
                phase = 2 * np.pi * i / n_shifts
                numerator += img.astype(np.float32) * np.sin(phase)
                denominator += img.astype(np.float32) * np.cos(phase)

            phase_map = np.arctan2(numerator, denominator)

            # Normalizza a [0, height-1]
            decoded_y = (phase_map + np.pi) / (2 * np.pi) * (height - 1)

        return decoded_x, decoded_y, mask.astype(np.uint8)

    def triangulate_points(
            self,
            decoded_x: np.ndarray,
            decoded_y: np.ndarray,
            mask: np.ndarray,
            texture_image: Optional[np.ndarray] = None
    ) -> ProcessingResult:
        """
        Triangola punti 3D dai pattern decodificati.

        Args:
            decoded_x: Coordinate X decodificate
            decoded_y: Coordinate Y decodificate
            mask: Maschera di validità
            texture_image: Immagine di texture

        Returns:
            Risultato dell'elaborazione
        """
        if self.calibration is None or not self.calibration.is_valid():
            logger.error("Dati di calibrazione mancanti o incompleti")
            return ProcessingResult()

        result = ProcessingResult()

        # Esempio di triangolazione (semplificato)
        # In un'implementazione reale, si userebbero le coordinate decodificate insieme
        # ai parametri di calibrazione per calcolare le coordinate 3D tramite triangolazione

        # Trova le coordinate 2D valide
        y_coords, x_coords = np.where(mask > 0)
        num_points = len(y_coords)

        if num_points == 0:
            logger.warning("Nessun punto valido per la triangolazione")
            return result

        # Array per i risultati
        points_3d = np.zeros((num_points, 3), dtype=np.float32)
        colors = np.zeros((num_points, 3), dtype=np.uint8) if texture_image is not None else None
        confidence = np.ones(num_points, dtype=np.float32)

        # Matrice camera e proiettore
        K_camera = self.calibration.camera_matrix
        K_proj = self.calibration.projector_matrix
        R = self.calibration.cam_to_proj_rotation
        T = self.calibration.cam_to_proj_translation

        # Matrice di proiezione per il proiettore
        P_proj = np.zeros((3, 4), dtype=np.float32)
        P_proj[:3, :3] = np.dot(K_proj, R)
        P_proj[:3, 3] = np.dot(K_proj, T).ravel()

        # Per ogni punto valido
        for i in range(num_points):
            x, y = x_coords[i], y_coords[i]

            # Coordinate 2D nella camera
            point_camera = np.array([x, y, 1.0], dtype=np.float32)

            # Coordinate 2D nel proiettore
            proj_x = decoded_x[y, x] if decoded_x is not None else 0
            proj_y = decoded_y[y, x] if decoded_y is not None else 0
            point_proj = np.array([proj_x, proj_y, 1.0], dtype=np.float32)

            # Triangolazione (semplificata)
            # Nota: questo è un esempio, non una vera triangolazione
            z = 1000.0 / (1.0 + np.abs(x - proj_x) / 100.0)  # Valore Z fittizio
            X = (x - K_camera[0, 2]) * z / K_camera[0, 0]
            Y = (y - K_camera[1, 2]) * z / K_camera[1, 1]

            points_3d[i] = [X, Y, z]

            # Estrai il colore dall'immagine di texture se disponibile
            if texture_image is not None and y < texture_image.shape[0] and x < texture_image.shape[1]:
                colors[i] = texture_image[y, x]

            # Calcola la confidenza (esempio)
            confidence[i] = mask[y, x] / 255.0

        # Aggiungi i risultati
        result.point_cloud = points_3d
        result.colors = colors
        result.confidence = confidence
        result.num_points = num_points

        if texture_image is not None:
            result.texture = texture_image

        return result


class ScanProcessor:
    """
    Classe per la gestione del processo di scansione 3D.
    """

    def __init__(self, client):
        """
        Inizializza il processore di scansione.

        Args:
            client: Istanza di UnlookClient
        """
        self.client = client
        self.calibration = CalibrationData()
        self.processor = StructuredLightProcessor(self.calibration)

        # Cache delle immagini
        self.pattern_images = []
        self.texture_image = None

    def load_calibration(self, filepath: str) -> bool:
        """
        Carica i dati di calibrazione da file.

        Args:
            filepath: Percorso del file

        Returns:
            True se caricato con successo, False altrimenti
        """
        calib = CalibrationData.load(filepath)
        if calib and calib.is_valid():
            self.calibration = calib
            self.processor.set_calibration(calib)
            logger.info("Dati di calibrazione caricati con successo")
            return True
        else:
            logger.error("Impossibile caricare i dati di calibrazione")
            return False

    def capture_gray_code_scan(
            self,
            camera_id: str,
            pattern_width: int = 1280,
            pattern_height: int = 800,
            direction: PatternDirection = PatternDirection.BOTH,
            capture_texture: bool = True,
            show_preview: bool = False
    ) -> Tuple[bool, ProcessingResult]:
        """
        Esegue una scansione utilizzando pattern di Gray code.

        Args:
            camera_id: ID della telecamera
            pattern_width: Larghezza del pattern
            pattern_height: Altezza del pattern
            direction: Direzione dei pattern
            capture_texture: Cattura un'immagine di texture
            show_preview: Mostra anteprima durante la scansione

        Returns:
            Tuple (successo, risultato)
        """
        if not self.client.connected:
            logger.error("Client non connesso")
            return False, ProcessingResult()

        # Genera i pattern di Gray code
        patterns = self.processor.generate_gray_code_patterns(
            pattern_width, pattern_height, direction
        )

        if not patterns:
            logger.error("Impossibile generare i pattern")
            return False, ProcessingResult()

        logger.info(f"Generati {len(patterns)} pattern di Gray code")

        # Configura il proiettore
        if not self.client.projector.set_test_pattern_mode():
            logger.error("Impossibile impostare il proiettore in modalità test pattern")
            return False, ProcessingResult()

        # Cattura immagini
        self.pattern_images = []
        counter = 0

        for pattern in patterns:
            # Proietta il pattern (semplificato)
            # In realtà dovresti convertire il pattern in un formato supportato dal proiettore
            # e inviarlo tramite le API del proiettore
            success = self.client.projector.show_solid_field("White")  # Provvisorio
            if not success:
                logger.error(f"Errore nella proiezione del pattern {counter}")
                continue

            # Attendi che il pattern sia proiettato
            time.sleep(0.2)

            # Cattura l'immagine
            image = self.client.camera.capture(camera_id)
            if image is None:
                logger.error(f"Errore nella cattura dell'immagine {counter}")
                continue

            # Aggiungi alla lista
            self.pattern_images.append(image)
            counter += 1

            # Visualizza anteprima
            if show_preview:
                preview = image.copy()
                cv2.putText(
                    preview,
                    f"Pattern {counter}/{len(patterns)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.imshow("Scansione", preview)
                cv2.waitKey(1)

        # Cattura immagine di texture
        self.texture_image = None
        if capture_texture:
            # Proietta pattern bianco per la texture
            self.client.projector.show_solid_field("White")
            time.sleep(0.2)

            # Cattura l'immagine
            self.texture_image = self.client.camera.capture(camera_id)

            if self.texture_image is None:
                logger.warning("Impossibile catturare l'immagine di texture")
            else:
                logger.info("Immagine di texture catturata")

                # Visualizza anteprima
                if show_preview:
                    cv2.imshow("Texture", self.texture_image)
                    cv2.waitKey(1)

        # Metti il proiettore in standby
        self.client.projector.set_standby()

        # Elabora le immagini
        if len(self.pattern_images) < 2:
            logger.error("Troppe poche immagini catturate")
            return False, ProcessingResult()

        # Calcola il numero di bit
        n_bits = len(self.pattern_images) // 2

        # Separa immagini normali e invertite
        normal_images = self.pattern_images[0:n_bits]
        inverted_images = self.pattern_images[n_bits:2 * n_bits]

        # Decodifica i pattern
        decoded_x, decoded_y, mask = self.processor.decode_gray_code(
            normal_images, inverted_images, direction
        )

        if mask is None or np.sum(mask) == 0:
            logger.error("Decodifica fallita: nessun punto valido")
            return False, ProcessingResult()

        logger.info(f"Decodifica completata: {np.sum(mask)} punti validi")

        # Triangola i punti 3D
        result = self.processor.triangulate_points(
            decoded_x, decoded_y, mask, self.texture_image
        )

        if not result.has_point_cloud():
            logger.error("Triangolazione fallita: nessun punto 3D")
            return False, result

        logger.info(f"Triangolazione completata: {result.num_points} punti 3D")

        # Aggiungi metadati
        result.scanner_uuid = self.client.scanner.uuid
        result.num_frames = len(self.pattern_images)
        result.capture_params = {
            "pattern_type": "gray_code",
            "direction": direction.value,
            "pattern_width": pattern_width,
            "pattern_height": pattern_height,
            "camera_id": camera_id
        }

        return True, result

    def capture_phase_shift_scan(
            self,
            camera_id: str,
            pattern_width: int = 1280,
            pattern_height: int = 800,
            direction: PatternDirection = PatternDirection.BOTH,
            num_shifts: int = 4,
            capture_texture: bool = True,
            show_preview: bool = False
    ) -> Tuple[bool, ProcessingResult]:
        """
        Esegue una scansione utilizzando pattern di spostamento di fase.

        Args:
            camera_id: ID della telecamera
            pattern_width: Larghezza del pattern
            pattern_height: Altezza del pattern
            direction: Direzione dei pattern
            num_shifts: Numero di spostamenti di fase
            capture_texture: Cattura un'immagine di texture
            show_preview: Mostra anteprima durante la scansione

        Returns:
            Tuple (successo, risultato)
        """
        # Simile a capture_gray_code_scan ma usa i pattern di spostamento di fase
        # Per brevità, questo metodo non è implementato completamente

        logger.warning("Il metodo capture_phase_shift_scan è uno stub")

        return False, ProcessingResult()

    def process_saved_images(
            self,
            image_paths: List[str],
            pattern_type: PatternType,
            direction: PatternDirection = PatternDirection.BOTH,
            texture_path: Optional[str] = None
    ) -> Tuple[bool, ProcessingResult]:
        """
        Elabora immagini salvate su disco.

        Args:
            image_paths: Lista di percorsi delle immagini
            pattern_type: Tipo di pattern
            direction: Direzione dei pattern
            texture_path: Percorso dell'immagine di texture

        Returns:
            Tuple (successo, risultato)
        """
        # Carica le immagini
        images = []
        for path in image_paths:
            try:
                img = cv2.imread(path)
                if img is None:
                    logger.error(f"Impossibile caricare l'immagine {path}")
                    continue

                images.append(img)
            except Exception as e:
                logger.error(f"Errore durante il caricamento dell'immagine {path}: {e}")

        if len(images) < 2:
            logger.error("Troppe poche immagini caricate")
            return False, ProcessingResult()

        # Carica immagine di texture
        texture = None
        if texture_path:
            try:
                texture = cv2.imread(texture_path)
                if texture is None:
                    logger.warning(f"Impossibile caricare l'immagine di texture {texture_path}")
            except Exception as e:
                logger.error(f"Errore durante il caricamento dell'immagine di texture: {e}")

        # Elabora le immagini in base al tipo di pattern
        result = ProcessingResult()

        if pattern_type == PatternType.GRAY_CODE:
            # Calcola il numero di bit
            n_bits = len(images) // 2

            # Separa immagini normali e invertite
            normal_images = images[0:n_bits]
            inverted_images = images[n_bits:2 * n_bits]

            # Decodifica i pattern
            decoded_x, decoded_y, mask = self.processor.decode_gray_code(
                normal_images, inverted_images, direction
            )

            if mask is None or np.sum(mask) == 0:
                logger.error("Decodifica fallita: nessun punto valido")
                return False, result

            logger.info(f"Decodifica completata: {np.sum(mask)} punti validi")

            # Triangola i punti 3D
            result = self.processor.triangulate_points(
                decoded_x, decoded_y, mask, texture
            )

        elif pattern_type == PatternType.PHASE_SHIFT:
            # Decodifica i pattern
            decoded_x, decoded_y, mask = self.processor.decode_phase_shift(
                images, direction
            )

            if mask is None or np.sum(mask) == 0:
                logger.error("Decodifica fallita: nessun punto valido")
                return False, result

            logger.info(f"Decodifica completata: {np.sum(mask)} punti validi")

            # Triangola i punti 3D
            result = self.processor.triangulate_points(
                decoded_x, decoded_y, mask, texture
            )

        else:
            logger.error(f"Tipo di pattern non supportato: {pattern_type}")
            return False, result

        if not result.has_point_cloud():
            logger.error("Triangolazione fallita: nessun punto 3D")
            return False, result

        logger.info(f"Triangolazione completata: {result.num_points} punti 3D")

        # Aggiungi metadati
        result.num_frames = len(images)
        result.capture_params = {
            "pattern_type": pattern_type.value,
            "direction": direction.value,
            "num_images": len(images)
        }

        return True, result