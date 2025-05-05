"""
Modulo per l'esportazione dei dati 3D in vari formati.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any, Union

import numpy as np

from .processing import ProcessingResult

logger = logging.getLogger(__name__)


class ModelExporter:
    """Classe per l'esportazione di modelli 3D in vari formati."""

    def __init__(self):
        """Inizializza l'esportatore."""
        pass

    def export_ply(self, result: ProcessingResult, filepath: str, binary: bool = False) -> bool:
        """
        Esporta una nuvola di punti in formato PLY.

        Args:
            result: Risultato dell'elaborazione
            filepath: Percorso del file PLY
            binary: Se True, salva in formato binario

        Returns:
            True se esportato con successo, False altrimenti
        """
        if not result.has_point_cloud():
            logger.error("Nessuna nuvola di punti da esportare")
            return False

        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            # Calcola quali dati sono disponibili
            has_colors = result.colors is not None
            has_normals = result.normals is not None
            has_confidence = result.confidence is not None

            if binary:
                # Formato PLY binario
                with open(filepath, 'wb') as f:
                    # Scrivi header
                    header = "ply\n"
                    header += "format binary_little_endian 1.0\n"
                    header += f"element vertex {result.num_points}\n"
                    header += "property float x\n"
                    header += "property float y\n"
                    header += "property float z\n"

                    if has_colors:
                        header += "property uchar red\n"
                        header += "property uchar green\n"
                        header += "property uchar blue\n"

                    if has_normals:
                        header += "property float nx\n"
                        header += "property float ny\n"
                        header += "property float nz\n"

                    if has_confidence:
                        header += "property float confidence\n"

                    header += "end_header\n"
                    f.write(header.encode('ascii'))

                    # Prepara i dati
                    vertices = result.point_cloud.astype(np.float32)

                    # Scrivi vertici
                    for i in range(result.num_points):
                        # Coordinate
                        f.write(vertices[i].tobytes())

                        # Colori
                        if has_colors:
                            f.write(result.colors[i].astype(np.uint8).tobytes())

                        # Normali
                        if has_normals:
                            f.write(result.normals[i].astype(np.float32).tobytes())

                        # Confidenza
                        if has_confidence:
                            f.write(np.array([result.confidence[i]], dtype=np.float32).tobytes())

            else:
                # Formato PLY ASCII
                with open(filepath, 'w') as f:
                    # Scrivi header
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {result.num_points}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")

                    if has_colors:
                        f.write("property uchar red\n")
                        f.write("property uchar green\n")
                        f.write("property uchar blue\n")

                    if has_normals:
                        f.write("property float nx\n")
                        f.write("property float ny\n")
                        f.write("property float nz\n")

                    if has_confidence:
                        f.write("property float confidence\n")

                    f.write("end_header\n")

                    # Scrivi vertici
                    for i in range(result.num_points):
                        # Coordinate
                        x, y, z = result.point_cloud[i]
                        f.write(f"{x} {y} {z}")

                        # Colori
                        if has_colors:
                            r, g, b = result.colors[i]
                            f.write(f" {int(r)} {int(g)} {int(b)}")

                        # Normali
                        if has_normals:
                            nx, ny, nz = result.normals[i]
                            f.write(f" {nx} {ny} {nz}")

                        # Confidenza
                        if has_confidence:
                            f.write(f" {result.confidence[i]}")

                        f.write("\n")

            logger.info(f"Nuvola di punti esportata in {filepath}")
            return True

        except Exception as e:
            logger.error(f"Errore durante l'esportazione della nuvola di punti: {e}")
            return False

    def export_obj(self, result: ProcessingResult, filepath: str, gen_mesh: bool = False) -> bool:
        """
        Esporta una nuvola di punti in formato OBJ.

        Args:
            result: Risultato dell'elaborazione
            filepath: Percorso del file OBJ
            gen_mesh: Se True, genera una mesh dalle nuvole di punti

        Returns:
            True se esportato con successo, False altrimenti
        """
        if not result.has_point_cloud():
            logger.error("Nessuna nuvola di punti da esportare")
            return False

        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            # Verifica se sono disponibili i colori
            has_colors = result.colors is not None

            # Crea i nomi dei file
            obj_file = filepath
            mtl_file = os.path.splitext(filepath)[0] + ".mtl"
            mtl_name = os.path.splitext(os.path.basename(filepath))[0]

            # Scrivi file OBJ
            with open(obj_file, 'w') as f:
                # Riferimento al file MTL
                if has_colors:
                    f.write(f"mtllib {os.path.basename(mtl_file)}\n")

                # Scrivi vertici
                for i in range(result.num_points):
                    x, y, z = result.point_cloud[i]
                    f.write(f"v {x} {y} {z}")

                    # Aggiungi colore ai vertici (solo supporto RGB)
                    if has_colors:
                        r, g, b = result.colors[i]
                        # Normalizza a [0,1]
                        f.write(f" {r / 255.0} {g / 255.0} {b / 255.0}")

                    f.write("\n")

                # Se richiesto, genera una mesh (non implementato)
                if gen_mesh:
                    logger.warning("Generazione mesh non implementata")
                else:
                    # Usa vertici come punti
                    f.write("# Points\n")
                    if has_colors:
                        f.write(f"usemtl {mtl_name}\n")
                    f.write("p")
                    for i in range(1, result.num_points + 1):
                        f.write(f" {i}")
                    f.write("\n")

            # Se ci sono colori, scrivi file MTL
            if has_colors:
                with open(mtl_file, 'w') as f:
                    f.write(f"newmtl {mtl_name}\n")
                    f.write("Ka 1.0 1.0 1.0\n")  # Colore ambientale
                    f.write("Kd 1.0 1.0 1.0\n")  # Colore diffuso
                    f.write("Ks 0.0 0.0 0.0\n")  # Colore speculare
                    f.write("d 1.0\n")  # Opacità
                    f.write("illum 1\n")  # Modello di illuminazione

            logger.info(f"Modello esportato in {filepath}")
            return True

        except Exception as e:
            logger.error(f"Errore durante l'esportazione del modello: {e}")
            return False

    def export_xyz(self, result: ProcessingResult, filepath: str, include_colors: bool = True) -> bool:
        """
        Esporta una nuvola di punti in formato XYZ.

        Args:
            result: Risultato dell'elaborazione
            filepath: Percorso del file XYZ
            include_colors: Se True, include i colori

        Returns:
            True se esportato con successo, False altrimenti
        """
        if not result.has_point_cloud():
            logger.error("Nessuna nuvola di punti da esportare")
            return False

        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            # Verifica se sono disponibili i colori
            has_colors = include_colors and result.colors is not None

            with open(filepath, 'w') as f:
                for i in range(result.num_points):
                    x, y, z = result.point_cloud[i]
                    f.write(f"{x} {y} {z}")

                    # Aggiungi colore
                    if has_colors:
                        r, g, b = result.colors[i]
                        f.write(f" {int(r)} {int(g)} {int(b)}")

                    f.write("\n")

            logger.info(f"Nuvola di punti esportata in {filepath}")
            return True

        except Exception as e:
            logger.error(f"Errore durante l'esportazione della nuvola di punti: {e}")
            return False

    def export_metadata(self, result: ProcessingResult, filepath: str) -> bool:
        """
        Esporta i metadati della scansione in formato JSON.

        Args:
            result: Risultato dell'elaborazione
            filepath: Percorso del file JSON

        Returns:
            True se esportato con successo, False altrimenti
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            metadata = {
                "timestamp": result.timestamp,
                "num_points": result.num_points,
                "num_frames": result.num_frames,
                "scanner_uuid": result.scanner_uuid,
                "capture_params": result.capture_params
            }

            with open(filepath, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Metadati esportati in {filepath}")
            return True

        except Exception as e:
            logger.error(f"Errore durante l'esportazione dei metadati: {e}")
            return False