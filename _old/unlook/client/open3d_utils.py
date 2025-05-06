"""
Utilità per l'integrazione con Open3D per visualizzazione e processamento di nuvole di punti.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

# Import condizionale di Open3D (potrebbe non essere installato)
try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

from .processing import ProcessingResult

logger = logging.getLogger(__name__)


class Open3DWrapper:
    """Wrapper per funzionalità di Open3D."""

    def __init__(self):
        """Inizializza il wrapper."""
        if not OPEN3D_AVAILABLE:
            logger.warning("Open3D non è disponibile. Installalo con: pip install open3d")
            raise ImportError("Open3D non disponibile")

    @staticmethod
    def process_result_to_pointcloud(result: ProcessingResult) -> "o3d.geometry.PointCloud":
        """
        Converte un risultato di elaborazione in una nuvola di punti Open3D.

        Args:
            result: Risultato dell'elaborazione

        Returns:
            Nuvola di punti Open3D
        """
        if not result.has_point_cloud():
            raise ValueError("Il risultato non contiene una nuvola di punti")

        # Crea una nuvola di punti Open3D
        pcd = o3d.geometry.PointCloud()

        # Imposta i punti
        pcd.points = o3d.utility.Vector3dVector(result.point_cloud)

        # Imposta i colori se disponibili
        if result.colors is not None:
            # Converti in float [0, 1]
            colors = result.colors.astype(np.float32) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Calcola le normali se non sono disponibili
        if result.normals is None:
            logger.info("Calcolo delle normali dalla nuvola di punti...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(100)
        else:
            pcd.normals = o3d.utility.Vector3dVector(result.normals)

        return pcd

    @staticmethod
    def points_to_pointcloud(points: np.ndarray, colors: Optional[np.ndarray] = None) -> "o3d.geometry.PointCloud":
        """
        Converte punti 3D in una nuvola di punti Open3D.

        Args:
            points: Array numpy di punti (N, 3)
            colors: Array numpy di colori (N, 3), opzionale

        Returns:
            Nuvola di punti Open3D
        """
        if points.shape[1] != 3:
            raise ValueError("I punti devono essere un array (N, 3)")

        # Crea una nuvola di punti Open3D
        pcd = o3d.geometry.PointCloud()

        # Imposta i punti
        pcd.points = o3d.utility.Vector3dVector(points)

        # Imposta i colori se disponibili
        if colors is not None:
            if colors.shape[0] != points.shape[0]:
                raise ValueError("Il numero di colori deve corrispondere al numero di punti")

            # Converti in float [0, 1] se necessario
            if colors.dtype == np.uint8:
                colors = colors.astype(np.float32) / 255.0

            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    @staticmethod
    def visualize_pointcloud(pcd: "o3d.geometry.PointCloud", window_name: str = "UnLook Point Cloud"):
        """
        Visualizza una nuvola di punti Open3D.

        Args:
            pcd: Nuvola di punti Open3D
            window_name: Nome della finestra di visualizzazione
        """
        # Crea un visualizzatore
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)

        # Aggiungi la nuvola di punti
        vis.add_geometry(pcd)

        # Opzioni di rendering
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 1.0

        # Esegui la visualizzazione
        vis.run()
        vis.destroy_window()

    @staticmethod
    def filter_statistical_outliers(pcd: "o3d.geometry.PointCloud", nb_neighbors: int = 20,
                                    std_ratio: float = 2.0) -> "o3d.geometry.PointCloud":
        """
        Filtra i valori anomali statistici da una nuvola di punti.

        Args:
            pcd: Nuvola di punti Open3D
            nb_neighbors: Numero di vicini da considerare
            std_ratio: Rapporto di deviazione standard

        Returns:
            Nuvola di punti filtrata
        """
        logger.info(f"Filtro statistico: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")

        # Applica il filtro
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        logger.info(f"Punti rimossi: {len(pcd.points) - len(ind)}")

        return cl

    @staticmethod
    def filter_radius_outliers(pcd: "o3d.geometry.PointCloud", nb_points: int = 16,
                               radius: float = 0.05) -> "o3d.geometry.PointCloud":
        """
        Filtra i valori anomali basati sul raggio da una nuvola di punti.

        Args:
            pcd: Nuvola di punti Open3D
            nb_points: Numero minimo di punti nel raggio
            radius: Raggio di ricerca

        Returns:
            Nuvola di punti filtrata
        """
        logger.info(f"Filtro raggio: nb_points={nb_points}, radius={radius}")

        # Applica il filtro
        cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)

        logger.info(f"Punti rimossi: {len(pcd.points) - len(ind)}")

        return cl

    @staticmethod
    def voxel_downsample(pcd: "o3d.geometry.PointCloud", voxel_size: float = 0.01) -> "o3d.geometry.PointCloud":
        """
        Esegue il downsampling di una nuvola di punti utilizzando la tecnica del voxel.

        Args:
            pcd: Nuvola di punti Open3D
            voxel_size: Dimensione del voxel

        Returns:
            Nuvola di punti sottocampionata
        """
        logger.info(f"Voxel downsampling: voxel_size={voxel_size}")

        # Applicazione del voxel downsampling
        downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)

        logger.info(f"Punti originali: {len(pcd.points)}, punti dopo downsampling: {len(downsampled.points)}")

        return downsampled

    @staticmethod
    def create_mesh_from_pointcloud(pcd: "o3d.geometry.PointCloud", method: str = "poisson",
                                    depth: int = 8) -> "o3d.geometry.TriangleMesh":
        """
        Crea una mesh da una nuvola di punti.

        Args:
            pcd: Nuvola di punti Open3D
            method: Metodo di ricostruzione ('poisson' o 'ball_pivoting')
            depth: Profondità dell'ottavalbero per Poisson

        Returns:
            Mesh triangolare
        """
        # Assicurati che le normali siano state calcolate
        if len(pcd.normals) == 0:
            logger.info("Calcolo delle normali per la ricostruzione della mesh...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(100)

        # Ricostruzione della mesh
        if method.lower() == "poisson":
            logger.info(f"Ricostruzione Poisson: depth={depth}")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

            # Opzionale: rimuovere i triangoli con bassa densità
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)

        elif method.lower() == "ball_pivoting":
            # Stima il raggio in base alla densità della nuvola di punti
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 3 * avg_dist

            logger.info(f"Ricostruzione Ball Pivoting: radius=[{radius}, {2 * radius}, {4 * radius}]")
            radii = [radius, 2 * radius, 4 * radius]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        else:
            raise ValueError(f"Metodo non supportato: {method}. Usa 'poisson' o 'ball_pivoting'")

        # Informazioni sulla mesh
        logger.info(f"Mesh creata: {len(mesh.triangles)} triangoli, {len(mesh.vertices)} vertici")

        return mesh

    @staticmethod
    def save_mesh(mesh: "o3d.geometry.TriangleMesh", filepath: str) -> bool:
        """
        Salva una mesh in un file.

        Args:
            mesh: Mesh triangolare Open3D
            filepath: Percorso del file

        Returns:
            True se il salvataggio ha successo, False altrimenti
        """
        try:
            # Ottieni l'estensione del file
            import os
            _, ext = os.path.splitext(filepath)

            # Verifica se l'estensione è supportata
            supported_extensions = [".obj", ".ply", ".stl", ".off", ".gltf"]
            if ext.lower() not in supported_extensions:
                logger.warning(f"Estensione non supportata: {ext}. Uso .ply come fallback.")
                filepath = filepath + ".ply"

            # Salva la mesh
            o3d.io.write_triangle_mesh(filepath, mesh)
            logger.info(f"Mesh salvata in: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Errore durante il salvataggio della mesh: {e}")
            return False

    @staticmethod
    def save_pointcloud(pcd: "o3d.geometry.PointCloud", filepath: str) -> bool:
        """
        Salva una nuvola di punti in un file.

        Args:
            pcd: Nuvola di punti Open3D
            filepath: Percorso del file

        Returns:
            True se il salvataggio ha successo, False altrimenti
        """
        try:
            # Ottieni l'estensione del file
            import os
            _, ext = os.path.splitext(filepath)

            # Verifica se l'estensione è supportata
            supported_extensions = [".ply", ".pcd", ".xyz"]
            if ext.lower() not in supported_extensions:
                logger.warning(f"Estensione non supportata: {ext}. Uso .ply come fallback.")
                filepath = filepath + ".ply"

            # Salva la nuvola di punti
            o3d.io.write_point_cloud(filepath, pcd)
            logger.info(f"Nuvola di punti salvata in: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Errore durante il salvataggio della nuvola di punti: {e}")
            return False