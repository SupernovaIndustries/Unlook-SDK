"""
Modulo per il processing avanzato di mesh e nuvole di punti.
Fornisce integrazioni con librerie di elaborazione 3D come PyMeshLab e trimesh.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from .processing import ProcessingResult

logger = logging.getLogger(__name__)


class MeshProcessor:
    """Base class per il processamento di mesh."""

    def __init__(self):
        """Inizializza il processore di mesh."""
        pass

    def process_point_cloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None) -> Any:
        """
        Processa una nuvola di punti.

        Args:
            points: Punti 3D come array numpy (N, 3)
            colors: Colori come array numpy (N, 3), opzionale

        Returns:
            Oggetto mesh elaborato
        """
        raise NotImplementedError("Metodo non implementato")

    def process_result(self, result: ProcessingResult) -> Any:
        """
        Processa un risultato di elaborazione.

        Args:
            result: Risultato dell'elaborazione

        Returns:
            Oggetto mesh elaborato
        """
        if not result.has_point_cloud():
            raise ValueError("Il risultato non contiene una nuvola di punti")

        return self.process_point_cloud(result.point_cloud, result.colors)

    def save_mesh(self, mesh: Any, filepath: str) -> bool:
        """
        Salva una mesh su file.

        Args:
            mesh: Oggetto mesh
            filepath: Percorso del file

        Returns:
            True se il salvataggio ha successo, False altrimenti
        """
        raise NotImplementedError("Metodo non implementato")


class PyMeshLabProcessor(MeshProcessor):
    """Processor di mesh basato su PyMeshLab."""

    def __init__(self):
        """Inizializza il processore PyMeshLab."""
        super().__init__()

        # Import condizionale di PyMeshLab
        try:
            import pymeshlab
            self.pymeshlab = pymeshlab
            self.available = True
        except ImportError:
            logger.warning("PyMeshLab non è disponibile. Installalo con: pip install pymeshlab")
            self.available = False

    def process_point_cloud(
            self,
            points: np.ndarray,
            colors: Optional[np.ndarray] = None,
            method: str = "ball_pivoting",
            point_cloud_simplification: bool = True,
            simplification_voxel: float = 0.5,
            reconstruction_depth: int = 8,
            mesh_cleaning: bool = True,
            mesh_smoothing: bool = True,
            smoothing_iterations: int = 3
    ) -> Any:
        """
        Processa una nuvola di punti con PyMeshLab.

        Args:
            points: Punti 3D come array numpy (N, 3)
            colors: Colori come array numpy (N, 3), opzionale
            method: Metodo di ricostruzione ('ball_pivoting' o 'poisson')
            point_cloud_simplification: Applica semplificazione della nuvola di punti
            simplification_voxel: Dimensione voxel per simplificazione
            reconstruction_depth: Profondità ricostruzione per Poisson
            mesh_cleaning: Applica pulizia della mesh
            mesh_smoothing: Applica smoothing della mesh
            smoothing_iterations: Numero di iterazioni di smoothing

        Returns:
            Oggetto MeshSet di PyMeshLab
        """
        if not self.available:
            raise ImportError("PyMeshLab non è disponibile")

        # Crea un MeshSet vuoto
        ms = self.pymeshlab.MeshSet()

        # Crea una nuvola di punti
        if colors is not None:
            # Assicurati che i colori siano nel formato corretto [0, 255]
            if colors.dtype != np.uint8:
                if colors.max() <= 1.0:
                    colors = (colors * 255).astype(np.uint8)
                else:
                    colors = colors.astype(np.uint8)

            mesh = self.pymeshlab.Mesh(points, colors=colors)
        else:
            mesh = self.pymeshlab.Mesh(points)

        # Aggiungi la nuvola di punti al MeshSet
        ms.add_mesh(mesh, "original_points")

        # Log delle informazioni iniziali
        logger.info(f"Nuvola di punti caricata: {len(points)} punti")

        # 1. Semplificazione della nuvola di punti (opzionale)
        if point_cloud_simplification:
            logger.info(f"Semplificazione della nuvola di punti (voxel size: {simplification_voxel})")
            ms.apply_filter('point_cloud_simplification', threshold=simplification_voxel)
            logger.info(f"Nuvola di punti dopo semplificazione: {ms.current_mesh().vertex_number()} punti")

        # 2. Calcola le normali se necessario per la ricostruzione
        if method == "poisson":
            logger.info("Calcolo delle normali per ricostruzione Poisson")
            ms.apply_filter('compute_normals_for_point_sets', k=10)

        # 3. Ricostruzione della superficie
        if method == "ball_pivoting":
            logger.info("Ricostruzione Ball Pivoting")
            # Stima automatica del raggio
            try:
                auto_ball_radius = ms.compute_geometric_measures()["avg_mesh_resolution"] * 2
                logger.info(f"Raggio automatico: {auto_ball_radius}")
                ms.apply_filter('surface_reconstruction_ball_pivoting', ballradius=auto_ball_radius)
            except Exception as e:
                logger.warning(f"Errore nella stima automatica del raggio: {e}")
                logger.info("Utilizzo di parametri di default per Ball Pivoting")
                ms.apply_filter('surface_reconstruction_ball_pivoting')
        elif method == "poisson":
            logger.info(f"Ricostruzione Poisson: depth={reconstruction_depth}")
            ms.apply_filter('surface_reconstruction_screened_poisson', depth=reconstruction_depth)

            # Rimuovi i triangoli con bassa confidenza
            ms.apply_filter('select_faces_by_quality_threshold', threshold=0.1, percentile=True, pickselection=False)
            ms.apply_filter('delete_selected_faces')
        else:
            raise ValueError(f"Metodo di ricostruzione non supportato: {method}")

        # 4. Pulizia della mesh (opzionale)
        if mesh_cleaning:
            logger.info("Pulizia della mesh")
            # Rimuovi facce isolate
            ms.apply_filter('remove_isolated_pieces_wrt_face_num', mincomponentsize=20)
            # Rimuovi vertici non referenziati
            ms.apply_filter('remove_unreferenced_vertices')
            # Ripara non-manifold
            ms.apply_filter('repair_non_manifold_edges')
            # Chiudi buchi (se ci sono)
            ms.apply_filter('close_holes', maxholesize=10)

        # 5. Smoothing della mesh (opzionale)
        if mesh_smoothing:
            logger.info(f"Smoothing della mesh: {smoothing_iterations} iterazioni")
            ms.apply_filter('laplacian_smooth', iterations=smoothing_iterations)

        # Log delle informazioni finali
        logger.info(
            f"Mesh finale: {ms.current_mesh().vertex_number()} vertici, {ms.current_mesh().face_number()} facce")

        return ms

    def save_mesh(self, mesh: Any, filepath: str) -> bool:
        """
        Salva una mesh su file.

        Args:
            mesh: Oggetto MeshSet di PyMeshLab
            filepath: Percorso del file

        Returns:
            True se il salvataggio ha successo, False altrimenti
        """
        if not self.available:
            raise ImportError("PyMeshLab non è disponibile")

        try:
            # Verifica che mesh sia un MeshSet
            if not isinstance(mesh, self.pymeshlab.MeshSet):
                raise ValueError("mesh deve essere un oggetto MeshSet di PyMeshLab")

            # Salva la mesh corrente
            mesh.save_current_mesh(filepath)
            logger.info(f"Mesh salvata in: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Errore durante il salvataggio della mesh: {e}")
            return False


class TrimeshProcessor(MeshProcessor):
    """Processor di mesh basato su trimesh."""

    def __init__(self):
        """Inizializza il processore trimesh."""
        super().__init__()

        # Import condizionale di trimesh
        try:
            import trimesh
            self.trimesh = trimesh
            self.available = True
        except ImportError:
            logger.warning("trimesh non è disponibile. Installalo con: pip install trimesh")
            self.available = False

    def process_point_cloud(
            self,
            points: np.ndarray,
            colors: Optional[np.ndarray] = None,
            method: str = "convex_hull",
            sampling_factor: float = 1.0,
            cleaning: bool = True,
            smoothing: bool = True
    ) -> Any:
        """
        Processa una nuvola di punti con trimesh.

        Args:
            points: Punti 3D come array numpy (N, 3)
            colors: Colori come array numpy (N, 3), opzionale
            method: Metodo di ricostruzione ('convex_hull' o 'alpha')
            sampling_factor: Fattore di campionamento dei punti
            cleaning: Applica pulizia della mesh
            smoothing: Applica smoothing della mesh

        Returns:
            Oggetto Trimesh
        """
        if not self.available:
            raise ImportError("trimesh non è disponibile")

        # Crea una nuvola di punti
        cloud = self.trimesh.points.PointCloud(points, colors=colors)

        # Log delle informazioni
        logger.info(f"Nuvola di punti caricata: {len(points)} punti")

        # Applica campionamento (se richiesto)
        if sampling_factor < 1.0:
            # Calcola il numero di punti da mantenere
            n_keep = int(len(points) * sampling_factor)
            if n_keep < 100:
                n_keep = min(100, len(points))

            # Campiona punti
            indices = np.random.choice(len(points), n_keep, replace=False)
            cloud = self.trimesh.points.PointCloud(points[indices], colors=None if colors is None else colors[indices])
            logger.info(f"Nuvola di punti dopo campionamento: {n_keep} punti")

        # Ricostruzione della superficie
        if method == "convex_hull":
            logger.info("Ricostruzione convex hull")
            mesh = cloud.convex_hull
        elif method == "alpha":
            # Nota: richiede scipy
            try:
                from scipy.spatial import Delaunay

                # Stima il valore alpha basato sulla distanza media tra punti vicini
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=2).fit(points)
                distances, _ = nbrs.kneighbors(points)
                alpha = distances[:, 1].mean() * 2.0

                logger.info(f"Ricostruzione alpha shape (alpha={alpha:.6f})")

                # Crea una triangolazione Delaunay
                tri = Delaunay(points)

                # Filtra i simplessi per creare una alpha shape
                mesh = self.trimesh.Trimesh(points, tri.simplices)

                # Usa la connettività di facce per rimuovere facce con spigoli lunghi
                edges = mesh.edges_unique
                edge_lengths = mesh.edges_unique_length
                edges_to_remove = edge_lengths > alpha
                faces_to_remove = np.zeros(len(mesh.faces), dtype=bool)

                for i, face in enumerate(mesh.faces):
                    for j in range(3):
                        edge = sorted([face[j], face[(j + 1) % 3]])
                        edge_idx = mesh.edges_sorted.tolist().index(edge)
                        if edges_to_remove[edge_idx]:
                            faces_to_remove[i] = True
                            break

                mesh.update_faces(~faces_to_remove)

            except ImportError:
                logger.warning("scipy o sklearn non disponibili, uso convex hull come fallback")
                mesh = cloud.convex_hull
        else:
            raise ValueError(f"Metodo di ricostruzione non supportato: {method}")

        # Pulizia della mesh (opzionale)
        if cleaning and hasattr(mesh, 'remove_degenerate_faces'):
            logger.info("Pulizia della mesh")
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.fill_holes()

        # Smoothing della mesh (opzionale)
        if smoothing and hasattr(mesh, 'smooth'):
            logger.info("Smoothing della mesh")
            mesh.smooth(iterations=5)

        # Log delle informazioni finali
        logger.info(f"Mesh finale: {len(mesh.vertices)} vertici, {len(mesh.faces)} facce")

        # Trasferisci i colori se disponibili
        if colors is not None and hasattr(mesh, 'visual'):
            # Trova le corrispondenze tra i punti originali e i vertici della mesh
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            _, indices = tree.query(mesh.vertices, k=1)

            # Assegna i colori
            if colors.dtype != np.uint8:
                if colors.max() <= 1.0:
                    colors = (colors * 255).astype(np.uint8)
                else:
                    colors = colors.astype(np.uint8)

            mesh.visual.vertex_colors = colors[indices]

        return mesh

    def save_mesh(self, mesh: Any, filepath: str) -> bool:
        """
        Salva una mesh su file.

        Args:
            mesh: Oggetto Trimesh
            filepath: Percorso del file

        Returns:
            True se il salvataggio ha successo, False altrimenti
        """
        if not self.available:
            raise ImportError("trimesh non è disponibile")

        try:
            # Verifica che mesh sia un oggetto Trimesh
            if not isinstance(mesh, self.trimesh.Trimesh):
                raise ValueError("mesh deve essere un oggetto Trimesh")

            # Salva la mesh
            mesh.export(filepath)
            logger.info(f"Mesh salvata in: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Errore durante il salvataggio della mesh: {e}")
            return False