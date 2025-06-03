#!/usr/bin/env python3
"""
Compare Reconstruction Methods
Confronta i diversi approcci di ricostruzione per vedere quale funziona meglio
"""

import numpy as np
import cv2
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReconstructionComparator:
    """
    Confronta diversi metodi di ricostruzione
    """
    
    def __init__(self, capture_dir="captured_data/20250531_005620"):
        self.capture_dir = Path(capture_dir)
        self.calib_file = "unlook/calibration/custom/stereo_calibration_fixed.json"
        self.output_dir = self.capture_dir / "comparison_results"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load calibration
        self.load_calibration()
        
    def load_calibration(self):
        """Carica la calibrazione corretta"""
        with open(self.calib_file, 'r') as f:
            calib = json.load(f)
        
        self.K1 = np.array(calib['camera_matrix_left'])
        self.K2 = np.array(calib['camera_matrix_right'])
        self.D1 = np.array(calib['dist_coeffs_left']).flatten()
        self.D2 = np.array(calib['dist_coeffs_right']).flatten()
        self.R = np.array(calib['R'])
        self.T = np.array(calib['T']).flatten()
        
        baseline = np.linalg.norm(self.T) * 1000.0
        logger.info(f"Calibration loaded - baseline: {baseline:.1f}mm")
        
    def rectify_images(self, left_img, right_img):
        """Rettifica le immagini stereo"""
        h, w = left_img.shape
        
        R1, R2, P1_new, P2_new, Q_new, roi1, roi2 = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, (w, h), self.R, self.T,
            alpha=0.0
        )
        
        map1x, map1y = cv2.initUndistortRectifyMap(self.K1, self.D1, R1, P1_new, (w, h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(self.K2, self.D2, R2, P2_new, (w, h), cv2.CV_32FC1)
        
        left_rect = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
        
        return left_rect, right_rect, Q_new
    
    def method_1_basic_sgbm(self, left_rect, right_rect):
        """Metodo 1: SGBM Basic (quello originale che causava problemi)"""
        logger.info("Testing Method 1: Basic SGBM...")
        
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=96,
            blockSize=9,
            P1=8 * 3 * 9**2,
            P2=32 * 3 * 9**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=2
        )
        
        disparity_raw = stereo.compute(left_rect, right_rect)
        disparity = disparity_raw.astype(np.float32) / 16.0
        
        # Basic triangulation
        points_3d = self.triangulate_basic(disparity)
        
        return points_3d, disparity
    
    def method_2_filtered_sgbm(self, left_rect, right_rect):
        """Metodo 2: SGBM con filtri morfologici"""
        logger.info("Testing Method 2: Filtered SGBM...")
        
        # SGBM più fine
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=96,
            blockSize=5,  # Più piccolo
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=5,  # Più stringente
            speckleWindowSize=100,
            speckleRange=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        disparity_raw = stereo.compute(left_rect, right_rect)
        disparity = disparity_raw.astype(np.float32) / 16.0
        
        # Applica filtri
        disparity_filtered = self.apply_morphological_filters(disparity)
        
        # Triangolazione con clustering
        points_3d = self.triangulate_with_clustering(disparity_filtered)
        
        return points_3d, disparity_filtered
    
    def method_3_stereo_bm(self, left_rect, right_rect):
        """Metodo 3: StereoBM IMPROVED (WIDE RANGE - fixes flat point cloud issue)"""
        logger.info("Testing Method 3: StereoBM IMPROVED...")
        
        # UPDATED PARAMETERS based on advanced disparity analysis
        stereo = cv2.StereoBM_create(numDisparities=144, blockSize=21)  # Wider range + larger block
        stereo.setPreFilterCap(63)        # Higher preprocessing
        stereo.setMinDisparity(0)
        stereo.setTextureThreshold(5)     # More sensitive texture detection
        stereo.setUniquenessRatio(10)     # Less strict matching
        stereo.setSpeckleWindowSize(100)  # Larger speckle window
        stereo.setSpeckleRange(4)         # Larger speckle range
        
        disparity_raw = stereo.compute(left_rect, right_rect)
        disparity = disparity_raw.astype(np.float32) / 16.0
        
        # Post-processing
        disparity_processed = self.apply_median_filter(disparity)
        
        # Triangolazione
        points_3d = self.triangulate_basic(disparity_processed)
        
        return points_3d, disparity_processed
    
    def method_4_multi_scale(self, left_rect, right_rect):
        """Metodo 4: Multi-scale approach"""
        logger.info("Testing Method 4: Multi-scale approach...")
        
        all_points = []
        
        # Prova multiple scale di immagine
        scales = [1.0, 0.75, 0.5]
        
        for scale in scales:
            if scale != 1.0:
                new_size = (int(left_rect.shape[1] * scale), int(left_rect.shape[0] * scale))
                left_scaled = cv2.resize(left_rect, new_size)
                right_scaled = cv2.resize(right_rect, new_size)
            else:
                left_scaled = left_rect
                right_scaled = right_rect
            
            # SGBM su questa scala
            num_disp = int(64 * scale)  # Scala le disparità
            if num_disp % 16 != 0:
                num_disp = ((num_disp // 16) + 1) * 16
            
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=num_disp,
                blockSize=7,
                P1=8 * 3 * 7**2,
                P2=32 * 3 * 7**2,
                uniquenessRatio=10
            )
            
            disparity_raw = stereo.compute(left_scaled, right_scaled)
            disparity = disparity_raw.astype(np.float32) / 16.0
            
            # Scala di nuovo la disparità
            if scale != 1.0:
                disparity = cv2.resize(disparity, (left_rect.shape[1], left_rect.shape[0]))
                disparity = disparity / scale  # Correggi scale della disparità
            
            # Triangola
            points = self.triangulate_basic(disparity)
            if points is not None and len(points) > 100:
                all_points.append(points)
        
        # Combina tutti i punti
        if all_points:
            combined_points = np.vstack(all_points)
            # Remove duplicates semplice
            unique_points = self.remove_duplicates_simple(combined_points)
            return unique_points, disparity
        
        return None, None
    
    def apply_morphological_filters(self, disparity):
        """Applica filtri morfologici"""
        valid_mask = disparity > 0
        
        if np.sum(valid_mask) == 0:
            return disparity
        
        # Chiusura morfologica
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        disparity_uint8 = (disparity * 4).astype(np.uint8)  # Scale per morfologia
        closed = cv2.morphologyEx(disparity_uint8, cv2.MORPH_CLOSE, kernel)
        closed = closed.astype(np.float32) / 4.0
        
        # Median filter
        median_filtered = cv2.medianBlur(closed.astype(np.uint8), 5)
        result = median_filtered.astype(np.float32)
        
        # Ripristina zero sui pixel originariamente invalidi
        result[~valid_mask] = 0
        
        return result
    
    def apply_median_filter(self, disparity):
        """Applica filtro mediano"""
        valid_mask = disparity > 0
        result = cv2.medianBlur(disparity.astype(np.uint8), 5).astype(np.float32)
        result[~valid_mask] = 0
        return result
    
    def triangulate_basic(self, disparity):
        """Triangolazione basic"""
        baseline_mm = 80.0
        focal_length = self.K1[0, 0]
        cx = self.K1[0, 2]
        cy = self.K1[1, 2]
        
        h, w = disparity.shape
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        valid_mask = disparity > 0.5
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        disp_valid = disparity[valid_mask]
        
        Z = (focal_length * baseline_mm) / disp_valid
        X = (x_valid - cx) * Z / focal_length
        Y = (y_valid - cy) * Z / focal_length
        
        points_3d = np.column_stack([X, Y, Z])
        
        # Filtro profondità
        depth_mask = (Z > 100) & (Z < 1500)
        filtered_points = points_3d[depth_mask]
        
        if len(filtered_points) == 0:
            return None
        
        return self.center_points(filtered_points)
    
    def triangulate_with_clustering(self, disparity):
        """Triangolazione con clustering semplice"""
        points_3d = self.triangulate_basic(disparity)
        
        if points_3d is None or len(points_3d) < 100:
            return points_3d
        
        # Clustering per Z (profondità)
        z_values = points_3d[:, 2]
        z_median = np.median(z_values)
        z_std = np.std(z_values)
        
        # Mantieni solo punti vicini alla profondità mediana
        depth_mask = np.abs(z_values - z_median) < (2 * z_std)
        clustered_points = points_3d[depth_mask]
        
        return clustered_points
    
    def remove_duplicates_simple(self, points, threshold=5.0):
        """Rimuove duplicati semplice"""
        if len(points) < 2:
            return points
        
        unique_points = [points[0]]
        
        for point in points[1:]:
            # Controlla se è troppo vicino ai punti già mantenuti
            is_duplicate = False
            for unique_point in unique_points[-50:]:  # Controlla solo gli ultimi 50
                distance = np.sqrt(np.sum((point - unique_point)**2))
                if distance < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_points.append(point)
        
        return np.array(unique_points)
    
    def center_points(self, points_3d):
        """Centra i punti PRESERVANDO la variazione di profondità"""
        if len(points_3d) == 0:
            return points_3d
        
        centroid = np.mean(points_3d, axis=0)
        centered_points = points_3d.copy()
        
        # Center X and Y only - PRESERVE Z variation!
        centered_points[:, 0] -= centroid[0]  # Center X
        centered_points[:, 1] -= centroid[1]  # Center Y
        # DO NOT modify Z coordinates to preserve depth variation!
        
        # Optional: shift mean Z to target depth but keep relative variation
        target_mean_z = 300.0
        z_shift = target_mean_z - centroid[2]
        centered_points[:, 2] += z_shift  # Shift all Z by same amount
        
        # Outlier removal based on 3D distance
        center = np.mean(centered_points, axis=0)
        distances = np.sqrt(np.sum((centered_points - center)**2, axis=1))
        distance_threshold = np.mean(distances) + 2 * np.std(distances)
        inlier_mask = distances < distance_threshold
        
        result = centered_points[inlier_mask]
        
        # Log depth variation to verify it's preserved
        if len(result) > 0:
            z_variation = np.max(result[:, 2]) - np.min(result[:, 2])
            logger.info(f"Depth variation preserved: {z_variation:.1f}mm")
        
        return result
    
    def save_point_cloud(self, points_3d, filename):
        """Salva la nuvola di punti"""
        if points_3d is None or len(points_3d) == 0:
            logger.warning(f"No points to save for {filename}")
            return None
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            for point in points_3d:
                f.write(f"{point[0]:.3f} {point[1]:.3f} {point[2]:.3f}\n")
        
        logger.info(f"Saved: {filename}")
        return filepath
    
    def analyze_point_cloud_quality(self, points_3d, method_name):
        """Analizza la qualità della nuvola di punti"""
        if points_3d is None or len(points_3d) == 0:
            return {
                'method': method_name,
                'points': 0,
                'quality': 0,
                'description': 'No points generated'
            }
        
        # Calcola metriche di qualità
        centroid = np.mean(points_3d, axis=0)
        
        # Compattezza (quanto sono concentrati i punti)
        distances = np.sqrt(np.sum((points_3d - centroid)**2, axis=1))
        compactness = 1.0 / (1.0 + np.std(distances))
        
        # Densità relativa
        volume = (np.max(points_3d[:,0]) - np.min(points_3d[:,0])) * \
                (np.max(points_3d[:,1]) - np.min(points_3d[:,1])) * \
                (np.max(points_3d[:,2]) - np.min(points_3d[:,2]))
        
        density = len(points_3d) / (volume + 1e-6)
        
        # Centratura (quanto è vicino a (0,0,300))
        target = np.array([0, 0, 300])
        centering_error = np.sqrt(np.sum((centroid - target)**2))
        centering_score = 1.0 / (1.0 + centering_error / 100)
        
        # Score finale
        quality_score = (compactness * 0.3 + centering_score * 0.4 + min(density/1000, 1.0) * 0.3) * 100
        
        # Descrizione qualitativa
        if quality_score > 70:
            description = "Excellent - Compact, well-centered surface"
        elif quality_score > 50:
            description = "Good - Reasonable surface structure"
        elif quality_score > 30:
            description = "Fair - Some surface features visible"
        else:
            description = "Poor - Scattered points, no clear surface"
        
        return {
            'method': method_name,
            'points': len(points_3d),
            'quality': quality_score,
            'centroid': centroid,
            'compactness': compactness,
            'density': density,
            'centering_score': centering_score,
            'description': description
        }
    
    def compare_all_methods(self):
        """Confronta tutti i metodi di ricostruzione"""
        logger.info("🔬 COMPARING RECONSTRUCTION METHODS")
        logger.info("="*60)
        
        # Trova la migliore coppia di immagini
        best_left = None
        best_right = None
        
        for left_file in self.capture_dir.glob("left_*phase_shift_f8_s0*"):
            right_file = Path(str(left_file).replace("left_", "right_"))
            if right_file.exists():
                best_left = str(left_file)
                best_right = str(right_file)
                break
        
        if not best_left:
            for left_file in self.capture_dir.glob("left_*phase_shift_f1_s0*"):
                right_file = Path(str(left_file).replace("left_", "right_"))
                if right_file.exists():
                    best_left = str(left_file)
                    best_right = str(right_file)
                    break
        
        if not best_left:
            logger.error("No suitable images found!")
            return False
        
        logger.info(f"Using: {Path(best_left).name} + {Path(best_right).name}")
        
        # Carica e rettifica immagini
        left_img = cv2.imread(best_left, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(best_right, cv2.IMREAD_GRAYSCALE)
        
        left_rect, right_rect, Q = self.rectify_images(left_img, right_img)
        
        # Testa tutti i metodi
        methods = [
            ("Basic SGBM", self.method_1_basic_sgbm),
            ("Filtered SGBM", self.method_2_filtered_sgbm),
            ("StereoBM", self.method_3_stereo_bm),
            ("Multi-Scale", self.method_4_multi_scale)
        ]
        
        results = []
        
        for method_name, method_func in methods:
            logger.info(f"\n--- Testing {method_name} ---")
            
            try:
                points_3d, disparity = method_func(left_rect, right_rect)
                
                # Salva risultato
                if points_3d is not None:
                    filename = f"method_{method_name.lower().replace(' ', '_')}.ply"
                    self.save_point_cloud(points_3d, filename)
                    
                    # Salva disparità
                    if disparity is not None:
                        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
                        disp_filename = f"disparity_{method_name.lower().replace(' ', '_')}.png"
                        cv2.imwrite(str(self.output_dir / disp_filename), disp_colored)
                
                # Analizza qualità
                quality = self.analyze_point_cloud_quality(points_3d, method_name)
                results.append(quality)
                
                logger.info(f"Result: {quality['points']} points, quality={quality['quality']:.1f}")
                
            except Exception as e:
                logger.error(f"Error in {method_name}: {e}")
                results.append({
                    'method': method_name,
                    'points': 0,
                    'quality': 0,
                    'description': f'Error: {e}'
                })
        
        # Confronto finale
        logger.info("\n" + "="*60)
        logger.info("📊 FINAL COMPARISON RESULTS")
        logger.info("="*60)
        
        # Ordina per qualità
        results.sort(key=lambda x: x['quality'], reverse=True)
        
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result['method']}")
            logger.info(f"   Points: {result['points']}")
            logger.info(f"   Quality: {result['quality']:.1f}/100")
            logger.info(f"   Status: {result['description']}")
            if 'centroid' in result:
                c = result['centroid']
                logger.info(f"   Centroid: ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})mm")
            logger.info("")
        
        # Raccomandazione
        best_method = results[0]
        logger.info(f"🏆 RECOMMENDED METHOD: {best_method['method']}")
        logger.info(f"   Best quality score: {best_method['quality']:.1f}/100")
        logger.info(f"   {best_method['description']}")
        
        return True

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        capture_dir = sys.argv[1]
    else:
        capture_dir = "captured_data/20250531_005620"
    
    if not Path(capture_dir).exists():
        print(f"❌ Capture directory not found: {capture_dir}")
        return
    
    comparator = ReconstructionComparator(capture_dir)
    success = comparator.compare_all_methods()
    
    if success:
        print("\n🎉 COMPARISON COMPLETED!")
        print("Check the comparison_results folder for all method outputs")
        print("Use the recommended method for best surface reconstruction!")
    else:
        print("\n❌ Comparison failed.")

if __name__ == "__main__":
    main()