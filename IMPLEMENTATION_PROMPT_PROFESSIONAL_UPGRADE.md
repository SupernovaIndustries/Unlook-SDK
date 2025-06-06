# PROMPT DETTAGLIATO PER UPGRADE PROFESSIONALE UNLOOK SDK

**Data**: 6 Gennaio 2025  
**Obiettivo**: Implementare miglioramenti per raggiungere qualità scanner da 60k€  
**Baseline attuale**: 67 punti, quality score 1.4/100, disparity maps che somigliano alle foto  
**Target finale**: 5000-15000 punti, quality score 85-95/100, accuratezza sub-millimetrica

---

## ISTRUZIONI GENERALI PER L'IMPLEMENTAZIONE

Sei un esperto ingegnere di computer vision che deve upgradare il sistema Unlook SDK da qualità consumer a qualità professionale. Segui questo piano step-by-step implementando SOLO le priorità indicate. Non saltare passaggi e testa ogni modifica prima di procedere.

**FILOSOFIA**: Abbandonare gradualmente OpenCV per algoritmi state-of-the-art, mantenendo CGAL, ma StereoBM come fallback, e la fusione multi-frame.

---

## FASE 1: IMMEDIATE IMPROVEMENTS (2 settimane)

### PRIORITY 1.1: INTEGRARE ELAS STEREO LIBRARY

**TASK**: Sostituire OpenCV SGBM con libelas per 10x migliore performance

**STEPS**:
1. **Scaricare e compilare libelas**:
   ```bash
   git clone https://github.com/maiermic/elas
   cd elas
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j8
   ```

2. **Creare wrapper Python per libelas**:
   - File: `unlook/client/lib/elas_wrapper.py`
   - Implementare binding ctypes/pybind11
   - Interface compatibile con current StereoBM API

3. **Integrare in stereobm_surface_reconstructor.py**:
   ```python
   def compute_elas_disparity(self, left_rect, right_rect):
       """10x faster stereo with sub-pixel accuracy"""
       from .lib.elas_wrapper import ELASMatcher
       
       matcher = ELASMatcher(
           disp_min=0, disp_max=256,
           support_threshold=0.85,
           support_texture=10,
           candidate_stepsize=5,
           incon_window_size=5,
           incon_threshold=5,
           incon_min_support=5,
           add_corners=False,
           grid_size=20,
           beta=0.02,
           gamma=3,
           sigma=1,
           sradius=2,
           match_texture=1,
           lr_threshold=2,
           speckle_sim_threshold=1,
           speckle_size=200,
           ipol_gap_width=3,
           filter_median=False,
           filter_adaptive_mean=True,
           postprocess_only_left=True,
           subsampling=False
       )
       
       return matcher.compute(left_rect, right_rect)
   ```

4. **Aggiungere option flag**:
   - In `process_offline.py`: `--use-elas`
   - Auto-detect se ELAS disponibile
   - Fallback graceful a OpenCV

**TEST**: Verificare che ELAS produca 3-5x più punti validi di SGBM

---

### PRIORITY 1.2: BUNDLE ADJUSTMENT CON CERES SOLVER

**TASK**: Migliorare calibrazione stereo del 50% con bundle adjustment

**STEPS**:
1. **Installare Ceres Solver**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libceres-dev libceres1
   
   # Windows - build from source o vcpkg
   vcpkg install ceres[suitesparse]
   ```

2. **Creare modulo bundle adjustment**:
   - File: `unlook/client/scanning/calibration/bundle_adjustment.py`
   - Wrapper per Ceres optimization
   - Implementare stereo reprojection error

3. **Implementazione**:
   ```python
   class StereoCalibrationOptimizer:
       def __init__(self):
           self.ceres_problem = None
           
       def optimize_stereo_calibration(self, image_points_left, image_points_right, 
                                     object_points, initial_params):
           """Refine stereo calibration using bundle adjustment"""
           
           # Setup Ceres problem
           import pyceres
           problem = pyceres.Problem()
           
           # Add residuals for each observation
           for left_pts, right_pts, obj_pts in zip(image_points_left, 
                                                  image_points_right, 
                                                  object_points):
               residual = StereoReprojectionError(left_pts, right_pts, obj_pts)
               problem.add_residual_block(residual, None, 
                                        [camera_params, extrinsics])
           
           # Solve
           options = pyceres.SolverOptions()
           options.linear_solver_type = pyceres.SPARSE_SCHUR
           options.minimizer_progress_to_stdout = True
           
           summary = pyceres.solve(options, problem)
           return optimized_params, summary
   ```

4. **Integrare nella pipeline di calibrazione**:
   - Modificare `camera_calibration.py`
   - Applicare bundle adjustment dopo calibrazione iniziale
   - Verificare che reprojection error < 0.1 pixel

**TEST**: Calibrazione deve avere RMS error < 0.5 pixel (attualmente ~0.78)

---

### PRIORITY 1.3: CONFIDENCE-BASED FILTERING

**TASK**: Implementare filtering intelligente basato su confidence maps

**STEPS**:
1. **Creare confidence estimator**:
   ```python
   class ConfidenceEstimator:
       def estimate_disparity_confidence(self, disparity, left_img, right_img):
           """Multi-criteria confidence estimation"""
           
           # 1. Left-Right consistency
           lr_confidence = self.compute_lr_consistency(disparity, left_img, right_img)
           
           # 2. Texture-based confidence  
           texture_confidence = self.compute_texture_confidence(left_img)
           
           # 3. Neighbor agreement
           neighbor_confidence = self.compute_neighbor_agreement(disparity)
           
           # 4. Photometric consistency
           photo_confidence = self.compute_photometric_consistency(
               disparity, left_img, right_img)
           
           # Combine confidences
           combined = (lr_confidence * 0.3 + 
                      texture_confidence * 0.2 +
                      neighbor_confidence * 0.3 +
                      photo_confidence * 0.2)
           
           return combined
   ```

2. **Integrare nel triangulate_points**:
   - Calcolare confidence per ogni pixel
   - Filtrare punti con confidence < threshold
   - Usare confidence per weighting nella fusion

**TEST**: Punti filtrati devono avere errore medio < 50% dei punti raw

---

### PRIORITY 1.4: SUB-PIXEL INTERPOLATION ENHANCEMENT

**TASK**: Migliorare accuratezza sub-pixel oltre Birchfield-Tomasi

**STEPS**:
1. **Implementare multiple sub-pixel methods**:
   ```python
   class SubPixelRefinement:
       def refine_disparity_subpixel(self, disparity, left_img, right_img, method='enhanced'):
           if method == 'birchfield_tomasi':
               return self.birchfield_tomasi_refinement(disparity, left_img, right_img)
           elif method == 'enhanced':
               return self.enhanced_subpixel_refinement(disparity, left_img, right_img)
           elif method == 'quadratic':
               return self.quadratic_interpolation(disparity)
           
       def enhanced_subpixel_refinement(self, disparity, left_img, right_img):
           """Enhanced sub-pixel using gradient-based optimization"""
           # Implement gradient descent for sub-pixel optimization
           # Use photometric error minimization
           pass
   ```

2. **Benchmark different methods**:
   - Testare accuracy su pattern noti
   - Misurare improvement rispetto a baseline
   - Scegliere best method per production

**TEST**: Sub-pixel accuracy deve migliorare di almeno 30% rispetto a current

---

## FASE 2: ADVANCED ALGORITHMS (1 mese)

### PRIORITY 2.1: RAFT-STEREO INTEGRATION

**TASK**: Integrare state-of-the-art RAFT-Stereo per 15-20% migliore accuratezza

**STEPS**:
1. **Setup RAFT-Stereo environment**:
   ```bash
   git clone https://github.com/princeton-vl/RAFT-Stereo
   cd RAFT-Stereo
   pip install -e .
   # Download pretrained models
   wget https://www.dropbox.com/s/p3962uw0fpsgp4d/raftstereo-realtime.pth
   ```

2. **Creare RAFT-Stereo wrapper**:
   ```python
   class RAFTStereoMatcher:
       def __init__(self, model_path='models/raftstereo-realtime.pth'):
           import torch
           from raft_stereo import RAFTStereo
           
           self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           self.model = RAFTStereo(args)
           self.model.load_state_dict(torch.load(model_path))
           self.model.eval()
           
       def compute_disparity(self, left_img, right_img):
           """State-of-the-art stereo matching"""
           with torch.no_grad():
               left_tensor = self.preprocess(left_img)
               right_tensor = self.preprocess(right_img)
               
               disparity = self.model(left_tensor, right_tensor)
               return self.postprocess(disparity)
   ```

3. **Integrare come option**:
   - Flag: `--use-raft-stereo`
   - Auto-fallback se GPU non disponibile
   - Benchmark vs ELAS e OpenCV

**TEST**: RAFT-Stereo deve produrre quality score >50% migliore di SGBM

---

### PRIORITY 2.2: NDR 3.0 - NEURAL DISPARITY REFINEMENT

**TASK**: Implementare neural refinement ultra-leggero per real-time

**STEPS**:
1. **Progettare architettura ultra-light**:
   ```python
   import torch.nn as nn
   
   class UltraLightNDR(nn.Module):
       def __init__(self):
           super().__init__()
           # 8KB model for real-time processing
           self.confidence_net = nn.Sequential(
               nn.Conv2d(2, 16, 3, padding=1),
               nn.ReLU(),
               nn.Conv2d(16, 8, 3, padding=1), 
               nn.ReLU(),
               nn.Conv2d(8, 1, 1),
               nn.Sigmoid()
           )
           
           self.refinement_net = nn.Sequential(
               nn.Conv2d(3, 16, 3, padding=1),
               nn.ReLU(), 
               nn.Conv2d(16, 8, 3, padding=1),
               nn.ReLU(),
               nn.Conv2d(8, 1, 1),
               nn.Tanh()
           )
       
       def forward(self, disparity, left_img, right_img):
           # Normalize inputs
           left_norm = left_img / 255.0
           right_norm = right_img / 255.0
           disp_norm = disparity / 256.0
           
           # Estimate confidence
           confidence = self.confidence_net(torch.cat([left_norm, right_norm], 1))
           
           # Estimate refinement
           refinement = self.refinement_net(torch.cat([disp_norm, left_norm, confidence], 1))
           
           # Apply refinement
           refined_disparity = disparity + refinement * confidence * 5.0
           return refined_disparity, confidence
   ```

2. **Training pipeline self-supervised**:
   ```python
   def train_ndr_self_supervised(model, dataloader, epochs=50):
       optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
       
       for epoch in range(epochs):
           for left_img, right_img, initial_disparity in dataloader:
               # Forward pass
               refined_disp, confidence = model(initial_disparity, left_img, right_img)
               
               # Self-supervised loss: photometric consistency
               warped_right = warp_image(right_img, refined_disp)
               photo_loss = F.l1_loss(left_img, warped_right, reduction='none')
               
               # Confidence-weighted loss
               loss = (photo_loss * confidence).mean()
               
               # Backward pass
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
   ```

3. **Integrazione in pipeline**:
   - Applicare dopo disparity computation
   - Real-time inference <20ms
   - Optional flag: `--use-ndr3`

**TEST**: NDR deve migliorare quality score di 20-30%

---

### PRIORITY 2.3: MULTI-FREQUENCY PHASE ANALYSIS

**TASK**: Implementare analisi frequenza domain per phase patterns

**STEPS**:
1. **Implementare frequency domain matching**:
   ```python
   class PhaseFrequencyAnalyzer:
       def __init__(self, frequencies=[1, 2, 4, 8]):
           self.frequencies = frequencies
           
       def analyze_phase_patterns(self, phase_images):
           """Extract disparity from multiple frequency phase patterns"""
           
           phase_maps = []
           for img in phase_images:
               # Extract phase using Fourier analysis
               fft = np.fft.fft2(img)
               phase = np.angle(fft)
               phase_maps.append(phase)
           
           # Multi-frequency unwrapping for disambiguation
           unwrapped = self.unwrap_multi_frequency(phase_maps, self.frequencies)
           
           # Convert phase to disparity
           disparity = self.phase_to_disparity(unwrapped)
           
           return disparity, self.estimate_phase_quality(phase_maps)
   ```

2. **Integrare nella fusion pipeline**:
   - Usare phase analysis per weight computation
   - Migliorare fusion accuracy
   - Phase-aware correspondence matching

**TEST**: Multi-frequency deve ridurre ambiguità del 80%

---

## FASE 3: PROFESSIONAL FEATURES (2 mesi)

### PRIORITY 3.1: CGAL SURFACE RECONSTRUCTION AVANZATA

**TASK**: Implementare Advancing Front e Scale Space reconstruction

**STEPS**:
1. **Implementare Advancing Front**:
   ```cpp
   // C++ module con pybind11
   #include <CGAL/Advancing_front_surface_reconstruction.h>
   #include <CGAL/Scale_space_surface_reconstruction_3.h>
   
   class AdvancedCGALReconstructor {
   public:
       Mesh reconstruct_advancing_front(const std::vector<Point_3>& points,
                                       const std::vector<float>& confidences) {
           // Implement confidence-weighted advancing front
       }
       
       Mesh reconstruct_scale_space(const std::vector<Point_3>& points,
                                   int num_scales = 4) {
           // Multi-scale surface reconstruction
       }
   };
   ```

2. **Python wrapper**:
   ```python
   class AdvancedCGALReconstructor:
       def reconstruct_surface_advanced(self, points_3d, confidences=None, method='advancing_front'):
           """Professional-grade surface reconstruction"""
           
           if method == 'advancing_front':
               return self.cgal_lib.reconstruct_advancing_front(points_3d, confidences)
           elif method == 'scale_space':
               return self.cgal_lib.reconstruct_scale_space(points_3d)
           elif method == 'confidence_poisson':
               return self.cgal_lib.reconstruct_confidence_poisson(points_3d, confidences)
   ```

**TEST**: Mesh quality deve essere watertight con <1% holes

---

### PRIORITY 3.2: MULTI-VIEW CONSISTENCY FUSION

**TASK**: Implementare fusion avanzata con consistency checks

**STEPS**:
1. **Consistency checker**:
   ```python
   class MultiViewConsistencyChecker:
       def check_disparity_consistency(self, disparity_maps, threshold=2.0):
           """Check consistency across multiple views"""
           
           consistency_map = np.ones_like(disparity_maps[0])
           
           for i in range(len(disparity_maps)):
               for j in range(i+1, len(disparity_maps)):
                   # Compare disparities between views
                   diff = np.abs(disparity_maps[i] - disparity_maps[j])
                   consistent = diff < threshold
                   consistency_map *= consistent
           
           return consistency_map
       
       def fuse_with_consistency(self, disparity_maps, confidence_maps):
           """Consistency-aware fusion"""
           consistency = self.check_disparity_consistency(disparity_maps)
           
           # Weight by both confidence and consistency
           weights = confidence_maps * consistency
           
           # Weighted fusion
           fused = np.sum(disparity_maps * weights, axis=0) / np.sum(weights, axis=0)
           return fused
   ```

**TEST**: Consistency fusion deve ridurre outliers del 90%

---

### PRIORITY 3.3: REAL-TIME QUALITY MONITORING

**TASK**: Implementare monitoring qualità in real-time

**STEPS**:
1. **Quality metrics calculator**:
   ```python
   class RealTimeQualityMonitor:
       def compute_quality_metrics(self, points_3d, disparity_map, coverage_target=0.8):
           """Real-time quality assessment"""
           
           metrics = {}
           
           # Coverage analysis
           valid_pixels = np.sum(disparity_map > 0)
           total_pixels = disparity_map.size
           metrics['coverage'] = valid_pixels / total_pixels
           
           # Point density
           if len(points_3d) > 0:
               bounds = np.max(points_3d, axis=0) - np.min(points_3d, axis=0)
               volume = np.prod(bounds)
               metrics['density'] = len(points_3d) / volume if volume > 0 else 0
           
           # Accuracy estimation (using overlap regions)
           metrics['estimated_accuracy'] = self.estimate_accuracy(points_3d)
           
           # Overall quality score
           metrics['quality_score'] = self.compute_overall_quality(metrics)
           
           return metrics
   ```

**TEST**: Quality monitoring deve essere <5ms overhead

---

## FASE 4: OPTIMIZATION & POLISH (3 mesi)

### PRIORITY 4.1: SUB-MILLIMETER ACCURACY TUNING

**TASK**: Raggiungere accuratezza ±0.1mm attraverso fine-tuning

**STEPS**:
1. **Thermal compensation**:
   ```python
   class ThermalCompensator:
       def compensate_thermal_drift(self, calibration_params, temperature_sensor):
           """Compensate for thermal expansion/contraction"""
           
           # Model thermal effects on baseline and focal length
           temp_delta = temperature_sensor.current_temp - temperature_sensor.calibration_temp
           
           # Thermal expansion coefficients for aluminum housing
           thermal_coef = 23e-6  # per degree C
           
           # Adjust baseline
           baseline_adj = calibration_params['baseline'] * (1 + thermal_coef * temp_delta)
           
           # Adjust focal lengths
           focal_adj = calibration_params['focal_length'] * (1 + thermal_coef * temp_delta)
           
           return adjusted_params
   ```

2. **Sub-pixel calibration refinement**:
   - Calibrazione su target sub-pixel 
   - Corner detection enhancement
   - Distortion model refinement

**TEST**: Accuratezza misurata su oggetti noti deve essere ±0.1mm

---

### PRIORITY 4.2: INDUSTRIAL ROBUSTNESS (OPTIONAL BUT KEEP IT IN MIND)

**TASK**: Testare e ottimizzare per condizioni industriali

**STEPS**:
1. **Robustness testing**:
   - Varie condizioni illuminazione
   - Range temperature 10-40°C
   - Vibration resistance
   - Long-term stability

2. **Auto-calibration system**:
   ```python
   class AutoCalibrationSystem:
       def monitor_calibration_drift(self):
           """Continuous monitoring of calibration accuracy"""
           # Use known reference objects in scene
           # Detect when recalibration needed
           pass
           
       def self_calibrate(self):
           """Automatic recalibration when drift detected"""
           # Project calibration patterns
           # Perform bundle adjustment
           # Update calibration parameters
           pass
   ```

**TEST**: Sistema deve funzionare 8+ ore continue senza degrado

---

## ACCEPTANCE CRITERIA FINALI

**Per considerare l'upgrade COMPLETATO, il sistema deve raggiungere**:

### QUANTITATIVE METRICS:
- **Point density**: ≥5000 punti per scan
- **Quality score**: ≥85/100
- **Accuracy**: ±0.2mm su oggetti 50-200mm
- **Coverage**: ≥90% surface coverage
- **Speed**: ≤30 secondi per scan completo
- **Reliability**: <5% failure rate su 100 scans diversi

### QUALITATIVE METRICS:
- **Mesh quality**: Watertight, smooth surfaces
- **Detail preservation**: Feature ≥1mm visibili
- **Edge accuracy**: Sharp edges preservati
- **Texture mapping**: Se richiesto, mapping accurato
- **Industrial robustness**: 8+ ore continuous operation

### BENCHMARKING:
- **Confronto con scanner da 60k€**: ≥80% della loro accuratezza
- **Open3D compatibility**: Import/export seamless
- **Professional workflows**: Compatible con CAD software
- **Speed competitive**: ≤2x tempo scanner professionali

---

## NOTE FINALI PER L'IMPLEMENTAZIONE

1. **Testing rigoroso**: Ogni modifica deve essere testata prima di procedere
2. **Backward compatibility**: Mantenere compatibility con API esistente
3. **Performance monitoring**: Misurare performance impact di ogni enhancement
4. **Fallback graceful**: Sistema deve funzionare anche se features avanzate non disponibili
5. **Documentation**: Documentare ogni nuovo feature per users
6. **Code quality**: Mantenere standard code quality elevati

**RICORDA**: L'obiettivo è raggiungere quality scanner da 60k€ mantenendo accessibilità e open-source nature di Unlook. Ogni step deve portare miglioramento misurabile verso questo target.

Implementa methodicamente, testa rigorosamente, documenta accuratamente. Il successo si misura in qualità finale del point cloud che deve essere indistinguibile da scanner professionali.