# Files da Pulire - Metodi Stereo Obsoleti

## ❌ Files da Eliminare (Metodi Stereo per Pattern Sinusoidali)

### Root Directory
- `demo_quality_multi_frame.py` - Demo stereo obsoleto  
- `demo_sgbm_only.py` - Test StereoSGBM obsoleto
- `balanced_stereo_reconstructor.py` - Reconstructor stereo obsoleto
- `ultra_advanced_stereo_reconstructor.py` - Reconstructor stereo avanzato obsoleto
- `advanced_disparity_analyzer.py` - Analyzer disparità obsoleto
- `compare_reconstruction_methods.py` - Comparazione metodi obsoleta

### Research Files (Spostare in `/archive/`)
- `PHASE_SHIFT_STRUCTURED_LIGHT_RESEARCH.md` - Ricerca (da archiviare)
- `ADVANCED_STEREO_OPTIMIZATION_*.md` - Ricerca stereo (da archiviare)
- `STEREO_*.md` - Documenti stereo (da archiviare) 
- `DEEP_STEREO_RESEARCH_FINDINGS.md` - Ricerca stereo (da archiviare)

### Examples/Scanning (Files di Test Obsoleti)
- `unlook/examples/scanning/depth_map_diagnostic.py` - Diagnostica stereo obsoleta
- `unlook/examples/scanning/enhanced_3d_scanning_pipeline_v2.py` - Pipeline stereo obsoleta
- `unlook/examples/scanning/enhanced_scanner_modular.py` - Scanner stereo obsoleto

### Client/Scanning/Reconstruction (Moduli Stereo Obsoleti)
- `unlook/client/scanning/reconstruction/stereobm_surface_reconstructor.py` - MANTENERE (per Gray code)
- `unlook/client/scanning/reconstruction/enhanced_*` - Moduli stereo avanzati (da valutare)
- `unlook/client/scanning/reconstruction/improved_*` - Moduli stereo migliorati (da valutare)

## ✅ Files da Mantenere (Compatibilità)

### Stereo per Gray Code (NECESSARI)
- `unlook/client/scanning/reconstruction/stereobm_surface_reconstructor.py` - Per Gray code patterns
- `unlook/client/scanning/advanced_stereo_matcher.py` - Per Gray code patterns
- `process_offline.py` con `--surface-reconstruction` - Per Gray code

### Pattern Generation
- `unlook/client/scanning/patterns/enhanced_phaseshift.py` - DA AGGIORNARE per sinusoidali
- `unlook/client/scanning/pattern_manager.py` - AGGIORNATO ✅

## 🔄 Files da Aggiornare

### Capture Patterns
- `unlook/examples/scanning/capture_patterns.py` - AGGIORNARE per pattern sinusoidali

### Server
- Server pattern generation - AGGIORNARE per supportare `sinusoidal_pattern`

### Process Offline  
- `process_offline.py` - AGGIORNATO con `--phase-shift` ✅

## 📁 Struttura Finale Pulita

```
unlook/
├── client/
│   ├── scanning/
│   │   ├── calibration/
│   │   │   └── projector_calibration.py ✅ NEW
│   │   ├── reconstruction/
│   │   │   ├── phase_shift_reconstructor.py ✅ NEW
│   │   │   └── stereobm_surface_reconstructor.py (per Gray code)
│   │   ├── patterns/
│   │   │   └── enhanced_phaseshift.py (aggiornato)
│   │   └── pattern_manager.py ✅ UPDATED
│   └── projector/ (unchanged)
├── examples/
│   ├── calibration/
│   │   └── calibrate_projector_camera.py ✅ NEW
│   └── scanning/
│       ├── capture_patterns.py (da aggiornare)
│       ├── process_offline.py ✅ UPDATED  
│       └── process_phase_shift_offline.py ✅ NEW (da creare)
```