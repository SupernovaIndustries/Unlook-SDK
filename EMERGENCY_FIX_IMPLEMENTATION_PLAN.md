# PIANO IMPLEMENTAZIONE FIX URGENTI - UNLOOK SDK

## PRIORITÀ 1: FIX IMMEDIATE (Da fare SUBITO)

### 1. DISABILITARE TUTTI I FILTRI AGGRESSIVI

**File**: `stereobm_surface_reconstructor.py`

#### A. In `compute_surface_disparity()` (linea ~256):
```python
# COMMENTARE:
# disparity = cv2.medianBlur(disparity, 5)  # DISABILITATO - rimuove dettagli
```

#### B. Parametri StereoBM (linea ~264):
```python
# VECCHI (troppo permissivi)
stereo.setTextureThreshold(1)
stereo.setUniquenessRatio(5)
stereo.setSpeckleWindowSize(100)

# NUOVI (bilanciati per structured light)
stereo.setTextureThreshold(10)     # Filtra solo vero rumore
stereo.setUniquenessRatio(15)      # Riduce match ambigui
stereo.setSpeckleWindowSize(50)    # Preserva dettagli piccoli
stereo.setNumDisparities(160)      # Range più ampio (era 96)
```

### 2. FIX VALID DISPARITY RANGE

**In `triangulate_points()` (linea ~726):**
```python
# VECCHIO (troppo restrittivo)
valid_disparity = (disparity > 16) & (disparity < 112)

# NUOVO (adattivo)
valid_disparity = (disparity > 0) & (~np.isnan(disparity))
```

### 3. DISABILITARE OBJECT DETECTION

**In `triangulate_points()` (linea ~750):**
```python
# TEMPORANEO - disabilita completamente
# if self.enable_object_detection:
#     points_3d = self._post_process_points(...)
# SKIP DIRETTAMENTE A:
return points_3d
```

### 4. FIX DEBUG IMAGE SAVING

**Nuovo metodo robusto:**
```python
def _save_debug_image_robust(self, image, filepath, description=""):
    """Salvataggio robusto che funziona sempre"""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Gestisci diversi tipi di immagini
        if isinstance(image, np.ndarray):
            if image.dtype in [np.float32, np.float64]:
                # Normalizza float a uint8
                img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                image = img_norm.astype(np.uint8)
            
            # Applica colormap se grayscale
            if len(image.shape) == 2:
                image_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            else:
                image_color = image
                
            # Salva con OpenCV
            success = cv2.imwrite(str(filepath), image_color)
            
            if success:
                logger.info(f"✓ Saved debug: {filepath.name} - {description}")
            else:
                # Fallback con PIL
                from PIL import Image
                if len(image.shape) == 2:
                    Image.fromarray(image, mode='L').save(str(filepath))
                else:
                    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(str(filepath))
                logger.info(f"✓ Saved debug (PIL): {filepath.name}")
                
    except Exception as e:
        logger.error(f"✗ Failed to save {filepath.name}: {e}")
```

## PRIORITÀ 2: MULTI-FRAME FUSION SEMPLIFICATO

### Sostituire `_fuse_disparity_maps()` con versione semplice:

```python
def _fuse_disparity_maps_simple(self, disparity_maps, valid_maps):
    """Fusione semplice che preserva dettagli"""
    if not disparity_maps:
        return None
        
    # Stack tutte le disparity
    disp_stack = np.stack(disparity_maps, axis=-1)
    
    # Usa MEDIAN (robusto e preserva edges)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fused = np.nanmedian(disp_stack, axis=-1)
    
    # Riempi buchi con nearest neighbor (no smoothing!)
    if np.any(np.isnan(fused)):
        from scipy.ndimage import distance_transform_edt
        mask = ~np.isnan(fused)
        indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        fused = fused[tuple(indices)]
    
    return fused
```

## PRIORITÀ 3: PARAMETRI SGBM PER STRUCTURED LIGHT

### In `compute_advanced_surface_disparity()`:

```python
# Parametri ottimizzati per phase-shift patterns
stereo_sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=256,       # Range completo (era 160)
    blockSize=5,              # Più piccolo per dettagli (era 7)
    P1=8 * 3 * 5**2,         # Ricalcolato
    P2=32 * 3 * 5**2,
    disp12MaxDiff=2,
    uniquenessRatio=10,       # Bilanciato (era 10)
    speckleWindowSize=50,     # Più piccolo (era 150)
    speckleRange=2,           # Più stretto (era 8)
    preFilterCap=31,          # Ridotto (era 63)
    mode=cv2.STEREO_SGBM_MODE_HH
)
```

## PRIORITÀ 4: TEST IMMEDIATO

### Comando test semplificato:
```bash
# Test con MINIMO processing
.venv/Scripts/python.exe unlook/examples/scanning/process_offline.py \
  --input unlook/examples/scanning/captured_data/test1_2k/20250603_201954 \
  --surface-reconstruction \
  --no-phase-optimization \
  --no-ndr \
  --debug \
  --save-intermediate
```

## CHECKLIST IMPLEMENTAZIONE

- [ ] Disabilita median blur in `compute_surface_disparity()`
- [ ] Aggiorna parametri StereoBM
- [ ] Rimuovi valid disparity range restrittivo
- [ ] Disabilita object detection
- [ ] Implementa debug saving robusto
- [ ] Semplifica multi-frame fusion
- [ ] Test con minimal processing
- [ ] Verifica che depth map mostri oggetto
- [ ] Verifica che point cloud corrisponda

## RISULTATI ATTESI

Dopo queste fix:
1. **Disparity map**: Dovrebbe mostrare chiaramente la forma dell'oggetto
2. **Depth map**: Dovrebbe essere una vera depth map con gradienti realistici
3. **Point cloud**: 5000-10000 punti che rappresentano l'oggetto
4. **Debug images**: Tutte salvate correttamente

## NOTA IMPORTANTE

Queste sono fix di **emergenza** per far funzionare il sistema. Dopo aver verificato che funziona, si può gradualmente riabilitare filtering selettivo dove migliora veramente la qualità.