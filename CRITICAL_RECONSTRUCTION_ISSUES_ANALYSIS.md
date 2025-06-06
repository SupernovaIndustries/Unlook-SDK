# ANALISI CRITICA PROBLEMI RICOSTRUZIONE STEREO - UNLOOK SDK

**Data**: 6 Gennaio 2025  
**Severità**: CRITICA - Point cloud non corrisponde alle foto

## 1. PROBLEMI IDENTIFICATI

### 1.1 SALVATAGGIO IMMAGINI DEBUG
**Problema**: Le immagini debug non vengono salvate correttamente
- **Causa**: Exception handling silenzioso in `save_rectification_debug()` 
- **File**: `stereobm_surface_reconstructor.py` linea 1632
- **Impatto**: Impossibile debuggare il processo

### 1.2 DISPARITY MAP TROPPO FILTRATA
**Problema**: La disparity map perde tutti i dettagli dell'oggetto
- **Cause principali**:
  ```python
  # Linea 290 - Median blur rimuove dettagli
  cv2.medianBlur(disparity, 5)
  
  # Parametri StereoBM troppo conservativi
  stereo.setTextureThreshold(1)      # Troppo basso
  stereo.setUniquenessRatio(5)       # Troppo basso
  stereo.setSpeckleWindowSize(100)   # Troppo grande
  ```
- **Impatto**: Disparity map appare uniforme invece di mostrare geometria oggetto

### 1.3 MULTI-FRAME FUSION FALLIMENTARE
**Problema**: La fusione multi-frame peggiora invece di migliorare
- **Metodo**: `_fuse_disparity_maps()` linea 2125
- **Cause**:
  - Pesi calcolati favoriscono regioni uniformi (background)
  - Kernel 5x5 per variance troppo grande, liscia i dettagli
  - Non gestisce correttamente edge cases
- **Impatto**: Oggetto sparisce nella fusione

### 1.4 DEPTH MAP NON REALISTICA
**Problema**: Depth map non assomiglia all'oggetto fotografato
- **Cause**:
  ```python
  # Valid disparity range troppo restrittivo
  valid_mask = (disparity > 16) & (disparity < 112)
  ```
  - Range fisso non adattivo per la scena
  - Filtra punti legittimi vicini/lontani
  - Post-processing applica 4 filtri che rimuovono ulteriori dettagli

### 1.5 OBJECT DETECTION AGGRESSIVA
**Problema**: "4-strategy intelligent object detection" rimuove l'oggetto stesso
- **File**: `_post_process_points()` linea 794
- **Strategie problematiche**:
  1. Pattern-based filtering - rimuove punti non-pattern
  2. Depth segmentation - assume background planare
  3. ROI filtering - taglia dati validi
  4. Statistical confidence - rimuove punti ai bordi
- **Impatto**: Da 15,000 punti → 200 punti

### 1.6 COORDINATE TRANSFORMATION ERRORS
**Problema**: Trasformazioni coordinate sbagliate
- **Cause**:
  - Q matrix scaling issues (risolto parzialmente)
  - Centering operation sposta punti incorrettamente
  - Nessun check su coordinate system consistency

## 2. ROOT CAUSE ANALYSIS

### FILOSOFIA SBAGLIATA
Il codice è ottimizzato per **riduzione del rumore** invece che per **preservazione dei dettagli**.

### PIPELINE ISSUES
1. **Over-filtering cascade**: Ogni stage filtra, l'effetto si accumula
2. **Parameter mismatch**: Parametri non ottimizzati per phase-shift patterns
3. **Invalid assumptions**: Assume background planare, oggetti compatti
4. **Missing validation**: Nessun check se filtering migliora o peggiora

## 3. SOLUZIONI PROPOSTE

### 3.1 IMMEDIATE FIXES

#### A. Disabilitare filtering aggressivo
```python
# PRIMA (troppo filtrato)
disparity = cv2.medianBlur(disparity, 5)

# DOPO (preserva dettagli)
# disparity = cv2.medianBlur(disparity, 5)  # COMMENTATO
```

#### B. Parametri StereoBM ottimizzati per structured light
```python
# Parametri per phase-shift patterns
stereo.setTextureThreshold(10)      # Era 1
stereo.setUniquenessRatio(15)       # Era 5  
stereo.setSpeckleWindowSize(50)     # Era 100
stereo.setSpeckleRange(2)           # Era 32
stereo.setMinDisparity(0)           # Full range
stereo.setNumDisparities(256)       # Era 96
```

#### C. Valid disparity range adattivo
```python
# PRIMA
valid_mask = (disparity > 16) & (disparity < 112)

# DOPO
mean_disp = np.median(disparity[disparity > 0])
valid_mask = (disparity > 0) & (disparity < mean_disp * 3)
```

### 3.2 MULTI-FRAME FUSION FIX

#### Nuovo algoritmo fusion
```python
def _fuse_disparity_maps_improved(self, disparity_maps, valid_maps):
    # 1. NON usare variance per weights
    # 2. Usare MEDIAN invece di weighted average
    # 3. Preservare high-frequency details
    
    # Stack all disparities
    disp_stack = np.stack(disparity_maps, axis=2)
    
    # Use median for robustness (preserva edges)
    fused = np.nanmedian(disp_stack, axis=2)
    
    # Fill holes with nearest valid
    # NON con smoothing
```

### 3.3 DISABLE OBJECT DETECTION

Temporaneamente disabilitare TUTTA l'object detection:
```python
# In triangulate_points()
if self.enable_object_detection and False:  # FORCE DISABLE
    points_3d = self._post_process_points(...)
```

### 3.4 DEBUG SAVING FIX

```python
def save_debug_robust(self, image, path, description):
    try:
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Force conversion to saveable format
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
            
        success = cv2.imwrite(str(path), image)
        if success:
            logger.info(f"Saved: {path}")
        else:
            logger.error(f"Failed to save: {path}")
            # Try alternative method
            from PIL import Image
            Image.fromarray(image).save(str(path))
            
    except Exception as e:
        logger.error(f"Debug save error: {e}")
        # Log but don't crash
```

## 4. PARAMETRI OTTIMALI RICERCA

### Per Phase-Shift Patterns (da letteratura):
- **Block size**: 9-15 (texture presente)
- **Uniqueness ratio**: 10-20 (riduce ambiguità)
- **Texture threshold**: 5-15 (filtra aree uniformi)
- **Speckle window**: 50-100 (bilancia rumore/dettagli)
- **Pre-filter cap**: 31-63 (preserva edges)

### Per Multi-Frame Fusion:
- **NON** usare weighted average con variance weights
- **USARE** median o trimmed mean
- **PRESERVARE** high frequency tramite detail layer separation

## 5. VALIDATION METRICS

Aggiungere metriche per validare ogni step:
```python
def compute_quality_metrics(self, data_before, data_after):
    return {
        'valid_pixels': np.sum(data_after > 0),
        'coverage': np.sum(data_after > 0) / data_after.size,
        'edge_preservation': compute_edge_similarity(data_before, data_after),
        'detail_loss': compute_frequency_loss(data_before, data_after)
    }
```

## 6. TESTING PROTOCOL

1. **Disabilitare TUTTI i filtri**
2. **Testare raw disparity** - dovrebbe mostrare oggetto
3. **Aggiungere filtri uno alla volta** - verificare miglioramento
4. **Multi-frame fusion** - deve MIGLIORARE non peggiorare
5. **Object detection** - deve preservare oggetto principale

## 7. EMERGENCY FIXES PER DEMO

Se serve risultato immediato:
1. Usa SOLO best frame (no fusion)
2. Disabilita TUTTI i filtri
3. Usa parametri permissivi per StereoBM
4. Niente object detection
5. Salva TUTTO per debug

## CONCLUSIONE

Il sistema attuale è **over-engineered** con troppi filtri che si accumulano. La filosofia deve cambiare da "rimuovi rumore" a "preserva dettagli". Phase-shift patterns già forniscono dati puliti, non servono filtri aggressivi.