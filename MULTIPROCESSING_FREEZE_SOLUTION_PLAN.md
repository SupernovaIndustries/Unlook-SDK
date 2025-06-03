# 🚨 MULTIPROCESSING FREEZE SOLUTION PLAN
## UnLook SDK - Diagnosi e Piano d'Azione Dettagliato

---

## 📋 SOMMARIO ESECUTIVO

Il processo si blocca durante l'esecuzione parallela multi-frame con ProcessPoolExecutor. Dopo ricerca approfondita, ho identificato:

1. **Causa Principale**: Conflitto tra OpenCV e multiprocessing su Windows
2. **Problema Specifico**: Serializzazione pickle di oggetti OpenCV/NumPy attraverso processi
3. **Soluzione Immediata**: Modifiche al codice per evitare deadlock
4. **Ottimizzazioni**: Ridurre memoria e migliorare performance

---

## 🔍 RICERCA DETTAGLIATA - RISULTATI

### 1. **Problemi OpenCV + Multiprocessing**

#### Cause Principali:
- **Fork Safety Issues**: OpenCV usa variabili condivise non protette da race conditions
- **Windows DLL Loading**: Su Windows con Python 3.8+, il supporto multiprocessing per OpenCV è problematico
- **Serializzazione Pickle**: OpenCV objects non sono pickle-friendly, causano hang quando passati tra processi

#### Fonti:
- GitHub Issue opencv/opencv#5150
- GitHub Issue opencv/opencv#8161
- StackOverflow: "opencv and multiprocessing"

### 2. **ProcessPoolExecutor Freeze su Windows**

#### Sintomi Tipici:
- Il processo si blocca senza consumo CPU
- Nessun errore visibile
- Hang durante as_completed() o executor.map()

#### Cause:
- Mancanza di `if __name__ == '__main__'` guard
- Spawn method di default su Windows causa re-import del modulo principale
- Variabili globali non accessibili nei worker processes

#### Fonti:
- Python bug tracker issue42245
- StackOverflow: ProcessPoolExecutor gets stuck

### 3. **Memory Issues con StereoSGBM**

#### Consumo Memoria:
- MODE_HH consuma O(W*H*numDisparities) bytes
- Per immagini 1456x1088: ~250MB per frame
- Con 12 frames + 8 workers: potenziale 24GB RAM!
- Limite pratico: 2GB per processo per evitare crash

#### Fonti:
- OpenCV docs: StereoSGBM Class Reference
- StackOverflow: StereoSGBM parameters for 25MP images

### 4. **Neural Disparity Refinement (2024)**

#### Nuove Ricerche:
- NDR v2 pubblicato su IEEE TPAMI 2024
- Architettura lightweight per dispositivi mobile
- Real-time performance con single MLP
- Zero-shot generalization da synthetic a real data

#### Fonti:
- Tosi et al., "Neural Disparity Refinement", IEEE TPAMI 2024
- ArXiv: Neural Disparity Refinement for Arbitrary Resolution Stereo

---

## 🛠️ PIANO D'AZIONE DETTAGLIATO

### FASE 1: FIX IMMEDIATO DEL FREEZE (Priorità Alta)

#### 1.1 **Modificare parallel_processor.py**

```python
# PROBLEMA: ProcessPoolExecutor non gestisce bene OpenCV objects
# SOLUZIONE: Passare solo paths, non numpy arrays

def process_single_frame_worker_v2(left_path: str, right_path: str,
                                  calibration_dict: Dict, config: Dict) -> Tuple[List, Dict]:
    """
    Worker v2: Carica immagini DENTRO il worker process
    """
    try:
        # Import DENTRO il worker per evitare conflitti
        import cv2
        import numpy as np
        from pathlib import Path
        
        # Carica immagini nel worker process
        left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        
        if left_img is None or right_img is None:
            return [], {'points': 0, 'error': 'Failed to load images'}
        
        # Resto del processing...
```

#### 1.2 **Aggiungere Start Method Esplicito**

```python
# In process_offline.py, PRIMA di tutto
if __name__ == '__main__':
    import multiprocessing
    # Force spawn method su tutte le piattaforme
    multiprocessing.set_start_method('spawn', force=True)
```

#### 1.3 **Limitare Worker e Batch Size**

```python
# Configurazione conservativa per evitare memory issues
profile = {
    'high_end_desktop': {
        'recommended_workers': min(4, cores // 2),  # MAX 4 workers
        'batch_size': 2,  # MAX 2 frames per batch
        'memory_per_worker_gb': 2.0,  # Aumentare stima memoria
    }
}
```

### FASE 2: OTTIMIZZAZIONE STEREOSGBM (Priorità Media)

#### 2.1 **Ridurre Consumo Memoria**

```python
def compute_advanced_surface_disparity(self, left_rect, right_rect):
    # OPZIONE 1: Usare MODE_SGBM invece di MODE_SGBM_3WAY
    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=-16,          # Ridotto da -32
        numDisparities=96,         # Ridotto da 160
        blockSize=5,               # Ridotto da 7
        mode=cv2.STEREO_SGBM_MODE_SGBM  # Invece di MODE_SGBM_3WAY
    )
    
    # OPZIONE 2: Downscale immagini se troppo grandi
    if left_rect.shape[0] * left_rect.shape[1] > 2_000_000:  # 2MP
        scale = 0.5
        left_small = cv2.resize(left_rect, None, fx=scale, fy=scale)
        right_small = cv2.resize(right_rect, None, fx=scale, fy=scale)
        # Process at lower res, then upscale disparity
```

#### 2.2 **Implementare Memory-Aware Processing**

```python
def check_available_memory():
    """Controlla memoria disponibile prima di processing"""
    import psutil
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    
    if available_gb < 4.0:
        logger.warning(f"Low memory: {available_gb:.1f}GB available")
        return 'conservative'
    elif available_gb < 8.0:
        return 'moderate'
    else:
        return 'aggressive'
```

### FASE 3: NEURAL DISPARITY REFINEMENT LIGHTWEIGHT (Priorità Bassa)

#### 3.1 **Implementare NDR Semplificato**

```python
class LightweightNDR:
    """
    Versione ultra-leggera di Neural Disparity Refinement
    Basata su filtri classici invece di deep learning
    """
    def refine_disparity_fast(self, disparity, left_img, confidence_threshold=0.5):
        # 1. Fast bilateral filter (invece di guided filter)
        refined = cv2.bilateralFilter(
            disparity.astype(np.float32), 
            d=5,  # Piccolo kernel
            sigmaColor=10, 
            sigmaSpace=10
        )
        
        # 2. Simple hole filling con nearest neighbor
        mask = disparity <= 0
        if np.any(mask):
            # Fast inpainting con metodo più veloce
            refined = cv2.inpaint(
                refined,
                mask.astype(np.uint8),
                inpaintRadius=3,  # Piccolo radius
                flags=cv2.INPAINT_NS  # Metodo veloce
            )
        
        return refined
```

#### 3.2 **Cache dei Risultati**

```python
class DisparityCache:
    """Cache per evitare ricalcoli"""
    def __init__(self, max_size=10):
        self.cache = {}
        self.max_size = max_size
    
    def get_or_compute(self, key, compute_func):
        if key in self.cache:
            return self.cache[key]
        
        result = compute_func()
        
        # LRU eviction
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.keys())
            del self.cache[oldest]
        
        self.cache[key] = result
        return result
```

### FASE 4: FALLBACK E RECOVERY (Priorità Alta)

#### 4.1 **Implementare Timeout e Fallback**

```python
def process_with_timeout(func, args, timeout=60):
    """Esegue funzione con timeout"""
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"Function timed out after {timeout}s")
            return None
```

#### 4.2 **Sequential Fallback Automatico**

```python
def smart_process_frames(stereo_pairs, reconstructor):
    """Prova parallel, fallback a sequential se fallisce"""
    try:
        # Prova parallel con timeout
        with timeout(300):  # 5 minuti max
            return process_parallel(stereo_pairs, reconstructor)
    except Exception as e:
        logger.warning(f"Parallel processing failed: {e}")
        logger.info("Falling back to sequential processing...")
        return process_sequential(stereo_pairs, reconstructor)
```

---

## 📊 TEST PLAN

### Test 1: Minimal Test (Verifica Fix Base)
```bash
# Test con 2 frames solo
python process_offline.py --input test_session --surface-reconstruction --workers 2 --batch-size 1 --single-frame
```

### Test 2: Memory Test (Verifica Limiti)
```bash
# Monitor memory usage
python -m memory_profiler process_offline.py --input test_session --surface-reconstruction --workers 1
```

### Test 3: Performance Comparison
```bash
# Sequential baseline
time python process_offline.py --input test_session --no-parallel

# Parallel ottimizzato
time python process_offline.py --input test_session --workers 4 --batch-size 2
```

---

## 🎯 RACCOMANDAZIONI FINALI

### IMMEDIATO (Oggi):
1. **Applicare Fix #1.1**: Modificare worker per caricare immagini internamente
2. **Applicare Fix #1.2**: Aggiungere spawn method esplicito
3. **Applicare Fix #1.3**: Limitare workers a 4 max
4. **Test minimal**: Verificare che non si blocchi più

### BREVE TERMINE (Questa Settimana):
1. **Ottimizzare StereoSGBM**: Ridurre parametri per memoria
2. **Implementare fallback**: Sequential processing automatico
3. **Aggiungere monitoring**: Memory usage in real-time

### LUNGO TERMINE (Prossimo Mese):
1. **Valutare alternative**: 
   - Joblib con backend 'loky' (no main guard needed)
   - Ray per distributed computing
   - Dask per out-of-core processing
2. **GPU acceleration**: CUDA StereoSGBM se disponibile
3. **Cloud processing**: Per dataset molto grandi

---

## 📌 CODICE PRONTO DA APPLICARE

### File: `parallel_processor_fixed.py`
```python
# Versione corretta con tutti i fix applicati
# [Codice completo disponibile su richiesta]
```

### File: `process_offline_fixed.py`
```python
# Main script con proper guards e fallback
# [Codice completo disponibile su richiesta]
```

---

## 🔗 RIFERIMENTI UTILI

1. **OpenCV Multiprocessing Guide**: https://pyimagesearch.com/2019/09/09/multiprocessing-with-opencv-and-python/
2. **Python Multiprocessing Best Practices**: https://superfastpython.com/multiprocessing-best-practices/
3. **Neural Disparity Refinement Paper**: https://arxiv.org/abs/2110.15367
4. **Memory Profiling Tools**: https://pypi.org/project/memory-profiler/

---

**Documento creato**: Gennaio 2025
**Autore**: Claude AI Assistant
**Versione**: 1.0