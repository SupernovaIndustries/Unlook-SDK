# Prompt per Implementazione Completa Calibrazione Proiettore-Camera con Gray Code

## 📚 Documenti di Riferimento da Leggere
- **OBBLIGATORIO**: Leggi attentamente `@PHASE_SHIFT_STRUCTURED_LIGHT_RESEARCH.md`
- **OBBLIGATORIO**: Leggi attentamente `@PROJECTOR_CAMERA_HARDWARE_SETUP_GUIDE.md`

## 📋 Stato Attuale del Sistema

### ✅ Cosa è Stato Fatto Finora:

1. **Calibrazione Camera Singola Completata**
   - File: `calibration_2k_left_only/camera_calibration.json`
   - Risoluzione: 1456x1088
   - Focale: fx=1700.93, fy=1704.14 pixel (corretto per obiettivo 6mm)
   - 20 immagini di calibrazione con RMS error: 1.056 pixel
   - Distorsione calibrata con 5 coefficienti

2. **Calibrazione Proiettore Iniziata (MA INCOMPLETA)**
   - File: `projector_camera_calibration.json`
   - PROBLEMA: Solo calibrazione camera, NON calibrazione proiettore
   - Nota nel file: "projector calibration requires full pattern decoding implementation"
   - Script modificato: `calibrate_projector_camera.py` ora supporta pattern switching (p key)
   - 5 pattern strutturali implementati ma manca decodifica Gray Code

3. **Configurazione Hardware**
   - Camera sinistra con obiettivo 6mm
   - Proiettore DLP342x (1280x720 nativo)
   - Distanza camera-proiettore: ~8cm
   - Setup: [Camera] ---- 8cm ---- [Projector]

## 🚨 PROBLEMA CRITICO DA RISOLVERE

**Il proiettore NON è ancora calibrato come "camera inversa"!**

Secondo `PHASE_SHIFT_STRUCTURED_LIGHT_RESEARCH.md` (riga 169-174), serve:
```python
# Il proiettore deve essere calibrato come una "camera inversa"
projector_intrinsics = calibrate_projector(checkerboard_images)
projector_camera_extrinsics = calibrate_projector_camera_pair()
```

Secondo `PROJECTOR_CAMERA_HARDWARE_SETUP_GUIDE.md` (riga 57-65):
> "Il proiettore deve proiettare Gray code patterns SULLA scacchiera:
> 1. Camera rileva angoli scacchiera
> 2. Proiettore proietta Gray code sulla stessa scacchiera
> 3. Camera cattura scacchiera + Gray code insieme
> 4. Algoritmo calcola quale pixel proiettore illumina ogni angolo"

## 🎯 COSA DEVE ESSERE IMPLEMENTATO

### 1. **Implementazione Gray Code Pattern Decoding**
Creare o aggiornare il modulo per:
- Generare sequenze Gray Code (7-10 bit per 1280x720)
- Proiettare pattern Gray Code durante calibrazione
- Decodificare quale pixel del proiettore corrisponde a ogni angolo della scacchiera
- Calcolare corrispondenze camera↔proiettore

### 2. **Aggiornare `calibrate_projector_camera.py`**
Deve implementare il workflow completo:
```python
def calibrate_projector_as_inverse_camera():
    # Step 1: Rileva angoli scacchiera
    corners = detect_checkerboard_corners(camera_image)
    
    # Step 2: Proietta Gray Code patterns
    for bit in range(num_bits):
        # Proietta pattern verticale bit N
        project_gray_code_vertical(bit)
        capture_with_pattern()
        
        # Proietta pattern orizzontale bit N  
        project_gray_code_horizontal(bit)
        capture_with_pattern()
    
    # Step 3: Decodifica posizione proiettore per ogni angolo
    projector_coords = decode_gray_code_at_corners(corners, captured_patterns)
    
    # Step 4: Calibra proiettore come camera
    projector_intrinsics, projector_distortion = calibrate_camera(
        projector_coords, 
        world_points, 
        image_size=(1280, 720)
    )
    
    # Step 5: Calcola trasformazione camera↔proiettore
    R, T = stereo_calibrate(camera_points, projector_points)
```

### 3. **Pattern Gray Code Specifici**
Implementare in `unlook/client/scanning/calibration/gray_code_decoder.py`:
- Generazione pattern Gray Code binari
- Decodifica robusta con soglia adattiva
- Gestione riflessioni e rumore
- Validazione corrispondenze

### 4. **Struttura Dati Calibrazione Completa**
Il file di calibrazione finale deve contenere:
```json
{
  "camera_intrinsics": {...},      // ✅ Già fatto
  "camera_distortion": {...},       // ✅ Già fatto
  "projector_intrinsics": {...},    // ❌ DA FARE
  "projector_distortion": {...},    // ❌ DA FARE
  "rotation_matrix": {...},         // ❌ DA FARE
  "translation_vector": {...},      // ❌ DA FARE
  "essential_matrix": {...},        // ❌ DA FARE
  "fundamental_matrix": {...},      // ❌ DA FARE
  "rectification_transforms": {...} // ❌ DA FARE
}
```

## 🔧 IMPLEMENTAZIONE STEP-BY-STEP

### Step 1: Gray Code Pattern Generator
```python
# In pattern_manager.py o nuovo file
def generate_gray_code_patterns(width=1280, height=720, num_bits=10):
    patterns = []
    
    # Vertical stripes (X coordinate encoding)
    for bit in range(num_bits):
        pattern = create_gray_code_pattern(width, height, bit, 'vertical')
        patterns.append({
            'name': f'gray_v_bit_{bit}',
            'type': 'gray_code',
            'orientation': 'vertical',
            'bit': bit,
            'image': pattern
        })
    
    # Horizontal stripes (Y coordinate encoding)
    for bit in range(num_bits):
        pattern = create_gray_code_pattern(width, height, bit, 'horizontal')
        patterns.append({
            'name': f'gray_h_bit_{bit}',
            'type': 'gray_code', 
            'orientation': 'horizontal',
            'bit': bit,
            'image': pattern
        })
    
    return patterns
```

### Step 2: Gray Code Decoder
```python
# Decodifica coordinate proiettore da sequenza Gray Code
def decode_projector_coordinates(gray_images, corners):
    projector_points = []
    
    for corner in corners:
        x, y = corner
        
        # Decodifica coordinata X da pattern verticali
        x_code = extract_gray_code(gray_images['vertical'], x, y)
        projector_x = gray_to_binary(x_code) * (1280 / (2**num_bits))
        
        # Decodifica coordinata Y da pattern orizzontali
        y_code = extract_gray_code(gray_images['horizontal'], x, y)
        projector_y = gray_to_binary(y_code) * (720 / (2**num_bits))
        
        projector_points.append([projector_x, projector_y])
    
    return np.array(projector_points)
```

### Step 3: Calibrazione Proiettore Completa
```python
# Calcola parametri intrinseci proiettore
def calibrate_projector(world_points, projector_points):
    # Usa cv2.calibrateCamera con punti proiettore
    ret, proj_matrix, proj_dist, rvecs, tvecs = cv2.calibrateCamera(
        [world_points], 
        [projector_points], 
        (1280, 720),
        None, 
        None
    )
    return proj_matrix, proj_dist
```

## 📐 VERIFICA E VALIDAZIONE

### Test di Verifica Post-Calibrazione:
1. **Reprojection Error**: Deve essere < 1 pixel per proiettore
2. **Epipolar Geometry**: Linee epipolari devono essere corrette
3. **3D Reconstruction Test**: Proietta pattern noto e verifica accuratezza 3D
4. **Pattern Alignment**: I pattern proiettati devono allinearsi perfettamente

### Metriche di Successo:
- ✅ Calibrazione proiettore con RMS < 1.0 pixel
- ✅ Triangolazione produce 50,000+ punti 3D
- ✅ Accuratezza 3D: ±0.1mm a 50cm distanza
- ✅ Phase unwrapping senza artefatti

## 🚀 RISULTATO FINALE ATTESO

Dopo implementazione completa:
1. **File calibrazione completo** con tutti i parametri camera+proiettore
2. **Triangolazione projector-camera** funzionante
3. **50,000-200,000 punti 3D** per scansione (vs 400-500 attuali)
4. **Accuratezza professionale** per scanning 3D

## 💡 NOTE IMPORTANTI

- La calibrazione del proiettore è ESSENZIALE per phase shift 3D
- Senza calibrazione proiettore, impossibile fare triangolazione corretta
- Gray Code è il metodo standard per calibrare proiettori
- Serve pazienza: calibrazione proiettore più complessa di calibrazione camera

## 📝 PRIORITÀ IMPLEMENTAZIONE

1. **URGENTE**: Implementa Gray Code pattern generation e decoding
2. **CRITICO**: Aggiorna calibrate_projector_camera.py per calibrazione completa
3. **IMPORTANTE**: Testa con phase shift patterns dopo calibrazione
4. **VALIDAZIONE**: Verifica 50,000+ punti 3D generati

Questo completerà la transizione da stereo vision a structured light professionale!