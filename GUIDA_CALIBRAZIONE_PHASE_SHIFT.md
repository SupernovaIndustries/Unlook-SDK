# 📋 GUIDA RAPIDA CALIBRAZIONE E PHASE SHIFT SCANNING

## 1️⃣ CALIBRAZIONE CAMERA SINISTRA (già catturata)
```bash
# Processare le immagini già catturate in calibration_2k_left_only/left
python unlook/examples/calibration/process_calibration.py \
  --input calibration_2k_left_only \
  --output calibration_2k_left_only/camera_left_calibration.json \
  --checkerboard-size 9x6 \
  --square-size 23.13 \
  --single-camera \
  --camera left
```

## 2️⃣ CALIBRAZIONE PROIETTORE-CAMERA (Gray Code)
```bash
python unlook/examples/calibration/calibrate_projector_camera.py --interactive --live-preview --num-positions 8 --gray-bits 6 --projector-width 1280 --projector-height 720 --checkerboard-size 9x6 --square-size 23.13 --save-images --led-intensity 0 --camera-calibration calibration_2k_left_only/camera_left_calibration.json --output projector_camera_calibration.json
  ```

## 3️⃣ CATTURA PATTERN PHASE SHIFT
```bash
python unlook/examples/scanning/capture_patterns.py \
  --pattern phase_shift \
  --num-steps 4 \
  --frequencies 1,8,64 \
  --output captured_data/phase_shift_test \
  --led-intensity 0 \
  --save-debug
```

## 4️⃣ RICOSTRUZIONE 3D CON TRIANGOLAZIONE
```bash
python unlook/examples/scanning/process_phase_shift_offline.py \
  --input captured_data/phase_shift_test/[SESSION_DIR] \
  --calibration projector_camera_calibration.json \
  --output results/phase_shift_reconstruction \
  --save-ply \
  --save-debug
```

## 📝 NOTE IMPORTANTI:
- SESSION_DIR = directory con timestamp creata da capture_patterns.py
- Assicurarsi che il proiettore sia ben focalizzato
- Ambiente con poca luce ambientale per phase shift
- Il file di calibrazione projector_camera_calibration.json è CRITICO per la triangolazione