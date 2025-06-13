# Posizioni Standard per Calibrazione Camera-Proiettore

## Panoramica
Questo documento descrive le 8 posizioni standard per la calibrazione camera-proiettore nel sistema UnLook SDK. Una calibrazione accurata è fondamentale per ottenere ricostruzioni 3D di qualità millimetrica.

## Specifiche della Scacchiera
- **Dimensioni**: 9x6 angoli interni (pattern asimmetrico per orientamento non ambiguo)
- **Dimensione quadrato**: 23.13 mm
- **Bordo bianco**: Almeno 1-2 quadrati di spazio bianco attorno al pattern per migliorare la detection
- **Materiale**: Stampato su superficie rigida e piatta (cartone spesso, pannello rigido)
- **Qualità**: Pattern ad alto contrasto, evitare superfici riflettenti

## Configurazione Hardware
- **Baseline**: 80mm tra camera e proiettore (come due camere stereo)
- **Altezza**: Camera e proiettore alla stessa altezza
- **Angolo**: Angolazione convergente verso l'area di lavoro
- **Illuminazione**: Luce ambientale uniforme, evitare ombre sulla scacchiera
- **Stabilità**: Montaggio fisso su treppiede (NON muovere dopo calibrazione)

## 8 Posizioni Standard per Calibrazione

### Posizione 1: Centro Frontale
- **Distanza**: 400-500mm dalla camera
- **Orientamento**: Parallelo al piano camera-proiettore
- **Rotazione**: 0° (frontale)
- **Copertura**: Centro del campo visivo
- **Note**: Posizione di riferimento base

### Posizione 2: Centro con Rotazione X
- **Distanza**: 400-500mm dalla camera
- **Orientamento**: Ruotato di +20° attorno all'asse X (inclinato verso l'alto)
- **Rotazione**: 0° attorno Z
- **Copertura**: Centro del campo visivo
- **Note**: Testa il modello di distorsione in direzione verticale

### Posizione 3: Centro con Rotazione Y
- **Distanza**: 400-500mm dalla camera
- **Orientamento**: Ruotato di +25° attorno all'asse Y (inclinato verso destra)
- **Rotazione**: 0° attorno Z
- **Copertura**: Centro del campo visivo
- **Note**: Testa il modello di distorsione in direzione orizzontale

### Posizione 4: Angolo Superiore Sinistro
- **Distanza**: 350-400mm dalla camera
- **Orientamento**: Ruotato di -15° attorno X, -20° attorno Y
- **Posizione**: Spostato verso l'angolo superiore sinistro del campo visivo
- **Copertura**: 20-30% del bordo superiore sinistro
- **Note**: Copre l'angolo critico per la calibrazione

### Posizione 5: Angolo Superiore Destro
- **Distanza**: 350-400mm dalla camera
- **Orientamento**: Ruotato di -15° attorno X, +20° attorno Y
- **Posizione**: Spostato verso l'angolo superiore destro del campo visivo
- **Copertura**: 20-30% del bordo superiore destro
- **Note**: Copre l'angolo opposto per simmetria

### Posizione 6: Angolo Inferiore Sinistro
- **Distanza**: 450-500mm dalla camera
- **Orientamento**: Ruotato di +15° attorno X, -20° attorno Y
- **Posizione**: Spostato verso l'angolo inferiore sinistro del campo visivo
- **Copertura**: 20-30% del bordo inferiore sinistro
- **Note**: Completa la copertura degli angoli

### Posizione 7: Angolo Inferiore Destro
- **Distanza**: 450-500mm dalla camera
- **Orientamento**: Ruotato di +15° attorno X, +20° attorno Y
- **Posizione**: Spostato verso l'angolo inferiore destro del campo visivo
- **Copertura**: 20-30% del bordo inferiore destro
- **Note**: Completa la copertura degli angoli

### Posizione 8: Distanza Ravvicinata
- **Distanza**: 200-250mm dalla camera
- **Orientamento**: Ruotato di -10° attorno X, +30° attorno Y
- **Posizione**: Riempie completamente il campo visivo
- **Copertura**: 80-90% dell'inquadratura
- **Note**: Testa la calibrazione a distanza di lavoro ravvicinata

## Schema di Copertura del Campo Visivo

```
+---+---+---+
| 4 | 2 | 5 |  Posizioni 4,5: Angoli superiori
+---+---+---+  Posizione 2: Centro superiore
| 3 | 1 | 3 |  Posizione 1: Centro
+---+---+---+  Posizione 3: Lati (rotazione Y)
| 6 | 2 | 7 |  Posizioni 6,7: Angoli inferiori
+---+---+---+  Posizione 8: Riempimento completo
```

## Linee Guida per l'Acquisizione

### Prima dell'Acquisizione
1. **Calibrazione LED**: Impostare intensità LED a 0mA (OFF) per evitare interferenze
2. **Focus**: Verificare che la scacchiera sia a fuoco sia per camera che proiettore
3. **Illuminazione**: Luce ambientale uniforme, evitare riflessi e ombre
4. **Stabilità**: Assicurarsi che hardware e scacchiera siano stabili

### Durante l'Acquisizione
1. **Sequenza Gray Code**: Automatica (14 pattern: 2 riferimenti + 12 Gray Code)
2. **Tempo di esposizione**: Mantenere fisso (disabilitare auto-esposizione)
3. **Verifica**: Controllare che tutti gli angoli siano visibili e rilevati
4. **Pattern proiettati**: Strisce enormi (400-900px) per visibilità ottimale

### Criteri di Qualità
- **Detection rate**: >95% degli angoli rilevati correttamente
- **Copertura**: Ogni posizione deve coprire una zona diversa del campo visivo
- **Contrasto**: Pattern Gray Code chiaramente visibili sul proiettore
- **Stabilità**: Nessun movimento durante la sequenza di acquisizione (3-5 secondi)

## Parametri di Calibrazione Raccomandati

### Camera
- **Risoluzione**: 1456x1088 (nativa)
- **Modello distorsione**: Fino al 4° ordine (radiale + tangenziale)
- **Flags**: `CALIB_USE_INTRINSIC_GUESS` se disponibile calibrazione precedente

### Proiettore
- **Risoluzione**: 1280x720
- **Modello**: Trattato come "camera inversa"
- **Pattern**: Gray Code 6-bit (minimo per linee visibili)
- **Calibrazione**: Simultanea con camera per eliminare errori di propagazione

### Stereo
- **Flags**: `CALIB_FIX_INTRINSIC` se camera già calibrata
- **Criteri**: `TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 100, 1e-5`
- **Alpha**: 0 per rettificazione senza perdita di dati

## Risultati Attesi

### Errori RMS Tipici
- **Camera**: <0.5 pixel
- **Proiettore**: <1.0 pixel
- **Stereo**: <0.8 pixel

### Baseline
- **Target**: 80.0 ± 2.0 mm
- **Precisione**: ±1% per applicazioni millimetriche

### Copertura
- **Campo visivo**: >80% coperto dalle 8 posizioni
- **Angoli**: Tutti e 4 gli angoli rappresentati
- **Distanze**: Range 200-500mm testato

## Risoluzione Problemi

### Bassa Detection Rate
- Verificare illuminazione uniforme
- Controllare focus di camera e proiettore
- Aumentare contrasto del pattern stampato
- Verificare che la scacchiera sia perfettamente piatta

### Errori RMS Elevati
- Aumentare numero di posizioni (fino a 12-15)
- Migliorare diversità angolare
- Verificare calibrazione intrinseca camera
- Controllare stabilità meccanica del setup

### Pattern Gray Code Non Visibili
- Aumentare intensità LED (ma mantenere 0mA durante detection scacchiera)
- Verificare focus e allineamento proiettore
- Controllare che i pattern abbiano strisce sufficientemente larghe (>400px)
- Verificare connessione e controllo proiettore

## Comandi di Calibrazione

### Modalità Interattiva con Live Preview
```bash
python calibrate_projector_camera.py --interactive --live-preview --led-intensity 0 --save-images
```

### Processamento Offline
```bash
python calibrate_projector_camera.py --process-captured calibration_capture --output projector_camera_calibration.json
```

### Con Calibrazione Camera Esistente
```bash
python calibrate_projector_camera.py --interactive --camera-calibration camera_calibration.json --output calibration_complete.json
```

## Note Finali

1. **Tempo richiesto**: 15-20 minuti per acquisizione completa
2. **Numero minimo**: 3 posizioni, raccomandato 8
3. **Hardware fisso**: NON muovere camera/proiettore dopo calibrazione
4. **Validazione**: Testare con scansione di oggetto noto prima dell'uso
5. **Backup**: Salvare sempre calibrazione e immagini di riferimento

Seguendo queste linee guida, si otterrà una calibrazione camera-proiettore di qualità professionale per il sistema UnLook SDK.