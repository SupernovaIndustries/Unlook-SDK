# UnLook SDK

SDK universale per scanner 3D a luce strutturata UnLook. Questa libreria fornisce un'interfaccia completa per la comunicazione, controllo e acquisizione di dati dagli scanner UnLook, con funzionalità di elaborazione 3D integrate.

## Caratteristiche principali

- **Discovery automatica**: trova automaticamente gli scanner disponibili nella rete locale
- **Controllo del proiettore**: genera e proietta pattern di luce strutturata
- **Gestione multicamera**: supporto nativo per sistemi stereo con 2 o più telecamere
- **Elaborazione stereo**: ricostruzione 3D tramite stereovisione passiva e attiva
- **Scansione 3D**: acquisizione ed elaborazione di immagini con pattern di luce strutturata
- **Ricostruzione 3D**: algoritmi per la triangolazione e ricostruzione di nuvole di punti 3D
- **Esportazione modelli**: supporto per formati standard (PLY, OBJ, XYZ)
- **Calibrazione**: strumenti per la calibrazione della telecamera, del sistema e stereo

## Architettura

L'SDK è strutturato in due parti principali:

1. **Client**: si connette allo scanner, invia comandi, riceve dati e si occupa dell'elaborazione
2. **Server**: gira su Raspberry Pi, controlla il proiettore e le telecamere, invia i dati al client

L'elaborazione avviene principalmente lato client per sfruttare la maggiore potenza di calcolo disponibile.

## Requisiti

### Client
- Python 3.8 o superiore
- OpenCV 4.5+
- NumPy
- PyZMQ
- Zeroconf

### Server (Raspberry Pi)
- Python 3.8 o superiore
- PiCamera2
- SMBus2
- PyZMQ
- Zeroconf

## Guida rapida

### Trovare e connettersi a uno scanner

```python
from unlook.client import UnlookClient

# Crea il client
client = UnlookClient()

# Avvia la discovery degli scanner
client.start_discovery()

# Attendi un po' per la discovery
import time
time.sleep(3)

# Ottieni la lista degli scanner trovati
scanners = client.get_discovered_scanners()
if scanners:
    # Connetti al primo scanner
    client.connect(scanners[0])
    
    # Usa lo scanner...
    
    # Disconnetti quando hai finito
    client.disconnect()

# Ferma la discovery
client.stop_discovery()
```

### Controllare il proiettore

```python
# Imposta la modalità test pattern
client.projector.set_test_pattern_mode()

# Proietta un pattern
client.projector.show_grid()

# Al termine, metti in standby
client.projector.set_standby()
```

### Catturare immagini da una coppia stereo

```python
# Ottieni gli ID della coppia stereo
left_camera_id, right_camera_id = client.camera.get_stereo_pair()

# Configura le telecamere
client.camera.configure(left_camera_id, {
    "resolution": [1920, 1080],
    "fps": 30
})
client.camera.configure(right_camera_id, {
    "resolution": [1920, 1080],
    "fps": 30
})

# Cattura immagini sincronizzate
stereo_images = client.camera.capture_multi([left_camera_id, right_camera_id])
left_image = stereo_images[left_camera_id]
right_image = stereo_images[right_camera_id]

# Oppure usa il metodo specifico per stereo
left_image, right_image = client.camera.capture_stereo_pair()
```

### Eseguire una scansione 3D con stereovisione

```python
from unlook.client import StereoProcessor, StereoCalibrationData

# Carica la calibrazione stereo
stereo_calib = StereoCalibrationData.load("stereo_calibration.json")

# Crea un processore stereo
stereo_processor = StereoProcessor(stereo_calib)

# Cattura una coppia di immagini stereo
left_image, right_image = client.camera.capture_stereo_pair()

# Esegui la ricostruzione 3D
points, colors, disparity_map = stereo_processor.compute_stereo_scan(left_image, right_image)

# Usa i risultati...
# points: nuvola di punti 3D
# colors: colori dei punti
# disparity_map: mappa di disparità
```

### Eseguire una scansione 3D con luce strutturata

```python
from unlook.client import ScanProcessor, PatternDirection, ModelExporter

# Crea un processore di scansione
scan_processor = ScanProcessor(client)

# Carica dati di calibrazione (opzionale ma consigliato)
from unlook.client import CalibrationData
calibration = CalibrationData.load("calibration.json")
scan_processor.calibration = calibration

# Esegui la scansione
camera_id = left_camera_id  # Usa la telecamera sinistra
success, result = scan_processor.capture_gray_code_scan(
    camera_id=camera_id,
    pattern_width=1280,
    pattern_height=800,
    direction=PatternDirection.BOTH,
    capture_texture=True,
    show_preview=True
)

if success and result.has_point_cloud():
    # Esporta il risultato
    exporter = ModelExporter()
    exporter.export_ply(result, "scan_result.ply")
```

## Avvio del server

Sul Raspberry Pi, esegui:

```python
from unlook.server import UnlookServer

# Crea e avvia il server
server = UnlookServer(
    name="MyUnLookScanner",
    control_port=5555,
    stream_port=5556
)

# Il server è già in ascolto grazie al parametro auto_start=True
# Il proiettore è automaticamente configurato per utilizzare il bus I2C 3 e l'indirizzo 0x1B

# Per arrestare manualmente
# server.stop()
```

## Esempi

La directory `examples/` contiene diversi esempi di utilizzo dell'SDK:

- `client_example.py`: esempio base di connessione e utilizzo
- `scanner_3d_example.py`: scansione 3D interattiva
- `stereo_calibration.py`: calibrazione di un sistema stereo
- `stereo_scan.py`: scansione 3D usando stereovisione
- `programmatic_sdk_usage.py`: esempio di utilizzo programmatico
- `minimal_scanner.py`: esempio minimo di controllo dello scanner
- `run_server.py`: esempio di avvio del server

## Integrazione con altre librerie

L'SDK è progettato per essere facilmente integrabile con librerie di elaborazione 3D come:

- **Open3D**: per la visualizzazione e l'elaborazione di nuvole di punti
- **Mesh processing**: per convertire nuvole di punti in mesh 3D

## Sviluppi futuri

1. **Raffinamento dei modelli 3D**: Aggiungere algoritmi di post-processing come mesh generation
2. **Interfaccia web**: Sviluppare una GUI web-based
3. **Supporto multi-camera avanzato**: Espandere per utilizzare più di 2 telecamere contemporaneamente
4. **Calibrazione avanzata**: Migliorare la precisione della calibrazione
5. **Ottimizzazione delle performance**: Accelerazione GPU dove possibile

## Licenza

MIT License

## Autori

UnLook Team