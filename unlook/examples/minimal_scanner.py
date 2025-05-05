"""
Esempio minimo di utilizzo dell'SDK UnLook.
Questo script mostra come utilizzare l'SDK in modo minimale per controllare lo scanner.
"""

import time
import cv2
import numpy as np
from unlook.client import UnlookClient, UnlookClientEvent

# Crea il client
client = UnlookClient(client_name="UnlookMinimal")

# Avvia la discovery degli scanner
print("Ricerca scanner in corso...")
client.start_discovery()

# Attendi 3 secondi per trovare gli scanner
time.sleep(3)

# Ottieni la lista degli scanner trovati
scanners = client.get_discovered_scanners()
if not scanners:
    print("Nessuno scanner trovato. Uscita.")
    client.stop_discovery()
    exit(1)

# Seleziona il primo scanner
scanner = scanners[0]
print(f"Connessione a: {scanner.name} ({scanner.uuid})...")

# Connetti allo scanner
if not client.connect(scanner):
    print("Impossibile connettersi allo scanner. Uscita.")
    client.stop_discovery()
    exit(1)

# Ottieni la lista delle telecamere
cameras = client.camera.get_cameras()
if not cameras:
    print("Nessuna telecamera disponibile. Uscita.")
    client.disconnect()
    client.stop_discovery()
    exit(1)

# Seleziona la prima telecamera
camera_id = cameras[0]["id"]
print(f"Telecamera selezionata: {cameras[0]['name']} ({camera_id})")

# Imposta il proiettore in modalità test pattern
print("Impostazione proiettore in modalità test pattern...")
client.projector.set_test_pattern_mode()

# Proietta diversi pattern
patterns = [
    ("Bianco", lambda: client.projector.show_solid_field("White")),
    ("Nero", lambda: client.projector.show_solid_field("Black")),
    ("Rosso", lambda: client.projector.show_solid_field("Red")),
    ("Verde", lambda: client.projector.show_solid_field("Green")),
    ("Blu", lambda: client.projector.show_solid_field("Blue")),
    ("Griglia", lambda: client.projector.show_grid()),
    ("Scacchiera", lambda: client.projector.show_checkerboard()),
    ("Barre orizzontali", lambda: client.projector.show_horizontal_lines()),
    ("Barre verticali", lambda: client.projector.show_vertical_lines()),
    ("Barre colore", lambda: client.projector.show_colorbars()),
]

# Crea una finestra per visualizzare le immagini
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

# Proietta ogni pattern e cattura un'immagine
for name, pattern_func in patterns:
    print(f"Proiezione pattern: {name}")

    # Proietta il pattern
    pattern_func()

    # Attendi che il pattern sia proiettato
    time.sleep(0.5)

    # Cattura un'immagine
    image = client.camera.capture(camera_id)

    if image is not None:
        # Aggiungi il nome del pattern all'immagine
        cv2.putText(
            image,
            name,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Visualizza l'immagine
        cv2.imshow("Camera", image)

        # Salva l'immagine
        filename = f"pattern_{name.replace(' ', '_').lower()}.jpg"
        cv2.imwrite(filename, image)
        print(f"Immagine salvata come: {filename}")

        # Attendi 1 secondo o la pressione di un tasto
        key = cv2.waitKey(1000)
        if key == 27:  # ESC
            break
    else:
        print(f"Errore durante la cattura dell'immagine per il pattern {name}")

# Chiudi la finestra OpenCV
cv2.destroyAllWindows()

# Metti il proiettore in standby
print("Proiettore in standby...")
client.projector.set_standby()

# Disconnetti e pulisci
client.disconnect()
client.stop_discovery()

print("Script terminato.")