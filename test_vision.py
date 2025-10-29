# test_vision.py (version multi-threadée et optimisée)
import cv2
import os
import time
from dotenv import load_dotenv
from ultralytics import YOLO
from threading import Thread
from queue import Queue

# --- Paramètres d'optimisation ---
FRAME_WIDTH_FOR_PROCESSING = 320

class VideoStream:
    """
    Classe pour gérer la capture vidéo dans un thread dédié afin d'éviter le décalage du tampon OpenCV.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print(f"Erreur : Impossible de se connecter au flux vidéo à l'adresse : {src}")
            raise IOError("Impossible d'ouvrir le flux vidéo")

        # Lire la première image pour initialiser
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Démarrer le thread pour lire les images du flux
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Boucle infinie pour lire les images
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Renvoyer la dernière image lue
        return self.frame

    def stop(self):
        # Indiquer que le thread doit s'arrêter
        self.stopped = True

def main():
    """
    Fonction principale avec lecture et traitement vidéo séparés.
    """
    # 1. Charger les variables d'environnement
    load_dotenv()
    video_stream_url = os.getenv("VIDEO_STREAM_URL")
    if not video_stream_url:
        print("Erreur : VIDEO_STREAM_URL n'est pas définie dans le fichier .env.")
        return

    # 2. Charger le modèle YOLOv8
    model = YOLO('./models/yolov8n.pt')

    # 3. Démarrer le flux vidéo dans un thread séparé
    print("Démarrage du flux vidéo...")
    vs = VideoStream(video_stream_url).start()
    time.sleep(2.0)  # Laisser le temps au tampon de se remplir un peu

    print("Démarrage de la boucle de traitement. Appuyez sur 'q' pour quitter.")
    
    # 4. Boucle principale de traitement
    while True:
        # Lire la dernière image disponible depuis le thread de lecture
        frame = vs.read()
        if frame is None:
            continue

        # --- Optimisation : Réduction de la résolution ---
        h, w, _ = frame.shape
        ratio = FRAME_WIDTH_FOR_PROCESSING / w
        new_height = int(h * ratio)
        resized_frame = cv2.resize(frame, (FRAME_WIDTH_FOR_PROCESSING, new_height))

        # 5. Effectuer la détection d'objets
        results = model(resized_frame, verbose=False) # verbose=False pour des logs plus propres

        # 6. Dessiner les résultats sur l'image originale
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                orig_x1, orig_y1, orig_x2, orig_y2 = int(x1 / ratio), int(y1 / ratio), int(x2 / ratio), int(y2 / ratio)
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 7. Afficher l'image
        cv2.imshow("VisionStream Sentinel - Test de Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 8. Libérer les ressources
    vs.stop()
    cv2.destroyAllWindows()
    print("Flux vidéo et fenêtres fermés proprement.")

if __name__ == "__main__":
    main()