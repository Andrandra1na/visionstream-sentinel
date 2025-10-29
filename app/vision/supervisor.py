import cv2
from threading import Thread
from ultralytics import YOLO

from app.core.config import settings

class VideoStream:
    """Gère la capture vidéo dans un thread dédié pour une latence minimale."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise IOError(f"Impossible d'ouvrir le flux vidéo à l'adresse : {src}")
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

class Supervisor:
    """
    Classe principale qui orchestre la capture, l'analyse et l'affichage.
    """
    def __init__(self):
        print("Chargement du modèle YOLOv8...")
        self.model = YOLO(settings.MODEL_PATH)
        print("Modèle chargé.")
        
        print("Démarrage du flux vidéo...")
        self.video_stream = VideoStream(settings.VIDEO_STREAM_URL).start()
        print("Flux vidéo démarré.")

    def run(self):
        """Lance la boucle principale de supervision."""
        print("La fenêtre de supervision va s'ouvrir. Appuyez sur 'q' pour quitter.")
        
        while True:
            frame = self.video_stream.read()
            if frame is None:
                continue

            # --- Analyse de l'image ---
            annotated_frame = self._analyze_frame(frame)

            # --- Affichage ---
            cv2.imshow("VisionStream Sentinel - Console de Supervision", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self._cleanup()

    def _analyze_frame(self, frame):
        """Prend une image, effectue la détection et retourne l'image annotée."""
        h, w, _ = frame.shape
        ratio = settings.FRAME_WIDTH_FOR_PROCESSING / w
        new_height = int(h * ratio)
        resized_frame = cv2.resize(frame, (settings.FRAME_WIDTH_FOR_PROCESSING, new_height))

        results = self.model(resized_frame, verbose=False)
        
        intrusion_detected = False
        
        for result in results:
            for box in result.boxes:
                x1_r, y1_r, x2_r, y2_r = [int(i) for i in box.xyxy[0]]
                
                is_person = int(box.cls[0]) == 0
                is_intrusion = False

                if is_person:
                    person_center_x = (x1_r + x2_r) / 2
                    fz = settings.FORBIDDEN_ZONE
                    if fz[0] <= person_center_x <= fz[2]:
                        is_intrusion = True
                        intrusion_detected = True
                
                x1, y1, x2, y2 = int(x1_r / ratio), int(y1_r / ratio), int(x2_r / ratio), int(y2_r / ratio)
                
                conf = float(box.conf[0])
                label = f"{self.model.names[int(box.cls[0])]} {conf:.2f}"
                color = (0, 0, 255) if is_intrusion else (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if intrusion_detected:
            cv2.putText(frame, "ALERTE INTRUSION !", (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2)

        return frame

    def _cleanup(self):
        """Nettoie les ressources à la fin."""
        self.video_stream.stop()
        cv2.destroyAllWindows()
        print("Supervision terminée.")