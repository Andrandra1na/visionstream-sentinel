import cv2
import time
import base64
from threading import Thread
from ultralytics import YOLO

from app.core.config import settings

class VideoProcessor:
    """
    Gère la capture, le traitement et la diffusion des images vidéo.
    """
    def __init__(self):
        self.model = YOLO(settings.MODEL_PATH)
        self.video_stream = self._start_video_stream()
        self.latest_frame_data = None
        self.processing_stopped = False

    def _start_video_stream(self):
        stream = cv2.VideoCapture(settings.VIDEO_STREAM_URL)
        if not stream.isOpened():
            raise IOError("Impossible d'ouvrir le flux vidéo")
        return stream

    def _processing_loop(self):
        """La boucle qui tourne en arrière-plan pour traiter le flux."""
        while not self.processing_stopped:
            ret, frame = self.video_stream.read()
            if not ret:
                print("Flux vidéo terminé ou erreur de lecture.")
                time.sleep(5) # Attendre avant de réessayer
                self.video_stream = self._start_video_stream()
                continue

            # Redimensionner pour le traitement
            h, w, _ = frame.shape
            ratio = settings.FRAME_WIDTH_FOR_PROCESSING / w
            new_height = int(h * ratio)
            resized_frame = cv2.resize(frame, (settings.FRAME_WIDTH_FOR_PROCESSING, new_height))

            # Détection
            results = self.model(resized_frame, verbose=False)

            detections_list = []
            intrusion_detected = False

            # Dessiner les résultats et vérifier les intrusions
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                    
                    # Vérifier l'intrusion
                    if box.cls[0] == 0: # 0 est généralement la classe 'person'
                        person_center_x = (x1 + x2) / 2
                        person_center_y = (y1 + y2) / 2
                        
                        fz = settings.FORBIDDEN_ZONE
                        if fz[0] <= person_center_x <= fz[2] and fz[1] <= person_center_y <= fz[3]:
                            intrusion_detected = True

                    # Remettre à l'échelle pour l'affichage
                    orig_x1, orig_y1, orig_x2, orig_y2 = int(x1 / ratio), int(y1 / ratio), int(x2 / ratio), int(y2 / ratio)
                    
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{self.model.names[cls]} {conf:.2f}"
                    
                    color = (0, 0, 255) if intrusion_detected and cls == 0 else (0, 255, 0)
                    cv2.rectangle(frame, (orig_x1, orig_y1), (orig_x2, orig_y2), color, 2)
                    cv2.putText(frame, label, (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    detections_list.append({
                        "x1": orig_x1, "y1": orig_y1, "x2": orig_x2, "y2": orig_y2,
                        "confidence": conf, "label": self.model.names[cls]
                    })
            
            # Encoder l'image en JPEG puis en Base64
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Mettre à jour le dernier frame disponible pour les clients WebSocket
            self.latest_frame_data = {
                "image_base64": image_base64,
                "detections": detections_list,
                "status": "INTRUSION" if intrusion_detected else "ok"
            }

    def start_processing(self):
        """Démarre la boucle de traitement dans un thread séparé."""
        self.processing_thread = Thread(target=self._processing_loop)
        self.processing_thread.start()

    def stop_processing(self):
        """Arrête la boucle de traitement."""
        self.processing_stopped = True
        if self.processing_thread:
            self.processing_thread.join()
        self.video_stream.release()

# Créer une instance unique du processeur pour toute l'application
video_processor = VideoProcessor()