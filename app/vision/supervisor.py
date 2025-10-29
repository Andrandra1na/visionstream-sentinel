# app/vision/supervisor.py

import cv2
import numpy as np  # Nécessaire pour les opérations sur les polygones
from threading import Thread
from ultralytics import YOLO

from app.core.config import settings

# --- Variables globales pour la zone interactive ---
# Ces variables sont globales au module pour être accessibles par le gestionnaire d'événements de la souris.

# Liste pour stocker les points du polygone cliqués par l'utilisateur
points = [] 
# Le polygone final une fois que l'utilisateur a terminé (un tableau NumPy)
defined_zone = None 
# Nom de la fenêtre OpenCV pour pouvoir y attacher des événements
window_name = "VisionStream Sentinel - Console de Supervision"


def mouse_event_handler(event, x, y, flags, param):
    """
    Fonction appelée par OpenCV à chaque événement de la souris sur la fenêtre spécifiée.
    C'est un "callback" : une fonction que l'on donne à un autre système pour qu'il l'appelle.
    """
    global points, defined_zone

    # Si l'utilisateur fait un clic gauche
    if event == cv2.EVENT_LBUTTONDOWN:
        # On vérifie qu'on ne définit pas une nouvelle zone alors qu'une est déjà active
        if defined_zone is None:
            # Ajouter le point cliqué à notre liste
            points.append((x, y))
            print(f"Point ajouté : {(x, y)}. Total : {len(points)} points. (Clic droit pour finaliser)")

    # Si l'utilisateur fait un clic droit, on finalise la zone
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 2: # Il faut au moins 3 points pour un polygone (un triangle)
            defined_zone = np.array(points, np.int32)
            print(f"Zone d'intrusion définie avec {len(points)} points.")
            points = [] # Vider la liste des points temporaires
        else:
            print("Pas assez de points pour définir une zone. Clic gauche pour ajouter des points.")


class VideoStream:
    """
    Gère la capture vidéo dans un thread dédié pour une latence minimale.
    Cette classe est inchangée.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise IOError(f"Impossible d'ouvrir le flux vidéo à l'adresse : {src}")
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Démarrer le thread pour lire les images du flux en continu
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        # Boucle infinie qui lit constamment la dernière image
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Renvoyer la dernière image lue par le thread
        return self.frame

    def stop(self):
        # Indiquer que le thread doit s'arrêter
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
        """Lance la boucle principale de supervision et gère les interactions utilisateur."""
        print("\n--- INSTRUCTIONS ---")
        print("1. CLIC GAUCHE sur la vidéo pour ajouter des points et définir la zone d'intrusion.")
        print("2. CLIC DROIT pour finaliser la zone (minimum 3 points).")
        print("3. Appuyez sur 'c' pour effacer la zone et recommencer.")
        print("4. Appuyez sur 'q' pour quitter.")
        print("--------------------\n")

        # Créer la fenêtre et y associer notre gestionnaire d'événements de la souris
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_event_handler)
        
        while True:
            frame = self.video_stream.read()
            if frame is None:
                continue

            # Dessiner la zone interactive (points en cours, polygone finalisé) sur l'image
            self._draw_interactive_zone(frame)

            # Analyser l'image pour la détection d'objets et d'intrusions
            annotated_frame = self._analyze_frame(frame)

            # Afficher l'image finale
            cv2.imshow(window_name, annotated_frame)

            # Gérer les entrées clavier
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): # Si 'q' est pressé, on quitte
                break
            elif key == ord('c'): # Si 'c' est pressé, on efface la zone
                global points, defined_zone
                points = []
                defined_zone = None
                print("Zone effacée. Vous pouvez redéfinir une nouvelle zone.")
        
        self._cleanup()

    def _draw_interactive_zone(self, frame):
        """Dessine la zone interactive (points et polygones) sur l'image."""
        # Dessiner les lignes entre les points en cours de sélection pour guider l'utilisateur
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i+1], (0, 255, 255), 2) # Lignes jaunes
        
        # Dessiner les points cliqués
        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1) # Cercles rouges pleins

        # Dessiner le polygone finalisé avec une transparence pour la visibilité
        if defined_zone is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [defined_zone], (0, 0, 255)) # Remplissage rouge
            
            # Appliquer la transparence en mélangeant l'overlay et l'image originale
            alpha = 0.3 # 30% d'opacité
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _analyze_frame(self, frame):
        """Prend une image, effectue la détection et retourne l'image avec les annotations."""
        # Redimensionner l'image pour une analyse plus rapide
        h, w, _ = frame.shape
        ratio = settings.FRAME_WIDTH_FOR_PROCESSING / w
        new_height = int(h * ratio)
        resized_frame = cv2.resize(frame, (settings.FRAME_WIDTH_FOR_PROCESSING, new_height))

        # Exécuter le modèle IA
        results = self.model(resized_frame, verbose=False)
        
        # Analyser les résultats uniquement si une zone d'intrusion a été définie
        if defined_zone is not None:
            intrusion_detected = False
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0: # 0 est l'ID de la classe 'person' dans le modèle COCO
                        # Obtenir les coordonnées dans la petite image redimensionnée
                        x1_r, y1_r, x2_r, y2_r = [int(i) for i in box.xyxy[0]]
                        
                        # Remettre à l'échelle les coordonnées pour l'image originale
                        x1 = int(x1_r / ratio)
                        y1 = int(y1_r / ratio)
                        x2 = int(x2_r / ratio)
                        y2 = int(y2_r / ratio)

                        # Le point de test pour l'intrusion : le centre en bas de la boîte
                        person_point = (int((x1 + x2) / 2), y2)

                        # --- AJOUT POUR LE DÉBOGAGE VISUEL ---
                        # Dessinons ce point pour voir où il se trouve !
                        cv2.circle(frame, person_point, 5, (255, 0, 0), -1) # Un cercle BLEU plein

                        # Vérifier si ce point est dans le polygone
                        is_intrusion = cv2.pointPolygonTest(defined_zone, person_point, False) >= 0
                        
                        # Imprimer le résultat du test dans la console pour le débogage
                        # -1.0 = dehors, 0.0 = sur le bord, 1.0 = dedans
                        print(f"Test d'intrusion pour la personne : {cv2.pointPolygonTest(defined_zone, person_point, False)}")

                        if is_intrusion:
                            intrusion_detected = True
                        
                        # Dessiner la boîte de détection et son label
                        conf = float(box.conf[0])
                        label = f"{self.model.names[0]} {conf:.2f}"
                        color = (0, 0, 255) if is_intrusion else (0, 255, 0) # Rouge si intrusion, vert sinon
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Si au moins une intrusion a été détectée dans l'image, afficher l'alerte globale
            if intrusion_detected:
                cv2.putText(frame, "ALERTE INTRUSION !", (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2)
        
        return frame

    def _cleanup(self):
        """Nettoie les ressources à la fin de l'exécution."""
        self.video_stream.stop()
        cv2.destroyAllWindows()
        print("Supervision terminée.")