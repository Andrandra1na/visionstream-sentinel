import cv2
import os
from dotenv import load_dotenv
from ultralytics import YOLO

def main():
    
    load_dotenv()
    video_stream_url = os.getenv("VIDEO_STREAM_URL")

    if not video_stream_url:
        print("Erreur : La variable d'environnement VIDEO_STREAM_URL n'est pas définie.")
        print("Veuillez créer un fichier .env et y ajouter la ligne :")
        print('VIDEO_STREAM_URL="http://VOTRE_IP:PORT/video"')
        return

    try:
        model = YOLO('./models/yolov8n.pt')
        print("Modèle YOLOv8 chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle YOLOv8 : {e}")
        return

    cap = cv2.VideoCapture(video_stream_url)

    if not cap.isOpened():
        print(f"Erreur : Impossible de se connecter au flux vidéo à l'adresse : {video_stream_url}")
        print("Vérifiez les points suivants :")
        print("- Votre téléphone et votre ordinateur sont sur le même réseau Wi-Fi.")
        print("- L'application 'IP Webcam' est en cours d'exécution sur votre téléphone.")
        print("- L'URL dans votre fichier .env est correcte.")
        return
    
    print("Connexion au flux vidéo réussie. Une fenêtre va s'ouvrir. Appuyez sur 'q' pour quitter.")

    # 4. Boucle principale pour traiter chaque image du flux
    while True:
        # Lire une image (frame) depuis le flux
        ret, frame = cap.read()

        # Si la lecture échoue (ex: fin du flux ou erreur), on sort de la boucle
        if not ret:
            print("Erreur : Impossible de recevoir l'image du flux. Fin du programme.")
            break

        # 5. Effectuer la détection d'objets sur l'image
        results = model(frame)

        # 6. Obtenir l'image avec les boîtes de détection dessinées
        # La méthode .plot() de YOLOv8 renvoie une image NumPy (BGR) avec les annotations
        annotated_frame = results[0].plot()

        # 7. Afficher l'image annotée dans une fenêtre
        cv2.imshow("VisionStream Sentinel - Test de Vision", annotated_frame)

        # Attendre 1ms et vérifier si la touche 'q' a été pressée pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 8. Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
    print("Flux vidéo et fenêtres fermés proprement.")

if __name__ == "__main__":
    main()