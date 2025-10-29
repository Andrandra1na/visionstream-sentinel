from app.vision.supervisor import Supervisor
from app.core.config import settings

def main():
    """
    Point d'entrée principal de l'application de supervision.
    """
    if not settings.VIDEO_STREAM_URL:
        print("Erreur critique : La variable d'environnement VIDEO_STREAM_URL n'est pas définie.")
        print("Veuillez vérifier votre fichier .env.")
        return

    try:
        supervisor_app = Supervisor()
        supervisor_app.run()
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        print("Assurez-vous que l'URL du flux vidéo est correcte et que le flux est actif.")

if __name__ == "__main__":
    main()