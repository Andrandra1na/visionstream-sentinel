import os
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

class Settings:
    """
    Classe pour gérer les paramètres de configuration de l'application.
    """
    VIDEO_STREAM_URL: str = os.getenv("VIDEO_STREAM_URL")
    
    # Paramètres d'optimisation
    FRAME_WIDTH_FOR_PROCESSING: int = 320
    
    # Paramètres du modèle
    MODEL_PATH: str = 'yolov8n.pt'
    
    # Coordonnées de la zone interdite (exemple : un rectangle)
    # Format : [x_min, y_min, x_max, y_max]
    # NOTE : Ces coordonnées sont basées sur la taille de l'image TRAITÉE (ex: 320x180)
    # Nous allons définir une zone dans le quart inférieur droit de l'image
    FORBIDDEN_ZONE: list[int] = [160, 90, 320, 180] 


settings = Settings()