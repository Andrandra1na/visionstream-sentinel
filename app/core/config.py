import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Classe pour centraliser tous les paramètres configurables."""
    
    # Source Vidéo
    VIDEO_STREAM_URL: str = os.getenv("VIDEO_STREAM_URL")
    
    # Paramètres du Modèle IA
    MODEL_PATH: str = 'yolov8n.pt'
    
    # Paramètres de Performance
    FRAME_WIDTH_FOR_PROCESSING: int = 320 # La largeur à laquelle l'image est redimensionnée pour l'analyse
    
    # Paramètres de la Logique Métier
    # Coordonnées [x_min, y_min, x_max, y_max] basées sur la résolution de traitement
    FORBIDDEN_ZONE: list[int] = [160, 90, 320, 180] 

# Création d'une instance unique des paramètres pour être importée dans d'autres fichiers
settings = Settings()