from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import asyncio
from fastapi.responses import FileResponse

from app.api.streaming import router as streaming_router, broadcast_frames
from app.vision.processor import video_processor

app = FastAPI(title="VisionStream Sentinel")

# Monter le dossier 'static' pour servir notre frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Inclure le routeur WebSocket
app.include_router(streaming_router)

@app.on_event("startup")
async def startup_event():
    """Actions à exécuter au démarrage de l'application."""
    print("Démarrage du traitement vidéo en arrière-plan...")
    video_processor.start_processing()
    # Lancer la tâche de diffusion en arrière-plan
    asyncio.create_task(broadcast_frames())
    print("Application démarrée et prête.")

@app.on_event("shutdown")
def shutdown_event():
    """Actions à exécuter à l'arrêt de l'application."""
    print("Arrêt du traitement vidéo...")
    video_processor.stop_processing()
    print("Application arrêtée.")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')