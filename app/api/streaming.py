import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List

from app.vision.processor import video_processor
from app.schemas.detection import WebSocketResponse

router = APIRouter()

class ConnectionManager:
    """Gère les connexions WebSocket actives."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        """Diffuse des données à toutes les connexions actives."""
        # Valider les données avec notre schéma Pydantic
        response = WebSocketResponse(**data)
        for connection in self.active_connections:
            await connection.send_json(response.dict())

manager = ConnectionManager()

async def broadcast_frames():
    """Tâche de fond qui diffuse les frames analysées."""
    while True:
        if video_processor.latest_frame_data:
            await manager.broadcast(video_processor.latest_frame_data)
        # Contrôle la fréquence de diffusion pour ne pas surcharger
        await asyncio.sleep(0.05) # Diffuse environ 20 FPS

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Maintenir la connexion ouverte
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)