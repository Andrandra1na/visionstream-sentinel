from pydantic import BaseModel
from typing import List

class BoundingBox(BaseModel):
    """
    Définit une boîte de détection.
    Les coordonnées sont pour l'image originale, pas l'image redimensionnée.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str

class WebSocketResponse(BaseModel):
    """
    Définit la structure du message envoyé via WebSocket.
    """
    image_base64: str  # L'image originale avec les détections, encodée en base64
    detections: List[BoundingBox]
    status: str  # ex: "ok" ou "INTRUSION"