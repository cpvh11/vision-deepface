"""Motor de visión por computadora: detección + embeddings + identificación con DeepFace.

Flujo:
    imagen -> detector (OpenCV) -> rostro alineado -> red neuronal (Facenet512)
           -> vector de 512 números (embedding) -> distancia coseno contra la base
"""
from __future__ import annotations

import base64
import os
import threading
from dataclasses import dataclass, field

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from deepface import DeepFace  # noqa: E402
from deepface.modules.verification import find_threshold  # noqa: E402

MODELO = os.getenv("FR_MODELO", "Facenet512")
DETECTOR = os.getenv("FR_DETECTOR", "opencv")
METRICA = "cosine"

# TensorFlow no es seguro con varios hilos usando el mismo modelo: serializamos.
_LOCK = threading.Lock()


@dataclass
class Rostro:
    caja: dict                       # {x, y, w, h}
    embedding: np.ndarray            # vector normalizado
    confianza: float = 1.0
    nombre: str | None = None        # id de la persona reconocida (o None)
    distancia: float | None = None
    extra: dict = field(default_factory=dict)

    def a_dict(self) -> dict:
        d = {"caja": self.caja, "id": self.nombre, "confianza": round(self.confianza, 3)}
        if self.distancia is not None:
            d["distancia"] = round(float(self.distancia), 3)
        d.update(self.extra)
        return d


def leer_imagen(origen) -> np.ndarray:
    """Acepta ruta, bytes, DataURL/base64 o un arreglo numpy y devuelve imagen BGR."""
    if isinstance(origen, np.ndarray):
        return origen
    if isinstance(origen, str) and os.path.exists(origen):
        img = cv2.imread(origen)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {origen}")
        return img
    if isinstance(origen, str):
        origen = base64.b64decode(origen.split(",", 1)[-1])
    img = cv2.imdecode(np.frombuffer(origen, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("La imagen recibida no es válida.")
    return img


def _normalizar(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


class MotorFacial:
    def __init__(self, modelo: str = MODELO, detector: str = DETECTOR):
        self.modelo = modelo
        self.detector = detector
        self.umbral = float(find_threshold(modelo, METRICA))

    def calentar(self) -> None:
        """Carga los modelos en memoria (y descarga los pesos la primera vez)."""
        self.detectar(np.zeros((160, 160, 3), dtype=np.uint8))

    def detectar(self, imagen) -> list[Rostro]:
        """Detecta todos los rostros de la imagen y calcula su embedding."""
        img = leer_imagen(imagen)
        try:
            with _LOCK:
                res = DeepFace.represent(
                    img, model_name=self.modelo, detector_backend=self.detector,
                    enforce_detection=True, align=True,
                )
        except ValueError:  # DeepFace lanza ValueError cuando no hay rostro
            return []
        rostros = []
        for r in res:
            a = r["facial_area"]
            if a.get("w", 0) < 40 or a.get("h", 0) < 40:  # descarta rostros diminutos / falsos positivos
                continue
            rostros.append(Rostro(
                caja={k: int(a[k]) for k in ("x", "y", "w", "h")},
                embedding=_normalizar(r["embedding"]),
                confianza=float(r.get("face_confidence") or 1.0),
            ))
        return rostros

    @staticmethod
    def distancia(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(_normalizar(a), _normalizar(b)))

    def identificar(self, rostro: Rostro, base: dict[str, list[np.ndarray]],
                    factor_umbral: float = 1.0) -> Rostro:
        """Reconocimiento 1:N — compara contra todas las personas registradas."""
        mejor, mejor_d = None, 9.0
        for pid, embs in base.items():
            if not embs:
                continue
            d = min(self.distancia(rostro.embedding, e) for e in embs)
            if d < mejor_d:
                mejor, mejor_d = pid, d
        rostro.distancia = mejor_d if mejor else None
        rostro.nombre = mejor if mejor and mejor_d <= self.umbral * factor_umbral else None
        return rostro

    def analizar_atributos(self, imagen, caja: dict, acciones=("emotion",)) -> dict:
        """Análisis de atributos faciales (emoción, edad) sobre el rostro ya recortado."""
        img = leer_imagen(imagen)
        x, y, w, h = caja["x"], caja["y"], caja["w"], caja["h"]
        m = int(0.15 * max(w, h))
        recorte = img[max(0, y - m): y + h + m, max(0, x - m): x + w + m]
        try:
            with _LOCK:
                r = DeepFace.analyze(recorte, actions=list(acciones), detector_backend="skip",
                                     enforce_detection=False, silent=True)
        except Exception:
            return {}
        r = r[0] if isinstance(r, list) else r
        out = {}
        if "dominant_emotion" in r:
            out["emocion"] = r["dominant_emotion"]
        if "age" in r:
            out["edad"] = int(r["age"])
        return out
