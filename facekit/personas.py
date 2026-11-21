"""Base de datos local de personas registradas (JSON en data/personas.json)."""
from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from datetime import datetime

import numpy as np

from .motor import MotorFacial

DATA_DIR = os.getenv("FR_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
ROLES = ("adulto", "nino")


def slug(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


class RegistroError(ValueError):
    pass


class BasePersonas:
    def __init__(self, motor: MotorFacial, ruta: str | None = None):
        self.motor = motor
        self.ruta = ruta or os.path.join(DATA_DIR, "personas.json")
        os.makedirs(os.path.dirname(self.ruta), exist_ok=True)
        self._lock = threading.Lock()
        self._datos: dict[str, dict] = {}
        if os.path.exists(self.ruta):
            with open(self.ruta, encoding="utf-8") as f:
                self._datos = json.load(f)

    # ---------- persistencia ----------
    def _guardar(self) -> None:
        tmp = self.ruta + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._datos, f, ensure_ascii=False)
        os.replace(tmp, self.ruta)

    # ---------- consultas ----------
    def listar(self) -> list[dict]:
        return [self.publico(pid) for pid in sorted(self._datos)]

    def publico(self, pid: str) -> dict:
        p = dict(self._datos[pid])
        p["muestras"] = len(p.pop("embeddings", []))
        return p

    def obtener(self, pid: str) -> dict | None:
        return self._datos.get(pid)

    def embeddings(self) -> dict[str, list[np.ndarray]]:
        return {pid: [np.asarray(e, dtype=np.float32) for e in p["embeddings"]] for pid, p in self._datos.items()}

    def nombre(self, pid: str | None) -> str:
        return self._datos[pid]["nombre"] if pid in self._datos else "Desconocido"

    # ---------- altas y bajas ----------
    def registrar(self, nombre: str, imagenes: list, rol: str = "adulto", grupo: str = "General",
                  contacto: str = "", preferencias: dict | None = None) -> dict:
        nombre = nombre.strip()
        if not nombre:
            raise RegistroError("El nombre es obligatorio.")
        if rol not in ROLES:
            raise RegistroError(f"Rol inválido '{rol}'. Usa: {', '.join(ROLES)}.")
        if not imagenes:
            raise RegistroError("Se necesita al menos una foto.")
        pid = slug(nombre)

        nuevos = []
        for i, img in enumerate(imagenes, 1):
            rostros = self.motor.detectar(img)
            if len(rostros) != 1:
                raise RegistroError(
                    f"Foto {i}: se detectaron {len(rostros)} rostros. Cada foto debe tener exactamente uno.")
            nuevos.append(rostros[0])

        # Consistencia: todas las fotos deben ser de la misma persona.
        ref = nuevos[0].embedding
        for i, r in enumerate(nuevos[1:], 2):
            if self.motor.distancia(ref, r.embedding) > self.motor.umbral:
                raise RegistroError(f"La foto {i} no parece ser de la misma persona que la foto 1.")

        # Evitar que alguien se registre con el rostro de otra persona ya registrada.
        otros = {k: v for k, v in self.embeddings().items() if k != pid}
        for r in nuevos:
            self.motor.identificar(r, otros)
            if r.nombre:
                raise RegistroError(f"Ese rostro ya está registrado como '{self.nombre(r.nombre)}'.")

        with self._lock:
            previo = self._datos.get(pid, {})
            self._datos[pid] = {
                "id": pid,
                "nombre": nombre,
                "rol": rol,
                "grupo": grupo.strip() or "General",
                "contacto": contacto.strip(),
                "preferencias": preferencias or previo.get("preferencias") or {},
                "registrado": previo.get("registrado") or datetime.now().isoformat(timespec="seconds"),
                "embeddings": previo.get("embeddings", []) + [r.embedding.round(6).tolist() for r in nuevos],
            }
            self._guardar()
        return self.publico(pid)

    def eliminar(self, pid: str) -> bool:
        with self._lock:
            if self._datos.pop(pid, None) is None:
                return False
            self._guardar()
            return True
