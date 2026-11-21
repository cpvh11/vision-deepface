"""Proyecto 2 — Inicio de sesión seguro con verificación facial (sin contraseña).

Medidas de seguridad implementadas:
- Verificación 1:1: el rostro se compara SOLO contra el usuario que intenta entrar,
  con un umbral más estricto (80%) que el de reconocimiento normal.
- Prueba multi-cuadro: se piden 3 capturas y deben aprobar al menos 3 de 3
  (reduce falsos positivos por un solo cuadro "afortunado").
- Rechazo si la cámara ve más de un rostro (evita que otro "ayude" a entrar).
- Control cruzado: si el rostro se parece más a OTRA persona registrada, se rechaza.
- Bloqueo temporal tras 3 intentos fallidos (5 minutos).
- Bitácora de accesos en data/accesos.log y sesiones con token y expiración.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from datetime import datetime

from facekit.motor import MotorFacial
from facekit.personas import DATA_DIR, BasePersonas

FACTOR_ESTRICTO = 0.8
CAPTURAS_REQUERIDAS = 3
MAX_FALLOS = 3
BLOQUEO_SEG = 5 * 60
DURACION_SESION_SEG = 30 * 60


class LoginFacial:
    def __init__(self, base: BasePersonas, motor: MotorFacial):
        self.base, self.motor = base, motor
        self.sesiones: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.bitacora = os.path.join(DATA_DIR, "accesos.log")
        # Los intentos fallidos se guardan en disco para que el bloqueo sobreviva reinicios.
        self._ruta_fallos = os.path.join(DATA_DIR, "intentos_fallidos.json")
        self.fallos: dict[str, list[float]] = {}
        if os.path.exists(self._ruta_fallos):
            with open(self._ruta_fallos, encoding="utf-8") as f:
                self.fallos = json.load(f)

    def _guardar_fallos(self) -> None:
        with open(self._ruta_fallos, "w", encoding="utf-8") as f:
            json.dump(self.fallos, f)

    def _log(self, usuario: str, resultado: str, detalle: str) -> None:
        linea = f"{datetime.now().isoformat(timespec='seconds')}\t{usuario}\t{resultado}\t{detalle}"
        print(f"[LOGIN] {linea}")
        with open(self.bitacora, "a", encoding="utf-8") as f:
            f.write(linea + "\n")

    def bloqueado(self, usuario: str) -> int:
        """Segundos restantes de bloqueo (0 si no está bloqueado)."""
        ahora = time.time()
        recientes = [t for t in self.fallos.get(usuario, []) if ahora - t < BLOQUEO_SEG]
        self.fallos[usuario] = recientes
        if len(recientes) >= MAX_FALLOS:
            return int(BLOQUEO_SEG - (ahora - recientes[0])) + 1
        return 0

    def iniciar_sesion(self, usuario: str, capturas: list) -> dict:
        usuario = usuario.strip().lower()
        perfil = self.base.obtener(usuario)
        if not perfil:
            self._log(usuario, "RECHAZADO", "usuario inexistente")
            return {"ok": False, "motivo": "Usuario no registrado."}
        with self._lock:
            espera = self.bloqueado(usuario)
        if espera:
            self._log(usuario, "BLOQUEADO", f"{espera}s restantes")
            return {"ok": False, "motivo": f"Cuenta bloqueada por intentos fallidos. Intenta en {espera} s."}
        if len(capturas) < CAPTURAS_REQUERIDAS:
            return {"ok": False, "motivo": f"Se requieren {CAPTURAS_REQUERIDAS} capturas."}

        base = self.base.embeddings()
        propios = base[usuario]
        umbral = self.motor.umbral * FACTOR_ESTRICTO
        detalle, aprobadas = [], 0
        for i, img in enumerate(capturas[:CAPTURAS_REQUERIDAS], 1):
            rostros = self.motor.detectar(img)
            if len(rostros) != 1:
                detalle.append({"captura": i, "ok": False, "motivo": f"{len(rostros)} rostros detectados"})
                continue
            r = rostros[0]
            d = min(self.motor.distancia(r.embedding, e) for e in propios)
            self.motor.identificar(r, base)  # ¿a quién se parece más en toda la base?
            otro = r.nombre and r.nombre != usuario
            ok = d <= umbral and not otro
            aprobadas += ok
            detalle.append({"captura": i, "ok": bool(ok), "distancia": round(d, 3),
                            "umbral": round(umbral, 3), **({"motivo": "se parece más a otra persona"} if otro else {})})

        exito = aprobadas >= CAPTURAS_REQUERIDAS
        with self._lock:
            if exito:
                self.fallos.pop(usuario, None)
                token = secrets.token_urlsafe(24)
                self.sesiones[token] = {"usuario": usuario, "nombre": perfil["nombre"],
                                        "inicio": datetime.now().isoformat(timespec="seconds"),
                                        "expira": time.time() + DURACION_SESION_SEG}
            else:
                self.fallos.setdefault(usuario, []).append(time.time())
                restantes = max(0, MAX_FALLOS - len(self.fallos[usuario]))
            self._guardar_fallos()
        self._log(usuario, "ACCESO" if exito else "RECHAZADO", f"{aprobadas}/{CAPTURAS_REQUERIDAS} capturas válidas")
        if exito:
            return {"ok": True, "token": token, "nombre": perfil["nombre"], "detalle": detalle}
        return {"ok": False, "detalle": detalle,
                "motivo": f"Verificación fallida ({aprobadas}/{CAPTURAS_REQUERIDAS}). Intentos restantes: {restantes}."}

    def sesion(self, token: str) -> dict | None:
        s = self.sesiones.get(token)
        if not s or s["expira"] < time.time():
            self.sesiones.pop(token, None)
            return None
        return {k: v for k, v in s.items() if k != "expira"}

    def cerrar_sesion(self, token: str) -> None:
        s = self.sesiones.pop(token, None)
        if s:
            self._log(s["usuario"], "LOGOUT", "sesión cerrada")

    def historial(self, usuario: str, n: int = 10) -> list[str]:
        if not os.path.exists(self.bitacora):
            return []
        with open(self.bitacora, encoding="utf-8") as f:
            lineas = [l.rstrip("\n") for l in f if f"\t{usuario}\t" in l]
        return lineas[-n:]
