"""Proyecto 3 — Reconocimiento facial para hogar inteligente.

- Persona registrada como ADULTO  -> escena relajante con SUS preferencias (música, luz, temperatura).
- Persona registrada como NIÑO    -> modo entretenimiento infantil + control parental.
- Además se analiza la EMOCIÓN del rostro y la casa se adapta (p. ej. triste -> música animada).
- Persona DESCONOCIDA             -> alerta de seguridad (y, opcional, estimación de edad).
- Si nadie aparece durante un rato -> modo ahorro de energía.

Los dispositivos están simulados con print(), tal como permiten los lineamientos.
"""
from __future__ import annotations

import os
import time
from datetime import datetime

from facekit.motor import MotorFacial
from facekit.personas import DATA_DIR, BasePersonas

ENFRIAMIENTO_SEG = 60      # no repetir la bienvenida a la misma persona antes de este tiempo
AUSENCIA_SEG = 120         # sin nadie en cámara -> modo ahorro

ESCENA_ADULTO = {"musica": "Jazz relajante", "luz": "cálida 60%", "temperatura": 23}
ESCENA_NINO = {"musica": "Canciones infantiles", "luz": "brillante 100%", "temperatura": 24,
               "tv": "Caricaturas (control parental activado)"}

AJUSTE_EMOCION = {
    "sad": ("te noto triste", {"musica": "Playlist animada", "luz": "cálida brillante 80%"}),
    "angry": ("te noto estresado/a", {"musica": "Sonidos de naturaleza", "luz": "tenue 40%"}),
    "fear": ("te noto inquieto/a", {"luz": "brillante 100%"}),
    "happy": ("¡qué buen humor!", {}),
}


class CasaInteligente:
    """Dispositivos simulados."""

    def __init__(self):
        self.estado = {"luz": "apagada", "musica": "ninguna", "temperatura": 25, "tv": "apagada",
                       "seguridad": "normal", "modo": "ahorro"}
        self.bitacora: list[str] = []

    def _accion(self, texto: str) -> None:
        linea = f"{datetime.now().strftime('%H:%M:%S')}  {texto}"
        print(f"[CASA] {linea}")
        self.bitacora.append(linea)
        self.bitacora = self.bitacora[-50:]
        with open(os.path.join(DATA_DIR, "hogar.log"), "a", encoding="utf-8") as f:
            f.write(linea + "\n")

    def aplicar(self, escena: dict, motivo: str) -> None:
        self._accion(f"▶ {motivo}")
        if "luz" in escena:
            self._accion(f"   💡 Ajustando luces: {escena['luz']}")
        if "musica" in escena:
            self._accion(f"   🎵 Poniendo música: {escena['musica']}")
        if "temperatura" in escena:
            self._accion(f"   🌡️ Termostato a {escena['temperatura']} °C")
        if "tv" in escena:
            self._accion(f"   📺 TV: {escena['tv']}")
        self.estado.update({k: v for k, v in escena.items() if k in self.estado})
        self.estado["modo"] = "ocupada"

    def alerta(self, texto: str) -> None:
        self.estado["seguridad"] = "ALERTA"
        self._accion(f"🚨 {texto}")
        self._accion("   📲 Notificando al dueño de la casa (simulado)")
        self._accion("   📸 Foto del visitante guardada en data/visitantes/")

    def ahorro(self) -> None:
        self.estado.update({"luz": "apagada", "musica": "ninguna", "tv": "apagada", "modo": "ahorro",
                            "seguridad": "normal"})
        self._accion("🌙 Nadie en casa: modo ahorro de energía (luces y música apagadas)")


class SistemaHogar:
    def __init__(self, base: BasePersonas, motor: MotorFacial, estimar_edad: bool = False):
        self.base, self.motor, self.estimar_edad = base, motor, estimar_edad
        self.casa = CasaInteligente()
        self.ultimo_visto: dict[str, float] = {}
        self.ultima_presencia = time.time()
        os.makedirs(os.path.join(DATA_DIR, "visitantes"), exist_ok=True)

    def procesar(self, imagen) -> dict:
        import cv2
        from facekit.motor import leer_imagen

        img = leer_imagen(imagen)
        ahora = time.time()
        rostros = self.motor.detectar(img)
        base = self.base.embeddings()
        eventos = []

        if not rostros:
            if self.casa.estado["modo"] == "ocupada" and ahora - self.ultima_presencia > AUSENCIA_SEG:
                self.casa.ahorro()
                eventos.append("modo ahorro")
            return {"rostros": [], "eventos": eventos, "casa": self.casa.estado}
        self.ultima_presencia = ahora

        for r in rostros:
            self.motor.identificar(r, base)
            clave = r.nombre or "desconocido"
            nuevo = ahora - self.ultimo_visto.get(clave, 0) > ENFRIAMIENTO_SEG
            acciones = ("emotion", "age") if (not r.nombre and self.estimar_edad) else ("emotion",)
            if nuevo:
                r.extra.update(self.motor.analizar_atributos(img, r.caja, acciones))
            self.ultimo_visto[clave] = ahora

            if r.nombre:
                p = self.base.obtener(r.nombre)
                r.extra["nombre"], r.extra["rol"] = p["nombre"], p["rol"]
                if not nuevo:
                    continue
                es_nino = p["rol"] == "nino"
                escena = dict(ESCENA_NINO if es_nino else ESCENA_ADULTO)
                escena.update({k: v for k, v in (p.get("preferencias") or {}).items() if v not in ("", None)})
                motivo = f"Bienvenido/a {p['nombre']} ({'niño/a' if es_nino else 'adulto'})"
                emo = r.extra.get("emocion")
                if emo in AJUSTE_EMOCION:
                    frase, ajuste = AJUSTE_EMOCION[emo]
                    if not es_nino:
                        escena.update(ajuste)
                    motivo += f" — {frase}"
                self.casa.aplicar(escena, motivo)
                eventos.append(motivo)
            elif nuevo:
                edad = r.extra.get("edad")
                if edad is not None and edad < 13:
                    texto = f"Niño/a desconocido/a detectado/a (edad estimada {edad})"
                else:
                    texto = "Persona DESCONOCIDA en la entrada" + (f" (edad estimada {edad})" if edad else "")
                x, y, w, h = r.caja.values()
                cv2.imwrite(os.path.join(DATA_DIR, "visitantes", f"{datetime.now():%Y%m%d_%H%M%S}.jpg"),
                            img[max(0, y - 20): y + h + 20, max(0, x - 20): x + w + 20])
                self.casa.alerta(texto)
                eventos.append(texto)

        return {"rostros": [r.a_dict() for r in rostros], "eventos": eventos, "casa": self.casa.estado}
