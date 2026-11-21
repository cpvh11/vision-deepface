"""Proyecto 1 — Sistema de asistencia automatizada con reconocimiento facial.

- Reconoce a varias personas en el mismo cuadro (salón u oficina).
- Registra la asistencia una sola vez por día en data/asistencia/AAAA-MM-DD.csv
- Marca "puntual" o "retardo" según la hora de entrada + tolerancia.
- Al cerrar el día genera la lista de ausentes y notifica (simulado) al contacto
  de cada ausente (padre de familia / supervisor).
"""
from __future__ import annotations

import csv
import os
import threading
from datetime import datetime, time, timedelta

from facekit.motor import MotorFacial
from facekit.personas import DATA_DIR, BasePersonas

CAMPOS = ["fecha", "hora", "id", "nombre", "grupo", "estado", "distancia"]


class SistemaAsistencia:
    def __init__(self, base: BasePersonas, motor: MotorFacial,
                 hora_entrada: str = "09:00", tolerancia_min: int = 10):
        self.base, self.motor = base, motor
        h, m = map(int, hora_entrada.split(":"))
        self.limite = (datetime.combine(datetime.today(), time(h, m)) + timedelta(minutes=tolerancia_min)).time()
        self.hora_entrada, self.tolerancia = hora_entrada, tolerancia_min
        self.carpeta = os.path.join(DATA_DIR, "asistencia")
        os.makedirs(self.carpeta, exist_ok=True)
        self._lock = threading.Lock()

    def _archivo(self, fecha: str) -> str:
        return os.path.join(self.carpeta, f"{fecha}.csv")

    def registros(self, fecha: str | None = None) -> list[dict]:
        ruta = self._archivo(fecha or datetime.now().strftime("%Y-%m-%d"))
        if not os.path.exists(ruta):
            return []
        with open(ruta, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def procesar(self, imagen, ahora: datetime | None = None) -> dict:
        """Analiza un cuadro de la cámara y registra a quien no tenga asistencia hoy."""
        ahora = ahora or datetime.now()
        fecha = ahora.strftime("%Y-%m-%d")
        rostros = self.motor.detectar(imagen)
        base = self.base.embeddings()
        nuevos = []
        with self._lock:
            ya = {r["id"] for r in self.registros(fecha)}
            for r in rostros:
                self.motor.identificar(r, base)
                if not r.nombre:
                    r.extra["estado"] = "desconocido"
                    continue
                p = self.base.obtener(r.nombre)
                r.extra["nombre"] = p["nombre"]
                if r.nombre in ya:
                    r.extra["estado"] = "ya registrado"
                    continue
                estado = "puntual" if ahora.time() <= self.limite else "retardo"
                fila = {"fecha": fecha, "hora": ahora.strftime("%H:%M:%S"), "id": r.nombre,
                        "nombre": p["nombre"], "grupo": p["grupo"], "estado": estado,
                        "distancia": f"{r.distancia:.3f}"}
                self._escribir(fecha, fila)
                ya.add(r.nombre)
                r.extra["estado"] = f"asistencia: {estado}"
                nuevos.append(fila)
                print(f"[ASISTENCIA] {fila['hora']}  {p['nombre']:<25} {p['grupo']:<12} {estado.upper()}")
        return {"rostros": [r.a_dict() for r in rostros], "nuevos": nuevos}

    def _escribir(self, fecha: str, fila: dict) -> None:
        ruta = self._archivo(fecha)
        nuevo = not os.path.exists(ruta)
        with open(ruta, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS)
            if nuevo:
                w.writeheader()
            w.writerow(fila)

    def resumen(self, grupo: str | None = None, fecha: str | None = None) -> dict:
        fecha = fecha or datetime.now().strftime("%Y-%m-%d")
        regs = {r["id"]: r for r in self.registros(fecha)}
        personas = [p for p in self.base.listar() if not grupo or p["grupo"] == grupo]
        presentes = [dict(regs[p["id"]]) for p in personas if p["id"] in regs]
        ausentes = [p for p in personas if p["id"] not in regs]
        return {
            "fecha": fecha, "grupo": grupo or "Todos", "hora_entrada": self.hora_entrada,
            "tolerancia_min": self.tolerancia, "total": len(personas),
            "presentes": presentes, "ausentes": [{"id": p["id"], "nombre": p["nombre"], "grupo": p["grupo"],
                                                    "contacto": p["contacto"]} for p in ausentes],
            "porcentaje": round(100 * len(presentes) / len(personas), 1) if personas else 0.0,
        }

    def cerrar_dia(self, grupo: str | None = None, fecha: str | None = None) -> dict:
        """Genera el reporte del día y envía (simulado) avisos por cada ausencia."""
        res = self.resumen(grupo, fecha)
        avisos = []
        log = os.path.join(DATA_DIR, "notificaciones.log")
        with open(log, "a", encoding="utf-8") as f:
            for a in res["ausentes"]:
                destino = a["contacto"] or "(sin contacto registrado)"
                msg = (f"Aviso: {a['nombre']} ({a['grupo']}) no registró asistencia el {res['fecha']}.")
                print(f"[NOTIFICACIÓN simulada] Enviando a {destino}: {msg}")
                f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{destino}\t{msg}\n")
                avisos.append({"para": destino, "mensaje": msg})
        res["notificaciones"] = avisos
        return res
