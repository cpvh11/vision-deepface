"""Servidor web local (FastAPI) — la cámara se abre en el navegador."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from facekit.motor import MotorFacial
from facekit.personas import BasePersonas, RegistroError
from proyectos.asistencia import SistemaAsistencia
from proyectos.hogar import SistemaHogar
from proyectos.login import LoginFacial

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class Imagen(BaseModel):
    imagen: str

class Alta(BaseModel):
    nombre: str
    rol: str = "adulto"
    grupo: str = "General"
    contacto: str = ""
    preferencias: dict = Field(default_factory=dict)
    fotos: list[str]

class Acceso(BaseModel):
    usuario: str
    fotos: list[str]


def crear_app(hora_entrada: str = "09:00", tolerancia: int = 10, estimar_edad: bool = False) -> FastAPI:
    motor = MotorFacial()
    base = BasePersonas(motor)
    asistencia = SistemaAsistencia(base, motor, hora_entrada, tolerancia)
    login = LoginFacial(base, motor)
    hogar = SistemaHogar(base, motor, estimar_edad)

    app = FastAPI(title="Visión por computadora con DeepFace")

    def _img(fn, *a):
        try:
            return fn(*a)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.on_event("startup")
    def _calentar():
        print("Cargando modelos de DeepFace (la primera vez se descargan los pesos)...")
        motor.calentar()
        print("Listo. Abre http://127.0.0.1:8000")

    @app.get("/")
    def inicio():
        return FileResponse(os.path.join(WEB, "index.html"))

    # ---------- personas ----------
    @app.get("/api/personas")
    def personas():
        return base.listar()

    @app.post("/api/personas")
    def alta(d: Alta):
        try:
            return base.registrar(d.nombre, d.fotos, d.rol, d.grupo, d.contacto, d.preferencias)
        except RegistroError as e:
            raise HTTPException(422, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/personas/{pid}")
    def baja(pid: str):
        if not base.eliminar(pid):
            raise HTTPException(404, "No existe esa persona.")
        return {"ok": True}

    # ---------- proyecto 1 ----------
    @app.post("/api/asistencia/cuadro")
    def asistencia_cuadro(d: Imagen):
        return _img(asistencia.procesar, d.imagen)

    @app.get("/api/asistencia/resumen")
    def asistencia_resumen(grupo: str | None = None):
        return asistencia.resumen(grupo or None)

    @app.post("/api/asistencia/cerrar")
    def asistencia_cerrar(grupo: str | None = None):
        return asistencia.cerrar_dia(grupo or None)

    # ---------- proyecto 2 ----------
    @app.post("/api/login")
    def entrar(d: Acceso):
        return _img(login.iniciar_sesion, d.usuario, d.fotos)

    @app.get("/api/sesion/{token}")
    def sesion(token: str):
        s = login.sesion(token)
        if not s:
            raise HTTPException(401, "Sesión inválida o expirada.")
        return {**s, "historial": login.historial(s["usuario"])}

    @app.post("/api/logout/{token}")
    def salir(token: str):
        login.cerrar_sesion(token)
        return {"ok": True}

    # ---------- proyecto 3 ----------
    @app.post("/api/hogar/cuadro")
    def hogar_cuadro(d: Imagen):
        r = _img(hogar.procesar, d.imagen)
        r["bitacora"] = hogar.casa.bitacora[-15:]
        return r

    return app
