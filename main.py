"""Punto de entrada único.

    python main.py web                         # app con cámara en el navegador (recomendado)
    python main.py registrar --nombre "Ana López" --rol adulto --grupo "3A" --fotos a.jpg b.jpg
    python main.py personas
    python main.py asistencia --imagenes carpeta_o_foto [--cerrar]
    python main.py login --usuario ana-lopez --fotos f1.jpg f2.jpg f3.jpg
    python main.py hogar --imagenes carpeta_o_foto
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _imagenes(rutas: list[str]) -> list[str]:
    salida = []
    for r in rutas:
        if os.path.isdir(r):
            salida += sorted(p for p in glob.glob(os.path.join(r, "*")) if p.lower().endswith(EXT))
        elif os.path.isfile(r):
            salida.append(r)
        else:
            sys.exit(f"No existe: {r}")
    if not salida:
        sys.exit("No se encontraron imágenes.")
    return salida


def _contexto():
    from facekit.motor import MotorFacial
    from facekit.personas import BasePersonas
    motor = MotorFacial()
    return motor, BasePersonas(motor)


def cmd_web(a):
    import threading
    import webbrowser

    import uvicorn

    from servidor import crear_app
    app = crear_app(a.hora_entrada, a.tolerancia, a.estimar_edad)
    if not a.sin_navegador:
        threading.Timer(4, lambda: webbrowser.open(f"http://127.0.0.1:{a.puerto}")).start()
    uvicorn.run(app, host="127.0.0.1", port=a.puerto, log_level="warning")


def cmd_registrar(a):
    from facekit.personas import RegistroError
    _, base = _contexto()
    prefs = {k: v for k, v in {"musica": a.musica, "luz": a.luz, "temperatura": a.temperatura}.items() if v}
    try:
        p = base.registrar(a.nombre, _imagenes(a.fotos), a.rol, a.grupo, a.contacto or "", prefs)
    except RegistroError as e:
        sys.exit(f"✖ {e}")
    print(f"✔ Registrado: {p['nombre']}  (usuario: {p['id']}, rol: {p['rol']}, grupo: {p['grupo']}, "
          f"muestras: {p['muestras']})")


def cmd_personas(a):
    _, base = _contexto()
    ps = base.listar()
    if not ps:
        print("No hay personas registradas.")
    for p in ps:
        print(f"- {p['id']:<20} {p['nombre']:<25} rol={p['rol']:<7} grupo={p['grupo']:<10} muestras={p['muestras']}")


def cmd_eliminar(a):
    _, base = _contexto()
    print("✔ Eliminado" if base.eliminar(a.usuario) else "✖ No existe ese usuario")


def cmd_asistencia(a):
    from proyectos.asistencia import SistemaAsistencia
    motor, base = _contexto()
    s = SistemaAsistencia(base, motor, a.hora_entrada, a.tolerancia)
    for img in _imagenes(a.imagenes):
        r = s.procesar(img)
        quienes = [x.get("nombre", "Desconocido") + f" [{x['estado']}]" for x in r["rostros"]]
        print(f"{os.path.basename(img)}: {len(r['rostros'])} rostro(s) -> {', '.join(quienes) or 'ninguno'}")
    res = s.cerrar_dia(a.grupo) if a.cerrar else s.resumen(a.grupo)
    print(f"\nResumen {res['fecha']} — grupo {res['grupo']}: {len(res['presentes'])}/{res['total']} "
          f"presentes ({res['porcentaje']}%)")
    for p in res["presentes"]:
        print(f"  ✔ {p['nombre']:<25} {p['hora']}  {p['estado']}")
    for p in res["ausentes"]:
        print(f"  ✖ {p['nombre']:<25} AUSENTE")


def cmd_login(a):
    from proyectos.login import LoginFacial
    motor, base = _contexto()
    lg = LoginFacial(base, motor)
    r = lg.iniciar_sesion(a.usuario, _imagenes(a.fotos))
    for d in r.get("detalle", []):
        print("  captura", json.dumps(d, ensure_ascii=False))
    if r["ok"]:
        print(f"\n✔ ACCESO CONCEDIDO. Bienvenido/a {r['nombre']}. Token de sesión: {r['token'][:12]}…")
    else:
        print(f"\n✖ ACCESO DENEGADO: {r['motivo']}")
        sys.exit(1)


def cmd_hogar(a):
    from proyectos.hogar import SistemaHogar
    motor, base = _contexto()
    h = SistemaHogar(base, motor, a.estimar_edad)
    for img in _imagenes(a.imagenes):
        print(f"\n📷 {os.path.basename(img)}")
        h.ultimo_visto.clear()  # cada foto es un evento de llegada independiente
        r = h.procesar(img)
        if not r["rostros"]:
            print("   (sin rostros)")
    print("\nEstado final de la casa:", json.dumps(h.casa.estado, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description="Visión por computadora con DeepFace — 3 proyectos")
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("web", help="App web con cámara (los 3 proyectos)")
    w.add_argument("--puerto", type=int, default=8000)
    w.add_argument("--hora-entrada", default="09:00")
    w.add_argument("--tolerancia", type=int, default=10)
    w.add_argument("--estimar-edad", action="store_true", help="Estima edad de desconocidos (descarga ~500MB)")
    w.add_argument("--sin-navegador", action="store_true")
    w.set_defaults(fn=cmd_web)

    r = sub.add_parser("registrar", help="Registrar una persona con 1 o más fotos")
    r.add_argument("--nombre", required=True)
    r.add_argument("--fotos", nargs="+", required=True, help="Fotos o carpeta (un rostro por foto)")
    r.add_argument("--rol", choices=["adulto", "nino"], default="adulto")
    r.add_argument("--grupo", default="General")
    r.add_argument("--contacto", help="Correo/teléfono para avisos de ausencia")
    r.add_argument("--musica"); r.add_argument("--luz"); r.add_argument("--temperatura", type=int)
    r.set_defaults(fn=cmd_registrar)

    sub.add_parser("personas", help="Listar personas registradas").set_defaults(fn=cmd_personas)

    e = sub.add_parser("eliminar", help="Eliminar una persona")
    e.add_argument("usuario")
    e.set_defaults(fn=cmd_eliminar)

    s = sub.add_parser("asistencia", help="Proyecto 1 con fotos")
    s.add_argument("--imagenes", nargs="+", required=True)
    s.add_argument("--grupo")
    s.add_argument("--hora-entrada", default="09:00")
    s.add_argument("--tolerancia", type=int, default=10)
    s.add_argument("--cerrar", action="store_true", help="Cerrar el día y notificar ausencias")
    s.set_defaults(fn=cmd_asistencia)

    lg = sub.add_parser("login", help="Proyecto 2 con fotos")
    lg.add_argument("--usuario", required=True)
    lg.add_argument("--fotos", nargs="+", required=True, help="3 capturas del rostro")
    lg.set_defaults(fn=cmd_login)

    h = sub.add_parser("hogar", help="Proyecto 3 con fotos")
    h.add_argument("--imagenes", nargs="+", required=True)
    h.add_argument("--estimar-edad", action="store_true")
    h.set_defaults(fn=cmd_hogar)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
