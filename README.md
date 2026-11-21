# 👁️ Visión por computadora con DeepFace — 3 proyectos

Implementación de los **tres proyectos** del checkpoint, construidos sobre un mismo motor de reconocimiento facial:

| # | Proyecto | Qué hace |
|---|----------|----------|
| 1 | **Asistencia automatizada** | Reconoce a varias personas en el mismo cuadro, registra su asistencia una vez por día (puntual / retardo), genera el reporte y avisa (simulado) al padre o supervisor de cada ausente. |
| 2 | **Inicio de sesión con verificación facial** | Login **sin contraseña**: verificación 1:1 con umbral estricto, 3 capturas obligatorias, rechazo si hay más de un rostro o si se parece más a otra persona, bloqueo tras 3 intentos fallidos y bitácora de accesos. |
| 3 | **Hogar inteligente** | Reconoce quién llega: adulto → escena relajante con sus preferencias; niño → modo infantil con control parental; además lee la **emoción** y ajusta la casa (triste → música animada). Un desconocido activa una alerta de seguridad. Si no hay nadie, entra en modo ahorro. Los dispositivos están simulados con `print`, como permiten los lineamientos. |

Se puede usar de dos formas:
- **App web local** (recomendada): la cámara se abre en el navegador y funciona igual en macOS, Windows y Linux.
- **Línea de comandos** con fotos, para probar sin cámara.

---

## 🚀 Instalación

Requisitos: **Python 3.10 – 3.13** (probado con 3.13) y conexión a internet la primera vez.

```bash
git clone <URL-de-este-repositorio>
cd <carpeta-del-repositorio>

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> La primera ejecución descarga los pesos preentrenados de DeepFace a `~/.deepface` (Facenet512, unos 95 MB, y el modelo de emociones, unos 6 MB). Solo ocurre una vez.

---

## ▶️ Opción A — App web con cámara (los 3 proyectos)

```bash
python main.py web
```

Se abre el navegador en **http://127.0.0.1:8000** (acepta el permiso de la cámara). Si no se abre solo, entra manualmente a esa dirección.

**Guía de prueba (unos 3 minutos):**

1. **Pestaña 0 · Registro**: escribe tu nombre, presiona **📸 Capturar** 3 veces (mueve un poco la cabeza entre capturas) y luego **Registrar**. Anota tu *usuario* (p. ej. `ana-lopez`). Puedes registrar a otra persona con rol *Niño/a* para probar el proyecto 3.
2. **Pestaña 1 · Asistencia**: **▶ Iniciar**. Tu rostro aparece en verde con *asistencia: puntual/retardo* y se agrega a la tabla. Si alguien no registrado entra al cuadro, aparece en rojo. Presiona **Cerrar día** para ver el reporte y las notificaciones de ausencia.
3. **Pestaña 2 · Login facial**: escribe tu usuario y presiona **🔐 Verificar rostro**. Entras al área privada con tu historial de accesos. Prueba con otra persona frente a la cámara: se rechaza, y al tercer intento la cuenta se bloquea 5 minutos.
4. **Pestaña 3 · Hogar inteligente**: **▶ Iniciar**. Al reconocerte, la casa ajusta luces, música y temperatura según tu rol, tus preferencias y tu emoción (lo verás en la bitácora y en la terminal). Un rostro desconocido dispara la alerta de seguridad.

¿No tienes cámara? Usa **🖼️ Usar foto** en cualquier pestaña para analizar una imagen.

Opciones: `--puerto 8000`, `--hora-entrada 08:00`, `--tolerancia 10` (minutos), `--estimar-edad` (estima la edad de desconocidos; descarga unos 500 MB adicionales).

---

## ⌨️ Opción B — Línea de comandos (sin cámara)

Usa tus propias fotos (un rostro por foto para el registro).

```bash
# Registro (1 a 5 fotos; carpeta o archivos)
python main.py registrar --nombre "Ana López" --rol adulto --grupo 3A \
    --contacto mama.ana@mail.com --musica "Bossa nova" --fotos ana1.jpg ana2.jpg ana3.jpg
python main.py registrar --nombre "Carlos" --rol nino --grupo 3A --fotos carlos1.jpg
python main.py personas

# Proyecto 1 — procesa fotos del salón (pueden tener varias personas) y cierra el día
python main.py asistencia --imagenes fotos_salon/ --hora-entrada 08:00 --cerrar

# Proyecto 2 — login con 3 capturas
python main.py login --usuario ana-lopez --fotos captura1.jpg captura2.jpg captura3.jpg

# Proyecto 3 — simula llegadas a casa
python main.py hogar --imagenes llegada1.jpg llegada2.jpg
```

Ejemplo de salida del proyecto 3:
```
📷 llegada1.jpg
[CASA] 09:47:57  ▶ Bienvenido/a Ana López (adulto) — ¡qué buen humor!
[CASA] 09:47:57     💡 Ajustando luces: cálida 60%
[CASA] 09:47:57     🎵 Poniendo música: Bossa nova
[CASA] 09:47:57     🌡️ Termostato a 23 °C
📷 llegada2.jpg
[CASA] 09:47:58  🚨 Persona DESCONOCIDA en la entrada
[CASA] 09:47:58     📲 Notificando al dueño de la casa (simulado)
```

---

## 🧠 Cómo funciona

```
Imagen ─▶ Detector de rostros (OpenCV) ─▶ Alineación ─▶ Red neuronal Facenet512
                                                              │
                                     embedding: vector de 512 números
                                                              │
                    Distancia coseno contra los embeddings registrados
                                                              │
                     distancia ≤ umbral (0.30) ─▶ misma persona
```

- **Registro**: por cada foto se guarda su *embedding* (no se guardan las fotos). Se valida que cada foto tenga un solo rostro, que todas sean de la misma persona y que ese rostro no esté ya registrado con otro nombre.
- **Reconocimiento 1:N** (asistencia y hogar): se busca la persona más cercana entre todas las registradas.
- **Verificación 1:1** (login): solo se compara contra el usuario que intenta entrar, con un umbral 20% más estricto (0.24).
- **Análisis de atributos** (hogar): `DeepFace.analyze` sobre el rostro recortado para obtener la emoción (y la edad, opcional).

Modelo y detector configurables con variables de entorno: `FR_MODELO` (p. ej. `ArcFace`) y `FR_DETECTOR` (p. ej. `retinaface`, más preciso pero más lento).

## 📂 Estructura

```
├── main.py                 # Punto de entrada (web y línea de comandos)
├── servidor.py             # API FastAPI
├── facekit/
│   ├── motor.py            # Detección, embeddings, identificación, atributos (DeepFace)
│   └── personas.py         # Base local de personas registradas
├── proyectos/
│   ├── asistencia.py       # Proyecto 1
│   ├── login.py            # Proyecto 2
│   └── hogar.py            # Proyecto 3
├── web/index.html          # Interfaz con cámara
├── requirements.txt
└── data/                   # Se crea al usarlo (ignorado por git)
    ├── personas.json       # Embeddings y perfiles
    ├── asistencia/AAAA-MM-DD.csv
    ├── notificaciones.log  # Avisos de ausencia (simulados)
    ├── accesos.log         # Bitácora del login
    ├── hogar.log           # Acciones de la casa
    └── visitantes/         # Recorte del rostro de desconocidos
```

## ⚠️ Limitaciones conocidas

- **Fotos de una foto**: el login no tiene detección de "vida" (*anti-spoofing*), así que una foto impresa de buena calidad podría engañarlo. DeepFace ofrece `anti_spoofing=True`, pero requiere instalar PyTorch (más de 700 MB). Lo dejé fuera para que la instalación sea ligera.
- Con iluminación pobre o rostros de perfil baja la tasa de reconocimiento; registra 3 fotos con luz frontal.
- Rendimiento: en CPU cada cuadro tarda entre 0.3 y 1 s, por lo que la app analiza un cuadro cada 1.2 s.
