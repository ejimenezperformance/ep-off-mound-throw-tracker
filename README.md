# EP Off-Mound Throw Tracker

**Emerson Jimenez | Emerson Performance**

[English](#english) · [Español](#español)

---

## English

### The problem

Teams track every pitch thrown from the mound. A large part of a pitcher's real arm workload happens **off the mound**: catch play, long toss, flat-ground work and warm-up throws. Those throws are usually not counted, or they are counted by hand.

### What this prototype does

It takes a video and automatically counts the throws made by the pitcher. It returns:

- the number of throws and the exact time of each one,
- the throwing arm (right or left),
- an annotated video with the skeleton, a throw counter and a "THROW DETECTED" flag,
- an arm-speed chart with every detected throw marked.

### How it works

| Step | What happens | Tool |
|---|---|---|
| 1. See the body | Finds every person and 17 body points (shoulders, elbows, wrists, hips, knees, ankles) in each frame | YOLO11 Pose (Ultralytics) |
| 2. Follow each person | Gives each person an ID and follows them frame by frame, so the pitcher is not mixed with coaches or other players | ByteTrack |
| 3. Measure the arm | Wrist speed **relative to the hips**, divided by torso length. Running and crow hops do not count, and it works near or far from the camera | NumPy / SciPy |
| 4. Find the throws | A peak in arm speed above the threshold (0.35 torso lengths/frame) = one throw | SciPy `find_peaks` |
| 5. Handle real video | Camera cuts are detected and ignored; only one thrower is counted at a time (the strongest arm); throws must be at least 2.5 s apart | OpenCV |

The method is **rule-based**: every decision can be explained to a coach, and it does not need a large labeled dataset.

### Validation

Each video was counted by hand first, then compared with the system.

**Tuning set** (the rules were adjusted while looking at these videos):

| Video | Type | Difficulty | Manual | Detected | False positives | Missed |
|---|---|---|---|---|---|---|
| Yamamoto | Long toss, 1 throw | Moving camera, view from behind | 1 | 1 | 0 | 0 |
| Musgrove | Bullpen | Pitching coach larger than the pitcher in the frame | 3 | 3 | 0 | 0 |
| Yamamoto | Long toss session | Camera cuts, slow motion, near and far shots, coaches in background | 6 | 6 | 0 | 0 |

**Independent test** (new video, rules frozen, no changes made after seeing it):

| Video | Type | Difficulty | Manual | Detected | False positives | Missed |
|---|---|---|---|---|---|---|
| Front-view test | Mound throws | Catcher's view (new angle); pitcher turns his back after each throw, so tracking is lost and recovered 7 times | 7 | 7 | 0 | 0 |

Full detail: [`results/validation.csv`](results/validation.csv). Per-video output: [`results/`](results/).

**CONFIRMED:** 17 of 17 throws detected with no false positives across 4 videos; **7 of 7 on the independent test video**.

**INTERPRETATION / LIMITATIONS:**
- The rules were tuned on the first 3 videos. The first long-toss run produced 3 false positives (a coach gesturing in the background and a post-throw recovery), fixed with the "one thrower at a time" and 2.5 s gap rules. Only the front-view video is an independent test, and one video is a small sample.
- **Throw time:** on the front-view test, 2 of 7 throws were timed about 0.4–0.5 s late. When the pitcher turns his back after release, the arm produces a spike larger than the release itself, and the system keeps the largest spike. The count is correct; the exact time is not.
- **Throwing hand is not reliable.** The field `signal_arm` is the wrist with the strongest 2D movement, not the pitcher's throwing hand. From the catcher's view, the throwing arm moves mostly toward the camera (little movement on screen), while the glove arm moves sideways, so a right-handed pitcher was reported as `L`. Forcing the right wrist made it worse (5 of 7 throws). Handedness should come from the roster, and a fix needs depth information (3D pose or a second camera).
- All throws here are high-effort. Soft, short catch-play throws will produce smaller arm-speed peaks and are the main risk of missed throws.
- The 2.5 s minimum gap can merge two real throws if they are very close together.
- The videos are not stored in this repo (third-party footage). The code runs on any video you provide.

### Next steps

1. Test on new catch-play videos with many soft throws and report precision and recall.
2. Classify each throw (catch play, long toss, flat ground) and estimate effort level.
3. Daily workload log per pitcher, ready to add to mound pitch counts.

### Run it

```bash
pip install -r requirements.txt
python src/throw_detector.py my_video.mp4 output/
```

The YOLO pose weights (`yolo11m-pose.pt`) are downloaded automatically on the first run. Output: `throws.json`, `arm_speed.csv`, `arm_speed.png` and `annotated.mp4`. A CPU processes about 1–2 frames per second with the medium model; use `--model yolo11n-pose.pt` for a faster, less precise run.

---

## Español

### El problema

Los equipos cuentan cada pitcheo desde el montículo. Pero una gran parte de la carga real del brazo ocurre **fuera del montículo**: catch, long toss, tiros en terreno plano y calentamiento. Esos tiros casi nunca se cuentan, o se cuentan a mano.

### Qué hace este prototipo

Toma un video y cuenta automáticamente los tiros del lanzador. Entrega:

- el número de tiros y el momento exacto de cada uno,
- el brazo de lanzar (derecho o izquierdo),
- un video anotado con el esqueleto, el contador de tiros y el aviso "THROW DETECTED",
- una gráfica de velocidad del brazo con cada tiro marcado.

### Cómo funciona

| Paso | Qué pasa | Herramienta |
|---|---|---|
| 1. Ver el cuerpo | Encuentra a cada persona y 17 puntos del cuerpo en cada cuadro | YOLO11 Pose (Ultralytics) |
| 2. Seguir a cada persona | Le pone un número a cada persona y la sigue, para no mezclar al pitcher con coaches u otros jugadores | ByteTrack |
| 3. Medir el brazo | Velocidad de la muñeca **respecto a la cadera**, dividida por el largo del torso. La carrera y el crow hop no cuentan, y funciona cerca o lejos de la cámara | NumPy / SciPy |
| 4. Encontrar los tiros | Un pico de velocidad del brazo por encima del umbral (0.35) = un tiro | SciPy `find_peaks` |
| 5. Video real | Detecta e ignora los cortes de cámara; cuenta un solo lanzador a la vez (el del brazo más fuerte); mínimo 2.5 s entre tiros | OpenCV |

El método está **basado en reglas**: cada decisión se le puede explicar a un coach, y no necesita miles de videos etiquetados.

### Validación

Cada video se contó primero a mano y luego se comparó con el sistema.

**Videos de ajuste** (las reglas se ajustaron mirando estos videos):

| Video | Tipo | Dificultad | A mano | Sistema | Falsos | Perdidos |
|---|---|---|---|---|---|---|
| Yamamoto | Long toss, 1 tiro | Cámara en movimiento, vista de atrás | 1 | 1 | 0 | 0 |
| Musgrove | Bullpen | El coach se ve más grande que el pitcher | 3 | 3 | 0 | 0 |
| Yamamoto | Sesión de long toss | Cortes de cámara, cámara lenta, tomas cerca y lejos, coaches al fondo | 6 | 6 | 0 | 0 |

**Prueba independiente** (video nuevo, reglas congeladas, sin cambios después de verlo):

| Video | Tipo | Dificultad | A mano | Sistema | Falsos | Perdidos |
|---|---|---|---|---|---|---|
| Prueba de frente | Tiros desde el montículo | Vista del catcher (ángulo nuevo); el pitcher se voltea de espaldas después de cada tiro y el sistema lo pierde y lo recupera 7 veces | 7 | 7 | 0 | 0 |

**CONFIRMADO:** 17 de 17 tiros detectados, sin tiros falsos, en 4 videos; **7 de 7 en la prueba independiente**.

**INTERPRETACIÓN / LIMITACIONES:**
- Las reglas se ajustaron con los primeros 3 videos. La primera corrida del long toss dio 3 tiros falsos (un coach haciendo gestos al fondo y la recuperación después de un tiro), corregidos con las reglas de "un lanzador a la vez" y "mínimo 2.5 s entre tiros". Solo el video de frente es una prueba independiente, y un video es una muestra pequeña.
- **Hora del tiro:** en la prueba de frente, 2 de 7 tiros quedaron marcados unos 0.4–0.5 s tarde. Cuando el pitcher se voltea de espaldas, el brazo hace un pico más alto que el del tiro, y el sistema se queda con el pico más alto. El conteo es correcto; el momento exacto no.
- **El brazo de lanzar no es confiable.** El campo `signal_arm` es la muñeca con más movimiento en 2D, no la mano de lanzar. Desde la vista del catcher, el brazo de lanzar se mueve sobre todo hacia la cámara (poco movimiento en pantalla), mientras el brazo del guante se mueve hacia el lado; por eso un pitcher derecho salió como `L`. Forzar la muñeca derecha lo empeoró (5 de 7 tiros). La mano de lanzar debe venir del roster, y corregirlo necesita profundidad (pose en 3D o una segunda cámara).
- Todos los tiros de estos videos son fuertes. Los tiros suaves y cortos de catch dan picos más bajos y son el mayor riesgo de tiros perdidos.
- La regla de 2.5 s puede juntar dos tiros reales si están muy seguidos.
- Los videos no están en este repo (son de terceros). El código funciona con cualquier video.

### Próximos pasos

1. Probar con videos nuevos de catch con muchos tiros suaves y reportar precisión y recall.
2. Clasificar cada tiro (catch, long toss, terreno plano) y estimar el nivel de esfuerzo.
3. Registro diario de carga por lanzador, listo para sumar a los pitcheos del montículo.

### Cómo correrlo

```bash
pip install -r requirements.txt
python src/throw_detector.py mi_video.mp4 output/
```

---

*Emerson Jimenez · Emerson Performance · [ejimenezperformance.github.io](https://ejimenezperformance.github.io)*
