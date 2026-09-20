# dual-view-tracker

English version: [README.md](README.md)

Seguimiento de personas en 3D con dos webcams comunes apuntando a la misma habitación desde ángulos distintos.

El plan: detectar puntos clave del cuerpo en cada cámara, emparejar las detecciones entre vistas usando la geometría de las cámaras y triangularlas a posiciones 3D que se puedan seguir en el tiempo.

## Estado actual

Captura de dos cámaras con marcas de tiempo, emparejamiento de cuadros y una prueba de sincronización con flash, manual o automatizada con la linterna LED de un celular. Todavía no hay calibración ni estimación de pose.

## Requisitos

- Windows 10/11 (las cámaras se leen por Media Foundation o DirectShow)
- Python 3.12 o más nuevo
- Dos webcams USB, o una webcam más un celular que sirva un stream de video

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Uso

Averiguá qué soportan de verdad tus cámaras. La prueba recorre los índices de dispositivo 0 a 5 en ambos backends, a 640x480, 1280x720 y 1920x1080, e informa la resolución real, la tasa de cuadros y los controles de foco y exposición:

```powershell
python scripts/probe_cameras.py
```

Abrí las dos cámaras y mostralas lado a lado:

```powershell
python -m dual_view_tracker.view --cam0 0 --cam1 1 --res0 1280x720 --res1 1920x1080
```

Esos valores son los predeterminados. Otras opciones: `--fps` (30), `--backend msmf|dshow` (msmf), `--max-wait` segundos antes de emparejar un cuadro sin pareja confirmada (0.1), `--display-height` píxeles (480), `--min-jump` la subida de brillo más chica, en escala 0-255, que cuenta como el flash (20).

`--cam0` y `--cam1` aceptan un índice de dispositivo o la URL de un stream de video. Un celular con una app de cámara que sirva un stream de video por la red (DroidCam, por ejemplo) puede ser la segunda cámara:

```powershell
python -m dual_view_tracker.view --cam0 0 --cam1 http://192.168.0.20:4747/video
```

Para un stream, el celular decide la resolución y la tasa de cuadros, así que `--res1` se ignora para él.

Al arrancar, el visor imprime la configuración pedida y la real de cada cámara. La superposición muestra la resolución real y la tasa de cuadros medida de cada cámara, y la diferencia de tiempo entre los dos cuadros emparejados en ese momento, marcada como `settled` o `UNSETTLED`. Apretá `q` o Esc para salir.

### Prueba de sincronización

Las webcams no se pueden sincronizar por hardware, así que el visor mide cuánto se desfasan las dos cámaras en el tiempo. Apuntá el flash de un celular de modo que las dos cámaras lo vean, apretá `f` y disparalo dentro de los tres segundos. El visor encuentra el cuadro donde salta el brillo en cada cámara e informa la diferencia (cámara 1 menos cámara 0, así que un valor positivo significa que la cámara 1 vio el flash más tarde). Si la subida es menor que `--min-jump` en alguna de las cámaras, la prueba se descarta; cada prueba imprime la mayor subida vista por cámara, así se puede diagnosticar un flash que no se detectó. Repetilo varias veces; la media y la dispersión se muestran en pantalla, se imprimen al salir y se escriben en `output/flash_offsets.csv` si al menos una prueba salió bien.

#### Lote automatizado

Si la cámara del celular corre DroidCam, el visor puede manejar la linterna LED del celular a través de la API de control remoto de DroidCam y correr las pruebas solo. Pasá `--remote` con la dirección de control remoto del celular y apretá `t`:

```powershell
python -m dual_view_tracker.view --cam0 0 --cam1 http://192.168.0.20:4747/video --remote http://192.168.0.20:4747
```

Un lote corre `--trials` destellos (20 por defecto). Se prende la linterna, se bloquea la exposición del celular mientras está iluminado para que mantenga su tasa de cuadros completa en una habitación oscura, y después la linterna se pulsa una vez por prueba mientras las dos cámaras registran el inicio de la subida de brillo. El desfase es el inicio en la cámara 1 menos el inicio en la cámara 0, igual que en la prueba manual. Cada prueba imprime su desfase, la tasa de cuadros y la mayor subida por cámara, y los cuadros perdidos; una prueba en la que alguna cámara bajó de 27 FPS se marca `[slow]` y queda afuera de las estadísticas del resumen, pero se conserva en el CSV. Al salir se imprime el resumen y todas las pruebas se escriben en `output/torch_offsets.csv`.

Notas prácticas:

- Las dos cámaras tienen que ver la superficie que ilumina la linterna, desde menos de un metro más o menos.
- Mantené la habitación oscura y quedate quieto durante el lote.
- La API remota de DroidCam solo ofrece toggles y no informa si la linterna o el bloqueo de exposición están activos, así que el visor lleva ese estado por su cuenta. Arrancá con la linterna apagada y la exposición sin bloquear; el lote deja el celular así cuando termina.
- Cerrá el cliente de DroidCam para PC: el celular acepta una sola conexión.

## Cómo funciona

- `capture.py` lee cada cámara en su propio hilo y guarda el último medio segundo de cuadros, cada uno marcado con su hora de llegada. Una URL de stream se lee a través de FFmpeg.
- `sync.py` empareja el cuadro más nuevo de la cámara de referencia con el cuadro más cercano de la otra cámara. Espera hasta que la otra cámara tenga un cuadro igual o posterior al de referencia, para que ningún cuadro posterior pueda ser mejor pareja. Esto cuesta como máximo un período de cuadro de latencia. Si la otra cámara se traba por más de `--max-wait`, el par se arma igual y se marca como no confirmado.
- `flash.py` encuentra el inicio del flash como la mayor subida de brillo entre cuadros consecutivos.
- `trials.py` corre la prueba manual y el lote de linterna como máquinas de pasos manejadas por el bucle del visor; `remote.py` manda los toggles de linterna y bloqueo de exposición a DroidCam.

## Pruebas

```powershell
pytest
```

Las pruebas cubren el emparejamiento, la detección del flash, el búfer de captura, la lógica de las pruebas y el lote de linterna, todo con cámaras y un celular simulados.

## Licencia

MIT
