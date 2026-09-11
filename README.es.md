# dual-view-tracker

English version: [README.md](README.md)

Seguimiento de personas en 3D con dos webcams comunes apuntando a la misma habitación desde ángulos distintos.

El plan: detectar puntos clave del cuerpo en cada cámara, emparejar las detecciones entre vistas usando la geometría de las cámaras y triangularlas a posiciones 3D que se puedan seguir en el tiempo.

## Estado actual

Captura de dos cámaras con marcas de tiempo, emparejamiento de cuadros y una prueba de sincronización con flash. Todavía no hay calibración ni estimación de pose.

## Requisitos

- Windows 10/11 (las cámaras se leen por Media Foundation o DirectShow)
- Python 3.12 o más nuevo
- Dos webcams USB

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

Esos valores son los predeterminados. Otras opciones: `--fps` (30), `--backend msmf|dshow` (msmf), `--max-wait` segundos antes de emparejar un cuadro sin pareja confirmada (0.1), `--display-height` píxeles (480).

Al arrancar, el visor imprime la configuración pedida y la real de cada cámara. La superposición muestra la resolución real y la tasa de cuadros medida de cada cámara, y la diferencia de tiempo entre los dos cuadros emparejados en ese momento, marcada como `settled` o `UNSETTLED`. Apretá `q` o Esc para salir.

### Prueba de sincronización

Las webcams no se pueden sincronizar por hardware, así que el visor mide cuánto se desfasan las dos cámaras en el tiempo. Apuntá el flash de un celular de modo que las dos cámaras lo vean, apretá `f` y disparalo dentro de los tres segundos. El visor encuentra el cuadro donde salta el brillo en cada cámara e informa la diferencia (cámara 1 menos cámara 0, así que un valor positivo significa que la cámara 1 vio el flash más tarde). Si el salto es demasiado chico en alguna de las cámaras, la prueba se descarta. Repetilo varias veces; la media y la dispersión se muestran en pantalla, se imprimen al salir y se escriben en `output/flash_offsets.csv` si al menos una prueba salió bien.

## Cómo funciona

- `capture.py` lee cada cámara en su propio hilo y guarda el último medio segundo de cuadros, cada uno marcado con su hora de llegada.
- `sync.py` empareja el cuadro más nuevo de la cámara de referencia con el cuadro más cercano de la otra cámara. Espera hasta que la otra cámara tenga un cuadro igual o posterior al de referencia, para que ningún cuadro posterior pueda ser mejor pareja. Esto cuesta como máximo un período de cuadro de latencia. Si la otra cámara se traba por más de `--max-wait`, el par se arma igual y se marca como no confirmado.
- `flash.py` encuentra el inicio del flash como la mayor subida de brillo entre cuadros consecutivos.

## Pruebas

```powershell
pytest
```

Las pruebas cubren el emparejamiento, la detección del flash, el búfer de captura y la lógica de prueba del visor, todo con cámaras simuladas.

## Licencia

MIT
