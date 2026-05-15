# Car Parts Detector

Aplicación de detección y segmentación de piezas de coche usando YOLOv8. El proyecto incluye entrenamiento, validación básica del dataset, una API con FastAPI y una interfaz web para subir imágenes, ajustar el umbral de confianza y visualizar resultados.

## Funcionalidades

- Inferencia en dos etapas: detección previa del vehículo y segmentación de piezas sobre el recorte.
- Detección de piezas visibles de coche.
- Segmentación con máscaras.
- Visualización de la imagen anotada con cajas y máscaras.
- Tabla de resultados con clase, confianza, caja y máscara.
- Filtro por clase y ordenación por confianza.
- Umbral de confianza configurable desde la web.
- API REST con FastAPI.
- Ejecución local o mediante Docker.

## Estructura del proyecto

```text
.
├── config/
│   ├── general.yaml
│   ├── inference.yaml
│   └── train.yaml
├── models/
│   └── previous_best_run/
│       └── weights/best.pt
├── src/
│   ├── api_inference.py
│   ├── main.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── tests/
├── web/
│   ├── static/
│   │   ├── css/styles.css
│   │   └── js/main.js
│   └── templates/index.html
├── Dockerfile
├── requirements.txt
└── pytest.ini
```

## Requisitos

- Python 3.10
- Modelo entrenado en `models/previous_best_run/weights/best.pt`
- Dependencias de `requirements.txt`
- Cuenta de Weights & Biases si se quiere registrar entrenamiento y artefactos.

## Instalación local

Crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Para ejecutar el flujo de entrenamiento con tracking en Weights & Biases, instala también `wandb` si no está disponible en tu entorno:

```bash
pip install wandb
```

## Ejecutar la aplicación web

```bash
python -m src.api_inference --config inference.yaml
```

Abrir en el navegador:

```text
http://localhost:8000
```

Desde la interfaz web:

1. Selecciona una imagen de un coche.
2. Ajusta el campo `Confidence (%)`, por defecto `20`.
3. Pulsa `Predict`.
4. Revisa la imagen anotada y la tabla de detecciones.

## Configuración de inferencia

El archivo [config/inference.yaml](config/inference.yaml) define:

```yaml
model_path : "models"
best_model_path : "previous_best_run"
imgsz: 640
```

El modelo cargado será:

```text
models/previous_best_run/weights/best.pt
```

## Flujo de inferencia

La inferencia se realiza en dos pasos:

1. **Detección previa del vehículo**
   - Se usa un modelo YOLO estándar `yolov8n.pt`.
   - Solo se buscan clases de vehículo de COCO:
     - `2`: car
     - `5`: bus
     - `7`: truck
   - Si se detectan varios vehículos, se selecciona el de mayor área.
   - La imagen se recorta usando la caja del vehículo más grande.

2. **Segmentación de piezas**
   - Sobre el recorte anterior se ejecuta el modelo entrenado del proyecto:

```text
models/previous_best_run/weights/best.pt
```

Este enfoque evita ejecutar la segmentación de piezas sobre toda la imagen cuando primero se puede aislar el vehículo principal.

Si no se detecta ningún vehículo en la primera etapa, el segundo modelo no se ejecuta. En ese caso la API devuelve una respuesta sin detecciones ni imagen anotada.

## API

### `GET /`

Devuelve la interfaz web.

### `POST /predict`

Recibe una imagen y un umbral de confianza.

Parámetros `multipart/form-data`:

- `file`: imagen de entrada.
- `confidence`: umbral entre `0` y `1`. La web envía este valor convertido desde porcentaje.

Ejemplo con `curl`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/ruta/a/imagen.jpg" \
  -F "confidence=0.2"
```

Respuesta:

```json
{
  "filename": "imagen.jpg",
  "detections": [
    {
      "class_name": "door",
      "confidence": 0.91,
      "bbox": [10.2, 20.1, 140.5, 180.7],
      "mask": [
        { "x": 12.0, "y": 25.0 }
      ]
    }
  ],
  "message": "Prediction completed successfully in 1.34s",
  "fileout": "/outputs/pred_<id>_imagen.jpg"
}
```

Respuesta cuando no se detecta ningún vehículo:

```json
{
  "filename": "imagen.jpg",
  "detections": [],
  "message": "No vehicle detected. Segmentation model was not executed. Completed in 0.42s",
  "fileout": null
}
```

## Interpretación de resultados

- `class_name`: pieza detectada.
- `confidence`: confianza del modelo entre `0` y `1`.
- `bbox`: caja delimitadora en formato `[x1, y1, x2, y2]`.
- `mask`: puntos de la máscara de segmentación, si existen.
- `fileout`: URL de la imagen anotada generada por la API.

Si `fileout` es `null`, significa que no se detectó ningún vehículo en la primera etapa y por tanto no se ejecutó el modelo de segmentación de piezas.

En la tabla de la web, la confianza se colorea así:

- Verde: más de 75%.
- Amarillo: entre 50% y 75%.
- Naranja: entre 25% y 50%.
- Rojo: menos de 25%.

## Docker

Construir la imagen:

```bash
docker build --no-cache -t car-parts-detector .
```

Ejecutar el contenedor:

```bash
docker run --rm -p 8000:8000 car-parts-detector
```

Abrir:

```text
http://localhost:8000
```

## Entrenamiento

El entrenamiento se lanza desde:

```bash
python -m src.main
```

La configuración principal está en:

- [config/general.yaml](config/general.yaml): rutas, semilla y dataset.
- [config/train.yaml](config/train.yaml): modelo base, epochs, batch size, device y aumentos.

El dataset se descarga automáticamente con KaggleHub si no existe en `data/CarParts`.

## Tracking con Weights & Biases

El proyecto usa Weights & Biases para registrar experimentos, métricas y artefactos del entrenamiento.

En [src/main.py](src/main.py), el entrenamiento inicializa una run con:

- Proyecto: `MLOps`
- Nombre de experimento: valor de `training.experiment_name` en [config/train.yaml](config/train.yaml)
- Tipo de job: `training`
- Modo: `online`

Antes de entrenar, inicia sesión en W&B:

```bash
wandb login
```

Durante el flujo de entrenamiento se registran:

- Configuración combinada de `general.yaml` y `train.yaml`.
- Splits del dataset como artefacto `carparts-splits`.
- Métricas de `results.csv`.
- Imágenes generadas durante entrenamiento y validación.
- Mejor checkpoint como artefacto `best-yolo-model`.

Los artefactos se registran desde [src/tracking.py](src/tracking.py):

- `log_split_artifact`
- `log_results_artifact`
- `log_results_metrics`
- `log_training_images`
- `log_model_artifact`

También existe el script [src/register_previous_run.py](src/register_previous_run.py), pensado para registrar en W&B el mejor modelo ya entrenado en `models/previous_best_run`.

Ejecutarlo:

```bash
python -m src.register_previous_run
```

Ese script registra:

- El modelo `models/previous_best_run/weights/best.pt`.
- El fichero `models/previous_best_run/results.csv`.
- Métricas históricas del entrenamiento.
- Imágenes de entrenamiento y validación guardadas en la carpeta de la run previa.

## Tests

Ejecutar tests:

```bash
pytest
```

Los tests validan aspectos básicos del dataset, imágenes, máscaras y preprocesamiento.

## Notas

- Las imágenes generadas por inferencia se guardan temporalmente en `temp/`.
- Cada predicción genera un nombre de salida único para evitar problemas de caché del navegador y de concurrencia de peticiones.
- La API sirve las imágenes generadas mediante la ruta `/outputs`.
- La segmentación solo se ejecuta si el detector previo encuentra un coche, autobús o camión.

## Autora
- Nombre: Delia Arjona Castillo
- GitHub: [@Deliius](https://github.com/Deliius)
- Repositorio: [Deliius/car-parts-detector](https://github.com/Deliius/car-parts-detector)
