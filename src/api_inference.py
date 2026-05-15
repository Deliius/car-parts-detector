import logging
import shutil
from time import perf_counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote
from uuid import uuid4

import argparse
import cv2
import numpy as np
import os
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from PIL.Image import Image as PILImage
from pydantic import BaseModel, ConfigDict, Field
from ultralytics import YOLO

from src.logging_config import setup_logger
from src.utils import get_project_folder, load_config

setup_logger('DEBUG')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    default="inference.yaml",
    help="Configuration YAML file"
)

args = parser.parse_args()

# Rutas base del proyecto
PROJECT_ROOT = get_project_folder()
TEMP_FOLDER = PROJECT_ROOT / "temp"
TEMP_FOLDER.mkdir(exist_ok=True)


def load_inference_parameters(config_filename: str) -> dict[str, Any]:
    # En local se utiliza el YAML; en despliegues se permite configurar
    # la inferencia con variables de entorno si el fichero no existe.
    config_file = PROJECT_ROOT / "config" / config_filename

    if config_file.exists():
        return load_config(config_filename)

    logger.warning(
        f"Configuration file not found: {config_file}. "
        "Loading inference configuration from environment variables."
    )

    return {
        "model_path": os.getenv("MODEL_PATH", "models"),
        "crop_model_path": os.getenv("CROP_MODEL_PATH", "yolov8n.pt"),
        "best_model_path": os.getenv("BEST_MODEL_PATH", "previous_best_run"),
        "imgsz": int(os.getenv("IMGSZ", "640")),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(f"Loading configuration: {args.config}")

    parameters = load_inference_parameters(args.config)
    app.state.parameters = parameters

    # Rutas de los dos modelos usados durante inferencia:
    # - modelo entrenado del proyecto para segmentar piezas
    # - modelo previo para detectar y recortar el vehículo principal
    model_path = Path(parameters["model_path"])
    best_model_path = Path(parameters["best_model_path"])
    crop_model_path = Path(parameters["crop_model_path"])

    MODEL_PATH = (
        PROJECT_ROOT
        / model_path
        / best_model_path
        / "weights"
        / "best.pt"
    )

    CROP_MODEL_PATH = (
        PROJECT_ROOT
        / model_path
        / crop_model_path
    )


    try:
        # Los modelos se cargan una sola vez al arrancar la API.
        # Así evitamos inicializarlos de nuevo en cada petición /predict.
        logger.info(f"Loading YOLO model from: {MODEL_PATH}")
        app.state.model = YOLO(MODEL_PATH)
        logger.info(f"Loading vehicle detector model: {crop_model_path}")
        app.state.vehicle_model = YOLO(CROP_MODEL_PATH)
        logger.info("YOLO model loaded successfully")

        yield

    except Exception as e:

        logger.error(f"Failed to load YOLO model: {e}")
        raise

    finally:

        logger.info("Shutting down API")

app = FastAPI(
    title="Car Parts Detection API",
    version="1.0.0",
    lifespan=lifespan
)

templates = Jinja2Templates(
    directory=str(PROJECT_ROOT / "web" / "templates")
)

# Archivos del frontend: CSS y JavaScript.
app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "web" / "static")),
    name="static"
)

# Imágenes generadas 
app.mount(
    "/outputs",
    StaticFiles(directory=str(TEMP_FOLDER)),
    name="outputs"
)

class MaskPoint(BaseModel):
    # Punto individual de una máscara de segmentación.
    x: float
    y: float


class Detection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Detección: clase, confianza, caja y máscara (opcional).
    class_name: str
    confidence: float
    bbox: list[float]
    mask: list[MaskPoint] | None = None

class PredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # fileout None cuando no se detecta vehículo.
    filename: str
    detections: list[Detection]
    message: str
    fileout: str | None = Field(serialization_alias="fileout")


def save_uploaded_file(file: UploadFile) -> Path:
    # Path(...).name evita que el nombre recibido contenga rutas del cliente.
    # uuid4 evita colisiones si se sube la misma imagen varias veces.
    filename = Path(str(file.filename)).name
    input_path = TEMP_FOLDER / f"upload_{uuid4().hex}_{filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return input_path

def run_prediction(app: FastAPI, input_path: Path, confidence: float) -> Any | None:
    parameters = app.state.parameters

    detector_model = app.state.vehicle_model
    segmentation_model = app.state.model

    # Primera etapa: detectar vehículos en la imagen original.
    # Clases de COCO: 2 (car), 5 (bus), 7 (truck).
    results_vehiculo = detector_model.predict(
        source=str(input_path),
        classes=[2, 5, 7],
        conf=0.5,
        device='cpu'
    )

    if len(results_vehiculo[0].boxes) == 0:
        # Si no hay vehículo, no se ejecuta el modelo de segmentación.
        logger.warning("No vehicle detected. Segmentation model was not executed.")
        return None

    # Si hay varios vehículos, se recorta el de mayor área.
    boxes = cast(np.ndarray[Any, Any], results_vehiculo[0].boxes.xyxy.cpu().numpy())
    areas = cast(list[float], ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).tolist())

    idx_max_area = max(range(len(areas)), key=areas.__getitem__)
    box = boxes[idx_max_area].astype(int)

    # Las coordenadas de YOLO se ajustan a los límites reales de la imagen.
    img_orig = cast(PILImage, Image.open(input_path))
    width, height = img_orig.size
    x1 = max(0, min(int(box[0]), width))
    y1 = max(0, min(int(box[1]), height))
    x2 = max(0, min(int(box[2]), width))
    y2 = max(0, min(int(box[3]), height))

    if x2 <= x1 or y2 <= y1:
        # Evita enviar un recorte vacío o inválido al segundo modelo.
        logger.warning("Vehicle crop is invalid. Segmentation model was not executed.")
        return None

    img_crop = img_orig.crop((x1, y1, x2, y2))

    logger.info("Cropping the largest detected vehicle.")

    # Segunda etapa: segmentar piezas sobre el recorte del vehículo.
    return segmentation_model.predict(
        source=img_crop,
        imgsz=parameters["imgsz"],
        conf=confidence,
        device="cpu",
    )
    



def build_mask_response(results: Any) -> list[list[MaskPoint]]:
    # Convierte las máscaras poligonales de Ultralytics a objetos JSON simples.
    result = results[0]
    result_masks = getattr(result, "masks", None)

    if result_masks is None:
        return []

    raw_masks = cast(list[Any], result_masks.xy)
    masks: list[list[MaskPoint]] = []

    for raw_mask in raw_masks:
        raw_points = cast(list[list[float]], raw_mask.tolist())
        masks.append([
            MaskPoint(x=float(point[0]), y=float(point[1]))
            for point in raw_points
        ])

    return masks


def build_detection_response(app: FastAPI, results: Any) -> list[Detection]:
    # Une cajas, clases, confianza y máscara en el formato de respuesta.

    detections: list[Detection] = []
    boxes = results[0].boxes
    masks = build_mask_response(results)

    for index, box in enumerate(boxes):
        class_id = int(box.cls[0])
        class_name = app.state.model.names[class_id]
        confidence = float(box.conf[0])
        bbox = [float(x) for x in box.xyxy[0].tolist()]
        mask = None

        if index < len(masks):
            mask = masks[index]

        detections.append(
            Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                mask=mask,
            )
        )

    return detections

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    confidence: float = Form(0.2, ge=0.0, le=1.0),
) -> PredictionResponse:

    logger.info(f"Received file for JSON prediction: {file.filename}")

    try:

        # Se mide el tiempo completo de la petición de inferencia.
        filename = Path(str(file.filename)).name
        start_time = perf_counter()
        input_path = save_uploaded_file(file)
        results = run_prediction(app, input_path, confidence)
        elapsed_time = perf_counter() - start_time

        if results is None:
            return PredictionResponse(
                filename=filename,
                detections=[],
                message=f"No vehicle detected. Segmentation model was not executed. Completed in {elapsed_time:.2f}s",
                fileout=None,
            )

        detections = build_detection_response(app, results)
        plotted = results[0].plot()
        # Nombre único para evitar caché del navegador y colisiones.
        output_path = TEMP_FOLDER / f"pred_{uuid4().hex}_{filename}"
        cv2.imwrite(str(output_path), plotted)
        elapsed_time = perf_counter() - start_time
        logger.info(f"Prediction completed in {elapsed_time:.2f}s. Detections: {len(detections)}")
        logger.info(f"Prediction image saved at: {output_path}")

        return PredictionResponse(
            filename=filename,
            detections=detections,
            message=f"Prediction completed successfully in {elapsed_time:.2f}s",
            fileout=f"/outputs/{quote(output_path.name)}"
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":

    uvicorn.run(
        "src.api_inference:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
