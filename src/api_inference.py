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

#Paths 
PROJECT_ROOT = get_project_folder()
TEMP_FOLDER = PROJECT_ROOT / "temp"
TEMP_FOLDER.mkdir(exist_ok=True)


def load_inference_parameters(config_filename: str) -> dict[str, Any]:
    config_file = PROJECT_ROOT / "config" / config_filename

    if config_file.exists():
        return load_config(config_filename)

    logger.warning(
        f"Configuration file not found: {config_file}. "
        "Loading inference configuration from environment variables."
    )

    return {
        "model_path": os.getenv("MODEL_PATH", "models"),
        "best_model_path": os.getenv("BEST_MODEL_PATH", "previous_best_run"),
        "imgsz": int(os.getenv("IMGSZ", "640")),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(f"Loading configuration: {args.config}")

    parameters = load_inference_parameters(args.config)
    app.state.parameters = parameters

    # Ruta al modelo
    model_path = Path(parameters["model_path"])
    best_model_path = Path(parameters["best_model_path"])
    MODEL_PATH = (
        PROJECT_ROOT
        / model_path
        / best_model_path
        / "weights"
        / "best.pt"
    )

    try:
        logger.info(f"Loading YOLO model from: {MODEL_PATH}")
        app.state.model = YOLO(MODEL_PATH)
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

app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "web" / "static")),
    name="static"
)

app.mount(
    "/outputs",
    StaticFiles(directory=str(TEMP_FOLDER)),
    name="outputs"
)

class MaskPoint(BaseModel):
    x: float
    y: float


class Detection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str
    confidence: float
    bbox: list[float]
    mask: list[MaskPoint] | None = None

class PredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    filename: str
    detections: list[Detection]
    message: str
    fileout: str | None = Field(serialization_alias="fileout")


def save_uploaded_file(file: UploadFile) -> Path:
    input_path = TEMP_FOLDER / str(file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return input_path

def run_prediction(app: FastAPI, input_path: Path, confidence: float) -> Any | None:
    parameters = app.state.parameters

    # Cargar un modelo YOLO estándar
    model_estandar = YOLO('yolov8n.pt')
    detector_model = cast(Any, model_estandar)
    segmentation_model = app.state.model

    # Detección en la imagen original
    # Clases de vehículos de COCO: 2 (car), 5 (bus), 7 (truck)
    results_vehiculo = detector_model.predict(
        source=str(input_path),
        classes=[2, 5, 7],
        conf=0.5,
        device='cpu'
    )

    if len(results_vehiculo[0].boxes) == 0:
        logger.warning("No se detectó ningún vehículo. No se ejecuta el modelo de segmentación.")
        return None

    # Calcular el área de todas las cajas detectadas
    boxes = cast(np.ndarray[Any, Any], results_vehiculo[0].boxes.xyxy.cpu().numpy())
    areas = cast(list[float], ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).tolist())

    #Obtener área más grande
    idx_max_area = max(range(len(areas)), key=areas.__getitem__)
    box = boxes[idx_max_area].astype(int)

    #Recorte
    img_orig = cast(PILImage, Image.open(input_path))
    img_crop = img_orig.crop((box[0], box[1], box[2], box[3]))

    logger.info("Recortando el vehículo más grande.")

    return segmentation_model.predict(
        source=img_crop,
        imgsz=parameters["imgsz"],
        conf=confidence,
        device="cpu",
    )
    



def build_mask_response(results: Any) -> list[list[MaskPoint]]:
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
    confidence: float = Form(0.2),
) -> PredictionResponse:

    logger.info(f"Received file for JSON prediction: {file.filename}")

    try:

        start_time = perf_counter()
        input_path = save_uploaded_file(file)
        results = run_prediction(app, input_path, confidence)
        elapsed_time = perf_counter() - start_time

        if results is None:
            return PredictionResponse(
                filename=str(file.filename),
                detections=[],
                message=f"No vehicle detected. Segmentation model was not executed. Completed in {elapsed_time:.2f}s",
                fileout=None,
            )

        detections = build_detection_response(app, results)
        plotted = results[0].plot()
        output_path = TEMP_FOLDER / f"pred_{uuid4().hex}_{file.filename}"
        cv2.imwrite(str(output_path), plotted)
        logger.info(f"Prediction completed in {elapsed_time:.2f}s. Detections: {len(detections)}")
        logger.info(f"Prediction image saved at: {output_path}")

        return PredictionResponse(
            filename=str(file.filename),
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
