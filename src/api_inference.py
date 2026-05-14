import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import argparse
import cv2
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
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


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(f"Loading configuration: {args.config}")

    parameters = load_config(args.config)
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

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: list[float]

class PredictionResponse(BaseModel):
    filename: str
    detections: list[Detection]
    message: str
    fileout: str


def save_uploaded_file(file: UploadFile) -> Path:
    input_path = TEMP_FOLDER / str(file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return input_path

def run_prediction(app: FastAPI, input_path: Path) -> Any:
    parameters = app.state.parameters
    
    return app.state.model.predict(
        source=str(input_path),
        imgsz=parameters["imgsz"],
        conf=parameters["conf"],
        device="cpu",
    )


def build_detection_response(app: FastAPI, results: Any) -> list[Detection]:

    detections: list[Detection] = []
    boxes = results[0].boxes

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = app.state.model.names[class_id]
        confidence = float(box.conf[0])
        bbox = [float(x) for x in box.xyxy[0].tolist()]

        detections.append(
            Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
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
async def predict(file: UploadFile = File(...)) -> PredictionResponse:

    logger.info(f"Received file for JSON prediction: {file.filename}")

    try:

        input_path = save_uploaded_file(file)
        results = run_prediction(app, input_path)
        detections = build_detection_response(app, results)
        logger.info(f"Prediction completed. Detections: {len(detections)}")
        plotted = results[0].plot()
        output_path = TEMP_FOLDER / f"pred_{file.filename}"
        cv2.imwrite(str(output_path), plotted)
        logger.info(f"Prediction image saved at: {output_path}")

        return PredictionResponse(
            filename=str(file.filename),
            detections=detections,
            message="Prediction completed successfully",
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
