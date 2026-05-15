from collections.abc import Generator
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from pytest import MonkeyPatch

from src import api_inference
from src.api_inference import Detection



# Fakes

class FakeYOLO:
    '''
    Simula un modelo YOLO sin cargar pesos reales.

    Se utiliza para que el lifespan de FastAPI no intente cargar modelos
    durante los tests.
    '''

    names = {0: "door"}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def predict(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


class FakeSegmentationResult:
    '''
    Simula el resultado devuelto por el modelo de segmentación.

    Solo implementa plot(), que es el método usado por el endpoint.
    '''

    def plot(self) -> NDArray[np.uint8]:
        return np.zeros((32, 32, 3), dtype=np.uint8)


# Fixtures

@pytest.fixture
def client(monkeypatch: MonkeyPatch) -> Generator[TestClient, None, None]:
    '''
    Crea un cliente de test para la aplicación FastAPI.

    Reemplaza YOLO por una clase fake para evitar cargar modelos reales
    al arrancar la aplicación.
    '''

    monkeypatch.setattr(api_inference, "YOLO", FakeYOLO)
    with TestClient(api_inference.app) as client:
        yield client


def fake_run_prediction(app: FastAPI, path: Path, confidence: float) -> None:
    '''
    Simula una predicción sin vehículo detectado.
    '''

    return None


def fake_run_prediction_with_detections(
    app: FastAPI,
    path: Path,
    confidence: float,
) -> list[FakeSegmentationResult]:
    '''
    Simula una predicción con resultados de segmentación.
    '''

    return [FakeSegmentationResult()]


def build_fake_detection_response(app: FastAPI, results: Any) -> list[Detection]:
    '''
    Construye una detección fake con el mismo contrato de la API.
    '''

    return [
        Detection(
            class_name="door",
            confidence=0.93,
            bbox=[10.0, 20.0, 110.0, 120.0],
            mask=[],
        )
    ]


def fake_imwrite(path: str, image: NDArray[Any]) -> bool:
    '''
    Sustituye cv2.imwrite durante los tests.

    Crea un fichero pequeño para que /outputs pueda servirlo sin depender
    de OpenCV escribiendo una imagen real.
    '''

    Path(path).write_bytes(b"fake image")
    return True


@pytest.fixture
def no_vehicle_prediction(monkeypatch: MonkeyPatch) -> None:
    '''
    Fuerza la rama en la que no se detecta ningún vehículo.
    '''

    monkeypatch.setattr(api_inference, "run_prediction", fake_run_prediction)


@pytest.fixture
def detections_prediction(monkeypatch: MonkeyPatch) -> None:
    '''
    Fuerza la rama en la que sí hay detecciones.
    '''

    monkeypatch.setattr(api_inference, "run_prediction", fake_run_prediction_with_detections)
    monkeypatch.setattr(api_inference, "build_detection_response", build_fake_detection_response)
    monkeypatch.setattr(api_inference.cv2, "imwrite", fake_imwrite)


# Home Endpoint Tests


def test_home_endpoint_returns_html(client: TestClient) -> None:
    '''
    Comprueba que la página principal devuelve HTML.
    '''

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# Environment Parameters Tests

def test_load_inference_parameters_from_env(
    monkeypatch: MonkeyPatch,
) -> None:
    '''
    Comprueba que la configuración puede cargarse desde variables de entorno
    si no existe el fichero YAML.
    '''

    monkeypatch.setenv("MODEL_PATH", "models")
    monkeypatch.setenv("CROP_MODEL", "yolov8n.pt")
    monkeypatch.setenv("BEST_MODEL_PATH", "previous_best_run")
    monkeypatch.setenv("IMGSZ", "640")

    parameters = api_inference.load_inference_parameters("missing.yaml")

    assert parameters["model_path"] == "models"
    assert parameters["crop_model"] == "yolov8n.pt"
    assert parameters["best_model_path"] == "previous_best_run"
    assert parameters["imgsz"] == 640


def test_save_uploaded_file_sanitizes_filename() -> None:
    '''
    Comprueba que el nombre del fichero subido no conserva rutas peligrosas.
    '''

    fake_file = BytesIO(b"fake image data")

    upload_file = UploadFile(
        filename="../dangerous.jpg",
        file=fake_file,
    )

    saved_path = api_inference.save_uploaded_file(upload_file)

    assert saved_path.name.endswith("dangerous.jpg")
    assert ".." not in saved_path.name



# Prediction Endpoint Tests

def test_predict_returns_no_vehicle(
    client: TestClient,
    no_vehicle_prediction: None,
) -> None:
    '''
    Comprueba la respuesta cuando no se detecta ningún vehículo.
    '''

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", BytesIO(b"fake image data"), "image/jpeg")},
        data={"confidence": "0.2"},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["filename"] == "test.jpg"
    assert payload["detections"] == []
    assert payload["fileout"] is None
    assert "No vehicle detected" in payload["message"]


def test_predict_returns_detections(
    client: TestClient,
    detections_prediction: None,
) -> None:
    '''
    Comprueba la respuesta cuando la predicción devuelve detecciones.
    '''

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", BytesIO(b"fake image data"), "image/jpeg")},
        data={"confidence": "0.2"},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["filename"] == "test.jpg"
    assert payload["detections"]
    assert payload["fileout"] is not None
    assert "Prediction completed successfully" in payload["message"]

    detection = payload["detections"][0]
    assert detection["class_name"] == "door"
    assert detection["confidence"] == 0.93
    assert detection["bbox"] == [10.0, 20.0, 110.0, 120.0]
    assert detection["mask"] == []


def test_predict_rejects_invalid_confidence(client: TestClient) -> None:
    '''
    Comprueba que FastAPI rechaza confidence fuera del rango permitido.
    '''

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", BytesIO(b"fake image data"), "image/jpeg")},
        data={"confidence": "1.5"},
    )

    assert response.status_code == 422


def test_predict_requires_file(client: TestClient) -> None:
    '''
    Comprueba que el endpoint exige recibir una imagen.
    '''

    response = client.post(
        "/predict",
        data={"confidence": "0.2"},
    )

    assert response.status_code == 422


def test_prediction_output_file_is_accessible(
    client: TestClient,
    detections_prediction: None,
) -> None:
    '''
    Comprueba que la imagen generada puede descargarse desde fileout.
    '''

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", BytesIO(b"fake image data"), "image/jpeg")},
        data={"confidence": "0.2"},
    )

    payload = response.json()
    fileout = payload["fileout"]

    assert fileout is not None

    image_response = client.get(fileout)

    assert image_response.status_code == 200
