from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from src.preprocessing import CarPartsDatasetAnalyzer
from src.utils import download_data, load_config


# =========================================================
# Fixtures
# =========================================================

# Carga la configuración general del proyecto desde YAML.
# Se ejecuta una sola vez por sesión de tests.
@pytest.fixture(scope="session")
def parameters() -> dict[str, Any]:
    return load_config("general.yaml")


# Devuelve la ruta principal del dataset.
# Además, descarga el dataset si todavía no existe.
@pytest.fixture(scope="session")
def data_path(parameters: dict[str, Any]) -> Path:
    path = Path(parameters["data_path"])
    download_data(path)
    return path


# Devuelve la ruta del fichero classes.csv
@pytest.fixture(scope="session")
def classes_path(parameters: dict[str, Any]) -> Path:
    return Path(parameters["classes_path"])


# Devuelve la ruta de la carpeta de imágenes
@pytest.fixture(scope="session")
def images_path(parameters: dict[str, Any]) -> Path:
    return Path(parameters["images_path"])


# Devuelve la ruta de la carpeta de máscaras
@pytest.fixture(scope="session")
def masks_path(parameters: dict[str, Any]) -> Path:
    return Path(parameters["masks_path"])


# Devuelve el nombre del fichero de cache generado
# durante el análisis del dataset.
@pytest.fixture(scope="session")
def cache_file(parameters: dict[str, Any]) -> str:
    return str(parameters["cache_file"])


# =========================================================
# Dataset Validation Tests
# =========================================================

# Comprueba que el dataset existe y no está vacío.
def test_dataset_not_empty(data_path: Path) -> None:
    assert data_path.exists()
    assert any(data_path.iterdir())


# Comprueba que existe el fichero classes.csv
def test_dataset_has_classes(classes_path: Path) -> None:
    assert classes_path.exists()
    assert classes_path.is_file()


# Comprueba que classes.csv contiene datos.
def test_classes_not_empty(classes_path: Path) -> None:
    df: DataFrame = pd.read_csv(classes_path)
    assert len(df) > 0


# Comprueba que classes.csv contiene las columnas esperadas.
def test_classes_has_expected_columns(classes_path: Path) -> None:
    df: DataFrame = pd.read_csv(classes_path)

    expected_columns = {"class", "color"}

    assert expected_columns.issubset(df.columns)


# Comprueba que las columnas críticas no contienen valores nulos.
def test_classes_has_no_nulls(classes_path: Path) -> None:
    df: DataFrame = pd.read_csv(classes_path)

    assert not df[["class", "color"]].isnull().any().any()


# Comprueba que existe la carpeta de imágenes
# y contiene imágenes válidas.
def test_dataset_has_images(images_path: Path) -> None:
    assert images_path.exists()
    assert images_path.is_dir()

    images = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))

    assert len(images) > 0


# Comprueba que existe la carpeta de máscaras
# y contiene máscaras válidas.
def test_dataset_has_masks(masks_path: Path) -> None:
    assert masks_path.exists()
    assert masks_path.is_dir()

    masks = list(masks_path.glob("*.png"))

    assert len(masks) > 0


# Comprueba que el número de imágenes coincide
# con el número de máscaras.
def test_dataset_number_files(
    images_path: Path,
    masks_path: Path
) -> None:
    images = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    masks = list(masks_path.glob("*.png"))

    assert len(images) == len(masks)


# Comprueba que cada imagen tiene una máscara asociada
# con el mismo nombre base.
def test_images_and_masks_have_matching_names(
    images_path: Path,
    masks_path: Path
) -> None:

    image_names = {
        p.stem
        for p in list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    }

    mask_names = {
        p.stem
        for p in masks_path.glob("*.png")
    }

    assert image_names == mask_names


# =========================================================
# Shape & Data Type Tests
# =========================================================

# Comprueba que una imagen y su máscara correspondiente
# tienen las mismas dimensiones espaciales.
def test_image_and_mask_have_same_shape(
    images_path: Path,
    masks_path: Path
) -> None:

    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    assert len(image_files) > 0

    image_file = image_files[0]
    mask_file = masks_path / f"{image_file.stem}.png"

    image = cv2.imread(str(image_file))
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

    assert image is not None
    assert mask is not None
    assert image.shape[:2] == mask.shape[:2]


# Comprueba que imágenes y máscaras utilizan uint8,
# formato estándar esperado por OpenCV y YOLO.
def test_image_and_mask_dtype(
    images_path: Path,
    masks_path: Path
) -> None:

    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    assert len(image_files) > 0

    image_file = image_files[0]
    mask_file = masks_path / f"{image_file.stem}.png"

    image = cv2.imread(str(image_file))
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

    assert image is not None
    assert mask is not None
    assert image.dtype == np.uint8
    assert mask.dtype == np.uint8


# =========================================================
# Preprocessing Tests
# =========================================================

# Comprueba que el analizador:
# - genera correctamente el DataFrame
# - crea el fichero de cache
# - contiene las columnas esperadas
def test_analyzer_creates_dataframe(
    classes_path: Path,
    data_path: Path,
    masks_path: Path,
    cache_file: str
) -> None:

    classes = pd.read_csv(classes_path)

    analyzer = CarPartsDatasetAnalyzer(
        classes=classes,
        dataset_path=data_path,
        masks_path=masks_path,
        cache_file=cache_file
    )

    df = analyzer.load_or_analyze()

    assert (data_path / cache_file).exists()

    assert len(df) > 0

    assert "image_name" in df.columns
    assert "mask_name" in df.columns