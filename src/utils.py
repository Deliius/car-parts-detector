import os
import random
import shutil
from pathlib import Path
from typing import Any

import kagglehub  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import yaml
import logging


def get_project_folder() -> Path:
    '''Devuelve la ruta raíz del proyecto.'''
    return Path(__file__).resolve().parents[1]


def load_config(filename: str) -> dict[str, Any]:
    ''' Carga un fichero YAML de configuración y devuelve su contenido como diccionario. '''

    logger = logging.getLogger(__name__)
    logger.info("Loading configuration")

    # Carpeta raíz del proyecto.
    project_folder = get_project_folder()

    # Ruta completa del fichero de configuración.
    config_file = project_folder / "config" / filename

    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file}")
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    # Cargamos el contenido YAML.
    with open(config_file, "r") as fichero:
        return yaml.safe_load(fichero)


def set_seeds(SEED: int) -> None:
    ''' Fija semillas aleatorias para reproducibilidad. '''

    # Semilla para librería random.
    random.seed(SEED)

    # Semilla para numpy.
    np.random.seed(SEED)


def download_data(ruta: Path) -> None:
    ''' Descarga el dataset desde KaggleHub si no existe en la carpeta destino.'''
    
    logger = logging.getLogger(__name__)
    logger.info("Downloading data")

    # Solo descargamos el dataset si no existe previamente.
    if not os.path.exists(ruta):

        # Descarga del dataset desde KaggleHub.
        downloaded_path = kagglehub.dataset_download(
            "dinislamgaraev/car-parts-image-masking"
        )

        # Creamos la carpeta padre si no existe.
        os.makedirs(os.path.dirname(ruta), exist_ok=True)

        # Copiamos el dataset descargado a la ruta destino.
        shutil.copytree(downloaded_path, ruta)

        logger.info(f"Dataset copied to: {ruta}")

    else:
        logger.info("Dataset already on destination folder")