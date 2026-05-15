from pathlib import Path
from typing import Any
import logging

import pandas as pd
import wandb

logger = logging.getLogger(__name__)


def log_split_artifact(data_path: Path) -> None:
    '''
    Registra en W&B los ficheros que definen el split del dataset.
    '''

    artifact = wandb.Artifact(
        name="carparts-splits",
        type="dataset",
        description="Train/validation/test split files"
    )

    # Añadimos el YAML de YOLO y los ficheros de partición.
    artifact.add_file(str(data_path / "car_parts.yaml"))
    artifact.add_file(str(data_path / "train.txt"))
    artifact.add_file(str(data_path / "val.txt"))
    artifact.add_file(str(data_path / "test.txt"))

    # Registramos el artefacto en la run activa.
    wandb.log_artifact(artifact)


def log_results_artifact(results_path: Path) -> None:
    '''
    Registra results.csv como artefacto de W&B.
    '''

    # Si no existe results.csv, no se registra ningún artefacto.
    if not results_path.exists():
        logger.warning(f"results.csv does not exist at: {results_path}")
        return

    artifact = wandb.Artifact(
        name="training-results",
        type="results",
        description="YOLO training metrics per epoch"
    )

    # Añadimos el fichero completo como artefacto versionado.
    artifact.add_file(str(results_path))

    wandb.log_artifact(artifact)


def log_model_artifact(model_path: Path) -> None:
    '''
    Registra el mejor checkpoint del modelo como artefacto.
    '''

    # Si el checkpoint no existe, evitamos fallar el flujo completo.
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        return

    artifact = wandb.Artifact(
        name="best-yolo-model",
        type="model"
    )

    # Añadimos el fichero best.pt al artefacto del modelo.
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)


def log_training_images(run_path: Path) -> None:
    '''
    Registra las imágenes generadas durante entrenamiento y validación.
    '''

    # Recuperamos todas las imágenes JPG generadas por Ultralytics.
    image_files = list(run_path.glob("*.jpg"))

    for image_file in image_files:

        # Cada imagen se registra con su nombre como clave y caption.
        wandb.log({
            image_file.stem: wandb.Image(
                str(image_file),
                caption=image_file.name
            )
        })


def log_results_metrics(results_path: Path) -> None:
    '''
    Registra las métricas de results.csv como series temporales en W&B.
    '''

    # Si no existe el fichero de resultados, no se registran métricas.
    if not results_path.exists():
        logger.warning(f"Results file not found: {results_path}")
        return
    
    # Cargamos el CSV generado por Ultralytics.
    df = pd.read_csv(results_path)

    for _, row in df.iterrows():

        # Convertimos cada fila a un diccionario compatible con wandb.log.
        metrics: dict[str, Any] = {
            str(k): v
            for k, v in row.to_dict().items()
        }

        wandb.log(metrics)
