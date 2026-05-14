from typing import Any
from pathlib import Path

from ultralytics import YOLO
import logging

def train_model(
    base_path: Path,
    model_path: Path,
    config: dict[str, Any]
) -> None:
    """
    Entrena un modelo YOLOv8 de segmentación.

    El entrenamiento utiliza:
    - un modelo preentrenado YOLOv8
    - augmentations
    - early stopping mediante patience
    """

    # Cargamos la configuración
    training = config["training"]
    augmentation = config["augmentation"]
    model_config = config["model"]

    model: YOLO = YOLO(
        model_config
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Training")
    logger.info(f"Experiment name: {training['experiment_name']}")
    logger.info(f"Epochs: {training['epochs']}")
    logger.info(f"Batch size: {training['batch_size']}")

    try:
        # Lanzamos el entrenamiento.
        model.train(  # pyright: ignore[reportUnknownMemberType]

            # =================================================
            # Dataset
            # =================================================

            data=str(base_path / "car_parts.yaml"),

            # =================================================
            # Entrenamiento
            # =================================================

            epochs=training["epochs"],
            patience=training["patience"],
            imgsz=training["imgsz"],
            batch=training["batch_size"],

            workers=training["workers"],
            amp=training["amp"],
            device=training["device"],

            # =================================================
            # Output
            # =================================================

            project=str(model_path),
            name=training["experiment_name"],
            save=True,

            # =================================================
            # Augmentations avanzadas
            # =================================================

            mosaic=augmentation["mosaic"],
            close_mosaic=augmentation["close_mosaic"],

            mixup=augmentation["mixup"],
            copy_paste=augmentation["copy_paste"],

            # =================================================
            # Transformaciones geométricas
            # =================================================

            degrees=augmentation["degrees"],
            translate=augmentation["translate"],
            scale=augmentation["scale"],
            shear=augmentation["shear"],
            perspective=augmentation["perspective"],

            fliplr=augmentation["fliplr"],
            flipud=augmentation["flipud"],

            # =================================================
            # Transformaciones de color
            # =================================================

            hsv_h=augmentation["hsv_h"],
            hsv_s=augmentation["hsv_s"],
            hsv_v=augmentation["hsv_v"],

            erasing=augmentation["erasing"]
        )

        logger.info("Training completed successfully")

    except Exception as e:

        logger.error(f"Training failed: {e}")

        raise