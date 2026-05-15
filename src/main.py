import pandas as pd
from pathlib import Path

import wandb
import logging

from src.annotations import YoloAnnotationConverter
from src.logging_config import setup_logger
from src.preprocessing import CarPartsDatasetAnalyzer, generate_yolo_yaml, split_train_val_test
from src.tracking import (
    log_model_artifact,
    log_results_artifact,
    log_results_metrics,
    log_split_artifact,
    log_training_images,
)
from src.train import train_model
from src.utils import download_data, load_config, set_seeds


def main() -> None:
    '''Función principal que lanza la ejecución de carga datos, preprocesado, split y entrenamiento.'''

    # Configuración del logger
    setup_logger("DEBUG")
    logger = logging.getLogger(__name__)

    logger.info("Starting execution")

    # Configuración general del proyecto
    parameters = load_config("general.yaml")

    # Configuración específica del entrenamiento
    train_config = load_config("train.yaml")

    # Configuración de W&B: parametros generales + entrenamiento
    wandb_config = {
        **parameters,
        **train_config,
    }

    # Inicialización de W&B
    wandb.init(
        project="MLOps",
        name=train_config["training"]["experiment_name"],
        config=wandb_config,
        job_type="training",
        mode="online",
    )

    # Rutas principales
    data_path = Path(parameters["data_path"])
    images_path = Path(parameters["images_path"])
    masks_path = Path(parameters["masks_path"])
    labels_path = Path(parameters["labels_path"])
    classes_path = Path(parameters["classes_path"])
    model_path = Path(parameters["model_path"])
    run_path = (
        model_path
        / train_config["training"]["experiment_name"]
    )

    # Fijar semillas
    set_seeds(int(parameters["seed"]))

    # Descargar datos si no existen
    download_data(data_path)

    # Cargar clases
    classes = pd.read_csv(classes_path)

    # Analizar dataset
    analyzer = CarPartsDatasetAnalyzer(
        classes=classes,
        dataset_path=data_path,
        masks_path=masks_path,
        cache_file=str(parameters["cache_file"]),
    )

    df_dataset = analyzer.load_or_analyze()

    # Generar anotaciones YOLO
    converter = YoloAnnotationConverter(
        class_to_id=analyzer.class_to_id,
        masks_path=masks_path,
    )

    converter.save_annotations(
        df=df_dataset,
        output_path=labels_path,
        min_area=20,
    )

    # Crear train/val/test
    split_train_val_test(
        data_path=str(data_path),
        images_path=str(images_path),
    )

    # Generar YAML de YOLO
    generate_yolo_yaml(
        path=str(data_path),
        file_name=str(parameters["yaml_filename"]),
    )

    # Registrar splits en W&B
    log_split_artifact(data_path)

    # Entrenamiento del modelo
    train_model(
        base_path=data_path,
        model_path=model_path,
        config=train_config,
    )

    # Registrar results.csv en W&B
    results_path = run_path / "results.csv"
    best_model_path = run_path / "weights" / "best.pt"

    log_results_artifact(results_path)
    log_results_metrics(results_path)
    log_training_images(run_path)
    log_model_artifact(best_model_path) 

    wandb.finish()

    logger.info("=======================")


if __name__ == "__main__":
    main()