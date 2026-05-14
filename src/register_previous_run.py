import logging
from pathlib import Path
from typing import Any

import wandb

from src.logging_config import setup_logger
from src.utils import load_config
from src.tracking import (
    log_model_artifact,
    log_results_artifact,
    log_results_metrics,
    log_training_images,
)

setup_logger("DEBUG")
logger = logging.getLogger(__name__)


def main() -> None:

    # Configuración general del proyecto
    parameters = load_config("general.yaml")

    # Configuración específica del entrenamiento
    train_config = load_config("train.yaml")

    wandb_config : dict[str, Any] = {
        **parameters,
        **train_config,
         "note": "Best checkpoint obtained in the previous computer vision practice and registered for the MLOps project."
    }

    previous_run_path = Path("models/previous_best_run")
    best_model_path = previous_run_path / "weights" / "best.pt"
    results_path = previous_run_path / "results.csv"

    wandb.init(
        project="MLOps",
        name="previous_best_model",
        config=wandb_config,
        job_type="training",
        mode="online"
    )

    run = wandb.init(
        entity="delia-arjona-universidad-polit-cnica-de-madrid",
        project="MLOps",
        name="previous-best-yolo-model",
        config=wandb_config,
        job_type="model-registration",
        mode="online",
    )


    logger.info(f"W&B run URL: {run.url}")

    log_model_artifact(best_model_path)
    log_results_artifact(results_path)
    log_results_metrics(results_path)
    log_training_images(previous_run_path)

    wandb.finish()


if __name__ == "__main__":
    main()