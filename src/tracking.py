from pathlib import Path
from typing import Any

import pandas as pd
import wandb


def log_split_artifact(data_path: Path) -> None:
    artifact = wandb.Artifact(
        name="carparts-splits",
        type="dataset",
        description="Train/validation/test split files"
    )

    artifact.add_file(str(data_path / "car_parts.yaml"))
    artifact.add_file(str(data_path / "train.txt"))
    artifact.add_file(str(data_path / "val.txt"))
    artifact.add_file(str(data_path / "test.txt"))

    wandb.log_artifact(artifact)


def log_results_artifact(results_path: Path) -> None:

    if not results_path.exists():
        print(f"No existe results.csv en: {results_path}")
        return

    artifact = wandb.Artifact(
        name="training-results",
        type="results",
        description="YOLO training metrics per epoch"
    )

    artifact.add_file(str(results_path))

    wandb.log_artifact(artifact)


def log_model_artifact(model_path: Path) -> None:
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    artifact = wandb.Artifact(
        name="best-yolo-model",
        type="model"
    )

    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)


def log_training_images(run_path: Path) -> None:

    image_files = list(run_path.glob("*.jpg"))

    for image_file in image_files:

        wandb.log({
            image_file.stem: wandb.Image(
                str(image_file),
                caption=image_file.name
            )
        })


def log_results_metrics(results_path: Path) -> None:

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
    
    df = pd.read_csv(results_path)

    for _, row in df.iterrows():

        metrics: dict[str, Any] = {
            str(k): v
            for k, v in row.to_dict().items()
        }

        wandb.log(metrics)
