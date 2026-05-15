import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from pandas import DataFrame
from tqdm import tqdm
import logging


class CarPartsDatasetAnalyzer:
    """
    Clase encargada de analizar el dataset de segmentación.

    A partir de las máscaras del dataset:
    - detecta qué piezas aparecen en cada imagen
    - genera un DataFrame resumen
    - almacena el resultado en cache para evitar recalcularlo
    """

    def __init__(
        self,
        classes: DataFrame,
        dataset_path: Path,
        masks_path: Path,
        cache_file: str
    ) -> None:

        # DataFrame con la información de clases del dataset.
        self.classes = classes

        # Ruta raíz del dataset.
        self.dataset_path = dataset_path

        # Ruta donde se encuentran las máscaras.
        self.masks_path = masks_path

        # Ruta completa del fichero cache.
        self.cache_file = self.dataset_path / cache_file

        # Diccionario que asigna un ID numérico a cada clase.
        # Ejemplo:
        # {
        #     "hood": 0,
        #     "wheel": 1,
        #     ...
        # }
        self.class_to_id: dict[str, int] = {
            str(row["class"]): idx
            for idx, (_, row) in enumerate(self.classes.iterrows())
        }
        self.logger = logging.getLogger(__name__)

    def load_or_analyze(self) -> DataFrame:
        """
        Carga el DataFrame cacheado si existe.
        En caso contrario, analiza el dataset y genera el cache.
        """

        if self.cache_file.exists():
            self.logger.info(f"Dataset loaded from cache: {self.cache_file}")
            return pd.read_csv(self.cache_file)

        # Si no existe cache, analizamos el dataset.
        df = self.analyze_dataset()

        # Guardamos el resultado para futuras ejecuciones.
        df.to_csv(self.cache_file, index=False)

        
        self.logger.info(f"Dataset analyzed and saved to: {self.cache_file}")

        return df

    def analyze_dataset(self) -> DataFrame:
        """
        Analiza todas las máscaras del dataset.

        Para cada imagen:
        - comprueba qué piezas aparecen
        - genera una fila binaria indicando presencia/ausencia
        """

        # Lista donde se almacenarán las filas del DataFrame.
        data: list[dict[str, Any]] = []

        # Recuperamos todas las máscaras PNG.
        mask_paths = list(self.masks_path.glob("*.png"))

        for mask_path in tqdm(mask_paths, desc="Analizando máscaras"):

            # Leemos la máscara en escala de grises.
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                continue

            # Fila base asociada a la imagen actual.
            row: dict[str, Any] = {
                "image_name": mask_path.name,
                "mask_name": mask_path.name
            }

            # Comprobamos presencia de cada clase.
            for part_name, class_id in self.class_to_id.items():

                # Ignoramos la clase de fondo.
                if part_name == "background":
                    continue

                # Añadimos 1 si la clase aparece en la máscara,
                # 0 en caso contrario.
                row[part_name] = (
                    1 if np.any(mask == class_id) else 0
                )

            data.append(row)

        # Convertimos el resultado a DataFrame.
        return pd.DataFrame(data)


def split_train_val_test(
    data_path: str,
    images_path: str,
    train_perc: float = 0.7,
    val_perc: float = 0.2,
    test_perc: float = 0.1
) -> None:
    """
    Divide el dataset en train, validation y test.

    Genera tres ficheros:
    - train.txt
    - val.txt
    - test.txt

    Cada fichero contiene las rutas absolutas de las imágenes.
    """

    logger = logging.getLogger(__name__)
    logger.info("Generating train/validation/test splits")

    data_path_path = Path(data_path)
    images_path_path = Path(images_path)


    # Si ya tenemos un split, no lo genera de nuevo
    if (
        (data_path_path / "train.txt").exists()
        and (data_path_path / "val.txt").exists()
        and (data_path_path / "test.txt").exists()
    ):
        logger.info("Dataset split already exists.")
        return

    # Recuperamos todas las imágenes.
    images = (
        list(images_path_path.glob("*.jpg")) +
        list(images_path_path.glob("*.png"))
    )

    # Mezclamos aleatoriamente las imágenes.
    random.shuffle(images)

    # Comprobamos que los porcentajes suman 1.
    assert abs(train_perc + val_perc + test_perc - 1.0) < 1e-6

    n = len(images)

    # Calculamos índices de separación.
    train_end = int(n * train_perc)
    val_end = int(n * (train_perc + val_perc))

    # División del dataset.
    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    def save_list(
        img_list: list[Path],
        filename: str
    ) -> None:
        """
        Guarda una lista de imágenes en un fichero .txt
        """

        with open(data_path_path / filename, "w") as f:
            for img in img_list:
                f.write(str(img.resolve()) + "\n")

    # Guardamos los splits.
    save_list(train_imgs, "train.txt")
    save_list(val_imgs, "val.txt")
    save_list(test_imgs, "test.txt")


def generate_yolo_yaml(
    path: str,
    file_name: str
) -> None:
    """
    Genera el fichero YAML requerido por YOLO.

    Este fichero define:
    - rutas de train/val/test
    - nombres de clases
    - estructura del dataset
    """

    yaml_path = Path(f"{path}/{file_name}")

    # Solo se crea si no existe previamente.
    if not yaml_path.exists():

        dataset_config: dict[str, Any] = {

            # Ruta raíz del dataset.
            "path": str(path),

            # Ficheros de splits.
            "train": "train.txt",
            "val": "val.txt",
            "test": "test.txt",

            # Diccionario de clases.
            "names": {
                0: "hood",
                1: "trunk",
                2: "windshield",
                3: "rear_window",
                4: "headlight",
                5: "tail_light",
                6: "front_door",
                7: "rear_door",
                8: "front_bumper",
                9: "rear_bumper",
                10: "wheel",
                11: "mirror"
            }
        }

        # Guardamos el YAML.
        with open(yaml_path, "w") as f:
            yaml.dump(
                dataset_config,
                f,
                default_flow_style=False
            )
