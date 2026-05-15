from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from tqdm import tqdm
import logging

class YoloAnnotationConverter:
    """
    Convierte máscaras de segmentación en anotaciones YOLO segmentation.

    Cada máscara contiene IDs de clase por píxel. Esta clase extrae los contornos
    de cada pieza del coche y los guarda en formato YOLO:
    
    class_id x1 y1 x2 y2 x3 y3 ...
    """

    def __init__(
        self,
        class_to_id: dict[str, int],
        masks_path: str | Path
    ) -> None:
        # Diccionario que relaciona nombre de clase con su ID numérico.
        self.class_to_id = class_to_id

        # Ruta donde están almacenadas las máscaras.
        self.masks_path = Path(masks_path)

    def process_single_mask(
        self,
        row: Mapping[str, Any],
        output: Path,
        min_area: int
    ) -> None:
        """
        Procesa una única máscara y genera su fichero .txt de anotación YOLO.
        """

        # Recuperamos la ruta de la máscara asociada a la fila actual.
        mask_path = self.masks_path / str(row["mask_name"])

        # Leemos la máscara en escala de grises.
        # Cada valor de píxel representa una clase.
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            return

        h, w = mask.shape

        # Lista donde se guardarán todas las líneas YOLO de esta imagen.
        lines: list[str] = []

        # Obtenemos los IDs de clase presentes en la máscara.
        present_ids = np.unique(mask)

        for class_id in present_ids:
            # Ignoramos el fondo.
            if class_id == 0:
                continue

            # Ignoramos valores que no correspondan a ninguna clase conocida.
            if int(class_id) not in self.class_to_id.values():
                continue

            # Creamos una máscara binaria solo para la clase actual.
            binary_piece: NDArray[np.uint8] = (mask == class_id).astype(np.uint8)

            # Extraemos los contornos de la clase actual.
            contours: Any
            contours, _ = cv2.findContours(  # pyright: ignore[reportCallIssue]
                binary_piece,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # YOLO espera clases empezando en 0.
            # Si la máscara usa 1..N, restamos 1.
            yolo_id = int(class_id) - 1

            for cnt in contours:
                # Un contorno necesita al menos 3 puntos para formar un polígono.
                if len(cnt) < 3:
                    continue

                # Descartamos regiones demasiado pequeñas.
                if cv2.contourArea(cnt) < min_area:
                    continue

                # Convertimos el contorno a coordenadas x, y.
                points = cnt.reshape(-1, 2).astype(np.float32)

                # Normalizamos coordenadas al rango [0, 1], como exige YOLO.
                points[:, 0] /= w
                points[:, 1] /= h
                points = np.clip(points, 0.0, 1.0)

                if len(points) < 3:
                    continue

                # Aplanamos los puntos:
                # [[x1, y1], [x2, y2]] -> [x1, y1, x2, y2]
                flattened = points.flatten()

                # Construimos la línea YOLO.
                line = f"{yolo_id} " + " ".join(
                    f"{x:.6f}" for x in flattened
                )

                lines.append(line)

        # El fichero de anotación debe tener el mismo nombre base que la imagen.
        txt_name = Path(str(row["image_name"])).stem + ".txt"

        # Guardamos las anotaciones.
        with open(output / txt_name, "w") as f:
            f.write("\n".join(lines))

    def save_annotations(
        self,
        df: DataFrame,
        output_path: str | Path,
        min_area: int = 20
    ) -> None:
        """
        Genera las anotaciones YOLO para todas las máscaras del dataset.
        """

        logger = logging.getLogger(__name__)
        logger.info("Generating YOLO annotations")

        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)

        existing_files = list(output.glob("*.txt"))

        # Si ya existe una anotación por cada fila del DataFrame,
        # evitamos recalcular todo.
        if len(existing_files) == len(df):
            logger.info(f"YOLO annotations already exist in: {output}")
            return

        rows = cast(list[dict[str, Any]], df.to_dict("records"))


        for row in tqdm(rows, desc="Progreso"):
            self.process_single_mask(row, output, min_area)
