import logging
from pathlib import Path

from src.utils import get_project_folder


def setup_logger(level : str, log_filename : str = "app.log") -> logging.Logger:
    '''
    Configura el sistema de logging del proyecto.

    Los logs se envían a:
    - consola
    - fichero dentro de la carpeta logs/
    '''

    # Recuperamos la ruta raíz del proyecto.
    project_folder = get_project_folder()

    # Creamos la carpeta logs si todavía no existe.
    Path(project_folder/"logs").mkdir(exist_ok=True)

    # Ruta completa del fichero de logs.
    log_file = project_folder/ "logs" / log_filename

    # Configuración global de logging:
    logging.basicConfig(
        level = getattr(logging, level.upper(), logging.DEBUG),
        format="%(asctime)s - %(levelname)-8s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    return logging.getLogger(__name__)
