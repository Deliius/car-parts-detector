import logging
from src.utils import get_project_folder
from pathlib import Path


def setup_logger(level : str, log_filename : str = "app.log") -> logging.Logger:

    project_folder = get_project_folder()
    Path(project_folder/"logs").mkdir(exist_ok=True)

    log_file = project_folder/ "logs" / log_filename

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