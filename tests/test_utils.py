import random
from pathlib import Path

import numpy as np
import pytest
from pytest import MonkeyPatch

from src import utils
from src.utils import get_project_folder, load_config, set_seeds


def test_get_project_folder_returns_existing_path() -> None:
    '''
    Comprueba que la ruta raíz del proyecto existe y es un directorio.
    '''

    project_folder = get_project_folder()

    assert isinstance(project_folder, Path)
    assert project_folder.exists()
    assert project_folder.is_dir()


def test_load_config_returns_dictionary() -> None:
    '''
    Comprueba que load_config carga un YAML como diccionario no vacío.
    '''

    config = load_config("general.yaml")

    assert isinstance(config, dict)
    assert len(config) > 0


def test_load_config_raises_file_not_found() -> None:
    '''
    Comprueba que load_config falla si el fichero no existe.
    '''

    with pytest.raises(FileNotFoundError):
        load_config("missing_file.yaml")


def test_set_seeds_is_reproducible() -> None:
    '''
    Comprueba que set_seeds hace reproducibles random y numpy.
    '''

    set_seeds(42)

    random_value_1 = random.random()
    numpy_value_1 = np.random.rand()

    set_seeds(42)

    random_value_2 = random.random()
    numpy_value_2 = np.random.rand()

    assert random_value_1 == random_value_2
    assert numpy_value_1 == numpy_value_2


def test_download_data_skips_if_exists(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    '''
    Comprueba que download_data no descarga nada si el dataset ya existe.
    '''

    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    # Bandera para comprobar si se llama a kagglehub.dataset_download.
    called = False

    def fake_download(*args: object, **kwargs: object) -> str:
        # Si esta función se ejecuta, el test debe detectarlo.
        nonlocal called
        called = True
        return ""

    # Sustituimos la descarga real por una función fake.
    monkeypatch.setattr(
        utils.kagglehub,
        "dataset_download",
        fake_download,
    )

    utils.download_data(dataset_path)

    assert called is False


def test_download_data_downloads_when_missing(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    '''
    Comprueba que download_data descarga y copia el dataset si no existe.
    '''

    dataset_path = tmp_path / "dataset"
    fake_download_path = tmp_path / "downloaded"

    # Creamos una carpeta que simula el dataset descargado por KaggleHub.
    fake_download_path.mkdir()
    (fake_download_path / "test.txt").write_text("data")

    def fake_download(*args: object, **kwargs: object) -> str:
        return str(fake_download_path)

    # Sustituimos KaggleHub por la ruta local fake.
    monkeypatch.setattr(
        utils.kagglehub,
        "dataset_download",
        fake_download,
    )

    utils.download_data(dataset_path)

    assert dataset_path.exists()
    assert (dataset_path / "test.txt").exists()
