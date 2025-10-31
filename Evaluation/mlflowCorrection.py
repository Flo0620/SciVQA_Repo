import mlflow
import json
import pandas as pd

from dataclasses import asdict, is_dataclass
from typing import Any

from omegaconf import ListConfig

from scivqa.evaluation.config import Config


def _process_values(value: Any) -> dict[str, str | int | float]:
    if value is None:
        return {"": "None"}
    if isinstance(value, bool):
        return {"": str(value)}
    if isinstance(value, (str, int, float)):
        return {"": value}
    if isinstance(value, (list, ListConfig)):
        return {
            f"{i}{k0}": v0 for i, v in enumerate(value) for k0, v0 in _process_values(v).items()
        }
    try:
        return {
            f"_{key}{k0}": v0 for key, v in value.items() for k0, v0 in _process_values(v).items()
        }
    except AttributeError as e:
        raise ValueError(f"Can not convert value {value} to scalar. Type is {type(value)}") from e


def flatten_dict(config: dict | Config | Any) -> dict[str, str | int | float]:
    """Convert a nested config to a flat dictionary suitable for TensorBoard logging.

    :param config: A dict or Config instance
    :return: A flat dictionary with keys of the form
        "outer_middle_inner".
    :raises ValueError: If values of config are not of type str, int,
        float or bool.
    """
    # Convert Config to a dictionary if it is a dataclass
    if is_dataclass(config):
        config = asdict(config)  # type: ignore
    elif not isinstance(config, dict):
        try:
            config = dict(config)
        except TypeError:
            raise ValueError("The provided config must be a dictionary or a dataclass.") from None

    # Flatten the dictionary
    return {
        f"{key}{k}": v for key, value in config.items() for k, v in _process_values(value).items()
    }
from dataclasses import asdict, is_dataclass
from typing import Any

from omegaconf import ListConfig

from scivqa.evaluation.config import Config
import numpy as np


def _process_values(value: Any) -> dict[str, str | int | float]:
    if value is None:
        return {"": "None"}
    if isinstance(value, bool):
        return {"": str(value)}
    if isinstance(value, (str, int, float)):
        return {"": value}
    if isinstance(value, (list, ListConfig)):
        return {
            f"{i}{k0}": v0 for i, v in enumerate(value) for k0, v0 in _process_values(v).items()
        }
    try:
        return {
            f"_{key}{k0}": v0 for key, v in value.items() for k0, v0 in _process_values(v).items()
        }
    except AttributeError as e:
        raise ValueError(f"Can not convert value {value} to scalar. Type is {type(value)}") from e


def flatten_dict(config: dict | Config | Any) -> dict[str, str | int | float]:
    """Convert a nested config to a flat dictionary suitable for TensorBoard logging.

    :param config: A dict or Config instance
    :return: A flat dictionary with keys of the form
        "outer_middle_inner".
    :raises ValueError: If values of config are not of type str, int,
        float or bool.
    """
    # Convert Config to a dictionary if it is a dataclass
    if is_dataclass(config):
        config = asdict(config)  # type: ignore
    elif not isinstance(config, dict):
        try:
            config = dict(config)
        except TypeError:
            raise ValueError("The provided config must be a dictionary or a dataclass.") from None

    # Flatten the dictionary
    return {
        f"{key}{k}": v for key, value in config.items() for k, v in _process_values(value).items()
    }


mlflow.set_tracking_uri("https://mlflow-g4k-serving-474827717259.europe-west3.run.app/")

with mlflow.start_run(run_id="86755bf6f4544c1e942522c76fc418aa") as run:

    # Read the JSON file
    with open("/ltstorage/home/9schleid/scivqa/outputs/25-04-23_12:43_gpt-4o-mini/inference_log_shortend.json", "r") as file:
        data = json.load(file)

    # Flatten each entry in the JSON
    json_dump = [flatten_dict(response) for response in data]
    mlflow.log_table(data=pd.DataFrame(json_dump), artifact_file="86755bf6f4544c1e942522c76fc418aa.json")