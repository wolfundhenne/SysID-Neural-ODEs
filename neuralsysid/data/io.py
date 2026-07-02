"""Utilities for loading output-control trajectory data from JSON files."""

from copy import deepcopy
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import boto3
import os


def infer_sample_time(t: np.ndarray) -> float:
    t_flat = np.asarray(t).reshape(-1)
    return float(t_flat[1] - t_flat[0])


def _extract_metadata(datadict: Dict[str, Any]) -> Dict[str, Any]:
    metadata = deepcopy(datadict.get("metadata", {}))
    if "simulation_config" in datadict:
        metadata["simulation_config"] = deepcopy(datadict["simulation_config"])
    if "grid_config" in datadict:
        metadata["grid_config"] = deepcopy(datadict["grid_config"])
    return metadata


def _extract_trajectory_dicts(
    datadict: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    trajectories = datadict["trajectories"]
    output_trajectories = trajectories.get(
        "outputs", trajectories.get("states")
    )
    control_trajectories = trajectories.get(
        "controls", trajectories.get("control_inputs")
    )
    if output_trajectories is None or control_trajectories is None:
        raise KeyError(
            "trajectories must contain outputs/states and controls/control_inputs"
        )
    return output_trajectories, control_trajectories


def load_data(
    datapath: str, extensions: List[str] = ["train", "dev", "test"]
) -> Tuple[List[Tuple[np.ndarray, Dict[str, np.ndarray]]], Dict[str, Any]]:
    """Load trajectory data from json files.

    The returned dataset uses ``Y`` for observed outputs and ``U`` for
    controls.

    Args:
        name (str): file name without extension
        extensions (List[str], optional): Extensions appended to the file name.
            Defaults to ["train", "dev", "test"].

    Returns:
        Tuple[List[Tuple[np.ndarray, Dict[str, np.ndarray]]], Dict[Any]]: time
            vector, as well as a dictionary with outputs and controls
            collected in a tuple. Respective tuples of each data set collected
            in a list.
    """
    data_sets = []
    metadata = None

    for ext in extensions:
        with open(datapath + ext + ".json", "r") as f:
            datadict = json.load(f)

        if metadata is None:
            metadata = _extract_metadata(datadict)

        output_trajectories, control_trajectories = _extract_trajectory_dicts(
            datadict
        )
        output_names = list(output_trajectories.keys())
        control_names = list(control_trajectories.keys())

        t = np.array(datadict["trajectories"]["t"])[None, :]
        outputs = np.array(
            [output_trajectories[name] for name in output_names]
        ).T[None, :]
        controls = np.array(
            [control_trajectories[name] for name in control_names]
        ).T[None, :]

        data = {
            "Y": outputs,
            "U": controls,
            "output_names": output_names,
            "control_names": control_names,
        }
        data_sets.append([t, data])

    return data_sets, metadata if metadata is not None else {}


def load_s3(data_name, extensions: List[str] = ["train", "dev", "test"]):
    boto3_client = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_USER"),
        aws_secret_access_key=os.getenv("MINIO_PASSWORD"),
    )

    for suffix in extensions:
        if not os.path.exists("data/" + data_name + suffix + ".json"):
            boto3_client.download_file(
                "simulated-datasets",
                "emt/" + data_name + suffix + ".json",
                "data/" + data_name + suffix + ".json",
            )
