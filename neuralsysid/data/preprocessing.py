"""This module contains functions for data preprocessing, including
normalization, resampling, and unnormalization of data."""

from typing import Dict, List, Tuple

import numpy as np


def resample_data(
    t: np.ndarray,
    data_set: Dict[str, np.ndarray],
    t_sample: float,
    t_step: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """This function resamples the data set to a new, smaller, time step.

    Args:
        t (np.ndarray): time vector
        data_set (Dict[str, np.ndarray]): data containing the state and control
            input trajectories
        T_sample (float): new sampling time step
        T_step (float): original sampling time step

    Returns:
        Tuple[np.ndarray, Dict[str, np.ndarray]]: resampled time vector and data
            set
    """

    every_nth = round(t_step / t_sample)
    t_res = t[:, ::every_nth]

    data_set_res = {}
    for key in data_set.keys():
        if key in {"output_names", "control_names"}:
            data_set_res[key] = data_set[key]
        else:
            data_set_res[key] = data_set[key][:, ::every_nth, :]

    return t_res, data_set_res


def normalize_data(
    data_set: Dict[str, np.ndarray], data_stats: Dict[str, np.ndarray] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """This function normalizes a data set. Depending on weather the statistics
    are provided or not, the function will either use the provided stats or
    compute the stats from the data set for the normalization.

    Args:
        data_set (Dict[str, np.ndarray]): data containing the state and control
            input trajectories
        data_stats (Dict[str, np.ndarray], optional): data statistics. Defaults
            to None.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: The normalized data
            set and the data statistics.
    """

    y = data_set["Y"].squeeze()
    u = data_set["U"].squeeze()

    if data_stats is None:
        mean_y = y.mean(axis=0)
        std_y = y.std(axis=0)
        mean_u = u.mean(axis=0)
        std_u = u.std(axis=0)

        std_y[np.abs(std_y) < 1e-10] = 1
        std_u[np.abs(std_u) < 1e-10] = 1
    else:
        mean_y = data_stats["mean_y"]
        std_y = data_stats["std_y"]
        mean_u = data_stats["mean_u"]
        std_u = data_stats["std_u"]

    y = normalize(y, mean_y, std_y)
    u = normalize(u, mean_u, std_u)

    y = y[np.newaxis, ...]
    u = u[np.newaxis, ...]

    normalized_data_set = {
        "Y": y,
        "U": u,
        "output_names": data_set["output_names"],
        "control_names": data_set["control_names"],
    }

    data_stats = {
        "mean_y": mean_y,
        "std_y": std_y,
        "mean_u": mean_u,
        "std_u": std_u,
    }

    return normalized_data_set, data_stats


def resample_data_sets(
    data_sets: List[Dict[str, np.ndarray]], t_sample: float, t_step: float
) -> List[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """This function resamples a set of data sets to a new, smaller, time step.

    Args:
        data_sets (List[Dict[str, np.ndarray]]): distinct data sets
        T_sample (float): new sampling time step
        T_step (float): original sampling time step

    Returns:
        List[Tuple[np.ndarray, Dict[str, np.ndarray]]]: resampled time vectors
            and resampled data sets
    """
    data_sets_res = []
    for t, data_set in data_sets:
        data_sets_res.append(resample_data(t, data_set, t_sample, t_step))

    return data_sets_res


def unnormalize(
    x: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray
) -> np.ndarray:
    """This function unnormalizes a data. It uses a given mean and standard
    deviation of the data to unnormalize the data.

    Args:
        X (np.ndarray): data set to be unnormalized
        x_mean (np.ndarray): mean of the data
        x_std (np.ndarray): standard deviation of the data

    Returns:
        np.ndarray: unnormalized data
    """
    return x * x_std + x_mean


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """This function normalizes given data. It uses a given mean and standard
    deviation of the data to normalize the data.

    Args:
        x (np.ndarray): data set to be normalized
        mean (np.ndarray): mean to be used for normalization
        std (np.ndarray): standard deviation to be used for normalization

    Returns:
        np.ndarray: normalized data
    """
    return (x - mean) / std
