"""This module implements functions to add noise to the data."""

import numpy as np

np.random.seed(42)


def add_noise(
    data_sets: np.ndarray, uncertainties: float, n_stds: int = 3
) -> np.ndarray:
    """Adds Gaussian noise to the data.

    Args:
        data (np.ndarray): Original data.
        noise_level (float): A dictionary containing the measurement uncertainty
            as a standard deviation for each state.

    Returns:
        np.ndarray: Noisy data.
    """

    for data_set in data_sets:
        data_set[1]["Y"] += (
            data_set[1]["Y"]
            * uncertainties[None, None, :]
            / n_stds
            * np.random.randn(*data_set[1]["Y"].shape)
        )


def add_noise_snr(data_sets: dict, snr_db: float = 25) -> dict:
    """Adds Gaussian noise, s.t. the zero-mean signal to noise ratio is snr_db."""

    for data_set in data_sets:
        signal_power = np.var(data_set[1]["Y"], axis=1).squeeze()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(
            0, np.sqrt(noise_power), data_set[1]["Y"].shape
        )
        data_set[1]["Y"] += noise

    return data_sets
