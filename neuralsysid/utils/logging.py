"""This module provides a function for logging experiments to mlflow."""

import os
from typing import Any, Dict, List

import mlflow
import numpy as np
from matplotlib import pyplot as plt
from torch import nn


def strip_weight_parametrizations(model):
    for module in model.modules():
        # fold all parametrizations on 'weight' into a single plain Parameter
        try:
            nn.utils.parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=True
            )
        except (ValueError, AttributeError):
            pass


def _serialize_config(value):
    if isinstance(value, dict):
        return {key: _serialize_config(entry) for key, entry in value.items()}
    if isinstance(value, list):
        return [_serialize_config(entry) for entry in value]
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, nn.Module):
        return str(value)
    return value


def _get_config_path(config, path):
    value = config
    for key in path.split("."):
        value = value[key]
    return value


def log_run(
    hyperparameters: List[str] = None,
    model_configuration: Dict[str, Any] = None,
    sim_configuration: Dict[str, Any] = None,
    dynamics_model: nn.Module = None,
    mean_train_loss: float = None,
    mean_dev_loss: float = None,
    mean_test_loss: float = None,
    train_loss_over_epochs: List[float] = None,
    dev_loss_over_epochs: List[float] = None,
    avg_time_epoch: float = None,
    time_vectors: List[np.ndarray] = None,
    data_sets: List[Dict[str, Any]] = None,
    chip_type: str = None,
):
    """This function logs the run to mlflow. Only the parameters that are not
    None are logged.

    Args:
        hyperparameters (List[str], optional): List of Hyperparameters used in
            some hyperparameter optimization task. Defaults to None.
        model_configuration (Dict[str, Any], optional): A dictionary containing
            the configuration of a model. Defaults to None.
        sim_configuration (Dict[str, Any], optional): A dictionary contining the
            configuration of the simulation. Defaults to None.
        dynamics_model (nn.Module, optional): the learned model. Defaults to
            None.
        mean_train_loss (float, optional): Mean training loss over the entire
            training data set. Defaults to None.
        mean_dev_loss (float, optional): Mean validation loss over the entire
            training data set. Defaults to None.
        mean_test_loss (float, optional): Mean test loss over the entire test
            data set. Defaults to None.
        train_loss_over_epochs (List[float], optional): The trajectory of the
            train loss evolving over the epochs. Defaults to None.
        dev_loss_over_epochs (List[float], optional): The trajectory of the
            dev loss evolving over the epochs. Defaults to None.
        avg_time_epoch (float, optional): Time needed for training per epoch.
            Defaults to None.
        time_vectors (List[np.ndarray], optional): The time vectors of the data-
            sets. Defaults to None.
        data_sets (List[Dict[str, Any]], optional): The datasets. Defaults to
            None.
    """

    if data_sets is not None and time_vectors is not None:
        data_sets = {
            "t_train": data_sets["t_train"],
            "t_dev": data_sets["t_dev"],
            "t_test": data_sets["t_test"],
            "train": data_sets["train"],
            "dev": data_sets["dev"],
            "test": data_sets["test"],
        }

    if train_loss_over_epochs is not None and dev_loss_over_epochs is not None:
        # log dictionaries
        loss_trajectories = {
            "train": train_loss_over_epochs,
            "dev": dev_loss_over_epochs,
        }
        mlflow.log_dict(loss_trajectories, "loss_trajectories.json")

        fig_trainloss = plt.figure()
        plt.plot(train_loss_over_epochs, label="train")
        plt.yscale("log")

        fig_devloss = plt.figure()
        plt.plot(dev_loss_over_epochs, label="dev")
        plt.yscale("log")

        mlflow.log_figure(fig_trainloss, "train_loss_over_epochs.png")
        mlflow.log_figure(fig_devloss, "dev_loss_over_epochs.png")

    if model_configuration is not None:
        mlflow.log_dict(
            _serialize_config(model_configuration),
            "model_configuration.json",
        )

    if sim_configuration is not None:
        for key in sim_configuration:
            if isinstance(sim_configuration[key], type):
                sim_configuration[key] = sim_configuration[key].__name__
            elif isinstance(sim_configuration[key], nn.Module):
                sim_configuration[key] = str(sim_configuration[key])
        mlflow.log_dict(sim_configuration, artifact_file="metadata.json")

    # log model parameters that were optimized
    if hyperparameters is not None and model_configuration is not None:
        for key in hyperparameters:
            mlflow.log_param(key, _get_config_path(model_configuration, key))
    if chip_type is not None:
        mlflow.log_param("chip_type", chip_type)
    if model_configuration is not None:
        mlflow.log_param(
            "batch_size", model_configuration["train"]["batch_size"]
        )

    if dynamics_model is not None:
        # log model
        strip_weight_parametrizations(dynamics_model)
        model_info = mlflow.pytorch.log_model(dynamics_model, "dynamics_model")
        model_info_as_dict = {
            "model_id": model_info.model_id,
            "model_uri": model_info.model_uri,
            "model_uuid": model_info.model_uuid,
        }
        mlflow.log_dict(model_info_as_dict, "model_info.json")

    if (
        avg_time_epoch is not None
        and mean_train_loss is not None
        and mean_dev_loss is not None
        and mean_test_loss is not None
    ):
        mlflow.log_metrics(
            {
                "avg_time_epoch": avg_time_epoch,
                "mean_train_loss": mean_train_loss,
                "mean_dev_loss": mean_dev_loss,
                "mean_test_loss": mean_test_loss,
            }
        )

    if os.getenv("GITLAB_CI"):
        mlflow.set_tags({"gitlab.CI_JOB_ID": os.getenv("CI_JOB_ID")})

    plt.close()
