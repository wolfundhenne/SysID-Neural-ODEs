import numpy
import torch
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
import sys
import os
from torch import nn
from torch.optim.optimizer import Optimizer
from typing import Callable, Optional, Any

from neuralsysid.data import datasets, preprocessing
from neuralsysid.models import (
    neuralnetworks,
    predictors,
    statedynamics,
    timesteppers,
)
from neuralsysid.training import trainer
from torch import Tensor
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Dataset for multi-step ahead predictions. Consider L is the sequence
    length, n_x is the dimension of the state and n_u is the dimension of the
    input. Then with k sequences, we get:

    Args:
        x_0 (Tensor): The initial state. Has dimensions (k, n_x).
        u (Tensor): The input sequence. (k, L, n_u).
        x (Tensor): The output sequence. (k, L, n_x).
    """

    def __init__(self, x_0: Tensor, u: Tensor, x: Tensor):
        super().__init__()
        self.len = x_0.shape[0]
        self.x_0 = x_0
        self.u = u
        self.x = x

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.len

    def __getitem__(self, idx):
        """Returns the sample at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[Tensor, Tensor]: The input and output features.
        """
        return (self.x_0[idx], self.u[idx]), self.x[idx]


def create_batches_sequence(
    t: numpy.ndarray,
    data_set: Dict[str, numpy.ndarray],
    used_states: List[int],
    used_inputs: List[int],
    batch_size: int,
    shuffle: bool = True,
    seq_len: int = 10,
    overlap: int = 0,
    cutoff_samples: int = 0,
) -> Tuple[numpy.ndarray, SequenceDataset, DataLoader]:
    """This function creates batches of data for training or evaluation for one
    dataset. A pytorch DataSet, a pytorch DataLoader and a time vector, which is
    not required for the training, are returned.

    Args:
        t (numpy.ndarray): time vector
        data_set (dict): dictionary containing the data
        used_states (List[int]): states used for training
        used_inputs (List[int]): control inputs used for training
        batch_size (int): size of one batch
        shuffle (bool, optional): wheather to shuffle the batches in training or
             not Defaults to True.

    Returns:
        Tuple[numpy.ndarray, SequenceDataset, DataLoader]: time
            vector, dataset and dataloader
    """

    # create sequences from one trajectory. sequences can be overlapping

    t = t.squeeze(0)
    x = data_set["X"].squeeze(0)[:, used_states]
    u = data_set["U"].squeeze(0)[:, used_inputs]

    # cut off initial samples if specified
    x = x[cutoff_samples:, :]
    u = u[cutoff_samples:, :]
    t = t[cutoff_samples:]

    n_timesteps = x.shape[0]
    n_sequences = (n_timesteps - seq_len) // (seq_len - overlap) + 1

    t_seq = numpy.zeros((n_sequences, seq_len))
    x_0 = numpy.zeros((n_sequences, x.shape[1]))
    u_in = numpy.zeros((n_sequences, seq_len, u.shape[1]))
    x_out = numpy.zeros((n_sequences, seq_len, x.shape[1]))
    for i_iter in range(n_sequences):
        start_idx = i_iter * (seq_len - overlap)
        end_idx = start_idx + seq_len

        t_seq[i_iter, :] = t[start_idx:end_idx]
        x_0[i_iter, :] = x[start_idx, :]
        u_in[i_iter, :, :] = u[start_idx:end_idx, :]
        x_out[i_iter, :, :] = x[start_idx:end_idx, :]

    x_0 = torch.tensor(x_0, dtype=torch.float32)
    u_in = torch.tensor(u_in, dtype=torch.float32)
    x_out = torch.tensor(x_out, dtype=torch.float32)

    data_set = SequenceDataset(x_0, u_in, x_out)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return (
        t_seq,
        data_set,
        data_loader,
    )


def generate_batch_sets_sequence(
    t_vecs: List[numpy.ndarray],
    data_sets: List[Dict[str, numpy.ndarray]],
    used_states: List[int],
    used_inputs: List[int],
    batch_size: int,
    shuffle: bool = True,
    seq_len: int = 10,
    overlap: int = 0,
) -> Tuple[Tuple[numpy.ndarray, SequenceDataset, DataLoader]]:
    """_summary_

    Args:
        t_vecs (List[numpy.ndarray]): independent time vectors
        data_sets (List[Dict[str, numpy.ndarray]]): independent datasets
        used_states (List[int]): states used for training
        used_inputs (List[int]): control inputs used for training
        batch_size (int): size of one batch
        shuffle (bool, optional): whether to shuffle the batches in training or
            not Defaults to True.

    Returns:
        Tuple[Tuple[numpy.ndarray, OneStepTrajectoriesDataset, DataLoader]]:
            time vector, dataset and dataloader for each dataset
    """
    batch_sets = []
    for t, data_set in zip(t_vecs, data_sets):
        batch_sets.append(
            create_batches_sequence(
                t,
                data_set,
                used_states,
                used_inputs,
                batch_size,
                shuffle=shuffle,
                seq_len=seq_len,
                overlap=overlap,
            )
        )

    return batch_sets


def learn_dynamics(
    model_configuration: Dict[str, Any],
    sim_configuration: Dict[str, Any],
    hyperparameters: List[str],
    chip_type: str,
    data_sets: Dict[str, numpy.ndarray],
    log_mlf: bool = False,
    normalize_data: bool = True,
) -> Tuple[timesteppers.TimeStepper, float]:
    """This function learns the dynamics of a system using a neural network. The
    model is configured by the respective dictionary. Three datasets are used
    in the learning task: training, validation and test set. Another dictionary
    defines some parameter of the data, like sampling time and which states
    and control inputs of the system shall be used for the training.

    Args:
        model_configuration (Dict[str, Any]): Contains the hyperparameter choice
            for the model.
        sim_configuration (Dict[str, Any]): Contains information about the data
            contained in the given datasets.
        hyperparameters (List[str]): Contains a list of hyperparameters, in case
            that the function is called from a hyperparameter optimization.
        chip_type (str): Device used for training.
        data_sets (Dict[str, np.ndarray]): Contains the datasets for training,
            validation and test.
        log_mlf (bool, optional): Weather to log the results to mlflow. Defaults
             to False.
        normalize_data (bool, optional): Weather to normalize the data or not.
            Defaults to True.

    Returns:
        Tuple[timesteppers.TimeStepper, float]: trained model and the mean test
            loss
    """
    if "gpu" in chip_type:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            raise ValueError("No GPU available")
    else:
        device = torch.device("cpu")

    # necessary for reproducible results and tests to be successful
    torch.manual_seed(0)

    run_name = "configuration_" + str(model_configuration["config_id"])

    print("testing model: ")
    for hyperparameter in hyperparameters:
        print(hyperparameter, model_configuration[hyperparameter])

    nx = len(model_configuration["used_outputs"])
    nu = len(model_configuration["used_controls"])

    data_sets = preprocessing.resample_data_sets(
        data_sets,
        t_sample=sim_configuration["T_sample"],
        t_step=model_configuration["t_step"],
    )

    t_train, data_set_train = data_sets[0]
    t_dev, data_set_dev = data_sets[1]
    t_test, data_set_test = data_sets[2]
    if normalize_data:
        data_set_train_normalized, data_stats_train = (
            preprocessing.normalize_data(data_set_train)
        )
        data_set_dev_normalized, _ = preprocessing.normalize_data(
            data_set_dev, data_stats_train
        )
        data_set_test_normalized, _ = preprocessing.normalize_data(
            data_set_test, data_stats_train
        )
        data_set_train_normalized["X"] = data_set_train_normalized["Y"]
        data_set_dev_normalized["X"] = data_set_dev_normalized["Y"]
        data_set_test_normalized["X"] = data_set_test_normalized["Y"]
    else:
        data_set_train_normalized = data_set_train
        data_set_dev_normalized = data_set_dev
        data_set_test_normalized = data_set_test

    batch_sets = generate_batch_sets_sequence(
        [t_train, t_dev, t_test],
        [
            data_set_train_normalized,
            data_set_dev_normalized,
            data_set_test_normalized,
        ],
        model_configuration["used_outputs"],
        model_configuration["used_controls"],
        batch_size=model_configuration["batch_size"],
        shuffle=False,  # required for reproducible results
        seq_len=model_configuration["seq_len"],
        overlap=model_configuration["overlap"],
    )
    _, _, dataloader_train = batch_sets[0]
    _, _, dataloader_dev = batch_sets[1]
    _, _, dataloader_test = batch_sets[2]

    if model_configuration["dynamics"] == "fxgu":
        f_dyn = neuralnetworks.MultiLayerPerceptron(
            n_inputs=nx,
            n_outputs=nx,
            n_layers=model_configuration["n_layers_A"],
            n_hidden=model_configuration["n_hidden_A"],
            activation=model_configuration["act_fct"],
        )
        g_dyn = neuralnetworks.MultiLayerPerceptron(
            n_inputs=nu,
            n_outputs=nx,
            n_layers=model_configuration["n_layers_B"],
            n_hidden=model_configuration["n_hidden_B"],
            activation=model_configuration["act_fct"],
        )
        rhs = statedynamics.FxGuDynamics(
            n_states=nx,
            n_inputs=nu,
            f_dynamics=f_dyn,
            g_dynamics=g_dyn,
        )
    elif model_configuration["dynamics"] == "fxu":
        f_dyn = neuralnetworks.MultiLayerPerceptron(
            n_inputs=nx + nu,
            n_outputs=nx,
            n_layers=model_configuration["n_layers_dyn"],
            n_hidden=model_configuration["n_hidden_dyn"],
            activation=model_configuration["act_fct"],
        )
        rhs = statedynamics.FxuDynamics(
            n_states=nx,
            n_inputs=nu,
            dynamics=f_dyn,
        )
    else:
        raise ValueError("Dynamics not defined")

    if model_configuration["stepper"] == "discrete":
        dynamics = timesteppers.Discrete(rhs)
    elif model_configuration["stepper"] == "residual":
        dynamics = timesteppers.Residual(rhs)
    elif model_configuration["stepper"] == "euler":
        dynamics = timesteppers.Euler(rhs, model_configuration["t_step"])
    elif model_configuration["stepper"] == "rk4":
        dynamics = timesteppers.RungeKutta4(rhs, model_configuration["t_step"])
    elif model_configuration["stepper"] == "dopri5":
        dynamics = timesteppers.Dopri5(rhs, model_configuration["t_step"])
    else:
        raise ValueError("Stepper not defined")

    dynamics_model = predictors.StateRollout(
        dynamics, model_configuration["seq_len"] - 1
    )

    optimizer = torch.optim.Adam(
        dynamics_model.parameters(), lr=model_configuration["learning_rate"]
    )

    model_trainer = SequenceTrainer(
        model=dynamics_model,
        optimizer=optimizer,
        loss_fct=nn.MSELoss(),
        train_loader=dataloader_train,
        dev_loader=dataloader_dev,
        epochs=model_configuration["n_epochs"],
        patience=model_configuration["patience"],
        warmup=model_configuration["warmup"],
        clip_grad_threshhold=model_configuration["grad_clip"],
        device=device,
    )

    model_trainer.train()

    dynamics_model.load_state_dict(model_trainer.best_model_state_dict)

    mean_train_loss = model_trainer.evaluate(dataloader_train)
    mean_dev_loss = model_trainer.evaluate(dataloader_dev)
    mean_test_loss = model_trainer.evaluate(dataloader_test)

    return dynamics_model, mean_test_loss


class SequenceTrainer(trainer.BaseTrainer):
    """This class implements the base trainer, but the objective is only cal-
    culated for a subset of the states. This is useful for training models

    Args:
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): The optimizer to be used for training.
        loss_fct (nn.Module): The loss function to be used for training.
        train_loader (DataLoader): The DataLoader for the training data.
        dev_loader (DataLoader): The DataLoader for the validation data.
        reduced_state_set (List[int]): The list of indices of the states to be
            used for training. Defaults to None, which means all states are
            used.
        epochs (int): The number of epochs to train for.
        patience (int): The number of epochs to wait before stopping training if
            no improvement is seen.
        warmup (int): The number of epochs to wait before starting to track
            improvements.
        clip_grad_threshhold (float): The threshold for gradient clipping.
            Defaults to float("inf") (no clipping).
        device (device): The device to use for training ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fct: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_loader: DataLoader,
        dev_loader: DataLoader,
        epochs: int = 1000,
        patience: int = 50,
        warmup: int = 50,
        clip_grad_threshhold: float = float("inf"),
        device: torch.device = torch.device("cpu"),
        trial=None,
    ):
        super().__init__(
            model,
            optimizer,
            loss_fct,
            train_loader,
            dev_loader,
            epochs,
            patience,
            warmup,
            clip_grad_threshhold,
            device,
            trial,
        )

    def calc_objective(self, in_features, out_features):
        """Calculates the objective function for a given batch of data.

        Args:
            in_features (Tuple(torch.Tensor)): The input features. x_0 is at
                index 0, u is at index 1.
            out_features (torch.Tensor): The output features.

        Returns:
            torch.Tensor: The objective function value.
        """
        if self.move_batch:
            in_features_x0 = in_features[0].to(device=self.device)
            in_features_u = in_features[1].to(device=self.device)
            out_features = out_features.to(device=self.device)
            out = self.model(in_features_x0, in_features_u)
        else:
            out = self.model(in_features[0], in_features[1])
        loss = self.loss_fct(out[:, 1:, :], out_features[:, 1:, :])
        return loss


class StateSubsetSequenceTrainer(
    SequenceTrainer
):  # should not be used anymore
    """This class implements the base trainer, but the objective is only cal-
    culated for a subset of the states. This is useful for training models

    Args:
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): The optimizer to be used for training.
        loss_fct (nn.Module): The loss function to be used for training.
        train_loader (DataLoader): The DataLoader for the training data.
        dev_loader (DataLoader): The DataLoader for the validation data.
        reduced_state_set (List[int]): The list of indices of the states to be
            used for training. Defaults to None, which means all states are
            used.
        epochs (int): The number of epochs to train for.
        patience (int): The number of epochs to wait before stopping training if
            no improvement is seen.
        warmup (int): The number of epochs to wait before starting to track
            improvements.
        clip_grad_threshhold (float): The threshold for gradient clipping.
            Defaults to float("inf") (no clipping).
        device (device): The device to use for training ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fct: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_loader: DataLoader,
        dev_loader: DataLoader,
        state_range_train: List[int],
        epochs: int = 1000,
        patience: int = 50,
        warmup: int = 50,
        clip_grad_threshhold: float = float("inf"),
        device: torch.device = torch.device("cpu"),
        trial=None,
    ):
        super().__init__(
            model,
            optimizer,
            loss_fct,
            train_loader,
            dev_loader,
            epochs,
            patience,
            warmup,
            clip_grad_threshhold,
            device,
            trial,
        )
        self.state_range_train = state_range_train

    def calc_objective(self, in_features, out_features):
        """Calculates the objective function for a given batch of data.

        Args:
            in_features (Tuple(torch.Tensor)): The input features. x_0 is at
                index 0, u is at index 1.
            out_features (torch.Tensor): The output features.

        Returns:
            torch.Tensor: The objective function value.
        """
        if self.move_batch:
            in_features_x0 = in_features[0].to(device=self.device)
            in_features_u = in_features[1].to(device=self.device)
            out_features = out_features.to(device=self.device)
            out = self.model(in_features_x0, in_features_u)
        else:
            out = self.model(in_features[0], in_features[1])
        loss = self.loss_fct(
            out[:, 1:, self.state_range_train[0] : self.state_range_train[1]],
            out_features[
                :, 1:, self.state_range_train[0] : self.state_range_train[1]
            ],
        )
        return loss


class StateSubsetSequenceTrainerFancy(
    SequenceTrainer
):  # should not be used anymore
    """This class implements the base trainer, but the objective is only cal-
    culated for a subset of the states. This is useful for training models

    Args:
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): The optimizer to be used for training.
        loss_fct (nn.Module): The loss function to be used for training.
        train_loader (DataLoader): The DataLoader for the training data.
        dev_loader (DataLoader): The DataLoader for the validation data.
        reduced_state_set (List[int]): The list of indices of the states to be
            used for training. Defaults to None, which means all states are
            used.
        epochs (int): The number of epochs to train for.
        patience (int): The number of epochs to wait before stopping training if
            no improvement is seen.
        warmup (int): The number of epochs to wait before starting to track
            improvements.
        clip_grad_threshhold (float): The threshold for gradient clipping.
            Defaults to float("inf") (no clipping).
        device (device): The device to use for training ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fct: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_loader: DataLoader,
        dev_loader: DataLoader,
        epochs: int = 1000,
        patience: int = 50,
        warmup: int = 50,
        clip_grad_threshhold: float = float("inf"),
        device: torch.device = torch.device("cpu"),
        state_set: List[int] = None,
        trial=None,
    ):
        super().__init__(
            model,
            optimizer,
            loss_fct,
            train_loader,
            dev_loader,
            epochs,
            patience,
            warmup,
            clip_grad_threshhold,
            device,
            trial,
        )
        self.state_set = state_set

    def calc_objective(self, in_features, out_features):
        """Calculates the objective function for a given batch of data.

        Args:
            in_features (Tuple(torch.Tensor)): The input features. x_0 is at
                index 0, u is at index 1.
            out_features (torch.Tensor): The output features.

        Returns:
            torch.Tensor: The objective function value.
        """
        if self.move_batch:
            in_features_x0 = in_features[0].to(device=self.device)
            in_features_u = in_features[1].to(device=self.device)
            out_features = out_features.to(device=self.device)
            out = self.model(in_features_x0, in_features_u)
        else:
            out = self.model(in_features[0], in_features[1])
        loss = self.loss_fct(
            out[:, 1:, self.state_set],
            out_features[:, 1:, self.state_set],
        )
        return loss
