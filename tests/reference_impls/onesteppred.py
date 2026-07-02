from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor
from torch import nn
from typing import Dict, List, Tuple, Any, Callable
import numpy
import time

from neuralsysid.data import datasets, preparation, preprocessing
from neuralsysid.models import neuralnetworks, statedynamics, timesteppers
from neuralsysid.training import trainer


class OneStepTrajectoriesDataset(Dataset):
    """Dataset for one-step ahead predictions.

    Args:
        xkm1 (Tensor): The state at time k-1.
        ukm1 (Tensor): The input at time k-1.
        xk (Tensor): The state at time k.
    """

    def __init__(self, xkm1: Tensor, ukm1: Tensor, xk: Tensor):
        super().__init__()
        self.len = xk.shape[0]
        self.in_features = torch.cat([xkm1, ukm1], dim=1)
        self.out_features = xk

        assert xk.shape == xkm1.shape, "state dimensions do not match"
        assert (
            ukm1.shape[0] == xk.shape[0]
        ), "number of samples between xk and ukm1 do not match"

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
        return self.in_features[idx], self.out_features[idx]


def create_batches(
    t: numpy.ndarray,
    data_set: Dict[str, numpy.ndarray],
    used_states: List[int],
    used_inputs: List[int],
    batch_size: int,
    shuffle: bool = True,
) -> Tuple[numpy.ndarray, OneStepTrajectoriesDataset, DataLoader]:
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
        Tuple[numpy.ndarray, OneStepTrajectoriesDataset, DataLoader]: time
            vector, dataset and dataloader
    """

    t = t.squeeze(0)
    x = data_set["X"].squeeze(0)[:, used_states]
    u = data_set["U"].squeeze(0)[:, used_inputs]

    t = t[:]
    x_in = x[:-1, :]
    u_in = u[:-1, :]
    x_out = x[1:, :]

    x_in = torch.tensor(x_in, dtype=torch.float32)
    u_in = torch.tensor(u_in, dtype=torch.float32)
    x_out = torch.tensor(x_out, dtype=torch.float32)

    data_set = OneStepTrajectoriesDataset(x_in, u_in, x_out)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return (
        t,
        data_set,
        data_loader,
    )


def generate_batch_sets(
    t_vecs: List[numpy.ndarray],
    data_sets: List[Dict[str, numpy.ndarray]],
    used_states: List[int],
    used_inputs: List[int],
    batch_size: int,
    shuffle: bool = True,
) -> Tuple[Tuple[numpy.ndarray, OneStepTrajectoriesDataset, DataLoader]]:
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
            create_batches(
                t,
                data_set,
                used_states,
                used_inputs,
                batch_size,
                shuffle=shuffle,
            )
        )

    return batch_sets


def cyclic_computation(
    model: timesteppers.TimeStepper,
    dataset: OneStepTrajectoriesDataset,
    prediction_horizon: int,
) -> torch.Tensor:
    """This function computes the trajectory of an identified model by repeated-
    ly solving the model for the next time step using the last prediction as the
    initial state.

    Args:
        model (TimeStepper): The identified model to be used for prediction.
        dataset (OneStepTrajectoriesDataset): The dataset containing the input-
            output features.
        prediction_horizon (int): The number of time steps to predict into the
            future.

    Returns:
        torch.Tensor: The predicted trajectory of the system.
    """

    n_states = model.dynamics.n_states
    n_inputs = model.dynamics.n_inputs
    n_samples = len(dataset)

    assert prediction_horizon > 0, "Prediction horizon must be greater than 0"
    assert (
        n_samples >= prediction_horizon - 1
    ), "Dataset must be greater than prediction horizon"
    assert (
        n_states + n_inputs == dataset[0][0].shape[0]
    ), "Number of states and inputs must match dataset"

    sol = torch.zeros((prediction_horizon + 1, n_states))
    sol[0, :] = dataset[0][0][:n_states].detach()

    for i in range(1, prediction_horizon + 1):
        in_feature = dataset[i - 1][0]
        x_in = sol[i - 1, :]
        u_in = in_feature[n_states:].detach()
        # in_feature = torch.cat((x_in, u_in), axis=0)[None, :]
        sol[i, :] = model(x_in, u_in)

    return sol


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

    batch_sets = generate_batch_sets(
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
        dynamics_model = timesteppers.Discrete(rhs)
    elif model_configuration["stepper"] == "residual":
        dynamics_model = timesteppers.Residual(rhs)
    elif model_configuration["stepper"] == "euler":
        dynamics_model = timesteppers.Euler(rhs, model_configuration["t_step"])
    elif model_configuration["stepper"] == "rk4":
        dynamics_model = timesteppers.RungeKutta4(
            rhs, model_configuration["t_step"]
        )
    elif model_configuration["stepper"] == "dopri5":
        dynamics_model = timesteppers.Dopri5(
            rhs, model_configuration["t_step"]
        )
    else:
        raise ValueError("Stepper not defined")

    optimizer = torch.optim.Adam(
        dynamics_model.parameters(), lr=model_configuration["learning_rate"]
    )

    model_trainer = OneStepTrainer(
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

    t0_train = time.time_ns()
    t0_train_process = time.process_time_ns()
    model_trainer.train()
    t1_train_process = time.process_time_ns()
    t1_train = time.time_ns()
    dur_training_process = (t1_train_process - t0_train_process) * 1e-9
    dur_training_process = (
        dur_training_process / model_configuration["n_epochs"]
    )
    dur_training = (t1_train - t0_train) * 1e-9
    dur_training = dur_training / model_configuration["n_epochs"]

    print("Training time per epoch: ", dur_training)
    print("Process time per epoch: ", dur_training_process)

    dynamics_model.load_state_dict(model_trainer.best_model_state_dict)

    mean_train_loss = model_trainer.evaluate(dataloader_train)
    mean_dev_loss = model_trainer.evaluate(dataloader_dev)
    mean_test_loss = model_trainer.evaluate(dataloader_test)

    return dynamics_model, mean_test_loss


class OneStepTrainer(trainer.BaseTrainer):
    """This class represents the base class for training procdures.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): The optimizer to be used for training.
        loss_fct (nn.Module): The loss function to be used for training.
        train_loader (DataLoader): The DataLoader for the training data.
        dev_loader (DataLoader): The DataLoader for the validation data.
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
        optimizer: nn.Module,
        loss_fct: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_loader: DataLoader,
        dev_loader: DataLoader,
        epochs: int = 1000,
        patience: int = 250,
        warmup: int = 100,
        clip_grad_threshhold: float = float("inf"),
        device: torch.device = torch.device("cpu"),
        trial: Any = None,
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
            in_features (torch.Tensor): The input features.
            out_features (torch.Tensor): The output features.

        Returns:
            torch.Tensor: The objective function value.
        """
        n_states = self.model.dynamics.n_states

        if self.move_batch:
            in_features = in_features.to(device=self.device)
            out_features = out_features.to(device=self.device)
        out = self.model(
            in_features[..., :n_states], in_features[..., n_states:]
        )
        loss = self.loss_fct(out, out_features)
        return loss
