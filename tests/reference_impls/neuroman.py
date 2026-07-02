import torch
from torch.utils.data import DataLoader
from neuromancer.dataset import DictDataset
import numpy as np
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss
from neuromancer.trainer import Trainer
from neuromancer.dynamics import integrators
from neuromancer.constraint import variable
from time import time_ns, process_time_ns

from neuralsysid.data.preprocessing import (
    resample_data_sets,
    normalize_data,
)


def create_batches(
    t,
    data_set,
    prediction_horizon,
    used_states,
    used_inputs,
    batch_size,
    data_set_type="train",
    test_loader=False,
    shuffle=True,
    device="cpu",
):

    # cut trajectories to multiple of prediction horizon
    n_inputs = len(used_inputs)
    n_states = len(used_states)
    n_samples = t.shape[1]
    n_batches = n_samples // (prediction_horizon + 1)
    n_samples = n_batches * (prediction_horizon + 1)

    t = t.squeeze()[:n_samples]
    X = data_set["X"].squeeze()[:n_samples, used_states]
    U = data_set["U"].squeeze()[:n_samples, used_inputs]

    # if one step prediction error, we need to account for inbetween steps of
    # sample trajectories
    if prediction_horizon == 1:
        X_ext = X[1:-1, :]
        U_ext = U[1:-1, :]
        t_ext = t[1:-1]

    if data_set_type == "train" or data_set_type == "dev" or test_loader:
        X = X.reshape(n_batches, prediction_horizon + 1, n_states)
        X = torch.tensor(X, dtype=torch.float32)

        U = U.reshape(n_batches, prediction_horizon + 1, n_inputs)
        U = torch.tensor(U, dtype=torch.float32)

        T = t.reshape(n_batches, prediction_horizon + 1)

        if prediction_horizon == 1:
            X_ext = X_ext.reshape(
                n_batches - 1, prediction_horizon + 1, n_states
            )
            X_ext = torch.tensor(X_ext, dtype=torch.float32)

            U_ext = U_ext.reshape(
                n_batches - 1, prediction_horizon + 1, n_inputs
            )
            U_ext = torch.tensor(U_ext, dtype=torch.float32)

            t_ext = t_ext.reshape(n_batches - 1, prediction_horizon + 1)

            # append the extended data to the data set
            X = torch.cat((X, X_ext), dim=0)
            U = torch.cat((U, U_ext), dim=0)

            T = np.concatenate((T, t_ext), axis=0)

        data_set = DictDataset(
            {"X": X, "xn": X[:, 0:1, :], "U": U}, name=data_set_type
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            collate_fn=data_set.collate_fn,
            shuffle=shuffle,
        )

    else:
        X = X.reshape(1, n_samples, n_states)
        X = torch.tensor(X, dtype=torch.float32)

        U = U.reshape(1, n_samples, n_inputs)
        U = torch.tensor(U, dtype=torch.float32)

        data_set = {"X": X, "xn": X[:, 0:1, :], "U": U}
        data_loader = None

        T = t.reshape(1, n_samples)

    return (
        T,
        data_set,
        data_loader,
    )


def create_batches_overlapping(
    t,
    data_set,
    prediction_horizon,
    used_states,
    used_inputs,
    batch_size,
    data_set_type="train",
    shuffle=True,
    device="cpu",
    overlap=1,
):

    n_samples = t.shape[1]

    t = t.squeeze(0)[:n_samples]
    X = data_set["X"].squeeze(0)[:, used_states]
    U = data_set["U"].squeeze(0)[:, used_inputs]

    X = X[:, np.newaxis, :]
    U = U[:, np.newaxis, :]
    T = t[:, np.newaxis]

    X_stack = X.copy()
    U_stack = U.copy()
    T_stack = T.copy()

    for shift in range(1, prediction_horizon + 1):
        X_shift = np.roll(X, -shift, axis=0)
        U_shift = np.roll(U, -shift, axis=0)
        T_shift = np.roll(T, -shift, axis=0)

        X_stack = np.concatenate((X_stack, X_shift), axis=1)
        U_stack = np.concatenate((U_stack, U_shift), axis=1)
        T_stack = np.concatenate((T_stack, T_shift), axis=1)

    X = X_stack[:-prediction_horizon, :, :]
    U = U_stack[:-prediction_horizon, :, :]
    T = T_stack[:-prediction_horizon, :]

    X = torch.tensor(X, dtype=torch.float32)
    U = torch.tensor(U, dtype=torch.float32)

    data_set = DictDataset(
        {"X": X, "xn": X[:, 0:1, :], "U": U}, name=data_set_type
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        collate_fn=data_set.collate_fn,
        shuffle=shuffle,
    )

    return (
        T,
        data_set,
        data_loader,
    )


def create_batches_overlapping_k(
    t,
    data_set,
    prediction_horizon,
    used_states,
    used_inputs,
    batch_size,
    data_set_type="train",
    shuffle=True,
    device="cpu",
    overlap=1,
):

    n_samples = t.shape[1]
    seq_len = prediction_horizon + 1

    t = t.squeeze(0)[:n_samples]
    X = data_set["X"].squeeze(0)[:, used_states]
    U = data_set["U"].squeeze(0)[:, used_inputs]

    X = X[np.newaxis, :, :]
    U = U[np.newaxis, :, :]
    T = t[np.newaxis, :]

    X_stack = X.copy()
    U_stack = U.copy()
    T_stack = T.copy()

    # for shift in range(1, prediction_horizon + 1):
    shift = seq_len - overlap
    while (
        shift + seq_len <= n_samples
    ):  # in range(prediction_horizon - 1 - overlap, prediction_horizon + overlap):
        X_shift = np.roll(X, -shift, axis=1)
        U_shift = np.roll(U, -shift, axis=1)
        T_shift = np.roll(T, -shift, axis=1)

        X_stack = np.concatenate((X_stack, X_shift), axis=0)
        U_stack = np.concatenate((U_stack, U_shift), axis=0)
        T_stack = np.concatenate((T_stack, T_shift), axis=0)

        shift += seq_len - overlap

    X = X_stack[:, :seq_len, :]
    U = U_stack[:, :seq_len, :]
    T = T_stack[:, :seq_len]

    X = torch.tensor(X, dtype=torch.float32)
    U = torch.tensor(U, dtype=torch.float32)

    data_set = DictDataset(
        {"X": X, "xn": X[:, 0:1, :], "U": U}, name=data_set_type
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        collate_fn=data_set.collate_fn,
        shuffle=shuffle,
    )

    return (
        T,
        data_set,
        data_loader,
    )


def generate_batch_sets(
    t_vecs,
    data_sets,
    prediction_horizon,
    used_states,
    used_inputs,
    batch_size,
    data_set_types=["train", "dev", "test"],
    test_loader=False,
    shuffle=True,
    device="cpu",
):
    batch_sets = []
    for t, data_set, data_set_type in zip(t_vecs, data_sets, data_set_types):
        batch_sets.append(
            create_batches(
                t,
                data_set,
                prediction_horizon,
                used_states,
                used_inputs,
                batch_size,
                data_set_type=data_set_type,
                test_loader=test_loader,
                shuffle=shuffle,
                device=device,
            )
        )

    return batch_sets


def generate_batch_sets_overlapping(
    t_vecs,
    data_sets,
    prediction_horizon,
    used_states,
    used_inputs,
    batch_size,
    data_set_types=["train", "dev", "test"],
    shuffle=True,
    device="cpu",
    overlap=1,
):
    batch_sets = []
    for t, data_set, data_set_type in zip(t_vecs, data_sets, data_set_types):
        batch_sets.append(
            create_batches_overlapping(
                t,
                data_set,
                prediction_horizon,
                used_states,
                used_inputs,
                batch_size,
                data_set_type=data_set_type,
                shuffle=shuffle,
                device=device,
                overlap=overlap,
            )
        )

    return batch_sets


def generate_batch_sets_overlapping_k(
    t_vecs,
    data_sets,
    prediction_horizon,
    used_states,
    used_inputs,
    batch_size,
    data_set_types=["train", "dev", "test"],
    shuffle=True,
    device="cpu",
    overlap=1,
):
    batch_sets = []
    for t, data_set, data_set_type in zip(t_vecs, data_sets, data_set_types):
        batch_sets.append(
            create_batches_overlapping_k(
                t,
                data_set,
                prediction_horizon,
                used_states,
                used_inputs,
                batch_size,
                data_set_type=data_set_type,
                shuffle=shuffle,
                device=device,
                overlap=overlap,
            )
        )

    return batch_sets


def learn_dynamics(
    model_configuration,
    sim_configuration,
    hyperparameters,
    chip_type,
    dataSets,
    log_mlf=False,
):

    if "gpu" in chip_type:
        if torch.cuda.is_available():
            if "0" in chip_type:
                device = torch.device("cuda:0")
            elif "1" in chip_type:
                device = torch.device("cuda:1")
            else:
                device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            raise ValueError("No GPU available")
    else:
        device = torch.device("cpu")

    torch.manual_seed(0)  # necessary for reproducible results

    run_name = "configuration_" + str(model_configuration["config_id"])

    print("testing model: ")
    for hyperparameter in hyperparameters:
        print(hyperparameter, model_configuration[hyperparameter])

    # TODO: possibly exchange this in the future so sim config does not need to be
    # used here
    nx = len(model_configuration["used_outputs"])
    nu = len(model_configuration["used_controls"])

    # TODO: possibly exchange this in the future so sim config does not need to be
    # used here (resample with interpolator or get sample time from data)
    dataSets = resample_data_sets(
        dataSets,
        t_sample=sim_configuration["T_sample"],
        t_step=model_configuration["t_step"],
    )

    tTrain, dataSetTrain = dataSets[0]
    tDev, dataSetDev = dataSets[1]
    tTest, dataSetTest = dataSets[2]

    # TODO: later export dataStats so it is known what normalization has to be applied
    # to data before it can be used for feed-forward calculation
    dataSetTrainNormalized, dataStatsTrain = normalize_data(dataSetTrain)
    dataSetDevNormalized, _ = normalize_data(dataSetDev, dataStatsTrain)
    dataSetTestNormalized, _ = normalize_data(dataSetTest, dataStatsTrain)
    dataSetTrainNormalized["X"] = dataSetTrainNormalized["Y"]
    dataSetDevNormalized["X"] = dataSetDevNormalized["Y"]
    dataSetTestNormalized["X"] = dataSetTestNormalized["Y"]

    # TODO: change so sim configuration does not need to be used here in the
    # future - maybe just overgive the names of inputs and states for a
    # learning task
    batchSets = generate_batch_sets_overlapping(
        [tTrain, tDev, tTest],
        [
            dataSetTrainNormalized,
            dataSetDevNormalized,
            dataSetTestNormalized,
        ],
        model_configuration["seq_len"] - 1,
        model_configuration["used_outputs"],
        model_configuration["used_controls"],
        batch_size=model_configuration["batch_size"],
        data_set_types=["train", "dev", "test"],
        shuffle=False,  # TODO: make this part of configuration
        device=device,
    )

    tTrainBatches, _, dataSetTrainLoader = batchSets[0]
    _, _, dataSetDevLoader = batchSets[1]
    _, _, dataSetTestLoader = batchSets[2]

    nsteps = tTrainBatches.shape[1]

    if model_configuration["dynamics"] == "fxgu":
        if model_configuration["stepper"] == "discrete":
            dynamics_model = construct_fxgu(
                nx=nx,
                nu=nu,
                nsteps=nsteps,
                Ts=1,
                nlayersA=model_configuration["n_layers_A"],
                nlayersB=model_configuration["n_layers_B"],
                nNeuronsA=model_configuration["n_hidden_A"],
                nNeuronsB=model_configuration["n_hidden_B"],
                activation=model_configuration["act_fct"],
                integrator=None,
            )
        elif model_configuration["stepper"] == "residual":
            dynamics_model = construct_fxgu(
                nx=nx,
                nu=nu,
                nsteps=nsteps,
                Ts=1,
                nlayersA=model_configuration["n_layers_A"],
                nlayersB=model_configuration["n_layers_B"],
                nNeuronsA=model_configuration["n_hidden_A"],
                nNeuronsB=model_configuration["n_hidden_B"],
                activation=model_configuration["act_fct"],
                integrator=integrators.Euler,
            )
        elif model_configuration["stepper"] == "euler":
            dynamics_model = construct_fxgu(
                nx=nx,
                nu=nu,
                nsteps=nsteps,
                Ts=model_configuration["t_step"],
                nlayersA=model_configuration["n_layers_A"],
                nlayersB=model_configuration["n_layers_B"],
                nNeuronsA=model_configuration["n_hidden_A"],
                nNeuronsB=model_configuration["n_hidden_B"],
                activation=model_configuration["act_fct"],
                integrator=integrators.Euler,
            )
        elif model_configuration["stepper"] == "rk4":
            dynamics_model = construct_fxgu(
                nx=nx,
                nu=nu,
                nsteps=nsteps,
                Ts=model_configuration["t_step"],
                nlayersA=model_configuration["n_layers_A"],
                nlayersB=model_configuration["n_layers_B"],
                nNeuronsA=model_configuration["n_hidden_A"],
                nNeuronsB=model_configuration["n_hidden_B"],
                activation=model_configuration["act_fct"],
                integrator=RK4_three_eightth,
            )
        else:
            raise ValueError(
                "Chosen stepper not implemented for chosen dynamics in nm."
            )
    elif model_configuration["dynamics"] == "fxu":
        if model_configuration["stepper"] == "discrete":
            dynamics_model = construct_fxu(
                nx=nx,
                nu=nu,
                nsteps=nsteps,
                Ts=1,
                nLayers=model_configuration["n_layers_dyn"],
                nNeurons=model_configuration["n_hidden_dyn"],
                activation=model_configuration["act_fct"],
                integrator=None,
            )
        elif model_configuration["stepper"] == "residual":
            dynamics_model = construct_fxu(
                nx=nx,
                nu=nu,
                nsteps=nsteps,
                Ts=1.0,
                nLayers=model_configuration["n_layers_dyn"],
                nNeurons=model_configuration["n_hidden_dyn"],
                activation=model_configuration["act_fct"],
                integrator=integrators.Euler,
            )
        elif model_configuration["stepper"] == "euler":
            dynamics_model = construct_fxu(
                nx=nx,
                nu=nu,
                nsteps=nsteps,
                Ts=model_configuration["t_step"],
                nLayers=model_configuration["n_layers_dyn"],
                nNeurons=model_configuration["n_hidden_dyn"],
                activation=model_configuration["act_fct"],
                integrator=integrators.Euler,
            )
        elif model_configuration["stepper"] == "rk4":
            dynamics_model = construct_fxu(
                nx=nx,
                nu=nu,
                nsteps=nsteps,
                Ts=model_configuration["t_step"],
                nLayers=model_configuration["n_layers_dyn"],
                nNeurons=model_configuration["n_hidden_dyn"],
                activation=model_configuration["act_fct"],
                integrator=RK4_three_eightth,
            )
        else:
            raise ValueError(
                "Chosen stepper not implemented for chosen dynamics in nm."
            )

    x = variable("X")[:, 1:, :]
    xhat = variable("xn")[:, 1:-1, :]

    objectives = []
    step_losses = [-1, 1]
    weight_step_losses = [
        model_configuration["reference_loss_weight"],
        model_configuration["one_step_loss_weight"],
    ]
    for horizon, weight in zip(step_losses, weight_step_losses):
        objectives.append(
            define_loss(x, xhat, horizon, weight, loss_type="mse")
        )

    constraints = []
    loss = PenaltyLoss(objectives, constraints)

    # optimization problem
    problem = Problem([dynamics_model], loss)

    # x = variable("X")[:, 1:, :]
    # xhat = variable("xn")[:, 1:, :]
    # objectives = [define_loss(x, xhat, -1, 1)]
    # constraints = []
    # loss = PenaltyLoss(objectives, constraints)
    # problem = Problem([dynamics_model], loss)

    # training
    optimizer = torch.optim.Adam(
        problem.parameters(), lr=model_configuration["learning_rate"]
    )

    if "mps" in device.type or "cuda" in device.type:
        problem.to(device)

    trainer = Trainer(
        problem,
        dataSetTrainLoader,
        dataSetDevLoader,
        dataSetTestLoader,
        optimizer,
        patience=model_configuration["patience"],
        warmup=model_configuration["warmup"],
        epochs=model_configuration["n_epochs"],
        eval_metric="mean_dev_loss",
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        device=device,
        clip=model_configuration["grad_clip"],
    )
    t0_train = time_ns()
    t0_train_process = process_time_ns()
    best_model = trainer.train()
    t1_train_process = process_time_ns()
    t1_train = time_ns()
    dur_training_process = (t1_train_process - t0_train_process) * 1e-9
    dur_training_process = (
        dur_training_process / model_configuration["n_epochs"]
    )
    dur_training = (t1_train - t0_train) * 1e-9
    dur_training = dur_training / model_configuration["n_epochs"]

    print("Training time per epoch: ", dur_training)
    print("Process time per epoch: ", dur_training_process)

    problem.load_state_dict(best_model)

    output = trainer.test(best_model)
    mean_train_loss = output["mean_train_loss"]
    mean_dev_loss = output["mean_dev_loss"]
    mean_test_loss = output["mean_test_loss"]

    return dynamics_model, mean_test_loss


class RK4_three_eightth(integrators.Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """

        :param block: (nn.Module) A state transition model.
        :param h: (float) integration step size
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, *args):
        h = self.h
        # k1 = self.block(x, *args)
        # k2 = self.block(x + h * k1 * 1 / 3, *args)
        # k3 = self.block(x + h * (k2 - k1 * 1 / 3), *args)
        # k4 = self.block(x + h * (k1 - k2 + k3), *args)
        # return x + (k1 + 3 * (k2 + k3) + k4) * h * 0.125

        k1 = self.block(x, *args)
        k2 = self.block(x + h * k1 * _one_third, *args)
        k3 = self.block(x + h * (k2 - k1 * _one_third), *args)
        k4 = self.block(x + h * (k1 - k2 + k3), *args)
        return x + (k1 + 3 * (k2 + k3) + k4) * h * 0.125


_one_third = 1 / 3


from neuromancer import blocks
from neuromancer.system import Node, System
import torch
import torch.nn as nn
from neuromancer.dynamics import integrators


class FxGu(nn.Module):
    """
    Baseline class for (neural) state space model (SSM)
    Implements discrete-time dynamical system:
        x_k+1 = fx(x_k) + fu(u_k)
    with variables:
        x_k - states
        u_k - control inputs
    """

    def __init__(self, fx, fu, nx, nu):
        super().__init__()
        self.fx, self.fu = fx, fu
        self.nx, self.nu = nx, nu
        self.in_features, self.out_features = nx + nu, nx

    def forward(self, x, u, d=None):
        """
        :param x: (torch.Tensor, shape=[batchsize, nx])
        :param u: (torch.Tensor, shape=[batchsize, nu])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        # state space model
        x = self.fx(x) + self.fu(u)
        return x


def construct_fxu(
    nx,
    nu,
    nsteps,
    Ts,
    nLayers=3,
    nNeurons=80,
    activation=nn.ReLU,
    integrator=integrators.RK4,
    torqdiffmethod=None,
):

    fx = blocks.MLP(
        nx + nu,
        nx,
        bias=True,
        linear_map=torch.nn.Linear,
        nonlin=activation,
        hsizes=nLayers * [nNeurons],
    )

    if integrator is not None:
        if torqdiffmethod is not None:
            stepper = integrator(fx, h=Ts, method=torqdiffmethod)
        else:
            stepper = integrator(fx, h=Ts)

        model = Node(stepper, ["xn", "U"], ["xn"], name="NODE")
        dynamics_model = System([model], name="system", nsteps=int(nsteps))
    else:
        model = Node(fx, ["xn", "U"], ["xn"], name="NODE")
        dynamics_model = System([model], name="system", nsteps=int(nsteps))

    return dynamics_model


def construct_fxgu(
    nx,
    nu,
    nsteps,
    Ts,
    nlayersA=2,
    nlayersB=2,
    nNeuronsA=40,
    nNeuronsB=40,
    activation=nn.ReLU,
    integrator=integrators.RK4,
    torqdiffmethod=None,
):

    A = blocks.MLP(
        nx,
        nx,
        bias=True,
        linear_map=torch.nn.Linear,
        nonlin=activation,
        hsizes=nlayersA * [nNeuronsA],
    )
    B = blocks.MLP(
        nu,
        nx,
        bias=True,
        linear_map=torch.nn.Linear,
        nonlin=activation,
        hsizes=nlayersB * [nNeuronsB],
    )
    ssm = FxGu(A, B, nx, nu)

    if integrator is not None:
        if torqdiffmethod is not None:
            stepper = integrator(ssm, h=Ts, method=torqdiffmethod)
        else:
            stepper = integrator(ssm, h=Ts)

        model = Node(stepper, ["xn", "U"], ["xn"], name="NODE")
        dynamics_model = System(
            [model],
            name="system",
            nsteps=int(nsteps),
        )
    else:
        model = Node(ssm, ["xn", "U"], ["xn"], name="NODE")
        dynamics_model = System(
            [model],
            name="system",
            nsteps=int(nsteps),
        )
    return dynamics_model


def define_loss(
    trueVar, estVar, horizon, weight, loss_type="mse", weight_variables=None
):
    if horizon == -1:
        loss_slice = estVar == trueVar
        step_loss_name = "referenceloss"
    else:
        loss_slice = estVar[:, :horizon, :] == trueVar[:, :horizon, :]
        step_loss_name = str(horizon) + "steploss"

    if loss_type == "mse":
        loss = weight * (loss_slice) ^ 2
    else:
        raise ValueError("Invalid loss type")

    loss.name = loss_type + "_" + step_loss_name + "_" + "weight" + str(weight)

    return loss
