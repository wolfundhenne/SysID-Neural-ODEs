"""Training pipeline for learning dynamical system models."""

from dataclasses import dataclass
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader

from configurations.utils import resolve_config_references
from neuralsysid.data import preparation, preprocessing
from neuralsysid.models import (
    neuralnetworks,
    outputdecoders,
    predictors,
    stateencoders,
    statedynamics,
    timesteppers,
)
from neuralsysid.utils import logging
from . import trainer


@dataclass(frozen=True)
class ModelDimensions:
    """Container for the relevant model dimensions."""

    n_outputs: int
    n_controls: int
    latent_state_dim: int
    n_latent_states: int


@dataclass(frozen=True)
class PreparedData:
    """Container for prepared dataloaders and normalization statistics."""

    train_loader: DataLoader
    dev_loader: DataLoader
    test_loader: DataLoader
    train_stats: Optional[Dict[str, np.ndarray]]


def _resolve_activation(activation: str | type[nn.Module]) -> type[nn.Module]:
    if not isinstance(activation, str):
        return activation

    activations = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "softplus": nn.Softplus,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }
    return activations[activation]


def _validate_optional_inputs(
    model: Optional[nn.Module],
    data_stats_train: Optional[Dict[str, np.ndarray]],
) -> None:
    """Validate the optional reuse inputs for model and normalization stats.

    Args:
        model (Optional[nn.Module]): Existing model to continue training with.
        data_stats_train (Optional[Dict[str, np.ndarray]]): Normalization
            statistics of the training data.
    """
    if bool(model is None) ^ bool(data_stats_train is None):
        raise ValueError(
            "Either both or neither of model and data_stats_train should be "
            + "provided"
        )


def _resolve_device(chip_type: str) -> torch.device:
    """Resolve the torch device from the configured chip type.

    Args:
        chip_type (str): Device specifier, e.g. ``cpu`` or ``cuda:0``.

    Returns:
        torch.device: Resolved torch device.
    """
    if "cuda" in chip_type:
        if torch.cuda.is_available():
            return torch.device(chip_type)
        raise ValueError(f"CUDA device {chip_type} not available")

    if "mps" in chip_type:
        if torch.backends.mps.is_available():
            return torch.device(chip_type)
        raise ValueError(f"MPS device {chip_type} not available")

    return torch.device(chip_type)


def _log_configuration(
    model_config: Dict[str, Any], hyperparameters: List[str]
) -> None:
    """Print the active hyperparameter configuration.

    Args:
        model_config (Dict[str, Any]): Configuration of the model.
        hyperparameters (List[str]): Hyperparameters to display.
    """
    print("testing model: ")
    for hyperparameter in hyperparameters:
        value = model_config
        for key in hyperparameter.split("."):
            value = value[key]
        print(hyperparameter, value)


def _get_model_dimensions(
    model_config: Dict[str, Any],
) -> ModelDimensions:
    """Extract the relevant model dimensions from the configuration.

    Args:
        model_config (Dict[str, Any]): Configuration of the model.

    Returns:
        ModelDimensions: Relevant state, input and output dimensions.
    """

    n_outputs = len(model_config["data"]["outputs"])
    n_controls = len(model_config["data"]["controls"])
    n_latent_states = model_config["model"]["dynamics"][
        "latent_state_dim"
    ] - len(model_config["data"]["outputs_as_latent_states"])

    if model_config["model"]["dynamics"]["type"] == "graph":
        n_controls = n_controls // model_config["model"]["graph"]["n_nodes"]
        n_outputs = n_outputs // model_config["model"]["graph"]["n_nodes"]

    return ModelDimensions(
        n_outputs=n_outputs,
        n_controls=n_controls,
        latent_state_dim=model_config["model"]["dynamics"]["latent_state_dim"],
        n_latent_states=n_latent_states,
    )


def prepare_data(
    model_config: Dict[str, Any],
    chip_type: str,
    data_sets: List[Tuple[np.ndarray, Dict[str, np.ndarray]]],
    normalize_data: bool,
    data_stats_train: Optional[Dict[str, np.ndarray]] = None,
) -> PreparedData:
    """Resample, normalize and batch the train, dev and test data.

    Args:
        model_config (Dict[str, Any]): Configuration of the model.
        chip_type (str): Device specifier used to determine worker count.
        data_sets (List[Tuple[np.ndarray, Dict[str, np.ndarray]]]): Train, dev
            and test datasets with their time vectors.
        normalize_data (bool): Whether to normalize the datasets.
        data_stats_train (Optional[Dict[str, np.ndarray]]): Existing training
            statistics for normalization.

    Returns:
        PreparedData: Train, dev and test dataloaders and the training
        normalization statistics.
    """

    t_flat = np.asarray(data_sets[0][0]).reshape(-1)
    sample_time = float(t_flat[1] - t_flat[0])
    data_sets = preprocessing.resample_data_sets(
        data_sets,
        t_sample=sample_time,
        t_step=model_config["data"]["t_step"],
    )

    t_train, data_set_train = data_sets[0]
    t_dev, data_set_dev = data_sets[1]
    t_test, data_set_test = data_sets[2]

    if normalize_data:
        if data_stats_train is None:
            data_set_train_normalized, data_stats_train = (
                preprocessing.normalize_data(data_set_train)
            )
        else:
            data_set_train_normalized, _ = preprocessing.normalize_data(
                data_set_train, data_stats_train
            )
        data_set_dev_normalized, _ = preprocessing.normalize_data(
            data_set_dev, data_stats_train
        )
        data_set_test_normalized, _ = preprocessing.normalize_data(
            data_set_test, data_stats_train
        )
    else:
        data_set_train_normalized = data_set_train
        data_set_dev_normalized = data_set_dev
        data_set_test_normalized = data_set_test

    workers = 8 if "cuda" in chip_type else 0
    sequence_dataloaders = preparation.build_sequence_dataloaders(
        [t_train, t_dev, t_test],
        [
            data_set_train_normalized,
            data_set_dev_normalized,
            data_set_test_normalized,
        ],
        model_config["data"]["outputs"],
        model_config["data"]["controls"],
        batch_size=model_config["train"]["batch_size"],
        shuffle=model_config["train"]["shuffle"],
        seq_len=model_config["data"]["seq_len"],
        overlap=model_config["data"]["overlap"],
        historic_seq_len=model_config["data"]["hist_seq_len"],
        encode=model_config["model"]["encoder"]["type"] != "identity",
        workers=workers,
        drop_last=model_config["train"]["drop_last"],
    )

    _, _, dataloader_train = sequence_dataloaders[0]
    _, _, dataloader_dev = sequence_dataloaders[1]
    _, _, dataloader_test = sequence_dataloaders[2]

    return PreparedData(
        train_loader=dataloader_train,
        dev_loader=dataloader_dev,
        test_loader=dataloader_test,
        train_stats=data_stats_train,
    )


def _create_dynamics(
    model_config: Dict[str, Any],
    dims: ModelDimensions,
    device: torch.device,
) -> statedynamics.DynamicsFunction:
    """Create the right-hand-side dynamics model.

    Args:
        model_config (Dict[str, Any]): Configuration of the model.
        dims (ModelDimensions): Relevant model dimensions.
        device (torch.device): Device used by graph dynamics.

    Returns:
        statedynamics.DynamicsFunction: Dynamics right-hand-side model.
    """
    if model_config["model"]["dynamics"]["type"] == "fxgu":
        f_dyn = neuralnetworks.MultiLayerPerceptron(
            n_inputs=dims.latent_state_dim,
            n_outputs=dims.latent_state_dim,
            n_layers=model_config["model"]["dynamics"]["n_layers_A"],
            n_hidden=model_config["model"]["dynamics"]["n_hidden_A"],
            activation=_resolve_activation(
                model_config["model"]["dynamics"]["act_fct"]
            ),
            dropout=model_config["model"]["dynamics"]["dropout"],
        )
        g_dyn = neuralnetworks.MultiLayerPerceptron(
            n_inputs=dims.n_controls,
            n_outputs=dims.latent_state_dim,
            n_layers=model_config["model"]["dynamics"]["n_layers_B"],
            n_hidden=model_config["model"]["dynamics"]["n_hidden_B"],
            activation=_resolve_activation(
                model_config["model"]["dynamics"]["act_fct"]
            ),
            dropout=model_config["model"]["dynamics"]["dropout"],
        )
        return statedynamics.FxGuDynamics(
            n_states=dims.latent_state_dim,
            n_inputs=dims.n_controls,
            f_dynamics=f_dyn,
            g_dynamics=g_dyn,
        )

    if model_config["model"]["dynamics"]["type"] == "fxu":
        dynamics_network = neuralnetworks.MultiLayerPerceptron(
            n_inputs=dims.latent_state_dim + dims.n_controls,
            n_outputs=dims.latent_state_dim,
            n_layers=model_config["model"]["dynamics"]["n_layers"],
            n_hidden=model_config["model"]["dynamics"]["n_hidden"],
            activation=_resolve_activation(
                model_config["model"]["dynamics"]["act_fct"]
            ),
            dropout=model_config["model"]["dynamics"]["dropout"],
        )
        return statedynamics.FxuDynamics(
            n_states=dims.latent_state_dim,
            n_inputs=dims.n_controls,
            dynamics=dynamics_network,
        )

    if model_config["model"]["dynamics"]["type"] == "graph":
        n_dynamic_context = model_config["model"]["dynamics"]["n_context"]
        n_node_embedding = model_config["model"]["graph"]["n_node_embedding"]
        n_edge_embedding = model_config["model"]["graph"]["n_edge_embedding"]

        if model_config["model"]["graph"]["additive_messages"]:
            in_dim_node = (
                dims.latent_state_dim
                + dims.n_controls
                + n_node_embedding
                + n_dynamic_context
            )
        else:
            in_dim_node = (
                dims.latent_state_dim
                + dims.n_controls
                + n_node_embedding
                + n_dynamic_context
                + model_config["model"]["graph"]["n_msg"]
            )

        in_dim_edge = 2 * dims.latent_state_dim + n_edge_embedding

        node_dyn = neuralnetworks.MultiLayerPerceptron(
            n_inputs=in_dim_node,
            n_outputs=dims.latent_state_dim,
            n_hidden=model_config["model"]["dynamics"]["n_hidden"],
            n_layers=model_config["model"]["dynamics"]["n_layers"],
            activation=_resolve_activation(
                model_config["model"]["dynamics"]["act_fct"]
            ),
            dropout=model_config["model"]["dynamics"]["dropout"],
        )
        edge_dyn = neuralnetworks.MultiLayerPerceptron(
            n_inputs=in_dim_edge,
            n_outputs=model_config["model"]["graph"]["n_msg"],
            n_hidden=model_config["model"]["graph"]["n_hidden_msg"],
            n_layers=model_config["model"]["graph"]["n_layers_msg"],
            activation=_resolve_activation(
                model_config["model"]["dynamics"]["act_fct"]
            ),
            dropout=model_config["model"]["graph"]["dropout_msg"],
        )
        return statedynamics.GraphDynamics(
            n_states_node=dims.latent_state_dim,
            n_inputs_node=dims.n_controls,
            node_function=node_dyn,
            edge_function=edge_dyn,
            n_msg=model_config["model"]["graph"]["n_msg"],
            n_ctx=n_dynamic_context,
            n_node_embedding=n_node_embedding,
            n_edge_embedding=n_edge_embedding,
            n_nodes=model_config["model"]["graph"]["n_nodes"],
            adjacency=model_config["model"]["graph"]["adjacency_list"],
            additive_messages=model_config["model"]["graph"][
                "additive_messages"
            ],
            device=device,
            share_edge_embeddings=model_config["model"]["graph"][
                "share_edge_embeddings"
            ],
            degree_normalization=model_config["model"]["graph"][
                "degree_normalization"
            ],
        )

    raise ValueError("Dynamics not defined")


def _create_time_stepper(
    model_config: Dict[str, Any],
    dynamics: statedynamics.DynamicsFunction,
    device: torch.device,
) -> timesteppers.TimeStepper:
    """Create the configured time stepping model.

    Args:
        model_config (Dict[str, Any]): Configuration of the model.
        dynamics (statedynamics.DynamicsFunction): Dynamics right-hand-side
            model.
        device (torch.device): Device used by selected steppers.

    Returns:
        timesteppers.TimeStepper: Configured time stepper.
    """

    if model_config["model"]["dynamics"]["stepper"] == "discrete":
        return timesteppers.Discrete(dynamics)
    if model_config["model"]["dynamics"]["stepper"] == "residual":
        return timesteppers.Residual(dynamics)
    if model_config["model"]["dynamics"]["stepper"] == "euler":
        return timesteppers.Euler(dynamics, model_config["data"]["t_step"])
    if model_config["model"]["dynamics"]["stepper"] == "rk4":
        return timesteppers.RungeKutta4(
            dynamics, model_config["data"]["t_step"], device=device
        )
    if model_config["model"]["dynamics"]["stepper"] == "dopri5":
        return timesteppers.Dopri5(dynamics, model_config["data"]["t_step"])
    raise ValueError("Stepper not defined")


def _create_predictor(
    model_config: Dict[str, Any],
    device: torch.device,
) -> predictors.StateRollout:
    """Create the predictor from dynamics, stepper and rollout horizon.

    Args:
        model_config (Dict[str, Any]): Configuration of the model.
        device (torch.device): Device used for model creation.

    Returns:
        predictors.StateRollout: Unrolled prediction model.
    """
    dims = _get_model_dimensions(model_config)
    dynamics = _create_dynamics(model_config, dims, device)
    time_stepper = _create_time_stepper(model_config, dynamics, device)
    return predictors.StateRollout(
        time_stepper, model_config["data"]["seq_len"] - 1, device=device
    )


def _get_encoder_output_dim(
    model_config: Dict[str, Any], dims: ModelDimensions
) -> int:
    """Determine the encoder output dimension.

    Args:
        model_config (Dict[str, Any]): Configuration of the model.
        dims (ModelDimensions): Relevant model dimensions.

    Returns:
        int: Output dimension of the encoder core.
    """
    if model_config["model"]["dynamics"]["type"] == "graph":
        return (
            dims.n_latent_states
            + model_config["model"]["dynamics"]["n_context"]
        )
    return dims.n_latent_states


def _create_encoder_core(
    model_config: Dict[str, Any],
    dims: ModelDimensions,
) -> Optional[nn.Module]:
    """Create the raw encoder core before state encoding wrapping.

    Args:
        model_config (Dict[str, Any]): Configuration of the model.
        dims (ModelDimensions): Relevant model dimensions.

    Returns:
        Optional[nn.Module]: Encoder core or ``None`` for identity encoding.
    """
    encoder_type = model_config["model"]["encoder"]["type"]
    n_outputs_encoder = _get_encoder_output_dim(model_config, dims)
    input_mode = model_config["model"]["encoder"]["input_mode"]
    input_sources = model_config["model"]["encoder"]["input_sources"]
    if input_sources == "y":
        n_encoder_features = dims.n_outputs
    elif input_sources == "u":
        n_encoder_features = dims.n_controls
    elif input_sources == "yu":
        n_encoder_features = dims.n_outputs + dims.n_controls
    else:
        raise ValueError("Encoder input sources not defined")

    n_encoder_inputs = n_encoder_features
    if input_mode == "vector":
        n_encoder_inputs *= model_config["data"]["hist_seq_len"]

    if "tcn" in encoder_type:
        kernel_size = 3
        dilation_base = 2

        if encoder_type == "restcn":
            levels = math.ceil(
                math.log(
                    (model_config["data"]["seq_len"] - 1)
                    * (dilation_base - 1)
                    / (2 * (kernel_size - 1))
                    + 1,
                    dilation_base,
                )
            )
            return neuralnetworks.ResTCNEncoder(
                input_channels=n_encoder_inputs,
                n_outputs=n_outputs_encoder,
                kernel_size=kernel_size,
                dilation_base=dilation_base,
                hidden_channels=model_config["model"]["encoder"]["n_hidden"],
                levels=levels,
                activation=_resolve_activation(
                    model_config["model"]["dynamics"]["act_fct"]
                ),
                dropout=model_config["model"]["encoder"]["dropout"],
            )

        if encoder_type == "tcn":
            levels = math.ceil(
                math.log(
                    (model_config["data"]["seq_len"] - 1)
                    * (dilation_base - 1)
                    / (kernel_size - 1)
                    + 1,
                    dilation_base,
                )
            )
            return neuralnetworks.ResTCNEncoder(
                input_channels=n_encoder_inputs,
                n_outputs=n_outputs_encoder,
                kernel_size=kernel_size,
                dilation_base=dilation_base,
                hidden_channels=model_config["model"]["encoder"]["n_hidden"],
                levels=levels,
                activation=nn.ReLU,
            )

    if encoder_type == "rnn":
        return neuralnetworks.RNNEncoder(
            n_inputs=n_encoder_inputs,
            n_outputs=n_outputs_encoder,
            n_layers=model_config["model"]["encoder"]["n_layers"],
            n_hidden=model_config["model"]["encoder"]["n_hidden"],
            dropout=model_config["model"]["encoder"]["dropout"],
            nonlin="relu",
        )
    if encoder_type == "lstm":
        return neuralnetworks.LSTMEncoder(
            n_inputs=n_encoder_inputs,
            n_outputs=n_outputs_encoder,
            n_layers=model_config["model"]["encoder"]["n_layers"],
            n_hidden=model_config["model"]["encoder"]["n_hidden"],
            dropout=model_config["model"]["encoder"]["dropout"],
        )
    if encoder_type == "gru":
        return neuralnetworks.GRUEncoder(
            n_inputs=n_encoder_inputs,
            n_outputs=n_outputs_encoder,
            n_layers=model_config["model"]["encoder"]["n_layers"],
            n_hidden=model_config["model"]["encoder"]["n_hidden"],
            dropout=model_config["model"]["encoder"]["dropout"],
        )
    if encoder_type == "mlp":
        return neuralnetworks.MultiLayerPerceptron(
            n_inputs=n_encoder_inputs,
            n_outputs=n_outputs_encoder,
            n_layers=model_config["model"]["encoder"]["n_layers"],
            n_hidden=model_config["model"]["encoder"]["n_hidden"],
            activation=_resolve_activation(
                model_config["model"]["dynamics"]["act_fct"]
            ),
            dropout=model_config["model"]["encoder"]["dropout"],
        )
    if encoder_type == "identity":
        return None

    raise ValueError("Encoder not defined")


def _create_encoder(
    model_config: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Create the complete encoder module.

    Args:
        model_config (Dict[str, Any]): Configuration of the model.
        device (torch.device): Device used for model creation.

    Returns:
        nn.Module: Encoder module used in the prediction pipeline.
    """
    dims = _get_model_dimensions(model_config)
    encoder_core = _create_encoder_core(model_config, dims)
    input_mode = model_config["model"]["encoder"]["input_mode"]
    input_sources = model_config["model"]["encoder"]["input_sources"]

    if model_config["model"]["encoder"]["type"] == "identity":
        return stateencoders.IdentityStateEncoder()

    if model_config["model"]["dynamics"]["type"] == "graph":
        return stateencoders.GraphStateEncoder(
            encoder=encoder_core,
            n_states_node=dims.latent_state_dim,
            n_inputs_node=dims.n_controls,
            n_outputs_node=dims.n_outputs,
            n_context_node=model_config["model"]["dynamics"]["n_context"],
            n_nodes=model_config["model"]["graph"]["n_nodes"],
            input_mode=input_mode,
            input_sources=input_sources,
            outputs_as_latent_states=model_config["data"][
                "outputs_as_latent_states"
            ],
            device=device,
        )

    return stateencoders.PartialStateEncoder(
        encoder=encoder_core,
        input_mode=input_mode,
        input_sources=input_sources,
        outputs_as_latent_states=model_config["data"][
            "outputs_as_latent_states"
        ],
        device=device,
    )


def _create_decoder(
    model_config: Dict[str, Any],
    dims: ModelDimensions,
) -> nn.Module:
    """Create the decoder mapping states to outputs.

    Args:
        model_config (Dict[str, Any]): Configuration of the model.
        dims (ModelDimensions): Relevant model dimensions.

    Returns:
        nn.Module: Decoder module.
    """
    if model_config["model"]["dynamics"]["type"] == "graph":
        decoder_network = None
        n_decoder_context = model_config["model"]["decoder"]["n_context"]
        n_decoder_node_embedding = model_config["model"]["decoder"][
            "n_node_embedding"
        ]
        decoder_input_sources = model_config["model"]["decoder"][
            "input_sources"
        ]
        n_decoded_outputs = dims.n_outputs - len(
            model_config["data"]["outputs_as_latent_states"]
        )
        if n_decoded_outputs > 0:
            in_dim_decoder = (
                dims.latent_state_dim
                + n_decoder_context
                + n_decoder_node_embedding
            )
            if decoder_input_sources == "xu":
                in_dim_decoder += dims.n_controls
            decoder_network = neuralnetworks.MultiLayerPerceptron(
                n_inputs=in_dim_decoder,
                n_outputs=n_decoded_outputs,
                n_layers=model_config["model"]["decoder"]["n_layers"],
                n_hidden=model_config["model"]["decoder"]["n_hidden"],
                activation=_resolve_activation(
                    model_config["model"]["dynamics"]["act_fct"]
                ),
                dropout=model_config["model"]["decoder"]["dropout"],
            )
        return outputdecoders.GraphOutputDecoder(
            n_states_node=dims.latent_state_dim,
            n_inputs_node=dims.n_controls,
            n_outputs_node=dims.n_outputs,
            n_ctx_node=n_decoder_context,
            n_node_embedding=n_decoder_node_embedding,
            n_nodes=model_config["model"]["graph"]["n_nodes"],
            decoder=decoder_network,
            outputs_as_latent_states=model_config["data"][
                "outputs_as_latent_states"
            ],
            input_sources=decoder_input_sources,
        )

    decoder_network = None
    n_decoded_outputs = dims.n_outputs - len(
        model_config["data"]["outputs_as_latent_states"]
    )
    if n_decoded_outputs > 0:
        decoder_network = neuralnetworks.MultiLayerPerceptron(
            n_inputs=dims.latent_state_dim + dims.n_controls,
            n_outputs=n_decoded_outputs,
            n_layers=model_config["model"]["decoder"]["n_layers"],
            n_hidden=model_config["model"]["decoder"]["n_hidden"],
            activation=_resolve_activation(
                model_config["model"]["dynamics"]["act_fct"]
            ),
            dropout=model_config["model"]["decoder"]["dropout"],
        )
    return outputdecoders.PartialOutputDecoder(
        n_states=dims.latent_state_dim,
        n_inputs=dims.n_controls,
        n_outputs=dims.n_outputs,
        decoder=decoder_network,
        outputs_as_latent_states=model_config["data"][
            "outputs_as_latent_states"
        ],
    )


def create_model(
    model_config: Dict[str, Any],
    chip_type: str,
    unit_test: bool = False,
) -> nn.Module:
    """Create the full encoder-predictor-decoder model.

    Args:
        model_config (Dict[str, Any]): Configuration of the model.
        chip_type (str): Device specifier used for model creation.
        unit_test (bool, optional): Whether to use deterministic setup for
            testing. Defaults to False.

    Returns:
        nn.Module: Created model.
    """
    device = _resolve_device(chip_type)

    if unit_test:
        torch.manual_seed(0)

    dims = _get_model_dimensions(model_config)
    encoder = _create_encoder(model_config, device)
    predictor = _create_predictor(model_config, device)
    decoder = _create_decoder(model_config, dims)

    return predictors.EncoderPredictorDecoder(
        predictor=predictor,
        encoder=encoder,
        decoder=decoder,
    )


def _build_trainer(
    model: nn.Module,
    model_config: Dict[str, Any],
    prepared_data: PreparedData,
    device: torch.device,
    trial: Optional[optuna.Trial],
) -> trainer.SequencePredictionTrainer:
    """Create the trainer and optimizer for the model.

    Args:
        model (nn.Module): Model to be trained.
        model_config (Dict[str, Any]): Configuration of the model.
        prepared_data (PreparedData): Prepared dataloaders and statistics.
        device (torch.device): Device used during training.
        trial (Optional[optuna.Trial]): Optuna trial for pruning.

    Returns:
        trainer.SequencePredictionTrainer: Configured trainer.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_config["train"]["learning_rate"]
    )

    return trainer.SequencePredictionTrainer(
        model=model,
        optimizer=optimizer,
        loss_fct=nn.MSELoss(),
        train_loader=prepared_data.train_loader,
        dev_loader=prepared_data.dev_loader,
        epochs=model_config["train"]["n_epochs"],
        patience=model_config["train"]["patience"],
        warmup=model_config["train"]["warmup"],
        clip_grad_threshhold=model_config["train"]["grad_clip"],
        device=device,
        trial=trial,
    )


def _run_training(
    trainer: trainer.SequencePredictionTrainer,
) -> Tuple[bool, float]:
    """Run the training routine and measure epoch duration.

    Args:
        trainer (trainer.SequencePredictionTrainer):
            Configured trainer.

    Returns:
        Tuple[bool, float]: Training success flag and average epoch duration.
    """
    t0_train = time.time_ns()
    t0_train_process = time.process_time_ns()
    success = trainer.train()
    if trainer.current_epoch == 0:
        trainer.current_epoch = 1
    t1_train_process = time.process_time_ns()
    t1_train = time.time_ns()

    dur_training_process = (t1_train_process - t0_train_process) * 1e-9
    dur_training_process = dur_training_process / trainer.current_epoch
    dur_training = (t1_train - t0_train) * 1e-9
    dur_training = dur_training / trainer.current_epoch

    print("Training time per epoch: ", dur_training)
    print("Process time per epoch: ", dur_training_process)

    return success, dur_training


def _evaluate_model(
    model: nn.Module,
    trainer: trainer.SequencePredictionTrainer,
    model_config: Dict[str, Any],
    hyperparameters: List[str],
    prepared_data: PreparedData,
) -> Tuple[float, float, float]:
    """Evaluate the trained model on train, dev and test data.

    Args:
        model (nn.Module): Trained model.
        trainer (trainer.SequencePredictionTrainer):
            Trainer containing the best checkpoint.
        model_config (Dict[str, Any]): Configuration of the model.
        hyperparameters (List[str]): Hyperparameters currently optimized.
        prepared_data (PreparedData): Prepared dataloaders and statistics.

    Returns:
        Tuple[float, float, float]: Mean train, dev and test losses.
    """
    model.load_state_dict(trainer.best_model_state_dict)

    train_loss = trainer.evaluate(prepared_data.train_loader)
    dev_loss = trainer.evaluate(prepared_data.dev_loader)
    test_loss = trainer.evaluate(prepared_data.test_loader)
    return train_loss, dev_loss, test_loss


def _handle_mlflow_logging(
    log_mlf: bool,
    run_name: str,
    success: bool,
    trial: Optional[optuna.Trial],
    model: nn.Module,
    trainer: trainer.SequencePredictionTrainer,
    train_loss: float,
    dev_loss: float,
    test_loss: float,
    dur_training: float,
    hyperparameters: List[str],
    model_config: Dict[str, Any],
    chip_type: str,
) -> None:
    """Log the completed training run to MLflow if enabled.

    Args:
        log_mlf (bool): Whether MLflow logging is enabled.
        run_name (str): Name of the MLflow run.
        success (bool): Whether training finished successfully.
        trial (Optional[optuna.Trial]): Optuna trial for pruning.
        model (nn.Module): Trained model.
        trainer (trainer.SequencePredictionTrainer):
            Trainer containing logged metrics.
        train_loss (float): Mean training loss.
        dev_loss (float): Mean validation loss.
        test_loss (float): Mean test loss.
        dur_training (float): Average duration per epoch.
        hyperparameters (List[str]): Hyperparameters currently optimized.
        model_config (Dict[str, Any]): Configuration of the model.
        chip_type (str): Device specifier used for training.
    """
    if not log_mlf:
        return

    if mlflow.active_run():
        mlflow.end_run(status="KILLED")
    mlflow.start_run(run_name=run_name)

    logging.log_run(
        dynamics_model=model,
        mean_train_loss=train_loss,
        mean_dev_loss=dev_loss,
        mean_test_loss=test_loss,
        train_loss_over_epochs=trainer.train_loss,
        dev_loss_over_epochs=trainer.dev_loss,
        avg_time_epoch=dur_training,
        time_vectors=None,
        data_sets=None,
        hyperparameters=hyperparameters,
        model_configuration=model_config,
        chip_type=chip_type,
    )

    if os.getenv("GITLAB_CI"):
        mlflow.set_tags({"gitlab.CI_JOB_ID": os.getenv("CI_JOB_ID")})

    if (not success) and trial:
        mlflow.end_run(status="KILLED")
        raise optuna.TrialPruned()

    mlflow.end_run()


def train_model(
    model_config: Dict[str, Any],
    hyperparameters: List[str],
    chip_type: str,
    model: nn.Module,
    prepared_data: PreparedData,
    log_mlf: bool = False,
    trial: Optional[optuna.Trial] = None,
) -> Tuple[nn.Module, float, Optional[Dict[str, np.ndarray]]]:
    """Train an existing dynamics model and evaluate it.

    Args:
        model_config (Dict[str, Any]): Configuration of the model.
        hyperparameters (List[str]): Hyperparameters currently optimized.
        chip_type (str): Device specifier used for training.
        model (nn.Module): Model to be trained.
        prepared_data (PreparedData): Prepared dataloaders and statistics.
        log_mlf (bool, optional): Whether to log the run to MLflow. Defaults to
            False.
        trial (Optional[optuna.Trial], optional): Optuna trial for pruning.
            Defaults to None.

    Returns:
        Tuple[nn.Module, float, Optional[Dict[str, np.ndarray]]]: Trained
            model, mean test loss, and training normalization statistics.
    """
    device = _resolve_device(chip_type)
    run_name = "configuration_" + str(model_config["meta"]["config_id"])
    _log_configuration(model_config, hyperparameters)

    model.set_n_steps(model_config["data"]["seq_len"] - 1)
    trainer = _build_trainer(
        model=model,
        model_config=model_config,
        prepared_data=prepared_data,
        device=device,
        trial=trial,
    )

    success, dur_training = _run_training(trainer)
    train_loss, dev_loss, test_loss = _evaluate_model(
        model=model,
        trainer=trainer,
        model_config=model_config,
        hyperparameters=hyperparameters,
        prepared_data=prepared_data,
    )

    _handle_mlflow_logging(
        log_mlf=log_mlf,
        run_name=run_name,
        success=success,
        trial=trial,
        model=model,
        trainer=trainer,
        train_loss=train_loss,
        dev_loss=dev_loss,
        test_loss=test_loss,
        dur_training=dur_training,
        hyperparameters=hyperparameters,
        model_config=model_config,
        chip_type=chip_type,
    )

    return model, test_loss, prepared_data.train_stats


def learn_dynamics(
    model_config: Dict[str, Any],
    hyperparameters: List[str],
    chip_type: str,
    data_sets: List[Tuple[np.ndarray, Dict[str, np.ndarray]]],
    log_mlf: bool = False,
    normalize_data: bool = True,
    trial: Optional[optuna.Trial] = None,
    model: Optional[nn.Module] = None,
    data_stats_train: Optional[Dict[str, np.ndarray]] = None,
    unit_test: bool = False,
) -> Tuple[nn.Module, float, Optional[Dict[str, np.ndarray]]]:
    """Create and train a dynamics model in one convenience function.

    Args:
        model_config (Dict[str, Any]): Configuration of the model.
        hyperparameters (List[str]): Hyperparameters currently optimized.
        chip_type (str): Device specifier used for training.
        data_sets (List[Tuple[np.ndarray, Dict[str, np.ndarray]]]): Train, dev
            and test datasets with their time vectors.
        log_mlf (bool, optional): Whether to log the run to MLflow. Defaults to
            False.
        normalize_data (bool, optional): Whether to normalize the datasets.
            Defaults to True.
        trial (Optional[optuna.Trial], optional): Optuna trial for pruning.
            Defaults to None.
        model (Optional[nn.Module], optional): Existing model to
            continue training with. Defaults to None.
        data_stats_train (Optional[Dict[str, np.ndarray]], optional): Existing
            normalization statistics of the training data. Defaults to None.
        unit_test (bool, optional): Whether to use deterministic setup for
            testing. Defaults to False.

    Returns:
        Tuple[nn.Module, float, Optional[Dict[str, np.ndarray]]]: Trained
            model, mean test loss, and training normalization statistics.
    """
    _validate_optional_inputs(model, data_stats_train)
    model_config = resolve_config_references(model_config, data_sets[0][1])

    prepared_data = prepare_data(
        model_config=model_config,
        chip_type=chip_type,
        data_sets=data_sets,
        normalize_data=normalize_data,
        data_stats_train=data_stats_train,
    )

    if model is None:
        model = create_model(
            model_config=model_config,
            chip_type=chip_type,
            unit_test=unit_test,
        )
    elif unit_test:
        torch.manual_seed(0)

    model, test_loss, train_stats = train_model(
        model_config=model_config,
        hyperparameters=hyperparameters,
        chip_type=chip_type,
        model=model,
        prepared_data=prepared_data,
        log_mlf=log_mlf,
        trial=trial,
    )
    return model, test_loss, train_stats
