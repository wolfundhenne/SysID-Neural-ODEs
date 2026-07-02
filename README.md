# neural-system-identification

Neural system identification for dynamical systems, centered on Neural ODE style models and encoder-predictor-decoder architectures.

The codebase supports two main usage modes:

1. Train from a nested config through the shared pipeline.
2. Build the model stack manually from encoder, dynamics, time-stepper, predictor, and decoder components.

## Installation

Create an environment and install the dependencies from [requirements.txt](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/requirements.txt:1).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository Layout

- [neuralsysid/](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/neuralsysid): main package
- [neuralsysid/training/pipeline.py](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/neuralsysid/training/pipeline.py:1): main training entry points
- [neuralsysid/data/io.py](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/neuralsysid/data/io.py:1): dataset loading
- [configurations/](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/configurations): experiment configs
- [notebooks/train_example.ipynb](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/notebooks/train_example.ipynb:1): compact end-to-end example

## Dataset Format

The loader expects `train`, `dev`, and `test` JSON files with the same prefix, for example:

- `data/2025-10-25_14-57-09_ieee9_kron_train.json`
- `data/2025-10-25_14-57-09_ieee9_kron_dev.json`
- `data/2025-10-25_14-57-09_ieee9_kron_test.json`

Loading is handled by [neuralsysid/data/io.py](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/neuralsysid/data/io.py:1). After loading, each split is represented as:

- `t`: time vector with shape `(1, T)`
- `data["Y"]`: outputs with shape `(1, T, n_outputs)`
- `data["U"]`: controls with shape `(1, T, n_controls)`
- `data["output_names"]`: output channel names
- `data["control_names"]`: control channel names

## Workflow 1: Train Through the Pipeline

This is the default path. You define or reuse a nested config, load the dataset, and call `pipeline.learn_dynamics(...)`.

```python
from copy import deepcopy
import importlib

from configurations.utils import resolve_config_references
from neuralsysid.data import io
from neuralsysid.training import pipeline

CONFIG_SET = "wolf_2026_augmented"
MODEL_TYPE = "rk4_ieee9_restcn_noangle"
DATA_NAME = "2025-10-25_14-57-09_ieee9_kron_"
CHIP_TYPE = "cpu"

config_module = importlib.import_module(f"configurations.{CONFIG_SET}.base")
model_config = deepcopy(config_module.BASE_CONFIGS[MODEL_TYPE])
model_config["meta"]["config_id"] = "example_run"

data_sets, metadata = io.load_data("data/" + DATA_NAME)

model, test_loss, train_stats = pipeline.learn_dynamics(
    model_config=model_config,
    hyperparameters=[],
    chip_type=CHIP_TYPE,
    data_sets=data_sets,
    normalize_data=True,
)

print(test_loss)
```

Use this path when:

- you already have a config in `configurations/*/base.py`
- you want the standard resampling, normalization, batching, model construction, and training loop
- you want to stay aligned with the scripts and notebook examples

## Going Lower Level

If you want to construct everything by hand, the relevant modules are:

- [neuralsysid/models/statedynamics.py](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/neuralsysid/models/statedynamics.py:1)
- [neuralsysid/models/timesteppers.py](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/neuralsysid/models/timesteppers.py:1)
- [neuralsysid/models/stateencoders.py](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/neuralsysid/models/stateencoders.py:1)
- [neuralsysid/models/outputdecoders.py](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/neuralsysid/models/outputdecoders.py:1)
- [neuralsysid/models/predictors.py](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/neuralsysid/models/predictors.py:1)
- [neuralsysid/training/trainer.py](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/neuralsysid/training/trainer.py:1)

Example for a monolithic `fxu` model with a residual TCN encoder:

```python
import torch
from torch import nn

from neuralsysid.data import io, preparation, preprocessing
from neuralsysid.models import (
    neuralnetworks,
    outputdecoders,
    predictors,
    stateencoders,
    statedynamics,
    timesteppers,
)
from neuralsysid.training import trainer

DATA_NAME = "2025-10-25_14-57-09_ieee9_kron_"
CHIP_TYPE = "cpu"

n_outputs = 6
n_controls = 3
latent_state_dim = 8
t_step = 0.01
seq_len = 64
dynamics_layers = 1
dynamics_neurons = 128
encoder_hidden_channels = 32
learning_rate = 1e-3
n_epochs = 1000

data_sets, metadata = io.load_data("data/" + DATA_NAME)
outputs = list(range(n_outputs))
controls = list(range(n_controls))
outputs_as_latent_states = outputs
encoder_output_dim = latent_state_dim - len(outputs_as_latent_states)
batch_size = 256

t_sample = io.infer_sample_time(data_sets[0][0])
data_sets = preprocessing.resample_data_sets(
    data_sets,
    t_sample=t_sample,
    t_step=t_step,
)
t_train, train_data = data_sets[0]
t_dev, dev_data = data_sets[1]
t_test, test_data = data_sets[2]

train_data, train_stats = preprocessing.normalize_data(train_data)
dev_data, _ = preprocessing.normalize_data(dev_data, train_stats)
test_data, _ = preprocessing.normalize_data(test_data, train_stats)

sequence_dataloaders = preparation.build_sequence_dataloaders(
    [t_train, t_dev, t_test],
    [train_data, dev_data, test_data],
    outputs,
    controls,
    batch_size=batch_size,
    shuffle=True,
    seq_len=seq_len,
    overlap=48,
    historic_seq_len=seq_len,
)
_, _, train_loader = sequence_dataloaders[0]
_, _, dev_loader = sequence_dataloaders[1]
_, _, test_loader = sequence_dataloaders[2]

device = torch.device(CHIP_TYPE)

dynamics_net = neuralnetworks.MultiLayerPerceptron(
    n_inputs=latent_state_dim + n_controls,
    n_outputs=latent_state_dim,
    n_layers=dynamics_layers,
    n_hidden=dynamics_neurons,
    activation=nn.SiLU,
    dropout=0.0,
)
dynamics = statedynamics.FxuDynamics(
    n_states=latent_state_dim,
    n_inputs=n_controls,
    dynamics=dynamics_net,
)
time_stepper = timesteppers.RungeKutta4(
    dynamics,
    t_step,
    device=device,
)
predictor = predictors.StateRollout(
    time_stepper,
    seq_len - 1,
    device=device,
)

encoder_core = neuralnetworks.ResTCNEncoder(
    input_channels=n_outputs + n_controls,
    n_outputs=encoder_output_dim,
    kernel_size=3,
    dilation_base=2,
    hidden_channels=encoder_hidden_channels,
    levels=6,
    activation=nn.SiLU,
    dropout=0.1,
)
encoder = stateencoders.PartialStateEncoder(
    encoder=encoder_core,
    input_mode="sequence",
    input_sources="yu",
    outputs_as_latent_states=outputs_as_latent_states,
    device=device,
)

decoder = outputdecoders.PartialOutputDecoder(
    n_states=latent_state_dim,
    n_inputs=n_controls,
    n_outputs=n_outputs,
    decoder=None,
    outputs_as_latent_states=outputs_as_latent_states,
)

model = predictors.EncoderPredictorDecoder(
    predictor=predictor,
    encoder=encoder,
    decoder=decoder,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)
my_trainer = trainer.SequencePredictionTrainer(
    model=model,
    optimizer=optimizer,
    loss_fct=nn.MSELoss(),
    train_loader=train_loader,
    dev_loader=dev_loader,
    epochs=n_epochs,
    patience=250,
    warmup=250,
    clip_grad_threshhold=0.1,
    device=device,
    trial=None,
)

my_trainer.train()
test_loss = my_trainer.evaluate(test_loader)
print(test_loss)
```

This path is useful when:

- you want to swap one component while keeping the rest fixed
- you want a custom trainer or optimizer setup
- you want to inspect the exact composition of the model stack

## Configs

Configs are nested dictionaries with four top-level sections:

- `data`
- `model`
- `train`
- `meta`

`resolve_config_references(...)` in [configurations/utils.py](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/configurations/utils.py:1) maps output/control names from the config to channel indices from the loaded dataset.

## Examples

- For a minimal publication-style example, use [notebooks/train_example.ipynb](/Users/hanneswolf/Documents/University%20Kassel/AI%20for%20power%20systems/neural-system-identification/notebooks/train_example.ipynb:1).
