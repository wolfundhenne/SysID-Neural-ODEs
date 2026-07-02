import torch

test_configs_onestep = {
    # data settings
    "t_step": 0.01,
    "used_outputs": [  # no angles observed
        1,
        2,
        3,
        4,
        5,
    ],
    "outputs": [  # no angles observed
        1,
        2,
        3,
        4,
        5,
    ],
    "used_controls": [
        0,
        2,
        4,
        6,
    ],
    "controls": [
        0,
        2,
        4,
        6,
    ],
    "outputs_as_states": list(range(5)),
    "outputs_as_latent_states": list(range(5)),
    "seq_len": 2,  # replace with "seq_len": 32,
    "overlap": 1,
    "hist_seq_len": 1,
    # system architecture
    "dynamics": ["fxgu", "fxu"],
    "stepper": ["discrete", "euler", "rk4"],
    "encoder": "identity",
    "n_states": 5,
    "latent_state_dim": 5,
    "n_context": 0,
    "persistent_context": False,
    # neural networks configuration
    "act_fct": torch.nn.ReLU,
    "n_hidden_A": 16,
    "n_layers_A": 2,
    "n_hidden_B": 16,
    "n_layers_B": 2,
    "dropout_dyn": 0.0,
    "n_hidden_enc": 32,
    "n_layers_enc": 1,
    "dropout_enc": 0.1,  # only for tcn
    "n_layers_dec": 2,
    "n_hidden_dec": 16,
    "dropout_dec": 0.0,
    # training settings
    "learning_rate": 0.003,
    "n_epochs": 10,
    "patience": 10,
    "warmup": 100,
    "batch_size": 10000,
    "grad_clip": 100.0,  # add as dynamic arg to old learn_dynamics,
    # others
    "encode": False,
    "shuffle": False,
    "config_id": "noid",
    # neuromancer
    "loss_type": "mse",
    "one_step_loss_weight": 0,
    "reference_loss_weight": 1,
}


test_configs_multistep = {
    # data settings
    "t_step": 0.01,
    "used_outputs": [  # no angles observed
        1,
        2,
        3,
        4,
        5,
    ],
    "outputs": [  # no angles observed
        1,
        2,
        3,
        4,
        5,
    ],
    "used_controls": [
        0,
        2,
        4,
        6,
    ],
    "controls": [
        0,
        2,
        4,
        6,
    ],
    "outputs_as_states": list(range(5)),
    "outputs_as_latent_states": list(range(5)),
    "seq_len": 10,
    "overlap": 1,
    "hist_seq_len": 1,
    # system architecture
    "dynamics": ["fxgu", "fxu"],
    "stepper": ["discrete", "euler", "rk4"],
    "encoder": "identity",
    "n_states": 5,
    "latent_state_dim": 5,
    "n_context": 0,
    "persistent_context": False,
    # neural networks configuration
    "act_fct": torch.nn.ReLU,
    "n_hidden_A": 16,
    "n_layers_A": 2,
    "n_hidden_B": 16,
    "n_layers_B": 2,
    "dropout_dyn": 0.0,
    "n_hidden_enc": 32,
    "n_layers_enc": 1,
    "dropout_enc": 0.0,
    "n_layers_dec": 2,
    "n_hidden_dec": 16,
    "dropout_dec": 0.0,
    # training settings
    "learning_rate": 0.003,
    "n_epochs": 10,
    "patience": 10,
    "warmup": 100,
    "batch_size": 10000,
    "grad_clip": 100.0,  # add as dynamic arg to old learn_dynamics,
    # others
    "encode": False,
    "shuffle": False,
    "config_id": "noid",
    # neuromancer
    "loss_type": "mse",
    "one_step_loss_weight": 0,
    "reference_loss_weight": 1,
}
