from copy import deepcopy


_OREO_NOANGLE_BASE = {
    "data": {
        "t_step": 0.01,
        "outputs": [
            "$p_{m1}$",
            "$v_{1}$",
            "$\\theta_{2,1}$",
            "$p_{m2}$",
            "$v_{2}$",
            "$\\theta_{3,1}$",
            "$p_{m3}$",
            "$v_{3}$",
            "$\\theta_{4,1}$",
            "$p_{m4}$",
            "$v_{4}$",
        ],
        "controls": [
            "$\\omega_{d1}$",
            "$v_{d1}$",
            "$\\omega_{d2}$",
            "$v_{d2}$",
            "$\\omega_{d3}$",
            "$v_{d3}$",
            "$\\omega_{d4}$",
            "$v_{d4}$",
        ],
        "outputs_as_latent_states": [
            "$p_{m1}$",
            "$v_{1}$",
            "$\\theta_{2,1}$",
            "$p_{m2}$",
            "$v_{2}$",
            "$\\theta_{3,1}$",
            "$p_{m3}$",
            "$v_{3}$",
            "$\\theta_{4,1}$",
            "$p_{m4}$",
            "$v_{4}$",
        ],
        "seq_len": 2,
        "hist_seq_len": 1,
        "overlap": 1,
    },
    "model": {
        "dynamics": {
            "type": "fxu",
            "latent_state_dim": 11,
            "act_fct": "softplus",
            "n_layers": 2,
            "n_hidden": 128,
            "dropout": 0.0,
        },
        "encoder": {
            "type": "identity",
            "input_mode": "sequence",
            "input_sources": "yu",
        },
        "decoder": None,
    },
    "train": {
        "learning_rate": 0.001,
        "n_epochs": 5000,
        "patience": 250,
        "warmup": 50,
        "grad_clip": float("inf"),
        "batch_size": 2**8,
        "shuffle": True,
        "drop_last": False,
    },
    "meta": {
        "config_id": "noid",
    },
}


def _with_stepper(stepper: str) -> dict:
    config = deepcopy(_OREO_NOANGLE_BASE)
    config["model"]["dynamics"]["stepper"] = stepper
    return config


BASE_CONFIGS = {
    "euler_oreo_onestep_no_encoding_noangle": _with_stepper("euler"),
    "rk4_oreo_onestep_no_encoding_noangle": _with_stepper("rk4"),
    "dopri5_oreo_onestep_no_encoding_noangle": _with_stepper("dopri5"),
}
