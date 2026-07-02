from copy import deepcopy

from configurations.search_types import categorical, log_float


_SEARCH_TEMPLATE = {
    "model": {
        "dynamics": {
            "act_fct": categorical(["softplus", "silu", "gelu"]),
            "n_layers": categorical([1, 2, 4]),
            "n_hidden": categorical([64, 128, 256, 512]),
            "dropout": categorical([0.0, 0.05, 0.1]),
        },
    },
    "train": {
        "learning_rate": log_float(0.0001, 0.01),
        "grad_clip": categorical([0.1, 1.0, 10.0, float("inf")]),
        "batch_size": categorical([2**7, 2**8, 2**9]),
    },
}


SEARCH_SPACES = {
    "euler_oreo_onestep_no_encoding_noangle": deepcopy(_SEARCH_TEMPLATE),
    "rk4_oreo_onestep_no_encoding_noangle": deepcopy(_SEARCH_TEMPLATE),
    "dopri5_oreo_onestep_no_encoding_noangle": deepcopy(_SEARCH_TEMPLATE),
}
