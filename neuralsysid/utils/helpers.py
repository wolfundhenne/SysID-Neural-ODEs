"""Small helper utilities used across the project."""

from typing import List

import torch
from torch import nn


def initialize_weights_and_biases(
    modules: nn.Module,
    init_weights: List[torch.Tensor],
    init_biases: List[torch.Tensor],
):
    """This function initializes the weights and biases of a list of a nn.Module
    objects. Each layer is iterated over and the weights and biases are set to
    the corresponding values in the init_weights and init_biases lists.

    Args:
        modules (nn.Module): A Sequence of nn.Module objects.
        init_weights (List[torch.Tensor]): A list of torch.Tensor objects with
            appropriate dimensions representing the weights of the neurons.
        init_biases (List[torch.Tensor]): A list of torch.Tensor objects with
            appropriate dimensions representing the biases of the neurons.
    """
    k = 0
    with torch.no_grad():
        for module in modules:
            if isinstance(module, torch.nn.Linear):
                module.weight.copy_(init_weights[k])
                module.bias.copy_(init_biases[k])
                k += 1
