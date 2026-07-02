"""Top-level package for neural system identification components."""

from . import data, models, training, utils
from .data import datasets, io, noise, preparation, preprocessing
from .models import (
    neuralnetworks,
    outputdecoders,
    predictors,
    statedynamics,
    stateencoders,
    timesteppers,
)
from .training import pipeline, trainer

__all__ = [
    "data",
    "datasets",
    "io",
    "models",
    "neuralnetworks",
    "noise",
    "outputdecoders",
    "pipeline",
    "predictors",
    "preparation",
    "preprocessing",
    "statedynamics",
    "stateencoders",
    "timesteppers",
    "trainer",
    "training",
    "utils",
]
