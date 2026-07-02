"""Helpers for resolving named config references against loaded datasets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def _require_name_list(values: Any, path: str) -> list[str]:
    if not isinstance(values, list):
        raise TypeError(f"{path} must be a list of names")
    if not all(isinstance(value, str) for value in values):
        raise TypeError(f"{path} must contain names only, not indices")
    if len(values) != len(set(values)):
        raise ValueError(f"{path} contains duplicate names: {values}")
    return values


def _name_indices(
    names: list[str],
    available_names: list[str],
    path: str,
) -> list[int]:
    unknown_names = [name for name in names if name not in available_names]
    if unknown_names:
        raise KeyError(
            f"Unknown names in {path}: {unknown_names}. Available names: "
            f"{available_names}"
        )
    return [available_names.index(name) for name in names]


def resolve_config_references(
    model_config: Dict[str, Any],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve output/control names in a nested config to indices."""
    resolved = deepcopy(model_config)
    config_data = resolved["data"]
    output_list = data["output_names"]
    control_list = data["control_names"]

    if config_data["outputs"] and isinstance(config_data["outputs"][0], int):
        return resolved

    output_names = _require_name_list(config_data["outputs"], "data.outputs")
    control_names = _require_name_list(config_data["controls"], "data.controls")
    direct_output_names = _require_name_list(
        config_data["outputs_as_latent_states"],
        "data.outputs_as_latent_states",
    )

    missing_direct_outputs = [
        name for name in direct_output_names if name not in output_names
    ]
    if missing_direct_outputs:
        raise KeyError(
            "data.outputs_as_latent_states must be a subset of data.outputs. "
            f"Missing from selected outputs: {missing_direct_outputs}"
        )

    config_data["outputs"] = _name_indices(
        output_names, output_list, "data.outputs"
    )
    config_data["controls"] = _name_indices(
        control_names, control_list, "data.controls"
    )
    config_data["outputs_as_latent_states"] = [
        output_names.index(name) for name in direct_output_names
    ]
    return resolved
