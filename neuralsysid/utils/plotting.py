"""Plotting helpers for datasets and prediction trajectories."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
from matplotlib import pyplot as plt

DEFAULT_INVERTER_STATE_NAMES = (r"$\theta$", r"$p_{m}$", r"$v$")
DEFAULT_INVERTER_CONTROL_NAMES = (
    r"$w_d$",
    r"$p_d$",
    r"$v_d$",
    r"$q_d$",
)


def _as_axes_array(axes: plt.Axes | np.ndarray) -> np.ndarray:
    """Normalize matplotlib axes output to a one-dimensional array."""
    if isinstance(axes, np.ndarray):
        return axes.reshape(-1)
    return np.array([axes], dtype=object)


def _slice_from_limits(x_limits: tuple[int, int] | None) -> slice:
    return slice(*x_limits) if x_limits is not None else slice(None)


def _default_channel_names(prefix: str, n_channels: int) -> list[str]:
    return [f"{prefix}{idx}" for idx in range(n_channels)]


def _infer_channel_count(*signals: np.ndarray | None) -> int:
    for signal in signals:
        if signal is not None:
            return int(signal.shape[1])
    raise ValueError("At least one signal must be provided.")


def _plot_channel_group(
    time: np.ndarray,
    values: np.ndarray,
    channel_names: Sequence[str],
    *,
    figure_title: str | None = None,
    x_label: str = "time",
    line_labels: Sequence[str] | None = None,
    line_styles: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
) -> plt.Figure:
    """Plot one or multiple trajectories for each signal channel."""
    n_channels = values.shape[-1]
    axes_count = len(channel_names)
    if axes_count != n_channels:
        raise ValueError("Number of channel names must match data channels.")

    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(14, max(4, 2.8 * n_channels)),
        layout="constrained",
        squeeze=False,
    )
    axes_flat = axes[:, 0]

    n_series = values.shape[0]
    line_labels = list(line_labels or [None] * n_series)
    line_styles = list(line_styles or ["-"] * n_series)
    colors = list(colors or [None] * n_series)

    for channel_idx, axis in enumerate(axes_flat):
        for series_idx in range(n_series):
            axis.plot(
                time[series_idx],
                values[series_idx, :, channel_idx],
                linestyle=line_styles[series_idx],
                color=colors[series_idx],
                label=line_labels[series_idx],
                linewidth=2.0,
            )
        axis.set_ylabel(channel_names[channel_idx], rotation=0, labelpad=20)
        axis.tick_params(labelbottom=channel_idx == n_channels - 1)
        if any(label is not None for label in line_labels):
            axis.legend()

    axes_flat[-1].set_xlabel(x_label)
    if figure_title is not None:
        fig.suptitle(figure_title)
    return fig


def plot_system_trajectories(
    time: np.ndarray,
    dataset: Mapping[str, np.ndarray],
    *,
    output_names_per_system: Sequence[str] = DEFAULT_INVERTER_STATE_NAMES,
    control_names_per_system: Sequence[str] = DEFAULT_INVERTER_CONTROL_NAMES,
    n_systems: int = 2,
    output_key: str = "X",
    control_key: str = "U",
    show_outputs: bool = True,
    show_controls: bool = True,
) -> tuple[plt.Figure | None, plt.Figure | None]:
    """Plot output and control trajectories grouped by local channel name.

    The function assumes a flattened channel layout where all channels of system
    0 come first, then all channels of system 1, and so on.
    """
    output_figure = None
    control_figure = None

    n_batches = time.shape[0]

    if show_outputs:
        output_values = dataset[output_key]
        n_output_channels = len(output_names_per_system)
        if output_values.shape[-1] != n_output_channels * n_systems:
            raise ValueError("Output channels do not match n_systems layout.")

        grouped_outputs = np.zeros(
            (n_batches * n_systems, output_values.shape[1], n_output_channels)
        )
        grouped_time = np.repeat(time, n_systems, axis=0)
        line_labels = []
        line_styles = []
        colors = []

        row = 0
        for batch_idx in range(n_batches):
            for system_idx in range(n_systems):
                start = system_idx * n_output_channels
                stop = start + n_output_channels
                grouped_outputs[row] = output_values[
                    batch_idx, :, start:stop
                ]
                line_labels.append(
                    f"system {system_idx}" if batch_idx == 0 else None
                )
                line_styles.append("-" if batch_idx % 2 == 0 else "--")
                colors.append(f"C{system_idx % 10}")
                row += 1

        output_figure = _plot_channel_group(
            grouped_time,
            grouped_outputs,
            output_names_per_system,
            figure_title="Output trajectories",
            x_label="t",
            line_labels=line_labels,
            line_styles=line_styles,
            colors=colors,
        )

    if show_controls:
        control_values = dataset[control_key]
        n_control_channels = len(control_names_per_system)
        if control_values.shape[-1] != n_control_channels * n_systems:
            raise ValueError("Control channels do not match n_systems layout.")

        grouped_controls = np.zeros(
            (
                n_batches * n_systems,
                control_values.shape[1],
                n_control_channels,
            )
        )
        grouped_time = np.repeat(time, n_systems, axis=0)
        line_labels = []
        line_styles = []
        colors = []

        row = 0
        for batch_idx in range(n_batches):
            for system_idx in range(n_systems):
                start = system_idx * n_control_channels
                stop = start + n_control_channels
                grouped_controls[row] = control_values[
                    batch_idx, :, start:stop
                ]
                line_labels.append(
                    f"system {system_idx}" if batch_idx == 0 else None
                )
                line_styles.append("-" if batch_idx % 2 == 0 else "--")
                colors.append(f"C{system_idx % 10}")
                row += 1

        control_figure = _plot_channel_group(
            grouped_time,
            grouped_controls,
            control_names_per_system,
            figure_title="Control trajectories",
            x_label="t",
            line_labels=line_labels,
            line_styles=line_styles,
            colors=colors,
        )

    return output_figure, control_figure


def plot_dataset_collection(
    times: Iterable[np.ndarray],
    datasets: Iterable[Mapping[str, np.ndarray]],
    *,
    output_names_per_system: Sequence[str] = DEFAULT_INVERTER_STATE_NAMES,
    control_names_per_system: Sequence[str] = DEFAULT_INVERTER_CONTROL_NAMES,
    n_systems: int = 2,
    output_key: str = "X",
    control_key: str = "U",
    show_outputs: bool = True,
    show_controls: bool = True,
) -> list[tuple[plt.Figure | None, plt.Figure | None]]:
    """Plot multiple datasets with the same channel layout."""
    figures = []
    for time, dataset in zip(times, datasets):
        figures.append(
            plot_system_trajectories(
                time,
                dataset,
                output_names_per_system=output_names_per_system,
                control_names_per_system=control_names_per_system,
                n_systems=n_systems,
                output_key=output_key,
                control_key=control_key,
                show_outputs=show_outputs,
                show_controls=show_controls,
            )
        )
    return figures


def plot_prediction_comparison(
    *,
    measured: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    predicted: np.ndarray | None = None,
    controls: np.ndarray | None = None,
    output_names: Sequence[str] | None = None,
    control_names: Sequence[str] | None = None,
    x_limits: tuple[int, int] | None = None,
) -> tuple[plt.Figure | None, plt.Figure | None]:
    """Plot measured, reference, and predicted output trajectories.

    All trajectory inputs are expected to have shape ``(time, channels)``.
    """
    plot_slice = _slice_from_limits(x_limits)
    output_figure = None
    control_figure = None

    if any(signal is not None for signal in (measured, reference, predicted)):
        n_outputs = _infer_channel_count(measured, reference, predicted)
        output_names = list(
            output_names or _default_channel_names("y", n_outputs)
        )
        output_series = []
        line_labels = []
        line_styles = []
        colors = []

        if measured is not None:
            output_series.append(measured[plot_slice][None, ...])
            line_labels.append("Measured")
            line_styles.append("-")
            colors.append("#9aa6a6")
        if reference is not None:
            output_series.append(reference[plot_slice][None, ...])
            line_labels.append("Reference")
            line_styles.append("-.")
            colors.append("k")
        if predicted is not None:
            output_series.append(predicted[plot_slice][None, ...])
            line_labels.append("Predicted")
            line_styles.append("--")
            colors.append("C2")

        output_values = np.concatenate(output_series, axis=0)
        output_time = np.tile(
            np.arange(output_values.shape[1], dtype=float),
            (output_values.shape[0], 1),
        )
        output_figure = _plot_channel_group(
            output_time,
            output_values,
            output_names,
            figure_title="Output comparison",
            line_labels=line_labels,
            line_styles=line_styles,
            colors=colors,
        )

    if controls is not None:
        control_names = list(
            control_names or _default_channel_names("u", controls.shape[1])
        )
        control_values = controls[plot_slice][None, ...]
        control_time = np.arange(control_values.shape[1], dtype=float)[None, :]
        control_figure = _plot_channel_group(
            control_time,
            control_values,
            control_names,
            figure_title="Control trajectories",
            line_labels=["Control"],
            colors=["C2"],
        )

    return output_figure, control_figure
