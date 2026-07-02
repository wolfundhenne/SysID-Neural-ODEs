import matplotlib
import numpy as np

from neuralsysid.utils import plotting


matplotlib.use("Agg")


def test_plot_prediction_comparison_returns_figures():
    measured = np.arange(12, dtype=float).reshape(4, 3)
    reference = measured + 1.0
    predicted = measured - 1.0
    controls = np.arange(8, dtype=float).reshape(4, 2)

    output_fig, control_fig = plotting.plot_prediction_comparison(
        measured=measured,
        reference=reference,
        predicted=predicted,
        controls=controls,
        output_names=["a", "b", "c"],
        control_names=["u1", "u2"],
        x_limits=(1, 4),
    )

    assert output_fig is not None
    assert control_fig is not None
    assert len(output_fig.axes) == 3
    assert len(control_fig.axes) == 2


def test_plot_system_trajectories_groups_flattened_channels():
    time = np.tile(np.arange(5, dtype=float), (2, 1))
    dataset = {
        "X": np.arange(2 * 5 * 6, dtype=float).reshape(2, 5, 6),
        "U": np.arange(2 * 5 * 4, dtype=float).reshape(2, 5, 4),
    }

    output_fig, control_fig = plotting.plot_system_trajectories(
        time,
        dataset,
        output_names_per_system=["x1", "x2", "x3"],
        control_names_per_system=["u1", "u2"],
        n_systems=2,
    )

    assert output_fig is not None
    assert control_fig is not None
    assert len(output_fig.axes) == 3
    assert len(control_fig.axes) == 2
