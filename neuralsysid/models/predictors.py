"""Prediction wrappers that combine steppers, encoders and decoders."""

import torch
from torch import Tensor, nn

from .timesteppers import TimeStepper


class StateRollout(nn.Module):
    """Roll out a one-step time-stepper over a fixed prediction horizon."""

    def __init__(
        self,
        time_stepper: TimeStepper,
        n_steps: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.time_stepper = time_stepper
        self.n_steps = n_steps
        self.device = device

    def forward(self, x_0: Tensor, u: Tensor) -> Tensor:
        """Compute the state trajectory starting from ``x_0``."""
        leading_shape = x_0.shape[:-1]
        x_traj = torch.zeros(
            (*leading_shape, self.n_steps + 1, x_0.shape[-1]),
            dtype=x_0.dtype,
            device=x_0.device,
        )
        x_traj[..., 0, :] = x_0
        x_last = x_0

        for i_iter in range(self.n_steps):
            x_next = self.time_stepper(x_last, u[..., i_iter, :])
            x_traj[..., i_iter + 1, :] = x_next
            x_last = x_next

        return x_traj


class EncoderPredictorDecoder(nn.Module):
    """Compose encoder, state rollout and decoder into a full predictor."""

    def __init__(
        self,
        predictor: StateRollout,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.predictor = predictor
        self.encoder = encoder
        self.decoder = decoder
        self.n_steps = 1

    def forward(
        self,
        y_hist: torch.Tensor,
        u_hist: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Predict outputs for the future control sequence ``u``."""
        encoded = self.encoder(y_hist, u_hist)
        if isinstance(encoded, tuple):
            x_0, ctx = encoded
            self.predictor.time_stepper.dynamics.set_context(ctx)
            self.decoder.set_context(ctx)
        else:
            x_0 = encoded
        x_pred = self.predictor(x_0, u)
        return self.decoder(x_pred, u)

    def set_n_steps(self, n_steps: int):
        """Update the prediction horizon of the wrapped rollout."""
        self.n_steps = n_steps
        self.predictor.n_steps = n_steps
