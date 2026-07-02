"""This module provides wrappers for encoders to be used on historic or future
data of system output measurements (or a state subset) in order to encode a
(partial) initial state."""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn


class PartialStateEncoder(nn.Module):
    """Wraps encoder cores to map available measurements to a partial state.

    Args:
        encoder (nn.Module): The encoder to wrap.
        input_mode (str): Interface expected by the encoder core. ``sequence``
            keeps a time dimension, while ``vector`` flattens all trailing
            dimensions into one feature vector per sample.
        input_sources (str): Which signals are provided to the encoder core.
            Must be one of ``y``, ``u`` or ``yu``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        input_mode: str,
        input_sources: str = "yu",
        outputs_as_latent_states: Optional[List[int]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.encoder = encoder
        if input_mode not in {"sequence", "vector"}:
            raise ValueError(
                "input_mode must be one of {'sequence', 'vector'}"
            )
        if input_sources not in {"y", "u", "yu"}:
            raise ValueError("input_sources must be one of {'y', 'u', 'yu'}")
        self.input_mode = input_mode
        self.input_sources = input_sources
        self.outputs_as_latent_states = (
            outputs_as_latent_states
            if outputs_as_latent_states is not None
            else []
        )
        self.device = device

    def _get_encoder_input(
        self,
        y_sequence: torch.Tensor,
        u_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the encoder input according to the configured interface."""
        if self.input_sources == "y":
            encoder_input = y_sequence
        elif self.input_sources == "u":
            encoder_input = u_sequence
        else:
            encoder_input = torch.cat((y_sequence, u_sequence), dim=-1)

        if self.input_mode == "vector":
            encoder_input = encoder_input.reshape(encoder_input.shape[0], -1)

        return encoder_input

    def forward(
        self,
        y_sequence: torch.Tensor,
        u_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the wrapped partial-state encoder.

        Args:
            y_sequence (torch.Tensor): Output sequence tensor of shape
                (batch_size, seq_len, measured_size).
            u_sequence (torch.Tensor): Control input sequence of shape
                (batch_size, seq_len, control_size).

        Returns:
            torch.Tensor: Initial state of shape (batch_size, encoded_size).
        """
        x_encoded = self.encoder(
            self._get_encoder_input(y_sequence, u_sequence)
        )

        if self.outputs_as_latent_states:
            x_0 = torch.cat(
                (
                    y_sequence[..., -1, self.outputs_as_latent_states],
                    x_encoded,
                ),
                dim=-1,
            )
        else:
            x_0 = x_encoded

        return x_0


class IdentityStateEncoder(nn.Module):
    """Identity encoder that returns the last measurement in the sequence as
    initial state.

    Args:
        n_x (int): Size of the state to return.
        x_meas_idcs (List[int], optional): Indices of the state that are
            measured. If provided, only these indices are returned. If None,
            the full state is returned. Defaults to None.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        y_sequence: torch.Tensor,
        u_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the identity encoder.

        Args:
            y_sequence (torch.Tensor): Output sequence tensor of shape
                (batch_size, seq_len, measured_size).
            u_sequence (torch.Tensor): Control input sequence of shape
                (batch_size, seq_len, control_size).

        Returns:
            torch.Tensor: Encoded state of shape (batch_size, encoded_size).
        """
        x_0 = y_sequence[..., -1, :]
        return x_0


class GraphStateEncoder(nn.Module):
    """Wraps graph-based encoders to an interface accepting input and
    state/output sequences to encode a part of or a full initial state.

    Args:
        encoder (nn.Module): The encoder to wrap.
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_states_node: int,
        n_inputs_node: int,
        n_outputs_node: int,
        n_context_node: int,
        n_nodes: int,
        input_mode: str = "sequence",
        input_sources: str = "yu",
        outputs_as_latent_states: Optional[List[int]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.encoder = encoder
        if input_mode not in {"sequence", "vector"}:
            raise ValueError(
                "input_mode must be one of {'sequence', 'vector'}"
            )
        if input_sources not in {"y", "u", "yu"}:
            raise ValueError("input_sources must be one of {'y', 'u', 'yu'}")
        self.n_states_node = n_states_node
        self.n_inputs_node = n_inputs_node
        self.n_outputs_node = n_outputs_node
        self.n_context_node = n_context_node
        self.n_nodes = n_nodes
        self.input_mode = input_mode
        self.input_sources = input_sources
        self.outputs_as_latent_states = (
            outputs_as_latent_states
            if outputs_as_latent_states is not None
            else []
        )
        self.device = device

    def _get_encoder_input(
        self,
        y_sequence_nodes: torch.Tensor,
        u_sequence_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the nodewise encoder input according to the interface."""
        if self.input_sources == "y":
            encoder_input = y_sequence_nodes
        elif self.input_sources == "u":
            encoder_input = u_sequence_nodes
        else:
            encoder_input = torch.cat(
                (y_sequence_nodes, u_sequence_nodes), dim=-1
            )

        if self.input_mode == "vector":
            encoder_input = encoder_input.reshape(encoder_input.shape[0], -1)

        return encoder_input

    def forward(
        self,
        y_sequence: torch.Tensor,
        u_sequence: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the graph-based encoder.

        Args:
            y_sequence (torch.Tensor): Output sequence tensor of shape
                (batch_size, seq_len, measured_size).
            u_sequence (torch.Tensor): Control input sequence of shape
                (batch_size, seq_len, control_size).

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Initial
            state, optionally together with dynamic node context.
        """
        n_samples, seq_len = y_sequence.shape[0], y_sequence.shape[1]

        y_sequence_nodes = (
            y_sequence.view(
                n_samples,
                seq_len,
                self.n_nodes,
                self.n_outputs_node,
            )
            .transpose(1, 2)
            .reshape(
                self.n_nodes * n_samples,
                seq_len,
                self.n_outputs_node,
            )
        )
        u_sequence_nodes = (
            u_sequence.view(
                n_samples,
                seq_len,
                self.n_nodes,
                self.n_inputs_node,
            )
            .transpose(1, 2)
            .reshape(
                self.n_nodes * n_samples,
                seq_len,
                self.n_inputs_node,
            )
        )

        x_encoded_nodes = self.encoder(
            self._get_encoder_input(y_sequence_nodes, u_sequence_nodes)
        )

        if self.outputs_as_latent_states:
            out = torch.cat(
                (
                    y_sequence_nodes[..., -1, self.outputs_as_latent_states],
                    x_encoded_nodes,
                ),
                dim=-1,
            )
        else:
            out = x_encoded_nodes

        x_0_nodes = out[..., : self.n_states_node]
        x_0 = x_0_nodes.reshape(n_samples, self.n_states_node * self.n_nodes)
        if self.n_context_node == 0:
            return x_0

        ctx_nodes = out[..., self.n_states_node :]
        ctx = ctx_nodes.view(n_samples, self.n_nodes, self.n_context_node)

        return x_0, ctx
