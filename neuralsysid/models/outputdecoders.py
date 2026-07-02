"""Decoder modules that map latent states back to measured outputs."""

from typing import List, Optional

import torch
from torch import Tensor, nn


class PartialOutputDecoder(nn.Module):
    """Maps latent states, optionally together with controls, to outputs."""

    def __init__(
        self,
        n_states: int,
        n_inputs: int,
        n_outputs: int,
        decoder: nn.Module = None,
        outputs_as_latent_states: Optional[List[int]] = None,
        input_sources: str = "xu",
    ):
        super().__init__()
        if input_sources not in {"x", "xu"}:
            raise ValueError("input_sources must be one of {'x', 'xu'}")
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.input_sources = input_sources
        self.outputs_as_latent_states = (
            outputs_as_latent_states
            if outputs_as_latent_states is not None
            else []
        )
        self.outputs_not_as_states = [
            out
            for out in range(n_outputs)
            if out not in self.outputs_as_latent_states
        ]
        if self.outputs_not_as_states:
            self.decoder = decoder
            if self.decoder is None:
                raise ValueError(
                    "A decoder network must be provided when outputs are not "
                    + "part of the state."
                )
        else:
            self.decoder = None

    def _get_decoder_input(self, x: Tensor, u: Tensor) -> Tensor:
        """Assemble the decoder input according to the configured interface."""
        if self.input_sources == "x":
            return x
        return torch.cat((x, u), dim=-1)

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward pass of the decoder.

        Args:
            x (Tensor): Current state trajectory.
            u (Tensor): Current control input trajectory.

        Returns:
            Tensor: Predicted output trajectory.
        """

        y = torch.empty(
            (*x.shape[:-1], self.n_outputs), device=x.device, dtype=x.dtype
        )

        if self.outputs_not_as_states:
            y[..., self.outputs_not_as_states] = self.decoder(
                self._get_decoder_input(x, u)
            )

        if self.outputs_as_latent_states:
            y[..., self.outputs_as_latent_states] = x[
                ..., : len(self.outputs_as_latent_states)
            ]

        return y


class GraphOutputDecoder(nn.Module):
    """Maps nodewise latent states to nodewise outputs on a graph."""

    def __init__(
        self,
        n_states_node: int,
        n_inputs_node: int,
        n_outputs_node: int,
        n_ctx_node: int,
        n_node_embedding: int,
        n_nodes: int,
        decoder: nn.Module = None,
        outputs_as_latent_states: Optional[List[int]] = None,
        input_sources: str = "x",
    ):
        super().__init__()
        if input_sources not in {"x", "xu"}:
            raise ValueError("input_sources must be one of {'x', 'xu'}")
        self.n_states_node = n_states_node
        self.n_inputs_node = n_inputs_node
        self.n_outputs_node = n_outputs_node
        self.n_ctx_node = n_ctx_node
        self.n_node_embedding = n_node_embedding
        self.n_nodes = n_nodes
        self.input_sources = input_sources
        self.outputs_as_latent_states = (
            outputs_as_latent_states
            if outputs_as_latent_states is not None
            else []
        )
        self.outputs_not_as_states = [
            out
            for out in range(n_outputs_node)
            if out not in self.outputs_as_latent_states
        ]
        self.register_buffer("ctx", torch.zeros(1, n_nodes, n_ctx_node))
        self.node_embedding = nn.Parameter(
            torch.empty(1, n_nodes, n_node_embedding)
        )
        if self.n_node_embedding > 0:
            nn.init.xavier_uniform_(self.node_embedding)
        if self.outputs_not_as_states:
            self.decoder = decoder
            if self.decoder is None:
                raise ValueError(
                    "A decoder network must be provided when outputs are not "
                    + "part of the node states."
                )
        else:
            self.decoder = None

    def set_context(self, ctx: Tensor):
        """Set the context for all nodes. The context is constant over the
        entire prediction horizon and must be set for heterogenous node dynamics
        before starting a prediciton with this dynamics model.

        Args:
            ctx (Tensor): Context tensor of shape (batch_size, n_nodes, n_ctx)
        """
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0)
        self.ctx = ctx

    def _get_decoder_input(
        self,
        x_nodes: Tensor,
        u_nodes: Tensor,
        leading_shape: torch.Size,
    ) -> Tensor:
        """Assemble the nodewise decoder input according to the interface."""
        decoder_inputs = [x_nodes]
        if self.input_sources == "xu":
            decoder_inputs.append(u_nodes)
        leading_ones = (1,) * len(leading_shape)
        decoder_inputs.append(
            self.node_embedding.view(
                *leading_ones, self.n_nodes, self.n_node_embedding
            ).expand(*leading_shape, self.n_nodes, self.n_node_embedding)
        )
        decoder_inputs.append(
            self.ctx.view(*leading_ones, self.n_nodes, self.n_ctx_node).expand(
                *leading_shape, self.n_nodes, self.n_ctx_node
            )
        )
        return torch.cat(decoder_inputs, dim=-1)

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward pass of the graph decoder.

        Args:
            x (Tensor): Current state trajectory.
            u (Tensor): Current control input trajectory.

        Returns:
            Tensor: Predicted output trajectory.
        """
        leading_shape = x.shape[:-1]
        x_nodes = x.view(*leading_shape, self.n_nodes, self.n_states_node)
        u_nodes = u.view(*leading_shape, self.n_nodes, self.n_inputs_node)

        y_nodes = torch.empty(
            (*x_nodes.shape[:-1], self.n_outputs_node),
            device=x.device,
            dtype=x.dtype,
        )

        if self.outputs_not_as_states:
            y_nodes[..., self.outputs_not_as_states] = self.decoder(
                self._get_decoder_input(x_nodes, u_nodes, leading_shape)
            )

        if self.outputs_as_latent_states:
            y_nodes[..., self.outputs_as_latent_states] = x_nodes[
                ..., : len(self.outputs_as_latent_states)
            ]

        return y_nodes.view(*leading_shape, self.n_nodes * self.n_outputs_node)
