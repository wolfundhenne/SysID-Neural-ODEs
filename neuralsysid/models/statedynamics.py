"""This module implements the right-hand side of the dynamics function a
neural network.
"""

import abc

import torch
from torch import Tensor, nn


class DynamicsFunction(nn.Module, abc.ABC):
    """Abstract base class for dynamics functions.

    Args:
        n_states (int): Number of states.
        n_inputs (int): Number of inputs.
    """

    def __init__(self, n_states: int, n_inputs: int):
        super().__init__()
        self.n_states = n_states
        self.n_inputs = n_inputs

    @abc.abstractmethod
    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward pass of the dynamics model.

        Args:
            x (Tensor): current state
            u (Tensor): current control input

        Returns:
            Tensor: state derivative
        """


class FxuDynamics(DynamicsFunction):
    """
    Represents dynamics function of the form dot dx/dt = f(x, u) as a neural
    network.

    Args:
        n_states (int): Number of states.
        n_inputs (int): Number of inputs.
        n_layers (int): Number of layers for the neural network f.
        n_hidden (int): Number of hidden units in each layer for f.
        activation (nn.Module): Activation function of the neurons.
    """

    def __init__(
        self,
        n_states: int,
        n_inputs: int,
        dynamics: nn.Module,
    ):
        super().__init__(n_states, n_inputs)
        self.dynamics = dynamics

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward pass of the dynamics model.

        Args:
            x (Tensor): current state
            u (Tensor): current control input

        Returns:
            Tensor: state derivative
        """
        out = self.dynamics(torch.cat([x, u], dim=-1))
        return out


class FxGuDynamics(DynamicsFunction):
    """Represents dynamics function of the form dx/dt = f(x) + g(u) as a neural
    network.

    Args:
        n_states (int): Number of states.
        n_inputs (int): Number of inputs.
        n_layers_states (int): Number of layers in the neural network for f.
        n_hidden_states (int): Number of hidden units in each layer for f.
        n_layers_inputs (int): Number of layers in the neural network for g.
        n_hidden_inputs (int): Number of hidden units in each layer for g.
        activation (nn.Module): Activation function of the neurons.
    """

    def __init__(
        self,
        n_states: int,
        n_inputs: int,
        f_dynamics: nn.Module,
        g_dynamics: nn.Module,
    ):
        super().__init__(n_states, n_inputs)
        self.f_dynamics = f_dynamics
        self.g_dynamics = g_dynamics

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward pass of the dynamics model.

        Args:
            x (Tensor): current state
            u (Tensor): current control input

        Returns:
            Tensor: state derivative
        """
        out = self.f_dynamics(x) + self.g_dynamics(u)
        return out


class GraphDynamics(DynamicsFunction):
    """Graph dynamics with dimension-driven node and edge heterogeneity.

    Persistent node heterogeneity is enabled by ``n_node_embedding > 0``.
    Sample-specific node context is enabled by ``n_ctx > 0``. Persistent edge
    heterogeneity is enabled by ``n_edge_embedding > 0`` and can be shared
    either per directed edge or per undirected line.
    """

    def __init__(
        self,
        n_states_node: int,
        n_inputs_node: int,
        n_nodes: int,
        n_msg: int,
        adjacency: list,
        node_function: nn.Module,
        edge_function: nn.Module,
        n_ctx: int = 0,
        n_node_embedding: int = 0,
        n_edge_embedding: int = 0,
        share_edge_embeddings: bool = False,
        additive_messages: bool = False,
        degree_normalization: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(n_states_node * n_nodes, n_inputs_node * n_nodes)
        self.n_states_node = n_states_node
        self.n_inputs_node = n_inputs_node
        self.n_nodes = n_nodes
        self.n_edges = len(adjacency)

        if additive_messages:
            assert (
                n_msg == n_states_node
            ), "For additive messages, message dimension must match node state dimension."

        self.node_function = node_function
        self.edge_function = edge_function
        self.adjacency = adjacency
        self.n_msg = n_msg
        self.n_ctx = n_ctx
        self.n_node_embedding = n_node_embedding
        self.n_edge_embedding = n_edge_embedding
        self.share_edge_embeddings = share_edge_embeddings

        if self.n_edges == 0:
            self.srcs = torch.empty(0, dtype=torch.long, device=device)
            self.dsts = torch.empty(0, dtype=torch.long, device=device)
        else:
            adjacency = torch.tensor(
                adjacency, dtype=torch.long, device=device
            ).T
            self.srcs = adjacency[0]
            self.dsts = adjacency[1]
        deg_in = torch.bincount(self.dsts, minlength=self.n_nodes).float()
        self.register_buffer("deg_in", deg_in.clamp_min(1.0))

        self.additive_messages = additive_messages
        self.degree_normalization = degree_normalization
        self.node_embedding = nn.Parameter(
            torch.empty(1, n_nodes, self.n_node_embedding, device=device)
        )
        if self.n_node_embedding > 0:
            nn.init.xavier_uniform_(self.node_embedding)

        self.register_buffer(
            "ctx", torch.zeros(1, n_nodes, self.n_ctx, device=device)
        )

        n_embed_entries, edge_to_line_idx = self._build_edge_index_map(
            share_edge_embeddings=self.share_edge_embeddings,
            device=device,
        )
        self.register_buffer("edge_to_line_idx", edge_to_line_idx)
        self.edge_embedding = nn.Parameter(
            torch.empty(n_embed_entries, self.n_edge_embedding, device=device)
        )
        if self.n_edge_embedding > 0:
            nn.init.xavier_uniform_(self.edge_embedding)

    def _build_edge_index_map(
        self, share_edge_embeddings: bool, device: torch.device
    ) -> tuple[int, Tensor]:
        """Map directed messages to persistent edge-embedding entries."""
        if share_edge_embeddings:
            undirected_to_idx = {}
            edge_to_line = []

            for src, dst in zip(self.srcs.tolist(), self.dsts.tolist()):
                key = (src, dst) if src < dst else (dst, src)
                if key not in undirected_to_idx:
                    undirected_to_idx[key] = len(undirected_to_idx)
                edge_to_line.append(undirected_to_idx[key])

            return len(undirected_to_idx), torch.tensor(
                edge_to_line, dtype=torch.long, device=device
            )

        return self.n_edges, torch.arange(
            self.n_edges, device=device, dtype=torch.long
        )

    def set_context(self, ctx: Tensor):
        """Set the sample-specific node context.

        Args:
            ctx (Tensor): Context tensor of shape ``(batch_size, n_nodes, n_ctx)``.
        """
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0)
        self.ctx = ctx

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward pass of the dynamics model.

        Args:
            x (Tensor): current state
            u (Tensor): current control input

        Returns:
            Tensor: state derivative
        """
        single_sample = x.dim() == 1
        if single_sample:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)

        x_nodes = x.view(-1, self.n_nodes, self.n_states_node)
        u_nodes = u.view(-1, self.n_nodes, self.n_inputs_node)
        batch_size = x_nodes.shape[0]

        x_srcs = x_nodes[..., self.srcs, :]
        x_dsts = x_nodes[..., self.dsts, :]
        edge_embedding_batch = (
            self.edge_embedding[self.edge_to_line_idx]
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        msgs = self.edge_function(
            torch.cat([x_srcs, x_dsts, edge_embedding_batch], dim=-1)
        )

        agg_msgs = x_nodes.new_zeros(batch_size, self.n_nodes, self.n_msg)
        agg_msgs.index_add_(1, self.dsts, msgs)

        if self.degree_normalization:
            agg_msgs = agg_msgs / self.deg_in.view(1, -1, 1)

        if self.additive_messages:
            dxdt_nodes = (
                self.node_function(
                    torch.cat(
                        [
                            x_nodes,
                            u_nodes,
                            self.node_embedding.expand(batch_size, -1, -1),
                            self.ctx.expand(batch_size, -1, -1),
                        ],
                        dim=-1,
                    )
                )
                + agg_msgs
            )
        else:
            dxdt_nodes = self.node_function(
                torch.cat(
                    [
                        x_nodes,
                        u_nodes,
                        self.node_embedding.expand(batch_size, -1, -1),
                        self.ctx.expand(batch_size, -1, -1),
                        agg_msgs,
                    ],
                    dim=-1,
                )
            )
        dxdt = dxdt_nodes.view(-1, self.n_states_node * self.n_nodes)

        if single_sample:
            dxdt = dxdt.squeeze(0)

        return dxdt
