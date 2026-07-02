"""This module implements Neural Networks to be used for system identification."""

import abc
from typing import Type

import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """This class implements a Multi-Layer Perceptron (MLP).

    Args:
        n_inputs (int): Number of input features.
        n_outputs (int): Number of output features.
        n_layers (int): Number of hidden layers.
        n_hidden (int): Number of neurons in each hidden layer.
        activation (nn.Module, optional): Activation function. Defaults to
            nn.ReLU.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_layers: int,
        n_hidden: int,
        activation: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_in_features = n_inputs
        self.n_out_features = n_outputs

        h_sizes = n_layers * [n_hidden]
        sizes = [self.n_in_features] + h_sizes + [self.n_out_features]
        module_list = []
        for k in range(n_layers):
            module_list.append(nn.Linear(sizes[k], sizes[k + 1]))
            module_list.append(activation())
            module_list.append(nn.Dropout(dropout))
        module_list.append(nn.Linear(sizes[-2], sizes[-1]))

        self.mlp = nn.Sequential(*module_list)

    def forward(self, in_features):
        """Forward pass of the MLP.

        Args:
            in_features (_type_): Input features to the MLP.

        Returns:
            Tensor: Output of the MLP.
        """
        out = self.mlp(in_features)
        return out


class RecurrentEncoder(nn.Module, abc.ABC):
    """Base Recurrent Encoder. Uses a observed sequence to encode an output at
    the final time instance.

    Args:
        n_out (int): Number of output units.
        n_hidden (int): Number of hidden units in the RNN.
    """

    recurrent: nn.RNNBase
    head: nn.Linear

    def __init__(self, n_hidden: int, n_outputs: int, dropout: float = 0.0):
        super().__init__()
        self.head = self.build_head(n_hidden, n_outputs)
        self.output_dropout = nn.Dropout(dropout)

    @abc.abstractmethod
    def build_recurrent(
        self, n_inputs: int, n_hidden: int, n_layers: int, dropout: float = 0.0
    ) -> nn.RNNBase:
        """Subclasses return a recurrent network (e.g., nn.RNN).

        Args:
            n_inputs (int): Number of input features.
            n_hidden (int): Number of hidden features.
            n_layers (int): Number of recurrent layers.
            dropout (float, optional): Dropout rate for the recurrent layers.
                Defaults to 0.0.

        Returns:
            nn.RNNBase: The recurrent network.
        """

    def build_head(self, n_hidden: int, n_outputs: int) -> nn.Linear:
        """Returns the projection head, i.e. a linear layer.

        Args:
            n_hidden (int): Dimension of hidden.
            n_outputs (int): Dimension of output.

        Returns:
            nn.Linear: The projection head.
        """
        return nn.Linear(n_hidden, n_outputs)

    def forward(self, in_sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass through the RNN encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len,
                state_size).
            u (torch.Tensor): Control input tensor of shape (batch_size, seq_len,
                control_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, state_size).
        """
        out_sequence, _ = self.recurrent(in_sequence)
        out_last = out_sequence[..., -1, :]
        out_last = self.output_dropout(out_last)
        return self.head(out_last)


class RNNEncoder(RecurrentEncoder):
    """RNN Encoder. Uses a observed sequence to encode an output at the final
    time instance.

    Args:
        n_controls (int): Number of control inputs of the system.
        n_states (int): Number of states of the system.
        n_hidden (int): Number of hidden units in the RNN.
        n_layers (int): Number of layers in the RNN.
        dropout (float, optional): Dropout rate for the RNN layers. Defaults to
            0.0.
        nonlin (str): Non-linearity to use in the RNN. "tanh" or "relu".
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_hidden: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        nonlin: str = "tanh",
    ):
        super().__init__(n_hidden, n_outputs, dropout)
        self.recurrent = self.build_recurrent(
            n_inputs, n_hidden, n_layers, dropout, nonlin
        )

    def build_recurrent(
        self,
        n_inputs: int,
        n_hidden: int,
        n_layers: int,
        dropout: float = 0.0,
        nonlin: str = "tanh",
    ) -> nn.RNN:
        """Returns a recurrent network, an RNN.

        Args:
            n_inputs (int): Number of input features.
            n_hidden (int): Number of hidden features.
            n_layers (int): Number of recurrent layers.
            nonlin (str): Non-linearity to use in the RNN. "tanh" or "relu".

        Returns:
            nn.RNN: The recurrent network, a recurrent neural network.
        """

        return nn.RNN(
            input_size=n_inputs,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            nonlinearity=nonlin,
        )


class LSTMEncoder(RecurrentEncoder):
    """LSTM Encoder. Uses a observed sequence to encode an output at the final
    time instance.

    Args:
        n_controls (int): Number of control inputs of the system.
        n_states (int): Number of states of the system.
        n_hidden (int): Number of hidden units in the LSTM.
        n_layers (int): Number of layers in the LSTM.
        dropout (float, optional): Dropout rate for the LSTM layers. Defaults
            to 0.0.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_hidden: int,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(n_hidden, n_outputs, dropout)
        self.recurrent = self.build_recurrent(
            n_inputs, n_hidden, n_layers, dropout
        )

    def build_recurrent(
        self, n_inputs: int, n_hidden: int, n_layers: int, dropout: float = 0.0
    ) -> nn.LSTM:
        """Returns a recurrent network, an LSTM.

        Args:
            n_inputs (int): Number of input features.
            n_hidden (int): Number of hidden features.
            n_layers (int): Number of recurrent layers.
            dropout (float, optional): Dropout rate for the LSTM layers.
                Defaults to 0.0.

        Returns:
            nn.LSTM: The recurrent layer, a Long Short-Term Memory network.
        """

        return nn.LSTM(
            input_size=n_inputs,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )


class GRUEncoder(RecurrentEncoder):
    """GRU Encoder. Uses a observed sequence to encode an output at the final
    time instance.

    Args:
        n_controls (int): Number of control inputs of the system.
        n_states (int): Number of states of the system.
        n_hidden (int): Number of hidden units in the GRU.
        n_layers (int): Number of layers in the GRU.
        dropout (float, optional): Dropout rate for the GRU layers. Defaults to
            0.0.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_hidden: int,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(n_hidden, n_outputs, dropout)
        self.recurrent = self.build_recurrent(
            n_inputs, n_hidden, n_layers, dropout
        )

    def build_recurrent(
        self, n_inputs: int, n_hidden: int, n_layers: int, dropout: float = 0.0
    ) -> nn.GRU:
        """Returns a recurrent network, a GRU.

        Args:
            n_inputs (int): Number of input features.
            n_hidden (int): Number of hidden features.
            n_layers (int): Number of recurrent layers.
            dropout (float, optional): Dropout rate for the GRU layers. Defaults
                to 0.0.

        Returns:
            nn.GRU: The recurrent layer, a Gated Recurrent Unit network.
        """
        return nn.GRU(
            input_size=n_inputs,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )


class TCNEncoderBase(nn.Module, abc.ABC):
    """Base class for Temporal Convolutional Network Encoder. Uses a observed
    sequence to encode an output at the final time instance.

    Args:
        n_hidden (int): Number of hidden units.
        n_outputs (int): Number of output units.
    """

    tcn: nn.Sequential
    head: nn.Linear

    def __init__(self, n_hidden: int, n_outputs: int):
        super().__init__()
        self.head = self.build_head(n_hidden, n_outputs)

    @abc.abstractmethod
    def build_tcn(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_base: int,
        levels: int,
        activation: Type[nn.Module],
    ) -> nn.Sequential:
        """Abstract method to build the TCN Encoder.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Size of the convolutional kernel.
            dilation_base (int): Base dilation factor for the convolutions.
            levels (int): Number of levels in the TCN.
            activation (nn.Module): Activation function to use.

        Returns:
            nn.Sequential: The constructed TCN.
        """

    def build_head(self, n_hidden: int, n_outputs: int) -> nn.Linear:
        """Returns the projection head, i.e. a linear layer.

        Args:
            n_hidden (int): Dimension of hidden.
            n_outputs (int): Dimension of output.

        Returns:
            nn.Linear: The projection head.
        """
        return nn.Linear(n_hidden, n_outputs)

    def forward(self, in_sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Residual TCN encoder.

        Args:
            in_sequence (torch.Tensor): Input tensor of shape (batch_size,
                seq_len, in_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels).
        """
        seq = in_sequence.transpose(-1, -2)
        out = self.tcn(seq)
        out_last = out[..., -1]
        return self.head(out_last)


class TCNEncoder(TCNEncoderBase):
    """Temporal Convolutional Network Encoder. Uses a observed sequence to
    encode an output at the final time instance.

    Args:
        input_channels (int): Number of input channels.
        n_outputs (int): Dimension of the encoded output.
        hidden_channels (int): Number of hidden channels in the TCN.
        kernel_size (int): Size of the convolutional kernel.
        dilation_base (int): Base dilation factor for the convolutions.
        levels (int): Number of levels in the TCN. Should be chosen s.t. the
            receptive field offers full history coverage. For that, with a
            dilation base b, a kernel size k and a sequence length l the minimum
            number of layers is given by ceil[log_b((l-1)*(b-1)/(k-1)+1)].
    """

    def __init__(
        self,
        input_channels: int,
        n_outputs: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dilation_base: int = 2,
        levels: int = 5,
        activation: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__(hidden_channels, n_outputs)
        self.tcn = self.build_tcn(
            input_channels,
            hidden_channels,
            kernel_size,
            dilation_base,
            levels,
            activation,
        )

    def build_tcn(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_base: int,
        levels: int,
        activation: Type[nn.Module],
    ) -> nn.Sequential:
        """Build the TCN module.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Size of the convolutional kernel.
            dilation_base (int): Base dilation factor for the convolutions.
            levels (int): Number of levels in the TCN.
            activation (nn.Module): Activation function to use.

        Returns:
            nn.Sequential: The constructed TCN.
        """

        layers = []
        in_ch = input_channels
        for i in range(levels):
            dilation = dilation_base**i
            layers.append(
                nn.ConstantPad1d(
                    padding=((kernel_size - 1) * dilation, 0), value=0
                )
            )
            layers.append(
                nn.Conv1d(
                    in_ch,
                    hidden_channels,
                    kernel_size,
                    padding=0,
                    dilation=dilation,
                ),
            )
            layers.append(activation())
            in_ch = hidden_channels
        return nn.Sequential(*layers)

    def forward(self, in_sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass through the TCN encoder.

        Args:
            in_sequence (torch.Tensor): Input tensor of shape (batch_size,
                seq_len, in_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_states_unobs).
        """
        seq = in_sequence.transpose(-1, -2)
        out = self.tcn(seq)
        out_last = out[..., -1]
        return self.head(out_last)


class ResTCNEncoder(TCNEncoderBase):
    """Temporal Convolutional Network Encoder. Uses a observed sequence to
    encode an output at the final time instance.

    Args:
        input_channels (int): Number of input channels.
        n_outputs (int): Dimension of the encoded output.
        hidden_channels (int): Number of hidden channels in the TCN.
        kernel_size (int): Size of the convolutional kernel.
        dilation_base (int): Base dilation factor for the convolutions.
        levels (int): Number of levels in the TCN. Should be chosen s.t. the
            receptive field offers full history coverage. For that, with a
            dilation base b, a kernel size k and a sequence length l the minimum
            number of layers is given by ceil[log_b((l-1)*(b-1)/(2*(k-1))+1)].
    """

    def __init__(
        self,
        input_channels: int,
        n_outputs: int,
        kernel_size: int = 3,
        dilation_base: int = 2,
        hidden_channels: int = 32,
        levels: int = 5,
        activation: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ):
        super().__init__(hidden_channels, n_outputs)
        self.tcn = self.build_tcn(
            input_channels,
            hidden_channels,
            kernel_size,
            dilation_base,
            levels,
            activation,
            dropout,
        )

    def build_tcn(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_base: int,
        levels: int,
        activation: Type[nn.Module],
        dropout: float,
    ) -> nn.Sequential:
        """Build the Residual TCN module.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Size of the convolutional kernel.
            dilation_base (int): Base dilation factor for the convolutions.
            levels (int): Number of levels in the TCN.
            activation (nn.Module): Activation function to use.

        Returns:
            nn.Sequential: The constructed TCN.
        """
        layers = []
        in_ch = input_channels
        for i in range(levels):
            dilation = dilation_base**i
            layers.append(
                ResBlockTCN(
                    input_channels=in_ch,
                    output_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation,
                    dropout=dropout,
                )
            )
            in_ch = hidden_channels
        return nn.Sequential(*layers)


class ResBlockTCN(nn.Module):
    """Residual Block for Temporal Convolutional Network. Contains two
    layers of padding, convolution, and activation.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation (int): Dilation factor for the convolutions.
        activation (nn.Module): Activation function to use.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        dilation: int = 2,
        activation: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        layers.append(
            nn.ConstantPad1d(
                padding=((kernel_size - 1) * dilation, 0), value=0
            )
        )
        layers.append(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=0,
                dilation=dilation,
            )
        )
        layers[-1] = nn.utils.parametrizations.weight_norm(layers[-1])
        layers.append(activation())
        layers.append(nn.Dropout(dropout))

        layers.append(
            nn.ConstantPad1d(
                padding=((kernel_size - 1) * dilation, 0), value=0
            )
        )
        layers.append(
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size,
                padding=0,
                dilation=dilation,
            )
        )
        layers[-1] = nn.utils.parametrizations.weight_norm(layers[-1])

        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        layers.append(activation())
        layers.append(nn.Dropout(dropout))

        self.tcn_block = nn.Sequential(*layers)
        if input_channels != output_channels:
            self.residual = nn.Conv1d(
                input_channels, output_channels, kernel_size=1
            )
        else:
            self.residual = nn.Identity()

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.

        Args:
            input_sequence (torch.Tensor): Input tensor of shape (batch_size,
                in_channels, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels,
                seq_len).
        """
        return self.tcn_block(input_sequence) + self.residual(input_sequence)
