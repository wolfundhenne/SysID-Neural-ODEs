"""This module impelements Datasets for the ML System Identification."""

from torch import Tensor
from torch.utils.data import Dataset


class SequenceDatasetHistoricEncoding(Dataset):
    """Dataset for multi-step ahead predictions. Consider L is the sequence
    length, n_x is the dimension of the state and n_u is the dimension of the
    input. Then with k sequences, we get:

    Args:
        y_h (Tensor): Historic measurements. Has dimensions (k, H, n_y).
        u_h (Tensor): Historic inputs. Has dimensions (k, H, n_u).
        u (Tensor): The input sequence. (k, L, n_u).
        y (Tensor): The output sequence. (k, L, n_y).
    """

    def __init__(self, y_h: Tensor, u_h: Tensor, y: Tensor, u: Tensor):
        super().__init__()
        self.len = y.shape[0]
        self.y_h = y_h
        self.u_h = u_h
        self.u = u
        self.y = y

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.len

    def __getitem__(self, idx):
        """Returns the sample at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[Tuple[Tensor, Tensor], Tensor],Tensor: The input and output
                features. The first Tuple are the input features for the encoded
                state. The encoded state together with the second entry of the
                inner Tuple are the features for the forward pass of the neural
                ODE.
        """
        return ((self.y_h[idx], self.u_h[idx]), self.u[idx]), self.y[idx]
