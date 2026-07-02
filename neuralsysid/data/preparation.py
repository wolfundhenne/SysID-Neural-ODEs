"""Build sequence datasets and dataloaders for training."""

from typing import Dict, List, Tuple

import numpy
import torch
from torch.utils.data import DataLoader

from neuralsysid.data.datasets import (
    SequenceDatasetHistoricEncoding,
)


def build_sequence_dataloader(
    t: numpy.ndarray,
    trajectory_data: Dict[str, numpy.ndarray],
    used_outputs: List[int],
    used_controls: List[int],
    *,
    batch_size: int,
    shuffle: bool = True,
    seq_len: int = 64,
    overlap: int = 0,
    historic_seq_len: int = 32,
    workers: int = 0,
    cutoff_samples: int = 0,
    encode: bool = True,
    drop_last: bool = False,
) -> Tuple[numpy.ndarray, SequenceDatasetHistoricEncoding, DataLoader]:
    """Build a sequence dataset and dataloader for one trajectory dataset.

    Args:
        t (numpy.ndarray): time vector
        trajectory_data (dict): dictionary containing the data
        used_outputs (List[int]): observed outputs used for training
        used_controls (List[int]): control inputs used for training
        batch_size (int): size of one batch
        shuffle (bool, optional): whether to shuffle the sequences in training
            or not. Defaults to True.

    Returns:
        Tuple[numpy.ndarray, OneStepTrajectoriesDataset, DataLoader]: time
            vector, dataset and dataloader
    """

    if not encode:
        assert (
            historic_seq_len == 1
        ), "If encoding is disabled, historic_seq_len must be 1."

    t = t.squeeze(0)
    y = trajectory_data["Y"].squeeze(0)[:, used_outputs]
    u = trajectory_data["U"].squeeze(0)[:, used_controls]

    # cut off initial samples if specified
    y = y[cutoff_samples:, :]
    u = u[cutoff_samples:, :]
    t = t[cutoff_samples:]

    n_timesteps = y.shape[0]
    if encode:
        n_sequences = (n_timesteps - seq_len - historic_seq_len) // (
            seq_len - overlap
        ) + 1
    else:
        n_sequences = (n_timesteps - seq_len) // (seq_len - overlap) + 1

    t_seq = numpy.zeros((n_sequences, seq_len))
    u_in = numpy.zeros((n_sequences, seq_len, u.shape[1]))
    y_out = numpy.zeros((n_sequences, seq_len, y.shape[1]))
    y_hist = numpy.zeros((n_sequences, historic_seq_len, y.shape[1]))
    if encode:
        u_hist = numpy.zeros((n_sequences, historic_seq_len, u.shape[1]))
    else:
        u_hist = numpy.zeros((n_sequences, 0, u.shape[1]))

    for i_iter in range(n_sequences):
        if encode:
            start_idx = i_iter * (seq_len - overlap) + historic_seq_len
        else:
            start_idx = i_iter * (seq_len - overlap)

        end_idx = start_idx + seq_len

        t_seq[i_iter, :] = t[start_idx:end_idx]
        u_in[i_iter, :, :] = u[start_idx:end_idx, :]
        y_out[i_iter, :, :] = y[start_idx:end_idx, :]

        y_hist[i_iter, :, :] = y[
            start_idx - historic_seq_len + 1 : start_idx + 1, :
        ]
        if encode:
            u_hist[i_iter, :, :] = u[
                start_idx - historic_seq_len : start_idx, :
            ]

    u_in = torch.tensor(u_in, dtype=torch.float32)
    y_out = torch.tensor(y_out, dtype=torch.float32)

    y_hist = torch.tensor(y_hist, dtype=torch.float32)
    u_hist = torch.tensor(u_hist, dtype=torch.float32)

    sequence_dataset = SequenceDatasetHistoricEncoding(
        y_hist, u_hist, y_out, u_in
    )

    data_loader = DataLoader(
        sequence_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=drop_last,
    )

    return (
        t_seq,
        sequence_dataset,
        data_loader,
    )


def build_sequence_dataloaders(
    t_vecs: List[numpy.ndarray],
    trajectory_data_sets: List[Dict[str, numpy.ndarray]],
    used_outputs: List[int],
    used_controls: List[int],
    *,
    batch_size: int,
    shuffle: bool = True,
    seq_len: int = 64,
    overlap: int = 0,
    historic_seq_len: int = 32,
    workers: int = 0,
    cutoff_samples: int = 0,
    encode: bool = True,
    drop_last: bool = False,
) -> Tuple[Tuple[numpy.ndarray, SequenceDatasetHistoricEncoding, DataLoader]]:
    """Build sequence datasets and dataloaders for multiple trajectories.

    Args:
        t_vecs (List[numpy.ndarray]): independent time vectors
        trajectory_data_sets (List[Dict[str, numpy.ndarray]]): independent
            datasets
        used_outputs (List[int]): observed outputs used for training
        used_controls (List[int]): control inputs used for training
        batch_size (int): size of one batch
        shuffle (bool, optional): whether to shuffle the sequences in training
            or not. Defaults to True.

    Returns:
        Tuple[Tuple[numpy.ndarray, SequenceDatasetHistoricEncoding, DataLoader]]:
            time vector, dataset and dataloader for each dataset
    """
    sequence_dataloaders = []
    for t, trajectory_data in zip(t_vecs, trajectory_data_sets):
        sequence_dataloaders.append(
            build_sequence_dataloader(
                t,
                trajectory_data,
                used_outputs=used_outputs,
                used_controls=used_controls,
                batch_size=batch_size,
                shuffle=shuffle,
                seq_len=seq_len,
                overlap=overlap,
                historic_seq_len=historic_seq_len,
                workers=workers,
                cutoff_samples=cutoff_samples,
                encode=encode,
                drop_last=drop_last,
            )
        )

    return sequence_dataloaders
