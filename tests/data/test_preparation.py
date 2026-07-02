"""Tests for batch construction in neuralsysid.data.preparation."""

import unittest

import numpy as np
import random
import torch

from neuralsysid.data import preparation
from tests.reference_impls import onesteppred, sequencepred

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class TestCreateBatches(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(8)[None, :]
        self.x = np.array(
            [
                [
                    [0.0, 10.0],
                    [1.0, 11.0],
                    [2.0, 12.0],
                    [3.0, 13.0],
                    [4.0, 14.0],
                    [5.0, 15.0],
                    [6.0, 16.0],
                    [7.0, 17.0],
                ]
            ]
        )
        self.u = np.array(
            [
                [
                    [100.0],
                    [101.0],
                    [102.0],
                    [103.0],
                    [104.0],
                    [105.0],
                    [106.0],
                    [107.0],
                ]
            ]
        )
        self.data_set = {"Y": self.x, "U": self.u}

    def test_build_sequence_dataloader_without_encoding_returns_expected_slices(self):
        t_seq, dataset, dataloader = preparation.build_sequence_dataloader(
            t=self.t,
            trajectory_data=self.data_set,
            used_outputs=[0, 1],
            used_controls=[0],
            batch_size=2,
            shuffle=False,
            seq_len=2,
            overlap=1,
            historic_seq_len=1,
            encode=False,
        )

        self.assertEqual(len(dataset), 7)
        self.assertEqual(len(dataloader), 4)
        np.testing.assert_array_equal(
            t_seq,
            np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                ]
            ),
        )

        ((y_h, u_h), u), y = dataset[0]
        self.assertTrue(
            torch.equal(y_h, torch.tensor([[0.0, 10.0]], dtype=torch.float32))
        )
        self.assertEqual(u_h.shape, (0, 1))
        self.assertTrue(
            torch.equal(
                u,
                torch.tensor([[100.0], [101.0]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                y,
                torch.tensor([[0.0, 10.0], [1.0, 11.0]], dtype=torch.float32),
            )
        )

    def test_build_sequence_dataloader_with_encoding_returns_expected_history(self):
        t_seq, dataset, _ = preparation.build_sequence_dataloader(
            t=self.t,
            trajectory_data=self.data_set,
            used_outputs=[0, 1],
            used_controls=[0],
            batch_size=2,
            shuffle=False,
            seq_len=2,
            overlap=1,
            historic_seq_len=3,
            encode=True,
        )

        self.assertEqual(len(dataset), 4)
        np.testing.assert_array_equal(
            t_seq,
            np.array(
                [
                    [3, 4],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                ]
            ),
        )

        ((y_h, u_h), u), y = dataset[0]
        self.assertTrue(
            torch.equal(
                y_h,
                torch.tensor(
                    [[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]],
                    dtype=torch.float32,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                u_h,
                torch.tensor([[100.0], [101.0], [102.0]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                u,
                torch.tensor([[103.0], [104.0]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                y,
                torch.tensor([[3.0, 13.0], [4.0, 14.0]], dtype=torch.float32),
            )
        )

    def test_encoding_with_history_one_matches_non_encoded_batches_after_cutoff(
        self,
    ):
        t_seq_encoded, dataset_encoded, _ = preparation.build_sequence_dataloader(
            t=self.t,
            trajectory_data=self.data_set,
            used_outputs=[0, 1],
            used_controls=[0],
            batch_size=2,
            shuffle=False,
            seq_len=2,
            overlap=1,
            historic_seq_len=1,
            encode=True,
        )
        t_seq_plain, dataset_plain, _ = preparation.build_sequence_dataloader(
            t=self.t,
            trajectory_data=self.data_set,
            used_outputs=[0, 1],
            used_controls=[0],
            batch_size=2,
            shuffle=False,
            seq_len=2,
            overlap=1,
            historic_seq_len=1,
            cutoff_samples=1,
            encode=False,
        )

        np.testing.assert_array_equal(t_seq_encoded, t_seq_plain)
        self.assertEqual(len(dataset_encoded), len(dataset_plain))

        for sample_idx in range(len(dataset_encoded)):
            ((y_h_encoded, _), u_encoded), y_encoded = dataset_encoded[
                sample_idx
            ]
            ((y_h_plain, u_h_plain), u_plain), y_plain = dataset_plain[
                sample_idx
            ]

            self.assertEqual(u_h_plain.shape, (0, 1))
            self.assertTrue(torch.equal(y_h_encoded, y_h_plain))
            self.assertTrue(torch.equal(u_encoded, u_plain))
            self.assertTrue(torch.equal(y_encoded, y_plain))

    def test_build_sequence_dataloader_requires_historic_seq_len_one_without_encoding(
        self,
    ):
        with self.assertRaises(AssertionError):
            preparation.build_sequence_dataloader(
                t=self.t,
                trajectory_data=self.data_set,
                used_outputs=[0, 1],
                used_controls=[0],
                batch_size=2,
                shuffle=False,
                seq_len=2,
                overlap=1,
                historic_seq_len=2,
                encode=False,
            )


class TestGenerateBatchSets(unittest.TestCase):
    def setUp(self):
        self.t_vecs = [
            np.arange(6)[None, :],
            np.arange(10, 16)[None, :],
        ]
        self.data_sets = [
            {
                "Y": np.arange(12, dtype=float).reshape(1, 6, 2),
                "U": np.arange(100, 106, dtype=float).reshape(1, 6, 1),
            },
            {
                "Y": np.arange(200, 212, dtype=float).reshape(1, 6, 2),
                "U": np.arange(300, 306, dtype=float).reshape(1, 6, 1),
            },
        ]

    def test_build_sequence_dataloaders_matches_repeated_build_sequence_dataloader(self):
        batch_sets = preparation.build_sequence_dataloaders(
            t_vecs=self.t_vecs,
            trajectory_data_sets=self.data_sets,
            used_outputs=[0, 1],
            used_controls=[0],
            batch_size=2,
            shuffle=False,
            seq_len=2,
            overlap=1,
            historic_seq_len=1,
            encode=True,
        )

        self.assertEqual(len(batch_sets), 2)

        for idx, batch_set in enumerate(batch_sets):
            expected = preparation.build_sequence_dataloader(
                t=self.t_vecs[idx],
                trajectory_data=self.data_sets[idx],
                used_outputs=[0, 1],
                used_controls=[0],
                batch_size=2,
                shuffle=False,
                seq_len=2,
                overlap=1,
                historic_seq_len=1,
                encode=True,
            )

            np.testing.assert_array_equal(batch_set[0], expected[0])
            self.assertEqual(len(batch_set[1]), len(expected[1]))
            self.assertEqual(len(batch_set[2]), len(expected[2]))

            for sample_idx in range(len(batch_set[1])):
                ((y_h_batch, u_h_batch), u_batch), y_batch = batch_set[1][
                    sample_idx
                ]
                ((y_h_expected, u_h_expected), u_expected), y_expected = (
                    expected[1][sample_idx]
                )
                self.assertTrue(torch.equal(y_h_batch, y_h_expected))
                self.assertTrue(torch.equal(u_h_batch, u_h_expected))
                self.assertTrue(torch.equal(u_batch, u_expected))
                self.assertTrue(torch.equal(y_batch, y_expected))


class TestDataPrepLegacyDegeneracies(unittest.TestCase):
    def setUp(self):
        n_tests = 20

        self.osh_datasets = []
        self.seq_datasets_osheq = []
        self.enc_datasets_osheq = []
        self.seq_times = []
        self.enc_times = []
        self.seq_datasets = []
        self.enc_datasets = []
        self.n_used_inputs = []
        self.n_used_states = []

        for _ in range(n_tests):
            n_states = np.random.randint(1, 30)
            n_inputs = np.random.randint(1, 20)
            n_instances = np.random.randint(20, 120)
            y = np.random.randn(1, n_instances, n_states)
            u = np.random.randn(1, n_instances, n_inputs)
            t = np.arange(0, n_instances)[None, :]

            data_set = {"X": y, "Y": y, "U": u}

            used_outputs = sorted(
                random.sample(range(n_states), k=random.randint(1, n_states))
            )
            used_controls = sorted(
                random.sample(range(n_inputs), k=random.randint(1, n_inputs))
            )
            self.n_used_states.append(len(used_outputs))
            self.n_used_inputs.append(len(used_controls))

            batch_size = random.randint(1, 20)
            seq_len = random.randint(2, max(2, n_instances // 4))
            overlap = random.randint(0, seq_len - 1)
            historic_seq_len = random.randint(1, max(1, n_instances // 10))

            _, osh_dataset, _ = onesteppred.create_batches(
                t=t,
                data_set=data_set,
                used_states=used_outputs,
                used_inputs=used_controls,
                batch_size=batch_size,
                shuffle=False,
            )
            self.osh_datasets.append(osh_dataset)

            _, seq_dataset_osheq, _ = sequencepred.create_batches_sequence(
                t=t,
                data_set=data_set,
                used_states=used_outputs,
                used_inputs=used_controls,
                batch_size=batch_size,
                shuffle=False,
                seq_len=2,
                overlap=1,
            )
            self.seq_datasets_osheq.append(seq_dataset_osheq)

            _, enc_dataset_osheq, _ = preparation.build_sequence_dataloader(
                t=t,
                trajectory_data=data_set,
                used_outputs=used_outputs,
                used_controls=used_controls,
                batch_size=batch_size,
                shuffle=False,
                seq_len=2,
                overlap=1,
                historic_seq_len=1,
                encode=False,
            )
            self.enc_datasets_osheq.append(enc_dataset_osheq)

            seq_t, seq_dataset, _ = sequencepred.create_batches_sequence(
                t=t,
                data_set=data_set,
                used_states=used_outputs,
                used_inputs=used_controls,
                batch_size=batch_size,
                shuffle=False,
                seq_len=seq_len,
                overlap=overlap,
                cutoff_samples=historic_seq_len,
            )
            self.seq_times.append(seq_t)
            self.seq_datasets.append(seq_dataset)

            enc_t, enc_dataset, _ = preparation.build_sequence_dataloader(
                t=t,
                trajectory_data=data_set,
                used_outputs=used_outputs,
                used_controls=used_controls,
                batch_size=batch_size,
                shuffle=False,
                seq_len=seq_len,
                overlap=overlap,
                historic_seq_len=historic_seq_len,
                encode=True,
            )
            self.enc_times.append(enc_t)
            self.enc_datasets.append(enc_dataset)

    def test_onestep_sequence_and_nonencoded_batches_are_equivalent(self):
        for ds_idx, (osh_set, seq_set, enc_set) in enumerate(
            zip(
                self.osh_datasets,
                self.seq_datasets_osheq,
                self.enc_datasets_osheq,
            )
        ):
            self.assertEqual(len(osh_set), len(seq_set))
            self.assertEqual(len(osh_set), len(enc_set))

            for sample_idx in range(len(osh_set)):
                in_features, out_features = osh_set[sample_idx]
                x_0_osh = in_features[: self.n_used_states[ds_idx]]
                u_osh = in_features[self.n_used_states[ds_idx] :]
                x_osh = out_features

                (x_0_seq, u_seq), x_seq = seq_set[sample_idx]
                u_seq = u_seq[0, :]
                x_seq = x_seq[1, :]

                ((y_h_enc, u_h_enc), u_enc), y_enc = enc_set[sample_idx]
                x_0_enc = y_h_enc[-1, :]
                u_enc = u_enc[0, :]
                x_enc = y_enc[1, :]

                self.assertTrue(
                    torch.allclose(x_0_osh, x_0_seq, atol=1e-20, rtol=0)
                )
                self.assertTrue(
                    torch.allclose(x_0_osh, x_0_enc, atol=1e-20, rtol=0)
                )
                self.assertEqual(u_h_enc.shape[0], 0)
                self.assertTrue(
                    torch.allclose(u_osh, u_seq, atol=1e-20, rtol=0)
                )
                self.assertTrue(
                    torch.allclose(u_osh, u_enc, atol=1e-20, rtol=0)
                )
                self.assertTrue(
                    torch.allclose(x_osh, x_seq, atol=1e-20, rtol=0)
                )
                self.assertTrue(
                    torch.allclose(x_osh, x_enc, atol=1e-20, rtol=0)
                )

    def test_encoded_batches_degenerate_to_sequence_batches_with_cutoff(self):
        for seq_t, enc_t, seq_set, enc_set in zip(
            self.seq_times,
            self.enc_times,
            self.seq_datasets,
            self.enc_datasets,
        ):
            self.assertTrue(np.array_equal(seq_t.squeeze(), enc_t.squeeze()))
            self.assertEqual(len(seq_set), len(enc_set))

            for sample_idx in range(len(seq_set)):
                (x_0_seq, u_seq), x_seq = seq_set[sample_idx]
                ((y_h_enc, _), u_enc), y_enc = enc_set[sample_idx]
                x_0_enc = y_h_enc[-1, :]

                self.assertTrue(
                    torch.allclose(x_0_seq, x_0_enc, atol=1e-20, rtol=0)
                )
                self.assertTrue(
                    torch.allclose(u_seq, u_enc, atol=1e-20, rtol=0)
                )
                self.assertTrue(
                    torch.allclose(x_seq, y_enc, atol=1e-20, rtol=0)
                )


if __name__ == "__main__":
    unittest.main()
