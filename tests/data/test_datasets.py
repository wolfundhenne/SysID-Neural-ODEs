import torch
import unittest

from torch.utils.data import DataLoader

from neuralsysid.data.datasets import SequenceDatasetHistoricEncoding


class TestSequenceDatasetHistoricEncoding(unittest.TestCase):
    def setUp(self):
        self.y_h = torch.arange(2 * 3 * 2, dtype=torch.float32).view(2, 3, 2)
        self.u_h = torch.arange(
            100, 100 + 2 * 3 * 1, dtype=torch.float32
        ).view(2, 3, 1)
        self.y = torch.arange(200, 200 + 2 * 4 * 2, dtype=torch.float32).view(
            2, 4, 2
        )
        self.u = torch.arange(300, 300 + 2 * 4 * 1, dtype=torch.float32).view(
            2, 4, 1
        )
        self.dataset = SequenceDatasetHistoricEncoding(
            y_h=self.y_h,
            u_h=self.u_h,
            y=self.y,
            u=self.u,
        )

    def test_length_matches_number_of_sequences(self):
        self.assertEqual(len(self.dataset), self.y.shape[0])

    def test_getitem_returns_expected_first_sample_structure(self):
        ((y_h, u_h), u), y = self.dataset[0]

        self.assertTrue(torch.equal(y_h, self.y_h[0]))
        self.assertTrue(torch.equal(u_h, self.u_h[0]))
        self.assertTrue(torch.equal(u, self.u[0]))
        self.assertTrue(torch.equal(y, self.y[0]))

    def test_getitem_returns_expected_last_sample_structure(self):
        ((y_h, u_h), u), y = self.dataset[len(self.dataset) - 1]

        self.assertTrue(torch.equal(y_h, self.y_h[-1]))
        self.assertTrue(torch.equal(u_h, self.u_h[-1]))
        self.assertTrue(torch.equal(u, self.u[-1]))
        self.assertTrue(torch.equal(y, self.y[-1]))

    def test_dataloader_collates_nested_structure(self):
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False)

        ((y_h_batch, u_h_batch), u_batch), y_batch = next(iter(dataloader))

        self.assertEqual(y_h_batch.shape, self.y_h.shape)
        self.assertEqual(u_h_batch.shape, self.u_h.shape)
        self.assertEqual(u_batch.shape, self.u.shape)
        self.assertEqual(y_batch.shape, self.y.shape)
        self.assertTrue(torch.equal(y_h_batch, self.y_h))
        self.assertTrue(torch.equal(u_h_batch, self.u_h))
        self.assertTrue(torch.equal(u_batch, self.u))
        self.assertTrue(torch.equal(y_batch, self.y))


if __name__ == "__main__":
    unittest.main()
