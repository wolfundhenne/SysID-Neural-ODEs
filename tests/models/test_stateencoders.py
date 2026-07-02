import torch
import unittest

from neuralsysid.models import (
    stateencoders,
)


class SelectLastStepFeatures(torch.nn.Module):
    def __init__(self, feature_indices):
        super().__init__()
        self.feature_indices = feature_indices

    def forward(self, x):
        return x[:, -1, self.feature_indices]


class CaptureModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        return x


class CaptureLastStepModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        if x.dim() == 3:
            return x[:, -1, :]
        return x


class TestPartialStateEncoder(unittest.TestCase):
    def test_sequence_mode_uses_sequence_interface(self):
        batch_size, seq_len, n_y, n_u = 4, 3, 2, 5
        encoder_core = torch.nn.Identity()
        encoder = stateencoders.PartialStateEncoder(
            encoder=encoder_core,
            input_mode="sequence",
            input_sources="yu",
        )

        y_sequence = torch.randn(batch_size, seq_len, n_y)
        u_sequence = torch.randn(batch_size, seq_len, n_u)

        out = encoder(y_sequence, u_sequence)

        self.assertTrue(
            torch.allclose(out, torch.cat((y_sequence, u_sequence), dim=-1))
        )

    def test_vector_mode_flattens_selected_inputs(self):
        batch_size, seq_len, n_y = 4, 3, 2
        encoder_core = torch.nn.Identity()
        encoder = stateencoders.PartialStateEncoder(
            encoder=encoder_core,
            input_mode="vector",
            input_sources="y",
        )

        y_sequence = torch.randn(batch_size, seq_len, n_y)
        u_sequence = torch.randn(batch_size, seq_len, 1)

        out = encoder(y_sequence, u_sequence)

        self.assertTrue(
            torch.allclose(out, y_sequence.reshape(batch_size, -1))
        )

    def test_outputs_as_latent_states_are_prepended(self):
        batch_size, seq_len, n_y, n_u = 4, 3, 3, 2
        encoder_core = torch.nn.Identity()
        encoder = stateencoders.PartialStateEncoder(
            encoder=encoder_core,
            input_mode="vector",
            input_sources="u",
            outputs_as_latent_states=[0, 2],
        )

        y_sequence = torch.randn(batch_size, seq_len, n_y)
        u_sequence = torch.randn(batch_size, seq_len, n_u)

        out = encoder(y_sequence, u_sequence)
        expected = torch.cat(
            (
                y_sequence[:, -1, [0, 2]],
                u_sequence.reshape(batch_size, -1),
            ),
            dim=-1,
        )

        self.assertTrue(torch.allclose(out, expected))

    def test_sequence_mode_without_batch_dimension(self):
        seq_len, n_y, n_u = 3, 2, 2
        encoder_core = torch.nn.Identity()
        encoder = stateencoders.PartialStateEncoder(
            encoder=encoder_core,
            input_mode="sequence",
            input_sources="yu",
        )

        y_sequence = torch.randn(seq_len, n_y)
        u_sequence = torch.randn(seq_len, n_u)

        out = encoder(y_sequence, u_sequence)

        self.assertTrue(
            torch.allclose(out, torch.cat((y_sequence, u_sequence), dim=-1))
        )

    def test_vector_mode_without_batch_dimension(self):
        seq_len, n_y, n_u = 3, 2, 1
        encoder_core = torch.nn.Identity()
        encoder = stateencoders.PartialStateEncoder(
            encoder=encoder_core,
            input_mode="vector",
            input_sources="yu",
        )

        y_sequence = torch.randn(seq_len, n_y)
        u_sequence = torch.randn(seq_len, n_u)

        out = encoder(y_sequence, u_sequence)

        self.assertTrue(
            torch.allclose(
                out,
                torch.cat((y_sequence, u_sequence), dim=-1).reshape(
                    seq_len, -1
                ),
            )
        )


class TestIdentityStateEncoder(unittest.TestCase):
    def test_returns_last_measurement_for_batched_sequence(self):
        encoder = stateencoders.IdentityStateEncoder()
        y_sequence = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ]
        )
        u_sequence = torch.tensor(
            [
                [[0.0], [1.0], [2.0]],
                [[3.0], [4.0], [5.0]],
            ]
        )

        out = encoder(y_sequence, u_sequence)

        self.assertTrue(torch.equal(out, y_sequence[:, -1, :]))

    def test_returns_last_measurement_for_single_sample_sequence(self):
        encoder = stateencoders.IdentityStateEncoder()
        y_sequence = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        u_sequence = torch.tensor([[[5.0], [6.0]]])

        out = encoder(y_sequence, u_sequence)

        self.assertTrue(torch.equal(out, torch.tensor([[3.0, 4.0]])))

    def test_returns_last_measurement_without_batch_dimension(self):
        encoder = stateencoders.IdentityStateEncoder()
        y_sequence = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        u_sequence = torch.tensor([[7.0], [8.0], [9.0]])

        out = encoder(y_sequence, u_sequence)

        self.assertTrue(torch.equal(out, torch.tensor([5.0, 6.0])))


class TestGraphStateEncoder(unittest.TestCase):
    def test_returns_state_only_when_n_context_node_is_zero(self):
        encoder = stateencoders.GraphStateEncoder(
            encoder=SelectLastStepFeatures([0]),
            n_states_node=1,
            n_inputs_node=1,
            n_outputs_node=1,
            n_context_node=0,
            n_nodes=2,
        )
        y_sequence = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        u_sequence = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])

        x_0 = encoder(y_sequence, u_sequence)

        self.assertFalse(isinstance(x_0, tuple))
        self.assertTrue(torch.equal(x_0, torch.tensor([[3.0, 4.0]])))

    def test_returns_state_and_context_when_context_is_configured(self):
        encoder = stateencoders.GraphStateEncoder(
            encoder=SelectLastStepFeatures([0, 1]),
            n_states_node=1,
            n_inputs_node=1,
            n_outputs_node=1,
            n_context_node=1,
            n_nodes=2,
        )
        y_sequence = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        u_sequence = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])

        x_0, ctx = encoder(y_sequence, u_sequence)

        self.assertTrue(torch.equal(x_0, torch.tensor([[3.0, 4.0]])))
        self.assertTrue(torch.equal(ctx, torch.tensor([[[7.0], [8.0]]])))

    def test_prepends_outputs_as_latent_states_before_state_context_split(
        self,
    ):
        encoder = stateencoders.GraphStateEncoder(
            encoder=SelectLastStepFeatures([0, 2]),
            n_states_node=2,
            n_inputs_node=1,
            n_outputs_node=2,
            n_context_node=1,
            n_nodes=2,
            outputs_as_latent_states=[1],
        )
        y_sequence = torch.tensor(
            [[[1.0, 10.0, 2.0, 20.0], [3.0, 30.0, 4.0, 40.0]]]
        )
        u_sequence = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])

        x_0, ctx = encoder(y_sequence, u_sequence)

        self.assertTrue(
            torch.equal(x_0, torch.tensor([[30.0, 3.0, 40.0, 4.0]]))
        )
        self.assertTrue(torch.equal(ctx, torch.tensor([[[7.0], [8.0]]])))

    def test_uses_configured_input_sources_for_nodewise_encoding(self):
        encoder_core = CaptureLastStepModule()
        encoder = stateencoders.GraphStateEncoder(
            encoder=encoder_core,
            n_states_node=1,
            n_inputs_node=1,
            n_outputs_node=1,
            n_context_node=0,
            n_nodes=2,
            input_mode="sequence",
            input_sources="y",
        )
        y_sequence = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        u_sequence = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])

        encoder(y_sequence, u_sequence)

        expected = torch.tensor(
            [
                [[1.0], [3.0]],
                [[2.0], [4.0]],
            ]
        )
        self.assertTrue(torch.equal(encoder_core.last_input, expected))

    def test_flattens_local_node_history_in_vector_mode(self):
        encoder_core = CaptureLastStepModule()
        encoder = stateencoders.GraphStateEncoder(
            encoder=encoder_core,
            n_states_node=2,
            n_inputs_node=1,
            n_outputs_node=1,
            n_context_node=0,
            n_nodes=2,
            input_mode="vector",
            input_sources="yu",
        )
        y_sequence = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        u_sequence = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])

        encoder(y_sequence, u_sequence)

        expected = torch.tensor(
            [
                [1.0, 5.0, 3.0, 7.0],
                [2.0, 6.0, 4.0, 8.0],
            ]
        )
        self.assertTrue(torch.equal(encoder_core.last_input, expected))


if __name__ == "__main__":
    unittest.main()
