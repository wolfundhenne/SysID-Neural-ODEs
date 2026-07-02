import unittest

import torch

from neuralsysid.models import outputdecoders as outputfunctions


class SliceRecorder(torch.nn.Module):
    def __init__(self, n_outputs: int):
        super().__init__()
        self.n_outputs = n_outputs
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        return x[..., : self.n_outputs]


class TestPartialOutputDecoder(unittest.TestCase):
    def test_uses_only_state_when_input_sources_is_x(self):
        decoder_core = SliceRecorder(n_outputs=1)
        decoder = outputfunctions.PartialOutputDecoder(
            n_states=2,
            n_inputs=1,
            n_outputs=1,
            decoder=decoder_core,
            input_sources="x",
        )
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        u = torch.tensor([[[10.0], [20.0]]])

        y = decoder(x, u)

        self.assertTrue(torch.equal(decoder_core.last_input, x))
        self.assertTrue(torch.equal(y, x[..., :1]))

    def test_uses_state_and_control_when_input_sources_is_xu(self):
        decoder_core = SliceRecorder(n_outputs=1)
        decoder = outputfunctions.PartialOutputDecoder(
            n_states=2,
            n_inputs=1,
            n_outputs=1,
            decoder=decoder_core,
            input_sources="xu",
        )
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        u = torch.tensor([[[10.0], [20.0]]])
        expected_input = torch.tensor([[[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]]])

        y = decoder(x, u)

        self.assertTrue(torch.equal(decoder_core.last_input, expected_input))
        self.assertTrue(torch.equal(y, expected_input[..., :1]))

    def test_combines_outputs_as_latent_states_with_learned_outputs(self):
        decoder_core = SliceRecorder(n_outputs=1)
        decoder = outputfunctions.PartialOutputDecoder(
            n_states=2,
            n_inputs=1,
            n_outputs=2,
            decoder=decoder_core,
            outputs_as_latent_states=[1],
            input_sources="x",
        )
        x = torch.tensor([[[5.0, 6.0]]])
        u = torch.tensor([[[7.0]]])

        y = decoder(x, u)
        expected = torch.tensor([[[5.0, 5.0]]])

        self.assertTrue(torch.equal(y, expected))

    def test_all_outputs_as_latent_states_does_not_require_decoder_network(
        self,
    ):
        decoder = outputfunctions.PartialOutputDecoder(
            n_states=2,
            n_inputs=1,
            n_outputs=2,
            outputs_as_latent_states=[0, 1],
        )
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        u = torch.tensor([[[5.0], [6.0]]])

        y = decoder(x, u)

        self.assertIsNone(decoder.decoder)
        self.assertTrue(torch.equal(y, x))

    def test_requires_decoder_when_outputs_are_not_states(self):
        with self.assertRaises(ValueError):
            outputfunctions.PartialOutputDecoder(
                n_states=2,
                n_inputs=1,
                n_outputs=2,
                outputs_as_latent_states=[0],
            )

    def test_rejects_invalid_input_sources(self):
        with self.assertRaises(ValueError):
            outputfunctions.PartialOutputDecoder(
                n_states=2,
                n_inputs=1,
                n_outputs=1,
                decoder=SliceRecorder(n_outputs=1),
                input_sources="u",
            )

    def test_accepts_inputs_without_batch_dimension(self):
        decoder_core = SliceRecorder(n_outputs=1)
        decoder = outputfunctions.PartialOutputDecoder(
            n_states=2,
            n_inputs=1,
            n_outputs=1,
            decoder=decoder_core,
            input_sources="xu",
        )
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        u = torch.tensor([[10.0], [20.0]])

        y = decoder(x, u)

        expected_input = torch.tensor([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]])
        self.assertTrue(torch.equal(decoder_core.last_input, expected_input))
        self.assertEqual(y.shape, (2, 1))
        self.assertTrue(torch.equal(y, expected_input[..., :1]))

    def test_accepts_single_state_without_batch_or_time_dimension(self):
        decoder_core = SliceRecorder(n_outputs=1)
        decoder = outputfunctions.PartialOutputDecoder(
            n_states=2,
            n_inputs=1,
            n_outputs=1,
            decoder=decoder_core,
            input_sources="xu",
        )
        x = torch.tensor([1.0, 2.0])
        u = torch.tensor([10.0])

        y = decoder(x, u)

        expected_input = torch.tensor([1.0, 2.0, 10.0])
        self.assertTrue(torch.equal(decoder_core.last_input, expected_input))
        self.assertEqual(y.shape, (1,))
        self.assertTrue(torch.equal(y, expected_input[:1]))


class TestGraphOutputDecoder(unittest.TestCase):
    def test_uses_only_state_when_input_sources_is_x(self):
        decoder_core = SliceRecorder(n_outputs=1)
        decoder = outputfunctions.GraphOutputDecoder(
            n_states_node=2,
            n_inputs_node=1,
            n_outputs_node=1,
            n_ctx_node=0,
            n_node_embedding=0,
            n_nodes=2,
            decoder=decoder_core,
            input_sources="x",
        )
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        u = torch.tensor([[[10.0, 20.0]]])

        y = decoder(x, u)

        expected_input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        expected_output = torch.tensor([[[1.0, 3.0]]])
        self.assertTrue(torch.equal(decoder_core.last_input, expected_input))
        self.assertTrue(torch.equal(y, expected_output))

    def test_uses_state_control_context_and_node_embeddings(self):
        decoder_core = SliceRecorder(n_outputs=1)
        decoder = outputfunctions.GraphOutputDecoder(
            n_states_node=1,
            n_inputs_node=1,
            n_outputs_node=1,
            n_ctx_node=1,
            n_node_embedding=1,
            n_nodes=2,
            decoder=decoder_core,
            input_sources="xu",
        )
        with torch.no_grad():
            decoder.node_embedding.copy_(torch.tensor([[[0.5], [1.5]]]))
        decoder.set_context(torch.tensor([[[2.0], [3.0]]]))
        x = torch.tensor([[[4.0, 5.0]]])
        u = torch.tensor([[[6.0, 7.0]]])

        y = decoder(x, u)

        expected_input = torch.tensor(
            [[[[4.0, 6.0, 0.5, 2.0], [5.0, 7.0, 1.5, 3.0]]]]
        )
        expected_output = torch.tensor([[[4.0, 5.0]]])
        self.assertTrue(torch.equal(decoder_core.last_input, expected_input))
        self.assertTrue(torch.equal(y, expected_output))

    def test_combines_outputs_as_latent_states_with_learned_outputs(self):
        decoder_core = SliceRecorder(n_outputs=1)
        decoder = outputfunctions.GraphOutputDecoder(
            n_states_node=2,
            n_inputs_node=1,
            n_outputs_node=2,
            n_ctx_node=0,
            n_node_embedding=0,
            n_nodes=2,
            decoder=decoder_core,
            outputs_as_latent_states=[1],
            input_sources="x",
        )
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        u = torch.tensor([[[5.0, 6.0]]])

        y = decoder(x, u)

        expected = torch.tensor([[[1.0, 1.0, 3.0, 3.0]]])
        self.assertTrue(torch.equal(y, expected))

    def test_broadcasts_single_sample_context_across_batch(self):
        decoder_core = SliceRecorder(n_outputs=1)
        decoder = outputfunctions.GraphOutputDecoder(
            n_states_node=1,
            n_inputs_node=1,
            n_outputs_node=1,
            n_ctx_node=1,
            n_node_embedding=0,
            n_nodes=2,
            decoder=decoder_core,
            input_sources="x",
        )
        decoder.set_context(torch.tensor([[2.0], [3.0]]))
        x = torch.tensor(
            [
                [[4.0, 5.0]],
                [[6.0, 7.0]],
            ]
        )
        u = torch.tensor(
            [
                [[8.0, 9.0]],
                [[10.0, 11.0]],
            ]
        )

        y = decoder(x, u)

        expected_input = torch.tensor(
            [
                [[[4.0, 2.0], [5.0, 3.0]]],
                [[[6.0, 2.0], [7.0, 3.0]]],
            ]
        )
        expected_output = torch.tensor(
            [
                [[4.0, 5.0]],
                [[6.0, 7.0]],
            ]
        )
        self.assertTrue(torch.equal(decoder_core.last_input, expected_input))
        self.assertTrue(torch.equal(y, expected_output))

    def test_all_outputs_as_latent_states_does_not_require_decoder_network(
        self,
    ):
        decoder = outputfunctions.GraphOutputDecoder(
            n_states_node=2,
            n_inputs_node=1,
            n_outputs_node=2,
            n_ctx_node=1,
            n_node_embedding=1,
            n_nodes=2,
            outputs_as_latent_states=[0, 1],
            input_sources="xu",
        )
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        u = torch.tensor([[[5.0, 6.0]]])

        y = decoder(x, u)

        self.assertIsNone(decoder.decoder)
        self.assertTrue(torch.equal(y, x))

    def test_rejects_invalid_input_sources(self):
        with self.assertRaises(ValueError):
            outputfunctions.GraphOutputDecoder(
                n_states_node=1,
                n_inputs_node=1,
                n_outputs_node=1,
                n_ctx_node=0,
                n_node_embedding=0,
                n_nodes=1,
                decoder=SliceRecorder(n_outputs=1),
                input_sources="u",
            )

    def test_accepts_unbatched_trajectory_input(self):
        decoder_core = SliceRecorder(n_outputs=1)
        decoder = outputfunctions.GraphOutputDecoder(
            n_states_node=1,
            n_inputs_node=1,
            n_outputs_node=1,
            n_ctx_node=1,
            n_node_embedding=1,
            n_nodes=2,
            decoder=decoder_core,
            input_sources="xu",
        )
        with torch.no_grad():
            decoder.node_embedding.copy_(torch.tensor([[[0.5], [1.5]]]))
        decoder.set_context(torch.tensor([[2.0], [3.0]]))
        x = torch.tensor([[4.0, 5.0], [6.0, 7.0]])
        u = torch.tensor([[8.0, 9.0], [10.0, 11.0]])

        y = decoder(x, u)

        expected_input = torch.tensor(
            [
                [[4.0, 8.0, 0.5, 2.0], [5.0, 9.0, 1.5, 3.0]],
                [[6.0, 10.0, 0.5, 2.0], [7.0, 11.0, 1.5, 3.0]],
            ]
        )
        expected_output = torch.tensor([[4.0, 5.0], [6.0, 7.0]])
        self.assertTrue(torch.equal(decoder_core.last_input, expected_input))
        self.assertEqual(y.shape, (2, 2))
        self.assertTrue(torch.equal(y, expected_output))

    def test_accepts_batched_single_step_input(self):
        decoder_core = SliceRecorder(n_outputs=1)
        decoder = outputfunctions.GraphOutputDecoder(
            n_states_node=1,
            n_inputs_node=1,
            n_outputs_node=1,
            n_ctx_node=1,
            n_node_embedding=1,
            n_nodes=2,
            decoder=decoder_core,
            input_sources="xu",
        )
        with torch.no_grad():
            decoder.node_embedding.copy_(torch.tensor([[[0.5], [1.5]]]))
        decoder.set_context(torch.tensor([[2.0], [3.0]]))
        x = torch.tensor(
            [
                [4.0, 5.0],
                [6.0, 7.0],
            ]
        )
        u = torch.tensor(
            [
                [8.0, 9.0],
                [10.0, 11.0],
            ]
        )

        y = decoder(x, u)

        expected_input = torch.tensor(
            [
                [[4.0, 8.0, 0.5, 2.0], [5.0, 9.0, 1.5, 3.0]],
                [[6.0, 10.0, 0.5, 2.0], [7.0, 11.0, 1.5, 3.0]],
            ]
        )
        expected_output = torch.tensor(
            [
                [4.0, 5.0],
                [6.0, 7.0],
            ]
        )
        self.assertTrue(torch.equal(decoder_core.last_input, expected_input))
        self.assertEqual(y.shape, (2, 2))
        self.assertTrue(torch.equal(y, expected_output))

    def test_accepts_single_state_without_batch_dimension(self):
        decoder_core = SliceRecorder(n_outputs=1)
        decoder = outputfunctions.GraphOutputDecoder(
            n_states_node=1,
            n_inputs_node=1,
            n_outputs_node=1,
            n_ctx_node=0,
            n_node_embedding=0,
            n_nodes=2,
            decoder=decoder_core,
            input_sources="x",
        )
        x = torch.tensor([1.0, 2.0])
        u = torch.tensor([3.0, 4.0])

        y = decoder(x, u)

        expected_input = torch.tensor([[1.0], [2.0]])
        expected_output = torch.tensor([1.0, 2.0])
        self.assertTrue(torch.equal(decoder_core.last_input, expected_input))
        self.assertEqual(y.shape, (2,))
        self.assertTrue(torch.equal(y, expected_output))


if __name__ == "__main__":
    unittest.main()
