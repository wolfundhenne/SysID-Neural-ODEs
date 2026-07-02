import numpy as np
import random
import torch
import unittest

# own ML models
import neuralsysid.models.neuralnetworks as nnets

# library ML models
import neuromancer.modules.blocks as nmblocks

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

from neuralsysid.utils.helpers import initialize_weights_and_biases


class TestMultiLayerPerceptron(unittest.TestCase):
    # this class should test single blocks of neural networks, like mlp

    def setUp(self):
        n_tests = 500  # 10000
        n_inputs = np.random.randint(1, 100, n_tests)
        n_outputs = np.random.randint(1, 100, n_tests)
        n_hidden = np.random.randint(1, 100, n_tests)
        n_layers = np.random.randint(1, 10, n_tests)

        # n_tests random activation functions
        act_fcts = [
            torch.nn.ReLU,
            torch.nn.Sigmoid,
            torch.nn.Tanh,
        ]
        act_fct = [
            act_fcts[np.random.randint(0, len(act_fcts))]
            for _ in range(n_tests)
        ]

        self.mlps = []
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.act_fct = act_fct

        for n_in, n_out, n_hid, n_lay, act in zip(
            n_inputs, n_outputs, n_hidden, n_layers, act_fct
        ):

            my_mlp = nnets.MultiLayerPerceptron(
                n_inputs=n_in,
                n_outputs=n_out,
                n_layers=n_lay,
                n_hidden=n_hid,
                activation=act,
            )

            nm_mlp = nmblocks.MLP(
                insize=n_in,
                outsize=n_out,
                bias=True,
                linear_map=torch.nn.Linear,
                nonlin=act,
                hsizes=[n_hid] * n_lay,
            )

            sizes = [n_in] + [n_hid] * n_lay + [n_out]

            weight_inits = [
                torch.randn(sizes[ii + 1], sizes[ii])
                for ii in range(len(sizes) - 1)
            ]
            bias_inits = [
                torch.randn(sizes[ii + 1]) for ii in range(len(sizes) - 1)
            ]

            initialize_weights_and_biases(my_mlp.mlp, weight_inits, bias_inits)

            initialize_weights_and_biases(
                nm_mlp.linear, weight_inits, bias_inits
            )

            self.mlps.append((my_mlp, nm_mlp))

        return

    def test_MLP_outputs(self):

        for idx, (n_in, n_out, n_hid, n_lay, act) in enumerate(
            zip(
                self.n_inputs,
                self.n_outputs,
                self.n_hidden,
                self.n_layers,
                self.act_fct,
            )
        ):

            my_mlp, nm_mlp = self.mlps[idx]

            test_inputs = torch.randn(int(np.random.randint(1, 100)), n_in)

            my_out = my_mlp(test_inputs)
            nm_out = nm_mlp(test_inputs)

            self.assertTrue(torch.equal(my_out, nm_out))

    def test_mlp_module_structure_single_hidden_layer(self):
        mlp = nnets.MultiLayerPerceptron(
            n_inputs=3,
            n_outputs=2,
            n_layers=1,
            n_hidden=5,
            activation=torch.nn.Tanh,
            dropout=0.25,
        )

        modules = list(mlp.mlp)

        self.assertEqual(len(modules), 4)
        self.assertIsInstance(modules[0], torch.nn.Linear)
        self.assertEqual(modules[0].in_features, 3)
        self.assertEqual(modules[0].out_features, 5)
        self.assertIsInstance(modules[1], torch.nn.Tanh)
        self.assertIsInstance(modules[2], torch.nn.Dropout)
        self.assertAlmostEqual(modules[2].p, 0.25)
        self.assertIsInstance(modules[3], torch.nn.Linear)
        self.assertEqual(modules[3].in_features, 5)
        self.assertEqual(modules[3].out_features, 2)

    def test_mlp_module_structure_multiple_hidden_layers(self):
        mlp = nnets.MultiLayerPerceptron(
            n_inputs=4,
            n_outputs=1,
            n_layers=3,
            n_hidden=6,
            activation=torch.nn.ReLU,
            dropout=0.1,
        )

        modules = list(mlp.mlp)

        self.assertEqual(len(modules), 10)
        self.assertTrue(
            all(
                isinstance(modules[idx], torch.nn.Linear)
                for idx in [0, 3, 6, 9]
            )
        )
        self.assertTrue(
            all(isinstance(modules[idx], torch.nn.ReLU) for idx in [1, 4, 7])
        )
        self.assertTrue(
            all(
                isinstance(modules[idx], torch.nn.Dropout) for idx in [2, 5, 8]
            )
        )

    def test_mlp_dropout_is_deterministic_in_eval_mode(self):
        torch.manual_seed(1)
        mlp = nnets.MultiLayerPerceptron(
            n_inputs=3,
            n_outputs=2,
            n_layers=2,
            n_hidden=4,
            dropout=0.5,
        )
        test_inputs = torch.randn(5, 3)

        mlp.eval()
        first_out = mlp(test_inputs)
        second_out = mlp(test_inputs)

        self.assertTrue(torch.equal(first_out, second_out))

    def test_mlp_dropout_changes_outputs_in_train_mode(self):
        torch.manual_seed(2)
        mlp = nnets.MultiLayerPerceptron(
            n_inputs=3,
            n_outputs=2,
            n_layers=2,
            n_hidden=4,
            dropout=0.5,
        )
        test_inputs = torch.randn(5, 3)

        mlp.train()
        first_out = mlp(test_inputs)
        second_out = mlp(test_inputs)

        self.assertFalse(torch.equal(first_out, second_out))


class TestRecurrentEncoder(unittest.TestCase):
    # this class should test single blocks of neural networks, like mlp

    def setUp(self):
        n_tests = 500  # 10000
        n_inputs = np.random.randint(1, 100, n_tests)
        n_outputs = np.random.randint(1, 100, n_tests)
        n_hidden = np.random.randint(1, 100, n_tests)
        n_layers = np.random.randint(1, 10, n_tests)

        # n_tests random activation functions
        act_fcts = ["relu", "tanh"]
        act_fct = [
            act_fcts[np.random.randint(0, len(act_fcts))]
            for _ in range(n_tests)
        ]
        encoder_types = ["rnn", "lstm", "gru"]
        encoder_type = [
            encoder_types[np.random.randint(0, len(encoder_types))]
            for _ in range(n_tests)
        ]

        self.encoders = []
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.act_fct = act_fct

        for n_in, n_out, n_hid, n_lay, act, enc_type in zip(
            n_inputs, n_outputs, n_hidden, n_layers, act_fct, encoder_type
        ):

            if enc_type == "rnn":
                my_encoder = nnets.RNNEncoder(
                    n_inputs=int(n_in),
                    n_outputs=int(n_out),
                    n_hidden=int(n_hid),
                    n_layers=int(n_lay),
                    nonlin=act,
                )
            elif enc_type == "lstm":
                my_encoder = nnets.LSTMEncoder(
                    n_inputs=int(n_in),
                    n_outputs=int(n_out),
                    n_hidden=int(n_hid),
                    n_layers=int(n_lay),
                )

            elif enc_type == "gru":
                my_encoder = nnets.GRUEncoder(
                    n_inputs=int(n_in),
                    n_outputs=int(n_out),
                    n_hidden=int(n_hid),
                    n_layers=int(n_lay),
                )
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")

            self.encoders.append(my_encoder)

        return

    def test_output_dimensions_sample(self):

        for idx, (n_in, n_out, n_hid, n_lay, act) in enumerate(
            zip(
                self.n_inputs,
                self.n_outputs,
                self.n_hidden,
                self.n_layers,
                self.act_fct,
            )
        ):

            my_encoder = self.encoders[idx]

            seq_len = torch.randint(1, 100, (1,)).item()

            test_input_sequence_sample = torch.randn(seq_len, n_in)

            my_out = my_encoder(test_input_sequence_sample)

            self.assertTrue(my_out.shape == (n_out,))

    def test_output_dimensions_batch(self):

        for idx, (n_in, n_out, n_hid, n_lay, act) in enumerate(
            zip(
                self.n_inputs,
                self.n_outputs,
                self.n_hidden,
                self.n_layers,
                self.act_fct,
            )
        ):

            my_encoder = self.encoders[idx]

            seq_len = torch.randint(1, 100, (1,)).item()
            batch_size = torch.randint(1, 100, (1,)).item()

            test_input_sequence_sample = torch.randn(batch_size, seq_len, n_in)

            my_out = my_encoder(test_input_sequence_sample)

            self.assertTrue(my_out.shape == (batch_size, n_out))

    def test_single_timestep_sequence_is_accepted_for_all_recurrent_encoders(
        self,
    ):
        encoders = [
            nnets.RNNEncoder(
                n_inputs=3,
                n_outputs=2,
                n_hidden=4,
                n_layers=1,
                nonlin="relu",
            ),
            nnets.LSTMEncoder(
                n_inputs=3,
                n_outputs=2,
                n_hidden=4,
                n_layers=1,
            ),
            nnets.GRUEncoder(
                n_inputs=3,
                n_outputs=2,
                n_hidden=4,
                n_layers=1,
            ),
        ]
        in_sequence = torch.randn(1, 3)

        for encoder in encoders:
            out = encoder(in_sequence)
            self.assertEqual(out.shape, (2,))

    def test_recurrent_dropout_is_deterministic_in_eval_mode(self):
        torch.manual_seed(3)
        encoder = nnets.LSTMEncoder(
            n_inputs=3,
            n_outputs=2,
            n_hidden=4,
            n_layers=2,
            dropout=0.4,
        )
        in_sequence = torch.randn(5, 6, 3)

        encoder.eval()
        first_out = encoder(in_sequence)
        second_out = encoder(in_sequence)

        self.assertTrue(torch.equal(first_out, second_out))

    def test_recurrent_dropout_changes_outputs_in_train_mode(self):
        torch.manual_seed(4)
        encoder = nnets.GRUEncoder(
            n_inputs=3,
            n_outputs=2,
            n_hidden=4,
            n_layers=2,
            dropout=0.4,
        )
        in_sequence = torch.randn(5, 6, 3)

        encoder.train()
        first_out = encoder(in_sequence)
        second_out = encoder(in_sequence)

        self.assertFalse(torch.equal(first_out, second_out))


class TestTCNEncoder(unittest.TestCase):

    def setUp(self):
        n_tests = 500  # 10000
        n_inputs = np.random.randint(1, 100, n_tests)
        n_outputs = np.random.randint(1, 100, n_tests)
        n_hidden = np.random.randint(1, 100, n_tests)
        n_layers = np.random.randint(1, 10, n_tests)

        # n_tests random activation functions
        act_fcts = [torch.nn.ReLU, torch.nn.Tanh]
        act_fct = [
            act_fcts[np.random.randint(0, len(act_fcts))]
            for _ in range(n_tests)
        ]
        encoder_types = ["nonres", "res"]
        encoder_type = [
            encoder_types[np.random.randint(0, len(encoder_types))]
            for _ in range(n_tests)
        ]

        self.encoders = []
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.act_fct = act_fct

        for n_in, n_out, n_hid, n_lay, act, enc_type in zip(
            n_inputs, n_outputs, n_hidden, n_layers, act_fct, encoder_type
        ):
            ker_size = random.randint(3, 8)
            dil_base = random.randint(1, ker_size - 1)
            lev = random.randint(1, 5)

            if enc_type == "nonres":
                my_encoder = nnets.TCNEncoder(
                    input_channels=n_in,
                    n_outputs=n_out,
                    hidden_channels=n_hid,
                    kernel_size=ker_size,
                    dilation_base=dil_base,
                    levels=lev,
                    activation=act,
                )
            elif enc_type == "res":
                my_encoder = nnets.ResTCNEncoder(
                    input_channels=n_in,
                    n_outputs=n_out,
                    hidden_channels=n_hid,
                    kernel_size=ker_size,
                    dilation_base=dil_base,
                    levels=lev,
                    activation=act,
                )
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")

            self.encoders.append(my_encoder)

        return

    def test_output_dimensions_sample(self):

        for idx, (n_in, n_out, n_hid, n_lay, act) in enumerate(
            zip(
                self.n_inputs,
                self.n_outputs,
                self.n_hidden,
                self.n_layers,
                self.act_fct,
            )
        ):

            my_encoder = self.encoders[idx]

            seq_len = torch.randint(10, 100, (1,)).item()

            test_input_sequence_sample = torch.randn(seq_len, n_in)

            my_out = my_encoder(test_input_sequence_sample)

            self.assertTrue(my_out.shape == (n_out,))

    def test_output_dimensions_batch(self):

        for idx, (n_in, n_out, n_hid, n_lay, act) in enumerate(
            zip(
                self.n_inputs,
                self.n_outputs,
                self.n_hidden,
                self.n_layers,
                self.act_fct,
            )
        ):

            my_encoder = self.encoders[idx]

            seq_len = torch.randint(1, 100, (1,)).item()
            batch_size = torch.randint(1, 100, (1,)).item()

            test_input_sequence_sample = torch.randn(batch_size, seq_len, n_in)

            my_out = my_encoder(test_input_sequence_sample)

            self.assertTrue(my_out.shape == (batch_size, n_out))

    def test_single_timestep_sequence_is_accepted_for_all_tcn_encoders(self):
        encoders = [
            nnets.TCNEncoder(
                input_channels=3,
                n_outputs=2,
                hidden_channels=4,
                kernel_size=3,
                dilation_base=2,
                levels=2,
                activation=torch.nn.ReLU,
            ),
            nnets.ResTCNEncoder(
                input_channels=3,
                n_outputs=2,
                hidden_channels=4,
                kernel_size=3,
                dilation_base=2,
                levels=2,
                activation=torch.nn.ReLU,
            ),
        ]
        in_sequence = torch.randn(1, 3)

        for encoder in encoders:
            out = encoder(in_sequence)
            self.assertEqual(out.shape, (2,))

    def test_restcn_dropout_changes_only_in_train_mode(self):
        torch.manual_seed(5)
        encoder = nnets.ResTCNEncoder(
            input_channels=3,
            n_outputs=2,
            hidden_channels=4,
            kernel_size=3,
            dilation_base=2,
            levels=2,
            activation=torch.nn.ReLU,
            dropout=0.4,
        )
        in_sequence = torch.randn(5, 6, 3)

        encoder.eval()
        eval_first_out = encoder(in_sequence)
        eval_second_out = encoder(in_sequence)
        self.assertTrue(torch.equal(eval_first_out, eval_second_out))

        encoder.train()
        train_first_out = encoder(in_sequence)
        train_second_out = encoder(in_sequence)
        self.assertFalse(torch.equal(train_first_out, train_second_out))


if __name__ == "__main__":
    unittest.main()
