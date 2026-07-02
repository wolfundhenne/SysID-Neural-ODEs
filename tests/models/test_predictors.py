import numpy as np
import torch
import unittest

from copy import deepcopy

from neuralsysid.models import (
    stateencoders,
    predictors,
    statedynamics as dynamicsfunctions,
    timesteppers,
    outputdecoders as outputfunctions,
    neuralnetworks,
)

# library ML models
from tests.reference_impls import neuroman
from neuromancer.dynamics import integrators

from neuralsysid.utils.helpers import initialize_weights_and_biases


class DummyContextDynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = None

    def set_context(self, ctx):
        self.ctx = ctx

    def forward(self, x, u):
        return x


class DummyContextDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = None

    def set_context(self, ctx):
        self.ctx = ctx

    def forward(self, x, u):
        return x


class DummyContextEncoder(torch.nn.Module):
    def __init__(self, x_0: torch.Tensor, ctx: torch.Tensor):
        super().__init__()
        self.x_0 = x_0
        self.ctx = ctx

    def forward(self, y_hist, u_hist):
        return self.x_0, self.ctx


class ConstantDerivativeDynamics(dynamicsfunctions.DynamicsFunction):
    def __init__(self, derivative: torch.Tensor, n_inputs: int):
        super().__init__(derivative.shape[-1], n_inputs)
        self.derivative = derivative

    def forward(self, x, u):
        return self.derivative.expand_as(x)


class TestEncoderPredictorDecoder(unittest.TestCase):
    def setUp(self):
        n_tests = 100
        n_states = np.random.randint(1, 100, n_tests)
        n_inputs = np.random.randint(1, 100, n_tests)

        act_fcts = [
            torch.nn.ReLU,
            torch.nn.Sigmoid,
            torch.nn.Tanh,
        ]
        act_fct = [
            act_fcts[np.random.randint(0, len(act_fcts))]
            for _ in range(n_tests)
        ]
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.act_fct = act_fct

        self.n_layers = np.random.randint(1, 10, n_tests)
        self.n_hidden = np.random.randint(1, 100, n_tests)

        self.fxu_models = []

        for n_s, n_i, n_l, n_h, act in zip(
            n_states, n_inputs, self.n_layers, self.n_hidden, act_fct
        ):
            my_mlp = neuralnetworks.MultiLayerPerceptron(
                n_inputs=n_s + n_i,
                n_outputs=n_s,
                n_layers=n_l,
                n_hidden=n_h,
                activation=act,
            )
            my_model = dynamicsfunctions.FxuDynamics(
                n_states=n_s,
                n_inputs=n_i,
                dynamics=my_mlp,
            )

            discretestepper_fxu = timesteppers.Euler(my_model, delta_t=0.1)
            unrolled_euler = predictors.StateRollout(
                discretestepper_fxu, n_steps=1
            )

            encoder_predictor = predictors.EncoderPredictorDecoder(
                encoder=stateencoders.IdentityStateEncoder(),
                predictor=deepcopy(unrolled_euler),
                decoder=outputfunctions.PartialOutputDecoder(
                    n_states=n_s,
                    n_inputs=n_i,
                    n_outputs=n_s,
                    outputs_as_latent_states=list(range(n_s)),
                ),
            )

            self.fxu_models.append((unrolled_euler, encoder_predictor))

    def test_forward_seq_equivalence_osh(self):
        for idx, (n_s, n_i) in enumerate(
            zip(
                self.n_states,
                self.n_inputs,
            )
        ):
            unrolled_model = self.fxu_models[idx][0]
            encoder_predictor = self.fxu_models[idx][1]

            n_traj = int(np.random.randint(1, 100))
            n_hist = int(np.random.randint(1, 100))
            n_pred = int(np.random.randint(1, 100))
            unrolled_model.n_steps = n_pred
            encoder_predictor.set_n_steps(n_pred)
            test_inputs = torch.randn(n_traj, n_hist + n_pred, n_i)
            test_meas = torch.randn(n_traj, n_hist, n_s)

            unrolled_out = unrolled_model(
                test_meas[:, -1, :], test_inputs[:, n_hist:]
            )
            encoder_out = encoder_predictor(
                test_meas,
                test_inputs[:, :n_hist, :],
                test_inputs[:, n_hist:, :],
            )

            self.assertTrue(torch.equal(encoder_out, unrolled_out))

    def test_sets_dynamics_and_decoder_context_when_encoder_returns_tuple(
        self,
    ):
        dynamics = DummyContextDynamics()
        decoder = DummyContextDecoder()
        time_stepper = timesteppers.Discrete(dynamics)
        predictor = predictors.StateRollout(time_stepper, n_steps=2)
        x_0 = torch.tensor([[1.0, 2.0]])
        ctx = torch.tensor([[[3.0], [4.0]]])
        model = predictors.EncoderPredictorDecoder(
            predictor=predictor,
            encoder=DummyContextEncoder(x_0=x_0, ctx=ctx),
            decoder=decoder,
        )

        out = model(
            y_hist=torch.zeros(1, 3, 2),
            u_hist=torch.zeros(1, 3, 2),
            u=torch.zeros(1, 2, 2),
        )
        expected = torch.tensor([[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]])

        self.assertTrue(torch.equal(dynamics.ctx, ctx))
        self.assertTrue(torch.equal(decoder.ctx, ctx))
        self.assertEqual(out.shape, (1, 3, 2))
        self.assertTrue(torch.equal(out, expected))


class TestStateRollout(unittest.TestCase):
    def setUp(self):
        n_tests = 50
        n_states = np.random.randint(1, 100, n_tests)
        n_inputs = np.random.randint(1, 100, n_tests)

        act_fcts = [
            torch.nn.ReLU,
            torch.nn.Sigmoid,
            torch.nn.Tanh,
        ]
        act_fct = [
            act_fcts[np.random.randint(0, len(act_fcts))]
            for _ in range(n_tests)
        ]
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.act_fct = act_fct

        self.n_layers = np.random.randint(1, 10, n_tests)
        self.n_hidden = np.random.randint(1, 100, n_tests)

        self.fxu_models = []

        for n_s, n_i, n_l, n_h, act in zip(
            n_states, n_inputs, self.n_layers, self.n_hidden, act_fct
        ):
            my_mlp = neuralnetworks.MultiLayerPerceptron(
                n_inputs=n_s + n_i,
                n_outputs=n_s,
                n_layers=n_l,
                n_hidden=n_h,
                activation=act,
            )
            my_model = dynamicsfunctions.FxuDynamics(
                n_states=n_s,
                n_inputs=n_i,
                dynamics=my_mlp,
            )

            discrete_stepper_fxu = timesteppers.Euler(my_model, delta_t=0.1)

            nm_model = neuroman.construct_fxu(
                nx=n_s,
                nu=n_i,
                nsteps=1,
                Ts=0.1,
                nLayers=n_l,
                nNeurons=n_h,
                activation=act,
                integrator=integrators.Euler,
            )

            sizes_nn = [n_s + n_i] + [n_h] * n_l + [n_s]

            weight_inits = [
                torch.randn(sizes_nn[ii + 1], sizes_nn[ii])
                for ii in range(len(sizes_nn) - 1)
            ]
            bias_inits = [
                torch.randn(sizes_nn[ii + 1])
                for ii in range(len(sizes_nn) - 1)
            ]

            initialize_weights_and_biases(
                my_model.dynamics.mlp, weight_inits, bias_inits
            )
            initialize_weights_and_biases(
                nm_model.nodes[0].callable.block.linear,
                weight_inits,
                bias_inits,
            )

            state_rollout = predictors.StateRollout(
                deepcopy(discrete_stepper_fxu), n_steps=1
            )

            self.fxu_models.append(
                (discrete_stepper_fxu, state_rollout, nm_model)
            )

    def test_forward_seq_equivalence_osh(self):
        for idx, (n_s, n_i) in enumerate(
            zip(
                self.n_states,
                self.n_inputs,
            )
        ):
            my_model = self.fxu_models[idx][0]
            state_rollout = self.fxu_models[idx][1]

            n_traj = int(np.random.randint(1, 100))
            test_inputs = torch.randn(n_traj, n_i)
            test_states = torch.randn(n_traj, n_s)

            my_out = my_model(test_states, test_inputs)
            test_inputs = test_inputs[:, None, :]
            test_inputs = torch.cat((test_inputs, test_inputs), dim=1)
            rollout_out = state_rollout(test_states, test_inputs)

            self.assertTrue(torch.equal(my_out, rollout_out[:, 1, :]))

    def test_forward_seq_equivalence_msh(self):
        for idx, (n_s, n_i) in enumerate(
            zip(
                self.n_states,
                self.n_inputs,
            )
        ):
            state_rollout = self.fxu_models[idx][1]
            unrolled_model_nm = self.fxu_models[idx][2]

            n_traj = int(np.random.randint(1, 100))
            seq_len = int(np.random.randint(2, 25))
            test_inputs = torch.randn(n_traj, seq_len, n_i)
            test_states = torch.randn(n_traj, seq_len, n_s)

            state_rollout.n_steps = seq_len - 1
            unrolled_model_nm.nsteps = seq_len - 1

            rollout_out = state_rollout(test_states[:, 0, :], test_inputs)
            unrolled_out_nm = unrolled_model_nm(
                {
                    "X": test_states,
                    "xn": test_states[:, 0:1, :],
                    "U": test_inputs,
                }
            )
            self.assertTrue(
                torch.allclose(
                    rollout_out, unrolled_out_nm["xn"], equal_nan=True
                )
            )

    def test_state_rollout_shape_and_initial_state(self):
        derivative = torch.tensor([1.0, -2.0])
        dynamics = ConstantDerivativeDynamics(derivative, n_inputs=2)
        stepper = timesteppers.Residual(dynamics)
        rollout = predictors.StateRollout(stepper, n_steps=3)
        x_0 = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        u = torch.zeros(2, 3, 2)

        out = rollout(x_0, u)

        self.assertEqual(out.shape, (2, 4, 2))
        self.assertTrue(torch.equal(out[:, 0, :], x_0))

    def test_state_rollout_applies_recurrence_over_multiple_steps(self):
        derivative = torch.tensor([1.0, -1.0])
        dynamics = ConstantDerivativeDynamics(derivative, n_inputs=2)
        stepper = timesteppers.Residual(dynamics)
        rollout = predictors.StateRollout(stepper, n_steps=3)
        x_0 = torch.tensor([[0.0, 0.0]])
        u = torch.zeros(1, 3, 2)

        out = rollout(x_0, u)
        expected = torch.tensor(
            [[[0.0, 0.0], [1.0, -1.0], [2.0, -2.0], [3.0, -3.0]]]
        )

        self.assertTrue(torch.equal(out, expected))

    def test_state_rollout_accepts_single_sample_without_batch_dim(self):
        derivative = torch.tensor([1.0, -1.0])
        dynamics = ConstantDerivativeDynamics(derivative, n_inputs=2)
        stepper = timesteppers.Residual(dynamics)
        rollout = predictors.StateRollout(stepper, n_steps=3)
        x_0 = torch.tensor([0.0, 0.0])
        u = torch.zeros(3, 2)

        out = rollout(x_0, u)
        expected = torch.tensor(
            [[0.0, 0.0], [1.0, -1.0], [2.0, -2.0], [3.0, -3.0]]
        )

        self.assertEqual(out.shape, (4, 2))
        self.assertTrue(torch.equal(out, expected))

    def test_state_rollout_uses_configured_device(self):
        derivative = torch.tensor([1.0, 0.0])
        dynamics = ConstantDerivativeDynamics(derivative, n_inputs=2)
        stepper = timesteppers.Residual(dynamics)
        rollout = predictors.StateRollout(
            stepper, n_steps=1, device=torch.device("cpu")
        )
        x_0 = torch.tensor([[1.0, 2.0]])
        u = torch.zeros(1, 1, 2)

        out = rollout(x_0, u)

        self.assertEqual(out.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
