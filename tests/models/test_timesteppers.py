import numpy as np
import torch
import unittest

from copy import deepcopy

# own ML models
import neuralsysid.models.statedynamics as sysmodels
import neuralsysid.models.timesteppers as timesteppers
from neuralsysid.models import neuralnetworks

# library ML models
from tests.reference_impls import neuroman
from neuromancer.dynamics import integrators

torch.manual_seed(0)
np.random.seed(0)

from neuralsysid.utils.helpers import initialize_weights_and_biases


class AffineDynamics(sysmodels.DynamicsFunction):
    def __init__(self, n_states: int, n_inputs: int):
        super().__init__(n_states, n_inputs)

    def forward(self, x, u):
        return x + 2.0 * u


class ConstantDerivativeDynamics(sysmodels.DynamicsFunction):
    def __init__(self, derivative: torch.Tensor, n_inputs: int):
        super().__init__(derivative.shape[-1], n_inputs)
        self.derivative = derivative

    def forward(self, x, u):
        return self.derivative.expand_as(x)


class TestDiscrete(unittest.TestCase):
    def setUp(self):
        n_tests = 50
        n_states = np.random.randint(1, 100, n_tests)
        n_inputs = np.random.randint(1, 100, n_tests)
        n_layers_states = np.random.randint(1, 10, n_tests)
        n_hidden_states = np.random.randint(1, 100, n_tests)
        n_layers_inputs = np.random.randint(1, 10, n_tests)
        n_hidden_inputs = np.random.randint(1, 100, n_tests)

        act_fcts = [
            torch.nn.ReLU,
            torch.nn.Sigmoid,
            torch.nn.Tanh,
        ]
        act_fct = [
            act_fcts[np.random.randint(0, len(act_fcts))]
            for _ in range(n_tests)
        ]
        dynamictypes = ["fxu", "fxgu"]

        self.models = []
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_layers_states = n_layers_states
        self.n_hidden_states = n_hidden_states
        self.n_layers_inputs = n_layers_inputs
        self.n_hidden_inputs = n_hidden_inputs
        self.act_fct = act_fct

        for n_s, n_i, n_ls, n_hs, n_li, n_hi, act in zip(
            n_states,
            n_inputs,
            n_layers_states,
            n_hidden_states,
            n_layers_inputs,
            n_hidden_inputs,
            act_fct,
        ):
            dynamic_type = dynamictypes[
                np.random.randint(0, len(dynamictypes))
            ]
            if dynamic_type == "fxu":
                my_mlp = neuralnetworks.MultiLayerPerceptron(
                    n_inputs=n_s + n_i,
                    n_outputs=n_s,
                    n_layers=n_ls,
                    n_hidden=n_hs,
                    activation=act,
                )
                my_model = sysmodels.FxuDynamics(
                    n_states=n_s,
                    n_inputs=n_i,
                    dynamics=my_mlp,
                )
                nm_model = neuroman.construct_fxu(
                    nx=n_s,
                    nu=n_i,
                    nsteps=1,
                    Ts=1,
                    nLayers=n_ls,
                    nNeurons=n_hs,
                    activation=act,
                    integrator=None,
                )
            elif dynamic_type == "fxgu":
                my_mlp_f = neuralnetworks.MultiLayerPerceptron(
                    n_inputs=n_s,
                    n_outputs=n_s,
                    n_layers=n_ls,
                    n_hidden=n_hs,
                    activation=act,
                )
                my_mlp_g = neuralnetworks.MultiLayerPerceptron(
                    n_inputs=n_i,
                    n_outputs=n_s,
                    n_layers=n_li,
                    n_hidden=n_hi,
                    activation=act,
                )
                my_model = sysmodels.FxGuDynamics(
                    n_states=n_s,
                    n_inputs=n_i,
                    f_dynamics=my_mlp_f,
                    g_dynamics=my_mlp_g,
                )
                nm_model = neuroman.construct_fxgu(
                    nx=n_s,
                    nu=n_i,
                    nsteps=1,
                    Ts=1,
                    nlayersA=n_ls,
                    nlayersB=n_li,
                    nNeuronsA=n_hs,
                    nNeuronsB=n_hi,
                    activation=act,
                    integrator=None,
                )

            stepper = timesteppers.Discrete(my_model)

            if dynamic_type == "fxu":
                sizes_nn = [n_s + n_i] + [n_hs] * n_ls + [n_s]
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
                    nm_model.nodes[0].callable.linear, weight_inits, bias_inits
                )
            elif dynamic_type == "fxgu":
                sizes_states_nn = [n_s] + [n_hs] * n_ls + [n_s]
                sizes_inputs_nn = [n_i] + [n_hi] * n_li + [n_s]
                weight_inits_f_dynamics = [
                    torch.randn(sizes_states_nn[ii + 1], sizes_states_nn[ii])
                    for ii in range(len(sizes_states_nn) - 1)
                ]
                bias_inits_f_dynamics = [
                    torch.randn(sizes_states_nn[ii + 1])
                    for ii in range(len(sizes_states_nn) - 1)
                ]
                weight_inits_g_dynamics = [
                    torch.randn(sizes_inputs_nn[ii + 1], sizes_inputs_nn[ii])
                    for ii in range(len(sizes_inputs_nn) - 1)
                ]
                bias_inits_g_dynamics = [
                    torch.randn(sizes_inputs_nn[ii + 1])
                    for ii in range(len(sizes_inputs_nn) - 1)
                ]
                initialize_weights_and_biases(
                    my_model.f_dynamics.mlp,
                    weight_inits_f_dynamics,
                    bias_inits_f_dynamics,
                )
                initialize_weights_and_biases(
                    my_model.g_dynamics.mlp,
                    weight_inits_g_dynamics,
                    bias_inits_g_dynamics,
                )
                initialize_weights_and_biases(
                    nm_model.nodes[0].callable.fx.linear,
                    weight_inits_f_dynamics,
                    bias_inits_f_dynamics,
                )
                initialize_weights_and_biases(
                    nm_model.nodes[0].callable.fu.linear,
                    weight_inits_g_dynamics,
                    bias_inits_g_dynamics,
                )

            self.models.append((stepper, nm_model))

        self.n_layers = np.random.randint(1, 10, n_tests)
        self.n_hidden = np.random.randint(1, 100, n_tests)

    def test_forward(self):

        for idx, (n_s, n_i) in enumerate(
            zip(
                self.n_states,
                self.n_inputs,
            )
        ):
            my_model = self.models[idx][0]
            nm_model = self.models[idx][1]

            n_traj = int(np.random.randint(1, 100))
            test_inputs = torch.randn(n_traj, n_i)
            test_states = torch.randn(n_traj, n_s)

            my_out = my_model(test_states, test_inputs)
            nm_out = nm_model.nodes[0].callable(test_states, test_inputs)

            self.assertTrue(torch.equal(my_out, nm_out))

            batch_nm = {
                "X": test_states[:, None, :],
                "xn": test_states[:, None, :],
                "U": test_inputs[:, None, :],
            }
            out = nm_model(batch_nm)
            prediction_nm = out["xn"][:, 1, :]
            self.assertTrue(torch.equal(my_out, prediction_nm))

    def test_discrete_forwards_dynamics_output(self):
        dynamics = AffineDynamics(n_states=2, n_inputs=2)
        stepper = timesteppers.Discrete(dynamics, delta_t=3.0)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        u = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        out = stepper(x, u)

        self.assertEqual(stepper.delta_t, 3.0)
        self.assertTrue(torch.equal(out, x + 2.0 * u))

    def test_discrete_accepts_inputs_without_batch_dimension(self):
        dynamics = AffineDynamics(n_states=2, n_inputs=2)
        stepper = timesteppers.Discrete(dynamics)
        x = torch.tensor([1.0, 2.0])
        u = torch.tensor([3.0, 4.0])

        out = stepper(x, u)

        self.assertEqual(out.shape, (2,))
        self.assertTrue(torch.equal(out, x + 2.0 * u))


class TestResidual(unittest.TestCase):
    def setUp(self):
        n_tests = 50
        n_states = np.random.randint(1, 100, n_tests)
        n_inputs = np.random.randint(1, 100, n_tests)
        n_layers_states = np.random.randint(1, 10, n_tests)
        n_hidden_states = np.random.randint(1, 100, n_tests)
        n_layers_inputs = np.random.randint(1, 10, n_tests)
        n_hidden_inputs = np.random.randint(1, 100, n_tests)

        act_fcts = [
            torch.nn.ReLU,
            torch.nn.Sigmoid,
            torch.nn.Tanh,
        ]
        act_fct = [
            act_fcts[np.random.randint(0, len(act_fcts))]
            for _ in range(n_tests)
        ]
        dynamictypes = ["fxu", "fxgu"]

        self.models = []
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_layers_states = n_layers_states
        self.n_hidden_states = n_hidden_states
        self.n_layers_inputs = n_layers_inputs
        self.n_hidden_inputs = n_hidden_inputs
        self.act_fct = act_fct

        for n_s, n_i, n_ls, n_hs, n_li, n_hi, act in zip(
            n_states,
            n_inputs,
            n_layers_states,
            n_hidden_states,
            n_layers_inputs,
            n_hidden_inputs,
            act_fct,
        ):
            dynamic_type = dynamictypes[
                np.random.randint(0, len(dynamictypes))
            ]
            if dynamic_type == "fxu":
                my_mlp = neuralnetworks.MultiLayerPerceptron(
                    n_inputs=n_s + n_i,
                    n_outputs=n_s,
                    n_layers=n_ls,
                    n_hidden=n_hs,
                    activation=act,
                )
                my_model = sysmodels.FxuDynamics(
                    n_states=n_s,
                    n_inputs=n_i,
                    dynamics=my_mlp,
                )
                nm_model = neuroman.construct_fxu(
                    nx=n_s,
                    nu=n_i,
                    nsteps=1,
                    Ts=1,
                    nLayers=n_ls,
                    nNeurons=n_hs,
                    activation=act,
                    integrator=integrators.Euler,
                )
            elif dynamic_type == "fxgu":
                my_mlp_f = neuralnetworks.MultiLayerPerceptron(
                    n_inputs=n_s,
                    n_outputs=n_s,
                    n_layers=n_ls,
                    n_hidden=n_hs,
                    activation=act,
                )
                my_mlp_g = neuralnetworks.MultiLayerPerceptron(
                    n_inputs=n_i,
                    n_outputs=n_s,
                    n_layers=n_li,
                    n_hidden=n_hi,
                    activation=act,
                )
                my_model = sysmodels.FxGuDynamics(
                    n_states=n_s,
                    n_inputs=n_i,
                    f_dynamics=my_mlp_f,
                    g_dynamics=my_mlp_g,
                )
                nm_model = neuroman.construct_fxgu(
                    nx=n_s,
                    nu=n_i,
                    nsteps=1,
                    Ts=1,
                    nlayersA=n_ls,
                    nlayersB=n_li,
                    nNeuronsA=n_hs,
                    nNeuronsB=n_hi,
                    activation=act,
                    integrator=integrators.Euler,
                )

            stepper = timesteppers.Residual(my_model)

            if dynamic_type == "fxu":
                sizes_nn = [n_s + n_i] + [n_hs] * n_ls + [n_s]
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
            elif dynamic_type == "fxgu":
                sizes_states_nn = [n_s] + [n_hs] * n_ls + [n_s]
                sizes_inputs_nn = [n_i] + [n_hi] * n_li + [n_s]
                weight_inits_f_dynamics = [
                    torch.randn(sizes_states_nn[ii + 1], sizes_states_nn[ii])
                    for ii in range(len(sizes_states_nn) - 1)
                ]
                bias_inits_f_dynamics = [
                    torch.randn(sizes_states_nn[ii + 1])
                    for ii in range(len(sizes_states_nn) - 1)
                ]
                weight_inits_g_dynamics = [
                    torch.randn(sizes_inputs_nn[ii + 1], sizes_inputs_nn[ii])
                    for ii in range(len(sizes_inputs_nn) - 1)
                ]
                bias_inits_g_dynamics = [
                    torch.randn(sizes_inputs_nn[ii + 1])
                    for ii in range(len(sizes_inputs_nn) - 1)
                ]
                initialize_weights_and_biases(
                    my_model.f_dynamics.mlp,
                    weight_inits_f_dynamics,
                    bias_inits_f_dynamics,
                )
                initialize_weights_and_biases(
                    my_model.g_dynamics.mlp,
                    weight_inits_g_dynamics,
                    bias_inits_g_dynamics,
                )
                initialize_weights_and_biases(
                    nm_model.nodes[0].callable.block.fx.linear,
                    weight_inits_f_dynamics,
                    bias_inits_f_dynamics,
                )
                initialize_weights_and_biases(
                    nm_model.nodes[0].callable.block.fu.linear,
                    weight_inits_g_dynamics,
                    bias_inits_g_dynamics,
                )

            self.models.append((stepper, nm_model))

        self.n_layers = np.random.randint(1, 10, n_tests)
        self.n_hidden = np.random.randint(1, 100, n_tests)

    def test_forward(self):

        for idx, (n_s, n_i) in enumerate(
            zip(
                self.n_states,
                self.n_inputs,
            )
        ):
            my_model = self.models[idx][0]
            nm_model = self.models[idx][1]

            n_traj = int(np.random.randint(1, 100))
            test_inputs = torch.randn(n_traj, n_i)
            test_states = torch.randn(n_traj, n_s)

            my_out = my_model(test_states, test_inputs)
            nm_out = nm_model.nodes[0].callable(test_states, test_inputs)

            self.assertTrue(torch.equal(my_out, nm_out))

            batch_nm = {
                "X": test_states[:, None, :],
                "xn": test_states[:, None, :],
                "U": test_inputs[:, None, :],
            }
            out = nm_model(batch_nm)
            prediction_nm = out["xn"][:, 1, :]
            self.assertTrue(torch.equal(my_out, prediction_nm))

    def test_residual_adds_dynamics_to_state(self):
        derivative = torch.tensor([0.5, -1.0])
        dynamics = ConstantDerivativeDynamics(derivative, n_inputs=2)
        stepper = timesteppers.Residual(dynamics, delta_t=2.0)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        u = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        out = stepper(x, u)

        self.assertEqual(stepper.delta_t, 2.0)
        self.assertTrue(torch.equal(out, x + derivative))

    def test_residual_accepts_inputs_without_batch_dimension(self):
        derivative = torch.tensor([1.0, -2.0])
        dynamics = ConstantDerivativeDynamics(derivative, n_inputs=2)
        stepper = timesteppers.Residual(dynamics)
        x = torch.tensor([3.0, 4.0])
        u = torch.tensor([5.0, 6.0])

        out = stepper(x, u)

        self.assertEqual(out.shape, (2,))
        self.assertTrue(torch.equal(out, x + derivative))


class TestEuler(unittest.TestCase):
    def setUp(self):
        n_tests = 50
        n_states = np.random.randint(1, 100, n_tests)
        n_inputs = np.random.randint(1, 100, n_tests)
        n_layers_states = np.random.randint(1, 10, n_tests)
        n_hidden_states = np.random.randint(1, 100, n_tests)
        n_layers_inputs = np.random.randint(1, 10, n_tests)
        n_hidden_inputs = np.random.randint(1, 100, n_tests)
        n_ts = np.random.choice([0.1, 0.5, 1, 2, 5], n_tests)

        act_fcts = [
            torch.nn.ReLU,
            torch.nn.Sigmoid,
            torch.nn.Tanh,
        ]
        act_fct = [
            act_fcts[np.random.randint(0, len(act_fcts))]
            for _ in range(n_tests)
        ]
        dynamictypes = ["fxu", "fxgu"]

        self.models = []
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_layers_states = n_layers_states
        self.n_hidden_states = n_hidden_states
        self.n_layers_inputs = n_layers_inputs
        self.n_hidden_inputs = n_hidden_inputs
        self.act_fct = act_fct

        for n_s, n_i, n_ls, n_hs, n_li, n_hi, act, ts in zip(
            n_states,
            n_inputs,
            n_layers_states,
            n_hidden_states,
            n_layers_inputs,
            n_hidden_inputs,
            act_fct,
            n_ts,
        ):
            dynamic_type = dynamictypes[
                np.random.randint(0, len(dynamictypes))
            ]
            if dynamic_type == "fxu":
                my_mlp = neuralnetworks.MultiLayerPerceptron(
                    n_inputs=n_s + n_i,
                    n_outputs=n_s,
                    n_layers=n_ls,
                    n_hidden=n_hs,
                    activation=act,
                )
                my_model = sysmodels.FxuDynamics(
                    n_states=n_s,
                    n_inputs=n_i,
                    dynamics=my_mlp,
                )
                nm_model = neuroman.construct_fxu(
                    nx=n_s,
                    nu=n_i,
                    nsteps=1,
                    Ts=ts,
                    nLayers=n_ls,
                    nNeurons=n_hs,
                    activation=act,
                    integrator=integrators.Euler,
                )
            elif dynamic_type == "fxgu":
                my_mlp_f = neuralnetworks.MultiLayerPerceptron(
                    n_inputs=n_s,
                    n_outputs=n_s,
                    n_layers=n_ls,
                    n_hidden=n_hs,
                    activation=act,
                )
                my_mlp_g = neuralnetworks.MultiLayerPerceptron(
                    n_inputs=n_i,
                    n_outputs=n_s,
                    n_layers=n_li,
                    n_hidden=n_hi,
                    activation=act,
                )
                my_model = sysmodels.FxGuDynamics(
                    n_states=n_s,
                    n_inputs=n_i,
                    f_dynamics=my_mlp_f,
                    g_dynamics=my_mlp_g,
                )
                nm_model = neuroman.construct_fxgu(
                    nx=n_s,
                    nu=n_i,
                    nsteps=1,
                    Ts=ts,
                    nlayersA=n_ls,
                    nlayersB=n_li,
                    nNeuronsA=n_hs,
                    nNeuronsB=n_hi,
                    activation=act,
                    integrator=integrators.Euler,
                )

            stepper = timesteppers.Euler(my_model, delta_t=ts)

            if dynamic_type == "fxu":
                sizes_nn = [n_s + n_i] + [n_hs] * n_ls + [n_s]
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
            elif dynamic_type == "fxgu":
                sizes_states_nn = [n_s] + [n_hs] * n_ls + [n_s]
                sizes_inputs_nn = [n_i] + [n_hi] * n_li + [n_s]
                weight_inits_f_dynamics = [
                    torch.randn(sizes_states_nn[ii + 1], sizes_states_nn[ii])
                    for ii in range(len(sizes_states_nn) - 1)
                ]
                bias_inits_f_dynamics = [
                    torch.randn(sizes_states_nn[ii + 1])
                    for ii in range(len(sizes_states_nn) - 1)
                ]
                weight_inits_g_dynamics = [
                    torch.randn(sizes_inputs_nn[ii + 1], sizes_inputs_nn[ii])
                    for ii in range(len(sizes_inputs_nn) - 1)
                ]
                bias_inits_g_dynamics = [
                    torch.randn(sizes_inputs_nn[ii + 1])
                    for ii in range(len(sizes_inputs_nn) - 1)
                ]
                initialize_weights_and_biases(
                    my_model.f_dynamics.mlp,
                    weight_inits_f_dynamics,
                    bias_inits_f_dynamics,
                )
                initialize_weights_and_biases(
                    my_model.g_dynamics.mlp,
                    weight_inits_g_dynamics,
                    bias_inits_g_dynamics,
                )
                initialize_weights_and_biases(
                    nm_model.nodes[0].callable.block.fx.linear,
                    weight_inits_f_dynamics,
                    bias_inits_f_dynamics,
                )
                initialize_weights_and_biases(
                    nm_model.nodes[0].callable.block.fu.linear,
                    weight_inits_g_dynamics,
                    bias_inits_g_dynamics,
                )

            self.models.append((stepper, nm_model))

        self.n_layers = np.random.randint(1, 10, n_tests)
        self.n_hidden = np.random.randint(1, 100, n_tests)

    def test_forward(self):

        for idx, (n_s, n_i) in enumerate(
            zip(
                self.n_states,
                self.n_inputs,
            )
        ):
            my_model = self.models[idx][0]
            nm_model = self.models[idx][1]

            n_traj = int(np.random.randint(1, 100))
            test_inputs = torch.randn(n_traj, n_i)
            test_states = torch.randn(n_traj, n_s)

            my_out = my_model(test_states, test_inputs)
            nm_out = nm_model.nodes[0].callable(test_states, test_inputs)

            self.assertTrue(torch.equal(my_out, nm_out))

            batch_nm = {
                "X": test_states[:, None, :],
                "xn": test_states[:, None, :],
                "U": test_inputs[:, None, :],
            }
            out = nm_model(batch_nm)
            prediction_nm = out["xn"][:, 1, :]
            self.assertTrue(torch.equal(my_out, prediction_nm))

    def test_euler_integrates_constant_derivative_exactly(self):
        derivative = torch.tensor([0.5, -1.5])
        dynamics = ConstantDerivativeDynamics(derivative, n_inputs=2)
        stepper = timesteppers.Euler(dynamics, delta_t=0.2, adjoint=False)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        u = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

        out = stepper(x, u)

        self.assertTrue(torch.allclose(out, x + 0.2 * derivative))

    def test_euler_adjoint_runs_on_constant_derivative(self):
        derivative = torch.tensor([1.0, -1.0])
        dynamics = ConstantDerivativeDynamics(derivative, n_inputs=2)
        stepper = timesteppers.Euler(dynamics, delta_t=0.1, adjoint=True)
        x = torch.tensor([[1.0, 2.0]])
        u = torch.tensor([[0.0, 0.0]])

        out = stepper(x, u)

        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(torch.allclose(out, x + 0.1 * derivative))


class TestRungeKutta4(unittest.TestCase):
    def setUp(self):
        n_tests = 50
        n_states = np.random.randint(1, 100, n_tests)
        n_inputs = np.random.randint(1, 100, n_tests)
        n_layers_states = np.random.randint(1, 10, n_tests)
        n_hidden_states = np.random.randint(1, 100, n_tests)
        n_layers_inputs = np.random.randint(1, 10, n_tests)
        n_hidden_inputs = np.random.randint(1, 100, n_tests)
        n_ts = np.random.choice([0.1, 0.5, 1, 2, 5], n_tests)

        act_fcts = [
            torch.nn.ReLU,
            torch.nn.Sigmoid,
            torch.nn.Tanh,
        ]
        act_fct = [
            act_fcts[np.random.randint(0, len(act_fcts))]
            for _ in range(n_tests)
        ]
        dynamictypes = ["fxu", "fxgu"]

        self.models = []
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_layers_states = n_layers_states
        self.n_hidden_states = n_hidden_states
        self.n_layers_inputs = n_layers_inputs
        self.n_hidden_inputs = n_hidden_inputs
        self.act_fct = act_fct

        for n_s, n_i, n_ls, n_hs, n_li, n_hi, act, ts in zip(
            n_states,
            n_inputs,
            n_layers_states,
            n_hidden_states,
            n_layers_inputs,
            n_hidden_inputs,
            act_fct,
            n_ts,
        ):
            dynamic_type = dynamictypes[
                np.random.randint(0, len(dynamictypes))
            ]
            if dynamic_type == "fxu":
                my_mlp = neuralnetworks.MultiLayerPerceptron(
                    n_inputs=n_s + n_i,
                    n_outputs=n_s,
                    n_layers=n_ls,
                    n_hidden=n_hs,
                    activation=act,
                )
                my_model = sysmodels.FxuDynamics(
                    n_states=n_s,
                    n_inputs=n_i,
                    dynamics=my_mlp,
                )
                nm_model = neuroman.construct_fxu(
                    nx=n_s,
                    nu=n_i,
                    nsteps=1,
                    Ts=ts,
                    nLayers=n_ls,
                    nNeurons=n_hs,
                    activation=act,
                    integrator=neuroman.RK4_three_eightth,
                )
            elif dynamic_type == "fxgu":
                my_mlp_f = neuralnetworks.MultiLayerPerceptron(
                    n_inputs=n_s,
                    n_outputs=n_s,
                    n_layers=n_ls,
                    n_hidden=n_hs,
                    activation=act,
                )
                my_mlp_g = neuralnetworks.MultiLayerPerceptron(
                    n_inputs=n_i,
                    n_outputs=n_s,
                    n_layers=n_li,
                    n_hidden=n_hi,
                    activation=act,
                )
                my_model = sysmodels.FxGuDynamics(
                    n_states=n_s,
                    n_inputs=n_i,
                    f_dynamics=my_mlp_f,
                    g_dynamics=my_mlp_g,
                )
                nm_model = neuroman.construct_fxgu(
                    nx=n_s,
                    nu=n_i,
                    nsteps=1,
                    Ts=ts,
                    nlayersA=n_ls,
                    nlayersB=n_li,
                    nNeuronsA=n_hs,
                    nNeuronsB=n_hi,
                    activation=act,
                    integrator=neuroman.RK4_three_eightth,
                )

            stepper = timesteppers.RungeKutta4(my_model, delta_t=ts)

            if dynamic_type == "fxu":
                sizes_nn = [n_s + n_i] + [n_hs] * n_ls + [n_s]
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
            elif dynamic_type == "fxgu":
                sizes_states_nn = [n_s] + [n_hs] * n_ls + [n_s]
                sizes_inputs_nn = [n_i] + [n_hi] * n_li + [n_s]
                weight_inits_f_dynamics = [
                    torch.randn(sizes_states_nn[ii + 1], sizes_states_nn[ii])
                    for ii in range(len(sizes_states_nn) - 1)
                ]
                bias_inits_f_dynamics = [
                    torch.randn(sizes_states_nn[ii + 1])
                    for ii in range(len(sizes_states_nn) - 1)
                ]
                weight_inits_g_dynamics = [
                    torch.randn(sizes_inputs_nn[ii + 1], sizes_inputs_nn[ii])
                    for ii in range(len(sizes_inputs_nn) - 1)
                ]
                bias_inits_g_dynamics = [
                    torch.randn(sizes_inputs_nn[ii + 1])
                    for ii in range(len(sizes_inputs_nn) - 1)
                ]
                initialize_weights_and_biases(
                    my_model.f_dynamics.mlp,
                    weight_inits_f_dynamics,
                    bias_inits_f_dynamics,
                )
                initialize_weights_and_biases(
                    my_model.g_dynamics.mlp,
                    weight_inits_g_dynamics,
                    bias_inits_g_dynamics,
                )
                initialize_weights_and_biases(
                    nm_model.nodes[0].callable.block.fx.linear,
                    weight_inits_f_dynamics,
                    bias_inits_f_dynamics,
                )
                initialize_weights_and_biases(
                    nm_model.nodes[0].callable.block.fu.linear,
                    weight_inits_g_dynamics,
                    bias_inits_g_dynamics,
                )

            self.models.append((stepper, nm_model))

        self.n_layers = np.random.randint(1, 10, n_tests)
        self.n_hidden = np.random.randint(1, 100, n_tests)

    def test_forward(self):

        for idx, (n_s, n_i) in enumerate(
            zip(
                self.n_states,
                self.n_inputs,
            )
        ):
            my_model = self.models[idx][0]
            nm_model = self.models[idx][1]

            n_traj = int(np.random.randint(1, 100))
            test_inputs = torch.randn(n_traj, n_i)
            test_states = torch.randn(n_traj, n_s)

            my_out = my_model(test_states, test_inputs)
            nm_out = nm_model.nodes[0].callable(test_states, test_inputs)

            self.assertTrue(torch.equal(my_out, nm_out))

            batch_nm = {
                "X": test_states[:, None, :],
                "xn": test_states[:, None, :],
                "U": test_inputs[:, None, :],
            }
            out = nm_model(batch_nm)
            prediction_nm = out["xn"][:, 1, :]
            self.assertTrue(torch.equal(my_out, prediction_nm))

    def test_rungekutta4_integrates_constant_derivative_exactly(self):
        derivative = torch.tensor([0.25, -0.75])
        dynamics = ConstantDerivativeDynamics(derivative, n_inputs=2)
        stepper = timesteppers.RungeKutta4(
            dynamics, delta_t=0.4, adjoint=False
        )
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        u = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

        out = stepper(x, u)

        self.assertTrue(torch.allclose(out, x + 0.4 * derivative))

    def test_rungekutta4_adjoint_runs_on_constant_derivative(self):
        derivative = torch.tensor([1.0, 2.0])
        dynamics = ConstantDerivativeDynamics(derivative, n_inputs=2)
        stepper = timesteppers.RungeKutta4(
            dynamics, delta_t=0.3, adjoint=True
        )
        x = torch.tensor([[1.0, 2.0]])
        u = torch.tensor([[0.0, 0.0]])

        out = stepper(x, u)

        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(torch.allclose(out, x + 0.3 * derivative))


class TestDopri5(unittest.TestCase):
    def test_dopri5_integrates_constant_derivative(self):
        derivative = torch.tensor([0.5, -1.0])
        dynamics = ConstantDerivativeDynamics(derivative, n_inputs=2)
        stepper = timesteppers.Dopri5(dynamics, delta_t=0.2, adjoint=False)
        x = torch.tensor([[1.0, 2.0]])
        u = torch.tensor([[0.0, 0.0]])

        out = stepper(x, u)

        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(torch.allclose(out, x + 0.2 * derivative, atol=1e-5))

    def test_dopri5_adjoint_runs_on_constant_derivative(self):
        derivative = torch.tensor([1.0, -1.0])
        dynamics = ConstantDerivativeDynamics(derivative, n_inputs=2)
        stepper = timesteppers.Dopri5(dynamics, delta_t=0.1, adjoint=True)
        x = torch.tensor([[1.0, 2.0]])
        u = torch.tensor([[0.0, 0.0]])

        out = stepper(x, u)

        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(torch.allclose(out, x + 0.1 * derivative, atol=1e-5))


class TestDopri8(unittest.TestCase):
    def test_dopri8_integrates_constant_derivative(self):
        derivative = torch.tensor([0.5, -1.0])
        dynamics = ConstantDerivativeDynamics(derivative, n_inputs=2)
        stepper = timesteppers.Dopri8(dynamics, delta_t=0.2, adjoint=False)
        x = torch.tensor([[1.0, 2.0]])
        u = torch.tensor([[0.0, 0.0]])

        out = stepper(x, u)

        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(torch.allclose(out, x + 0.2 * derivative, atol=1e-5))

    def test_dopri8_adjoint_runs_on_constant_derivative(self):
        derivative = torch.tensor([1.0, -1.0])
        dynamics = ConstantDerivativeDynamics(derivative, n_inputs=2)
        stepper = timesteppers.Dopri8(dynamics, delta_t=0.1, adjoint=True)
        x = torch.tensor([[1.0, 2.0]])
        u = torch.tensor([[0.0, 0.0]])

        out = stepper(x, u)

        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(torch.allclose(out, x + 0.1 * derivative, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
