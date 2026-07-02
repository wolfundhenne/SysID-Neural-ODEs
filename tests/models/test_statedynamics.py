import numpy as np
import torch
import unittest

# own ML models
import neuralsysid.models.statedynamics as sysmodels
from neuralsysid.models import neuralnetworks

# library ML models
from tests.reference_impls import neuroman

torch.manual_seed(0)
np.random.seed(0)

from neuralsysid.utils.helpers import initialize_weights_and_biases


class CaptureModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        return x


class ScaleModule(torch.nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        return self.factor * x


class ConstantSliceModule(torch.nn.Module):
    def __init__(self, out_dim: int, start_idx: int = 0):
        super().__init__()
        self.out_dim = out_dim
        self.start_idx = start_idx
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        return x[..., self.start_idx : self.start_idx + self.out_dim]


class ZeroModule(torch.nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        return x.new_zeros(*x.shape[:-1], self.out_dim)


class TestFxu(unittest.TestCase):
    # this class should test system models like discrete and continuous,
    # input affine and non input affine systems

    def setUp(self):
        n_tests = 50
        n_states = np.random.randint(1, 100, n_tests)
        n_inputs = np.random.randint(1, 100, n_tests)
        n_layers = np.random.randint(1, 10, n_tests)
        n_hidden = np.random.randint(1, 100, n_tests)

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

        self.fxu_models = []
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.act_fct = act_fct

        for n_s, n_i, n_l, n_h, act in zip(
            n_states,
            n_inputs,
            n_layers,
            n_hidden,
            act_fct,
        ):

            my_mlp = neuralnetworks.MultiLayerPerceptron(
                n_inputs=n_s + n_i,
                n_outputs=n_s,
                n_layers=n_l,
                n_hidden=n_h,
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
                Ts=1.0,
                nLayers=n_l,
                nNeurons=n_h,
                activation=act,
                integrator=None,
            )

            # define initials for weights to make all inits equal
            sizes_nn = [n_s + n_i] + [n_h] * n_l + [n_s]

            weight_inits_dynamics = [
                torch.randn(sizes_nn[ii + 1], sizes_nn[ii])
                for ii in range(len(sizes_nn) - 1)
            ]
            bias_inits_dynamics = [
                torch.randn(sizes_nn[ii + 1])
                for ii in range(len(sizes_nn) - 1)
            ]

            initialize_weights_and_biases(
                my_model.dynamics.mlp,
                weight_inits_dynamics,
                bias_inits_dynamics,
            )

            initialize_weights_and_biases(
                nm_model.nodes[0].callable.linear,
                weight_inits_dynamics,
                bias_inits_dynamics,
            )

            self.fxu_models.append((my_model, nm_model))

    def test_forward(self):

        for idx, (n_s, n_i, n_l, n_h, act) in enumerate(
            zip(
                self.n_states,
                self.n_inputs,
                self.n_layers,
                self.n_hidden,
                self.act_fct,
            )
        ):

            my_model = self.fxu_models[idx][0]
            nm_model = self.fxu_models[idx][1]

            n_traj = int(np.random.randint(1, 100))
            test_inputs = torch.randn(n_traj, n_i)
            test_states = torch.randn(n_traj, n_s)

            # check forward pass of the callable neuromancer
            my_out = my_model(test_states, test_inputs)
            nm_out = nm_model.nodes[0].callable(test_states, test_inputs)

            self.assertTrue(torch.equal(my_out, nm_out))

            # check the forward pass if whole neuromancer module is taken
            batch_nm = {
                "X": test_states[:, None, :],
                "xn": test_states[:, None, :],
                "U": test_inputs[:, None, :],
            }
            out = nm_model(batch_nm)
            prediction_nm = out["xn"][:, 1, :]
            self.assertTrue(torch.equal(my_out, prediction_nm))

    def test_constructor_stores_dimensions(self):
        dynamics = torch.nn.Identity()
        model = sysmodels.FxuDynamics(
            n_states=3,
            n_inputs=2,
            dynamics=dynamics,
        )

        self.assertEqual(model.n_states, 3)
        self.assertEqual(model.n_inputs, 2)
        self.assertIs(model.dynamics, dynamics)

    def test_forward_preserves_batch_and_state_dimensions(self):
        dynamics = neuralnetworks.MultiLayerPerceptron(
            n_inputs=5,
            n_outputs=3,
            n_layers=1,
            n_hidden=4,
            activation=torch.nn.ReLU,
        )
        model = sysmodels.FxuDynamics(
            n_states=3,
            n_inputs=2,
            dynamics=dynamics,
        )
        x = torch.randn(7, 3)
        u = torch.randn(7, 2)

        out = model(x, u)

        self.assertEqual(out.shape, (7, 3))

    def test_forward_concatenates_state_and_input_in_last_dimension(self):
        dynamics = CaptureModule()
        model = sysmodels.FxuDynamics(
            n_states=3,
            n_inputs=2,
            dynamics=dynamics,
        )
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        u = torch.tensor([[7.0, 8.0], [9.0, 10.0]])

        out = model(x, u)
        expected = torch.cat([x, u], dim=-1)

        self.assertTrue(torch.equal(dynamics.last_input, expected))
        self.assertTrue(torch.equal(out, expected))

    def test_forward_accepts_inputs_without_batch_dimension(self):
        dynamics = neuralnetworks.MultiLayerPerceptron(
            n_inputs=4,
            n_outputs=2,
            n_layers=1,
            n_hidden=3,
            activation=torch.nn.Tanh,
        )
        model = sysmodels.FxuDynamics(
            n_states=2,
            n_inputs=2,
            dynamics=dynamics,
        )
        x = torch.randn(2)
        u = torch.randn(2)

        out = model(x, u)

        self.assertEqual(out.shape, (2,))


class TestFxGu(unittest.TestCase):

    def setUp(self):
        n_tests = 50
        n_states = np.random.randint(1, 100, n_tests)
        n_inputs = np.random.randint(1, 100, n_tests)
        n_layers_states = np.random.randint(1, 10, n_tests)
        n_hidden_states = np.random.randint(1, 100, n_tests)
        n_layers_inputs = np.random.randint(1, 10, n_tests)
        n_hidden_inputs = np.random.randint(1, 100, n_tests)

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

        self.fxgu_models = []
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
                Ts=1.0,
                nlayersA=n_ls,
                nlayersB=n_li,
                nNeuronsA=n_hs,
                nNeuronsB=n_hi,
                activation=act,
                integrator=None,
            )

            # define initials for weights to make all inits equal
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

            self.fxgu_models.append((my_model, nm_model))

    def test_forward(self):

        for idx, (n_s, n_i, n_ls, n_hs, n_li, n_hi, act) in enumerate(
            zip(
                self.n_states,
                self.n_inputs,
                self.n_layers_states,
                self.n_hidden_states,
                self.n_layers_inputs,
                self.n_hidden_inputs,
                self.act_fct,
            )
        ):

            my_model = self.fxgu_models[idx][0]
            nm_model = self.fxgu_models[idx][1]

            n_traj = int(np.random.randint(1, 100))
            test_inputs = torch.randn(n_traj, n_i)
            test_states = torch.randn(n_traj, n_s)

            # check forward pass of the callable neuromancer
            my_out = my_model(test_states, test_inputs)
            nm_out = nm_model.nodes[0].callable(test_states, test_inputs)

            self.assertTrue(torch.equal(my_out, nm_out))

            # check the forward pass if whole neuromancer module is taken
            batch_nm = {
                "X": test_states[:, None, :],
                "xn": test_states[:, None, :],
                "U": test_inputs[:, None, :],
            }
            out = nm_model(batch_nm)
            prediction_nm = out["xn"][:, 1, :]
            self.assertTrue(torch.equal(my_out, prediction_nm))

    def test_constructor_stores_dimensions(self):
        f_dynamics = torch.nn.Identity()
        g_dynamics = torch.nn.Identity()
        model = sysmodels.FxGuDynamics(
            n_states=3,
            n_inputs=3,
            f_dynamics=f_dynamics,
            g_dynamics=g_dynamics,
        )

        self.assertEqual(model.n_states, 3)
        self.assertEqual(model.n_inputs, 3)
        self.assertIs(model.f_dynamics, f_dynamics)
        self.assertIs(model.g_dynamics, g_dynamics)

    def test_forward_preserves_batch_and_state_dimensions(self):
        f_dynamics = neuralnetworks.MultiLayerPerceptron(
            n_inputs=3,
            n_outputs=3,
            n_layers=1,
            n_hidden=4,
            activation=torch.nn.ReLU,
        )
        g_dynamics = neuralnetworks.MultiLayerPerceptron(
            n_inputs=2,
            n_outputs=3,
            n_layers=1,
            n_hidden=4,
            activation=torch.nn.ReLU,
        )
        model = sysmodels.FxGuDynamics(
            n_states=3,
            n_inputs=2,
            f_dynamics=f_dynamics,
            g_dynamics=g_dynamics,
        )
        x = torch.randn(7, 3)
        u = torch.randn(7, 2)

        out = model(x, u)

        self.assertEqual(out.shape, (7, 3))

    def test_forward_is_sum_of_state_and_input_dynamics(self):
        f_dynamics = ScaleModule(2.0)
        g_dynamics = ScaleModule(-3.0)
        model = sysmodels.FxGuDynamics(
            n_states=2,
            n_inputs=2,
            f_dynamics=f_dynamics,
            g_dynamics=g_dynamics,
        )
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        u = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        out = model(x, u)
        expected = 2.0 * x + (-3.0) * u

        self.assertTrue(torch.equal(f_dynamics.last_input, x))
        self.assertTrue(torch.equal(g_dynamics.last_input, u))
        self.assertTrue(torch.equal(out, expected))

    def test_forward_accepts_inputs_without_batch_dimension(self):
        f_dynamics = neuralnetworks.MultiLayerPerceptron(
            n_inputs=2,
            n_outputs=2,
            n_layers=1,
            n_hidden=3,
            activation=torch.nn.Tanh,
        )
        g_dynamics = neuralnetworks.MultiLayerPerceptron(
            n_inputs=2,
            n_outputs=2,
            n_layers=1,
            n_hidden=3,
            activation=torch.nn.Tanh,
        )
        model = sysmodels.FxGuDynamics(
            n_states=2,
            n_inputs=2,
            f_dynamics=f_dynamics,
            g_dynamics=g_dynamics,
        )
        x = torch.randn(2)
        u = torch.randn(2)

        out = model(x, u)

        self.assertEqual(out.shape, (2,))


class TestGraphDynamics(unittest.TestCase):
    def test_single_node_edgeless_graph_matches_fxu_dynamics(self):
        dynamics = neuralnetworks.MultiLayerPerceptron(
            n_inputs=5,
            n_outputs=3,
            n_layers=2,
            n_hidden=4,
            activation=torch.nn.Tanh,
        )
        fxu_model = sysmodels.FxuDynamics(
            n_states=3,
            n_inputs=2,
            dynamics=dynamics,
        )
        graph_model = sysmodels.GraphDynamics(
            n_states_node=3,
            n_inputs_node=2,
            n_nodes=1,
            n_msg=3,
            adjacency=[],
            node_function=dynamics,
            edge_function=ZeroModule(out_dim=3),
            additive_messages=True,
        )
        x = torch.randn(7, 3)
        u = torch.randn(7, 2)

        fxu_out = fxu_model(x, u)
        graph_out = graph_model(x, u)

        self.assertTrue(torch.allclose(graph_out, fxu_out))

    def test_constructor_creates_zero_width_optional_tensors(self):
        model = sysmodels.GraphDynamics(
            n_states_node=1,
            n_inputs_node=1,
            n_nodes=2,
            n_msg=1,
            adjacency=[(0, 1), (1, 0)],
            node_function=ConstantSliceModule(out_dim=1, start_idx=0),
            edge_function=CaptureModule(),
        )

        self.assertEqual(model.node_embedding.shape, (1, 2, 0))
        self.assertEqual(model.ctx.shape, (1, 2, 0))
        self.assertEqual(model.edge_embedding.shape, (2, 0))
        self.assertTrue(
            torch.equal(model.edge_to_line_idx, torch.tensor([0, 1]))
        )

    def test_set_context_replaces_stored_context(self):
        model = sysmodels.GraphDynamics(
            n_states_node=1,
            n_inputs_node=1,
            n_nodes=2,
            n_msg=1,
            adjacency=[(0, 1), (1, 0)],
            node_function=ConstantSliceModule(out_dim=1, start_idx=0),
            edge_function=CaptureModule(),
            n_ctx=2,
        )
        ctx = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

        model.set_context(ctx)

        self.assertIs(model.ctx, ctx)

    def test_edge_function_input_is_states_only_in_homogeneous_case(self):
        edge_function = ConstantSliceModule(out_dim=1, start_idx=0)
        model = sysmodels.GraphDynamics(
            n_states_node=1,
            n_inputs_node=1,
            n_nodes=2,
            n_msg=1,
            adjacency=[(0, 1), (1, 0)],
            node_function=ConstantSliceModule(out_dim=1, start_idx=0),
            edge_function=edge_function,
        )

        model(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))

        self.assertTrue(
            torch.equal(
                edge_function.last_input,
                torch.tensor([[[1.0, 2.0], [2.0, 1.0]]]),
            )
        )

    def test_node_function_input_in_additive_excludes_aggregated_messages(
        self,
    ):
        node_function = ConstantSliceModule(out_dim=1, start_idx=0)
        model = sysmodels.GraphDynamics(
            n_states_node=1,
            n_inputs_node=1,
            n_nodes=2,
            n_msg=1,
            adjacency=[(0, 1), (1, 0)],
            node_function=node_function,
            edge_function=ConstantSliceModule(out_dim=1, start_idx=0),
            additive_messages=True,
        )

        model(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))

        self.assertTrue(
            torch.equal(
                node_function.last_input,
                torch.tensor([[[1.0, 3.0], [2.0, 4.0]]]),
            )
        )

    def test_node_function_input_in_nonadditive_includes_aggregated_messages(
        self,
    ):
        node_function = ConstantSliceModule(out_dim=1, start_idx=0)
        model = sysmodels.GraphDynamics(
            n_states_node=1,
            n_inputs_node=1,
            n_nodes=2,
            n_msg=1,
            adjacency=[(0, 1), (1, 0)],
            node_function=node_function,
            edge_function=ConstantSliceModule(out_dim=1, start_idx=0),
            additive_messages=False,
        )

        model(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))

        self.assertTrue(
            torch.equal(
                node_function.last_input,
                torch.tensor([[[1.0, 3.0, 2.0], [2.0, 4.0, 1.0]]]),
            )
        )

    def test_forward_aggregates_messages_to_destination_nodes(self):
        model = sysmodels.GraphDynamics(
            n_states_node=1,
            n_inputs_node=1,
            n_nodes=3,
            n_msg=1,
            adjacency=[(0, 2), (1, 2)],
            node_function=ConstantSliceModule(out_dim=1, start_idx=2),
            edge_function=ConstantSliceModule(out_dim=1, start_idx=0),
            additive_messages=False,
        )

        out = model(
            torch.tensor([[2.0, 4.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
        )

        self.assertTrue(torch.equal(out, torch.tensor([[0.0, 0.0, 6.0]])))

    def test_forward_degree_normalization_divides_by_in_degree(self):
        model = sysmodels.GraphDynamics(
            n_states_node=1,
            n_inputs_node=1,
            n_nodes=3,
            n_msg=1,
            adjacency=[(0, 2), (1, 2)],
            node_function=ConstantSliceModule(out_dim=1, start_idx=2),
            edge_function=ConstantSliceModule(out_dim=1, start_idx=0),
            additive_messages=False,
            degree_normalization=True,
        )

        out = model(
            torch.tensor([[2.0, 4.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
        )

        self.assertTrue(torch.equal(out, torch.tensor([[0.0, 0.0, 3.0]])))

    def test_constructor_shared_edges_map_both_directions_to_same_embedding(
        self,
    ):
        model = sysmodels.GraphDynamics(
            n_states_node=1,
            n_inputs_node=1,
            n_nodes=3,
            n_msg=1,
            adjacency=[(0, 1), (1, 0), (1, 2), (2, 1)],
            node_function=ConstantSliceModule(out_dim=1, start_idx=0),
            edge_function=ConstantSliceModule(out_dim=1, start_idx=0),
            n_edge_embedding=2,
            share_edge_embeddings=True,
        )

        self.assertEqual(model.edge_embedding.shape, (2, 2))
        self.assertTrue(
            torch.equal(model.edge_to_line_idx, torch.tensor([0, 0, 1, 1]))
        )

    def test_forward_output_shape_matches_flattened_node_states(self):
        model = sysmodels.GraphDynamics(
            n_states_node=2,
            n_inputs_node=1,
            n_nodes=3,
            n_msg=2,
            adjacency=[(0, 1), (1, 2), (2, 0)],
            node_function=ZeroModule(out_dim=2),
            edge_function=ZeroModule(out_dim=2),
        )
        x = torch.randn(5, 6)
        u = torch.randn(5, 3)

        out = model(x, u)

        self.assertEqual(out.shape, (5, 6))

    def test_node_function_input_includes_context(self):
        node_function = ConstantSliceModule(out_dim=1, start_idx=0)
        model = sysmodels.GraphDynamics(
            n_states_node=1,
            n_inputs_node=1,
            n_nodes=2,
            n_msg=1,
            adjacency=[(0, 1), (1, 0)],
            node_function=node_function,
            edge_function=ConstantSliceModule(out_dim=1, start_idx=0),
            n_ctx=1,
            additive_messages=True,
        )
        model.set_context(torch.tensor([[[10.0], [20.0]]]))

        model(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))

        self.assertTrue(
            torch.equal(
                node_function.last_input,
                torch.tensor([[[1.0, 3.0, 10.0], [2.0, 4.0, 20.0]]]),
            )
        )

    def test_node_function_input_includes_node_embeddings(self):
        node_function = ConstantSliceModule(out_dim=1, start_idx=0)
        model = sysmodels.GraphDynamics(
            n_states_node=1,
            n_inputs_node=1,
            n_nodes=2,
            n_msg=1,
            adjacency=[(0, 1), (1, 0)],
            node_function=node_function,
            edge_function=ConstantSliceModule(out_dim=1, start_idx=0),
            n_node_embedding=1,
            additive_messages=True,
        )
        with torch.no_grad():
            model.node_embedding.copy_(torch.tensor([[[10.0], [20.0]]]))

        model(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))

        self.assertTrue(
            torch.equal(
                node_function.last_input,
                torch.tensor([[[1.0, 3.0, 10.0], [2.0, 4.0, 20.0]]]),
            )
        )

    def test_edge_function_input_uses_distinct_directed_edge_embeddings(
        self,
    ):
        edge_function = ConstantSliceModule(out_dim=1, start_idx=0)
        model = sysmodels.GraphDynamics(
            n_states_node=1,
            n_inputs_node=1,
            n_nodes=2,
            n_msg=1,
            adjacency=[(0, 1), (1, 0)],
            node_function=ConstantSliceModule(out_dim=1, start_idx=0),
            edge_function=edge_function,
            n_edge_embedding=1,
        )
        with torch.no_grad():
            model.edge_embedding.copy_(torch.tensor([[10.0], [20.0]]))

        model(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))

        self.assertTrue(
            torch.equal(
                edge_function.last_input,
                torch.tensor([[[1.0, 2.0, 10.0], [2.0, 1.0, 20.0]]]),
            )
        )

    def test_edge_function_input_uses_shared_edge_embeddings(
        self,
    ):
        edge_function = ConstantSliceModule(out_dim=1, start_idx=0)
        model = sysmodels.GraphDynamics(
            n_states_node=1,
            n_inputs_node=1,
            n_nodes=2,
            n_msg=1,
            adjacency=[(0, 1), (1, 0)],
            node_function=ConstantSliceModule(out_dim=1, start_idx=0),
            edge_function=edge_function,
            n_edge_embedding=1,
            share_edge_embeddings=True,
        )
        with torch.no_grad():
            model.edge_embedding.copy_(torch.tensor([[10.0]]))

        model(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))

        self.assertTrue(
            torch.equal(
                edge_function.last_input,
                torch.tensor([[[1.0, 2.0, 10.0], [2.0, 1.0, 10.0]]]),
            )
        )

    def test_node_function_input_includes_context_and_node_embeddings_together(
        self,
    ):
        node_function = ConstantSliceModule(out_dim=1, start_idx=0)
        model = sysmodels.GraphDynamics(
            n_states_node=1,
            n_inputs_node=1,
            n_nodes=2,
            n_msg=1,
            adjacency=[(0, 1), (1, 0)],
            node_function=node_function,
            edge_function=ConstantSliceModule(out_dim=1, start_idx=0),
            n_ctx=1,
            n_node_embedding=1,
            additive_messages=True,
        )
        with torch.no_grad():
            model.node_embedding.copy_(torch.tensor([[[10.0], [20.0]]]))
        model.set_context(torch.tensor([[[30.0], [40.0]]]))

        model(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))

        self.assertTrue(
            torch.equal(
                node_function.last_input,
                torch.tensor(
                    [[[1.0, 3.0, 10.0, 30.0], [2.0, 4.0, 20.0, 40.0]]]
                ),
            )
        )

    def test_unbatched_forward_supports_full_graph_heterogeneity(self):
        node_function = ConstantSliceModule(out_dim=1, start_idx=0)
        edge_function = ConstantSliceModule(out_dim=1, start_idx=0)
        model = sysmodels.GraphDynamics(
            n_states_node=1,
            n_inputs_node=1,
            n_nodes=2,
            n_msg=1,
            adjacency=[(0, 1), (1, 0)],
            node_function=node_function,
            edge_function=edge_function,
            n_ctx=1,
            n_node_embedding=1,
            n_edge_embedding=1,
            share_edge_embeddings=True,
            additive_messages=False,
        )
        with torch.no_grad():
            model.node_embedding.copy_(torch.tensor([[[10.0], [20.0]]]))
            model.edge_embedding.copy_(torch.tensor([[30.0]]))
        model.set_context(torch.tensor([[40.0], [50.0]]))

        out = model(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))

        self.assertEqual(out.shape, (2,))
        self.assertTrue(
            torch.equal(
                edge_function.last_input,
                torch.tensor([[[1.0, 2.0, 30.0], [2.0, 1.0, 30.0]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                node_function.last_input,
                torch.tensor(
                    [[[1.0, 3.0, 10.0, 40.0, 2.0], [2.0, 4.0, 20.0, 50.0, 1.0]]]
                ),
            )
        )
        self.assertTrue(torch.equal(out, torch.tensor([1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
