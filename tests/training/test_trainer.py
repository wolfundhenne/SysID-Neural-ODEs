import numpy as np
import random
import torch
import unittest
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

# own ML models
import neuralsysid.training.trainer as mytrainer
from neuralsysid.data import preparation
from neuralsysid.models import (
    neuralnetworks,
    outputdecoders as outputfunctions,
    predictors,
    stateencoders,
    statedynamics as dynamicsfunctions,
    timesteppers,
)

# neuromancer modules
import neuromancer
import neuromancer.trainer as nmtrainer
from neuromancer.dynamics import integrators as nmintegrators

# own modules making use of neuromancer
from tests.reference_impls import neuroman, onesteppred, sequencepred

from neuralsysid.utils.helpers import initialize_weights_and_biases

# set seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def loss_per_batch(input, target, model):
    output = model(input)
    loss = torch.nn.MSELoss()(output, target)

    return loss


def loss_per_batch_ownmse(input, target, model):
    output = model(input)
    loss = torch.mean((output - target) ** 2)

    return loss


class SimpleRegressionTrainer(mytrainer.BaseTrainer):
    def calc_objective(self, in_features, out_features):
        out = self.model(in_features)
        return self.loss_fct(out, out_features)


class EpochLossTrainer(mytrainer.BaseTrainer):
    def __init__(self, *args, train_losses, dev_losses, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_losses = train_losses
        self._dev_losses = dev_losses

    def calc_objective(self, in_features, out_features):
        raise NotImplementedError

    def train_loop(self):
        with torch.no_grad():
            self.model.weight.fill_(float(self.current_epoch))
        self.train_loss.append(self._train_losses[self.current_epoch])

    def eval_loop(self):
        self.dev_loss.append(self._dev_losses[self.current_epoch])


class FakeTrial:
    def __init__(self, should_prune=False):
        self.reported = []
        self._should_prune = should_prune

    def report(self, loss, epoch):
        self.reported.append((loss, epoch))

    def should_prune(self):
        return self._should_prune


class FakeEncodedModel(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.recorded_y_h = None
        self.recorded_u_h = None
        self.recorded_u = None

    def forward(self, y_h, u_h, u):
        self.recorded_y_h = y_h
        self.recorded_u_h = u_h
        self.recorded_u = u
        return self.output


class TestBaseTrainer(unittest.TestCase):
    def setUp(self):
        features = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        targets = 2.0 * features
        dataset = TensorDataset(features, targets)
        self.loader = DataLoader(dataset, batch_size=2, shuffle=False)
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(0.0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.loss_fct = torch.nn.MSELoss()

    def _make_trainer(self, **kwargs):
        model = deepcopy(self.model)
        return SimpleRegressionTrainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_fct=torch.nn.MSELoss(),
            train_loader=self.loader,
            dev_loader=self.loader,
            epochs=3,
            patience=1,
            warmup=0,
            device=torch.device("cpu"),
            **kwargs,
        )

    def test_prune_nan(self):
        trainer = self._make_trainer()

        self.assertTrue(trainer.prune_nan(float("nan")))
        self.assertFalse(trainer.prune_nan(1.0))

    def test_prune_unpromising_without_trial_returns_false(self):
        trainer = self._make_trainer()
        trainer.current_epoch = 10

        self.assertFalse(trainer.prune_unpromising(1.0))

    def test_prune_unpromising_reports_and_respects_trial(self):
        trial = FakeTrial(should_prune=True)
        trainer = self._make_trainer(trial=trial)
        trainer.current_epoch = 10

        should_prune = trainer.prune_unpromising(2.5, report_interval=10)

        self.assertTrue(should_prune)
        self.assertEqual(trial.reported, [(2.5, 10)])

    def test_train_loop_appends_mean_loss(self):
        trainer = self._make_trainer()

        trainer.train_loop()

        self.assertEqual(len(trainer.train_loss), 1)
        self.assertGreaterEqual(trainer.train_loss[0], 0.0)

    def test_eval_loop_appends_mean_loss(self):
        trainer = self._make_trainer()

        trainer.eval_loop()

        self.assertEqual(len(trainer.dev_loss), 1)
        self.assertGreaterEqual(trainer.dev_loss[0], 0.0)

    def test_evaluate_returns_mean_loss(self):
        trainer = self._make_trainer()

        loss = trainer.evaluate(self.loader)

        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)

    def test_train_stops_early_after_patience(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = EpochLossTrainer(
            model=model,
            optimizer=optimizer,
            loss_fct=torch.nn.MSELoss(),
            train_loader=self.loader,
            dev_loader=self.loader,
            epochs=10,
            patience=1,
            warmup=0,
            device=torch.device("cpu"),
            train_losses=[5.0, 4.0, 4.0, 4.0],
            dev_losses=[3.0, 3.0, 3.0, 3.0],
        )

        finished = trainer.train(print_progress=False)

        self.assertTrue(finished)
        self.assertEqual(trainer.current_epoch, 2)
        self.assertEqual(trainer.no_improvement_counter, 2)

    def test_train_tracks_best_model_state_dict(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = EpochLossTrainer(
            model=model,
            optimizer=optimizer,
            loss_fct=torch.nn.MSELoss(),
            train_loader=self.loader,
            dev_loader=self.loader,
            epochs=3,
            patience=5,
            warmup=0,
            device=torch.device("cpu"),
            train_losses=[5.0, 4.0, 3.0],
            dev_losses=[5.0, 2.0, 4.0],
        )

        finished = trainer.train(print_progress=False)

        self.assertTrue(finished)
        self.assertEqual(float(trainer.best_dev_loss), 2.0)
        self.assertTrue(
            torch.equal(
                trainer.best_model_state_dict["weight"],
                torch.tensor([[1.0]]),
            )
        )


class TestSequencePredictionTrainerDeterministic(unittest.TestCase):
    def test_calc_objective_uses_full_output_tensor(self):
        y_hist = torch.tensor(
            [
                [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]],
                [[3.0, 30.0, 300.0], [4.0, 40.0, 400.0]],
            ]
        )
        u_hist = torch.tensor(
            [
                [[0.1], [0.2]],
                [[0.3], [0.4]],
            ]
        )
        u_pred = torch.tensor(
            [
                [[1.0], [2.0], [3.0]],
                [[4.0], [5.0], [6.0]],
            ]
        )
        out_features = torch.tensor(
            [
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                [[20.0, 21.0, 22.0], [23.0, 24.0, 25.0], [26.0, 27.0, 28.0]],
            ]
        )
        model_output = torch.tensor(
            [
                [[30.0, 31.0, 32.0], [33.0, 34.0, 35.0], [36.0, 37.0, 38.0]],
                [[40.0, 41.0, 42.0], [43.0, 44.0, 45.0], [46.0, 47.0, 48.0]],
            ]
        )
        model = FakeEncodedModel(model_output)
        optimizer = torch.optim.SGD(
            [torch.nn.Parameter(torch.tensor(0.0))], lr=0.1
        )
        trainer = mytrainer.SequencePredictionTrainer(
            model=model,
            optimizer=optimizer,
            loss_fct=torch.nn.MSELoss(),
            train_loader=[],
            dev_loader=[],
            device=torch.device("cpu"),
        )

        loss = trainer.calc_objective(
            ((y_hist, u_hist), u_pred),
            out_features,
        )

        expected_y_h = y_hist
        expected_loss = torch.nn.MSELoss()(
            model_output[:, 1:, :],
            out_features[:, 1:, :],
        )

        self.assertTrue(torch.equal(model.recorded_y_h, expected_y_h))
        self.assertTrue(torch.equal(model.recorded_u_h, u_hist))
        self.assertTrue(torch.equal(model.recorded_u, u_pred))
        self.assertTrue(torch.isclose(loss, expected_loss))


class TestTrainerOneStepPred(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.training_epochs = 10
        n_tests = 5
        n_states = np.random.randint(1, 100, n_tests)
        n_inputs = np.random.randint(1, 100, n_tests)
        n_layers_states = np.random.randint(1, 10, n_tests)
        n_hidden_states = np.random.randint(1, 100, n_tests)
        n_layers_inputs = np.random.randint(1, 10, n_tests)
        n_hidden_inputs = np.random.randint(1, 100, n_tests)
        self.n_batches = np.random.randint(1, 100, n_tests)

        # for trainer test, we additionally need
        self.T_train_list = []
        self.dataset_train_list = []
        self.dataloader_train_list = []

        self.T_dev_list = []
        self.dataset_dev_list = []
        self.dataloader_dev_list = []

        self.T_test_list = []
        self.dataset_test_list = []
        self.dataloader_test_list = []

        traj_length = np.random.randint(5, 500, n_tests)
        batch_size = np.random.randint(1, 600, n_tests)  # this is for

        activations = [
            torch.nn.ReLU,
        ]
        integrators = [
            (timesteppers.Euler, nmintegrators.Euler),
        ]
        dynamicstypes = ["fxu"]
        act_fcts = random.choices(activations, k=n_tests)
        dyn_types = random.choices(dynamicstypes, k=n_tests)
        int_types = random.choices(integrators, k=n_tests)

        self.n_tests = n_tests
        self.models = []
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_layers_states = n_layers_states
        self.n_hidden_states = n_hidden_states
        self.act_fcts = act_fcts

        for (
            n_s,
            n_i,
            n_ls,
            n_hs,
            n_li,
            n_hi,
            act,
            traj_len,
            bs,
            dyn_typ,
            int_typ,
        ) in zip(
            n_states,
            n_inputs,
            n_layers_states,
            n_hidden_states,
            n_layers_inputs,
            n_hidden_inputs,
            act_fcts,
            traj_length,
            batch_size,
            dyn_types,
            int_types,
        ):
            if dyn_typ == "fxu":
                my_mlp = neuralnetworks.MultiLayerPerceptron(
                    n_inputs=n_s + n_i,
                    n_outputs=n_s,
                    n_layers=n_ls,
                    n_hidden=n_hs,
                    activation=act,
                )
                my_dynamics = dynamicsfunctions.FxuDynamics(
                    n_states=n_s,
                    n_inputs=n_i,
                    dynamics=my_mlp,
                )
                stepper_model = int_typ[0](
                    my_dynamics,
                    delta_t=(
                        0.1
                        if int_typ[0] is not timesteppers.Discrete
                        and int_typ[0] is not timesteppers.Residual
                        else 1.0
                    ),
                )

                nm_model = neuroman.construct_fxu(
                    nx=n_s,
                    nu=n_i,
                    nsteps=1,
                    Ts=(
                        0.1
                        if int_typ[0] is not timesteppers.Discrete
                        and int_typ[0] is not timesteppers.Residual
                        else 1.0
                    ),
                    nLayers=n_ls,
                    nNeurons=n_hs,
                    activation=act,
                    integrator=int_typ[1],
                )

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
                    my_dynamics.dynamics.mlp, weight_inits, bias_inits
                )
                if int_typ[0] == timesteppers.Discrete:
                    initialize_weights_and_biases(
                        nm_model.nodes[0].callable.linear,
                        weight_inits,
                        bias_inits,
                    )
                else:
                    initialize_weights_and_biases(
                        nm_model.nodes[0].callable.block.linear,
                        weight_inits,
                        bias_inits,
                    )
            elif dyn_typ == "fxgu":
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
                my_dynamics = dynamicsfunctions.FxGuDynamics(
                    n_states=n_s,
                    n_inputs=n_i,
                    f_dynamics=my_mlp_f,
                    g_dynamics=my_mlp_g,
                )
                stepper_model = int_typ[0](
                    my_dynamics,
                    (
                        0.1
                        if int_typ[0] is not timesteppers.Discrete
                        and int_typ[0] is not timesteppers.Residual
                        else 1.0
                    ),
                )
                nm_model = neuroman.construct_fxgu(
                    nx=n_s,
                    nu=n_i,
                    nsteps=1,
                    Ts=(
                        0.1
                        if int_typ[0] is not timesteppers.Discrete
                        and int_typ[0] is not timesteppers.Residual
                        else 1.0
                    ),
                    nlayersA=n_ls,
                    nlayersB=n_li,
                    nNeuronsA=n_hs,
                    nNeuronsB=n_hi,
                    activation=act,
                    integrator=int_typ[1],
                )

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
                    my_dynamics.f_dynamics.mlp,
                    weight_inits_f_dynamics,
                    bias_inits_f_dynamics,
                )
                initialize_weights_and_biases(
                    my_dynamics.g_dynamics.mlp,
                    weight_inits_g_dynamics,
                    bias_inits_g_dynamics,
                )
                if int_typ[0] == timesteppers.Discrete:
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
                else:
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
            else:
                raise ValueError("Unknown dynamics type")

            unrolled_model = predictors.StateRollout(
                deepcopy(stepper_model), n_steps=1
            )

            unrolled_model_enc = predictors.EncoderPredictorDecoder(
                deepcopy(unrolled_model),
                stateencoders.IdentityStateEncoder(),
                outputfunctions.PartialOutputDecoder(
                    n_states=n_s,
                    n_inputs=n_i,
                    n_outputs=n_s,
                    outputs_as_latent_states=list(range(n_s)),
                ),
            )

            self.models.append(
                (
                    stepper_model,
                    nm_model,
                    unrolled_model,
                    unrolled_model_enc,
                )
            )

            for datatype in ["train", "dev", "test"]:

                state_trajectories = np.random.randn(1, traj_len, n_s)
                input_trajectories = np.random.randn(1, traj_len, n_i)
                time = np.arange(traj_len)[np.newaxis, :]

                dataset_dict = {
                    "X": state_trajectories,
                    "Y": state_trajectories,
                    "U": input_trajectories,
                }

                # one step ahead prediciton datasets.
                T_osh, dataset_osh, dataloader_osh = (
                    onesteppred.create_batches(
                        t=time,
                        data_set=dataset_dict,
                        used_states=list(range(n_s)),
                        used_inputs=list(range(n_i)),
                        batch_size=int(bs),
                        shuffle=False,
                    )
                )
                T_nm, dataset_nm, dataloader_nm = (
                    neuroman.create_batches_overlapping_k(
                        t=time,
                        data_set=dataset_dict,
                        prediction_horizon=1,
                        used_states=list(range(n_s)),
                        used_inputs=list(range(n_i)),
                        batch_size=int(bs),
                        shuffle=False,
                        data_set_type=datatype,
                        overlap=1,
                    )
                )
                T_seq, dataset_seq, dataloader_seq = (
                    sequencepred.create_batches_sequence(
                        t=time,
                        data_set=dataset_dict,
                        used_states=list(range(n_s)),
                        used_inputs=list(range(n_i)),
                        batch_size=int(bs),
                        shuffle=False,
                        seq_len=2,
                        overlap=1,
                    )
                )
                T_enc, dataset_enc, dataloader_enc = (
                    preparation.build_sequence_dataloader(
                        t=time,
                        trajectory_data=dataset_dict,
                        used_outputs=list(range(n_s)),
                        used_controls=list(range(n_i)),
                        batch_size=int(bs),
                        shuffle=False,
                        seq_len=2,
                        overlap=1,
                        historic_seq_len=1,
                        encode=False,
                    )
                )

                if datatype == "train":
                    self.T_train_list.append((T_osh, T_nm, T_seq, T_enc))
                    self.dataset_train_list.append(
                        (dataset_osh, dataset_nm, dataset_seq, dataset_enc)
                    )
                    self.dataloader_train_list.append(
                        (
                            dataloader_osh,
                            dataloader_nm,
                            dataloader_seq,
                            dataloader_enc,
                        )
                    )

                elif datatype == "dev":
                    self.T_dev_list.append((T_osh, T_nm, T_seq, T_enc))
                    self.dataset_dev_list.append(
                        (dataset_osh, dataset_nm, dataset_seq, dataset_enc)
                    )
                    self.dataloader_dev_list.append(
                        (
                            dataloader_osh,
                            dataloader_nm,
                            dataloader_seq,
                            dataloader_enc,
                        )
                    )
                elif datatype == "test":
                    self.T_test_list.append((T_osh, T_nm, T_seq, T_enc))
                    self.dataset_test_list.append(
                        (dataset_osh, dataset_nm, dataset_seq, dataset_enc)
                    )
                    self.dataloader_test_list.append(
                        (
                            dataloader_osh,
                            dataloader_nm,
                            dataloader_seq,
                            dataloader_enc,
                        )
                    )

    def test_loss(self):
        # TODO: add test for encoded version (with identity encoder)

        for idx, (n_s, n_i) in enumerate(
            zip(
                self.n_states,
                self.n_inputs,
            )
        ):
            unrolled_model = self.models[idx][2]
            unrolled_model_nm = self.models[idx][1]
            stepper_model = self.models[idx][0]
            unrolled_model_enc = self.models[idx][3]

            n_traj = int(np.random.randint(1, 100))
            seq_len = 2
            test_inputs = torch.randn(n_traj, seq_len, n_i)
            test_states = torch.randn(n_traj, seq_len, n_s)

            unrolled_model.n_steps = seq_len - 1
            unrolled_model_nm.nsteps = seq_len - 1
            unrolled_model_enc.set_n_steps(seq_len - 1)

            batch_nm = {
                "X": test_states,
                "xn": test_states[:, 0:1, :],
                "U": test_inputs,
                "name": "train",
                "epoch": 0,
            }
            x = neuromancer.constraint.variable("X")[:, 1:, :]
            xhat = neuromancer.constraint.variable("xn")[:, 1:, :]
            objectives = [neuroman.define_loss(x, xhat, -1, 1)]
            constraints = []
            loss = neuromancer.loss.PenaltyLoss(objectives, constraints)

            problem = neuromancer.problem.Problem([unrolled_model_nm], loss)

            unrolled_out_nm = problem(batch_nm)
            unrolled_loss_nm = unrolled_out_nm["train_loss"]

            unrolled_out = unrolled_model(test_states[:, 0, :], test_inputs)
            unrolled_loss = torch.nn.MSELoss()(
                unrolled_out[:, 1:, :], test_states[:, 1:, :]
            )
            stepper_out = stepper_model(
                test_states[:, 0, :], test_inputs[:, 0, :]
            )
            stepper_loss = torch.nn.MSELoss()(
                stepper_out, test_states[:, 1, :]
            )
            unrolled_out_enc = unrolled_model_enc(
                test_states[:, 0:1, :], test_inputs[:, 0:1, :], test_inputs
            )
            unrolled_loss_enc = torch.nn.MSELoss()(
                unrolled_out_enc[:, 1:, :], test_states[:, 1:, :]
            )

            self.assertTrue(
                torch.isclose(
                    unrolled_loss,
                    stepper_loss,
                    rtol=0,
                    atol=1e-20,
                    equal_nan=True,
                )
            )
            self.assertTrue(
                torch.isclose(
                    unrolled_loss,
                    unrolled_loss_nm,
                    rtol=0,
                    atol=1e-20,
                    equal_nan=True,
                )
            )
            self.assertTrue(
                torch.isclose(
                    unrolled_loss,
                    unrolled_loss_enc,
                    rtol=0,
                    atol=1e-20,
                    equal_nan=True,
                )
            )

    def test_training(self):

        n_epochs = self.training_epochs
        clip_grads_val = 100.0
        warmup = 10

        for idx, _ in enumerate(self.n_states):

            unrolled_model = self.models[idx][2]
            model_osh = self.models[idx][0]
            unrolled_model_nm = self.models[idx][1]
            unrolled_model_enc = self.models[idx][3]

            unrolled_model.n_steps = 1
            unrolled_model_nm.nsteps = 1
            unrolled_model_enc.set_n_steps(1)

            seq_dataloader_train = self.dataloader_train_list[idx][2]
            osh_dataloader_train = self.dataloader_train_list[idx][0]
            nm_dataloader_train = self.dataloader_train_list[idx][1]
            enc_dataloader_train = self.dataloader_train_list[idx][3]
            seq_dataloader_dev = self.dataloader_dev_list[idx][2]
            osh_dataloader_dev = self.dataloader_dev_list[idx][0]
            nm_dataloader_dev = self.dataloader_dev_list[idx][1]
            enc_dataloader_dev = self.dataloader_dev_list[idx][3]
            seq_dataloader_test = self.dataloader_test_list[idx][2]
            osh_dataloader_test = self.dataloader_test_list[idx][0]
            nm_dataloader_test = self.dataloader_test_list[idx][1]
            enc_dataloader_test = self.dataloader_test_list[idx][3]

            x = neuromancer.constraint.variable("X")[:, 1:, :]
            xhat = neuromancer.constraint.variable("xn")[:, 1:, :]
            objectives = [neuroman.define_loss(x, xhat, -1, 1)]
            constraints = []
            loss = neuromancer.loss.PenaltyLoss(objectives, constraints)
            problem = neuromancer.problem.Problem([unrolled_model_nm], loss)

            optimizer_unrolled_nm = torch.optim.Adam(
                problem.parameters(), lr=0.01
            )
            optimizer_unrolled = torch.optim.Adam(
                unrolled_model.parameters(), lr=0.01
            )
            optimizer_unrolled_osh = torch.optim.Adam(
                model_osh.parameters(), lr=0.01
            )
            optimizer_unrolled_enc = torch.optim.Adam(
                unrolled_model_enc.parameters(), lr=0.01
            )

            osh_trainer = onesteppred.OneStepTrainer(
                model=model_osh,
                optimizer=optimizer_unrolled_osh,
                loss_fct=torch.nn.MSELoss(),
                train_loader=osh_dataloader_train,
                dev_loader=osh_dataloader_dev,
                epochs=n_epochs,
                patience=n_epochs,
                clip_grad_threshhold=clip_grads_val,
                warmup=warmup,
                device=torch.device("cpu"),
            )

            seq_trainer = sequencepred.SequenceTrainer(
                model=unrolled_model,
                optimizer=optimizer_unrolled,
                loss_fct=torch.nn.MSELoss(),
                train_loader=seq_dataloader_train,
                dev_loader=seq_dataloader_dev,
                epochs=n_epochs,
                patience=n_epochs,
                clip_grad_threshhold=clip_grads_val,
                warmup=warmup,
                device=torch.device("cpu"),
            )

            nm_trainer = nmtrainer.Trainer(
                problem=problem,
                train_data=nm_dataloader_train,
                dev_data=nm_dataloader_dev,
                test_data=nm_dataloader_test,
                optimizer=optimizer_unrolled_nm,
                epochs=n_epochs,
                patience=n_epochs,
                clip=clip_grads_val,
                warmup=warmup,
                device=torch.device("cpu"),
                eval_metric="mean_dev_loss",
            )

            enc_trainer = mytrainer.SequencePredictionTrainer(
                model=unrolled_model_enc,
                optimizer=optimizer_unrolled_enc,
                loss_fct=torch.nn.MSELoss(),
                train_loader=enc_dataloader_train,
                dev_loader=enc_dataloader_dev,
                epochs=n_epochs,
                patience=n_epochs,
                clip_grad_threshhold=clip_grads_val,
                warmup=warmup,
                device=torch.device("cpu"),
            )

            seq_trainer.train()
            osh_trainer.train()
            nm_trainer.train()
            enc_trainer.train()

            unrolled_model.eval()
            model_osh.eval()
            problem.eval()
            unrolled_model_enc.eval()

            unrolled_model.load_state_dict(seq_trainer.best_model_state_dict)
            model_osh.load_state_dict(osh_trainer.best_model_state_dict)
            problem.load_state_dict(nm_trainer.best_model)
            unrolled_model_enc.load_state_dict(
                enc_trainer.best_model_state_dict
            )

            my_out = []
            for (x_0, u), _ in seq_dataloader_test:
                out = unrolled_model(x_0, u)[:, 1, :]
                my_out.append(out)

            osh_out = []
            for feature, _ in osh_dataloader_test:
                out = model_osh(
                    feature[..., : model_osh.dynamics.n_states],
                    feature[..., model_osh.dynamics.n_states :],
                )
                osh_out.append(out)

            nm_out = []
            for batch in nm_dataloader_test:
                out = problem(batch)
                out = out["test_xn"][:, 1:, :]
                out = torch.squeeze(out, dim=1)
                nm_out.append(out)

            enc_out = []
            for ((y_hist, u_hist), u_pred), _ in enc_dataloader_test:
                out = unrolled_model_enc(y_hist, u_hist, u_pred)[:, 1, :]
                enc_out.append(out)

            for my_o, osh_o in zip(my_out, osh_out):
                self.assertTrue(
                    torch.allclose(
                        my_o, osh_o, rtol=1e-4, atol=1e-4, equal_nan=True
                    )
                )
            for my_o, nm_o in zip(my_out, nm_out):
                print("max error: ", torch.max(my_o - nm_o))
                self.assertTrue(
                    torch.allclose(
                        my_o, nm_o, rtol=1e-4, atol=1e-4, equal_nan=True
                    )
                )
            for my_o, enc_o in zip(my_out, enc_out):
                self.assertTrue(
                    torch.allclose(
                        my_o, enc_o, rtol=1e-4, atol=1e-4, equal_nan=True
                    )
                )


class TestTrainerMultiStepPred(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.training_epochs = 10
        n_tests = 5
        n_states = np.random.randint(1, 100, n_tests)
        n_inputs = np.random.randint(1, 100, n_tests)
        n_layers_states = np.random.randint(1, 10, n_tests)
        n_hidden_states = np.random.randint(1, 100, n_tests)
        n_layers_inputs = np.random.randint(1, 10, n_tests)
        n_hidden_inputs = np.random.randint(1, 100, n_tests)
        self.n_batches = np.random.randint(1, 100, n_tests)

        # for trainer test, we additionally need
        self.T_train_list = []
        self.dataset_train_list = []
        self.dataloader_train_list = []

        self.T_dev_list = []
        self.dataset_dev_list = []
        self.dataloader_dev_list = []

        self.T_test_list = []
        self.dataset_test_list = []
        self.dataloader_test_list = []

        self.seq_lens = []

        traj_length = np.random.randint(5, 500, n_tests)
        batch_size = np.random.randint(1, 600, n_tests)  # this is for

        activations = [
            torch.nn.ReLU,
        ]
        integrators = [
            (timesteppers.Euler, nmintegrators.Euler),
        ]
        dynamicstypes = ["fxu"]
        act_fcts = random.choices(activations, k=n_tests)
        dyn_types = random.choices(dynamicstypes, k=n_tests)
        int_types = random.choices(integrators, k=n_tests)

        self.n_tests = n_tests
        self.models = []
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_layers_states = n_layers_states
        self.n_hidden_states = n_hidden_states
        self.act_fcts = act_fcts

        for (
            n_s,
            n_i,
            n_ls,
            n_hs,
            n_li,
            n_hi,
            act,
            traj_len,
            bs,
            dyn_typ,
            int_typ,
        ) in zip(
            n_states,
            n_inputs,
            n_layers_states,
            n_hidden_states,
            n_layers_inputs,
            n_hidden_inputs,
            act_fcts,
            traj_length,
            batch_size,
            dyn_types,
            int_types,
        ):
            seq_len = int(np.random.randint(2, traj_len))
            overlap = int(np.random.randint(1, seq_len))
            self.seq_lens.append(seq_len)

            if dyn_typ == "fxu":
                my_mlp = neuralnetworks.MultiLayerPerceptron(
                    n_inputs=n_s + n_i,
                    n_outputs=n_s,
                    n_layers=n_ls,
                    n_hidden=n_hs,
                    activation=act,
                )
                my_dynamics = dynamicsfunctions.FxuDynamics(
                    n_states=n_s,
                    n_inputs=n_i,
                    dynamics=my_mlp,
                )
                stepper_model = int_typ[0](
                    my_dynamics,
                    delta_t=(
                        0.1
                        if int_typ[0] is not timesteppers.Discrete
                        and int_typ[0] is not timesteppers.Residual
                        else 1.0
                    ),
                )

                nm_model = neuroman.construct_fxu(
                    nx=n_s,
                    nu=n_i,
                    nsteps=seq_len - 1,
                    Ts=(
                        0.1
                        if int_typ[0] is not timesteppers.Discrete
                        and int_typ[0] is not timesteppers.Residual
                        else 1.0
                    ),
                    nLayers=n_ls,
                    nNeurons=n_hs,
                    activation=act,
                    integrator=int_typ[1],
                )

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
                    my_dynamics.dynamics.mlp, weight_inits, bias_inits
                )
                if int_typ[0] == timesteppers.Discrete:
                    initialize_weights_and_biases(
                        nm_model.nodes[0].callable.linear,
                        weight_inits,
                        bias_inits,
                    )
                else:
                    initialize_weights_and_biases(
                        nm_model.nodes[0].callable.block.linear,
                        weight_inits,
                        bias_inits,
                    )
            elif dyn_typ == "fxgu":
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
                my_dynamics = dynamicsfunctions.FxGuDynamics(
                    n_states=n_s,
                    n_inputs=n_i,
                    f_dynamics=my_mlp_f,
                    g_dynamics=my_mlp_g,
                )
                stepper_model = int_typ[0](
                    my_dynamics,
                    (
                        0.1
                        if int_typ[0] is not timesteppers.Discrete
                        and int_typ[0] is not timesteppers.Residual
                        else 1.0
                    ),
                )

                nm_model = neuroman.construct_fxgu(
                    nx=n_s,
                    nu=n_i,
                    nsteps=seq_len - 1,
                    Ts=(
                        0.1
                        if int_typ[0] is not timesteppers.Discrete
                        and int_typ[0] is not timesteppers.Residual
                        else 1.0
                    ),
                    nlayersA=n_ls,
                    nlayersB=n_li,
                    nNeuronsA=n_hs,
                    nNeuronsB=n_hi,
                    activation=act,
                    integrator=int_typ[1],
                )

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
                    my_dynamics.f_dynamics.mlp,
                    weight_inits_f_dynamics,
                    bias_inits_f_dynamics,
                )
                initialize_weights_and_biases(
                    my_dynamics.g_dynamics.mlp,
                    weight_inits_g_dynamics,
                    bias_inits_g_dynamics,
                )
                if int_typ[0] == timesteppers.Discrete:
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
                else:
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
            else:
                raise ValueError("Unknown dynamics type")

            unrolled_model = predictors.StateRollout(
                deepcopy(stepper_model), n_steps=seq_len - 1
            )

            unrolled_model_enc = predictors.EncoderPredictorDecoder(
                deepcopy(unrolled_model),
                stateencoders.IdentityStateEncoder(),
                outputfunctions.PartialOutputDecoder(
                    n_states=n_s,
                    n_inputs=n_i,
                    n_outputs=n_s,
                    outputs_as_latent_states=list(range(n_s)),
                ),
            )

            self.models.append((nm_model, unrolled_model, unrolled_model_enc))

            for datatype in ["train", "dev", "test"]:

                state_trajectories = np.random.randn(1, traj_len, n_s)
                input_trajectories = np.random.randn(1, traj_len, n_i)
                time = np.arange(traj_len)[np.newaxis, :]

                dataset_dict = {
                    "X": state_trajectories,
                    "Y": state_trajectories,
                    "U": input_trajectories,
                }

                # multi step ahead prediction datasets.
                T_nm, dataset_nm, dataloader_nm = (
                    neuroman.create_batches_overlapping_k(
                        t=time,
                        data_set=dataset_dict,
                        prediction_horizon=seq_len - 1,
                        used_states=list(range(n_s)),
                        used_inputs=list(range(n_i)),
                        batch_size=int(bs),
                        shuffle=False,
                        data_set_type=datatype,
                        overlap=overlap,
                    )
                )
                T_seq, dataset_seq, dataloader_seq = (
                    sequencepred.create_batches_sequence(
                        t=time,
                        data_set=dataset_dict,
                        used_states=list(range(n_s)),
                        used_inputs=list(range(n_i)),
                        batch_size=int(bs),
                        shuffle=False,
                        seq_len=seq_len,
                        overlap=overlap,
                    )
                )
                T_enc, dataset_enc, dataloader_enc = (
                    preparation.build_sequence_dataloader(
                        t=time,
                        trajectory_data=dataset_dict,
                        used_outputs=list(range(n_s)),
                        used_controls=list(range(n_i)),
                        batch_size=int(bs),
                        shuffle=False,
                        seq_len=seq_len,
                        overlap=overlap,
                        historic_seq_len=1,
                        encode=False,
                    )
                )

                if datatype == "train":
                    self.T_train_list.append((T_nm, T_seq, T_enc))
                    self.dataset_train_list.append(
                        (dataset_nm, dataset_seq, dataset_enc)
                    )
                    self.dataloader_train_list.append(
                        (dataloader_nm, dataloader_seq, dataloader_enc)
                    )
                elif datatype == "dev":
                    self.T_dev_list.append((T_nm, T_seq, T_enc))
                    self.dataset_dev_list.append(
                        (dataset_nm, dataset_seq, dataset_enc)
                    )
                    self.dataloader_dev_list.append(
                        (dataloader_nm, dataloader_seq, dataloader_enc)
                    )
                elif datatype == "test":
                    self.T_test_list.append((T_nm, T_seq, T_enc))
                    self.dataset_test_list.append(
                        (dataset_nm, dataset_seq, dataset_enc)
                    )
                    self.dataloader_test_list.append(
                        (dataloader_nm, dataloader_seq, dataloader_enc)
                    )

    def test_loss(self):
        # TODO: add test for encoded version (with identity encoder)

        for idx, (n_s, n_i) in enumerate(
            zip(
                self.n_states,
                self.n_inputs,
            )
        ):
            unrolled_model = self.models[idx][1]
            unrolled_model_nm = self.models[idx][0]
            unrolled_model_enc = self.models[idx][2]

            n_traj = int(np.random.randint(1, 100))
            seq_len = int(np.random.randint(2, 25))
            test_inputs = torch.randn(n_traj, seq_len, n_i)
            test_states = torch.randn(n_traj, seq_len, n_s)

            unrolled_model.n_steps = seq_len - 1
            unrolled_model_nm.nsteps = seq_len - 1
            unrolled_model_enc.set_n_steps(seq_len - 1)

            unrolled_out = unrolled_model(test_states[:, 0, :], test_inputs)
            unrolled_loss = torch.nn.MSELoss()(
                unrolled_out[:, 1:, :], test_states[:, 1:, :]
            )
            unrolled_out_enc = unrolled_model_enc(
                test_states[:, 0:1, :], test_inputs[:, 0:1, :], test_inputs
            )
            unrolled_loss_enc = torch.nn.MSELoss()(
                unrolled_out_enc[:, 1:, :], test_states[:, 1:, :]
            )

            batch_nm = {
                "X": test_states,
                "xn": test_states[:, 0:1, :],
                "U": test_inputs,
                "name": "train",
                "epoch": 0,
            }
            x = neuromancer.constraint.variable("X")[:, 1:, :]
            xhat = neuromancer.constraint.variable("xn")[:, 1:, :]
            objectives = [neuroman.define_loss(x, xhat, -1, 1)]
            constraints = []
            loss = neuromancer.loss.PenaltyLoss(objectives, constraints)

            problem = neuromancer.problem.Problem([unrolled_model_nm], loss)

            out = problem(batch_nm)
            unrolled_loss_nm = out["train_loss"]

            self.assertTrue(
                torch.isclose(
                    unrolled_loss,
                    unrolled_loss_nm,
                    rtol=0,
                    atol=1e-20,
                    equal_nan=True,
                )
            )
            self.assertTrue(
                torch.isclose(
                    unrolled_loss,
                    unrolled_loss_enc,
                    rtol=0,
                    atol=1e-20,
                    equal_nan=True,
                )
            )

    def test_training(self):

        n_epochs = self.training_epochs
        clip_grads_val = 100.0
        warmup = 10

        for idx, _ in enumerate(self.n_states):

            unrolled_model = self.models[idx][1]
            unrolled_model_nm = self.models[idx][0]
            unrolled_model_enc = self.models[idx][2]

            seq_len = self.seq_lens[idx]
            unrolled_model.n_steps = seq_len - 1
            unrolled_model_nm.nsteps = seq_len - 1
            unrolled_model_enc.set_n_steps(seq_len - 1)

            seq_dataloader_train = self.dataloader_train_list[idx][1]
            nm_dataloader_train = self.dataloader_train_list[idx][0]
            enc_dataloader_train = self.dataloader_train_list[idx][2]
            seq_dataloader_dev = self.dataloader_dev_list[idx][1]
            nm_dataloader_dev = self.dataloader_dev_list[idx][0]
            enc_dataloader_dev = self.dataloader_dev_list[idx][2]
            seq_dataloader_test = self.dataloader_test_list[idx][1]
            nm_dataloader_test = self.dataloader_test_list[idx][0]
            enc_dataloader_test = self.dataloader_test_list[idx][2]

            x = neuromancer.constraint.variable("X")[:, 1:, :]
            xhat = neuromancer.constraint.variable("xn")[:, 1:, :]
            objectives = [neuroman.define_loss(x, xhat, -1, 1)]
            constraints = []
            loss = neuromancer.loss.PenaltyLoss(objectives, constraints)
            problem = neuromancer.problem.Problem([unrolled_model_nm], loss)

            optimizer_unrolled = torch.optim.Adam(
                unrolled_model.parameters(), lr=0.01
            )
            optimizer_unrolled_nm = torch.optim.Adam(
                problem.parameters(), lr=0.01
            )
            optimizer_unrolled_enc = torch.optim.Adam(
                unrolled_model_enc.parameters(), lr=0.01
            )

            my_trainer = sequencepred.SequenceTrainer(
                model=unrolled_model,
                optimizer=optimizer_unrolled,
                loss_fct=torch.nn.MSELoss(),
                train_loader=seq_dataloader_train,
                dev_loader=seq_dataloader_dev,
                epochs=n_epochs,
                patience=n_epochs,
                clip_grad_threshhold=clip_grads_val,
                warmup=warmup,
                device=torch.device("cpu"),
            )

            nm_trainer = nmtrainer.Trainer(
                problem=problem,
                train_data=nm_dataloader_train,
                dev_data=nm_dataloader_dev,
                test_data=nm_dataloader_test,
                optimizer=optimizer_unrolled_nm,
                epochs=n_epochs,
                patience=n_epochs,
                clip=clip_grads_val,
                warmup=warmup,
                device=torch.device("cpu"),
                eval_metric="mean_dev_loss",
            )

            enc_trainer = mytrainer.SequencePredictionTrainer(
                model=unrolled_model_enc,
                optimizer=optimizer_unrolled_enc,
                loss_fct=torch.nn.MSELoss(),
                train_loader=enc_dataloader_train,
                dev_loader=enc_dataloader_dev,
                epochs=n_epochs,
                patience=n_epochs,
                clip_grad_threshhold=clip_grads_val,
                warmup=warmup,
                device=torch.device("cpu"),
            )

            my_trainer.train()
            nm_trainer.train()
            enc_trainer.train()

            unrolled_model.eval()
            problem.eval()
            unrolled_model_enc.eval()

            unrolled_model.load_state_dict(my_trainer.best_model_state_dict)
            problem.load_state_dict(nm_trainer.best_model)
            unrolled_model_enc.load_state_dict(
                enc_trainer.best_model_state_dict
            )

            my_out = []
            for (x_0, u), _ in seq_dataloader_test:
                out = unrolled_model(x_0, u)[:, 1:, :]
                my_out.append(out)

            nm_out = []
            for batch in nm_dataloader_test:
                out = problem(batch)
                out = out["test_xn"][:, 1:, :]
                nm_out.append(out)

            enc_out = []
            for ((y_hist, u_hist), u_pred), _ in enc_dataloader_test:
                out = unrolled_model_enc(y_hist, u_hist, u_pred)[:, 1:, :]
                enc_out.append(out)

            for my_o, nm_o in zip(my_out, nm_out):
                self.assertTrue(
                    torch.allclose(
                        my_o, nm_o, rtol=1e-4, atol=1e-4, equal_nan=True
                    )
                )

            for my_o, enc_o in zip(my_out, enc_out):
                self.assertTrue(
                    torch.allclose(
                        my_o, enc_o, rtol=1e-4, atol=1e-4, equal_nan=True
                    )
                )


if __name__ == "__main__":
    unittest.main()
