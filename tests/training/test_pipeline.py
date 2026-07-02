import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

# import utils_nm.learn_dynamics_nm as ldnm
# import utils_nm.preparation_nm as dpnm
from tests.reference_impls import neuroman, onesteppred, sequencepred

# import utils.preparation as dp
# import utils.cyclic_computation as cc

from neuralsysid.data.preprocessing import (
    normalize_data,
    resample_data_sets,
    unnormalize,
)

import configurations.test as mdconfigs
from neuralsysid.data import io, preparation
from neuralsysid.training import pipeline

torch.manual_seed(0)
np.random.seed(0)


def _nested_test_config(flat_config):
    dynamics = {
        "type": flat_config["dynamics"],
        "stepper": flat_config["stepper"],
        "latent_state_dim": flat_config["latent_state_dim"],
        "n_context": flat_config.get("n_context", 0),
        "act_fct": flat_config["act_fct"],
        "dropout": flat_config.get("dropout_dyn", 0.0),
    }
    if flat_config["dynamics"] == "fxu":
        dynamics["n_layers"] = flat_config["n_layers_dyn"]
        dynamics["n_hidden"] = flat_config["n_hidden_dyn"]
    else:
        dynamics["n_layers_A"] = flat_config["n_layers_A"]
        dynamics["n_hidden_A"] = flat_config["n_hidden_A"]
        dynamics["n_layers_B"] = flat_config["n_layers_B"]
        dynamics["n_hidden_B"] = flat_config["n_hidden_B"]

    return {
        "data": {
            "t_step": flat_config["t_step"],
            "outputs": deepcopy(flat_config["outputs"]),
            "controls": deepcopy(flat_config["controls"]),
            "outputs_as_latent_states": deepcopy(
                flat_config["outputs_as_latent_states"]
            ),
            "seq_len": flat_config["seq_len"],
            "hist_seq_len": flat_config["hist_seq_len"],
            "overlap": flat_config["overlap"],
        },
        "model": {
            "dynamics": dynamics,
            "encoder": {
                "type": flat_config["encoder"],
                "n_layers": flat_config["n_layers_enc"],
                "n_hidden": flat_config["n_hidden_enc"],
                "dropout": flat_config["dropout_enc"],
                "input_mode": "sequence",
                "input_sources": "yu",
            },
            "decoder": None,
        },
        "train": {
            "learning_rate": flat_config["learning_rate"],
            "n_epochs": flat_config["n_epochs"],
            "patience": flat_config["patience"],
            "warmup": flat_config["warmup"],
            "grad_clip": flat_config["grad_clip"],
            "batch_size": flat_config["batch_size"],
            "shuffle": flat_config["shuffle"],
            "drop_last": False,
        },
        "meta": {
            "config_id": flat_config["config_id"],
        },
    }


class TestLearnDynamics(unittest.TestCase):

    def setUp(self):
        self.data_name = "tests/testdata/2024-08-13_11-24-24_reeses_"
        self.test_cases = 10

    def test_onestep_equivalence(self):

        dataSets, metadata = io.load_data(
            self.data_name, extensions=["train", "dev", "test"]
        )
        sim_config = metadata["simulation_config"]
        for _, data_set in dataSets:
            data_set["X"] = data_set["Y"]

        mod_configs = mdconfigs.test_configs_onestep

        chip_type = "cpu"
        hyperparameters = []
        for _ in range(self.test_cases):
            mod_config = deepcopy(mod_configs)
            mod_config["dynamics"] = np.random.choice(mod_config["dynamics"])
            mod_config["stepper"] = np.random.choice(mod_config["stepper"])

            if mod_config["dynamics"] == "fxu":
                mod_config["n_hidden_dyn"] = mod_config.pop("n_hidden_A")
                mod_config["n_layers_dyn"] = mod_config.pop("n_layers_A")
                mod_config.pop("n_hidden_B")
                mod_config.pop("n_layers_B")

            print(
                "testing configuration: ",
                mod_config["dynamics"],
                mod_config["stepper"],
            )
            self.learn_predict_and_check_onestep(
                mod_config=mod_config,
                sim_config=sim_config,
                hyperparameters=hyperparameters,
                chip_type=chip_type,
                dataSets=dataSets,
            )

    def test_multistep_equivalence(self):

        dataSets, metadata = io.load_data(
            self.data_name, extensions=["train", "dev", "test"]
        )
        sim_config = metadata["simulation_config"]
        for _, data_set in dataSets:
            data_set["X"] = data_set["Y"]

        mod_configs = mdconfigs.test_configs_multistep

        chip_type = "cpu"
        hyperparameters = []
        for _ in range(self.test_cases):
            mod_config = deepcopy(mod_configs)
            mod_config["dynamics"] = np.random.choice(mod_config["dynamics"])
            mod_config["stepper"] = np.random.choice(mod_config["stepper"])

            if mod_config["dynamics"] == "fxu":
                mod_config["n_hidden_dyn"] = mod_config.pop("n_hidden_A")
                mod_config["n_layers_dyn"] = mod_config.pop("n_layers_A")
                mod_config.pop("n_hidden_B")
                mod_config.pop("n_layers_B")

            print(
                "testing configuration: ",
                mod_config["dynamics"],
                mod_config["stepper"],
            )
            self.learn_predict_and_check_multistep(
                mod_config=mod_config,
                sim_config=sim_config,
                hyperparameters=hyperparameters,
                chip_type=chip_type,
                dataSets=dataSets,
            )

    def learn_predict_and_check_onestep(
        self,
        mod_config,
        sim_config,
        hyperparameters,
        chip_type,
        dataSets,
    ):

        dynamics_model_nm, _ = neuroman.learn_dynamics(
            deepcopy(mod_config),
            deepcopy(sim_config),
            deepcopy(hyperparameters),
            deepcopy(chip_type),
            deepcopy(dataSets),
        )

        dynamics_model, _ = onesteppred.learn_dynamics(
            deepcopy(mod_config),
            deepcopy(sim_config),
            deepcopy(hyperparameters),
            deepcopy(chip_type),
            deepcopy(dataSets),
        )

        dynamics_model_seq, _ = sequencepred.learn_dynamics(
            deepcopy(mod_config),
            deepcopy(sim_config),
            deepcopy(hyperparameters),
            deepcopy(chip_type),
            deepcopy(dataSets),
        )

        dynamics_model_enc, _, _ = pipeline.learn_dynamics(
            _nested_test_config(deepcopy(mod_config)),
            deepcopy(hyperparameters),
            deepcopy(chip_type),
            deepcopy(dataSets),
            unit_test=True,
        )

        pred_traj_nm, true_traj_nm, input_traj_nm = self.predict_nm_traj(
            dynamics_model_nm,
            deepcopy(dataSets),
            deepcopy(sim_config),
            deepcopy(mod_config),
        )

        pred_traj, true_traj, input_traj = self.predict_traj_osh(
            dynamics_model,
            deepcopy(dataSets),
            deepcopy(sim_config),
            deepcopy(mod_config),
        )

        pred_traj_seq, true_traj_seq, input_traj_seq = self.predict_traj_seq(
            dynamics_model_seq,
            deepcopy(dataSets),
            deepcopy(sim_config),
            deepcopy(mod_config),
        )

        pred_traj_enc, true_traj_enc, input_traj_enc = self.predict_traj_enc(
            dynamics_model_enc,
            deepcopy(dataSets),
            deepcopy(sim_config),
            deepcopy(mod_config),
        )

        self.assertTrue(
            np.allclose(pred_traj, pred_traj_nm, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            np.allclose(true_traj, true_traj_nm, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            np.allclose(input_traj, input_traj_nm, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            np.allclose(pred_traj_seq, pred_traj_nm, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            np.allclose(true_traj_seq, true_traj_nm, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            np.allclose(input_traj_seq, input_traj_nm, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            np.allclose(pred_traj_enc, pred_traj_nm, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            np.allclose(true_traj_enc, true_traj_nm, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            np.allclose(input_traj_enc, input_traj_nm, atol=1e-6, rtol=1e-6)
        )

    def learn_predict_and_check_multistep(
        self,
        mod_config,
        sim_config,
        hyperparameters,
        chip_type,
        dataSets,
    ):

        dynamics_model_nm, _ = sequencepred.learn_dynamics(
            deepcopy(mod_config),
            deepcopy(sim_config),
            deepcopy(hyperparameters),
            deepcopy(chip_type),
            deepcopy(dataSets),
        )

        dynamics_model, _, _ = pipeline.learn_dynamics(
            _nested_test_config(deepcopy(mod_config)),
            deepcopy(hyperparameters),
            deepcopy(chip_type),
            deepcopy(dataSets),
            unit_test=True,
        )

        pred_traj_nm, true_traj_nm, input_traj_nm = self.predict_traj_seq(
            dynamics_model_nm,
            deepcopy(dataSets),
            deepcopy(sim_config),
            deepcopy(mod_config),
        )

        pred_traj, true_traj, input_traj = self.predict_traj_enc(
            dynamics_model,
            deepcopy(dataSets),
            deepcopy(sim_config),
            deepcopy(mod_config),
        )

        self.assertTrue(
            np.allclose(pred_traj, pred_traj_nm, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            np.allclose(true_traj, true_traj_nm, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            np.allclose(input_traj, input_traj_nm, atol=1e-6, rtol=1e-6)
        )

    def predict_nm_traj(
        self, dynamics_model, dataSets, sim_config, mod_config
    ):

        dataToPlot = 0

        dataSets = resample_data_sets(
            dataSets,
            t_sample=sim_config["T_sample"],
            t_step=mod_config["t_step"],
        )

        tTrain, dataSetTrain = dataSets[0]
        tDev, dataSetDev = dataSets[1]
        tTest, dataSetTest = dataSets[2]

        dataSetTrainNormalized, dataStatsTrain = normalize_data(dataSetTrain)
        dataSetDevNormalized, _ = normalize_data(dataSetDev, dataStatsTrain)
        dataSetTestNormalized, _ = normalize_data(dataSetTest, dataStatsTrain)
        dataSetTrainNormalized["X"] = dataSetTrainNormalized["Y"]
        dataSetDevNormalized["X"] = dataSetDevNormalized["Y"]
        dataSetTestNormalized["X"] = dataSetTestNormalized["Y"]

        plottableSets = neuroman.generate_batch_sets(
            [tTrain, tDev, tTest],
            [
                dataSetTrainNormalized,
                dataSetDevNormalized,
                dataSetTestNormalized,
            ],
            1,  # this is irrelevant here as dataSetType=test automatically wraps it correctly
            sim_config["used_states"],
            sim_config["used_controls"],
            batch_size=100,  # irrelevant here because automatic wrapping of dataSetType=test
            data_set_types=["test", "test", "test"],
        )

        tDataEval, evaluatedDataSet, _ = plottableSets[dataToPlot]

        prediction_horizon = evaluatedDataSet["X"].shape[1]
        dynamics_model.nsteps = prediction_horizon

        test_outputs = dynamics_model(evaluatedDataSet)

        nx = len(sim_config["used_states"])
        nu = len(sim_config["used_controls"])

        # test_outputs = dynamics_model(evaluatedDataSet)
        pred_traj_norm = (
            test_outputs["xn"][:, :-1, :].detach().numpy().reshape(-1, nx)
        )
        true_traj_norm = evaluatedDataSet["X"].detach().numpy().reshape(-1, nx)
        input_traj_norm = (
            evaluatedDataSet["U"].detach().numpy().reshape(-1, nu)
        )

        pred_traj = unnormalize(
            pred_traj_norm,
            dataStatsTrain["mean_y"][sim_config["used_states"]],
            dataStatsTrain["std_y"][sim_config["used_states"]],
        )
        true_traj = unnormalize(
            true_traj_norm,
            dataStatsTrain["mean_y"][sim_config["used_states"]],
            dataStatsTrain["std_y"][sim_config["used_states"]],
        )
        input_traj = unnormalize(
            input_traj_norm,
            dataStatsTrain["mean_u"][sim_config["used_controls"]],
            dataStatsTrain["std_u"][sim_config["used_controls"]],
        )

        return pred_traj, true_traj, input_traj

    def predict_traj_osh(
        self, dynamics_model, dataSets, sim_config, mod_config
    ):

        dataToPlot = 0

        dataSets = resample_data_sets(
            dataSets,
            t_sample=sim_config["T_sample"],
            t_step=mod_config["t_step"],
        )

        tTrain, dataSetTrain = dataSets[0]
        tDev, dataSetDev = dataSets[1]
        tTest, dataSetTest = dataSets[2]

        dataSetTrainNormalized, dataStatsTrain = normalize_data(dataSetTrain)
        dataSetDevNormalized, _ = normalize_data(dataSetDev, dataStatsTrain)
        dataSetTestNormalized, _ = normalize_data(dataSetTest, dataStatsTrain)
        dataSetTrainNormalized["X"] = dataSetTrainNormalized["Y"]
        dataSetDevNormalized["X"] = dataSetDevNormalized["Y"]
        dataSetTestNormalized["X"] = dataSetTestNormalized["Y"]

        plottableSets = onesteppred.generate_batch_sets(
            [tTrain, tDev, tTest],
            [
                dataSetTrainNormalized,
                dataSetDevNormalized,
                dataSetTestNormalized,
            ],
            sim_config["used_states"],
            sim_config["used_controls"],
            batch_size=mod_config["batch_size"],
            shuffle=False,
        )

        tDataEval, evaluatedDataSet, _ = plottableSets[dataToPlot]

        prediction_horizon = evaluatedDataSet.in_features.shape[0]
        pred_traj_norm = (
            onesteppred.cyclic_computation(
                dynamics_model, evaluatedDataSet, prediction_horizon
            )
            .detach()
            .numpy()
        )
        x_0 = evaluatedDataSet.in_features[
            0:1, : dynamics_model.dynamics.n_states
        ]
        x_1K = evaluatedDataSet.out_features
        true_traj_norm = torch.cat((x_0, x_1K), 0).detach().numpy()

        u_0Km1 = evaluatedDataSet.in_features[
            :, dynamics_model.dynamics.n_states :
        ]
        u_K = evaluatedDataSet.in_features[
            -1:, dynamics_model.dynamics.n_states :
        ]  # just repeat the last input
        input_traj_norm = torch.cat((u_0Km1, u_K), 0).detach().numpy()

        pred_traj = unnormalize(
            pred_traj_norm,
            dataStatsTrain["mean_y"][sim_config["used_states"]],
            dataStatsTrain["std_y"][sim_config["used_states"]],
        )
        true_traj = unnormalize(
            true_traj_norm,
            dataStatsTrain["mean_y"][sim_config["used_states"]],
            dataStatsTrain["std_y"][sim_config["used_states"]],
        )
        input_traj = unnormalize(
            input_traj_norm,
            dataStatsTrain["mean_u"][sim_config["used_controls"]],
            dataStatsTrain["std_u"][sim_config["used_controls"]],
        )

        return pred_traj, true_traj, input_traj

    def predict_traj_seq(
        self, dynamics_model, dataSets, sim_config, mod_config
    ):

        dataToPlot = 0

        dataSets = resample_data_sets(
            dataSets,
            t_sample=sim_config["T_sample"],
            t_step=mod_config["t_step"],
        )

        tTrain, dataSetTrain = dataSets[0]
        tDev, dataSetDev = dataSets[1]
        tTest, dataSetTest = dataSets[2]

        dataSetTrainNormalized, dataStatsTrain = normalize_data(dataSetTrain)
        dataSetDevNormalized, _ = normalize_data(dataSetDev, dataStatsTrain)
        dataSetTestNormalized, _ = normalize_data(dataSetTest, dataStatsTrain)
        dataSetTrainNormalized["X"] = dataSetTrainNormalized["Y"]
        dataSetDevNormalized["X"] = dataSetDevNormalized["Y"]
        dataSetTestNormalized["X"] = dataSetTestNormalized["Y"]

        seq_len = tTrain.shape[1]

        plottableSets = sequencepred.generate_batch_sets_sequence(
            [tTrain, tDev, tTest],
            [
                dataSetTrainNormalized,
                dataSetDevNormalized,
                dataSetTestNormalized,
            ],
            sim_config["used_states"],
            sim_config["used_controls"],
            batch_size=mod_config["batch_size"],
            shuffle=False,
            seq_len=seq_len,
            overlap=0,
        )

        tDataEval, evaluatedDataSet, evaluatedDataloader = plottableSets[
            dataToPlot
        ]
        dynamics_model.n_steps = seq_len - 1

        (x_0, input_traj_norm), true_traj_norm = next(
            iter(evaluatedDataloader)
        )
        pred_traj_norm = dynamics_model(x_0, input_traj_norm).detach().numpy()

        pred_traj_norm = np.squeeze(pred_traj_norm, axis=0)
        true_traj_norm = np.squeeze(true_traj_norm.detach().numpy(), axis=0)
        input_traj_norm = np.squeeze(input_traj_norm.detach().numpy(), axis=0)

        pred_traj = unnormalize(
            pred_traj_norm,
            dataStatsTrain["mean_y"][sim_config["used_states"]],
            dataStatsTrain["std_y"][sim_config["used_states"]],
        )
        true_traj = unnormalize(
            true_traj_norm,
            dataStatsTrain["mean_y"][sim_config["used_states"]],
            dataStatsTrain["std_y"][sim_config["used_states"]],
        )
        input_traj = unnormalize(
            input_traj_norm,
            dataStatsTrain["mean_u"][sim_config["used_controls"]],
            dataStatsTrain["std_u"][sim_config["used_controls"]],
        )

        return pred_traj, true_traj, input_traj

    def predict_traj_enc(
        self, dynamics_model, dataSets, sim_config, mod_config
    ):

        dataToPlot = 0

        dataSets = resample_data_sets(
            dataSets,
            t_sample=sim_config["T_sample"],
            t_step=mod_config["t_step"],
        )

        tTrain, dataSetTrain = dataSets[0]
        tDev, dataSetDev = dataSets[1]
        tTest, dataSetTest = dataSets[2]

        dataSetTrainNormalized, dataStatsTrain = normalize_data(dataSetTrain)
        dataSetDevNormalized, _ = normalize_data(dataSetDev, dataStatsTrain)
        dataSetTestNormalized, _ = normalize_data(dataSetTest, dataStatsTrain)
        dataSetTrainNormalized["X"] = dataSetTrainNormalized["Y"]
        dataSetDevNormalized["X"] = dataSetDevNormalized["Y"]
        dataSetTestNormalized["X"] = dataSetTestNormalized["Y"]

        seq_len = tTrain.shape[1]

        plottableSets = preparation.build_sequence_dataloaders(
            [tTrain, tDev, tTest],
            [
                dataSetTrainNormalized,
                dataSetDevNormalized,
                dataSetTestNormalized,
            ],
            sim_config["used_states"],
            sim_config["used_controls"],
            batch_size=mod_config["batch_size"],
            shuffle=False,
            seq_len=seq_len,
            overlap=0,
            historic_seq_len=1,
            encode=False,
        )

        tDataEval, evaluatedDataSet, evaluatedDataLoader = plottableSets[
            dataToPlot
        ]
        dynamics_model.set_n_steps(seq_len - 1)

        ((y_hist, u_hist), input_traj_norm), true_traj_norm = next(
            iter(evaluatedDataLoader)
        )
        pred_traj_norm = (
            dynamics_model(y_hist, u_hist, input_traj_norm).detach().numpy()
        )

        pred_traj_norm = np.squeeze(pred_traj_norm, axis=0)
        true_traj_norm = np.squeeze(true_traj_norm.detach().numpy(), axis=0)
        input_traj_norm = np.squeeze(input_traj_norm.detach().numpy(), axis=0)

        pred_traj = unnormalize(
            pred_traj_norm,
            dataStatsTrain["mean_y"][sim_config["used_states"]],
            dataStatsTrain["std_y"][sim_config["used_states"]],
        )
        true_traj = unnormalize(
            true_traj_norm,
            dataStatsTrain["mean_y"][sim_config["used_states"]],
            dataStatsTrain["std_y"][sim_config["used_states"]],
        )
        input_traj = unnormalize(
            input_traj_norm,
            dataStatsTrain["mean_u"][sim_config["used_controls"]],
            dataStatsTrain["std_u"][sim_config["used_controls"]],
        )

        return pred_traj, true_traj, input_traj


if __name__ == "__main__":
    unittest.main()
