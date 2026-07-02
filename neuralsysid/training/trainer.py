"""This module implements training and evaluation procdures for a model."""

from copy import deepcopy
from typing import Callable, List
import math
from abc import abstractmethod, ABC

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import optuna


class BaseTrainer(ABC):
    """This class represents the base class for training procdures.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): The optimizer to be used for training.
        loss_fct (nn.Module): The loss function to be used for training.
        train_loader (DataLoader): The DataLoader for the training data.
        dev_loader (DataLoader): The DataLoader for the validation data.
        epochs (int): The number of epochs to train for.
        patience (int): The number of epochs to wait before stopping training if
            no improvement is seen.
        warmup (int): The number of epochs to wait before starting to track
            improvements.
        clip_grad_threshhold (float): The threshold for gradient clipping.
            Defaults to float("inf") (no clipping).
        device (device): The device to use for training ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fct: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_loader: DataLoader,
        dev_loader: DataLoader,
        epochs: int = 1000,
        patience: int = 250,
        warmup: int = 100,
        clip_grad_threshhold: float = float("inf"),
        device: torch.device = torch.device("cpu"),
        trial: optuna.Trial = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fct = loss_fct
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.epochs = epochs
        self.patience = patience
        self.warmup = warmup
        self.device = device
        self.clip_grad_threshhold = clip_grad_threshhold
        self.train_loss = []
        self.dev_loss = []
        self.best_model_state_dict = deepcopy(self.model.state_dict())
        self.best_dev_loss = torch.tensor(float("inf"))
        self.no_improvement_counter = 0
        self.current_epoch = 0
        self.trial = trial

        if "mps" in device.type or "cuda" in device.type:
            self.model.to(device=device)
            self.loss_fct.to(device=device)
            self.move_batch = True
        else:
            self.move_batch = False

    def prune_nan(self, loss: float):
        """Prunes a trial soon as a Nan value is encountered in the loss.

        Args:
            loss (torch.Tensor) : loss to be checked for nan.

        Returns:
            bool: True if trial should be pruned.
        """
        if math.isnan(loss):
            return True

        return False

    def prune_unpromising(self, loss: float, report_interval: int = 10):
        """Prunes a trial if the loss is not promising. Therefore reports to
        the RDB backend. Optuna then decides if the trial should be pruned or
        not.

        Args:
            loss (torch.Tensor): loss to be reported to RDB.
            report_interval (int, optional): Interval of epochs to report
                results. Defaults to 20.

        Returns:
            bool: True if trial should be pruned.
        """
        if self.trial and self.current_epoch % report_interval == 0:
            self.trial.report(loss, self.current_epoch)
            if self.trial.should_prune():
                return True

        return False

    def train(self, print_progress: bool = True):
        """Training routine for the model. Runs for a specified number of epochs
        or quits prematurely if no improvement is seen on the dev set.
        Alternately calls the train and eval loops.

        Args:
            print_progress (bool, optional): Whether to print the progress of
                training. Defaults to True.

        Returns:
            bool: True if training completed successfully, False if pruned due
            to NaN values or unpromising results.
        """
        if print_progress:
            dash = "-" * 50
            print("\n")
            print(dash)
            print(f"{'Epoch':<10}{'Train Loss':<25}{'Dev Loss':<25}")
            print(dash)

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            if self.no_improvement_counter > self.patience:
                self.current_epoch -= 1
                print(
                    f"No improvement in dev loss over {self.patience} epochs. Early stopping."
                )
                return True

            self.train_loop()
            if self.prune_nan(self.train_loss[-1]):
                self.train_loss.pop()
                self.current_epoch -= 1
                return False

            self.eval_loop()
            if self.prune_nan(self.dev_loss[-1]):
                self.train_loss.pop()
                self.dev_loss.pop()
                self.current_epoch -= 1
                return False
            if self.dev_loss[-1] < self.best_dev_loss:
                self.best_model_state_dict = deepcopy(self.model.state_dict())
                self.best_dev_loss = self.dev_loss[-1]
                self.no_improvement_counter = 0
            else:
                if self.current_epoch > self.warmup:
                    self.no_improvement_counter += 1

            if self.prune_unpromising(self.dev_loss[-1]):
                return False

            if print_progress:
                print(
                    f"{epoch:<10d}{self.train_loss[-1]:<25.15f}{self.dev_loss[-1]:<25.15f}"
                )

        return True

    def train_loop(self):
        """Training for one epoch. Loops over the batches of the training data
        and updates the model parameters using the optimizer and loss function.
        """
        self.model.train()
        losses = []
        for _, (in_features, out_features) in enumerate(self.train_loader):
            loss = self.calc_objective(in_features, out_features)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_threshhold
            )
            self.optimizer.step()
            losses.append(loss.item())

        train_loss = sum(losses) / len(losses)
        self.train_loss.append(train_loss)

    def eval_loop(self):
        """Evaluation for one epoch. Loops over batches of the validation data
        and computes the mean loss on the dev set.
        """
        self.model.eval()
        losses = []
        with torch.no_grad():
            for _, (in_features, out_features) in enumerate(self.dev_loader):
                loss = self.calc_objective(in_features, out_features)
                losses.append(loss.item())
            dev_loss = sum(losses) / len(losses)
            self.dev_loss.append(dev_loss)

    @abstractmethod
    def calc_objective(self, in_features, out_features):
        """Calculates the objective function for a given batch of data.

        Args:
            in_features (torch.Tensor): The input features.
            out_features (torch.Tensor): The output features.

        Returns:
            torch.Tensor: The objective function value.
        """

    def evaluate(self, data_loader: DataLoader) -> torch.Tensor:
        """Evaluates the model on a given dataset. Loops over the data and
        computes the loss.

        Args:
            data_loader (DataLoader): The DataLoader for the data to be
                evaluated.

        Returns:
            torch.Tensor: The mean loss over the dataset.
        """
        self.model.eval()
        losses = []
        with torch.no_grad():
            for _, (in_features, out_features) in enumerate(data_loader):
                loss = self.calc_objective(in_features, out_features)
                losses.append(loss.item())
            loss = sum(losses) / len(losses)

        self.prune_nan(loss)

        return loss


class SequencePredictionTrainer(BaseTrainer):
    """Trainer for encoder-predictor-decoder sequence prediction models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fct: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_loader: DataLoader,
        dev_loader: DataLoader,
        epochs: int = 1000,
        patience: int = 50,
        warmup: int = 50,
        clip_grad_threshhold: float = float("inf"),
        device: torch.device = torch.device("cpu"),
        trial: optuna.Trial = None,
    ):
        super().__init__(
            model,
            optimizer,
            loss_fct,
            train_loader,
            dev_loader,
            epochs,
            patience,
            warmup,
            clip_grad_threshhold,
            device,
            trial,
        )

    def calc_objective(self, in_features, out_features):
        """Calculates the objective function for a given batch of data.

        Args:
            in_features (Tuple(torch.Tensor)): The input features. x_0 is at
                index 0, u is at index 1.
            out_features (torch.Tensor): The output features.

        Returns:
            torch.Tensor: The objective function value.
        """
        if self.move_batch:
            in_features_y_h = in_features[0][0].to(device=self.device)
            in_features_u_h = in_features[0][1].to(device=self.device)
            in_features_u = in_features[1].to(device=self.device)
            out_features = out_features.to(device=self.device)
            out = self.model(in_features_y_h, in_features_u_h, in_features_u)
        else:
            in_features_y_h = in_features[0][0]
            in_features_u_h = in_features[0][1]
            in_features_u = in_features[1]
            out = self.model(in_features_y_h, in_features_u_h, in_features_u)
        loss = self.loss_fct(
            out[:, 1:, :],
            out_features[:, 1:, :],
        )

        return loss
