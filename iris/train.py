# iris/train.py
import json
from argparse import Namespace
from typing import Dict, Tuple
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from iris import data, eval, models, utils

lr = 1e-2


class Trainer:
    """Object used to facilitate training"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        trial: optuna.trial._trial.Trial = None,
    ):
        # set params
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trial = trial

    def train_step(self, dataloader: torch.utils.data.DataLoader):
        """Train step
        Args:
            dataloader: torch dataloader to load batches from
        """
        # set model to train mode
        self.model.train()
        loss = 0.0

        # iterate over train batches
        for i, batch in enumerate(dataloader):
            # step
            batch = [item.to(self.device) for item in batch]  # set device
            inputs, targets = batch[:-1][0], batch[-1]
            self.optimizer.zero_grad()  # reset gradients
            z = self.model(inputs)  # Forward pass
            J = self.loss_fn(z, targets)  # Define loss
            J.backward()  # backward pass
            self.optimizer.step()  # update weights

            # Cumulative Metrics
            loss += (J.detach().item() - loss) / (i + 1)

        return loss

    def eval_step(self, dataloader: torch.utils.data.DataLoader):
        """Evaluation (val/test) step
        Args:
            dataloader : torch.dataloader to load batches from
        """
        # set model to eval mode
        self.model.eval()
        loss = 0.0
        y_trues, y_probs = [], []

        # Iterate over val batches
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                # Step
                batch = [item.to(self.device) for item in batch]  # set device
                inputs, y_true = batch[:-1][0], batch[-1]
                z = self.model(inputs)  # forward pass
                J = self.loss_fn(z, y_true).item()

                # Cumelative metrics
                loss += (J - loss) / (i + 1)

                # store outputs
                y_prob = torch.sigmoid(z).cpu().numpy()
                y_probs.extend(y_prob)
                y_trues.extend(y_true.cpu().numpy())

        return loss, np.vstack(y_trues), np.vstack(y_probs)

    def predict_step(self, dataloader: torch.utils.data.DataLoader):
        """Predictin function ( inference step)

        Note:
            Loss is not calculated for this loop
        Args:
            dataloader : torch dataloader to load batches from
        """
        # set model to eval mode
        self.model.eval()
        y_trues, y_probs = [], []

        # Iterate over batchs
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                # Forward pass
                batch = [item.to(self.device) for item in batch]
                inputs, y_true = batch[:-1][0], batch[-1]
                z = self.model(inputs)

                # Store outputs
                y_prob = torch.sigmoid(z).cpu().numpy()
                y_probs.extend(y_prob)
                y_trues.extend(y_true.cpu().numpy())

        return np.vstack(y_trues), np.vstack(y_probs)

    def train(
        self,
        num_epochs: int,
        patience: int,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
    ) -> Tuple:
        """Training loop
        Args:
            num epochs (int): max num of epochs to train for 9 can stop early if not model not imporving
            patience (int): Number of acceptable epochs for continuous degrading performance.
            train_dataloader: dataloader object with trainig data split
            val_dataloader: dataloader object with validation data split
        Raises:
            optuna.TrialPruned: Early stopping of the optimization trial if poor performance.

        Returns:
            The best validation loss and the trained model from that point.
        """

        best_val_loss = np.inf
        best_model = None
        _patience = patience
        for epoch in range(num_epochs):
            # steps
            train_loss = self.train_step(dataloader=train_dataloader)
            val_loss, _, _ = self.eval_step(dataloader=val_dataloader)
            self.scheduler.step(val_loss)

            # Pruning based on the intemediate valus
            if self.trial:
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    print("failure trials pruned!")
                    raise optuna.TrialPruned()

            # Early stoping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                _patience = patience  # reset _patience
            else:
                _patience -= 1
            if not _patience:
                print("Stopping early!")
                break

            # to logging future
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}, "
                f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, "
                f"_patience: {_patience}"
            )
        return best_val_loss, best_model


def train(params: Namespace = None, trial: optuna.trial._trial.Trial = None) -> Dict:
    """Operations for training
    ARGS:
        params (Namespace): Iput params for operations.
        trial ( optuna.trial._trial.Trail,optional): Optuna optimization trial, defaults to None
    Returns:
        Artifacts to save and load for later"""

    utils.set_seed(seed=params.seed)
    device = utils.set_device(cuda=params.cuda)

    # Get data this data clensing can be done seperately later
    path = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    df = pd.read_csv(path, header=None, names=["f1", "f2", "f3", "f4", "class"])
    df = df.sample(frac=1).reset_index(drop=True)
    df["class"] = LabelEncoder().fit_transform(df["class"])
    train_df = df[: int(len(df) * 0.8)].reset_index(drop=True)
    test_df = df[int(len(df) * 0.8) : int(len(df) * 0.9)].reset_index(drop=True)
    val_df = df[int(len(df) * 0.9) :].reset_index(drop=True)
    X_train, y_train = train_df.values[:, :-1], train_df.values[:, -1]
    X_val, y_val = val_df.values[:, :-1], val_df.values[:, -1]

    train_dataset = data.CSVDataset(X=X_train, y=y_train)
    val_dataset = data.CSVDataset(X=X_val, y=y_val)
    train_dataloader = train_dataset.get_dataloader(batch_size=params.batch_size)
    val_dataloader = val_dataset.get_dataloader(batch_size=params.batch_size)

    model = models.initialize_model(device=torch.device("cpu"))

    # Trainer module
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )
    trainer = Trainer(
        model=model,
        device=device,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        trial=trial,
    )

    # Train
    best_val_loss, best_model = trainer.train(100, 10, train_dataloader, val_dataloader)

    # Find best threshold
    # y_true, y_prob = trainer.eval_step(dataloader=train_dl)
    # params.threshold = find_best_threshold(y_true=y_true, y_prob=y_prob)

    # Evaluate model
    artifacts = {
        "params": params,
        "model": best_model,
        "loss": best_val_loss,
    }
    device = torch.device("cpu")
    y_true, y_pred, performance = eval.evaluate(df=test_df, artifacts=artifacts)
    artifacts["performance"] = performance

    return artifacts


def objective(params: Namespace, trial: optuna.trial._trial.Trial) -> float:
    """Objective function for optimization trials.

    Args:
        params (Namespace): Input parameters for each trial (see `config/params.json`).
        trial (optuna.trial._trial.Trial): Optuna optimization trial.

    Returns:
        F1 score from evaluating the trained model on the test data split.
    """
    # Paramters (to tune)
    params.hidden_dim = trial.suggest_int("hidden_dim", 16, 32)
    params.dropout_p = trial.suggest_uniform("dropout_p", 0.3, 0.8)
    params.lr = trial.suggest_loguniform("lr", 5e-5, 5e-4)

    # Train (can move some of these outside for efficiency)
    print(f"\nTrial {trial.number}:")
    print(json.dumps(trial.params, indent=2))
    artifacts = train(params=params, trial=trial)

    # Set additional attributes
    params = artifacts["params"]
    performance = artifacts["performance"]
    print(json.dumps(performance["overall"], indent=2))
    trial.set_user_attr("precision", performance["overall"]["precision"])
    trial.set_user_attr("recall", performance["overall"]["recall"])
    trial.set_user_attr("f1", performance["overall"]["f1"])

    return performance["overall"]["f1"]