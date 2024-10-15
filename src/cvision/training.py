"""This module provides a ModelTrainer class that can be used to train models with a convenient API.
"""

import datetime
import mlflow
import torch
import uuid
from torch import nn
from torch.utils.data import DataLoader
from typing import Any, Literal, Callable


type Device = Literal["cpu", "cuda", "mps"]


class ModelTrainer:
    """Model trainer with built in MLFlow support."""

    def __init__(
        self,
        *args,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        loss_fn: nn.Module,
        accuracy_fn: Callable[[torch.Tensor, torch.Tensor], float],
        optimizer: torch.optim.Optimizer,
        device: Device = "cpu",
        mlflow_uri: str | None = None,
        mlflow_exp_name: str | None = None
    ):
        """Model trainer with built in MLFlow support.
    
        Args
            model (nn.Module):
                PyTorch model instance.
            train_loader (DataLoader):
                PyTorch compatible dataloader built around the training dataset.
            test_loader (DataLoader):
                PyTorch compatible dataloader built around the validation dataset.
            loss_fn (nn.Module):
                Loss function instance. Supported types -> https://pytorch.org/docs/stable/nn.html#loss-functions
            accuracy_fn (callable[[torch.Tensor, torch.Tensor], float]):
                Function used to calculate % accuracy. Should accept 2 PyTorch tensors and return a float.
                Provided tensors contain actual and predicted values.
            optimizer (torch.optim.Optimizer):
                One of optimizer instances supported by PyTorch -> https://pytorch.org/docs/stable/optim.html#algorithms
            device (Device):
                Tensor processing device. Defaults to CPU, also supports CUDA and MPS. However, MPS implementation
                is sometimes incompatible with models.
            mlflow_uri (str | None):
                Optional URI of MLFlow tracking server. If not provided, then experiment tracking won't be used.
            mlflow_exp_name (str | None):
                Optional MLFlow experiment name. Will be randomly generated if not provided and tracking is used.
        """
        self._model = model
        self._train_loader = train_loader
        self._eval_loader = eval_loader
        self._loss_fn = loss_fn
        self._accuracy_fn = accuracy_fn
        self._optimizer = optimizer
        self._device = device
        self._mlflow_uri = mlflow_uri
        self._client = None
        self._mlflow_exp_name = mlflow_exp_name
        self._mlflow_exp_id = None

        if self._mlflow_uri is not None:
            # TODO: validate uri & server connection
            self._client = mlflow.tracking.MlflowClient(tracking_uri=self._mlflow_uri)
            if self._mlflow_exp_name is None:
                self._mlflow_exp_name = str(uuid.uuid4())
            experiment = self._client.get_experiment_by_name(self._mlflow_exp_name)
            if experiment is None:
                self._client.create_experiment(self._mlflow_exp_name)
                experiment = self._client.get_experiment_by_name(self._mlflow_exp_name)
            self._mlflow_exp_id = experiment.experiment_id


    def _train_step(self) -> dict[str, float]:
        """Performs a single training step. Equivalent to one epoch.
        
        Returns:
            Dictionary containing average loss and accuracy for training epoch.
        """
        # TODO: verbose mode
        self._model.train()
        self._model.to(self._device)
        epoch_loss, epoch_acc = 0.0, 0.0

        for batch, (X, y) in enumerate(self._train_loader):
            X, y = X.to(self._device), y.to(self._device)
            forward_pass = self._model(X)
            batch_loss = self._loss_fn(forward_pass, y)
            epoch_loss += batch_loss.item()
            self._optimizer.zero_grad()
            batch_loss.backward()
            self._optimizer.step()
            probs = torch.softmax(forward_pass, dim=1)
            y_hat = torch.argmax(probs, dim=1)
            batch_acc = self._accuracy_fn(y, y_hat)
            epoch_acc += batch_acc

        epoch_loss /= len(self._train_loader)
        epoch_acc /= len(self._train_loader)

        return {"loss": epoch_loss, "accuracy": epoch_acc}


    def _test_step(self) -> dict[str, float]:
        """Performs a single validation step across all batches in dataloader.

        Returns:
            Dictionary containing average loss and accuracy for all batches in validation set.
        """
        # TODO: verbose mode
        self._model.eval()
        self._model.to(self._device)
        total_loss, total_acc = 0.0, 0.0

        with torch.inference_mode():
            for batch, (X, y) in enumerate(self._eval_loader):
                X, y = X.to(self._device), y.to(self._device)
                forward_pass = self._model(X)
                batch_loss = self._loss_fn(forward_pass, y)
                total_loss += batch_loss.item()
                probs = torch.softmax(forward_pass, dim=1)
                y_hat = torch.argmax(probs, dim=1)
                batch_acc = self._accuracy_fn(y, y_hat)
                total_acc += batch_acc

            total_loss /= len(self._eval_loader)
            total_acc /= len(self._eval_loader)

        return {"loss": total_loss, "accuracy": total_acc}


    def train(self, n_epochs: int):
        """Runs a full training pass for given number of epochs.
        
        Args:
            n_epochs (int):
                # of training epochs.
        """
        log = self._client is not None
        run = None
        if log:
            run = self._client.create_run(
                experiment_id=self._mlflow_exp_id,
                run_name=f"{self._mlflow_exp_name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
            )

        for epoch in range(n_epochs):
            epoch_res = self._train_step()
            epoch_eval = self._test_step()
            # TODO: additional logs
            if log:
                self._client.log_metric(
                    run_id=run.info.run_id, key="train_loss", value=epoch_res["loss"], step=epoch
                )
                self._client.log_metric(
                    run_id=run.info.run_id, key="train_acc", value=epoch_res["accuracy"], step=epoch
                )
                self._client.log_metric(
                    run_id=run.info.run_id, key="val_loss", value=epoch_eval["loss"], step=epoch
                )
                self._client.log_metric(
                    run_id=run.info.run_id, key="val_acc", value=epoch_eval["accuracy"], step=epoch
                )
                
            print(f"Epoch #{epoch+1} | Train loss {epoch_res['loss']:.4f} | Train acc {epoch_res['accuracy']:.4f}")
            print(f"Epoch #{epoch+1} | Val loss {epoch_eval['loss']:.4f} | Val acc {epoch_eval['accuracy']:.4f}")