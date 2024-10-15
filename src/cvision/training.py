"""This module provides a ModelTrainer class that can be used to train models with a convenient API.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Any, Literal


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
        accuracy_fn: callable[[torch.Tensor, torch.Tensor], float],
        optimizer: torch.optim.Optimizer,
        device: Device = "cpu"
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
    """
        self._model = model
        self._train_loader = train_loader
        self._eval_loader = eval_loader
        self._loss_fn = loss_fn
        self._accuracy_fn = accuracy_fn
        self._optimizer = optimizer
        self._device = device

    def _train_step():
        """Performs a single training step. Equivalent to one epoch."""
        pass


    def _test_step():
        """Performs a single validation step across all batches in dataloader."""
        pass


    def train():
        """Runs a full training pass for given number of epochs."""
        pass