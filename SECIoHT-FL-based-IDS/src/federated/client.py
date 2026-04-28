"""
Federated Learning Client with Differential Privacy.

Trains models locally with privacy guarantees using Opacus.
"""

import copy
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Federated learning client with differential privacy support.

    Args:
        client_id: Unique identifier for this client
        model: Neural network model
        train_loader: DataLoader for training data
        device: Device to run computations on
        learning_rate: Learning rate for optimizer
        enable_dp: Whether to enable differential privacy
        epsilon: Privacy budget (lower = more private)
        delta: Privacy parameter
        max_grad_norm: Maximum gradient norm for clipping
        noise_multiplier: Noise multiplier for DP

    Example:
        >>> model = MLP(44, [128, 64, 32], 2)
        >>> client = FederatedClient(
        ...     client_id='client_0',
        ...     model=model,
        ...     train_loader=train_loader,
        ...     device=torch.device('cpu'),
        ...     enable_dp=True,
        ...     epsilon=10.0
        ... )
        >>> client.train_one_epoch(epoch=0)
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.01,
        enable_dp: bool = True,
        epsilon: float = 10.0,
        delta: float = 1e-4,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
    ):
        self.client_id = client_id
        self.device = device
        self.enable_dp = enable_dp

        # Create a fresh copy of the model for this client
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.criterion = nn.CrossEntropyLoss()

        # Privacy parameters
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=0.9
        )

        # Initialize privacy engine if DP is enabled
        self.privacy_engine = None
        if self.enable_dp:
            self._setup_privacy_engine()

        logger.info(
            f"{client_id} initialized with DP={'enabled' if enable_dp else 'disabled'}"
        )

    def _setup_privacy_engine(self) -> None:
        """Set up the Opacus privacy engine for differential privacy."""
        self.privacy_engine = PrivacyEngine(secure_mode=False)

        # Make model, optimizer, and data loader private
        (
            self.model,
            self.optimizer,
            self.train_loader,
        ) = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        logger.info(
            f"{self.client_id}: Privacy engine initialized "
            f"(ε={self.epsilon}, δ={self.delta})"
        )

    @property
    def parameters(self) -> List[torch.Tensor]:
        """Get model parameters as a list."""
        return [param.data.clone() for param in self.model.parameters()]

    @property
    def num_samples(self) -> int:
        """Get the number of training samples."""
        return len(self.train_loader.dataset)

    def update_parameters(self, global_parameters: List[torch.Tensor]) -> None:
        """
        Update local model with parameters from the global model.

        Args:
            global_parameters: List of parameter tensors from the server
        """
        with torch.no_grad():
            for local_param, global_param in zip(
                self.model.parameters(), global_parameters
            ):
                local_param.copy_(global_param.to(self.device))

        logger.debug(f"{self.client_id}: Updated parameters from server")

    def train_one_epoch(self, epoch: int, verbose: bool = True) -> Dict[str, float]:
        """
        Train the model for one epoch on local data.

        Args:
            epoch: Current epoch number
            verbose: Whether to print progress

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        losses = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Ensure correct data types
            data = data.float()
            target = target.long()

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            losses.append(loss.item())
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

        # Calculate average metrics
        avg_loss = np.mean(losses)
        accuracy = correct / total

        # Get privacy spent if DP is enabled
        if self.enable_dp and self.privacy_engine:
            epsilon = self.privacy_engine.get_epsilon(self.delta)
        else:
            epsilon = float("inf")

        if verbose:
            log_msg = (
                f"{self.client_id} | Epoch {epoch} | "
                f"Loss: {avg_loss: .4f} | "
                f"Acc: {accuracy: .4f}"
            )
            if self.enable_dp:
                log_msg += f" | (ε={epsilon: .2f}, δ={self.delta: .0e})"
            print(log_msg)

        return {"loss": avg_loss, "accuracy": accuracy, "epsilon": epsilon}

    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get the current privacy budget spent.

        Returns:
            Tuple of (epsilon, delta)
        """
        if self.enable_dp and self.privacy_engine:
            epsilon = self.privacy_engine.get_epsilon(self.delta)
            return epsilon, self.delta
        else:
            return float("inf"), 0.0
