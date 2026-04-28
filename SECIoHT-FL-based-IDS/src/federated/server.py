"""
Federated Learning Server implementation.

Coordinates training across multiple clients and aggregates their updates.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FederatedServer:
    """
    Federated learning server that coordinates training across clients.

    Args:
        model: The global model architecture
        device: Device to run computations on
        aggregation_method: Method for aggregating client updates (default: 'fedavg')

    Example:
        >>> model = MLP(26, [128, 64, 32], 2)
        >>> server = FederatedServer(model, torch.device('cpu'))
        >>> # ... client training ...
        >>> server.add_client_update(client_params, num_samples=100)
        >>> server.aggregate_parameters()
    """

    def __init__(
        self, model: nn.Module, device: torch.device, aggregation_method: str = "fedavg"
    ):
        self.global_model = model.to(device)
        self.device = device
        self.aggregation_method = aggregation_method
        self.round_num = 0

        # Storage for client updates
        self._client_parameters: List[List[torch.Tensor]] = []
        self._client_samples_processed: List[int] = []

        logger.info(f"Server initialized with {aggregation_method} aggregation")

    @property
    def model(self) -> nn.Module:
        """Get the global model."""
        return self.global_model

    @property
    def parameters(self) -> List[torch.Tensor]:
        """Get global model parameters as a list."""
        return [param.data.clone() for param in self.global_model.parameters()]

    def add_client_update(
        self, parameters: List[torch.Tensor], num_samples: int
    ) -> None:
        """
        Add a client's model update for aggregation.

        Args:
            parameters: Client's model parameters (as list of tensors)
            num_samples: Number of samples the client trained on
        """
        # Convert to list and detach from computation graph
        param_list = [p.detach().clone() for p in parameters]

        self._client_parameters.append(param_list)
        self._client_samples_processed.append(num_samples)

        logger.debug(f"Added update from client with {num_samples} samples")

    def _weight_scaling_factor(self, client_idx: int) -> float:
        """
        Calculate weight for a client based on number of samples.

        Args:
            client_idx: Index of the client

        Returns:
            Weight factor (proportion of total samples)
        """
        global_count = sum(self._client_samples_processed)
        local_count = self._client_samples_processed[client_idx]
        return local_count / global_count

    def _scale_model_parameters(self, client_idx: int) -> None:
        """
        Scale a client's parameters by their weight.

        Args:
            client_idx: Index of the client
        """
        scalar = self._weight_scaling_factor(client_idx)
        parameters = self._client_parameters[client_idx]

        # Scale each parameter
        scaled_parameters = [param * scalar for param in parameters]

        # Update in place
        self._client_parameters[client_idx] = scaled_parameters

    def aggregate_parameters(self) -> None:
        """
        Aggregate client updates and update the global model.

        Uses FedAvg (Federated Averaging): weighted average based on
        number of samples each client trained on.
        """
        if not self._client_parameters:
            logger.warning("No client updates to aggregate")
            return

        logger.info(f"Aggregating updates from {len(self._client_parameters)} clients")

        # Scale all client parameters by their weights
        for i in range(len(self._client_parameters)):
            self._scale_model_parameters(i)

        # Initialize aggregated parameters with zeros
        aggregated_parameters = [
            torch.zeros_like(self._client_parameters[0][j])
            for j in range(len(self._client_parameters[0]))
        ]

        # Sum all scaled client parameters
        for client_params in self._client_parameters:
            for j, param in enumerate(client_params):
                aggregated_parameters[j] += param

        # Update global model with aggregated parameters
        with torch.no_grad():
            for model_param, new_param in zip(
                self.global_model.parameters(), aggregated_parameters
            ):
                model_param.data = new_param.to(self.device)

        # Clear stored updates for next round
        self._client_parameters = []
        self._client_samples_processed = []

        self.round_num += 1
        logger.info(f"Completed round {self.round_num}")

    def save_checkpoint(self, filepath: str, metadata: Optional[Dict] = None) -> None:
        """
        Save server state to a checkpoint file.

        Args:
            filepath: Path to save the checkpoint
            metadata: Additional metadata to save
        """
        checkpoint = {
            "round": self.round_num,
            "model_state_dict": self.global_model.state_dict(),
            "aggregation_method": self.aggregation_method,
        }

        if metadata:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str) -> Dict:
        """
        Load server state from a checkpoint file.

        Args:
            filepath: Path to the checkpoint file

        Returns:
            Dictionary containing checkpoint metadata
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.global_model.load_state_dict(checkpoint["model_state_dict"])
        self.round_num = checkpoint["round"]

        logger.info(f"Loaded checkpoint from {filepath} (round {self.round_num})")

        return checkpoint.get("metadata", {})
