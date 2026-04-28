"""
Multi-Layer Perceptron (MLP) for network intrusion detection.

This module provides a flexible MLP architecture for binary classification
with configurable layers, dropout, and batch normalization.
"""

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for binary classification.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        output_dim: Number of output classes (default: 2 for binary)
        dropout_rate: Dropout probability (default: 0.5)
        activation: Activation function (default: 'relu')

    Example:
        >>> model = MLP(input_dim=26, hidden_dims=[128, 64, 32, 16], output_dim=2)
        >>> x = torch.randn(32, 26)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 2])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 2,
        dropout_rate: float = 0.5,
        activation: str = "relu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # Build the network layers
        self.layers = self._build_layers(activation)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Initialize weights
        self._initialize_weights()

    def _build_layers(self, activation: str) -> nn.ModuleList:
        """Build the hidden layers of the network."""
        layers = nn.ModuleList()

        # Get activation function
        if activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh()
        else:
            act_fn = nn.ReLU()  # Default

        # Input to first hidden layer
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Activation
            layers.append(act_fn)
            # Dropout
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

            prev_dim = hidden_dim

        return layers

    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Pass through hidden layers
        for layer in self.layers:
            x = layer(x)

        # Output layer
        x = self.output_layer(x)

        return x

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
