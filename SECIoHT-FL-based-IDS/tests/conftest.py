"""
Pytest configuration and shared fixtures.

This file provides reusable test fixtures for all tests.
"""
import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def device():
    """Get test device (CPU for tests)."""
    return torch.device("cpu")


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    np.random.seed(42)

    # Create synthetic data similar to real dataset
    n_samples = 100
    n_features = 44

    data = {
        **{f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)},
        "Label": np.random.choice([0, 1], n_samples),
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    from src.models.mlp import MLP

    model = MLP(
        input_dim=10,
        hidden_dims=[20, 10],
        output_dim=2,
        dropout_rate=0.0,  # No dropout for deterministic tests
    )
    return model


@pytest.fixture
def complex_model():
    """Create a complex model matching production."""
    from src.models.mlp import MLP

    model = MLP(
        input_dim=44, hidden_dims=[128, 64, 32, 16], output_dim=2, dropout_rate=0.3
    )
    return model


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        "data": {
            "path": "data/raw/wustl.csv",
            "train_test_split": 0.8,
            "balance_classes": True,
            "random_seed": 42,
        },
        "model": {
            "input_dim": 44,
            "hidden_layers": [128, 64, 32, 16],
            "output_dim": 2,
            "dropout_rate": 0.3,
        },
        "federated": {
            "num_clients": 2,
            "num_rounds": 5,
            "aggregation_method": "fedavg",
        },
        "training": {"batch_size": 64, "learning_rate": 0.01},
        "privacy": {
            "enabled": True,
            "target_epsilon": 10.0,
            "target_delta": 1e-4,
            "max_grad_norm": 1.0,
            "noise_multiplier": 1.0,
        },
    }
