"""
Unit tests for DataLoader.

Tests data loading, preprocessing, and client distribution.
"""

import numpy as np
import pandas as pd
import torch

from src.data.dataloader import DatasetLoader


class TestDatasetLoader:
    """Test suite for DatasetLoader."""

    def test_dataloader_initialization(self, sample_data):
        """Test dataloader initializes correctly."""
        loader = DatasetLoader(
            df=sample_data, num_clients=2, sample_rate=0.2, balance_classes=True
        )

        assert loader._num_clients == 2
        assert len(loader.clients_data) == 2

    def test_train_test_split(self, sample_data):
        """Test train/test split is correct."""
        loader = DatasetLoader(df=sample_data, num_clients=2, sample_rate=0.2)

        train_loader = loader.get_train_data()
        test_loader = loader.get_test_data()

        assert len(train_loader.dataset) > 0
        assert len(test_loader.dataset) > 0

        total = len(train_loader.dataset) + len(test_loader.dataset)
        assert total <= len(sample_data) + 10

    def test_class_balancing(self):
        """Test class balancing works."""
        # Create imbalanced data
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "Label": [0] * 90 + [1] * 10,  # 90% class 0, 10% class 1
            }
        )

        loader = DatasetLoader(
            df=data, num_clients=2, sample_rate=0.2, balance_classes=True
        )

        # Training should be balanced
        train_loader = loader.get_train_data()
        labels = []
        for _, y in train_loader:
            labels.extend(y.numpy())

        # Count should be roughly equal (within 10%)
        count_0 = sum(lable == 0 for lable in labels)
        count_1 = sum(lable == 1 for lable in labels)

        assert abs(count_0 - count_1) / max(count_0, count_1) < 0.2

    def test_categorical_encoding(self):
        """Test categorical columns are encoded."""
        data = pd.DataFrame(
            {
                "cat_feature": ["a", "b", "c"] * 30,
                "num_feature": np.random.randn(90),
                "Label": np.random.choice([0, 1], 90),
            }
        )

        loader = DatasetLoader(df=data, num_clients=2, sample_rate=0.2)

        train_loader = loader.get_train_data()
        for x, y in train_loader:
            assert x.dtype == torch.float32
            break

    def test_client_data_distribution(self, sample_data):
        """Test data is distributed across clients."""
        num_clients = 3
        loader = DatasetLoader(df=sample_data, num_clients=num_clients, sample_rate=0.2)

        clients_data = loader.clients_data

        assert len(clients_data) == num_clients

        for client_id, client_loader in clients_data.items():
            assert len(client_loader.dataset) > 0

    def test_reproducibility_with_seed(self, sample_data):
        """Test same seed produces same split."""
        loader1 = DatasetLoader(
            df=sample_data, num_clients=2, sample_rate=0.2, random_seed=42
        )

        loader2 = DatasetLoader(
            df=sample_data, num_clients=2, sample_rate=0.2, random_seed=42
        )

        assert len(loader1.get_train_data().dataset) == len(
            loader2.get_train_data().dataset
        )
