"""
Data loading and preprocessing for federated learning.

This module handles loading network traffic data, preprocessing,
and distributing data across federated clients.
"""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


class DatasetLoader:
    """
    Load and prepare data for federated learning.

    Handles both numeric and categorical features automatically.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        num_clients: int,
        sample_rate: float,
        balance_classes: bool = True,
        random_seed: int = 42,
    ):
        self._num_clients = num_clients
        self._sample_rate = sample_rate
        self._balance_classes = balance_classes
        self._random_seed = random_seed

        np.random.seed(random_seed)

        print("Preprocessing data...")
        # Preprocess the dataframe
        df = self._preprocess_dataframe(df)

        # Load and split data
        (
            self._X_train,
            self._y_train,
            self._X_test,
            self._y_test,
        ) = self._load_train_test_data(df)

        # Create client data distributions
        self._clients_data = self._assign_data_to_clients(self._X_train, self._y_train)

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess dataframe: handle non-numeric columns and missing values.
        """
        df = df.copy()

        print(f"   Original shape: {df.shape}")

        # Ensure Label column exists
        if "Label" not in df.columns:
            raise ValueError("Dataset must have a 'Label' column")

        # Separate features and label
        label = df["Label"]
        features = df.drop("Label", axis=1)

        # Identify column types
        numeric_cols = features.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = features.select_dtypes(
            include=["object", "category"]
        ).columns

        print(f"   Numeric columns: {len(numeric_cols)}")
        print(f"   Categorical columns: {len(categorical_cols)}")

        # Handle categorical columns
        if len(categorical_cols) > 0:
            print(f"   Encoding categorical columns: {list(categorical_cols)}")
            for col in categorical_cols:
                # Use label encoding for categorical features
                le = LabelEncoder()
                # Handle any missing values
                features[col] = features[col].fillna("missing")
                features[col] = le.fit_transform(features[col].astype(str))

        # Handle missing values in numeric columns
        if features.isnull().any().any():
            print("   Handling missing values...")
            features = features.fillna(features.median())

        # Combine back with label
        df = pd.concat([features, label], axis=1)

        print(f"   Final shape: {df.shape}")
        print(f"   Final columns: {len(df.columns) - 1} features + 1 label")

        return df

    @property
    def clients_data(self) -> dict:
        """Get dictionary of client data loaders."""
        return self._clients_data

    def get_train_data(self) -> DataLoader:
        """Get training data loader (all training data)."""
        train_dataset = TensorDataset(
            torch.tensor(self._X_train, dtype=torch.float32),
            torch.tensor(self._y_train, dtype=torch.float32),
        )
        return DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False
        )

    def get_test_data(self) -> DataLoader:
        """Get test data loader."""
        test_dataset = TensorDataset(
            torch.tensor(self._X_test, dtype=torch.float32),
            torch.tensor(self._y_test, dtype=torch.float32),
        )
        return DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False
        )

    def _load_train_test_data(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and split data into train/test sets.
        """
        # Split indices by class
        attack_indices = df.loc[df.Label == 1].index
        normal_indices = df.loc[df.Label == 0].index

        print("\n Class Distribution:")
        print(f"   Attack samples: {len(attack_indices)}")
        print(f"   Normal samples: {len(normal_indices)}")

        # Random 80/20 split
        attack_train_mask = np.random.random(len(attack_indices)) < 0.80
        normal_train_mask = np.random.random(len(normal_indices)) < 0.80

        # Create train set
        train_attack = df.iloc[attack_indices[attack_train_mask]]
        train_normal = df.iloc[normal_indices[normal_train_mask]]
        train_df = pd.concat([train_attack, train_normal])

        # Create test set
        test_attack = df.iloc[attack_indices[~attack_train_mask]]
        test_normal = df.iloc[normal_indices[~normal_train_mask]]
        test_df = pd.concat([test_attack, test_normal])

        # Shuffle
        train_df = train_df.sample(frac=1, random_state=self._random_seed).reset_index(
            drop=True
        )
        test_df = test_df.sample(frac=1, random_state=self._random_seed).reset_index(
            drop=True
        )

        # Balance training classes if requested
        if self._balance_classes:
            attack = train_df.loc[train_df.Label == 1]
            normal = train_df.loc[train_df.Label == 0][: len(attack)]
            train_df = pd.concat([attack, normal])
            train_df = train_df.sample(
                frac=1, random_state=self._random_seed
            ).reset_index(drop=True)
            print("  Balanced training set")

        # Extract features and labels
        X_train = train_df.drop("Label", axis=1).values
        y_train = train_df.Label.values
        X_test = test_df.drop("Label", axis=1).values
        y_test = test_df.Label.values

        # Scale features
        print("\n Scaling features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        print("\n Data loaded:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {X_train.shape[1]}")
        print("   Train attack/normal: {sum(y_train == 1)} / {sum(y_train == 0)}")
        print(f"   Test attack/normal: {sum(y_test == 1)} / {sum(y_test == 0)}")

        return X_train, y_train, X_test, y_test

    def _assign_data_to_clients(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """
        Distribute training data across clients.
        """
        num_samples_per_client = X_train.shape[0] // self._num_clients
        clients_all_data = {}

        print(f"\n👥 Distributing data to {self._num_clients} clients: ")

        for i in range(self._num_clients):
            start_idx = i * num_samples_per_client
            end_idx = (i + 1) * num_samples_per_client

            batched_x = X_train[start_idx:end_idx]
            batched_y = y_train[start_idx:end_idx]

            client_dataset = TensorDataset(
                torch.tensor(batched_x, dtype=torch.float32),
                torch.tensor(batched_y, dtype=torch.float32),
            )

            data_loader = DataLoader(
                client_dataset,
                batch_size=64,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )

            clients_all_data[f"client_{i}"] = data_loader
            print(f"   Client {i}: {len(batched_x)} samples")

        return clients_all_data
