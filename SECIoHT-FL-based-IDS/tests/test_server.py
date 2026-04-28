"""
Unit tests for Federated Server.

Tests parameter aggregation, client management, and checkpointing.
"""

import torch

from src.federated.server import FederatedServer


class TestFederatedServer:
    """Test suite for Federated Server."""

    def test_server_initialization(self, simple_model, device):
        """Test server initializes correctly."""
        server = FederatedServer(simple_model, device)

        assert server.global_model is not None
        assert server.device == device
        assert server.round_num == 0
        assert server.aggregation_method == "fedavg"

    def test_add_client_update(self, simple_model, device):
        """Test adding client updates."""
        server = FederatedServer(simple_model, device)

        # Simulate client parameters
        client_params = [p.clone() for p in simple_model.parameters()]

        server.add_client_update(client_params, num_samples=100)

        assert len(server._client_parameters) == 1
        assert len(server._client_samples_processed) == 1
        assert server._client_samples_processed[0] == 100

    def test_aggregate_two_clients(self, simple_model, device):
        """Test aggregation with two clients."""
        server = FederatedServer(simple_model, device)

        # Save initial parameters
        initial_params = [p.clone() for p in server.parameters]

        # Add two client updates
        for i in range(2):
            client_params = [
                p.clone() + torch.randn_like(p) * 0.01
                for p in simple_model.parameters()
            ]
            server.add_client_update(client_params, num_samples=100)

        # Aggregate
        server.aggregate_parameters()

        updated_params = server.parameters
        assert not all(
            torch.allclose(p1, p2) for p1, p2 in zip(initial_params, updated_params)
        )

        assert len(server._client_parameters) == 0
        assert len(server._client_samples_processed) == 0

        assert server.round_num == 1

    def test_weighted_aggregation(self, simple_model, device):
        """Test that aggregation weights by sample count."""
        server = FederatedServer(simple_model, device)

        client1_params = [torch.ones_like(p) for p in simple_model.parameters()]
        server.add_client_update(client1_params, num_samples=100)

        client2_params = [torch.ones_like(p) * 3.0 for p in simple_model.parameters()]
        server.add_client_update(client2_params, num_samples=300)

        server.aggregate_parameters()

        for param in server.global_model.parameters():
            expected = torch.ones_like(param) * 2.5
            assert torch.allclose(param.data, expected, atol=1e-5)

    def test_aggregation_with_no_clients(self, simple_model, device):
        """Test aggregation with no client updates."""
        server = FederatedServer(simple_model, device)

        server.aggregate_parameters()

        assert server.round_num == 0

    def test_multiple_rounds(self, simple_model, device):
        """Test multiple rounds of aggregation."""
        server = FederatedServer(simple_model, device)

        for round_num in range(5):
            # Add client updates
            for i in range(2):
                client_params = [p.clone() for p in simple_model.parameters()]
                server.add_client_update(client_params, num_samples=100)

            server.aggregate_parameters()

        assert server.round_num == 5

    def test_checkpoint_save_load(self, simple_model, device, tmp_path):
        """Test saving and loading checkpoints."""
        server = FederatedServer(simple_model, device)

        # Train for a few rounds
        for _ in range(3):
            client_params = [p.clone() for p in simple_model.parameters()]
            server.add_client_update(client_params, num_samples=100)
            server.aggregate_parameters()

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        server.save_checkpoint(str(checkpoint_path), metadata={"test": True})

        assert checkpoint_path.exists()

        # Create new server and load checkpoint
        new_server = FederatedServer(simple_model, device)
        metadata = new_server.load_checkpoint(str(checkpoint_path))

        assert new_server.round_num == 3
        assert metadata["test"]

        # Parameters should match
        for p1, p2 in zip(
            server.global_model.parameters(), new_server.global_model.parameters()
        ):
            assert torch.allclose(p1, p2)
