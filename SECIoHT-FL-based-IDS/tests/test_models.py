"""
Unit tests for MLP model.

Tests model initialization, forward pass, predictions, and edge cases.
"""

import pytest
import torch

from src.models.mlp import MLP


class TestMLPModel:
    """Test suite for MLP model."""

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = MLP(input_dim=26, hidden_dims=[128, 64, 32], output_dim=2)

        assert model.input_dim == 26
        assert model.hidden_dims == [128, 64, 32]
        assert model.output_dim == 2
        assert len(list(model.parameters())) > 0

    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        model = MLP(input_dim=26, hidden_dims=[64, 32], output_dim=2)
        batch_size = 16
        x = torch.randn(batch_size, 26)

        output = model(x)

        assert output.shape == (batch_size, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_pass_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        model = MLP(input_dim=10, hidden_dims=[20], output_dim=2)

        for batch_size in [1, 8, 32, 128]:
            x = torch.randn(batch_size, 10)
            output = model(x)
            assert output.shape == (batch_size, 2)

    def test_num_parameters(self):
        """Test parameter counting is correct."""
        model = MLP(input_dim=10, hidden_dims=[20], output_dim=2)

        expected = (10 * 20 + 20) + (20 * 2 + 2)
        assert model.num_parameters() == expected

    def test_dropout_in_training_mode(self, simple_model):
        """Test dropout is active in training mode."""
        model = MLP(input_dim=10, hidden_dims=[20], output_dim=2, dropout_rate=0.5)
        x = torch.randn(10, 10)

        model.train()
        out1 = model(x)
        out2 = model(x)

        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_dropout_in_eval_mode(self):
        """Test dropout is disabled in eval mode."""
        model = MLP(input_dim=10, hidden_dims=[20], output_dim=2, dropout_rate=0.5)
        x = torch.randn(10, 10)

        model.eval()
        out1 = model(x)
        out2 = model(x)

        assert torch.allclose(out1, out2)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = MLP(input_dim=10, hidden_dims=[20], output_dim=2)
        x = torch.randn(10, 10, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_different_activations(self):
        """Test model works with different activation functions."""
        activations = ["relu", "tanh", "leaky_relu"]

        for activation in activations:
            model = MLP(
                input_dim=10, hidden_dims=[20], output_dim=2, activation=activation
            )
            x = torch.randn(5, 10)
            output = model(x)
            assert output.shape == (5, 2)

    def test_single_hidden_layer(self):
        """Test model with single hidden layer."""
        model = MLP(input_dim=10, hidden_dims=[20], output_dim=2)
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 2)

    def test_deep_network(self):
        """Test model with many hidden layers."""
        model = MLP(input_dim=10, hidden_dims=[64, 32, 16, 8], output_dim=2)
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 2)

    def test_model_on_different_devices(self, simple_model):
        """Test model can be moved to different devices."""
        model = simple_model

        # Test CPU
        model_cpu = model.to("cpu")
        x = torch.randn(5, 10)
        output = model_cpu(x)
        assert output.device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to("cuda")
            x_cuda = x.to("cuda")
            output = model_cuda(x_cuda)
            assert output.device.type == "cuda"

    def test_zero_input(self):
        """Test model handles zero input."""
        model = MLP(input_dim=10, hidden_dims=[20], output_dim=2)
        x = torch.zeros(5, 10)
        output = model(x)

        assert output.shape == (5, 2)
        assert not torch.isnan(output).any()

    def test_large_input_values(self):
        """Test model handles large input values."""
        model = MLP(input_dim=10, hidden_dims=[20], output_dim=2)
        x = torch.randn(5, 10) * 1000
        output = model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_deterministic_with_seed(self):
        """Test model gives same results with same seed."""
        torch.manual_seed(42)
        model1 = MLP(input_dim=10, hidden_dims=[20], output_dim=2, dropout_rate=0.0)

        torch.manual_seed(42)
        model2 = MLP(input_dim=10, hidden_dims=[20], output_dim=2, dropout_rate=0.0)

        x = torch.randn(5, 10)

        model1.eval()
        model2.eval()

        output1 = model1(x)
        output2 = model2(x)

        assert torch.allclose(output1, output2, atol=1e-6)

    def test_invalid_input_dimensions(self):
        """Test model raises error with wrong input dimensions."""
        model = MLP(input_dim=10, hidden_dims=[20], output_dim=2)
        x = torch.randn(5, 15)

        with pytest.raises(RuntimeError):
            model(x)

    def test_model_repr(self):
        """Test model string representation."""
        model = MLP(input_dim=10, hidden_dims=[20], output_dim=2)
        repr_str = repr(model)

        assert "MLP" in repr_str
        assert "10" in repr_str
        assert "2" in repr_str
