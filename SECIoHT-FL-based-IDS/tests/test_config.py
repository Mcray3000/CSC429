"""
Unit tests for configuration management.

Tests config loading, validation, and error handling.
"""

import pytest
import yaml

from src.utils.config import load_config, save_config, validate_config


class TestConfiguration:
    """Test suite for configuration management."""

    def test_load_valid_config(self, sample_config, tmp_path):
        """Test loading valid configuration."""
        # Save config to temp file
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Load it back
        loaded_config = load_config(str(config_file))

        assert loaded_config == sample_config

    def test_load_nonexistent_config(self):
        """Test loading non-existent config raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_validate_valid_config(self, sample_config):
        """Test validation passes for valid config."""
        assert validate_config(sample_config)

    def test_validate_missing_section(self, sample_config):
        """Test validation fails with missing required section."""
        del sample_config["privacy"]

        with pytest.raises(ValueError, match="Missing required config section"):
            validate_config(sample_config)

    def test_validate_invalid_epsilon(self, sample_config):
        """Test validation fails with invalid epsilon."""
        sample_config["privacy"]["target_epsilon"] = -1.0

        with pytest.raises(ValueError, match="target_epsilon must be positive"):
            validate_config(sample_config)

    def test_validate_invalid_delta(self, sample_config):
        """Test validation fails with invalid delta."""
        sample_config["privacy"]["target_delta"] = 1.5

        with pytest.raises(ValueError, match="target_delta must be in"):
            validate_config(sample_config)

    def test_save_config(self, sample_config, tmp_path):
        """Test saving configuration."""
        output_file = tmp_path / "saved_config.yaml"

        save_config(sample_config, str(output_file))

        assert output_file.exists()

        # Load and verify
        with open(output_file, "r") as f:
            loaded = yaml.safe_load(f)

        assert loaded == sample_config
