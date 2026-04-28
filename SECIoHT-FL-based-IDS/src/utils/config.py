"""
Configuration management for federated learning.

Loads and validates configuration from YAML files.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from: {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration has required fields.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ["data", "model", "federated", "training", "privacy"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate privacy parameters
    privacy = config["privacy"]
    if privacy["enabled"]:
        if privacy["target_epsilon"] <= 0:
            raise ValueError("target_epsilon must be positive")
        if not (0 < privacy["target_delta"] < 1):
            raise ValueError("target_delta must be in (0, 1)")

    # Validate model parameters
    model = config["model"]
    if len(model["hidden_layers"]) == 0:
        raise ValueError("Model must have at least one hidden layer")

    logger.info("Configuration validation passed")
    return True


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration to: {output_path}")
