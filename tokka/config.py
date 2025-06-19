#!/usr/bin/env python3
"""
Configuration handling for multilingual tokenizer training.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from omegaconf import DictConfig, OmegaConf


@dataclass
class DatasetSubsetConfig:
    """Configuration for a single dataset subset."""

    name: str
    priority: int = 1
    split: str = "train"
    text_column: str = "text"  # Default column name for text data


@dataclass
class DatasetConfig:
    """Configuration for a dataset with multiple subsets."""

    path: str
    subsets: List[DatasetSubsetConfig]
    description: str = ""


def load_config(config_path: str, cli_overrides: List[str] = None) -> DictConfig:
    """Load configuration from YAML file with optional CLI overrides."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load base configuration
    cfg = OmegaConf.load(config_path)

    # Apply CLI overrides
    if cli_overrides:
        override_cfg = OmegaConf.from_dotlist(cli_overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def parse_datasets_from_config(cfg: DictConfig) -> List[DatasetConfig]:
    """Parse dataset configurations from OmegaConf config."""
    datasets = []
    for dataset_cfg in cfg.datasets:
        subsets = []
        for subset_cfg in dataset_cfg.subsets:
            subsets.append(
                DatasetSubsetConfig(
                    name=subset_cfg.name,
                    priority=subset_cfg.priority,
                    split=subset_cfg.get("split", "train"),
                    text_column=subset_cfg.get("text_column", "text"),
                )
            )

        datasets.append(
            DatasetConfig(
                path=dataset_cfg.path,
                subsets=subsets,
                description=dataset_cfg.get("description", ""),
            )
        )

    return datasets


def validate_config(cfg: DictConfig) -> None:
    """Validate that the configuration has all required fields."""
    if not hasattr(cfg, "training"):
        raise ValueError("Configuration must contain 'training' section")

    if not hasattr(cfg, "datasets"):
        raise ValueError("Configuration must contain 'datasets' section")

    # Validate training section
    required_training_fields = [
        "total_samples",
        "output_dir",
        "temperature",
        "min_samples_per_lang",
        "max_samples_per_lang",
    ]

    for field in required_training_fields:
        if not hasattr(cfg.training, field):
            raise ValueError(f"Training configuration missing required field: {field}")

    # Validate datasets
    if len(cfg.datasets) == 0:
        raise ValueError("At least one dataset must be configured")


def print_config_summary(cfg: DictConfig) -> None:
    """Print a summary of the loaded configuration."""
    print("🔧 Configuration loaded:")
    print(f"  Total samples: {cfg.training.total_samples:,}")
    print(f"  Output directory: {cfg.training.output_dir}")
    print(f"  Temperature: {cfg.training.temperature}")
    print(f"  Streaming: {cfg.training.streaming_enabled}")
    print(f"  Min samples per lang: {cfg.training.min_samples_per_lang:,}")
    print(f"  Max samples per lang: {cfg.training.max_samples_per_lang:,}")
    print()
