#!/usr/bin/env python3
"""
Dataset utilities for multilingual tokenizer training.
"""

from typing import Any, Dict, Iterator, List

from datasets import interleave_datasets, load_dataset

from .config import DatasetConfig


def generate_datasets_config(
    datasets: List[DatasetConfig],
    total_samples: int,
    temperature: float,
    min_samples_per_lang: int,
    max_samples_per_lang: int,
) -> List[Dict[str, Any]]:
    """Generate dataset configuration using priority-based sampling for balanced tokenizer training."""
    all_subsets = []
    for dataset in datasets:
        for subset in dataset.subsets:
            all_subsets.append(
                {
                    "path": dataset.path,
                    "name": subset.name,
                    "priority": subset.priority,
                    "split": subset.split,
                    "text_column": subset.text_column,
                    "description": dataset.description,
                }
            )

    # Calculate weights using priority^temperature
    total_weight = sum(subset["priority"] ** temperature for subset in all_subsets)

    # Generate final configuration
    datasets_config = []
    for subset in all_subsets:
        # Calculate samples based on priority weight
        weight = (subset["priority"] ** temperature) / total_weight
        samples = int(total_samples * weight)

        # Apply min/max constraints
        samples = max(min_samples_per_lang, min(max_samples_per_lang, samples))

        datasets_config.append(
            {
                "path": subset["path"],
                "name": subset["name"],
                "split": subset["split"],
                "text_column": subset["text_column"],
                "samples": samples,
                "percent": samples / total_samples,
                "priority": subset["priority"],
                "description": subset["description"],
            }
        )

    return datasets_config


def load_and_interleave_datasets(
    datasets_config: List[Dict[str, Any]], streaming_enabled: bool
) -> tuple:
    """Load datasets and create interleaved dataset for training."""
    print("Loading datasets with streaming...")

    # Store both the dataset and its successful config
    successful_loads = []
    for dataset_cfg in datasets_config:
        print(
            f"Loading {dataset_cfg['path']} ({dataset_cfg.get('name', 'default')}) - samples: {dataset_cfg['samples']:,}"
        )
        try:
            load_kwargs = {
                "path": dataset_cfg["path"],
                "split": dataset_cfg["split"],
                "streaming": streaming_enabled,
            }

            # Handle different ways to specify subsets
            if dataset_cfg.get("name"):
                # For datasets like fineweb-2 that use 'name' parameter
                if "fineweb" in dataset_cfg["path"]:
                    load_kwargs["name"] = dataset_cfg["name"]
                # For datasets like starcoderdata and the-stack that use 'data_dir' parameter
                elif (
                    "stack" in dataset_cfg["path"] or "starcoder" in dataset_cfg["path"]
                ):
                    load_kwargs["data_dir"] = dataset_cfg["name"]

            dataset = load_dataset(**load_kwargs)
            dataset = dataset.take(dataset_cfg["samples"])
            # Append both the dataset and its config
            successful_loads.append({"dataset": dataset, "config": dataset_cfg})
        except Exception as e:
            print(f"Warning: Could not load {dataset_cfg['path']}: {e}")
            continue

    # Check if any datasets were loaded successfully
    if not successful_loads:
        raise RuntimeError(
            "No datasets could be loaded successfully. Aborting training."
        )

    print(f"Successfully loaded {len(successful_loads)} datasets.")

    # Use HuggingFace's native interleaving
    print("Interleaving datasets with native HuggingFace support...")
    if len(successful_loads) > 1:
        # Build datasets and probabilities from the successful loads list
        datasets_to_interleave = [item["dataset"] for item in successful_loads]
        interleave_probabilities = [
            item["config"]["percent"] for item in successful_loads
        ]

        # It's good practice to normalize probabilities to sum to 1
        total_prob = sum(interleave_probabilities)
        if total_prob > 0:
            interleave_probabilities = [
                p / total_prob for p in interleave_probabilities
            ]

        interleaved_dataset = interleave_datasets(
            datasets_to_interleave, probabilities=interleave_probabilities, seed=42
        )
    else:
        interleaved_dataset = successful_loads[0]["dataset"]
        interleave_probabilities = [1.0]

    print(f"Interleaved dataset ready with probabilities: {interleave_probabilities}")
    return interleaved_dataset, interleave_probabilities


def create_text_iterator(interleaved_dataset) -> Iterator[str]:
    """Create an iterator that yields text from the interleaved dataset."""
    count = 0
    for sample in interleaved_dataset:
        # Try different common column names for text content
        text = (
            sample.get("text", "")
            or sample.get("content", "")
            or sample.get("code", "")
        )
        if text and len(text.strip()) > 10:  # Filter very short texts
            count += 1
            if count % 5000 == 0:
                print(f"  Processed {count:,} samples...")
            yield text


def print_dataset_distribution(datasets_config: List[Dict[str, Any]]) -> None:
    """Print a summary of the dataset distribution."""
    print(f"📊 Dataset distribution ({len(datasets_config)} subsets):")
    for dataset_cfg in sorted(
        datasets_config, key=lambda x: x["samples"], reverse=True
    )[:10]:
        print(
            f"  {dataset_cfg['name']}: {dataset_cfg['samples']:,} samples ({dataset_cfg['percent']:.1%})"
        )
    if len(datasets_config) > 10:
        print(f"  ... and {len(datasets_config) - 10} more")
    print()
