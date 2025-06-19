#!/usr/bin/env python3
"""
Multilingual tokenizer training with OmegaConf configuration support.

Usage:
    python train_bpe_new.py --config configs/simple_500k.yaml
    python train_bpe_new.py --config configs/simple_50m.yaml training.total_samples=25000000
    python train_bpe_new.py --config configs/complex_500m.yaml training.output_dir=./custom-output
"""

import argparse

from tokka.config import (
    load_config,
    parse_datasets_from_config,
    print_config_summary,
    validate_config,
)
from tokka.dataset_utils import (
    create_text_iterator,
    generate_datasets_config,
    load_and_interleave_datasets,
    print_dataset_distribution,
)
from tokka.tokenizer_utils import (
    create_tokenizer,
    create_trainer,
    save_tokenizer,
    setup_post_processor,
    test_tokenizer,
    train_tokenizer_on_data,
)


def train_tokenizer_pipeline(cfg):
    """Main tokenizer training pipeline."""

    # Print configuration summary
    print_config_summary(cfg)

    # Parse datasets from config
    datasets = parse_datasets_from_config(cfg)

    # Generate dataset configuration with priority-based sampling
    datasets_config = generate_datasets_config(
        datasets=datasets,
        total_samples=cfg.total_samples,
        temperature=cfg.temperature,
        min_samples_per_lang=cfg.min_samples_per_lang,
        max_samples_per_lang=cfg.max_samples_per_lang,
    )

    # Print dataset distribution
    print_dataset_distribution(datasets_config)

    # Load and interleave datasets
    interleaved_dataset, probabilities = load_and_interleave_datasets(
        datasets_config, cfg.streaming_enabled
    )

    # Create tokenizer and trainer
    tokenizer = create_tokenizer()
    trainer = create_trainer(vocab_size=cfg.vocab_size)

    # Create text iterator and train tokenizer
    text_iterator = create_text_iterator(interleaved_dataset)
    train_tokenizer_on_data(tokenizer, trainer, text_iterator)

    # Set up post-processor
    setup_post_processor(tokenizer)

    # Save tokenizer and related files
    save_tokenizer(tokenizer, cfg)

    # Test the tokenizer
    test_tokenizer(tokenizer)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Train multilingual BPE tokenizer with OmegaConf configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_bpe_new.py --config configs/simple_500k.yaml
  python train_bpe_new.py --config configs/simple_50m.yaml training.total_samples=25000000
  python train_bpe_new.py --config configs/complex_500m.yaml training.output_dir=./custom-output

CLI overrides use dot notation:
  training.total_samples=1000000
  training.temperature=0.4
  training.output_dir=./my-tokenizer
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "overrides",
        nargs="*",
        help="Configuration overrides in key=value format (e.g., training.total_samples=1000000)",
    )

    args = parser.parse_args()

    try:
        # Load and validate configuration
        cfg = load_config(args.config, args.overrides)
        validate_config(cfg)

        print(f"🚀 Starting tokenizer training with config: {args.config}")
        if args.overrides:
            print(f"📝 CLI overrides: {args.overrides}")
        print()

        # Run the training pipeline
        train_tokenizer_pipeline(cfg)

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
