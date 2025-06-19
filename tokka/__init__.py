"""
Tokka - Multilingual tokenizer training toolkit.

A modern Python package for training multilingual BPE tokenizers with
comprehensive configuration support and dataset management.
"""

__version__ = "0.1.0"

# Import main functions and classes
from .config import (
    DatasetConfig,
    DatasetSubsetConfig,
    load_config,
    parse_datasets_from_config,
    print_config_summary,
    validate_config,
)
from .dataset_utils import (
    create_text_iterator,
    generate_datasets_config,
    load_and_interleave_datasets,
    print_dataset_distribution,
)
from .tokenizer_utils import (
    create_tokenizer,
    create_trainer,
    get_special_tokens,
    save_tokenizer,
    setup_post_processor,
    test_tokenizer,
    train_tokenizer_on_data,
)

__all__ = [
    # Configuration
    "DatasetConfig",
    "DatasetSubsetConfig",
    "load_config",
    "parse_datasets_from_config",
    "print_config_summary",
    "validate_config",
    # Dataset utilities
    "create_text_iterator",
    "generate_datasets_config",
    "load_and_interleave_datasets",
    "print_dataset_distribution",
    # Tokenizer utilities
    "create_tokenizer",
    "create_trainer",
    "get_special_tokens",
    "save_tokenizer",
    "setup_post_processor",
    "test_tokenizer",
    "train_tokenizer_on_data",
]
