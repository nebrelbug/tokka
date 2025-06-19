#!/usr/bin/env python3
"""
Tokenizer utilities for multilingual BPE tokenizer training.
"""

import os
from pathlib import Path
from typing import Iterator

from omegaconf import DictConfig, OmegaConf
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)

# Set tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def create_tokenizer() -> Tokenizer:
    """Create a BPE tokenizer with byte fallback and conservative normalization."""
    print("Initializing tokenizer with byte fallback...")
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # The decoder must be the corresponding ByteLevel decoder to ensure
    # the process is perfectly reversible.
    tokenizer.decoder = decoders.ByteLevel()

    # Conservative normalization for multilingual support
    print("Setting up conservative normalization...")
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Strip(),  # Remove leading/trailing whitespace only
            normalizers.NFC(),  # Canonical composition (safest Unicode normalization)
        ]
    )

    return tokenizer


def get_special_tokens() -> list:
    """Get the list of special tokens for the tokenizer."""
    # Special tokens for various tasks (256 total)
    special_tokens = [
        # Core tokens
        "<|startoftext|>",  # Start of text/document
        "<|endoftext|>",  # End of text/document
        "<|pad|>",  # Padding token
        "<|mask|>",  # Masked language modeling
        "<|sep|>",  # Separator for multi-sequence tasks
        "<|user|>",  # User turn (for chat/instruction tuning)
        "<|assistant|>",  # Assistant turn (for chat/instruction tuning)
        "<|system|>",  # System prompt (for chat/instruction tuning)
    ]

    # Add 248 reserved special tokens (for future use)
    special_tokens.extend([f"<|reserved_special_token_{i}|>" for i in range(248)])

    return special_tokens


def create_trainer(vocab_size: int = 128000) -> trainers.BpeTrainer:
    """Create a BPE trainer with the specified vocabulary size."""
    special_tokens = get_special_tokens()

    print(f"Total special tokens: {len(special_tokens)}")

    # BPE trainer with large vocab for multilingual support
    # Total vocab: 128,000 = 256 special + 256 byte fallbacks + 127,488 trained BPE
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,  # Only keep tokens that appear at least twice
    )

    print(f"Training vocab size: {vocab_size}")
    print(f"Special tokens: {len(special_tokens)}")
    print(f"Byte alphabet: {len(pre_tokenizers.ByteLevel.alphabet())}")
    print(
        f"Trained BPE tokens: {vocab_size - len(special_tokens) - len(pre_tokenizers.ByteLevel.alphabet())}"
    )

    return trainer


def train_tokenizer_on_data(
    tokenizer: Tokenizer, trainer: trainers.BpeTrainer, text_iterator: Iterator[str]
) -> None:
    """Train the tokenizer on the provided text iterator."""
    print("Training tokenizer on multilingual data...")
    print("This may take several minutes depending on dataset size...")

    print("Starting BPE training...")
    tokenizer.train_from_iterator(text_iterator, trainer)
    print("BPE training completed!")


def setup_post_processor(tokenizer: Tokenizer) -> None:
    """Set up post-processor with correct token IDs after training."""
    print("Setting up post-processor with trained token IDs...")
    vocab = tokenizer.get_vocab()
    try:
        start_token_id = vocab["<|startoftext|>"]
        end_token_id = vocab["<|endoftext|>"]

        tokenizer.post_processor = processors.TemplateProcessing(
            single="<|startoftext|> $A <|endoftext|>",
            special_tokens=[
                ("<|startoftext|>", start_token_id),
                ("<|endoftext|>", end_token_id),
            ],
        )
        print("Post-processor configured successfully")
    except KeyError as e:
        print(f"Warning: Could not set up post-processor, missing token: {e}")
        print("Tokenizer will work but won't automatically add special tokens")


def save_tokenizer(tokenizer: Tokenizer, cfg: DictConfig) -> None:
    """Save the trained tokenizer and related files."""
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save tokenizer
    tokenizer_file = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))

    # Save readable vocab file for inspection
    vocab_file = output_dir / "vocab.txt"
    vocab = tokenizer.get_vocab()
    with open(vocab_file, "w", encoding="utf-8") as f:
        # Sort by token ID
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        for token, token_id in sorted_vocab:
            f.write(f"{token_id}\t{token}\n")

    # Save the final configuration used
    config_file = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_file)

    print(f"Tokenizer saved to {tokenizer_file}")
    print(f"Vocabulary saved to {vocab_file}")
    print(f"Configuration saved to {config_file}")
    print(f"Final vocabulary size: {tokenizer.get_vocab_size()}")
    print("Training complete!")


def test_tokenizer(tokenizer: Tokenizer) -> None:
    """Test the tokenizer with sample texts."""
    print("\n--- Tokenizer Test ---")
    test_texts = [
        "Hello, how are you today?",
        "Bonjour, comment allez-vous?",
        "Hola, ¿cómo estás?",
        "Hallo, wie geht es dir?",
    ]

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"Original: {text}")
        print(f"Tokens: {encoded.tokens}")
        print(f"Decoded: {decoded}")
        print(f"Round-trip match: {text == decoded}")
        print()
