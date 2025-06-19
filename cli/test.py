#!/usr/bin/env python3
"""
Test the trained BPE tokenizer by loading it from saved files.
"""

from pathlib import Path

from tokenizers import Tokenizer


def main(tokenizer_path: str = "./train-500k/tokenizer.json"):
    """Load and test a saved tokenizer with proper security handling."""

    print(f"Loading tokenizer from {tokenizer_path}...")

    # Check if tokenizer exists
    if not Path(tokenizer_path).exists():
        print(f"Tokenizer not found at {tokenizer_path}")
        print("Please run train_bpe.py first to train the tokenizer.")
        return None

    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # IMPORTANT: Set this to True to treat special tokens in user input as regular text
    tokenizer.encode_special_tokens = True

    print(f"Loaded tokenizer with vocab size: {tokenizer.get_vocab_size()}")

    # Test cases covering various Unicode scenarios
    print("\nTesting tokenizer...")
    test_cases = [
        "Hello world! This is a test with émojis 🚀",
        "Unicode test: café, naïve, résumé",
        "Emojis: 👋 🌍 🔥 💯 🎉",
        "Mixed: Hello 世界! Bonjour le monde 🌟",
        "Special chars: <|startoftext|> content <|endoftext|>",
        "Edge case: 𝕳𝖊𝖑𝖑𝖔 (mathematical symbols)",
    ]

    for test_text in test_cases:
        # Use add_special_tokens=False for raw user text to prevent injection
        encoding = tokenizer.encode(test_text, add_special_tokens=False)
        decoded = tokenizer.decode(encoding.ids)
        match = "✓" if test_text == decoded else "✗"
        print(f"{match} {test_text} -> {len(encoding.tokens)} tokens")

    # Comprehensive multilingual tests with larger text samples
    print("\n=== COMPREHENSIVE MULTILINGUAL TESTS ===")

    multilingual_tests = [
        {
            "name": "Spanish",
            "text": "¡Hola mundo! Este es un test en español. Estamos probando que el tokenizador procese correctamente los caracteres en español, incluyendo acentos y la letra ñ. España y América Latina tienen una rica diversidad cultural.",
        },
        {
            "name": "Hindi",
            "text": "नमस्ते दुनिया! यह हिंदी भाषा का परीक्षण है। हम देख रहे हैं कि टोकनाइज़र देवनागरी लिपि को सही तरीके से संसाधित करता है या नहीं। भारत एक विविधतापूर्ण देश है।",
        },
        {
            "name": "Russian",
            "text": "Привет, как дела? Это тест русского языка. Мы проверяем, что токенизатор правильно обрабатывает кириллические символы. Россия — большая страна с богатой историей и культурой. Москва является столицей России.",
        },
        {
            "name": "Chinese (Simplified)",
            "text": "你好，世界！这是一个中文测试。我们正在测试分词器是否能正确处理中文字符。中国有着悠久的历史和丰富的文化。北京是中国的首都，有很多著名的景点。",
        },
        {
            "name": "Japanese",
            "text": "こんにちは、世界！これは日本語のテストです。トークナイザーが日本語の文字を正しく処理できるかどうかをテストしています。日本は美しい国で、豊かな文化と歴史があります。東京は日本の首都です。",
        },
        {
            "name": "German",
            "text": "Hallo Welt! Dies ist ein Test der deutschen Sprache. Wir testen, ob der Tokenizer deutsche Zeichen korrekt verarbeitet. Deutschland ist ein schönes Land mit reicher Geschichte und Kultur. Berlin ist die Hauptstadt von Deutschland.",
        },
        {
            "name": "Mixed Languages",
            "text": "Hello world! ¡Hola mundo! 你好世界! नमस्ते दुनिया! Привет мир! こんにちは世界! Hallo Welt! This sentence mixes multiple languages including Spanish, Chinese, Hindi, Russian, Japanese, and German.",
        },
    ]

    for test in multilingual_tests:
        print(f"\n--- {test['name']} Test ---")
        text = test["text"]
        encoding = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoding.ids)

        print(f"Original text ({len(text)} chars): {text}")
        print(f"Tokenized to {len(encoding.tokens)} tokens")
        print(f"First 10 tokens: {encoding.tokens[:10]}")
        print(f"Token IDs: {encoding.ids[:10]}...")
        print(f"Decoded text: {decoded}")
        print(f"Round-trip perfect: {'✓' if text == decoded else '✗'}")

        if text != decoded:
            print("⚠️  MISMATCH DETECTED!")
            print(f"Original bytes: {text.encode('utf-8')[:50]}...")
            print(f"Decoded bytes:  {decoded.encode('utf-8')[:50]}...")

        # Calculate compression ratio
        compression_ratio = len(text.encode("utf-8")) / len(encoding.tokens)
        print(
            f"Compression: {len(text.encode('utf-8'))} bytes -> {len(encoding.tokens)} tokens (ratio: {compression_ratio:.2f})"
        )

    print("\n=== FILE-BASED MULTILINGUAL ANALYSIS ===")

    # Test files with substantial content in different languages
    test_files = [
        {"name": "English", "file": "sample_texts/english.txt"},
        {"name": "Spanish", "file": "sample_texts/spanish.txt"},
        {"name": "Chinese", "file": "sample_texts/chinese.txt"},
        {"name": "Russian", "file": "sample_texts/russian.txt"},
        {"name": "Japanese", "file": "sample_texts/japanese.txt"},
        {"name": "Hindi", "file": "sample_texts/hindi.txt"},
        {"name": "Arabic", "file": "sample_texts/arabic.txt"},
        {"name": "German", "file": "sample_texts/german.txt"},
    ]

    for test_file in test_files:
        file_path = Path(test_file["file"])
        if not file_path.exists():
            print(f"⚠️  {test_file['name']} file not found: {file_path}")
            continue

        print(f"\n--- {test_file['name']} File Analysis ---")

        # Read file content
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            continue

        # Tokenize content
        encoding = tokenizer.encode(content, add_special_tokens=False)
        decoded = tokenizer.decode(encoding.ids)

        # Calculate metrics
        char_count = len(content)
        byte_count = len(content.encode("utf-8"))
        token_count = len(encoding.tokens)
        word_count = len(content.split())

        # Compression ratios
        chars_per_token = char_count / token_count if token_count > 0 else 0
        bytes_per_token = byte_count / token_count if token_count > 0 else 0
        tokens_per_word = token_count / word_count if word_count > 0 else 0

        print(
            f"File size: {char_count:,} chars, {byte_count:,} bytes, {word_count:,} words"
        )
        print(f"Tokenized to: {token_count:,} tokens")
        print("Compression ratios:")
        print(f"  • {chars_per_token:.2f} chars/token")
        print(f"  • {bytes_per_token:.2f} bytes/token")
        print(f"  • {tokens_per_word:.2f} tokens/word")

        # Round-trip accuracy test
        accuracy = "✓ Perfect" if content == decoded else "✗ MISMATCH"
        print(f"Round-trip accuracy: {accuracy}")

        if content != decoded:
            print("⚠️  Encoding/decoding mismatch detected!")
            # Show first difference
            for i, (orig, dec) in enumerate(zip(content, decoded)):
                if orig != dec:
                    print(f"   First difference at position {i}: '{orig}' vs '{dec}'")
                    break
            print(f"   Original length: {len(content)}, Decoded length: {len(decoded)}")

        # Show sample tokens for analysis
        sample_tokens = (
            encoding.tokens[:15] if len(encoding.tokens) >= 15 else encoding.tokens
        )
        print(f"Sample tokens: {sample_tokens}")

    print("\n=== TOKENIZER STATISTICS ===")
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    print("File-based multilingual analysis completed!")

    # Test special token IDs
    start_id = tokenizer.token_to_id("<|startoftext|>")
    end_id = tokenizer.token_to_id("<|endoftext|>")
    print(f"\nSpecial tokens: <|startoftext|>={start_id}, <|endoftext|>={end_id}")

    # Show how to properly add special tokens programmatically
    print("\nProgrammatic special token usage:")
    content = "This is my content"
    content_tokens = tokenizer.encode(content, add_special_tokens=False)
    # Manually add special tokens by ID
    full_sequence = [start_id] + content_tokens.ids + [end_id]
    full_text = tokenizer.decode(full_sequence)
    print(f"Content: {content}")
    print(f"With special tokens: {full_text}")

    print("\nAll tests completed!")
    return tokenizer


if __name__ == "__main__":
    main()
