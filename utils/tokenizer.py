"""
Tokenizer utilities.

Uses tiktoken (GPT-4 tokenizer) or can be swapped for Llama tokenizer.
"""

import tiktoken
from typing import List


class SimpleTokenizer:
    """
    Wrapper around tiktoken for compatibility with our model.

    For production, replace with actual Llama 3.3 tokenizer.
    """
    def __init__(self, vocab_size: int = 32000):
        # Use cl100k_base encoding (GPT-4 style)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = vocab_size

        # Special tokens
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.encoding.encode(text)

        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]

        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        if skip_special_tokens:
            # Filter out special tokens
            tokens = [t for t in tokens if t not in [
                self.pad_token_id, self.eos_token_id, self.bos_token_id
            ]]

        try:
            return self.encoding.decode(tokens)
        except:
            # Fallback for invalid tokens
            return "<invalid>"

    def __len__(self):
        return self.vocab_size


def create_tokenizer(vocab_size: int = 32000):
    """
    Create tokenizer for the model.

    Args:
        vocab_size: Vocabulary size (should match model config)

    Returns:
        Tokenizer instance
    """
    return SimpleTokenizer(vocab_size)
