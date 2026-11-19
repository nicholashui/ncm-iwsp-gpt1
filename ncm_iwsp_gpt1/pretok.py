"""Frozen GPT-2 BPE pretok, the same encoding NCM-GPT1 uses.

Token ids come from the GPT-2 byte-level BPE (50257 ids, end-of-text at
50256). ``<|pad|>`` is a new id, 50257, and the embedding table is padded to
50304, a multiple of 64, matching ``config/gpt1.yaml`` in NCM-GPT1. This is
not a trained MiniMind vocabulary and it is not a download of model weights.
"""

from __future__ import annotations

import tiktoken


class GPT2BPE:
    eos_token = "<|endoftext|>"
    pad_token = "<|pad|>"
    kind = "gpt2"

    def __init__(self, pad_vocab_to: int = 50304):
        self.enc = tiktoken.get_encoding("gpt2")
        self.base_vocab = int(self.enc.n_vocab)
        self.eos_token_id = int(self.enc.eot_token)
        if self.base_vocab != 50257 or self.eos_token_id != 50256:
            raise RuntimeError(
                f"unexpected GPT-2 encoding: vocab={self.base_vocab} eos={self.eos_token_id}"
            )
        # One past the last GPT-2 id. NCM-GPT1 adds <|pad|> the same way.
        self.pad_token_id = self.base_vocab
        self.used_vocab_size = self.base_vocab + 1
        pad = int(pad_vocab_to)
        if pad < self.used_vocab_size:
            raise ValueError(
                f"pad_vocab_to={pad} is smaller than the used vocab {self.used_vocab_size}"
            )
        if pad % 64 != 0:
            raise ValueError(f"pad_vocab_to={pad} must be a multiple of 64, as in NCM-GPT1")
        self.vocab_size = pad

    @classmethod
    def from_config(cls, cfg: dict) -> "GPT2BPE":
        block = cfg["pretok"] if "pretok" in cfg else cfg
        kind = str(block.get("kind", "gpt2"))
        if kind != "gpt2":
            raise ValueError(f"NCM-iWSP-GPT1 pretok kind must be 'gpt2', got {kind!r}")
        return cls(pad_vocab_to=int(block.get("pad_vocab_to", 50304)))

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        ids = self.enc.encode(
            text,
            allowed_special={self.eos_token},
            disallowed_special=(),
        )
        return [int(i) for i in ids]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        skip = {self.pad_token_id}
        if skip_special_tokens:
            skip.add(self.eos_token_id)
        kept = []
        for i in ids:
            i = int(i)
            if i in skip or i < 0 or i >= self.base_vocab:
                continue
            kept.append(i)
        return self.enc.decode(kept)

    def token_stats(self, texts) -> dict:
        chars = 0
        toks = 0
        for text in texts:
            chars += len(text)
            toks += len(self.encode(text))
        return {
            "kind": self.kind,
            "vocab_size": self.vocab_size,
            "used_vocab_size": self.used_vocab_size,
            "base_vocab": self.base_vocab,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "chars": chars,
            "tokens": toks,
            "chars_per_token": (chars / toks) if toks else None,
            "note": "GPT-2 BPE. <|pad|> is id 50257 and is not part of the original 50257-id table.",
        }
