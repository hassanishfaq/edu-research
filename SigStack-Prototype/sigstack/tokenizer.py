"""Token-length accounting with the real RoBERTa byte-level BPE tokenizer when `transformers` can
provide it (cached after the first download), otherwise an approximate splitter (~2.5 characters
per token on serialised behaviour rows).  Used in every mode: the n = 512 truncation policy of
Table 4.14 / Section 4.4.5 is a property of the input, not of the encoder that reads it."""
from __future__ import annotations

import re
import time
from typing import Any

_APPROX = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


class TokenCounter:
    def __init__(self, name: str | None):
        self.name = name or ""
        self.kind = "approximate"
        self.tok = None
        self.error = None
        if self.name:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from transformers import AutoTokenizer
                    self.tok = AutoTokenizer.from_pretrained(self.name)
                self.kind = "roberta-bpe"
            except Exception as e:  # offline and not cached, or transformers missing
                self.error = str(e)[:200]

    def count(self, text: str, max_length: int = 512) -> dict[str, Any]:
        t = time.perf_counter()
        if self.tok is not None:
            ids = self.tok(text, add_special_tokens=False, truncation=False)["input_ids"]
            n = len(ids)
            # character offset of the truncation point (for the segment report)
            cut_chars = None
            if n > max_length:
                cut_chars = len(self.tok.decode(ids[:max_length]))
        else:
            pieces = _APPROX.findall(text)
            n = int(round(len(pieces) * 1.15))
            cut_chars = None
            if n > max_length:
                keep = int(len(pieces) * max_length / max(n, 1))
                cut_chars = len(" ".join(pieces[:keep]))
        return {"n_tokens": n, "max_length": max_length, "truncated": n > max_length, "cut_chars": cut_chars,
                "kind": self.kind, "ms": round((time.perf_counter() - t) * 1000, 2)}

    def count_many(self, texts: list[str]) -> list[int]:
        if self.tok is not None:
            out = []
            B = 512
            for i in range(0, len(texts), B):
                enc = self.tok(texts[i:i + B], add_special_tokens=False, truncation=False)["input_ids"]
                out.extend(len(x) for x in enc)
            return out
        return [int(round(len(_APPROX.findall(t)) * 1.15)) for t in texts]

    def info(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name or None, "vocab": int(self.tok.vocab_size) if self.tok is not None else None, "error": self.error}
