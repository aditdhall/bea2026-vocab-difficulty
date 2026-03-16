"""Group E: L1-specific context features. Do not compare across L1s."""

import pandas as pd
import numpy as np


def _tokenize(text: str) -> list:
    return (text or "").split()


def context_length(df: pd.DataFrame) -> pd.Series:
    """Token count of L1_context."""
    return df["L1_context"].fillna("").astype(str).apply(lambda t: len(_tokenize(t)))


def context_ttr(df: pd.DataFrame) -> pd.Series:
    """Type-token ratio of L1_context."""
    def ttr(text):
        toks = _tokenize(text)
        if not toks:
            return 0.0
        return len(set(toks)) / len(toks)
    return df["L1_context"].fillna("").astype(str).apply(ttr)


def target_position_ratio(df: pd.DataFrame) -> pd.Series:
    """Position of L1_source_word in L1_context (first occurrence) / total tokens."""
    contexts = df["L1_context"].fillna("").astype(str)
    sources = df["L1_source_word"].fillna("").astype(str)

    def pos_ratio(row):
        ctx, src = row.iloc[0], row.iloc[1]
        toks = _tokenize(ctx)
        if not toks or not src:
            return 0.5
        src_lower = src.lower()
        for i, t in enumerate(toks):
            if t.lower() == src_lower or src_lower in t.lower():
                return (i + 1) / len(toks)
        return 0.5
    paired = pd.concat([contexts, sources], axis=1)
    return paired.apply(pos_ratio, axis=1)
