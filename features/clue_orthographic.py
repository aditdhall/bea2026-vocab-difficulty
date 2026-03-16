"""Group A: Clue and orthographic features. No external data needed."""

import pandas as pd
import numpy as np

# Standard English letter frequencies (percent, then normalized)
_ENGLISH_LETTER_FREQ = {
    "e": 12.7, "t": 9.1, "a": 8.2, "o": 7.5, "i": 7.0, "n": 6.7,
    "s": 6.3, "h": 6.1, "r": 6.0, "d": 4.3, "l": 4.0, "c": 2.8,
    "u": 2.8, "m": 2.4, "w": 2.4, "f": 2.2, "g": 2.0, "y": 2.0,
    "p": 1.9, "b": 1.5, "v": 1.0, "k": 0.8, "j": 0.2, "x": 0.2,
    "q": 0.1, "z": 0.1,
}
_DEFAULT_FREQ = np.mean(list(_ENGLISH_LETTER_FREQ.values()))

_COMMON_PREFIXES = ("un-", "re-", "pre-", "dis-", "mis-", "over-", "under-", "out-")
_COMMON_SUFFIXES = (
    "-tion", "-sion", "-ing", "-ly", "-ness", "-ment", "-able", "-ible",
    "-ful", "-less", "-ous", "-ive", "-al", "-er", "-est", "-ed",
)


def _revealed_count(clue: str) -> int:
    """Count non-underscore, non-space characters in clue."""
    if pd.isna(clue) or not isinstance(clue, str):
        return 0
    return sum(1 for c in clue if c not in ("_", " ") and c.strip())


def reveal_ratio(df: pd.DataFrame) -> pd.Series:
    """Revealed chars in clue / len(en_target_word). Expect positive r with GLMM_score."""
    words = df["en_target_word"].fillna("").astype(str)
    clues = df["en_target_clue"].fillna("").astype(str)
    lens = words.str.len().replace(0, np.nan)
    revealed = clues.apply(_revealed_count)
    return (revealed / lens).fillna(0)


def hidden_chars(df: pd.DataFrame) -> pd.Series:
    """len(en_target_word) - revealed letter count."""
    words = df["en_target_word"].fillna("").astype(str)
    clues = df["en_target_clue"].fillna("").astype(str)
    revealed = clues.apply(_revealed_count)
    return words.str.len() - revealed


def word_length(df: pd.DataFrame) -> pd.Series:
    """Length of en_target_word."""
    return df["en_target_word"].fillna("").astype(str).str.len()


def syllable_count(df: pd.DataFrame) -> pd.Series:
    """Syllable count of en_target_word. More syllables = harder."""
    try:
        import syllables
    except ImportError:
        # Fallback: rough heuristic
        def _syllables(w):
            w = (w or "").lower()
            if not w:
                return 0
            count = max(1, w.count("a") + w.count("e") + w.count("i") + w.count("o") + w.count("u") + w.count("y"))
            return min(count, 8)
        return df["en_target_word"].fillna("").astype(str).apply(_syllables)
    return df["en_target_word"].fillna("").astype(str).apply(
        lambda w: syllables.estimate(w) if w else 0
    )


def has_prefix(df: pd.DataFrame) -> pd.Series:
    """1 if word starts with common prefix, else 0."""
    words = df["en_target_word"].fillna("").astype(str).str.lower()
    return words.apply(
        lambda w: 1 if any(w.startswith(p.rstrip("-")) or w == p.rstrip("-") for p in _COMMON_PREFIXES) else 0
    ).astype(int)


def has_suffix(df: pd.DataFrame) -> pd.Series:
    """1 if word ends with common suffix, else 0."""
    words = df["en_target_word"].fillna("").astype(str).str.lower()
    return words.apply(
        lambda w: 1 if any(w.endswith(s.lstrip("-")) for s in _COMMON_SUFFIXES) else 0
    ).astype(int)


def letter_frequency_score(df: pd.DataFrame) -> pd.Series:
    """Average English letter frequency of HIDDEN characters only. Rare letters (x,q,z) = harder."""
    words = df["en_target_word"].fillna("").astype(str).str.lower()
    clues = df["en_target_clue"].fillna("").astype(str).str.lower()

    def avg_hidden_freq(row):
        w, c = row.iloc[0], row.iloc[1]
        if not w or len(w) == 0:
            return _DEFAULT_FREQ
        revealed = set()
        ci = 0
        for ch in w:
            if ci < len(c) and c[ci] not in ("_", " "):
                revealed.add(ch)
            if ch not in ("_", " "):
                ci += 1
        hidden = [ch for ch in w if ch not in revealed]
        if not hidden:
            return _DEFAULT_FREQ
        freqs = [_ENGLISH_LETTER_FREQ.get(ch, _DEFAULT_FREQ) for ch in hidden]
        return np.mean(freqs)

    paired = pd.concat([words, clues], axis=1)
    return paired.apply(avg_hidden_freq, axis=1)
