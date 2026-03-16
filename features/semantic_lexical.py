"""Group D: WordNet, POS, CEFR (closed-track compliant)."""

import os
import pandas as pd
import numpy as np

try:
    from nltk.corpus import wordnet as wn
except ImportError:
    wn = None

# CEFR levels A1–C2 -> 1–6 (e.g. Oxford 5000 style)
_CEFR_ORDER = {"a1": 1, "a2": 2, "b1": 3, "b2": 4, "c1": 5, "c2": 6}


def _safe_synsets(word: str):
    if wn is None:
        return []
    w = (word or "").strip().lower()
    if not w:
        return []
    try:
        return wn.synsets(w)
    except Exception:
        return []


def wordnet_num_senses(df: pd.DataFrame) -> pd.Series:
    """Number of WordNet synsets. More polysemous = harder."""
    return df["en_target_word"].fillna("").astype(str).apply(
        lambda w: len(_safe_synsets(w))
    )


def wordnet_depth(df: pd.DataFrame) -> pd.Series:
    """Max depth in hypernym hierarchy. Deeper = rarer = harder."""
    def max_depth(word):
        syns = _safe_synsets(word)
        if not syns:
            return 0
        depths = []
        for s in syns:
            d = 0
            while s:
                d += 1
                hyp = s.hypernyms()
                s = hyp[0] if hyp else None
            depths.append(d)
        return max(depths) if depths else 0
    return df["en_target_word"].fillna("").astype(str).apply(max_depth)


def pos_encoded(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode en_target_pos into NOUN, VERB, ADJ, ADV."""
    pos = df["en_target_pos"].fillna("").astype(str).str.upper()
    out = pd.DataFrame(index=df.index)
    for p in ["NOUN", "VERB", "ADJ", "ADV"]:
        out[f"pos_{p}"] = (pos.str.contains(p, na=False) | (pos == p)).astype(int)
    return out


def cefr_level(df: pd.DataFrame, resources_dir: str = "resources/", train_cefr_median: float | None = None) -> pd.Series:
    """CEFR level 1–6. If word not in list, impute with train_cefr_median."""
    path = os.path.join(resources_dir, "cefr_wordlist.csv")
    if not os.path.isfile(path):
        # No CEFR file: return constant
        med = train_cefr_median if train_cefr_median is not None else 3.0
        return pd.Series(med, index=df.index)
    try:
        cefr_df = pd.read_csv(path)
        word_col = [c for c in cefr_df.columns if "word" in c.lower() or c == "Word"][:1]
        level_col = [c for c in cefr_df.columns if "cefr" in c.lower() or "level" in c.lower() or c in ("Level", "CEFR")][:1]
        if not word_col or not level_col:
            return pd.Series(train_cefr_median or 3.0, index=df.index)
        cefr_df = cefr_df.rename(columns={word_col[0]: "word", level_col[0]: "level"})
        cefr_df["word"] = cefr_df["word"].astype(str).str.lower().str.strip()
        cefr_df["level"] = cefr_df["level"].astype(str).str.lower().str.strip().map(
            lambda x: _CEFR_ORDER.get(x, np.nan)
        )
        cefr_df = cefr_df.dropna(subset=["level"])
        cefr_df = cefr_df.groupby("word", as_index=False)["level"].min()
        words = df["en_target_word"].fillna("").astype(str).str.lower().str.strip()
        merged = words.to_frame("word").merge(cefr_df, on="word", how="left")["level"]
        med = train_cefr_median if train_cefr_median is not None else merged.median()
        return merged.fillna(med if pd.notna(med) else 3.0)
    except Exception:
        return pd.Series(train_cefr_median or 3.0, index=df.index)
