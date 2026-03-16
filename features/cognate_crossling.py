"""Group C: Cognate and cross-lingual features (en_target_word, L1_source_word, L1)."""

import re
import json
import os
import pandas as pd
import numpy as np

try:
    import Levenshtein
except ImportError:
    Levenshtein = None


def _norm_edit_distance(s1: str, s2: str) -> float:
    """Normalized Levenshtein: 0 = identical, 1 = completely different."""
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0
    if Levenshtein is None:
        # Fallback: use simple ratio
        return 1.0 - (2 * sum(a == b for a, b in zip(s1, s2)) / (len(s1) + len(s2)) if (s1 or s2) else 0)
    d = Levenshtein.distance(s1, s2)
    return d / max(len(s1), len(s2), 1)


def _char_bigram_jaccard(s1: str, s2: str) -> float:
    """Jaccard similarity of character bigrams."""
    def bigrams(s):
        s = (s or "").lower().strip()
        return set(s[i : i + 2] for i in range(len(s) - 1)) if len(s) >= 2 else set()
    a, b = bigrams(s1), bigrams(s2)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _unihan_stroke_path():
    """Path to Unihan data or None."""
    for base in ["resources", "resources/Unihan", "."]:
        for fname in ["Unihan_IRGSources.txt", "Unihan_DictionaryLikeData.txt", "Unihan_StrokeCount.txt"]:
            p = os.path.join(base, fname)
            if os.path.isfile(p):
                return p
    return None


class CognateFeatures:
    """Cognate/cross-lingual features. L1-specific: de gets compound; cn gets stroke complexity."""

    def __init__(self, l1: str):
        self.l1 = l1.lower()
        self._stroke_map = None

    def _load_strokes(self) -> dict:
        if self._stroke_map is not None:
            return self._stroke_map
        path = _unihan_stroke_path()
        out = {}
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("U+") and "\t" in line:
                            parts = line.strip().split("\t")
                            if len(parts) >= 2 and parts[1] == "kTotalStrokes":
                                try:
                                    code = chr(int(parts[0].replace("U+", ""), 16))
                                    out[code] = int(parts[2].split(".")[0])
                                except (ValueError, IndexError):
                                    pass
            except Exception:
                pass
        self._stroke_map = out
        return out

    def fit(self, train_df: pd.DataFrame) -> "CognateFeatures":
        if self.l1 == "cn":
            self._load_strokes()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        en = df["en_target_word"].fillna("").astype(str).str.strip()
        l1 = df["L1_source_word"].fillna("").astype(str).str.strip()

        out = pd.DataFrame(index=df.index)

        # Edit distance: 0=identical, 1=diff. Expect NEGATIVE r for es/de (cognate = easier = higher GLMM).
        out["edit_distance"] = [ _norm_edit_distance(a, b) for a, b in zip(en, l1) ]

        # Character bigram Jaccard
        out["char_ngram_overlap"] = [ _char_bigram_jaccard(a, b) for a, b in zip(en, l1) ]

        # Binary cognate: edit_distance < 0.4 (meaningful for es/de; cn ~1.0)
        out["is_cognate"] = (out["edit_distance"] < 0.4).astype(int)

        # Length ratio
        len_en = en.str.len().replace(0, np.nan)
        len_l1 = l1.str.len().replace(0, np.nan)
        out["length_ratio"] = (len_en / len_l1).fillna(1.0)

        if self.l1 == "de":
            out["l1_is_compound"] = (l1.str.len() > 12).astype(int)

        if self.l1 == "cn":
            stroke_map = self._load_strokes()
            if not stroke_map:
                import warnings
                warnings.warn(
                    "Unihan stroke data not loaded — cn_stroke_complexity will be all zeros. "
                    "Run: python scripts/download_resources.py"
                )
            def total_strokes(s):
                return sum(stroke_map.get(c, 0) for c in (s or ""))
            out["cn_stroke_complexity"] = l1.apply(total_strokes)

        return out
