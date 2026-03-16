"""Group B: External psycholinguistic norms (SUBTLEX, AoA, concreteness, MRC)."""

import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

RESOURCE_FILES = {
    "subtlex": "subtlex_us.csv",
    "aoa": "aoa.csv",
    "concreteness": "concreteness.csv",
    "mrc": "mrc.csv",
}


class FrequencyNormFeatures:
    """Load norm files from resources_dir; fit medians on train; transform with imputation."""

    def __init__(self, resources_dir: str = "resources/"):
        self.resources_dir = resources_dir.rstrip("/") + "/"
        self.medians_ = {}
        self._tables = {}

    def _load_table(self, name: str, word_col: str, value_col: str) -> pd.DataFrame | None:
        path = os.path.join(self.resources_dir, RESOURCE_FILES.get(name, name))
        if not os.path.isfile(path):
            logger.warning("Resource not found: %s", path)
            return None
        try:
            df = pd.read_csv(path)
            if word_col not in df.columns or value_col not in df.columns:
                logger.warning("%s: expected columns %s, %s", path, word_col, value_col)
                return None
            df = df[[word_col, value_col]].copy()
            df.columns = ["word", "value"]
            df["word"] = df["word"].astype(str).str.lower().str.strip()
            df = df.dropna(subset=["value"])
            return df.groupby("word", as_index=False)["value"].first()
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            return None

    def fit(self, train_df: pd.DataFrame) -> "FrequencyNormFeatures":
        """Load resources and compute coverage + medians for imputation."""
        word_col = "en_target_word"
        words = train_df[word_col].fillna("").astype(str).str.lower().str.strip()

        # SUBTLEX: word frequency -> log1p
        sub = self._load_table("subtlex", "Word", "FREQcount")
        if sub is not None:
            sub["value"] = np.log1p(sub["value"].astype(float))
            self._tables["subtlex_freq"] = sub
            m = sub.merge(words.to_frame("word"), on="word", how="right")["value"]
            self.medians_["subtlex_freq"] = m.median()
            cov = m.notna().sum() / len(train_df)
            if cov < 0.6:
                logger.warning("SUBTLEX coverage %.2f%% < 60%%", cov * 100)

        # AoA
        aoa = self._load_table("aoa", "Word", "AoA") or self._load_table("aoa", "word", "aoa")
        if aoa is not None:
            aoa["value"] = pd.to_numeric(aoa["value"], errors="coerce").dropna()
            self._tables["aoa_score"] = aoa.dropna()
            m = aoa.merge(words.to_frame("word"), on="word", how="right")["value"]
            self.medians_["aoa_score"] = m.median()
            if m.notna().sum() / len(train_df) < 0.6:
                logger.warning("AoA coverage < 60%%")

        # Concreteness
        conc = self._load_table("concreteness", "Word", "Conc.M") or self._load_table("concreteness", "word", "concreteness")
        if conc is not None:
            conc["value"] = pd.to_numeric(conc["value"], errors="coerce")
            self._tables["concreteness"] = conc.dropna(subset=["value"])
            m = conc.merge(words.to_frame("word"), on="word", how="right")["value"]
            self.medians_["concreteness"] = m.median()
            if m.notna().sum() / len(train_df) < 0.6:
                logger.warning("Concreteness coverage < 60%%")

        # MRC imageability
        mrc = self._load_table("mrc", "Word", "IMAGE") or self._load_table("mrc", "word", "imageability")
        if mrc is not None:
            mrc["value"] = pd.to_numeric(mrc["value"], errors="coerce")
            self._tables["imageability"] = mrc.dropna(subset=["value"])
            m = mrc.merge(words.to_frame("word"), on="word", how="right")["value"]
            self.medians_["imageability"] = m.median()
            if m.notna().sum() / len(train_df) < 0.6:
                logger.warning("MRC imageability coverage < 60%%")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge features and impute missing with stored medians."""
        out = pd.DataFrame(index=df.index)
        words = df["en_target_word"].fillna("").astype(str).str.lower().str.strip()

        for feat, table in self._tables.items():
            merged = words.to_frame("word").merge(table, on="word", how="left")["value"]
            med = self.medians_.get(feat)
            if med is not None and pd.notna(med):
                merged = merged.fillna(med)
            out[feat] = merged.values
        return out
