"""Group B: External psycholinguistic norms (SUBTLEX-US, AoA, concreteness)."""

import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Actual filenames and column specs
SUBTLEX_FILE = "SUBTLEXus74286wordstextversion.txt"  # tab-separated: Word, FREQcount
AOA_FILE = "aoa_kuperman.xlsx"                        # Excel: Word, AoA_Kup
CONC_FILE = "concreteness.txt"                        # tab-separated: Word, Conc.M
# MRC imageability skipped for now


class FrequencyNormFeatures:
    """Load norm files from resources_dir; fit medians on train; transform with imputation."""

    def __init__(self, resources_dir: str = "resources/"):
        self.resources_dir = os.path.normpath(resources_dir.rstrip("/")) + os.sep
        self.medians_ = {}
        self._tables = {}

    def _load_subtlex(self) -> pd.DataFrame | None:
        """SUBTLEX-US: tab-separated, Word, FREQcount. Feature = np.log1p(FREQcount)."""
        path = os.path.join(self.resources_dir, SUBTLEX_FILE)
        if not os.path.isfile(path):
            logger.warning("Resource not found: %s", path)
            return None
        try:
            df = pd.read_csv(path, sep="\t")
            if "Word" not in df.columns or "FREQcount" not in df.columns:
                logger.warning("%s: expected columns Word, FREQcount", path)
                return None
            df = df[["Word", "FREQcount"]].copy()
            df["Word"] = df["Word"].astype(str).str.lower().str.strip()
            df["value"] = np.log1p(pd.to_numeric(df["FREQcount"], errors="coerce"))
            df = df.dropna(subset=["value"]).rename(columns={"Word": "word"})[["word", "value"]]
            return df.groupby("word", as_index=False)["value"].first()
        except Exception as e:
            logger.warning("Failed to load SUBTLEX %s: %s", path, e)
            return None

    def _load_aoa(self) -> pd.DataFrame | None:
        """AoA Kuperman: Excel, Word, AoA_Kup. Feature = AoA_Kup (lower = easier)."""
        path = os.path.join(self.resources_dir, AOA_FILE)
        if not os.path.isfile(path):
            logger.warning("Resource not found: %s", path)
            return None
        try:
            df = pd.read_excel(path)
            if "Word" not in df.columns or "AoA_Kup" not in df.columns:
                logger.warning("%s: expected columns Word, AoA_Kup", path)
                return None
            df = df[["Word", "AoA_Kup"]].copy()
            df["Word"] = df["Word"].astype(str).str.lower().str.strip()
            df["value"] = pd.to_numeric(df["AoA_Kup"], errors="coerce")
            df = df.dropna(subset=["value"]).rename(columns={"Word": "word"})[["word", "value"]]
            return df.groupby("word", as_index=False)["value"].first()
        except Exception as e:
            logger.warning("Failed to load AoA %s: %s", path, e)
            return None

    def _load_concreteness(self) -> pd.DataFrame | None:
        """Concreteness: tab-separated, Word, Conc.M. Feature = Conc.M (higher = easier)."""
        path = os.path.join(self.resources_dir, CONC_FILE)
        if not os.path.isfile(path):
            logger.warning("Resource not found: %s", path)
            return None
        try:
            df = pd.read_csv(path, sep="\t")
            if "Word" not in df.columns or "Conc.M" not in df.columns:
                logger.warning("%s: expected columns Word, Conc.M", path)
                return None
            df = df[["Word", "Conc.M"]].copy()
            df["Word"] = df["Word"].astype(str).str.lower().str.strip()
            df["value"] = pd.to_numeric(df["Conc.M"], errors="coerce")
            df = df.dropna(subset=["value"]).rename(columns={"Word": "word"})[["word", "value"]]
            return df.groupby("word", as_index=False)["value"].first()
        except Exception as e:
            logger.warning("Failed to load concreteness %s: %s", path, e)
            return None

    def fit(self, train_df: pd.DataFrame) -> "FrequencyNormFeatures":
        """Load resources; compute coverage and store training medians for imputation."""
        words = train_df["en_target_word"].fillna("").astype(str).str.lower().str.strip().to_frame("word")

        # SUBTLEX-US: log1p(FREQcount)
        sub = self._load_subtlex()
        if sub is not None:
            self._tables["subtlex_freq"] = sub
            m = words.merge(sub, on="word", how="left")["value"]
            self.medians_["subtlex_freq"] = m.median()
            if m.notna().sum() / len(train_df) < 0.6:
                logger.warning("SUBTLEX coverage %.2f%% < 60%%", m.notna().sum() / len(train_df) * 100)

        # AoA
        aoa = self._load_aoa()
        if aoa is not None:
            self._tables["aoa_score"] = aoa
            m = words.merge(aoa, on="word", how="left")["value"]
            self.medians_["aoa_score"] = m.median()
            if m.notna().sum() / len(train_df) < 0.6:
                logger.warning("AoA coverage %.2f%% < 60%%", m.notna().sum() / len(train_df) * 100)

        # Concreteness
        conc = self._load_concreteness()
        if conc is not None:
            self._tables["concreteness"] = conc
            m = words.merge(conc, on="word", how="left")["value"]
            self.medians_["concreteness"] = m.median()
            if m.notna().sum() / len(train_df) < 0.6:
                logger.warning("Concreteness coverage %.2f%% < 60%%", m.notna().sum() / len(train_df) * 100)

        # MRC imageability skipped

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge on en_target_word (lowercased); impute missing with training median."""
        out = pd.DataFrame(index=df.index)
        words = df["en_target_word"].fillna("").astype(str).str.lower().str.strip().to_frame("word")

        for feat, table in self._tables.items():
            merged = words.merge(table, on="word", how="left")["value"]
            med = self.medians_.get(feat)
            if med is not None and pd.notna(med):
                merged = merged.fillna(med)
            out[feat] = merged.values
        return out
