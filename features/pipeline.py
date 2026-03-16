"""Feature pipeline: fit on train only, transform with stored statistics."""

import json
import os
import pandas as pd

from features import clue_orthographic
from features import context_features
from features import cognate_crossling
from features import frequency_norms
from features import semantic_lexical


class FeaturePipeline:
    """Single pipeline per L1. Fit on train; transform uses stored stats. Optional frozen feature list."""

    def __init__(self, l1: str, frozen_features_path: str | None = None, resources_dir: str = "resources/"):
        self.l1 = l1.lower()
        self.resources_dir = resources_dir
        self._frozen_names: list[str] | None = None
        if frozen_features_path and os.path.isfile(frozen_features_path):
            with open(frozen_features_path, "r") as f:
                data = json.load(f)
            shared = data.get("shared", [])
            es_de = data.get("es_de_specific", [])
            cn_spec = data.get("cn_specific", [])
            self._frozen_names = list(shared)
            if self.l1 in ("es", "de"):
                self._frozen_names += [x for x in es_de if x not in self._frozen_names]
            if self.l1 == "cn":
                self._frozen_names += [x for x in cn_spec if x not in self._frozen_names]

        self._freq_norms: frequency_norms.FrequencyNormFeatures | None = None
        self._cognate: cognate_crossling.CognateFeatures | None = None
        self._cefr_median: float | None = None

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        """Compute and store statistics from training data only."""
        self._freq_norms = frequency_norms.FrequencyNormFeatures(resources_dir=self.resources_dir)
        self._freq_norms.fit(train_df)

        self._cognate = cognate_crossling.CognateFeatures(self.l1)
        self._cognate.fit(train_df)

        # CEFR median from train for imputation
        cefr_ser = semantic_lexical.cefr_level(train_df, resources_dir=self.resources_dir, train_cefr_median=None)
        self._cefr_median = cefr_ser.median()
        if pd.isna(self._cefr_median):
            self._cefr_median = 3.0
        return self

    def _compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build full feature DataFrame (no filtering, no GLMM yet)."""
        blocks = []

        # Group A: clue/orthographic
        blocks.append(clue_orthographic.reveal_ratio(df).rename("reveal_ratio"))
        blocks.append(clue_orthographic.hidden_chars(df).rename("hidden_chars"))
        blocks.append(clue_orthographic.word_length(df).rename("word_length"))
        blocks.append(clue_orthographic.syllable_count(df).rename("syllable_count"))
        blocks.append(clue_orthographic.has_prefix(df).rename("has_prefix"))
        blocks.append(clue_orthographic.has_suffix(df).rename("has_suffix"))
        blocks.append(clue_orthographic.letter_frequency_score(df).rename("letter_frequency_score"))

        # Group B: frequency norms
        if self._freq_norms is not None:
            blocks.append(self._freq_norms.transform(df))

        # Group C: cognate
        if self._cognate is not None:
            blocks.append(self._cognate.transform(df))

        # Group D: semantic/lexical
        blocks.append(semantic_lexical.wordnet_num_senses(df).rename("wordnet_num_senses"))
        blocks.append(semantic_lexical.wordnet_depth(df).rename("wordnet_depth"))
        blocks.append(semantic_lexical.pos_encoded(df))
        blocks.append(semantic_lexical.cefr_level(df, self.resources_dir, self._cefr_median).rename("cefr_level"))

        # Group E: context
        blocks.append(context_features.context_length(df).rename("context_length"))
        blocks.append(context_features.context_ttr(df).rename("context_ttr"))
        blocks.append(context_features.target_position_ratio(df).rename("target_position_ratio"))

        out = pd.concat(blocks, axis=1)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature computations using stored statistics. Returns feature columns + GLMM_score if present."""
        if self._freq_norms is None or self._cognate is None:
            raise RuntimeError("Pipeline not fitted. Call fit(train_df) first.")
        feats = self._compute_all(df)
        if self._frozen_names is not None:
            available = [c for c in self._frozen_names if c in feats.columns]
            if available:
                feats = feats[available].copy()
        if "GLMM_score" in df.columns:
            feats = feats.copy()
            feats["GLMM_score"] = df["GLMM_score"].values
        return feats

    def get_feature_names(self) -> list[str]:
        """Return list of feature column names (excluding target)."""
        actual_norm_names = ["subtlex_freq", "aoa_score", "concreteness", "imageability"]
        cognate_base = ["edit_distance", "char_ngram_overlap", "is_cognate", "length_ratio"]
        if self.l1 == "de":
            cognate_base = cognate_base + ["l1_is_compound"]
        if self.l1 == "cn":
            cognate_base = cognate_base + ["cn_stroke_complexity"]
        pos_cols = ["pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV"]
        semantic = ["wordnet_num_senses", "wordnet_depth", "cefr_level"] + pos_cols
        context = ["context_length", "context_ttr", "target_position_ratio"]
        all_names = (
            ["reveal_ratio", "hidden_chars", "word_length", "syllable_count", "has_prefix", "has_suffix", "letter_frequency_score"]
            + [n for n in actual_norm_names if not self._freq_norms or n in getattr(self._freq_norms, "_tables", {})]
            + cognate_base
            + semantic
            + context
        )
        if self._frozen_names is not None:
            return [n for n in self._frozen_names if n in all_names or n.startswith("pos_")]
        return all_names
