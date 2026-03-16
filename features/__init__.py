"""BEA 2026 Vocabulary Difficulty — feature extraction modules."""

from features.pipeline import FeaturePipeline
from features.clue_orthographic import (
    reveal_ratio,
    hidden_chars,
    word_length,
    syllable_count,
    has_prefix,
    has_suffix,
    letter_frequency_score,
)
from features.frequency_norms import FrequencyNormFeatures
from features.cognate_crossling import CognateFeatures
from features.semantic_lexical import (
    wordnet_num_senses,
    wordnet_depth,
    pos_encoded,
    cefr_level,
)
from features.context_features import (
    context_length,
    context_ttr,
    target_position_ratio,
)

__all__ = [
    "FeaturePipeline",
    "reveal_ratio",
    "hidden_chars",
    "word_length",
    "syllable_count",
    "has_prefix",
    "has_suffix",
    "letter_frequency_score",
    "FrequencyNormFeatures",
    "CognateFeatures",
    "wordnet_num_senses",
    "wordnet_depth",
    "pos_encoded",
    "cefr_level",
    "context_length",
    "context_ttr",
    "target_position_ratio",
]
