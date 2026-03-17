"""
Phase 5, Analysis 1: Ablation Table
=====================================
Removes one feature group at a time from the hybrid model, retrains,
and records dev RMSE to show which features contribute most.

Uses single seed (10) for speed. ~2.8 hours on A100 with 3 groups × 3 L1s.
Requires GPU.
"""
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

import sys
import os
import json
import torch
import numpy as np
import pandas as pd

for mod in list(sys.modules.keys()):
    if mod.startswith(('features', 'models')):
        del sys.modules[mod]

from features.pipeline import FeaturePipeline
from models.hybrid_transformer import HybridTransformerModel, VocabularyDataset, train_hybrid, predict_hybrid
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

TRAIN = {l1: f'data/train/{l1}/kvl_shared_task_{l1}_train.csv' for l1 in ['es','de','cn']}
DEV = {l1: f'data/dev/{l1}/kvl_shared_task_{l1}_dev.csv' for l1 in ['es','de','cn']}
ENCODER = 'xlm-roberta-large'

# Feature groups to ablate
# Adjust these lists to match your actual feature names in frozen_features.json
GROUPS = {
    'cognate': ['edit_distance', 'char_ngram_overlap', 'is_cognate', 'length_ratio', 'l1_is_compound'],
    'frequency': ['subtlex_freq', 'aoa_score', 'concreteness', 'cefr_level'],
    'orthographic': ['reveal_ratio', 'word_length', 'syllable_count', 'has_suffix'],
    'semantic': ['wordnet_num_senses', 'wordnet_depth', 'pos_NOUN', 'pos_VERB', 'pos_ADJ'],
    'context': ['context_length', 'target_position_ratio'],
}


def train_ablation(l1, exclude_features, tokenizer, device):
    """Train hybrid model with specified features excluded. Returns dev RMSE."""
    train_df = pd.read_csv(TRAIN[l1])
    dev_df = pd.read_csv(DEV[l1])

    pipe = FeaturePipeline(l1=l1, frozen_features_path='frozen_features.json', resources_dir='resources')
    pipe.fit(train_df)
    X_tr = pipe.transform(train_df)
    X_dev = pipe.transform(dev_df)

    feat_cols = [c for c in X_tr.columns if c != 'GLMM_score']
    feat_cols = [c for c in feat_cols if c not in exclude_features]

    X_tr_f = X_tr[feat_cols].fillna(0).values.astype(np.float32)
    X_dev_f = X_dev[feat_cols].fillna(0).values.astype(np.float32)

    texts_tr = (train_df['L1_source_word'].fillna('') + ' [SEP] ' +
                train_df['L1_context'].fillna('') + ' [SEP] ' +
                train_df['en_target_clue'].fillna('') + ' [SEP] ' +
                train_df['en_target_word'].fillna('')).tolist()
    texts_dev = (dev_df['L1_source_word'].fillna('') + ' [SEP] ' +
                 dev_df['L1_context'].fillna('') + ' [SEP] ' +
                 dev_df['en_target_clue'].fillna('') + ' [SEP] ' +
                 dev_df['en_target_word'].fillna('')).tolist()

    train_ds = VocabularyDataset(texts_tr, X_tr_f, train_df['GLMM_score'].values, tokenizer)
    dev_ds = VocabularyDataset(texts_dev, X_dev_f, dev_df['GLMM_score'].values, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=16)

    torch.manual_seed(10)
    np.random.seed(10)
    torch.cuda.manual_seed_all(10)

    model = HybridTransformerModel(encoder_name=ENCODER, feature_dim=X_tr_f.shape[1])
    model = train_hybrid(model, train_loader, dev_loader,
                         transformer_lr=1e-5, mlp_lr=1e-4,
                         epochs=8, patience=3, device=device)
    preds = predict_hybrid(model, dev_loader, device)
    rmse = np.sqrt(np.mean((dev_df['GLMM_score'].values - preds) ** 2))

    del model
    torch.cuda.empty_cache()
    return rmse


def main():
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    os.makedirs('results', exist_ok=True)

    results = []

    for l1 in ['es', 'de', 'cn']:
        print(f"\n{'='*60}")
        print(f"  ABLATION — {l1.upper()}")
        print(f"{'='*60}")

        # Full model (no ablation)
        full_rmse = train_ablation(l1, exclude_features=[], tokenizer=tokenizer, device=device)
        print(f"  ALL features: {full_rmse:.4f}")
        results.append({'l1': l1, 'condition': 'all_features', 'rmse': full_rmse, 'delta': 0.0})

        for group_name, group_feats in GROUPS.items():
            rmse = train_ablation(l1, exclude_features=group_feats, tokenizer=tokenizer, device=device)
            delta = rmse - full_rmse
            print(f"  minus {group_name}: {rmse:.4f} (delta={delta:+.4f})")
            results.append({'l1': l1, 'condition': f'minus_{group_name}', 'rmse': rmse, 'delta': delta})

    # Summary table
    print(f"\n{'='*70}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*70}")
    results_df = pd.DataFrame(results)
    pivot = results_df.pivot(index='condition', columns='l1', values='rmse')
    pivot = pivot[['es', 'de', 'cn']]
    print(pivot.to_string(float_format='%.4f'))

    delta_pivot = results_df.pivot(index='condition', columns='l1', values='delta')
    delta_pivot = delta_pivot[['es', 'de', 'cn']]
    print(f"\nDeltas (positive = removing hurt = feature was useful):")
    print(delta_pivot.to_string(float_format='%+.4f'))

    # Save
    results_df.to_csv('results/ablation_results.csv', index=False)
    print("\nSaved to results/ablation_results.csv")


if __name__ == '__main__':
    main()
