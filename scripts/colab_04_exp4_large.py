"""
Phase 4, Exp 4: Hybrid Model — xlm-roberta-large + features (5-seed ensemble)
==============================================================================
MAIN CONTRIBUTION. Combines transformer [CLS] embedding with frozen
psycholinguistic features via MLP. Uses xlm-roberta-large encoder.

Architecture:
  xlm-roberta-large [CLS] (1024) + frozen features (N) → concat →
  Linear(1024+N, 256) → ReLU → Dropout(0.1) → Linear(256, 1)

Two learning rates: transformer 1e-5, MLP 1e-4
5 seeds: [10, 42, 123, 456, 789]

CLOSED TRACK: Per-L1 training. Encoder-only model. Compliant.
Requires GPU (A100 recommended). ~1 hour total.
"""
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

import sys
import os
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
CLOSED_BASELINES = {'es': 1.357, 'de': 1.328, 'cn': 1.175}
ENCODER = 'xlm-roberta-large'
SEEDS = [10, 42, 123, 456, 789]
PRED_DIR = 'results/predictions'
CKPT_DIR = 'models/checkpoints'


def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)

    all_results = {}

    for l1 in ['es', 'de', 'cn']:
        print(f"\n{'='*60}")
        print(f"  {l1.upper()} — xlm-roberta-large hybrid, 5 seeds")
        print(f"{'='*60}")

        train_df = pd.read_csv(TRAIN[l1])
        dev_df = pd.read_csv(DEV[l1])

        pipe = FeaturePipeline(l1=l1, frozen_features_path='frozen_features.json', resources_dir='resources')
        pipe.fit(train_df)
        X_tr = pipe.transform(train_df)
        X_dev = pipe.transform(dev_df)
        feat_cols = [c for c in X_tr.columns if c != 'GLMM_score']
        X_tr_f = X_tr[feat_cols].fillna(0).values.astype(np.float32)
        X_dev_f = X_dev[feat_cols].fillna(0).values.astype(np.float32)

        # Input text: L1_source_word [SEP] L1_context [SEP] en_target_clue [SEP] en_target_word
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

        feature_dim = X_tr_f.shape[1]
        seed_rmses = []
        seed_preds = []

        for seed in SEEDS:
            print(f"\n  seed={seed}...")
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)

            model = HybridTransformerModel(encoder_name=ENCODER, feature_dim=feature_dim)
            model = train_hybrid(model, train_loader, dev_loader,
                                 transformer_lr=1e-5, mlp_lr=1e-4,
                                 epochs=8, patience=3, device=device)
            preds = predict_hybrid(model, dev_loader, device)
            rmse = np.sqrt(np.mean((dev_df['GLMM_score'].values - preds) ** 2))
            seed_rmses.append(rmse)
            seed_preds.append(preds)
            print(f"  seed={seed} RMSE={rmse:.4f}")

            # Save per-seed predictions
            out = pd.DataFrame({'item_id': dev_df['item_id'], 'prediction': preds})
            out.to_csv(f'{PRED_DIR}/exp4_large_{l1}_seed{seed}_dev.csv', index=False)

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

        # Ensemble
        ensemble_preds = np.mean(seed_preds, axis=0)
        ensemble_rmse = np.sqrt(np.mean((dev_df['GLMM_score'].values - ensemble_preds) ** 2))
        out = pd.DataFrame({'item_id': dev_df['item_id'], 'prediction': ensemble_preds})
        out.to_csv(f'{PRED_DIR}/exp4_large_ensemble_{l1}_dev.csv', index=False)

        mean_rmse = np.mean(seed_rmses)
        std_rmse = np.std(seed_rmses)
        all_results[l1] = {'mean': mean_rmse, 'std': std_rmse, 'ensemble': ensemble_rmse, 'seeds': seed_rmses}

        print(f"\n  {l1} mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        print(f"  {l1} ensemble RMSE: {ensemble_rmse:.4f}")
        print(f"  {l1} seeds: {[f'{r:.4f}' for r in seed_rmses]}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS — xlm-roberta-large hybrid + features")
    print(f"{'='*70}")
    print(f"{'L1':<4} {'Mean±Std':<18} {'Ensemble':<10} {'Baseline':<10} {'vs Baseline':<12}")
    print(f"{'-'*70}")
    for l1 in ['es', 'de', 'cn']:
        r = all_results[l1]
        delta = r['ensemble'] - CLOSED_BASELINES[l1]
        print(f"{l1:<4} {r['mean']:.4f}±{r['std']:.4f}   {r['ensemble']:.4f}     {CLOSED_BASELINES[l1]:.3f}     {delta:+.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
