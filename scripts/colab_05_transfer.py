"""
Phase 5, Analysis 4: Cross-L1 Transfer
=======================================
Trains XGBoost on one L1, evaluates on all L1s using shared features only.
Tests whether vocabulary difficulty is L1-specific.

No GPU needed — XGBoost trains in seconds.
"""
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

import sys
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from features.pipeline import FeaturePipeline

TRAIN = {l1: f'data/train/{l1}/kvl_shared_task_{l1}_train.csv' for l1 in ['es','de','cn']}
DEV = {l1: f'data/dev/{l1}/kvl_shared_task_{l1}_dev.csv' for l1 in ['es','de','cn']}
CLOSED_BASELINES = {'es': 1.357, 'de': 1.328, 'cn': 1.175}

# Only shared features (no L1-specific ones) so cross-L1 comparison is fair
SHARED_FEATS = [
    'reveal_ratio', 'word_length', 'syllable_count', 'has_suffix',
    'subtlex_freq', 'aoa_score', 'concreteness', 'cefr_level',
    'wordnet_num_senses', 'wordnet_depth', 'pos_NOUN', 'pos_VERB',
    'pos_ADJ', 'context_length', 'target_position_ratio',
]


def main():
    # Prepare features for all L1s
    train_data = {}
    dev_data = {}
    for l1 in ['es', 'de', 'cn']:
        train_df = pd.read_csv(TRAIN[l1])
        dev_df = pd.read_csv(DEV[l1])
        pipe = FeaturePipeline(l1=l1, frozen_features_path='frozen_features.json', resources_dir='resources')
        pipe.fit(train_df)
        X_tr = pipe.transform(train_df)
        X_dev = pipe.transform(dev_df)
        cols = [c for c in SHARED_FEATS if c in X_tr.columns]
        train_data[l1] = {'X': X_tr[cols].fillna(0), 'y': train_df['GLMM_score'].values}
        dev_data[l1] = {'X': X_dev[cols].fillna(0), 'y': dev_df['GLMM_score'].values}

    # Train on each L1, evaluate on all L1s
    print(f"{'='*60}")
    print(f"  CROSS-L1 TRANSFER (XGBoost, shared features only)")
    print(f"{'='*60}")
    print(f"\n{'Train on':<12} {'Eval es':>10} {'Eval de':>10} {'Eval cn':>10}")
    print(f"{'-'*42}")

    transfer_results = {}
    for train_l1 in ['es', 'de', 'cn']:
        model = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        model.fit(train_data[train_l1]['X'], train_data[train_l1]['y'])

        row = {}
        for eval_l1 in ['es', 'de', 'cn']:
            preds = model.predict(dev_data[eval_l1]['X'])
            rmse = np.sqrt(np.mean((dev_data[eval_l1]['y'] - preds) ** 2))
            row[eval_l1] = rmse

        transfer_results[train_l1] = row
        in_domain = row[train_l1]
        print(f"{train_l1:<12} {row['es']:>10.4f} {row['de']:>10.4f} {row['cn']:>10.4f}  (in-domain: {in_domain:.4f})")

    # Transfer degradation summary
    print(f"\n{'='*60}")
    print(f"  TRANSFER DEGRADATION (cross-L1 RMSE minus in-domain RMSE)")
    print(f"{'='*60}")
    print(f"\n{'Train on':<12} {'→ es':>10} {'→ de':>10} {'→ cn':>10}")
    print(f"{'-'*42}")
    for train_l1 in ['es', 'de', 'cn']:
        in_domain = transfer_results[train_l1][train_l1]
        deltas = {eval_l1: transfer_results[train_l1][eval_l1] - in_domain for eval_l1 in ['es', 'de', 'cn']}
        print(f"{train_l1:<12} {deltas['es']:>+10.4f} {deltas['de']:>+10.4f} {deltas['cn']:>+10.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
