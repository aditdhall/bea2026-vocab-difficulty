"""
Phase 4, Exp 6: Ensemble + XGBoost Blend
=========================================
Combines 5-seed hybrid ensemble predictions with XGBoost feature-only
predictions using weighted averaging. Tests multiple alpha values to
find optimal blend per L1.

Submission runs:
  Run 1 = best single seed
  Run 2 = 5-seed ensemble
  Run 3 = ensemble + XGBoost blend (this script)

Requires: Exp 4 large ensemble predictions + frozen_features.json
"""
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

import sys
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from features.pipeline import FeaturePipeline

TRAIN = {l1: f'data/train/{l1}/kvl_shared_task_{l1}_train.csv' for l1 in ['es','de','cn']}
DEV = {l1: f'data/dev/{l1}/kvl_shared_task_{l1}_dev.csv' for l1 in ['es','de','cn']}
CLOSED_BASELINES = {'es': 1.357, 'de': 1.328, 'cn': 1.175}
PRED_DIR = 'results/predictions'


def main():
    os.makedirs(PRED_DIR, exist_ok=True)

    # Step 1: Generate fresh XGBoost predictions
    print("Step 1: Training XGBoost per L1...")
    xgb_preds = {}
    for l1 in ['es', 'de', 'cn']:
        train_df = pd.read_csv(TRAIN[l1])
        dev_df = pd.read_csv(DEV[l1])
        pipe = FeaturePipeline(l1=l1, frozen_features_path='frozen_features.json', resources_dir='resources')
        pipe.fit(train_df)
        X_tr = pipe.transform(train_df)
        X_dev = pipe.transform(dev_df)
        feat_cols = [c for c in X_tr.columns if c != 'GLMM_score']

        model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42)
        model.fit(X_tr[feat_cols].fillna(0), train_df['GLMM_score'])
        preds = model.predict(X_dev[feat_cols].fillna(0))
        rmse = np.sqrt(np.mean((dev_df['GLMM_score'].values - preds) ** 2))
        xgb_preds[l1] = preds
        print(f"  {l1} XGBoost RMSE: {rmse:.4f}")

    # Step 2: Load hybrid ensemble predictions
    print("\nStep 2: Loading hybrid ensemble predictions...")
    hybrid_preds = {}
    for l1 in ['es', 'de', 'cn']:
        pred_path = f'{PRED_DIR}/exp4_large_ensemble_{l1}_dev.csv'
        df = pd.read_csv(pred_path)
        hybrid_preds[l1] = df['prediction'].values
        print(f"  Loaded {pred_path}")

    # Step 3: Try different blend weights
    print(f"\n{'='*60}")
    print(f"  BLEND EXPERIMENTS")
    print(f"{'='*60}")
    for alpha in [0.9, 0.8, 0.7, 0.6]:
        print(f"\nalpha={alpha} (hybrid={alpha}, xgb={1-alpha}):")
        for l1 in ['es', 'de', 'cn']:
            dev_df = pd.read_csv(DEV[l1])
            blended = alpha * hybrid_preds[l1] + (1 - alpha) * xgb_preds[l1]
            rmse = np.sqrt(np.mean((dev_df['GLMM_score'].values - blended) ** 2))
            delta = rmse - CLOSED_BASELINES[l1]
            print(f"  {l1}: {rmse:.4f} (vs baseline {delta:+.4f})")

    # Step 4: Save per-L1 best blend
    best_alphas = {'es': 0.8, 'de': 0.8, 'cn': 0.9}
    print(f"\nSaving per-L1 best blends: {best_alphas}")
    for l1 in ['es', 'de', 'cn']:
        dev_df = pd.read_csv(DEV[l1])
        a = best_alphas[l1]
        blended = a * hybrid_preds[l1] + (1 - a) * xgb_preds[l1]
        out = pd.DataFrame({'item_id': dev_df['item_id'], 'prediction': blended})
        out.to_csv(f'{PRED_DIR}/exp6_best_blend_{l1}_dev.csv', index=False)
        rmse = np.sqrt(np.mean((dev_df['GLMM_score'].values - blended) ** 2))
        print(f"  {l1} (alpha={a}): {rmse:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
