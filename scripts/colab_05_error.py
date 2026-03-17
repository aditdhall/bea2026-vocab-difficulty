"""
Phase 5, Analysis 3: Error Analysis
====================================
Analyzes top prediction errors per L1 from the hybrid ensemble.
Categorizes false-easy and false-hard predictions.

Run after Exp 4 ensemble predictions are saved.
"""
import pandas as pd
import numpy as np
import os

TRAIN = {l1: f'data/train/{l1}/kvl_shared_task_{l1}_train.csv' for l1 in ['es','de','cn']}
DEV = {l1: f'data/dev/{l1}/kvl_shared_task_{l1}_dev.csv' for l1 in ['es','de','cn']}
CLOSED_BASELINES = {'es': 1.357, 'de': 1.328, 'cn': 1.175}
PRED_DIR = 'results/predictions'


def main():
    for l1 in ['es', 'de', 'cn']:
        print(f"\n{'='*70}")
        print(f"  ERROR ANALYSIS — {l1.upper()}")
        print(f"{'='*70}")

        dev_df = pd.read_csv(DEV[l1])
        pred_path = f'{PRED_DIR}/exp4_large_ensemble_{l1}_dev.csv'
        if not os.path.exists(pred_path):
            print(f"  Predictions not found: {pred_path}. Run Exp 4 first.")
            continue

        pred_df = pd.read_csv(pred_path)
        merged = dev_df.merge(pred_df, on='item_id')
        merged['error'] = merged['prediction'] - merged['GLMM_score']
        merged['abs_error'] = merged['error'].abs()

        # Overall stats
        print(f"\n  Mean abs error: {merged['abs_error'].mean():.4f}")
        print(f"  Median abs error: {merged['abs_error'].median():.4f}")
        print(f"  Max abs error: {merged['abs_error'].max():.4f}")

        # FALSE EASY: model predicted high (easy) but actually hard (low GLMM)
        false_easy = merged[merged['error'] > 0].nlargest(10, 'abs_error')
        print(f"\n  FALSE EASY (model thought easy, actually hard):")
        print(f"  {'word':<20} {'true':>8} {'pred':>8} {'error':>8} {'L1_word':<20}")
        print(f"  {'-'*64}")
        for _, row in false_easy.iterrows():
            print(f"  {row['en_target_word']:<20} {row['GLMM_score']:>8.3f} {row['prediction']:>8.3f} {row['error']:>+8.3f} {str(row['L1_source_word']):<20}")

        # FALSE HARD: model predicted low (hard) but actually easy (high GLMM)
        false_hard = merged[merged['error'] < 0].nsmallest(10, 'error')
        print(f"\n  FALSE HARD (model thought hard, actually easy):")
        print(f"  {'word':<20} {'true':>8} {'pred':>8} {'error':>8} {'L1_word':<20}")
        print(f"  {'-'*64}")
        for _, row in false_hard.iterrows():
            print(f"  {row['en_target_word']:<20} {row['GLMM_score']:>8.3f} {row['prediction']:>8.3f} {row['error']:>+8.3f} {str(row['L1_source_word']):<20}")

        # Error by POS
        print(f"\n  Error by POS:")
        pos_err = merged.groupby('en_target_pos')['abs_error'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        for pos, row in pos_err.iterrows():
            print(f"  {pos:<8} mean_err={row['mean']:.4f}  n={int(row['count'])}")

    print("\nDone!")


if __name__ == '__main__':
    main()
