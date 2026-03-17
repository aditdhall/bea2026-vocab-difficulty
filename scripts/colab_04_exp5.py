"""
Phase 4, Exp 5: Hyperparameter Tuning
======================================
Random search over learning rate, batch size, warmup ratio, and epochs.
Max 15 configurations to avoid overfitting to dev set.

CLOSED TRACK: Per-L1 training. Encoder-only. Compliant.
Requires GPU. ~2 hrs on A100.
"""
import sys, os, torch, numpy as np, pandas as pd, itertools, random

for mod in list(sys.modules.keys()):
    if mod.startswith(('features', 'models')):
        del sys.modules[mod]

from models.hybrid_transformer import HybridTransformerModel, VocabularyDataset, train_hybrid, predict_hybrid
from features.pipeline import FeaturePipeline
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

TRAIN = {l1: f'data/train/{l1}/kvl_shared_task_{l1}_train.csv' for l1 in ['es','de','cn']}
DEV = {l1: f'data/dev/{l1}/kvl_shared_task_{l1}_dev.csv' for l1 in ['es','de','cn']}
ENCODER = 'xlm-roberta-large'
PRED_DIR = 'results/predictions'

# Hyperparameter search space
HP_SPACE = {
    'transformer_lr': [5e-6, 1e-5, 2e-5],
    'batch_size': [8, 16],
    'warmup_ratio': [0.05, 0.1, 0.2],
    'epochs': [5, 8],
}
MAX_CONFIGS = 15


def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)

    # Generate all combos, sample MAX_CONFIGS
    all_combos = list(itertools.product(
        HP_SPACE['transformer_lr'], HP_SPACE['batch_size'],
        HP_SPACE['warmup_ratio'], HP_SPACE['epochs'],
    ))
    random.seed(42)
    sampled = random.sample(all_combos, min(MAX_CONFIGS, len(all_combos)))

    print(f"{'='*60}")
    print(f"  EXP 5: Hyperparameter Tuning ({len(sampled)} configs)")
    print(f"{'='*60}")

    # Run on ES only first (fastest L1 to iterate on)
    l1 = 'es'
    print(f"\nRunning on {l1.upper()} only for speed...")

    train_df = pd.read_csv(TRAIN[l1])
    dev_df = pd.read_csv(DEV[l1])

    pipe = FeaturePipeline(l1=l1, frozen_features_path='frozen_features.json', resources_dir='resources')
    pipe.fit(train_df)
    X_tr = pipe.transform(train_df)
    X_dev = pipe.transform(dev_df)
    feat_cols = [c for c in X_tr.columns if c != 'GLMM_score']
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

    results = []

    for i, (t_lr, bs, wr, ep) in enumerate(sampled):
        print(f"\n  Config {i+1}/{len(sampled)}: lr={t_lr}, bs={bs}, warmup={wr}, epochs={ep}")

        train_ds = VocabularyDataset(texts_tr, X_tr_f, train_df['GLMM_score'].values, tokenizer)
        dev_ds = VocabularyDataset(texts_dev, X_dev_f, dev_df['GLMM_score'].values, tokenizer)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        dev_loader = DataLoader(dev_ds, batch_size=bs)

        torch.manual_seed(10)
        np.random.seed(10)
        torch.cuda.manual_seed_all(10)

        model = HybridTransformerModel(encoder_name=ENCODER, feature_dim=X_tr_f.shape[1])
        model = train_hybrid(model, train_loader, dev_loader,
                             transformer_lr=t_lr, mlp_lr=t_lr * 10,
                             epochs=ep, patience=3, device=device)
        preds = predict_hybrid(model, dev_loader, device)
        rmse = np.sqrt(np.mean((dev_df['GLMM_score'].values - preds)**2))
        print(f"  RMSE: {rmse:.4f}")

        results.append({
            'config_id': i+1,
            'transformer_lr': t_lr,
            'batch_size': bs,
            'warmup_ratio': wr,
            'epochs': ep,
            'rmse': rmse,
        })

        del model
        torch.cuda.empty_cache()

    # Summary
    results_df = pd.DataFrame(results).sort_values('rmse')
    print(f"\n{'='*60}")
    print(f"  HYPERPARAMETER SEARCH RESULTS (sorted by RMSE)")
    print(f"{'='*60}")
    print(results_df.to_string(index=False, float_format='%.4f'))

    best = results_df.iloc[0]
    print(f"\n  BEST CONFIG: lr={best['transformer_lr']}, bs={int(best['batch_size'])}, "
          f"warmup={best['warmup_ratio']}, epochs={int(best['epochs'])}, RMSE={best['rmse']:.4f}")

    results_df.to_csv('results/hparam_search.csv', index=False)
    print("\nSaved to results/hparam_search.csv")


if __name__ == '__main__':
    main()
