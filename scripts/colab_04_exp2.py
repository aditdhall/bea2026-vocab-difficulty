"""
Phase 4, Exp 2: Better Transformer — mdeberta-v3-base
======================================================
Same hybrid architecture as Exp 4 but uses microsoft/mdeberta-v3-base
instead of xlm-roberta-large. Tests whether a different encoder
architecture improves results.

CLOSED TRACK: mdeberta is encoder-only. Compliant.
Requires GPU. ~1 hr on A100.
"""
import sys, os, torch, numpy as np, pandas as pd

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
CLOSED_BASELINES = {'es': 1.357, 'de': 1.328, 'cn': 1.175}
ENCODER = 'microsoft/mdeberta-v3-base'
SEEDS = [10, 42, 123]  # 3 seeds for comparison (not full 5 — this is exploratory)
PRED_DIR = 'results/predictions'


def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)

    print(f"{'='*60}")
    print(f"  EXP 2: mdeberta-v3-base hybrid, 3 seeds")
    print(f"{'='*60}")

    for l1 in ['es', 'de', 'cn']:
        print(f"\n--- {l1.upper()} ---")
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

        train_ds = VocabularyDataset(texts_tr, X_tr_f, train_df['GLMM_score'].values, tokenizer)
        dev_ds = VocabularyDataset(texts_dev, X_dev_f, dev_df['GLMM_score'].values, tokenizer)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        dev_loader = DataLoader(dev_ds, batch_size=16)

        seed_rmses = []
        seed_preds = []

        for seed in SEEDS:
            print(f"  seed={seed}...")
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)

            model = HybridTransformerModel(encoder_name=ENCODER, feature_dim=X_tr_f.shape[1])
            model = train_hybrid(model, train_loader, dev_loader,
                                 transformer_lr=1e-5, mlp_lr=1e-4,
                                 epochs=8, patience=3, device=device)
            preds = predict_hybrid(model, dev_loader, device)
            rmse = np.sqrt(np.mean((dev_df['GLMM_score'].values - preds)**2))
            seed_rmses.append(rmse)
            seed_preds.append(preds)
            print(f"  seed={seed} RMSE={rmse:.4f}")

            out = pd.DataFrame({'item_id': dev_df['item_id'], 'prediction': preds})
            out.to_csv(f'{PRED_DIR}/exp2_mdeberta_{l1}_seed{seed}_dev.csv', index=False)

            del model
            torch.cuda.empty_cache()

        # Ensemble
        ensemble_preds = np.mean(seed_preds, axis=0)
        ensemble_rmse = np.sqrt(np.mean((dev_df['GLMM_score'].values - ensemble_preds)**2))
        out = pd.DataFrame({'item_id': dev_df['item_id'], 'prediction': ensemble_preds})
        out.to_csv(f'{PRED_DIR}/exp2_mdeberta_ensemble_{l1}_dev.csv', index=False)

        print(f"  {l1} mean: {np.mean(seed_rmses):.4f}±{np.std(seed_rmses):.4f}, ensemble: {ensemble_rmse:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
