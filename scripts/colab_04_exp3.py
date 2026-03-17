"""
Phase 4, Exp 3: Structured Input Format
========================================
Uses explicit field labels in the input text instead of naive concatenation.
Tests whether labeling helps the transformer understand field boundaries.

Input format:
  "English: {word} | POS: {pos} | Clue: {clue} | L1 word: {L1_word} | Context: {context}"

vs Exp 4 default:
  "{L1_word} [SEP] {context} [SEP] {clue} [SEP] {word}"

CLOSED TRACK: Just a formatting change. Fully compliant.
Requires GPU. ~30 min on A100 (single seed per L1).
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
ENCODER = 'xlm-roberta-large'
PRED_DIR = 'results/predictions'


def structured_text(df):
    """Create structured input text with explicit field labels."""
    return df.apply(lambda r: (
        f"English: {r['en_target_word']} | "
        f"POS: {r['en_target_pos']} | "
        f"Clue: {r['en_target_clue']} | "
        f"L1 word: {r['L1_source_word']} | "
        f"Context: {r['L1_context']}"
    ), axis=1).tolist()


def default_text(df):
    """Default concatenation (same as Exp 4)."""
    return (df['L1_source_word'].fillna('') + ' [SEP] ' +
            df['L1_context'].fillna('') + ' [SEP] ' +
            df['en_target_clue'].fillna('') + ' [SEP] ' +
            df['en_target_word'].fillna('')).tolist()


def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)

    print(f"{'='*60}")
    print(f"  EXP 3: Structured vs Default Input Format")
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

        # Compare both formats
        for fmt_name, fmt_fn in [('structured', structured_text), ('default', default_text)]:
            texts_tr = fmt_fn(train_df)
            texts_dev = fmt_fn(dev_df)

            print(f"  Format: {fmt_name}")
            print(f"    Example: {texts_tr[0][:100]}...")

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
            rmse = np.sqrt(np.mean((dev_df['GLMM_score'].values - preds)**2))
            print(f"    RMSE: {rmse:.4f}")

            out = pd.DataFrame({'item_id': dev_df['item_id'], 'prediction': preds})
            out.to_csv(f'{PRED_DIR}/exp3_{fmt_name}_{l1}_dev.csv', index=False)

            del model
            torch.cuda.empty_cache()

    print("\nDone!")


if __name__ == '__main__':
    main()
