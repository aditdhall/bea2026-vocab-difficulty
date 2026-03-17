"""
Phase 5, Analysis 2: SHAP on Final Hybrid Model
=================================================
Trains the hybrid model for one seed, then runs SHAP on the MLP head
to show which features the end-to-end system relies on.

Compares with XGBoost SHAP from Phase 3 to see if the transformer
already captures some feature signals.

Requires GPU. ~15 min per L1 on A100.
"""
import sys, os, torch, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


class MLPWrapper:
    """Wraps the MLP head for SHAP — takes concatenated [CLS + features] as input."""
    def __init__(self, mlp, device):
        self.mlp = mlp
        self.device = device

    def __call__(self, X):
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).to(self.device)
            out = self.mlp(t).cpu().numpy().flatten()
        return out


def main():
    os.makedirs('results/figures', exist_ok=True)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)

    print(f"{'='*60}")
    print(f"  SHAP ANALYSIS ON FINAL HYBRID MODEL")
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

        # Train model
        print("  Training model...")
        torch.manual_seed(10)
        np.random.seed(10)
        torch.cuda.manual_seed_all(10)

        model = HybridTransformerModel(encoder_name=ENCODER, feature_dim=X_tr_f.shape[1])
        model = train_hybrid(model, train_loader, dev_loader,
                             transformer_lr=1e-5, mlp_lr=1e-4,
                             epochs=8, patience=3, device=device)

        # Extract CLS embeddings for dev set
        print("  Extracting CLS embeddings...")
        model.eval()
        all_cls = []
        all_feats = []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                features = batch['features'].to(device)
                cls_emb = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
                all_cls.append(cls_emb.cpu().numpy())
                all_feats.append(features.cpu().numpy())

        cls_array = np.vstack(all_cls)
        feat_array = np.vstack(all_feats)
        combined = np.hstack([cls_array, feat_array])

        # Feature names: first 1024 are CLS dimensions, rest are handcrafted
        feature_names = [f'cls_{i}' for i in range(cls_array.shape[1])] + feat_cols

        # Run SHAP on MLP head
        print("  Running SHAP (this may take a few minutes)...")
        try:
            import shap
            mlp_wrapper = MLPWrapper(model.mlp, device)

            # Use small background sample for speed
            bg_idx = np.random.choice(len(combined), min(100, len(combined)), replace=False)
            explainer = shap.KernelExplainer(mlp_wrapper, combined[bg_idx])
            shap_values = explainer.shap_values(combined[:100], nsamples=50)

            # Focus on handcrafted features only (skip CLS dimensions)
            n_cls = cls_array.shape[1]
            handcrafted_shap = shap_values[:, n_cls:]
            handcrafted_data = combined[:100, n_cls:]

            # Summary plot
            plt.figure(figsize=(8, 5))
            shap.summary_plot(handcrafted_shap, handcrafted_data,
                              feature_names=feat_cols, show=False, max_display=15)
            plt.title(f'SHAP — Handcrafted Features ({l1.upper()})')
            plt.tight_layout()
            plt.savefig(f'results/figures/shap_final_{l1}.pdf')
            plt.close()
            print(f"  Saved results/figures/shap_final_{l1}.pdf")

            # Print top features by mean |SHAP|
            mean_shap = np.abs(handcrafted_shap).mean(axis=0)
            top_idx = np.argsort(mean_shap)[::-1][:10]
            print(f"\n  Top 10 features by mean |SHAP| ({l1}):")
            for idx in top_idx:
                print(f"    {feat_cols[idx]:<25} {mean_shap[idx]:.4f}")

        except ImportError:
            print("  SHAP not installed. Run: pip install shap")
        except Exception as e:
            print(f"  SHAP failed: {e}")

        del model
        torch.cuda.empty_cache()

    print("\nDone!")


if __name__ == '__main__':
    main()
