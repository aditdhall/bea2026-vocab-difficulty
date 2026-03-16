"""Colab script: Exp 4 — Hybrid model (MAIN CONTRIBUTION).

Transformer [CLS] (768) + frozen handcrafted features (N) → MLP → scalar.
One model per L1 × 5 seeds. Saves predictions and 5-seed ensemble.

Runs on Google Colab Pro with A100 GPU.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Ensure project root is on path when run from scripts/
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_PROJECT_ROOT))

# NLTK for WordNet (used by feature pipeline)
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from features.pipeline import FeaturePipeline
from models.hybrid_transformer import (
    HybridTransformerModel,
    VocabularyDataset,
    train_hybrid,
    predict_hybrid,
    _load_frozen_feature_names,
)


TRAIN = {l1: f"data/train/{l1}/kvl_shared_task_{l1}_train.csv" for l1 in ["es", "de", "cn"]}
DEV = {l1: f"data/dev/{l1}/kvl_shared_task_{l1}_dev.csv" for l1 in ["es", "de", "cn"]}

CLOSED_BASELINES = {"es": 1.357, "de": 1.328, "cn": 1.175}
EXP0_RESULTS = {"es": 1.338, "de": 1.299, "cn": 1.222}

ENCODER_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 32
TRANSFORMER_LR = 1e-5
MLP_LR = 1e-4
EPOCHS = 8
PATIENCE = 3
SEEDS = [10, 42, 123, 456, 789]
WEIGHT_DECAY = {"es": 0.1, "de": 0.0, "cn": 0.1}
FROZEN_FEATURES_PATH = "frozen_features.json"
RESOURCES_DIR = "resources"
PRED_DIR = "results/predictions"
CKPT_DIR = "models/checkpoints"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def build_text_official(row: pd.Series) -> str:
    """Official order: L1_source_word [SEP] L1_context [SEP] en_target_clue [SEP] en_target_word."""
    return (
        f"{row.get('L1_source_word', '')} [SEP] "
        f"{row.get('L1_context', '')} [SEP] "
        f"{row.get('en_target_clue', '')} [SEP] "
        f"{row.get('en_target_word', '')}"
    )


def get_feature_counts_per_l1() -> dict[str, int]:
    """Return number of frozen features per L1 (for logging)."""
    if not os.path.isfile(FROZEN_FEATURES_PATH):
        return {}
    with open(FROZEN_FEATURES_PATH) as f:
        data = json.load(f)
    shared = len(data.get("shared", []))
    es_de = len(data.get("es_de_specific", []))
    cn = len(data.get("cn_specific", []))
    return {
        "es": shared + es_de,
        "de": shared + es_de,
        "cn": shared + cn,
    }


def run_l1_seed(
    l1: str,
    seed: int,
    device: torch.device,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    pipeline: FeaturePipeline,
    feat_cols: list[str],
    tokenizer,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Train one hybrid model for one L1 and one seed. Returns dev_rmse, dev_preds, dev_item_ids."""
    set_seed(seed)

    # Feature matrices (already fitted pipeline)
    X_train = pipeline.transform(train_df)
    X_dev = pipeline.transform(dev_df)
    X_train_f = X_train[feat_cols].fillna(0).astype(np.float32).values
    X_dev_f = X_dev[feat_cols].fillna(0).astype(np.float32).values

    texts_train = [build_text_official(train_df.iloc[i]) for i in range(len(train_df))]
    texts_dev = [build_text_official(dev_df.iloc[i]) for i in range(len(dev_df))]

    train_ds = VocabularyDataset(
        texts_train,
        X_train_f,
        train_df["GLMM_score"].values,
        tokenizer,
        max_length=MAX_LENGTH,
    )
    dev_ds = VocabularyDataset(
        texts_dev,
        X_dev_f,
        dev_df["GLMM_score"].values,
        tokenizer,
        max_length=MAX_LENGTH,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = HybridTransformerModel(
        encoder_name=ENCODER_NAME,
        feature_dim=len(feat_cols),
        hidden_dim=256,
        dropout=0.1,
    )

    train_hybrid(
        model,
        train_loader,
        dev_loader,
        transformer_lr=TRANSFORMER_LR,
        mlp_lr=MLP_LR,
        epochs=EPOCHS,
        patience=PATIENCE,
        device=device,
        weight_decay=WEIGHT_DECAY[l1],
    )

    dev_preds = predict_hybrid(model, dev_loader, device)
    dev_rmse_val = rmse(dev_df["GLMM_score"].values, dev_preds)

    os.makedirs(PRED_DIR, exist_ok=True)
    pred_path = os.path.join(PRED_DIR, f"exp4_hybrid_{l1}_seed{seed}_dev.csv")
    pd.DataFrame({"item_id": dev_df["item_id"].values, "prediction": dev_preds}).to_csv(pred_path, index=False)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"exp4_{l1}_seed{seed}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "encoder_name": ENCODER_NAME,
            "feature_dim": len(feat_cols),
            "max_length": MAX_LENGTH,
        },
        ckpt_path,
    )

    return dev_rmse_val, dev_preds, dev_df["item_id"].values


def main() -> None:
    # Run from project root so data/ and frozen_features.json resolve
    os.chdir(_PROJECT_ROOT)

    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model: hybrid =", ENCODER_NAME, "+ frozen features → MLP(256) → 1")
    print("Input order: L1_source_word [SEP] L1_context [SEP] en_target_clue [SEP] en_target_word")
    print("Hyperparams: transformer_lr=1e-5, mlp_lr=1e-4, epochs=8, patience=3, batch_size=32")
    print("Seeds:", SEEDS)

    feat_counts = get_feature_counts_per_l1()
    for l1 in ["es", "de", "cn"]:
        n = feat_counts.get(l1, "?")
        print(f"  {l1}: {n} frozen features")

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)

    all_results = []

    for l1 in tqdm(["es", "de", "cn"], desc="L1"):
        train_path = TRAIN[l1]
        dev_path = DEV[l1]
        if not os.path.isfile(train_path) or not os.path.isfile(dev_path):
            raise FileNotFoundError(f"Missing data for {l1}: {train_path}, {dev_path}")

        train_df = pd.read_csv(train_path)
        dev_df = pd.read_csv(dev_path)

        frozen_path = FROZEN_FEATURES_PATH
        resources_dir = RESOURCES_DIR
        pipeline = FeaturePipeline(
            l1=l1,
            frozen_features_path=frozen_path,
            resources_dir=resources_dir,
        )
        pipeline.fit(train_df)
        X_train = pipeline.transform(train_df)
        feat_cols = [c for c in X_train.columns if c != "GLMM_score"]
        if not feat_cols:
            raise RuntimeError(f"No frozen features for L1={l1}")

        seed_rmses = []
        seed_preds = []

        for seed in tqdm(SEEDS, desc=f"{l1} seeds", leave=False):
            dev_rmse_val, dev_preds, _ = run_l1_seed(
                l1=l1,
                seed=seed,
                device=device,
                train_df=train_df,
                dev_df=dev_df,
                pipeline=pipeline,
                feat_cols=feat_cols,
                tokenizer=tokenizer,
            )
            seed_rmses.append(dev_rmse_val)
            seed_preds.append(dev_preds)

        mean_rmse = float(np.mean(seed_rmses))
        std_rmse = float(np.std(seed_rmses))
        print(f"L1={l1} | mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} (seeds: {seed_rmses})")

        # 5-seed ensemble (average)
        ensemble_preds = np.mean(seed_preds, axis=0)
        ensemble_path = os.path.join(PRED_DIR, f"exp4_ensemble_{l1}_dev.csv")
        pd.DataFrame({"item_id": dev_df["item_id"].values, "prediction": ensemble_preds}).to_csv(
            ensemble_path, index=False
        )
        ensemble_rmse = rmse(dev_df["GLMM_score"].values, ensemble_preds)
        print(f"L1={l1} | ensemble dev RMSE: {ensemble_rmse:.4f} -> {ensemble_path}")

        all_results.append({
            "l1": l1,
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "ensemble_rmse": ensemble_rmse,
            "baseline": CLOSED_BASELINES[l1],
            "exp0": EXP0_RESULTS[l1],
        })

    print("\n=== Exp 4 summary (hybrid: transformer + frozen features) ===")
    print(f"{'L1':<4} {'mean±std':<14} {'ensemble':<10} {'baseline':<10} {'Exp0':<10} {'vs base':<10} {'vs Exp0':<10}")
    print("-" * 70)
    for r in all_results:
        vs_base = r["mean_rmse"] - r["baseline"]
        vs_exp0 = r["mean_rmse"] - r["exp0"]
        print(
            f"{r['l1']:<4} {r['mean_rmse']:.4f}±{r['std_rmse']:.4f}   "
            f"{r['ensemble_rmse']:.4f}     {r['baseline']:.3f}      {r['exp0']:.3f}      "
            f"{vs_base:+.4f}     {vs_exp0:+.4f}"
        )
    print("\nClosed baselines: es=1.357, de=1.328, cn=1.175")
    print("Exp 0 (our repro): es=1.338, de=1.299, cn=1.222")
    print("Ensemble predictions: results/predictions/exp4_ensemble_{l1}_dev.csv")


if __name__ == "__main__":
    main()
