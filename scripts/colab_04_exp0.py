"""Colab script: Exp 0 — reproduce official closed-track baseline.

Trains xlm-roberta-base as a regression model to predict GLMM_score,
one model per L1 (es/de/cn). Saves dev predictions to:
  results/predictions/exp0_{l1}_dev.csv

Official hyperparams (from upstream/model_parameters.csv):
  - model: xlm-roberta-base
  - component_order: "L1_source_word [SEP] L1_context [SEP] en_target_clue [SEP] en_target_word"
  - batch_size: 32
  - learning_rate: 3e-5
  - warmup_ratio: 0.1
  - epochs: 5
  - weight_decay: es=0.1, de=0.0, cn=0.1
  - seed: 10
  - max_length: 128
"""

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup


TRAIN = {l1: f"data/train/{l1}/kvl_shared_task_{l1}_train.csv" for l1 in ["es", "de", "cn"]}
DEV = {l1: f"data/dev/{l1}/kvl_shared_task_{l1}_dev.csv" for l1 in ["es", "de", "cn"]}

CLOSED_BASELINES = {"es": 1.357, "de": 1.328, "cn": 1.175}

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 32
LR = 3e-5
WARMUP_RATIO = 0.1
EPOCHS = 5
SEED = 10
WEIGHT_DECAY = {"es": 0.1, "de": 0.0, "cn": 0.1}


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


class BEAClosedDataset(Dataset):
    """Dataset that builds the official input string order and tokenizes to max_length."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128, with_labels: bool = True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_labels = with_labels

    @staticmethod
    def build_text(row: pd.Series) -> str:
        # Official order:
        # L1_source_word [SEP] L1_context [SEP] en_target_clue [SEP] en_target_word
        return (
            f"{row.get('L1_source_word', '')} [SEP] "
            f"{row.get('L1_context', '')} [SEP] "
            f"{row.get('en_target_clue', '')} [SEP] "
            f"{row.get('en_target_word', '')}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = self.build_text(row)
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.with_labels:
            item["labels"] = torch.tensor(float(row["GLMM_score"]), dtype=torch.float32)
        item["item_id"] = torch.tensor(int(row["item_id"]), dtype=torch.long)
        return item


class XLMRRegression(nn.Module):
    """xlm-roberta-base encoder + linear regression head on [CLS]."""

    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(getattr(self.encoder.config, "hidden_size", 768))
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # (B, 768)
        pred = self.regressor(cls).squeeze(-1)  # (B,)
        return pred


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, ys, item_ids = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        p = model(input_ids=input_ids, attention_mask=attention_mask).detach().cpu().numpy()
        preds.append(p)
        item_ids.append(batch["item_id"].cpu().numpy())
        if "labels" in batch:
            ys.append(batch["labels"].cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    ids = np.concatenate(item_ids, axis=0)
    y_true = np.concatenate(ys, axis=0) if ys else np.array([])
    return ids, y_true, y_pred


def train_one_l1(l1: str, device: torch.device) -> dict:
    set_seed(SEED)

    train_path = TRAIN[l1]
    dev_path = DEV[l1]
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_ds = BEAClosedDataset(train_df, tokenizer, max_length=MAX_LENGTH, with_labels=True)
    dev_ds = BEAClosedDataset(dev_df, tokenizer, max_length=MAX_LENGTH, with_labels=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = XLMRRegression(MODEL_NAME).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=float(WEIGHT_DECAY[l1]),
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(math.ceil(WARMUP_RATIO * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_rmse = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"{l1} epoch {epoch}/{EPOCHS}", leave=False)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            preds = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.functional.mse_loss(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        ids, y_true, y_pred = predict(model, dev_loader, device)
        dev_rmse = rmse(y_true, y_pred)
        print(f"L1={l1} | epoch {epoch} | dev RMSE: {dev_rmse:.4f}")

        if dev_rmse < best_rmse:
            best_rmse = dev_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save best model checkpoint (optional, but useful on Colab)
    ckpt_dir = os.path.join("models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"exp0_{l1}_best.pt")
    torch.save({"state_dict": model.state_dict(), "model_name": MODEL_NAME, "max_length": MAX_LENGTH}, ckpt_path)

    # Final dev predictions
    ids, y_true, y_pred = predict(model, dev_loader, device)
    pred_dir = os.path.join("results", "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    out_path = os.path.join(pred_dir, f"exp0_{l1}_dev.csv")
    out_df = pd.DataFrame({"item_id": ids.astype(int), "prediction": y_pred.astype(float)})
    out_df.to_csv(out_path, index=False)

    return {
        "l1": l1,
        "dev_rmse": float(best_rmse),
        "baseline_rmse": float(CLOSED_BASELINES[l1]),
        "delta": float(best_rmse - CLOSED_BASELINES[l1]),
        "pred_path": out_path,
        "ckpt_path": ckpt_path,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model:", MODEL_NAME)
    print("Official input order: L1_source_word [SEP] L1_context [SEP] en_target_clue [SEP] en_target_word")
    print("Hyperparams:", {"batch_size": BATCH_SIZE, "lr": LR, "warmup_ratio": WARMUP_RATIO, "epochs": EPOCHS, "seed": SEED, "max_length": MAX_LENGTH})

    results = []
    for l1 in ["es", "de", "cn"]:
        if not os.path.isfile(TRAIN[l1]) or not os.path.isfile(DEV[l1]):
            raise FileNotFoundError(f"Missing data for {l1}. Expected: {TRAIN[l1]} and {DEV[l1]}")
        results.append(train_one_l1(l1, device))

    print("\n=== Exp 0 summary (closed track baselines) ===")
    for r in results:
        print(
            f"L1={r['l1']} | dev RMSE: {r['dev_rmse']:.4f} | baseline: {r['baseline_rmse']:.3f} | "
            f"delta: {r['delta']:+.4f} | pred: {r['pred_path']}"
        )
    print("Target: within ±0.02 of baselines: es=1.357, de=1.328, cn=1.175")


if __name__ == "__main__":
    main()
