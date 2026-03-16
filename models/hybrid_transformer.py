"""Hybrid model: transformer [CLS] + frozen feature vector -> MLP. Two LRs."""

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


def _load_frozen_feature_names(l1: str, frozen_path: str = "frozen_features.json") -> list[str]:
    """Load feature names for L1 from frozen_features.json."""
    if not os.path.isfile(frozen_path):
        return []
    with open(frozen_path) as f:
        data = json.load(f)
    names = list(data.get("shared", []))
    names += list(data.get("es_de_specific", [])) if l1 in ("es", "de") else []
    names += list(data.get("cn_specific", [])) if l1 == "cn" else []
    return names


class VocabularyDataset(Dataset):
    """Dataset yielding (input_ids, attention_mask, feature_vec, label)."""

    def __init__(self, texts: list[str], feature_matrix: np.ndarray, labels: np.ndarray | None, tokenizer, max_length: int = 128):
        self.texts = texts
        self.features = torch.tensor(feature_matrix, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "features": self.features[i],
        }
        if self.labels is not None:
            out["labels"] = self.labels[i]
        return out


class HybridTransformerModel(nn.Module):
    """Transformer [CLS] + feature vector -> MLP -> scalar. Two param groups for different LRs."""

    def __init__(
        self,
        encoder_name: str = "xlm-roberta-base",
        feature_dim: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder_hidden = self.encoder.config.hidden_size  # 768
        self.feature_dim = feature_dim
        input_dim = self.encoder_hidden + feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask, features):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = enc.last_hidden_state[:, 0, :]  # (B, 768)
        combined = torch.cat([cls_vec, features], dim=1)
        return self.mlp(combined).squeeze(-1)


def train_hybrid(
    model: HybridTransformerModel,
    train_loader,
    dev_loader,
    transformer_lr: float = 1e-5,
    mlp_lr: float = 1e-4,
    epochs: int = 5,
    patience: int = 2,
    device: torch.device | None = None,
    weight_decay: float = 0.0,
):
    """Train with AdamW, linear warmup 10%, early stopping on dev RMSE."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    transformer_params = list(model.encoder.parameters())
    mlp_params = list(model.mlp.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": transformer_params, "lr": transformer_lr, "weight_decay": weight_decay},
            {"params": mlp_params, "lr": mlp_lr, "weight_decay": weight_decay},
        ]
    )
    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_dev_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            loss = nn.functional.mse_loss(logits, labels.squeeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        dev_rmse = _eval_rmse(model, dev_loader, device)
        if dev_rmse < best_dev_rmse:
            best_dev_rmse = dev_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _eval_rmse(model, data_loader, device) -> float:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            preds.append(logits.cpu().numpy())
            labels.append(batch["labels"].numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def predict_hybrid(model: HybridTransformerModel, data_loader, device: torch.device | None = None) -> np.ndarray:
    """Return predictions as numpy array."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    preds = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            preds.append(logits.cpu().numpy())
    return np.concatenate(preds)
