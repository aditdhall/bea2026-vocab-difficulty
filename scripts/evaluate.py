"""Standalone evaluation script compatible with official BEA format."""

import argparse
import pandas as pd
import numpy as np
from scipy import stats

CLOSED_BASELINES = {"es": 1.357, "de": 1.328, "cn": 1.175}


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict with rmse, pearson_r, pearson_p."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r, p = stats.pearsonr(y_true, y_pred)
    return {"rmse": rmse, "pearson_r": float(r), "pearson_p": float(p)}


def evaluate_from_files(pred_csv_path: str, gold_csv_path: str, l1: str) -> None:
    """Merge on item_id, compute RMSE and Pearson r, print formatted results."""
    pred_df = pd.read_csv(pred_csv_path)
    gold_df = pd.read_csv(gold_csv_path)
    if "item_id" not in pred_df.columns or "prediction" not in pred_df.columns:
        raise ValueError("Prediction CSV must have columns: item_id, prediction")
    if "item_id" not in gold_df.columns or "GLMM_score" not in gold_df.columns:
        raise ValueError("Gold CSV must have columns: item_id, GLMM_score")
    merged = pred_df[["item_id", "prediction"]].merge(
        gold_df[["item_id", "GLMM_score"]], on="item_id", how="inner"
    )
    if len(merged) == 0:
        raise ValueError("No matching item_id between pred and gold")
    y_true = merged["GLMM_score"].values
    y_pred = merged["prediction"].values
    res = evaluate_predictions(y_true, y_pred)
    baseline = CLOSED_BASELINES.get(l1.lower(), None)
    delta = (res["rmse"] - baseline) if baseline is not None else None
    delta_str = f" | Delta from baseline: {delta:+.4f}" if delta is not None else ""
    print(f"L1: {l1} | RMSE: {res['rmse']:.4f} | Pearson r: {res['pearson_r']:.4f}{delta_str}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BEA 2026 predictions")
    parser.add_argument("--pred", required=True, help="Path to prediction CSV (item_id, prediction)")
    parser.add_argument("--gold", required=True, help="Path to gold CSV (item_id, GLMM_score, ...)")
    parser.add_argument("--l1", required=True, choices=["es", "de", "cn"], help="L1 code")
    args = parser.parse_args()
    evaluate_from_files(args.pred, args.gold, args.l1)


if __name__ == "__main__":
    main()
