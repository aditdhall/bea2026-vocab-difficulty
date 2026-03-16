# BEA 2026 — L1-Aware Vocabulary Difficulty Prediction (Closed Track)

**Team size:** 5

This project implements systems for the **BEA 2026 Shared Task on Vocabulary Difficulty Prediction** in the **closed track only**. The task predicts `GLMM_score` (word difficulty for English learners) across three L1 backgrounds: **Spanish (es)**, **German (de)**, and **Mandarin (cn)**.

## Task Description

Given an English target word, its part-of-speech, a clue, the L1 translation/source word, and L1 context, the system predicts a continuous **GLMM_score**. **Lower GLMM_score = more difficult** for learners; a word with score -2 is harder than one with +2.

## Closed Track Rules (Mandatory)

- **One model per L1** — never mix es/de/cn training data.
- **Encoder-only models only**: `xlm-roberta-base`, `xlm-roberta-large`, `microsoft/mdeberta-v3-base`. No LLMs, no decoders (no GPT, Claude, Gemini, LLaMA).
- **Standard NLP resources allowed**: WordNet, SUBTLEX-US, AoA norms, concreteness ratings, CEFR word lists, Unicode Unihan database.
- **Official closed track baselines to beat:**
  - **es:** RMSE = 1.357  
  - **de:** RMSE = 1.328  
  - **cn:** RMSE = 1.175  

## Data

Data comes from [britishcouncil/bea2026st](https://github.com/britishcouncil/bea2026st).

**CSV paths (same for all tracks):**

| Split | es | de | cn |
|-------|----|----|-----|
| Train | `data/train/es/kvl_shared_task_es_train.csv` | `data/train/de/kvl_shared_task_de_train.csv` | `data/train/cn/kvl_shared_task_cn_train.csv` |
| Dev   | `data/dev/es/kvl_shared_task_es_dev.csv`   | `data/dev/de/kvl_shared_task_de_dev.csv`   | `data/dev/cn/kvl_shared_task_cn_dev.csv`   |

**CSV columns:** `item_id`, `L1`, `en_target_word`, `en_target_pos`, `en_target_clue`, `L1_source_word`, `L1_context`, `GLMM_score`

Items with the same `item_id` across L1 files refer to the same English word in different L1 contexts.

**Submission format:** CSV with columns `item_id`, `prediction`.

## Quick Start

1. **Setup (Colab):** Run `notebooks/00_setup_and_verify.ipynb` to mount Drive, clone repo, install deps, and copy data/upstream from Drive.
2. **Download resources:** `python scripts/download_resources.py`
3. **EDA:** Run `notebooks/01_eda.ipynb`.
4. **Features:** Run `notebooks/02_feature_engineering.ipynb` and `03_feature_selection.ipynb` to build and freeze features.
5. **Experiments:** Run `notebooks/04_experiments.ipynb` (Exp 0–6).
6. **Evaluate:** `python scripts/evaluate.py --pred path/to/predictions.csv --gold data/dev/es/kvl_shared_task_es_dev.csv --l1 es`

## Project Structure

- `configs/` — experiment configuration (YAML).
- `data/` — train/dev CSVs per L1 (gitignored; copied at runtime).
- `features/` — feature pipeline and modules (clue, frequency, cognate, semantic, context).
- `models/` — XGBoost baseline and hybrid transformer.
- `notebooks/` — setup, EDA, feature engineering, selection, experiments, analysis.
- `scripts/` — evaluate, download_resources, freeze_features.
- `results/` — predictions and figures.
- `resources/` — external lexicons (gitignored; downloaded by script).
- `frozen_features.json` — selected feature list after Phase 3.

## Requirements

- Python 3.8+
- PyTorch, Transformers, XGBoost, scikit-learn, pandas, and others (see `requirements.txt`).
- Training is designed for **Google Colab Pro with GPU**.
