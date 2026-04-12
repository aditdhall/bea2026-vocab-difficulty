# BEA 2026 — L1-Aware Vocabulary Difficulty Prediction (Closed Track)

**Team:** SAAKTH · **Team size:** 5 · **Track:** Closed · **Task:** [BEA 2026 Shared Task on Vocabulary Difficulty Prediction](https://github.com/britishcouncil/bea2026st)

**Authors:** Karthik Mattu, Adit Dhall, Arshad Naguru, Shubh Sehgal, Thejas Gowda, Hakyung Sung · Rochester Institute of Technology

This repository contains our system for predicting English vocabulary difficulty (`GLMM_score`) for learners from three L1 backgrounds: **Spanish (es)**, **German (de)**, and **Mandarin (cn)**. Our approach combines a fine-tuned `xlm-roberta-large` encoder with handcrafted psycholinguistic features, trained with 5-seed ensembling and optional XGBoost blending.

---

## Results

### Test Set (Official Evaluation)

| L1 | Official Baseline | Run 1: Best single seed (xlm-roberta-large) | Run 2: 5-seed ensemble (xlm-roberta-large) | Run 3: Ensemble + XGBoost blend |
|----|:-----------------:|:-------------------------------------------:|:------------------------------------------:|:-------------------------------:|
| Spanish | 1.257 | 1.087 | 1.053 | **1.045** |
| German | 1.258 | 1.000 | 1.012 | **0.994** |
| Mandarin | 1.140 | 0.913 | 0.911 | **0.900** |

All values are RMSE (↓ better). Run 3 is our best submission: a weighted blend of the 5-seed `xlm-roberta-large` ensemble with an XGBoost model trained on psycholinguistic features (blend weights: es=0.8/0.2, de=0.8/0.2, cn=0.9/0.1).

### Dev Set

| L1 | Official Baseline | Best single seed (xlm-roberta-large) | 5-seed ensemble (xlm-roberta-large) | Ensemble + XGBoost blend |
|----|:-----------------:|:------------------------------------:|:-----------------------------------:|:------------------------:|
| Spanish | 1.357 | 1.057 | 1.021 | **0.997** |
| German | 1.328 | 1.010 | 1.013 | **1.002** |
| Mandarin | 1.175 | 0.952 | 0.940 | **0.932** |

---

## Pretrained Checkpoints

Best-seed checkpoints are available on HuggingFace:  
**[aditdhall/bea2026-vocab-difficulty](https://huggingface.co/aditdhall/bea2026-vocab-difficulty)**

| File | L1 | Seed | Dev RMSE |
|------|----|:----:|:--------:|
| `exp4_large_es_seed42.pt` | Spanish | 42 | 1.057 |
| `exp4_large_de_seed789.pt` | German | 789 | 1.010 |
| `exp4_large_cn_seed42.pt` | Mandarin | 42 | 0.952 |

---

## Closed Track Rules

- One model per L1 — training data is never mixed across languages.
- Encoder-only models only: `xlm-roberta-base`, `xlm-roberta-large`, `microsoft/mdeberta-v3-base`.
- Standard NLP resources allowed: WordNet, SUBTLEX-US, AoA norms, concreteness ratings, CEFR word lists, Unicode Unihan database.

---

## Data

Data is not included in this repository. Clone the official shared task repo:

```bash
git clone https://github.com/britishcouncil/bea2026st.git
```

Then copy the train/dev/test CSVs into this repo's `data/` directory:

```
data/
├── train/{es,de,cn}/kvl_shared_task_{l1}_train.csv
├── dev/{es,de,cn}/kvl_shared_task_{l1}_dev.csv
└── test/{es,de,cn}/kvl_shared_task_{l1}_test.csv
```

**CSV columns:** `item_id`, `L1`, `en_target_word`, `en_target_pos`, `en_target_clue`, `L1_source_word`, `L1_context`, `GLMM_score`

---

## External Resources

The following resources must be downloaded and placed in `resources/` before running the feature pipeline. Run:

```bash
python scripts/download_resources.py
```

This will automatically download:
- **SUBTLEX-US** (Brysbaert & New, 2009) — word frequency norms
- **Kuperman AoA** (Kuperman et al., 2012) — age-of-acquisition norms
- **Brysbaert Concreteness** (Brysbaert et al., 2014) — concreteness ratings
- **Oxford 5000 CEFR list** — CEFR level labels
- **Unicode Unihan** — Chinese stroke counts (for cn only)

WordNet is downloaded automatically via NLTK at runtime.

---

## Reproducing Results

All experiments were run on **Google Colab Pro (A100 GPU)**. The canonical reproduction path is `notebooks/BEATest.ipynb`, which retrains from scratch and generates all three submission runs.

### Setup

In a fresh Colab session:

1. Mount Google Drive and clone this repo:
```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/aditdhall/bea2026-vocab-difficulty.git
%cd bea2026-vocab-difficulty
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install xgboost shap
```

> **Important:** Before running, set your Colab runtime to A100 GPU via Runtime → Change runtime type → A100 GPU. Training takes ~3.5 hours on A100.

3. Copy data and resources from Drive (or download fresh as described above).

### Run the submission notebook

Open and run `notebooks/BEATest.ipynb`. It will:
1. Retrain `xlm-roberta-large` hybrid (5 seeds × 3 L1s, ~3.5 hours on A100)
2. Generate three submission runs per L1 (best seed / ensemble / ensemble+XGBoost blend)
3. Package everything into `submission.zip`

### Individual experiment scripts

The `scripts/` directory contains standalone scripts for each experiment:

| Script | Description |
|--------|-------------|
| `colab_04_exp0.py` | Exp 0: Reproduce official baseline (xlm-roberta-base) |
| `colab_04_exp05.py` | Exp 0.5: Target variable scaling |
| `colab_04_exp2.py` | Exp 2: mdeberta-v3-base hybrid |
| `colab_04_exp3.py` | Exp 3: Structured vs default input format |
| `colab_04_exp4.py` | Exp 4: xlm-roberta-base hybrid |
| `colab_04_exp4_large.py` | Exp 4 (large): xlm-roberta-large hybrid, 5 seeds |
| `colab_04_exp5.py` | Exp 5: Hyperparameter search (Spanish only) |
| `colab_04_exp6.py` | Exp 6: Ensemble + XGBoost blend |
| `colab_05_ablation.py` | Ablation study by feature group |
| `colab_05_shap.py` | SHAP feature importance analysis |
| `colab_05_error.py` | Error analysis |
| `colab_05_transfer.py` | Cross-L1 transfer experiments |
| `colab_05_cluster.py` | Difficulty cluster analysis |

Each script is self-contained and can be run via `%run scripts/<script>.py` in Colab after setup.

### Evaluate predictions

```bash
python scripts/evaluate.py \
  --pred results/predictions/my_preds_es.csv \
  --gold data/dev/es/kvl_shared_task_es_dev.csv \
  --l1 es
```

---

## Project Structure

```
bea2026-vocab-difficulty/
├── features/                  # Feature pipeline and modules
│   ├── pipeline.py            # Main FeaturePipeline class
│   ├── clue_orthographic.py   # Clue and orthographic features
│   ├── cognate_crossling.py   # Cognate and cross-lingual features
│   ├── context_features.py    # Context complexity features
│   ├── frequency_norms.py     # SUBTLEX, AoA, concreteness
│   └── semantic_lexical.py    # WordNet, CEFR features
├── models/
│   ├── hybrid_transformer.py  # HybridTransformerModel + training loop
│   └── xgboost_baseline.py    # XGBoost baseline
├── notebooks/
│   ├── BEATest.ipynb          # ← Canonical submission notebook
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_experiments.ipynb
│   └── 05_analysis.ipynb
├── scripts/                   # Standalone experiment scripts (run in Colab)
├── configs/                   # Experiment configuration (YAML)
├── data/                      # Train/dev/test CSVs (gitignored)
├── resources/                 # External lexicons (gitignored)
├── results/                   # Predictions and figures (gitignored)
├── frozen_features.json       # Final frozen feature set
└── requirements.txt
```

---

## Feature Set

Features are frozen in `frozen_features.json`. Active counts differ by L1:

| Feature Group | Features | es | de | cn |
|---------------|----------|:--:|:--:|:--:|
| Clue/orthographic | `reveal_ratio`, `word_length`, `syllable_count`, `has_suffix` | ✓ | ✓ | ✓ |
| Frequency/psycholinguistic | `subtlex_freq`, `aoa_score`, `concreteness` | ✓ | ✓ | ✓ |
| Semantic | `cefr_level`, `wordnet_num_senses`, `wordnet_depth` | ✓ | ✓ | ✓ |
| POS | `pos_NOUN`, `pos_VERB`, `pos_ADJ` | ✓ | ✓ | ✓ |
| Context | `context_length`, `target_position_ratio` | ✓ | ✓ | ✓ |
| Cognate (es/de only) | `edit_distance`, `char_ngram_overlap`, `is_cognate`, `length_ratio` | ✓ | ✓ | ✗ |
| German compound (de only) | `l1_is_compound` | ✗ | ✓ | ✗ |
| Stroke complexity (cn only) | `cn_stroke_complexity` | ✗ | ✗ | ✓ |
| **Total active** | | **19** | **20** | **16** |

---

## Requirements

- Python 3.8+
- PyTorch, Transformers, XGBoost, scikit-learn, pandas, numpy, shap (see `requirements.txt`)
- Training requires **Google Colab Pro with A100 GPU** (~3.5 hours for full submission run)
