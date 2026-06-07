# Text Binary Classification Skill

[**中文**](./README_ZH.md)

A Claude Code skill for end-to-end multilingual text binary classifier training and deployment. Covers **150+ model variants** across traditional machine learning, deep learning, and Transformer architectures — with Optuna hyperparameter tuning, MLflow experiment tracking, and one-click API deployment.

---

## Model Coverage

| Category | Representative Models |
|---|---|
| **Traditional ML** | SVM, Logistic Regression, Random Forest, Multinomial NB, etc. |
| **Deep Learning** | LSTM, BiLSTM, GRU, BiGRU, CNN + Attention, etc. |
| **Transformers** | BERT, RoBERTa, DeBERTa, DistilBERT, XLNet, etc. (multilingual models such as XLM-RoBERTa, mBERT also supported) |

Each model supports **baseline** (default hyperparameters) and **tuned** (Optuna-optimized) training modes.

---

## Pipeline

```
step1_analyze.py  →  step2_split.py  →  step3_scheme.py  →  step4_train.py  →  step5_save.py
  Data Analysis      Data Split         Scheme Gen         Model Training      Deployment Export
```

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `step1_analyze.py` | Data exploration, encoding detection, class distribution, hardware detection |
| 2 | `step2_split.py` | Train/val/test split with stratification |
| 3 | `step3_scheme.py` | Model scheme generation with 150+ candidates |
| 4 | `step4_train.py` | Training with Optuna hyperparameter tuning + MLflow tracking |
| 5 | `step5_save.py` | Save best model, generate API deployment artifacts |

---

## Key Features

- **GPU Auto-detection** — CUDA/MPS/CPU with fallback
- **Optuna Hyperparameter Tuning** — Configurable trials per model
- **MLflow Experiment Tracking** — Full metric logging, artifact storage
- **Robust CSV Handling** — Auto-detects encoding from 8 codecs
- **Class Imbalance Handling** — Class weights, stratified splits
- **One-click Deployment** — FastAPI/Flask server, Dockerfile, requirements.txt

---

## Dependencies

```
numpy, pandas, scipy, scikit-learn, nltk, joblib, tqdm, psutil
torch, transformers, tokenizers, peft
optuna, mlflow
fastapi, uvicorn, pydantic
pyyaml, packaging, langdetect
```

---

## Quick Start

```bash
# Step 1: Analyze your dataset
python scripts/step1_analyze.py \
  --csv data.csv \
  --text-col text \
  --label-col label \
  --output-dir output

# Step 2: Split data
python scripts/step2_split.py \
  --csv data.csv \
  --text-col text \
  --label-col label \
  --output-dir output

# Step 3: Generate model schemes
python scripts/step3_scheme.py \
  --csv data.csv \
  --text-col text \
  --label-col label \
  --output-dir output

# Step 4: Train models
python scripts/step4_train.py \
  --csv data.csv \
  --text-col text \
  --label-col label \
  --scheme output/model_scheme.json \
  --models logistic_regression,svm_linear,bert-base-uncased \
  --mode both \
  --tune-trials 50 \
  --output-dir output

# Step 5: Export best model
python scripts/step5_save.py \
  --results output/training_results.json \
  --output-dir output
```

---

## Deployment

Generates a production-ready API server:

- `api_server.py` — FastAPI or Flask inference server
- `requirements.txt` — Pinned dependencies
- `Dockerfile` — Containerized deployment
- `monitoring.md` — Monitoring and alerting guide

```bash
# Start the API server
uvicorn api_server:app --host 0.0.0.0 --port 8000

# Test inference
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!"}'
```

---

## Language-specific Skills

| Language | Directory |
|---|---|
| English | [`classify-text-binary-en/`](./classify-text-binary-en/) |

---

## License

MIT
