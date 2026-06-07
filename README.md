# Text Binary Classification Skill

[**中文**](./README_ZH.md)

A Claude Code skill for end-to-end multilingual text binary classifier training and deployment. Covers **150+ model variants** across traditional machine learning, deep learning, and Transformer architectures — with Optuna hyperparameter tuning, MLflow experiment tracking, and one-click API deployment.

---

## Installation

This skill follows the **Agent Skills** open standard and works across multiple AI coding tools.

### Claude Code

```bash
# User-level (available across all projects)
git clone https://github.com/xiaoman-innovation/skill-classify-text-binary.git \
  ~/.claude/skills/classify-text-binary-en/

# Project-level (shared with your team)
git clone https://github.com/xiaoman-innovation/skill-classify-text-binary.git \
  .claude/skills/classify-text-binary-en/
```

Then invoke with: `/text-binary-classification` or describe your task in natural language (e.g. "train a sentiment classifier on my CSV").

### Codex CLI (OpenAI)

```bash
# Personal
git clone https://github.com/xiaoman-innovation/skill-classify-text-binary.git \
  ~/.codex/skills/classify-text-binary-en/

# Project-shared
git clone https://github.com/xiaoman-innovation/skill-classify-text-binary.git \
  .codex/skills/classify-text-binary-en/
```

Then invoke with: `$text-binary-classification` or let Codex auto-detect based on your task.

### Trae (ByteDance)

```bash
# Project-level
git clone https://github.com/xiaoman-innovation/skill-classify-text-binary.git \
  .trae/skills/classify-text-binary-en/
```

Or import via Trae UI: **Settings → Skills → Import from URL** → paste the repo URL.

---

## Model Coverage

| Category | Representative Models |
|---|---|
| **Traditional ML** | SVM, Logistic Regression, Random Forest, Multinomial NB, etc. |
| **Deep Learning** | LSTM, BiLSTM, GRU, BiGRU, CNN + Attention, etc. |
| **Transformers** | BERT, RoBERTa, DeBERTa, DistilBERT, XLNet, etc. (multilingual models such as XLM-RoBERTa, mBERT also supported) |

Each model supports **baseline** (default hyperparameters) and **tuned** (Optuna-optimized) training modes.

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

## License

MIT
