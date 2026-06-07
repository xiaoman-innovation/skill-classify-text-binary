# English Text Binary Classification Skill

> 英文文本二分类 — 端到端训练与部署

A Claude Code skill for end-to-end binary text classifier training and deployment. Covers **150+ model variants** across traditional machine learning, deep learning, and Transformer architectures — with Optuna hyperparameter tuning, MLflow experiment tracking, and one-click API deployment.

一个 Claude Code 技能，用于端到端的英文文本二分类模型训练和部署。覆盖 **150+ 种模型变体**，涵盖传统机器学习、深度学习和 Transformer 架构，支持 Optuna 超参数调优、MLflow 实验追踪和一键 API 部署。

---

## Model Coverage / 模型覆盖

| Category / 类别 | Models / 模型 | Embeddings / 嵌入 |
|---|---|---|
| **Traditional ML** 传统机器学习 | SVM (Linear/RBF), Logistic Regression, Random Forest, Multinomial NB | BoW, TF-IDF, GloVe, Word2Vec, fastText |
| **Deep Learning** 深度学习 | CNN, LSTM, BiLSTM, GRU, BiGRU + Attention variants | GloVe, fastText (frozen / fine-tuned) |
| **Transformers** | BERT, RoBERTa, DeBERTa, DistilBERT, ALBERT, ELECTRA, XLNet | Feature extraction, partial freeze, LoRA, full fine-tuning |

Each model supports **baseline** (default hyperparameters) and **tuned** (Optuna-optimized) training modes.

每个模型支持 **baseline**（默认超参数）和 **tuned**（Optuna 优化）两种训练模式。

---

## Key Features / 核心特性

- **GPU Auto-detection** / 自动检测 GPU — CUDA/MPS/CPU with fallback
- **Optuna Hyperparameter Tuning** / 超参数调优 — Configurable trials per model
- **MLflow Experiment Tracking** / 实验追踪 — Full metric logging, artifact storage
- **Robust CSV Handling** / 健壮的 CSV 处理 — Auto-detects encoding from 8 codecs
- **Class Imbalance Handling** / 类别不平衡处理 — Class weights, stratified splits
- **One-click Deployment** / 一键部署 — FastAPI/Flask server, Dockerfile, requirements.txt

---

## Dependencies / 依赖

```
numpy, pandas, scipy, scikit-learn, nltk, joblib, tqdm, psutil
torch, transformers, tokenizers, peft
optuna, mlflow
fastapi, uvicorn, pydantic
pyyaml, packaging, langdetect
```

---

## License / 许可证

MIT
