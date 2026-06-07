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

## Pipeline / 流水线

```
step1_analyze.py  →  step2_split.py  →  step3_scheme.py  →  step4_train.py  →  step5_save.py
   数据探索              数据划分              方案生成              模型训练              部署导出
```

| Step | Script | Purpose / 用途 |
|------|--------|---------------|
| 1 | `step1_analyze.py` | Data exploration, encoding detection, class distribution, hardware detection / 数据探索、编码检测、类别分布、硬件检测 |
| 2 | `step2_split.py` | Train/val/test split with stratification / 分层训练/验证/测试集划分 |
| 3 | `step3_scheme.py` | Model scheme generation with 150+ candidates / 生成 150+ 候选模型方案 |
| 4 | `step4_train.py` | Training with Optuna hyperparameter tuning + MLflow tracking / 训练 + Optuna 超参优化 + MLflow 追踪 |
| 5 | `step5_save.py` | Save best model, generate FastAPI/Flask deployment artifacts / 保存最佳模型，生成 API 部署文件 |

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

## Quick Start / 快速开始

```bash
# Step 1: Analyze your dataset / 分析数据
python scripts/step1_analyze.py \
  --csv data.csv \
  --text-col text \
  --label-col label \
  --output-dir output

# Step 2: Split data / 划分数据
python scripts/step2_split.py \
  --csv data.csv \
  --text-col text \
  --label-col label \
  --output-dir output

# Step 3: Generate model schemes / 生成模型方案
python scripts/step3_scheme.py \
  --csv data.csv \
  --text-col text \
  --label-col label \
  --output-dir output

# Step 4: Train models / 训练模型
python scripts/step4_train.py \
  --csv data.csv \
  --text-col text \
  --label-col label \
  --scheme output/model_scheme.json \
  --models logistic_regression,svm_linear,bert-base-uncased \
  --mode both \
  --tune-trials 50 \
  --output-dir output

# Step 5: Export best model / 导出最佳模型
python scripts/step5_save.py \
  --results output/training_results.json \
  --output-dir output
```

---

## Deployment / 部署

Generates a production-ready API server / 生成生产就绪的 API 服务：

- `api_server.py` — FastAPI or Flask inference server / FastAPI 或 Flask 推理服务
- `requirements.txt` — Pinned dependencies / 固定版本依赖
- `Dockerfile` — Containerized deployment / 容器化部署
- `monitoring.md` — Monitoring and alerting guide / 监控告警指南

```bash
# Start the API server / 启动 API 服务
uvicorn api_server:app --host 0.0.0.0 --port 8000

# Test inference / 测试推理
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'
```

---

## License / 许可证

MIT
