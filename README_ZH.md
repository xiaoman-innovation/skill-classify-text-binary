# 文本二分类技能

[**English**](./README.md)

一个 Claude Code 技能，用于端到端的多语言文本二分类模型训练和部署。覆盖 **150+ 种模型变体**，涵盖传统机器学习、深度学习和 Transformer 架构，支持 Optuna 超参数调优、MLflow 实验追踪和一键 API 部署。

---

## 模型覆盖

| 类别 | 代表模型 |
|---|---|
| **传统机器学习** | SVM、逻辑回归、随机森林、多项式朴素贝叶斯 等 |
| **深度学习** | LSTM、BiLSTM、GRU、BiGRU、CNN + Attention 等 |
| **Transformers** | BERT、RoBERTa、DeBERTa、DistilBERT、XLNet 等（同时支持 XLM-RoBERTa、mBERT 等多语言模型） |

每个模型均支持 **baseline**（默认超参数）和 **tuned**（Optuna 优化）两种训练模式。

---

## 流水线

```
step1_analyze.py  →  step2_split.py  →  step3_scheme.py  →  step4_train.py  →  step5_save.py
   数据探索              数据划分              方案生成              模型训练              部署导出
```

| 步骤 | 脚本 | 用途 |
|------|------|------|
| 1 | `step1_analyze.py` | 数据探索、编码检测、类别分布统计、硬件环境检测 |
| 2 | `step2_split.py` | 分层训练/验证/测试集划分 |
| 3 | `step3_scheme.py` | 生成 150+ 候选模型方案 |
| 4 | `step4_train.py` | 训练 + Optuna 超参优化 + MLflow 追踪 |
| 5 | `step5_save.py` | 保存最佳模型，生成 API 部署文件 |

---

## 核心特性

- **GPU 自动检测** — 自动识别 CUDA/MPS/CPU 并回退
- **Optuna 超参数调优** — 每个模型可配置搜索次数
- **MLflow 实验追踪** — 完整的指标记录和产物存储
- **健壮的 CSV 处理** — 自动从 8 种编码中检测正确编码
- **类别不平衡处理** — 类别权重、分层抽样
- **一键部署** — 自动生成 FastAPI/Flask 服务、Dockerfile、requirements.txt

---

## 依赖

```
numpy、pandas、scipy、scikit-learn、nltk、joblib、tqdm、psutil
torch、transformers、tokenizers、peft
optuna、mlflow
fastapi、uvicorn、pydantic
pyyaml、packaging、langdetect
```

---

## 快速开始

```bash
# 步骤 1：分析数据
python scripts/step1_analyze.py \
  --csv data.csv \
  --text-col text \
  --label-col label \
  --output-dir output

# 步骤 2：划分数据
python scripts/step2_split.py \
  --csv data.csv \
  --text-col text \
  --label-col label \
  --output-dir output

# 步骤 3：生成模型方案
python scripts/step3_scheme.py \
  --csv data.csv \
  --text-col text \
  --label-col label \
  --output-dir output

# 步骤 4：训练模型
python scripts/step4_train.py \
  --csv data.csv \
  --text-col text \
  --label-col label \
  --scheme output/model_scheme.json \
  --models logistic_regression,svm_linear,bert-base-uncased \
  --mode both \
  --tune-trials 50 \
  --output-dir output

# 步骤 5：导出最佳模型
python scripts/step5_save.py \
  --results output/training_results.json \
  --output-dir output
```

---

## 部署

一键生成生产就绪的 API 服务：

- `api_server.py` — FastAPI 或 Flask 推理服务
- `requirements.txt` — 固定版本依赖
- `Dockerfile` — 容器化部署
- `monitoring.md` — 监控告警指南

```bash
# 启动 API 服务
uvicorn api_server:app --host 0.0.0.0 --port 8000

# 测试推理
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "这个产品太棒了！"}'
```

---

## 各语言技能

| 语言 | 目录 |
|---|---|
| English 英文 | [`classify-text-binary-en/`](./classify-text-binary-en/) |

---

## 许可证

MIT
