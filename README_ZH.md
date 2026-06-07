# 文本二分类技能

[**English**](./README.md)

一个 Claude Code 技能，用于端到端的多语言文本二分类模型训练和部署。覆盖 **150+ 种模型变体**，涵盖传统机器学习、深度学习和 Transformer 架构，支持 Optuna 超参数调优、MLflow 实验追踪和一键 API 部署。

---

## 安装方式

本技能遵循 **Agent Skills** 开放标准，适用于多种 AI 编程工具。

### Claude Code

```bash
# 用户级安装（所有项目可用）
git clone https://github.com/xiaoman-innovation/skill-classify-text-binary.git \
  ~/.claude/skills/classify-text-binary-en/

# 项目级安装（团队共享）
git clone https://github.com/xiaoman-innovation/skill-classify-text-binary.git \
  .claude/skills/classify-text-binary-en/
```

调用方式：输入 `/text-binary-classification`，或直接用自然语言描述任务（如"帮我用这个 CSV 训练一个情感分类器"）。

### Codex CLI (OpenAI)

```bash
# 个人安装
git clone https://github.com/xiaoman-innovation/skill-classify-text-binary.git \
  ~/.codex/skills/classify-text-binary-en/

# 项目共享安装
git clone https://github.com/xiaoman-innovation/skill-classify-text-binary.git \
  .codex/skills/classify-text-binary-en/
```

调用方式：输入 `$text-binary-classification`，或让 Codex 根据任务自动匹配。

### Trae（字节跳动）

```bash
# 项目级安装
git clone https://github.com/xiaoman-innovation/skill-classify-text-binary.git \
  .trae/skills/classify-text-binary-en/
```

也可通过 Trae 界面导入：**设置 → Skills → 从 URL 导入** → 粘贴仓库地址。

---

## 模型覆盖

| 类别 | 代表模型 |
|---|---|
| **传统机器学习** | SVM、逻辑回归、随机森林、多项式朴素贝叶斯 等 |
| **深度学习** | LSTM、BiLSTM、GRU、BiGRU、CNN + Attention 等 |
| **Transformers** | BERT、RoBERTa、DeBERTa、DistilBERT、XLNet 等（同时支持 XLM-RoBERTa、mBERT 等多语言模型） |

每个模型均支持 **baseline**（默认超参数）和 **tuned**（Optuna 优化）两种训练模式。

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

## 许可证

MIT
