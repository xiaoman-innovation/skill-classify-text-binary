---
name: text-binary-classification
description: >
  English text binary classification end-to-end training and deployment.
  Triggers when the user mentions: text classification, sentiment analysis
  (binary/positive-negative), spam detection, 0/1-label text classification,
  "train a text classifier", "build a binary classification model", "NLP
  classification model", "train a sentiment model with CSV", or any task
  involving classifying English text into two categories.

  Covers traditional ML (BoW/TF-IDF + SVM/LR/RF/NB), deep learning (CNN/LSTM/
  GRU + GloVe/fastText embeddings, frozen/fine-tuned), and Transformers
  (BERT/RoBERTa/DeBERTa/DistilBERT/ALBERT/ELECTRA/XLNet, feature extraction,
  partial freeze, LoRA fine-tuning, full fine-tuning).

  中文触发词：文本分类、情感分析（二分类/正负面）、垃圾邮件检测、0/1标签
  文本分类、"训练文本分类器"、"构建二分类模型"、"NLP分类模型"、
  "用CSV训练情感模型"。

  All training includes baseline defaults + Optuna hyperparameter tuning,
  MLflow experiment tracking, GPU auto-detection, progress bars, and
  deployment artifact generation. Use this skill when the user wants to
  train, tune, or deploy an English text binary classifier, even if they
  don't specify a model or framework.
---

# English Text Binary Classification Skill
# 英文文本二分类技能

End-to-end binary text classifier training and deployment pipeline.
Covers **150+ model variants** across three categories: traditional ML, deep learning, and Transformers.

---

## ⛔ 常见 AI 违规行为 — 执行此技能前必读
## ⛔ Common AI Violations — READ BEFORE EXECUTING THIS SKILL

The following 12 violations occur in >80% of AI executions of this skill. You MUST actively guard against each one:

| # | Violation | Correct Behavior |
|---|-----------|-----------------|
| 1 | **Skipping `Read` when script already outputs to stdout** — AI sees Bash stdout and decides the content is "already visible," so it skips the mandatory Read step. | stdout is NOT a substitute for Read. The all-153 path REQUIRES you to call the Read tool on `model_list.md`, regardless of what stdout shows. |
| 2 | **"As shown above" shortcut** — AI uses phrases like "如上所示" / "详见输出" / "已在对话中列出" instead of actually copying all 153 rows. | You MUST copy every single row verbatim. No summarization. No range abbreviations. No omission of P3 rows. |
| 3 | **AskUserQuestion misuse** — AI calls AskUserQuestion when the skill explicitly says "wait for text input in conversation." This happens most often after the all-153 list output. | After outputting the all-153 model list, you MUST NOT call AskUserQuestion. Just wait silently for the user to type their selection in the conversation. |
| 4 | **Skipping the self-check** — AI outputs the model list but never verifies the last line matches `> **Total: 153 models** (P1 ★ = N, P2 = N — recommended ✓, P3 = N)`. | After every all-153 output, you MUST output the self-check template (see below) and confirm every checkbox is ticked. If any box is unchecked, re-do the step from Read onward. |
| 5 | **Ambiguous Step 2 transition** — With the all-153 path, AI incorrectly assumes "training mode is encoded in the model number" means baseline/tune/both is also skipped. | The model number encodes full_ft/LoRA/feature_extraction/frozen/fine-tuned. The execution MODE (baseline/tune/both) is a SEPARATE question and MUST be asked via AskUserQuestion in Step 2 for every selected model. |
| 6 | **No fallback when AskUserQuestion dialog fails** — AskUserQuestion is a UI modal that can fail to render (network glitch, UI bug, platform compatibility). AI calls AskUserQuestion, user can't see/interact with it, rejects it — AI retries the same call or stalls. This creates a deadlock: the skill mandates AskUserQuestion, but the tool can't deliver. | After ANY AskUserQuestion is rejected (user denies) or fails twice, you MUST fall back to **inline text prompts**: output options as numbered plain text in the conversation body, wait for the user to type their choice. See "AskUserQuestion Fallback Protocol" below. |
| 7 | **Skipping or truncating the smart recommendation table** — AI rushes to AskUserQuestion without first displaying the full P1+P2 recommendation table. Instead, it embeds a partial list inside AskUserQuestion options (e.g. "仅训练 #19 SVM + #21 LR") or shows only P1 models. The user sees nothing to choose from, so they reject the dialog. | Before the select/all AskUserQuestion, you MUST display the COMPLETE smart recommendation table with ALL P1+P2 models in every category (Traditional ML, Deep Learning, Transformer), each with its global #. The table must be shown as inline markdown text in the conversation body — NOT inside AskUserQuestion option labels. |
| 8 | **Wrong category counts in smart recommendation headers** — AI writes headers like "传统机器学习（9 个）" / "深度学习（6 个）" / "Transformer（4 个）" by eyeballing the model list, without cross-referencing actual counts. | After assembling the recommendation table, you MUST verify `[N1]` / `[N2]` / `[N3]` header counts against `step3_scheme.py` stdout's "═══ P1+P2 by category ═══" section which prints ground-truth counts. |
| 9 | **Using `python3` on Windows** — Windows installs Python as `python`, not `python3`. Running `python3` causes exit code 49 and/or opens the Microsoft Store. | On Windows, use `python` (NOT `python3`) for ALL Bash tool calls. This rule has NO exceptions. |
| 10 | **Inline Python to explore CSV files** — AI writes `python -c "import csv; open('file.csv', encoding='utf-8')"` which hardcodes the wrong encoding and crashes on non-UTF-8 files. | NEVER use inline Python to read/explore CSV files. Use `step1_analyze.py` (auto-detects encoding from 8 codecs) or `head`/`wc` bash commands instead. If the script itself fails with encoding error, only then add `--encoding <detected>`. |
| 11 | **Unix-style paths (`/c/Users/...`) passed to Python** — Git Bash auto-translates `/c/Users/...` for shell commands, but Python's `open()`/`pd.read_csv()` bypass the shell and see the literal Unix path, which does not exist on Windows. | Bash commands (ls, head, wc) can use `/c/Users/...` paths. Python scripts MUST use Windows-native paths: `C:\Users\...` or `C:/Users/...`. Before any Python script call, verify `--csv` and `--output-dir` use Windows format. |
| 12 | **Language mismatch — AI replies in wrong language** — User's first message is in English but AI replies in Chinese, or vice versa. AI defaults to the skill document's language instead of detecting the user's language. This is the #1 most frequent violation across all executions. | ⛔ Before ANY output (including the Stage 1 analysis summary), you MUST detect the user's language from their FIRST message in the conversation. See "⛔ LANGUAGE DETECTION" section below. At every user-facing output point, run the ⛔ 2-second language self-check: "What language did the user write their first message in?" → reply in THAT language. |

**⛔ When you encounter text marked with ⛔ in this skill, treat it as non-negotiable. ⛔ directives supersede any heuristic you may have learned from other contexts.**

---

## ⛔ AskUserQuestion Fallback Protocol

AskUserQuestion renders as a UI modal dialog. This dialog can fail to display due to Claude Code UI bugs, network issues, or platform quirks. When this happens, the user sees nothing clickable — they can only type in the chat input.

**Detection rule:** if an AskUserQuestion is rejected (the tool result says "The user doesn't want to proceed with this tool use") **twice for the same question**, the dialog is likely invisible to the user. Do NOT call AskUserQuestion a third time for the same question.

**Fallback procedure:**
1. Output the options as **numbered inline text** in the conversation body, using the exact same labels and descriptions from the AskUserQuestion definition.
2. End with a clear prompt like: "请直接回复选项编号（如 `1`）" / "Reply with the option number (e.g. `1`)."
3. Wait for the user's text reply in the conversation.
4. Parse the user's reply and proceed exactly as if AskUserQuestion had returned that value.

**Example fallback (ZH):**

```
对话框未能显示，以下为文本选项：

[1] 从推荐模型中选择（直接输入编号）→ 如 21,19,22,32,81,82,92,96
[2] 显示全部 153 个模型，我按编号选择

请直接回复选项编号（如 1 或 2）。
```

**This fallback applies to EVERY 🔴 阻断点 in this skill.** Each AskUserQuestion in the pipeline is a potential failure point. The fallback must preserve all option labels, descriptions, and semantics.

---

## ⛔ LANGUAGE DETECTION (READ THIS FIRST — VIOLATION #12)

⛔ **This section is the SINGLE MOST VIOLATED rule in this skill. Read it before you produce ANY output — including any analysis summary, any table, any AskUserQuestion, any inline text. A single line of output in the wrong language is a violation.**

### Detection Rule

Detect the user's language from their **first message** in the conversation (the message that triggered this skill):

| User's first message | ALL subsequent output language |
|---------------------|-------------------------------|
| **Chinese** (contains CJK characters as primary script) | **Chinese** (中文) — every line of text, every table header, every AskUserQuestion label/description/preview, every result summary, every model card, every training plan |
| **English** (Latin script only, or Latin-dominant) | **English** — every line of text, every table header, every AskUserQuestion label/description/preview, every result summary, every model card, every training plan |

This rule applies to EVERY user-facing interaction point:
Stage 1 analysis summary, Stage 2 split options, Stage 3 interactive model selection, Stage 4 training results, Stage 5 final artifact confirmation.

### ⛔ Mandatory Pre-Output Language Self-Check

**Before EVERY user-facing output block** (after each Bash tool call, before each AskUserQuestion, before each inline text prompt), run this 2-second mental check:

```
⛔ Language Self-Check:
1. What language did the user write their first message in? → [EN / ZH]
2. What language am I about to output? → [EN / ZH]
3. Do they match? → [YES / NO — if NO, STOP and rewrite]
```

**If #3 is NO, do NOT output. Rewrite in the correct language.**

### Detection Heuristics

**Chinese-first indicators** (if ANY are true, treat as Chinese):
- Primary content script is CJK (Hanzi): 中文, 文本分类, 情感分析, 训练模型, etc.
- Message contains CJK punctuation: 。、「」
- Message mixes Chinese + English but the INTENT is Chinese (e.g. "用这个 CSV train 一个模型" → Chinese)

**English-first indicators** (all must be true, otherwise default to Chinese if CJK present):
- Primary content script is Latin: "train a model", "use this csv", etc.
- NO CJK characters in the primary message content
- Message may contain English + code/file paths (e.g. "use desktop nlp_test_5000.csv to model" → English)

**Edge cases:**
- Pure file path / command only (e.g. `/classify-text-binary-us`): check the message BODY for language cues. If truly ambiguous, default to English.
- Mixed script with CJK: ALWAYS treat as Chinese (CJK presence overrides Latin)
- User switches language mid-conversation: STICK with the language detected from the FIRST message — do not switch

### ⛔ First-Output Gate

**Before your FIRST text output after this skill is triggered**, you MUST:

1. Re-read the user's first message
2. Identify the primary script (CJK or Latin)
3. Decide: Chinese or English
4. State the decision silently (do NOT output "Detected language: X" — just output in the correct language)
5. Produce ALL subsequent output in that language

**⛔ If you catch yourself about to output in the wrong language, STOP immediately. Delete the draft. Rewrite in the user's language.**

### Common Failure Patterns (DO NOT REPEAT)

| Pattern | Example | Why it happens | Fix |
|---------|---------|---------------|-----|
| Skill doc influence | User writes English, AI replies in Chinese because skill doc is bilingual | AI defaults to skill doc's mixed language instead of detecting user's language | ⛔ The skill doc's language is IRRELEVANT. Only the user's first message matters. |
| Template copy-paste | AI copies ZH template verbatim for an EN user | AI uses template without checking which language variant to use | Always check: "Which language templates should I use for THIS user?" |
| Mid-conversation drift | AI starts in EN, gradually adds ZH phrases | AI forgets the language rule after many tool calls | Re-run the self-check before EVERY output block |
| Mixed AskUserQuestion | AI creates AskUserQuestion with EN question but ZH labels | AI mixes templates from both languages | Use ONLY the language variant matching the user |

**DO NOT default to the language of this skill document. DO NOT mix languages. The user's first-message language overrides everything.**

The templates below are provided in BOTH languages — use ONLY the set matching the user's language. **Before using any template, verify it matches the detected language.**

## 流水线概览 / Pipeline Overview

共 5 个步骤 / 5 stages:

```
Stage 1 (分析+环境) → Stage 2 (划分) → Stage 3 (方案) → Stage 4 (训练) → Stage 5 (保存)
Stage 1 (Analysis+Env) → Stage 2 (Split) → Stage 3 (Scheme) → Stage 4 (Train) → Stage 5 (Save)
       🔴 问答 Q&A            🔴 问答 Q&A          🔴 问答×4 Q&A×4       🔴 问答 Q&A        🔴 问答 Q&A
```
🔴 = 必须调用 AskUserQuestion 的交互步骤，任何情况下不得跳过。
🔴 = Mandatory AskUserQuestion interaction point. Must NOT be skipped under any circumstance.

所有脚本位于 `scripts/` 目录下，可独立运行。
All scripts are in the `scripts/` directory and can run independently.
共享模块 / Shared modules: `utils.py`, `preprocessing.py`, `model_factory.py`,
`mlflow_utils.py`, `report.py`, `deploy.py`.

---

## ⛔ 全局输出格式规则（GLOBAL — ALL STAGES）
## ⛔ Global Output Format Rule

**训练结果汇总表必须使用平铺列名格式，违反此规则的输出将被视为格式错误。**
**Training result summary tables MUST use flat column headers. Any output violating this format is an error.**

**列名格式 / Column format:** `{Dataset}_{Metric} (N={sample_count})` — 全部写在一行表头内 / all in a single header row.

**3-way split example (applies to Stage 4 & Stage 5):**

```
| Model | Mode | Train_Acc (N=3999) | Train_F1 (N=3999) | Train_AUC (N=3999) | Valid_Acc (N=501) | Valid_F1 (N=501) | Valid_AUC (N=501) | Test_Acc (N=500) | Test_F1 (N=500) | Test_AUC (N=500) | Time |
```

**2-way split example:**

```
| Model | Mode | Train_Acc (N=4500) | Train_F1 (N=4500) | Train_AUC (N=4500) | Val_Acc (N=CV) | Val_F1 (N=CV) | Val_AUC (N=CV) | Test_Acc (N=500) | Test_F1 (N=500) | Test_AUC (N=500) | Time |
```

**禁止事项 / Prohibited (never use under any circumstance):**
- ❌ 两行表头 / two-row headers (`group_header`)
- ❌ 合并单元格 / merged cells (`colspan`/`rowspan`)
- ❌ 仅标注 `Acc`/`F1`/`AUC` 而不注明属于哪个数据集 / metrics without dataset prefix
- ❌ Train/Val/Test 作为父行、子行只有指标名的层级结构 / hierarchical header structure

**Python enforcement:** `utils.print_table(group_header=...)` with any non-None value raises `ValueError`.

---

## 中文乱码预防 / UTF-8 Encoding (Windows)

**关键 / Critical**: On Windows, set UTF-8 encoding before running any Python script:

```bash
# Run before every Python script call:
export PYTHONIOENCODING=utf-8
# Or in PowerShell:
$env:PYTHONIOENCODING="utf-8"
```

- 所有脚本在调用时都应加上 `PYTHONIOENCODING=utf-8` 环境变量
- Always prefix python commands with `PYTHONIOENCODING=utf-8`
- 使用 Bash 工具运行命令时，始终在 python 命令前设置此变量

---

## Windows 路径处理（Git Bash / MSYS 环境）
## Windows Path Handling (Git Bash / MSYS)

**问题 / Problem**: Git Bash auto-translates `/c/Users/...` → `C:\Users\...` for shell commands, but Python's `open()` / `pd.read_csv()` bypass the shell and see the literal Unix path, which does not exist on Windows.

- `head -3 /c/Users/zhenp/Desktop/file.csv` → **Bash OK** (shell translates)
- `pd.read_csv('/c/Users/zhenp/Desktop/file.csv')` → **Python FileNotFoundError**

**规则 / Rule**: In Windows, pass **Windows-native paths** to Python scripts:

```
# Correct ✓ — Windows native paths
python script.py --csv "C:\Users\zhenp\Desktop\file.csv"
python script.py --csv "C:/Users/zhenp/Desktop/file.csv"

# Wrong ✗ — Unix-style paths (Python cannot resolve)
python script.py --csv "/c/Users/zhenp/Desktop/file.csv"
```

**AI self-check before every Python call:**
1. Is the current system Windows?
2. Are paths passed to `--csv`, `--output-dir` etc. in `C:\...` or `C:/...` format?
3. If a path came from a Bash tool output, has it been converted to Windows format?

---

## ⛔ Windows Python 命令（禁止 `python3`）/ ⛔ Windows Python Command (no `python3`)

Windows 上 Python 默认安装为 `python`（不是 `python3`）。使用 `python3` 会导致 exit code 49 或弹出 Microsoft Store。

| 平台 | 命令 |
|------|------|
| Windows | **`python`**（永远使用） |
| Linux / macOS | `python3` 或 `python`（优先 `python3`） |

**⛔ 在 Windows 下，所有 Bash 调用必须在 Python 命令前使用 `python`，不得使用 `python3`。** 此规则无例外。

---

## ⛔ 禁止内联 Python 探索 CSV（编码安全）/ ⛔ No Inline Python for CSV Exploration

Bash 工具中编写内联 Python（`python -c "..."`）探索 CSV 文件时，AI 经常硬编码 `encoding='utf-8'`，导致 latin-1 等非 UTF-8 文件解码失败（exit code 1）。

**⛔ 规则：不得使用内联 Python 读取或探索 CSV 文件。** 以下为禁止示例：

```bash
# ✗ 禁止 — 内联 Python 硬编码编码
python -c "import csv; ... open('file.csv', encoding='utf-8')"
python -c "import pandas as pd; pd.read_csv('file.csv')"
```

**替代方案（按优先级）：**

1. **首选**：直接运行 `step1_analyze.py`（不加 `--encoding`，让其自动检测）
   ```bash
   PYTHONIOENCODING=utf-8 python scripts/step1_analyze.py --csv <path> --text-col <name> --label-col <name>
   ```
2. **次选**：使用 `head` / `cut` / `wc` 等 Bash 命令快速查看文件结构
   ```bash
   head -5 /c/Users/.../file.csv
   wc -l /c/Users/.../file.csv
   ```
3. **仅当脚本本身报编码错误时**：才手动指定 `--encoding`
   ```bash
   PYTHONIOENCODING=utf-8 python scripts/step1_analyze.py --csv <path> ... --encoding latin-1
   ```

**原因**：所有 step 脚本内部使用 `utils.read_csv_safe()`，该函数自动尝试 `['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'cp1252', 'ISO-8859-1']` 编码链。内联 Python 跳过了此保护机制。

---

## 如何使用此技能 / How to Use This Skill

Trigger后，按照以下 4 步骤执行。
After triggering, follow the 4-stage pipeline below.

**> 🚫 Core Rule / 核心规则: Every interaction step marked 🔴 or `AskUserQuestion` in any Stage is MANDATORY — must NOT be skipped under any circumstance. After outputting a table/result, you MUST immediately call AskUserQuestion to confirm with the user; do NOT stop at a text description and wait for the user to reply. If AskUserQuestion is rejected twice, trigger the AskUserQuestion Fallback Protocol (see above).**

每个脚本阶段独立运行；如需了解内部细节可以读取脚本文件。
Each script stage runs independently; read the script file for internal details if needed.
脚本会自动将自身加入 `sys.path` 以便直接导入同级模块。
Scripts auto-add themselves to `sys.path` to import sibling modules directly.

---

## ⛔ Stage 0：操作前置检查 / Pre-Flight Check (READ BEFORE EVERY BASH CALL)

**在任何 Python 脚本调用之前，必须完成以下 4 项检查。违反任一项 = 必出错。**

```
┌─────────────────────────────────────────────────────────────┐
│                    ⛔ STAGE 0 CHECKLIST                      │
│                                                             │
│  □ 1. Python cmd: Windows → python / Linux → python3        │
│  □ 2. Path format: Python → Windows native (C:\... or C:/…) │
│  □ 3. Encoding: no inline Python for CSV → use step1_analyze│
│  □ 4. UTF-8 env: export PYTHONIOENCODING=utf-8              │
│                                                             │
│  ⛔ ALL 4 boxes MUST be [x] before running ANY python cmd.  │
└─────────────────────────────────────────────────────────────┘
```

| # | Check | Why | Correct |
|---|-------|-----|---------|
| 1 | **Use `python` not `python3`** | Windows only has `python`; `python3` → exit 49 | `python script.py ...` |
| 2 | **Windows-format paths for Python** | `/c/Users/...` works for Bash, not Python | `--csv "C:\Users\..."` or `C:/Users/...` |
| 3 | **No inline Python for CSV** | `python -c "open('f.csv', encoding='utf-8')"` dies on latin-1/gbk | Run `step1_analyze.py` directly (auto-detects 8 encodings) |
| 4 | **Set PYTHONIOENCODING=utf-8** | Windows console GBK encoding causes garbled output | Prefix every python command with `PYTHONIOENCODING=utf-8` |

**最安全的启动命令模板（Windows）：**

```bash
# 第一步永远是：直接运行 step1_analyze.py，不先探索 CSV
cd "技能的 scripts 目录" && PYTHONIOENCODING=utf-8 python scripts/step1_analyze.py \
  --csv "C:\Users\...\file.csv" --text-col <列名> --label-col <列名> \
  --output-dir "C:\...\output"
```

**如果 step1_analyze.py 报编码错误（罕见），再加 `--encoding`：**

```bash
# step1_analyze.py 内部已自动尝试 8 种编码。仅当它报错时才手动加：
PYTHONIOENCODING=utf-8 python scripts/step1_analyze.py --csv "C:\..." ... --encoding latin-1
```

**⛔ 此 Stage 0 检查在每次新的 Bash 调用前必须在脑中过一遍。违反 #1-#3 中任一项 = 浪费 3-5 轮对话修复可避免的错误。**

---

### Stage 1：数据分析 / Data Analysis

Run `scripts/step1_analyze.py` to analyze the dataset and environment:

```bash
PYTHONIOENCODING=utf-8 python scripts/step1_analyze.py \
  --csv <path> --text-col <name> --label-col <name> \
  [--output-dir output] [--encoding <enc>]
```

Generates `output/analysis.json` and outputs to console. Analysis dimensions:

**Part 1: Environment Analysis / 第 1 部分：环境分析**

1. **OS & Hardware / 操作系统与硬件**：
   - OS type (Linux / Windows / macOS)
   - CPU: physical cores, logical cores, frequency (GHz)
   - RAM: total (GB)
   - GPU: name, count, VRAM (GB), CUDA cores, CUDA support

2. **Storage / 存储**：
   - Free space (GB) / total (GB)
   - Disk type (SSD / HDD / NVMe)
   - Read/write speed (MB/s, via 50MB temp file benchmark)
   - Warns if free space < 10GB (large model downloads may fail)

3. **Python Environment / Python 环境** (against `references/requirements.yaml`):
   - Total required, installed (version-matched)
   - **Missing packages** (not installed)
   - **Outdated packages** (installed but version mismatch)
   - "All dependencies ready" when all pass

**Part 2: Modeling Sample Analysis / 第 2 部分：建模样本分析**

1. **Data Scale & Distribution / 数据规模与分布**：
   - Total samples, class distribution (0/1 counts & ratio), class ratio
   - Text length distribution (mean, median, min/max, stddev, P25/P75/P90/P95/P99)
   - Missing label count

2. **Data Quality & Features / 数据质量与特征**：
   - Missing rate, duplicate rate
   - **Vocabulary Richness / 词汇丰富度**: vocab size, total tokens, Type-Token Ratio (TTR), Hapax Legomena ratio (words appearing only once), repeated word ratio
   - **Syntactic Complexity / 句法复杂度**: avg sentence length (words), avg sentences per text, avg clauses per sentence (heuristic subordinating conjunction detection)

3. **Language Characteristics / 语言特性**：
   - Is English dominant (langdetect preferred, Latin char ratio fallback)
   - **Language distribution**: sample count & ratio per language code (sampled 500 texts)

**Warning Mechanisms / 警告机制**:
- Class imbalance (minority class <5%)
- High missing rate
- Non-English content >30%
- Low TTR (highly repetitive vocabulary, likely narrow-domain text)
- High TTR (extremely diverse vocabulary, potentially broad-domain or noisy text)
- High Hapax ratio (many rare words, data may be sparse)
- Very short/long sentences (social media fragments vs academic/legal text)
- Very low clause count (text may be overly simple/informal)
- Disk free space < 10GB
- Python dependencies missing or version mismatch

If validation fails (wrong column names, labels not 0/1), the script exits with a clear error message.
Non-English >30% triggers a warning but continues.
**Report the analysis summary to the user** (matching the user's language per the Language Detection rule), highlighting all warnings, then proceed to Environment Confirmation below.

---

### 环境确认 / Environment Confirmation

分析结果展示后，根据分析结果与环境要求进行对比，与用户交互确认后才能进入 Stage 2。
After showing analysis results, compare against environment requirements and get user confirmation before Stage 2.

#### Step 1: 环境评估 / Environment Assessment

Call `utils.evaluate_environment(analysis)` to evaluate.

**Pass conditions / 通过条件：**
- Python >= 3.8
- RAM >= 4 GB
- Disk free space >= 5 GB
- Key packages (numpy, pandas, scikit-learn, torch, transformers) installed

**Results display (ZH example):** (use matching language per Language Detection rule)

```
## 环境评估结果

| 检查项 | 要求 | 当前值 | 状态 |
|--------|------|--------|------|
| Python 版本 | >= 3.8 | 3.11.9 | 通过 |
| 内存 | >= 4 GB | 16.0 GB | 通过 |
| 磁盘剩余空间 | >= 5 GB | 167.4 GB | 通过 |
| GPU | 推荐 | RTX 4060 (8GB, 3840 CUDA) | 充足 |
| Python 包关键依赖 | 已安装 | 18/19 | 通过 |

评估结论: 环境满足要求，可以继续。
```

#### Step 2: 不满足要求时的处理 / Handling Unmet Requirements

If `can_proceed == False`, inform the user which conditions are not met:

> **EN**: "The environment does not meet the minimum requirements for training. The following issues must be resolved first:"
> **ZH**: "当前环境不满足建模训练的最低要求，以下问题需要先解决："

列出所有 `blockers`，每条以 `- [BLOCKED]` 开头。 / List blockers, each prefixed with `- [BLOCKED]`.
列出 `warnings`，每条以 `- [WARN]` 开头。 / List warnings, each prefixed with `- [WARN]`.
列出 `recommendations`，每条以 `- [TIP]` 开头。 / List recommendations, each prefixed with `- [TIP]`.

**Do NOT continue to Stage 2.** Ask the user to resolve issues and re-run Stage 1.

#### Step 3: 满足要求 — 确认安装缺失的 Python 包 / Install Missing Packages

If `can_proceed == True`:

> **🚫 Blocking point — You MUST call AskUserQuestion here to ask whether to install missing packages. Do not skip.**

Use **AskUserQuestion** to ask whether to install missing packages.

Only ask this question when `python_env.missing` is non-empty.

> **EN Question**: "Some Python packages are missing. Do you want to install them now?"
> **ZH Question**: "部分 Python 包未安装，是否现在安装？"

| Value | EN Label | ZH Label |
|-------|----------|----------|
| yes | "Yes, install missing packages" | "是，安装缺失的包" |
| no | "No, skip (I'll install manually)" | "否，跳过（我将手动安装）" |

如果用户选择 **yes**：
- 运行 `utils.install_missing_packages(missing_packages)`
- 显示安装结果（installed / failed）

```bash
PYTHONIOENCODING=utf-8 python -c "
import sys; sys.path.insert(0, 'scripts')
from utils import install_missing_packages
result = install_missing_packages(<missing_list>)
"
```

如果有包安装失败，提醒用户手动安装：`pip install <failed_packages>`

#### Step 4: 指定项目目录 / Specify Project Directory

Prompt the user for the project output directory:

> **EN**: "Where should the project output be saved? (Enter a path, or type 'auto' to use output/)"
> **ZH**: "项目输出保存到哪个目录？（直接输入路径，或回复「auto」使用默认 output/）"

- User enters `auto` or blank: use `output/` under the current working directory.
- User enters a path: verify parent exists and is writable; ask whether to create if not.
- All subsequent stages use `--output-dir <user-specified-path>`.

#### Step 5: 确认并进入 Stage 2 / Confirm and Proceed to Stage 2

After package installation and directory setup, output confirmation:

**EN:**
```
## Environment Setup Complete

- Project directory: <path>
- Python packages: all ready
- Ready to proceed to Stage 2: Model Scheme Generation.
```

**ZH:**
```
## 环境配置完成

- 项目目录：<path>
- Python 依赖：已就绪
- 即将进入 Stage 2：模型方案生成。
```

然后直接进入 Stage 2。

---

### Stage 2：样本划分 / Data Split

Run `scripts/step2_split.py` to perform train/validation/test split:

```bash
PYTHONIOENCODING=utf-8 python scripts/step2_split.py \
  --csv <path> --text-col <name> --label-col <name> \
  --split-type <random_2way|column_2way|random_3way|column_3way> \
  [--train-ratio 0.8] [--valid-ratio 0.1] [--test-ratio 0.2] \
  [--split-column <name>] [--train-value <val>] [--valid-value <val>] [--test-value <val>] \
  [--output-dir <dir>] [--encoding <enc>] [--seed 42]
```

Generates `<project_dir>/split_info.json`.

4 split types supported / 脚本支持 4 种划分方式：

| Type | Description / 说明 |
|------|------|
| `random_2way` | Random train-test (default 8:2), stratified / 随机 train-test（默认 8:2），分层保持类别比例 |
| `column_2way` | CSV column value-based train-test / 基于 CSV 列值划分 train-test |
| `random_3way` | Random train-valid-test (default 8:1:1), stratified / 随机 train-valid-test（默认 8:1:1），分层 |
| `column_3way` | CSV column value-based train-valid-test / 基于 CSV 列值划分 train-valid-test |

**Stage 2 Interactive Split Flow / 交互式样本划分流程：**

> **🚫 Blocking point — You MUST call AskUserQuestion here to ask how the data should be split. Do not skip.**

Present these options via AskUserQuestion:

> **EN Question**: "How should the data be split?"
> **ZH Question**: "请选择数据划分方式："

| Value | EN Label | ZH Label |
|-------|----------|----------|
| random_2way | "Random train-test (8:2) — CV on train, test held out" | "随机划分 train-test（8:2）— train 上做 CV，test 保留不参与训练" |
| column_2way | "Use CSV column to split train-test" | "使用 CSV 列值划分 train-test" |
| random_3way | "Random train-valid-test (8:1:1) — tune on train+valid, test held out" | "随机划分 train-valid-test（8:1:1）— train+valid 上调优，test 保留" |
| column_3way | "Use CSV column to split train-valid-test" | "使用 CSV 列值划分 train-valid-test" |

- If user chooses `column_2way` or `column_3way`: ask for the split column name and values for each set.
- If user chooses random split: ask for custom ratios (optional, defaults 8:2 or 8:1:1).

After confirmation, run the script, display split summary (sample counts per set, class distribution), save `split_info.json`.

**All subsequent stages use this split. The test set stays held out throughout training, used only for final comparison. Confirm with user before Stage 5 full-retrain.**

---

### Stage 3：模型方案 / Model Scheme

Run `scripts/step3_scheme.py` to generate model recommendations:

```bash
PYTHONIOENCODING=utf-8 python scripts/step3_scheme.py \
  --analysis <project_dir>/analysis.json \
  [--output-dir <project_dir>]
```

`<project_dir>` is the project directory from Stage 1 (default `output/`).

**Important**: NO web search needed before running this script. Default params come from `references/model_params.md`. Only search the web if that file lacks needed info or the user explicitly asks about specific parameter details.

Generates `<project_dir>/model_scheme.json` with three model categories:

- **A. Traditional ML**: Sparse (Count/TF-IDF/OneHot, 1-gram+2-gram) with SVM(LinearSVC)/SVM(RBF)/LogisticRegression/RandomForest/MultinomialNB; plus Dense (GloVe/Word2Vec/fastText averaged) with SVM/LR/RF
- **B. Deep Learning**: TextCNN, BiLSTM, StackedLSTM, LSTMAttention, BiGRU, StackedGRU, GRUAttention — each with GloVe/Word2Vec/fastText embeddings (frozen + fine-tuned)
- **C. Transformers**: BERT, RoBERTa, DeBERTa, DistilBERT, ALBERT, ELECTRA, XLNet — each supports full fine-tuning, feature extraction, partial freeze, LoRA (PEFT)

Priority assignment via data-driven heuristic rules:
- **P1 (Baseline)**: Best traditional ML models — fast, reliable performance reference
- **P2 (Recommended)**: Best fit for the current data — covers multiple architectures
- **P3 (Exploratory)**: Worth trying if resources permit

**After Stage 2 completes, follow the interactive model selection flow below. All choices are user-driven; P1 is no longer auto-trained by default.**

---

## Stage 3 Interactive Model Selection (Smart Defaults) / Stage 3 交互式模型选择

以下所有 AskUserQuestion 选项、模型卡片和训练计划均有中英文两套模板。
**仅使用与用户语言匹配的版本。**

4 步交互流程：

```
第一步：智能推荐+确认 → 第二步：训练方式 → 第三步：调参配置 → 第四步：确认方案
```

**核心设计**：基于 Stage 1 数据分析结果自动筛选高优先级模型组合，
用户只需确认或微调，无需从 153 个模型中逐一挑选。

**关键规则**：用户第一步确认/调整后的模型，后续每一步对话框中都必须列出这些已选模型，
让用户清楚当前正在为哪些模型做配置。

---

### 第一步：智能推荐 + 用户确认 / Step 1: Smart Recommendation + User Confirmation

运行 `step3_scheme.py` 后，读取 `model_scheme.json`，根据以下启发式规则自动筛选模型：

**筛选规则 / Selection Rules（executed in priority order / 按优先级从高到低执行）：**

> Rules based on 2024-2025 text classification benchmarks:
> - Reusens et al. (2024) "Evaluating text classification: a benchmark study" — comprehensive Trad ML vs Transformer comparison
> - 2024 scikit-learn + TF-IDF benchmarks: Linear SVM > Logistic Regression > Random Forest > Multinomial NB
> - SVM beat CNN/RNN/LSTM on 20 Newsgroups benchmark
> - BERT training ~360× slower than SVM, inference 50-200× slower, accuracy gain only 2-5% (0-3% on clear-topic text)

---

**Rule 1 / 规则 1: Data Size Threshold (most critical decision factor / 最关键的决策依据)**

| Samples / 样本量 | Threshold / 阈值判断 |
|--------|---------|
| **< 1,000** | **Traditional ML only / 传统 ML 独占**. Transformer/DL overfit too easily, excluded. Use only sparse vectorizers (Count/TF-IDF), no dense embeddings. |
| **1,000 – 10,000** | **Transition zone / 过渡区**. Traditional ML still primary. If GPU available, add 1-2 lightweight Transformers (DistilBERT / ALBERT) as reference. |
| **10,000 – 100,000** | **Transformer advantage zone / Transformer 优势区**. Pre-trained models start consistently beating traditional ML. Traditional ML kept as baseline reference; Transformers become primary recommendations. |
| **> 100,000** | **Large-scale zone / 大规模区**. Transformers fully dominant. Traditional ML reduced to strongest baseline only (SVM + TF-IDF bigram). Consider large models and more DL variants. |

---

**Rule 2 / 规则 2: Mandatory Baselines (always included, regardless of sample size / 必选基线，始终包含，不论数据量)**

- SVM (LinearSVC) + TF-IDF (1,2-gram) — Traditional ML text classification ceiling (#1 in multiple 2024 benchmarks), seconds-level training, < 10ms inference
- Logistic Regression + TF-IDF (1,2-gram) — Calibrated probability output, < 2% F1 gap vs SVM

---

**Rule 3 / 规则 3: Class Imbalance (minority < 20%) → Add / 类别不平衡（少数类占比 < 20%）→ 追加**

- Random Forest + TF-IDF (1,2-gram) — Tree models have different inductive bias for imbalanced data; `class_weight='balanced'` enabled by default
- If GPU available and samples >= 5K: + SVM (RBF) + GloVe 300d — RBF kernel significantly outperforms sparse TF-IDF on dense vectors

---

**Rule 4 / 规则 4: Graduated Additions by Sample Size / 按样本量梯度追加**

**4a. Tiny (< 1,000) / 微小样本：**
- Multinomial NB + TF-IDF (1-gram) — Extremely fast (< 1s training), Naive Bayes assumption is an advantage on small data
- No dense embeddings, DL, or Transformers

**4b. Small (1,000 – 5,000) / 小样本：**
- Multinomial NB + Count (1,2-gram) — Discrete count features friendlier to NB
- SVM (RBF) + GloVe 300d — Dense embeddings more expressive than sparse on small data
- No DL/Transformers (insufficient data for stable training)

**4c. Medium (5,000 – 10,000) / 中等样本：**
- SVM (RBF) + GloVe 300d, Logistic Regression + GloVe 300d — Dense embedding variants
- If GPU available + VRAM >= 4GB: DistilBERT (full_ft) — lightweight, only needs 4-6GB VRAM
- If GPU available + VRAM >= 8GB: BERT base (full_ft) — standard text classification reference
- If GPU available: TextCNN + FastText 300d (frozen + fine-tuned), BiLSTM/BiGRU + FastText 300d (frozen) — lightweight DL, fast training, good DL baselines

**4d. Medium-Large (10,000 – 50,000) / 中大规模：**
- Random Forest + GloVe 300d — More data unlocks tree model potential
- Logistic Regression + FastText 300d — Subword coverage helpful for domain text
- If GPU + VRAM >= 4GB: DistilBERT (full_ft), ALBERT base (full_ft)
- If GPU + VRAM >= 8GB: BERT base (full_ft), ELECTRA base (full_ft), RoBERTa base (full_ft)
- DL: TextCNN + GloVe 300d (frozen), BiLSTM + GloVe 300d (frozen)

**4e. Large (50,000 – 100,000) / 大规模：**
- On top of 4d, add:
- Trad ML: SVM (RBF) + Word2Vec 300d, Random Forest + FastText 300d
- DL: BiGRU + GloVe 300d (frozen), LSTM + Attention + GloVe 300d (frozen)
- If GPU >= 12GB: DeBERTa v3 base (full_ft), RoBERTa large (full_ft)

**4f. Very Large (> 100,000) / 超大规模：**
- On top of 4e, add:
- Trad ML trimmed to 3-4 strongest baselines (SVM TF-IDF bigram + SVM GloVe + LR TF-IDF bigram + RF GloVe)
- Transformers expanded: add BERT large, DeBERTa large, XLNet base
- DL: add Stacked LSTM + FastText (fine-tuned), LSTM + Attention + FastText (fine-tuned)

---

**Rule 5 / 规则 5: GPU VRAM Constraints / GPU 显存约束**

| VRAM | Traditional ML | Deep Learning | Transformer |
|------|---------|---------|-------------|
| **No GPU / CPU-only** | All available (recommend 5-8 dense+sparse combos) | Skip (CPU DL training extremely slow) | Skip (BERT CPU training 10-50× slower) |
| **< 4 GB** | All available | TextCNN + frozen embeddings (lightweight) | DistilBERT feature_extraction / LoRA only |
| **4-6 GB** | All available | All frozen embedding variants | DistilBERT / ALBERT / ELECTRA small (full_ft + LoRA) |
| **6-8 GB** | All available | All frozen + fine-tuned | BERT base / RoBERTa base (full_ft, FP16, batch ≤ 16) |
| **8-12 GB** | All available | All | BERT/RoBERTa/ELECTRA/DeBERTa base (full_ft, FP32), XLNet base |
| **>= 12 GB** | All available | All | All base + large models (full_ft, FP32) |

**Emergency VRAM fallback / 显存紧急回退规则：**
- If VRAM < 8GB but samples > 10K: Transformers default to LoRA (~0.1-0.5% params trained, 60-70% VRAM reduction)
- If VRAM < 4GB and Transformer is required: only DistilBERT feature_extraction (fully frozen encoder)

---

**Rule 6 / 规则 6: Text Feature Adaptation / 文本特征适配**

| Feature / 特征 | Condition / 条件 | Recommended Adjustment / 推荐调整 |
|------|------|---------|
| Short texts / 短文本 | Avg length < 30 words | TextCNN ahead of BiLSTM (CNN more effective on short text); MNB performance improved |
| Long texts / 长文本 | Avg length > 100 words | LSTM/GRU + Attention ahead of plain BiLSTM; XLNet ahead of BERT (better long-range dependencies) |
| High vocab diversity / 高词汇多样性 | TTR > 0.7 | FastText embeddings preferred (subword coverage for rare words) |
| Low vocab diversity / 低词汇多样性 | TTR < 0.3 | TF-IDF sparse features sufficient; dense embedding gains limited |
| Non-English > 10% | langdetect | FastText embeddings preferred (better multilingual subword coverage); warn user of data quality issue |
| High Hapax ratio / 高 Hapax 比率 | > 30% words appear only once | Prefer FastText embeddings; skip-gram Word2Vec; Transformer feature extraction mode (freeze bottom layers, avoid overfitting noise) |

---

**Rule 7 / 规则 7: Domain Text Judgment / 领域文本判断**

- **General domain / 通用领域** (news, social media, reviews): Pre-trained model domain match high; Transformer recommendation strength +1
- **Vertical/Technical domain / 垂直/技术领域** (medical, legal, IT tickets, fintech compliance): Pre-trained models may underperform traditional ML (BERT subword tokenization breaks domain terminology)
  - Alert user: traditional ML often matches or beats BERT on domain text
  - If using Transformer: recommend LoRA/partial_ft (preserve general language knowledge + adapt to domain)
  - Prefer FastText embeddings (subword coverage for technical terms)

---

**Rule 8 / 规则 8: Upper-Bound Control / 上限控制**

| Category / 类别 | Min / 最小 | Max / 最大 | Note / 说明 |
|------|------|------|------|
| Traditional ML / 传统 ML | 2 (SVM+LR mandatory baselines / 必选基线) | 10 | Sparse + dense embeddings total / 稀疏 + 稠密嵌入总计 |
| Deep Learning / 深度学习 | 0 (no GPU / samples < 5K) | 6 | 1 per encoder + embedding combo / 每个编码器 + 词向量组合计 1 个 |
| Transformer | 0 (no GPU / samples < 1K) | 5 | 1 per model + training mode combo / 每个模型 + 训练模式组合计 1 个 |
| **Total / 总计** | **4** | **15** | When exceeding limit, trim by priority, keep P1 and P2 first / 超出上限时按优先级裁剪，优先保留 P1 和 P2 |

**Trim priority when exceeding limit / 裁剪优先级（超出上限时从低优先级开始移除）：**
1. Remove P3 exploratory models first / 先移除 P3 探索模型
2. Deduplicate similar models (e.g., both SVM+TF-IDF unigram and bigram → keep bigram) / 同类模型去重
3. Remove redundant dense embedding variants (keep GloVe primary, Word2Vec/FastText secondary) / 移除冗余稠密嵌入变体
4. Finally remove Transformer large variants, keep base / 最后移除 Transformer large 变体保留 base

---

---

⛔ **Mandatory Display Rules (SMART RECOMMENDATION — non-negotiable / 强制展示规则 — 不可绕过):**

**Rule S1 — Must display complete recommendation table / 必须展示完整推荐表:** Before calling the select/all AskUserQuestion, you MUST display the full smart recommendation table with ALL P1+P2 models in the conversation body (grouped by Traditional ML / Deep Learning / Transformer, listing every ✓-marked model in each group). **Do NOT** embed the table inside AskUserQuestion option labels. **Do NOT** show only the P1 subset. **Do NOT** abbreviate with "and N others."

**Rule S2 — Category counts must be correct / 计数必须是正确的:** The group header counts `### Traditional ML ([N1] models)` / `### Deep Learning ([N2] models)` / `### Transformer ([N3] models)` MUST match the P1+P2 model counts per category from `model_scheme.json`. Before filling in counts, verify via either:
  - (A) Run `python -c "import json; ..."` to count models with priority<=2 per category from `model_scheme.json`
  - (B) Read `step3_scheme.py` stdout's "═══ P1+P2 by category ═══" section which gives ground-truth counts

**Rule S3 — Post-display self-check / 展示后自检:** After displaying the smart recommendation table, you MUST output the following self-check template in conversation and verify every box:

```
✅ Smart Recommendation Self-Check:
- [ ] Displayed ALL P1+P2 models (total X: P1=N1, P2=N2)
- [ ] Traditional ML / Deep Learning / Transformer all three groups listed
- [ ] Each group header count [N] matches model_scheme.json category priority<=2 count
- [ ] Every row includes global model number
- [ ] Data diagnostics table displayed
```

**Rule S4 — AskUserQuestion limited to select/all / AskUserQuestion 仅限 select/all:** After displaying the table and completing self-check, call AskUserQuestion with options **only** `select` (choose from recommended) and `all` (show all 153 models). **Do NOT** provide baseline/tune/both options here. **Do NOT** pre-judge which models the user wants.

⛔ **If any self-check checkbox is [ ] unchecked, do NOT proceed to AskUserQuestion. Fix and re-display.**

---

筛选完成后，以文本表格展示推荐结果：

**⛔ 强制规则：智能推荐表的每一行必须包含全局编号（# 列）。** 编号从 `step3_scheme.py` 脚本输出的 `#` 列中获取。脚本已确保输出包含编号列，你只需要照原样使用。**不得展示无编号列的推荐表。** 此规则无例外。

**ZH template:**

```
## 智能模型推荐

💡 直接输入全局编号选择模型，如 `14,19,21,22,23,81,92`。支持逗号分隔和范围选取。

### 数据诊断
| 维度 | 数值 | 判断 |
|------|------|------|
| 样本量 | [N] | [微小/小/中等/中大规模/大规模/超大] — [对应规则 4a-4f] |
| 平均文本长度 | [L] 词 | [短文本/中等/长文本] — [对应编码器偏好] |
| 词汇量 / TTR | [V] 词 / [TTR] | [低多样性/正常/高多样性] — [对应嵌入偏好] |
| GPU | [名称 + 显存]（或"仅 CPU"） | VRAM [档位] — [可用模型级别] |
| 类别比 | [ratio] | [平衡/轻度不平衡/严重不平衡] — [追加 RF/不追加] |
| 语言 | [英语占比]% 英语 | [纯净/混合] — [优先 FastText/标准] |
| Hapax 比率 | [H%] | [正常/偏高] — [冻结嵌入/特征提取优先] |

### 传统机器学习（[N1] 个）
| 编号 | 优先级 | 模型 | 向量化器 | 推荐理由 |
|------|--------|------|---------|---------|
| 19 | P2 | SVM (LinearSVC) | TF-IDF (1,2-gram) | 多项 2024 基准第一；快速可靠 |
| 21 | P1 ★ | Logistic Regression | TF-IDF (1,2-gram) | 校准概率输出，与 SVM 差距 < 2% F1 |
| 22 | P2 | Random Forest | TF-IDF (1,2-gram) | 类别不平衡，树模型不同归纳偏置 |
| ... | ... | ... | ... | ... |

### 深度学习（[N2] 个）
| 编号 | 优先级 | 编码器 | 词向量 | 推荐理由 |
|------|--------|--------|-------|---------|
| 81 | P2 | TextCNN | FastText 300d (微调) | CNN 擅长短文本，参数高效，VRAM 友好 |
| 82 | P2 | BiLSTM | FastText 300d (微调) | 序列建模标准基线 |
| ... | ... | ... | ... | ... |

### Transformer（[N3] 个）
| 编号 | 优先级 | 模型 | 训练方式 | 推荐理由 |
|------|--------|------|---------|---------|
| 92 | P2 | RoBERTa base (125M) | full_ft | 文本分类标准参照，VRAM 8GB 可跑 |
| 96 | P2 | DeBERTa v3 base (140M) | full_ft | 解耦注意力，文本分类常达最优 |
| ... | ... | ... | ... | ... |

> 💡 推荐理由基于 2024-2025 年多项基准研究。请在下方选项中选择下一步操作。
```

**EN template:**

**⛔ Mandatory rule: Every row of the smart recommendation table MUST include the global model number (# column).** Numbers come from the `#` column output by `step3_scheme.py`. The script now guarantees this column is present — use it as-is. **Never present a recommendation table without the ID column.** No exceptions.

```
## Smart Model Recommendation

💡 Enter global model numbers to select, e.g. `14,19,21,22,23,81,92`. Supports comma-separated and ranges.

### Data Diagnostics
| Dimension | Value | Assessment |
|-----------|-------|------------|
| Samples | [N] | [tiny/small/medium/medium-large/large/very large] — [rule 4a-4f] |
| Avg text length | [L] words | [short/medium/long] — [encoder preference] |
| Vocab / TTR | [V] words / [TTR] | [low-diversity/normal/high-diversity] — [embedding preference] |
| GPU | [name + VRAM] (or "CPU-only") | VRAM [tier] — [available model level] |
| Class ratio | [ratio] | [balanced/mild imbalance/severe imbalance] — [+RF / skip] |
| Language | [EN%]% English | [pure/mixed] — [prefer FastText/standard] |
| Hapax ratio | [H%] | [normal/high] — [freeze embeddings/feature extraction preferred] |

### Traditional ML ([N1] models)
| # | Priority | Model | Vectorizer | Why |
|---|----------|-------|-----------|-----|
| 19 | P2 | SVM (LinearSVC) | TF-IDF (1,2-gram) | #1 in multiple 2024 benchmarks; fast, reliable |
| 21 | P1 ★ | Logistic Regression | TF-IDF (1,2-gram) | Calibrated probabilities; < 2% F1 gap vs SVM |
| 22 | P2 | Random Forest | TF-IDF (1,2-gram) | Class imbalance — different inductive bias |
| ... | ... | ... | ... | ... |

### Deep Learning ([N2] models)
| # | Priority | Encoder | Embedding | Why |
|---|----------|---------|----------|-----|
| 81 | P2 | TextCNN | FastText 300d (fine-tuned) | CNN excels at short texts; VRAM-efficient |
| 82 | P2 | BiLSTM | FastText 300d (fine-tuned) | Standard sequence modeling baseline |
| ... | ... | ... | ... | ... |

### Transformer ([N3] models)
| # | Priority | Model | Training Mode | Why |
|---|----------|-------|-------------|-----|
| 92 | P2 | RoBERTa base (125M) | full_ft | Standard reference for text classification; 8GB VRAM |
| 96 | P2 | DeBERTa v3 base (140M) | full_ft | Disentangled attention, often SOTA |
| ... | ... | ... | ... | ... |

> 💡 Recommendations based on 2024-2025 benchmark studies. Use the options below to proceed.
```

---

> **🚫 阻断点 — 显示完智能推荐表格后，你必须在此处调用 AskUserQuestion 让用户选择 select 或 all，不得跳过。若 AskUserQuestion 被拒两次，立即执行 Fallback Protocol 切换为文本选项。**

然后使用 **AskUserQuestion** 确认：

> **ZH Question**: "以上推荐方案是否合适？"
> **EN Question**: "Does the recommendation above look good?"

| Value | ZH Label | EN Label |
|-------|----------|----------|
| select | "从推荐模型中选择（直接输入编号）" | "Select from recommended models (enter numbers directly)" |
| all | "显示全部 153 个模型，我按编号选择" | "Show all 153 models for manual selection by number" |

- **select**: User enters global model numbers directly from the smart recommendation table (e.g. `14,19,21,22,23,81,82,92,96`). **Do NOT use AskUserQuestion** — just wait silently for the user to type their selection in the conversation. Parse the input, confirm, then proceed to Step 2 (training mode).
- **all**: Display all 153 models for manual selection. **After displaying all models, do NOT use AskUserQuestion — just wait silently for the user to type model numbers.** Procedure:

  Run `step3_scheme.py --list-all <path to model_scheme.json> --output-dir <project_dir>`.

  ⛔ **Mandatory Display Rules (STRUCTURAL — non-negotiable / 强制展示规则 — 不可绕过):**

  The script outputs the full model list to stdout between `<!-- FULL_MODEL_LIST_START -->` and `<!-- FULL_MODEL_LIST_END -->`, and also saves it to `<project_dir>/model_list.md`.

  **⛔ Critical Warning / 重要警告: Bash stdout output ≠ you have displayed it. Stdout is visible only to you, not to the user. You MUST use the Read tool to read the file separately and copy its contents into the conversation.**

  **⛔ Step 1/3 — Read / 步骤 1/3:** After running the script, use the Read tool on `<project_dir>/model_list.md`. Even if Bash stdout already showed the content, you MUST still call Read.
  **⛔ Step 2/3 — Copy / 步骤 2/3:** Copy ALL content between `<!-- FULL_MODEL_LIST_START -->` and `<!-- FULL_MODEL_LIST_END -->` line-by-line into the conversation. No merging, abbreviation, or omission of P3 rows.
  **⛔ Step 3/3 — Self-Check / 步骤 3/3:** After copying, output the ⛔ mandatory self-check template below and confirm every box. Any unchecked box = MUST redo from Step 1.

  **⛔ Prohibited (never do / 禁止，任何情况下不得使用):**
  - Use phrases like "as shown above" / "如上所示" / "see output above" instead of actual display
  - Show only recommended model subset (P1+P2) and omit P3
  - Merge rows, abbreviate, or use range notation instead of listing every row
  - Rely on Bash tool output as the display mechanism (user may not see it)
  - Skip the self-check template or output an incomplete one

  **⛔ No exceptions. Not outputting all 153 model rows in full = violation. Not outputting the self-check template = violation.**

  格式如下：

  💡 推荐标注说明：P1 ★ + P2 = 智能推荐（✓），P3 = 探索（—）。推荐总数显示在表格末尾。

  **ZH template（全部模型展示）:**

  \`\`\`
  ## 全部 153 个模型（按模块分类）

  💡 直接输入编号选择模型，如 \`19,21,90,92,96,98\`。支持逗号分隔和范围选取（如 \`1-5,8,12-15\`）。

  ### A. 传统机器学习 — 稀疏特征（23个）
  | 编号 | 推荐 | 模型 | 向量化器 | 优先级 |
  |------|------|------|---------|--------|
  | 1 | — | SVM (LinearSVC) | Count (1-gram) | P3 |
  | 2 | — | SVM (RBF kernel) | Count (1-gram) | P3 |
  | 3 | — | Logistic Regression | Count (1-gram) | P3 |
  | 4 | — | Random Forest | Count (1-gram) | P3 |
  | 5 | — | Multinomial NB | Count (1-gram) | P3 |
  | 6 | — | SVM (LinearSVC) | Count (1,2-gram) | P3 |
  | 7 | — | SVM (RBF kernel) | Count (1,2-gram) | P3 |
  | 8 | — | Logistic Regression | Count (1,2-gram) | P3 |
  | 9 | — | Random Forest | Count (1,2-gram) | P3 |
  | 10 | — | Multinomial NB | Count (1,2-gram) | P3 |
  | 11 | — | SVM (LinearSVC) | OneHot (1-gram) | P3 |
  | 12 | — | Logistic Regression | OneHot (1-gram) | P3 |
  | 13 | — | Multinomial NB | OneHot (1-gram) | P3 |
  | 14 | — | SVM (LinearSVC) | TF-IDF (1-gram) | P3 |
  | 15 | — | SVM (RBF kernel) | TF-IDF (1-gram) | P3 |
  | 16 | — | Logistic Regression | TF-IDF (1-gram) | P3 |
  | 17 | — | Random Forest | TF-IDF (1-gram) | P3 |
  | 18 | — | Multinomial NB | TF-IDF (1-gram) | P3 |
  | **19** | **✓** | **SVM (LinearSVC)** | **TF-IDF (1,2-gram)** | **P2** |
  | 20 | — | SVM (RBF kernel) | TF-IDF (1,2-gram) | P3 |
  | **21** | **✓** | **Logistic Regression** | **TF-IDF (1,2-gram)** | **P1 \u2605** |
  | **22** | **✓** | **Random Forest** | **TF-IDF (1,2-gram)** | **P2** |
  | 23 | — | Multinomial NB | TF-IDF (1,2-gram) | P3 |

  ### B. 传统机器学习 — 稠密嵌入（12个）
  | 编号 | 推荐 | 模型 | 词嵌入 | 优先级 |
  |------|------|------|--------|--------|
  | 24 | — | SVM (LinearSVC) | GloVe 300d | P3 |
  | **25** | **✓** | **SVM (RBF kernel)** | **GloVe 300d** | **P2** |
  | **26** | **✓** | **Logistic Regression** | **GloVe 300d** | **P2** |
  | 27 | — | Random Forest | GloVe 300d | P3 |
  | 28 | — | SVM (LinearSVC) | Word2Vec 300d | P3 |
  | 29 | — | SVM (RBF kernel) | Word2Vec 300d | P3 |
  | 30 | — | Logistic Regression | Word2Vec 300d | P3 |
  | 31 | — | Random Forest | Word2Vec 300d | P3 |
  | 32 | — | SVM (LinearSVC) | FastText 300d | P3 |
  | 33 | — | SVM (RBF kernel) | FastText 300d | P3 |
  | 34 | — | Logistic Regression | FastText 300d | P3 |
  | 35 | — | Random Forest | FastText 300d | P3 |

  ### C. 深度学习（54个）
  | 编号 | 推荐 | 编码器 | 词嵌入 | 嵌入方式 | 优先级 |
  |------|------|--------|--------|---------|--------|
  | 36 | — | TextCNN | GloVe 300d | 冻结 | P3 |
  | 37 | BiLSTM | GloVe 300d | 冻结 | P3 |
  | 38 | LSTM | GloVe 300d | 冻结 | P3 |
  | 39 | Stacked LSTM | GloVe 300d | 冻结 | P3 |
  | 40 | LSTM + Attention | GloVe 300d | 冻结 | P3 |
  | **41** | **BiGRU** | **GloVe 300d** | **冻结** | **P2** |
  | 42 | GRU | GloVe 300d | 冻结 | P3 |
  | 43 | Stacked GRU | GloVe 300d | 冻结 | P3 |
  | 44 | GRU + Attention | GloVe 300d | 冻结 | P3 |
  | **45** | **TextCNN** | **GloVe 300d** | **微调** | **P2** |
  | **46** | **BiLSTM** | **GloVe 300d** | **微调** | **P2** |
  | 47 | LSTM | GloVe 300d | 微调 | P3 |
  | 48 | Stacked LSTM | GloVe 300d | 微调 | P3 |
  | 49 | LSTM + Attention | GloVe 300d | 微调 | P3 |
  | 50 | BiGRU | GloVe 300d | 微调 | P3 |
  | 51 | GRU | GloVe 300d | 微调 | P3 |
  | 52 | Stacked GRU | GloVe 300d | 微调 | P3 |
  | **53** | **GRU + Attention** | **GloVe 300d** | **微调** | **P2** |
  | 54 | TextCNN | Word2Vec 300d | 冻结 | P3 |
  | 55 | BiLSTM | Word2Vec 300d | 冻结 | P3 |
  | **56** | **LSTM** | **Word2Vec 300d** | **冻结** | **P2** |
  | 57 | Stacked LSTM | Word2Vec 300d | 冻结 | P3 |
  | 58 | LSTM + Attention | Word2Vec 300d | 冻结 | P3 |
  | 59 | BiGRU | Word2Vec 300d | 冻结 | P3 |
  | 60 | GRU | Word2Vec 300d | 冻结 | P3 |
  | 61 | Stacked GRU | Word2Vec 300d | 冻结 | P3 |
  | 62 | GRU + Attention | Word2Vec 300d | 冻结 | P3 |
  | 63 | TextCNN | Word2Vec 300d | 微调 | P3 |
  | 64 | BiLSTM | Word2Vec 300d | 微调 | P3 |
  | 65 | LSTM | Word2Vec 300d | 微调 | P3 |
  | 66 | Stacked LSTM | Word2Vec 300d | 微调 | P3 |
  | 67 | LSTM + Attention | Word2Vec 300d | 微调 | P3 |
  | 68 | BiGRU | Word2Vec 300d | 微调 | P3 |
  | 69 | GRU | Word2Vec 300d | 微调 | P3 |
  | 70 | Stacked GRU | Word2Vec 300d | 微调 | P3 |
  | 71 | GRU + Attention | Word2Vec 300d | 微调 | P3 |
  | **72** | **TextCNN** | **FastText 300d** | **冻结** | **P2** |
  | 73 | BiLSTM | FastText 300d | 冻结 | P3 |
  | 74 | LSTM | FastText 300d | 冻结 | P3 |
  | 75 | Stacked LSTM | FastText 300d | 冻结 | P3 |
  | 76 | LSTM + Attention | FastText 300d | 冻结 | P3 |
  | 77 | BiGRU | FastText 300d | 冻结 | P3 |
  | 78 | GRU | FastText 300d | 冻结 | P3 |
  | 79 | Stacked GRU | FastText 300d | 冻结 | P3 |
  | 80 | GRU + Attention | FastText 300d | 冻结 | P3 |
  | 81 | TextCNN | FastText 300d | 微调 | P3 |
  | 82 | BiLSTM | FastText 300d | 微调 | P3 |
  | 83 | LSTM | FastText 300d | 微调 | P3 |
  | 84 | Stacked LSTM | FastText 300d | 微调 | P3 |
  | 85 | LSTM + Attention | FastText 300d | 微调 | P3 |
  | 86 | BiGRU | FastText 300d | 微调 | P3 |
  | 87 | GRU | FastText 300d | 微调 | P3 |
  | 88 | Stacked GRU | FastText 300d | 微调 | P3 |
  | 89 | GRU + Attention | FastText 300d | 微调 | P3 |

  ### D. Transformer — Full Fine-tuning（16个）
  | 编号 | 推荐 | 模型 | 参数量 | 优先级 |
  |------|------|------|--------|--------|
  | **90** | **✓** | **BERT base (uncased)** | 110M | **P2** |
  | 91 | — | BERT large (uncased) | 340M | P3 |
  | **92** | **RoBERTa base** | 125M | **P2** |
  | 93 | — | RoBERTa large | 355M | P3 |
  | 94 | — | DeBERTa base | 140M | P3 |
  | 95 | — | DeBERTa large | 400M | P3 |
  | **96** | **DeBERTa v3 base** | 140M | **P2** |
  | 97 | — | DeBERTa v3 large | 400M | P3 |
  | **98** | **DistilBERT base** | 66M | **P2** |
  | **99** | **ALBERT base v2** | 12M | **P2** |
  | 100 | — | ALBERT large v2 | 18M | P3 |
  | 101 | — | ELECTRA small | 13M | P3 |
  | **102** | **ELECTRA base** | 110M | **P2** |
  | 103 | — | ELECTRA large | 335M | P3 |
  | **104** | **XLNet base (cased)** | 110M | **P2** |
  | 105 | — | XLNet large (cased) | 340M | P3 |

  ### D. Transformer — Feature Extraction（16个）
  | 编号 | 推荐 | 模型 | 优先级 |
  |------|------|------|--------|
  | 106 | — | — | BERT base — feature extraction | P3 |
  | 107 | BERT large — feature extraction | P3 |
  | 108 | RoBERTa base — feature extraction | P3 |
  | 109 | RoBERTa large — feature extraction | P3 |
  | 110 | DeBERTa base — feature extraction | P3 |
  | 111 | DeBERTa large — feature extraction | P3 |
  | 112 | DeBERTa v3 base — feature extraction | P3 |
  | 113 | DeBERTa v3 large — feature extraction | P3 |
  | 114 | DistilBERT base — feature extraction | P3 |
  | 115 | ALBERT base v2 — feature extraction | P3 |
  | 116 | ALBERT large v2 — feature extraction | P3 |
  | 117 | ELECTRA small — feature extraction | P3 |
  | 118 | ELECTRA base — feature extraction | P3 |
  | 119 | ELECTRA large — feature extraction | P3 |
  | 120 | XLNet base — feature extraction | P3 |
  | 121 | XLNet large — feature extraction | P3 |

  ### D. Transformer — Partial Fine-tuning（16个）
  | 编号 | 推荐 | 模型 | 优先级 |
  |------|------|------|--------|
  | 122 | — | — | BERT base — partial fine-tuning | P3 |
  | 123 | BERT large — partial fine-tuning | P3 |
  | 124 | RoBERTa base — partial fine-tuning | P3 |
  | 125 | RoBERTa large — partial fine-tuning | P3 |
  | 126 | DeBERTa base — partial fine-tuning | P3 |
  | 127 | DeBERTa large — partial fine-tuning | P3 |
  | 128 | DeBERTa v3 base — partial fine-tuning | P3 |
  | 129 | DeBERTa v3 large — partial fine-tuning | P3 |
  | 130 | DistilBERT base — partial fine-tuning | P3 |
  | 131 | ALBERT base v2 — partial fine-tuning | P3 |
  | 132 | ALBERT large v2 — partial fine-tuning | P3 |
  | 133 | ELECTRA small — partial fine-tuning | P3 |
  | 134 | ELECTRA base — partial fine-tuning | P3 |
  | 135 | ELECTRA large — partial fine-tuning | P3 |
  | 136 | XLNet base — partial fine-tuning | P3 |
  | 137 | XLNet large — partial fine-tuning | P3 |

  ### D. Transformer — LoRA / PEFT（16个）
  | 编号 | 推荐 | 模型 | 优先级 |
  |------|------|------|--------|
  | 138 | — | — | BERT base — LoRA | P3 |
  | 139 | BERT large — LoRA | P3 |
  | 140 | RoBERTa base — LoRA | P3 |
  | 141 | RoBERTa large — LoRA | P3 |
  | 142 | DeBERTa base — LoRA | P3 |
  | 143 | DeBERTa large — LoRA | P3 |
  | 144 | DeBERTa v3 base — LoRA | P3 |
  | 145 | DeBERTa v3 large — LoRA | P3 |
  | 146 | DistilBERT base — LoRA | P3 |
  | 147 | ALBERT base v2 — LoRA | P3 |
  | 148 | ALBERT large v2 — LoRA | P3 |
  | 149 | ELECTRA small — LoRA | P3 |
  | 150 | ELECTRA base — LoRA | P3 |
  | 151 | ELECTRA large — LoRA | P3 |
  | 152 | XLNet base — LoRA | P3 |
  | 153 | XLNet large — LoRA | P3 |

  > **总计：153 个模型**（P1 ★ = 1，P2 = 17 — 推荐 ✓，P3 = 135）
  \`\`\`

  **EN template:**

  \`\`\`
  ## All 153 Models (by Module)

  💡 Enter numbers to select models, e.g. \`19,21,90,92,96,98\`. Supports comma-separated and ranges (e.g. \`1-5,8,12-15\`).
  💡 Models marked **✓** are smart-recommended (P1+P2).

  ### A. Traditional ML — Sparse Features (23 models)
  | # | Rec | Model | Vectorizer | Priority |
  |---|-----|-------|-----------|----------|
  | 1 | — | SVM (LinearSVC) | Count (1-gram) | P3 |
  | 2 | — | SVM (RBF kernel) | Count (1-gram) | P3 |
  | 3 | — | Logistic Regression | Count (1-gram) | P3 |
  | 4 | — | Random Forest | Count (1-gram) | P3 |
  | 5 | — | Multinomial NB | Count (1-gram) | P3 |
  | 6 | — | SVM (LinearSVC) | Count (1,2-gram) | P3 |
  | 7 | — | SVM (RBF kernel) | Count (1,2-gram) | P3 |
  | 8 | — | Logistic Regression | Count (1,2-gram) | P3 |
  | 9 | — | Random Forest | Count (1,2-gram) | P3 |
  | 10 | — | Multinomial NB | Count (1,2-gram) | P3 |
  | 11 | — | SVM (LinearSVC) | OneHot (1-gram) | P3 |
  | 12 | — | Logistic Regression | OneHot (1-gram) | P3 |
  | 13 | — | Multinomial NB | OneHot (1-gram) | P3 |
  | 14 | — | SVM (LinearSVC) | TF-IDF (1-gram) | P3 |
  | 15 | — | SVM (RBF kernel) | TF-IDF (1-gram) | P3 |
  | 16 | — | Logistic Regression | TF-IDF (1-gram) | P3 |
  | 17 | — | Random Forest | TF-IDF (1-gram) | P3 |
  | 18 | — | Multinomial NB | TF-IDF (1-gram) | P3 |
  | **19** | **✓** | **SVM (LinearSVC)** | **TF-IDF (1,2-gram)** | **P2** |
  | 20 | — | SVM (RBF kernel) | TF-IDF (1,2-gram) | P3 |
  | **21** | **✓** | **Logistic Regression** | **TF-IDF (1,2-gram)** | **P1 \u2605** |
  | **22** | **✓** | **Random Forest** | **TF-IDF (1,2-gram)** | **P2** |
  | 23 | — | Multinomial NB | TF-IDF (1,2-gram) | P3 |

  ### B. Traditional ML — Dense Embeddings (12 models)
  | # | Rec | Model | Embedding | Priority |
  |---|-----|-------|----------|----------|
  | 24 | — | SVM (LinearSVC) | GloVe 300d | P3 |
  | **25** | **✓** | **SVM (RBF kernel)** | **GloVe 300d** | **P2** |
  | **26** | **✓** | **Logistic Regression** | **GloVe 300d** | **P2** |
  | 27 | — | Random Forest | GloVe 300d | P3 |
  | 28 | — | SVM (LinearSVC) | Word2Vec 300d | P3 |
  | 29 | — | SVM (RBF kernel) | Word2Vec 300d | P3 |
  | 30 | — | Logistic Regression | Word2Vec 300d | P3 |
  | 31 | — | Random Forest | Word2Vec 300d | P3 |
  | 32 | — | SVM (LinearSVC) | FastText 300d | P3 |
  | 33 | — | SVM (RBF kernel) | FastText 300d | P3 |
  | 34 | — | Logistic Regression | FastText 300d | P3 |
  | 35 | — | Random Forest | FastText 300d | P3 |

  ### C. Deep Learning (54 models)
  | # | Rec | Encoder | Embedding | Mode | Priority |
  |---|-----|---------|----------|------|----------|
  | 36 | — | TextCNN | GloVe 300d | frozen | P3 |
  | 37 | BiLSTM | GloVe 300d | frozen | P3 |
  | 38 | LSTM | GloVe 300d | frozen | P3 |
  | 39 | Stacked LSTM | GloVe 300d | frozen | P3 |
  | 40 | LSTM + Attention | GloVe 300d | frozen | P3 |
  | **41** | **BiGRU** | **GloVe 300d** | **frozen** | **P2** |
  | 42 | GRU | GloVe 300d | frozen | P3 |
  | 43 | Stacked GRU | GloVe 300d | frozen | P3 |
  | 44 | GRU + Attention | GloVe 300d | frozen | P3 |
  | **45** | **TextCNN** | **GloVe 300d** | **fine-tuned** | **P2** |
  | **46** | **BiLSTM** | **GloVe 300d** | **fine-tuned** | **P2** |
  | 47 | LSTM | GloVe 300d | fine-tuned | P3 |
  | 48 | Stacked LSTM | GloVe 300d | fine-tuned | P3 |
  | 49 | LSTM + Attention | GloVe 300d | fine-tuned | P3 |
  | 50 | BiGRU | GloVe 300d | fine-tuned | P3 |
  | 51 | GRU | GloVe 300d | fine-tuned | P3 |
  | 52 | Stacked GRU | GloVe 300d | fine-tuned | P3 |
  | **53** | **GRU + Attention** | **GloVe 300d** | **fine-tuned** | **P2** |
  | 54 | TextCNN | Word2Vec 300d | frozen | P3 |
  | 55 | BiLSTM | Word2Vec 300d | frozen | P3 |
  | **56** | **LSTM** | **Word2Vec 300d** | **frozen** | **P2** |
  | 57 | Stacked LSTM | Word2Vec 300d | frozen | P3 |
  | 58 | LSTM + Attention | Word2Vec 300d | frozen | P3 |
  | 59 | BiGRU | Word2Vec 300d | frozen | P3 |
  | 60 | GRU | Word2Vec 300d | frozen | P3 |
  | 61 | Stacked GRU | Word2Vec 300d | frozen | P3 |
  | 62 | GRU + Attention | Word2Vec 300d | frozen | P3 |
  | 63 | TextCNN | Word2Vec 300d | fine-tuned | P3 |
  | 64 | BiLSTM | Word2Vec 300d | fine-tuned | P3 |
  | 65 | LSTM | Word2Vec 300d | fine-tuned | P3 |
  | 66 | Stacked LSTM | Word2Vec 300d | fine-tuned | P3 |
  | 67 | LSTM + Attention | Word2Vec 300d | fine-tuned | P3 |
  | 68 | BiGRU | Word2Vec 300d | fine-tuned | P3 |
  | 69 | GRU | Word2Vec 300d | fine-tuned | P3 |
  | 70 | Stacked GRU | Word2Vec 300d | fine-tuned | P3 |
  | 71 | GRU + Attention | Word2Vec 300d | fine-tuned | P3 |
  | **72** | **TextCNN** | **FastText 300d** | **frozen** | **P2** |
  | 73 | BiLSTM | FastText 300d | frozen | P3 |
  | 74 | LSTM | FastText 300d | frozen | P3 |
  | 75 | Stacked LSTM | FastText 300d | frozen | P3 |
  | 76 | LSTM + Attention | FastText 300d | frozen | P3 |
  | 77 | BiGRU | FastText 300d | frozen | P3 |
  | 78 | GRU | FastText 300d | frozen | P3 |
  | 79 | Stacked GRU | FastText 300d | frozen | P3 |
  | 80 | GRU + Attention | FastText 300d | frozen | P3 |
  | 81 | TextCNN | FastText 300d | fine-tuned | P3 |
  | 82 | BiLSTM | FastText 300d | fine-tuned | P3 |
  | 83 | LSTM | FastText 300d | fine-tuned | P3 |
  | 84 | Stacked LSTM | FastText 300d | fine-tuned | P3 |
  | 85 | LSTM + Attention | FastText 300d | fine-tuned | P3 |
  | 86 | BiGRU | FastText 300d | fine-tuned | P3 |
  | 87 | GRU | FastText 300d | fine-tuned | P3 |
  | 88 | Stacked GRU | FastText 300d | fine-tuned | P3 |
  | 89 | GRU + Attention | FastText 300d | fine-tuned | P3 |

  ### D. Transformers — Full Fine-tuning (16 models)
  | # | Rec | Model | Params | Priority |
  |---|-----|-------|--------|----------|
  | **90** | **✓** | **BERT base (uncased)** | 110M | **P2** |
  | 91 | — | BERT large (uncased) | 340M | P3 |
  | **92** | **RoBERTa base** | 125M | **P2** |
  | 93 | — | RoBERTa large | 355M | P3 |
  | 94 | — | DeBERTa base | 140M | P3 |
  | 95 | — | DeBERTa large | 400M | P3 |
  | **96** | **DeBERTa v3 base** | 140M | **P2** |
  | 97 | — | DeBERTa v3 large | 400M | P3 |
  | **98** | **DistilBERT base** | 66M | **P2** |
  | **99** | **ALBERT base v2** | 12M | **P2** |
  | 100 | — | ALBERT large v2 | 18M | P3 |
  | 101 | — | ELECTRA small | 13M | P3 |
  | **102** | **ELECTRA base** | 110M | **P2** |
  | 103 | — | ELECTRA large | 335M | P3 |
  | **104** | **XLNet base (cased)** | 110M | **P2** |
  | 105 | — | XLNet large (cased) | 340M | P3 |

  ### D. Transformers — Feature Extraction (16 models)
  | # | Rec | Model | Priority |
  |---|-----|-------|----------|
  | 106 | — | BERT base — feature extraction | P3 |
  | 107 | BERT large — feature extraction | P3 |
  | 108 | RoBERTa base — feature extraction | P3 |
  | 109 | RoBERTa large — feature extraction | P3 |
  | 110 | DeBERTa base — feature extraction | P3 |
  | 111 | DeBERTa large — feature extraction | P3 |
  | 112 | DeBERTa v3 base — feature extraction | P3 |
  | 113 | DeBERTa v3 large — feature extraction | P3 |
  | 114 | DistilBERT base — feature extraction | P3 |
  | 115 | ALBERT base v2 — feature extraction | P3 |
  | 116 | ALBERT large v2 — feature extraction | P3 |
  | 117 | ELECTRA small — feature extraction | P3 |
  | 118 | ELECTRA base — feature extraction | P3 |
  | 119 | ELECTRA large — feature extraction | P3 |
  | 120 | XLNet base — feature extraction | P3 |
  | 121 | XLNet large — feature extraction | P3 |

  ### D. Transformers — Partial Fine-tuning (16 models)
  | # | Rec | Model | Priority |
  |---|-----|-------|----------|
  | 122 | — | BERT base — partial fine-tuning | P3 |
  | 123 | BERT large — partial fine-tuning | P3 |
  | 124 | RoBERTa base — partial fine-tuning | P3 |
  | 125 | RoBERTa large — partial fine-tuning | P3 |
  | 126 | DeBERTa base — partial fine-tuning | P3 |
  | 127 | DeBERTa large — partial fine-tuning | P3 |
  | 128 | DeBERTa v3 base — partial fine-tuning | P3 |
  | 129 | DeBERTa v3 large — partial fine-tuning | P3 |
  | 130 | DistilBERT base — partial fine-tuning | P3 |
  | 131 | ALBERT base v2 — partial fine-tuning | P3 |
  | 132 | ALBERT large v2 — partial fine-tuning | P3 |
  | 133 | ELECTRA small — partial fine-tuning | P3 |
  | 134 | ELECTRA base — partial fine-tuning | P3 |
  | 135 | ELECTRA large — partial fine-tuning | P3 |
  | 136 | XLNet base — partial fine-tuning | P3 |
  | 137 | XLNet large — partial fine-tuning | P3 |

  ### D. Transformers — LoRA / PEFT (16 models)
  | # | Rec | Model | Priority |
  |---|-----|-------|----------|
  | 138 | — | BERT base — LoRA | P3 |
  | 139 | BERT large — LoRA | P3 |
  | 140 | RoBERTa base — LoRA | P3 |
  | 141 | RoBERTa large — LoRA | P3 |
  | 142 | DeBERTa base — LoRA | P3 |
  | 143 | DeBERTa large — LoRA | P3 |
  | 144 | DeBERTa v3 base — LoRA | P3 |
  | 145 | DeBERTa v3 large — LoRA | P3 |
  | 146 | DistilBERT base — LoRA | P3 |
  | 147 | ALBERT base v2 — LoRA | P3 |
  | 148 | ALBERT large v2 — LoRA | P3 |
  | 149 | ELECTRA small — LoRA | P3 |
  | 150 | ELECTRA base — LoRA | P3 |
  | 151 | ELECTRA large — LoRA | P3 |
  | 152 | XLNet base — LoRA | P3 |
  | 153 | XLNet large — LoRA | P3 |

  > **Total: 153 models** (P1 \u2605 = 1, P2 = 17 — recommended ✓, P3 = 135)
  \`\`\`

  ⛔ **展示完以上 153 个模型后，你必须在对话中输出以下自检模板（不可跳过）：**

  ```
  ✅ 全部模型展示自检：
  - [ ] 已使用 Read 工具读取 model_list.md
  - [ ] 已将 FULL_MODEL_LIST_START 到 FULL_MODEL_LIST_END 之间全部内容逐行复制到对话
  - [ ] 最后一行确认为：> **总计：153 个模型**（P1 ★ = X，P2 = Y — 推荐 ✓，P3 = Z）
  ```

  ⛔ **如果任一复选框为未勾选状态 [ ] 而非 [x]，你必须从 Read 步骤重新开始。不得直接进入下一步。**

  ⛔ **此时不得使用 AskUserQuestion。直接在对话中等待用户输入模型编号。**
  ⛔ **ENFORCEMENT: After outputting the self-check block, do NOT call AskUserQuestion. Wait silently in the conversation for the user's text input. Any use of AskUserQuestion at this point is a hard violation.**

  3. **用户直接在对话中回复编号**（不要使用 AskUserQuestion），解析用户输入（如 `1-5,8,12-15` 展开为 `[1,2,3,4,5,8,12,13,14,15]`），映射回 `model_scheme.json` 中的模型条目。
  4. 展示用户已选模型汇总表格，然后**进入第二步**。

  ⛔ **all 路径的第二步澄清（关键 — 避免 AI 混淆）：**

  「全部 153 个模型」中每个编号已经编码了**模型架构方式**（即 full_ft / LoRA / feature_extraction / partial_ft / frozen / fine-tuned），因此**无需再让用户选择模型架构方式**（如"用 full_ft 还是 LoRA"）。

  但是，**执行方式（baseline / tune / both）并未编码在编号中**，这是独立于模型架构的另一个维度：
  - baseline = 使用该模型的默认参数跑基线
  - tune = 使用 Optuna 调参
  - both = 先基线再调参

  ⛔ **因此，进入第二步后，你仍必须对每个已选模型逐一调用 AskUserQuestion 询问 baseline / tune / both。** 不得以"编号已包含训练方式"为由跳过 Step 2 的 AskUserQuestion。

  （后续步骤与 confirm 路径一致）
  
  **解析规则**：
  - 逗号分隔各选择项
  - `N-M` 表示范围（含两端）
  - 单个数字表示单个模型
  - 忽略无效编号（< 1 或 > 153），提示用户修正
  - 至少选择 1 个模型，否则提示重新输入
  
  选择完成后，汇总选定的模型列表，**进入第二步（训练方式选择）**。模型架构方式（full_ft/LoRA/feature_extraction/partial_ft/frozen/fine-tuned）已编码在编号中无需再选，但执行方式（baseline/tune/both）仍必须通过 AskUserQuestion 逐一询问。



### 第二步：训练方式 / Step 2: Training Mode

对第一步确定的所有具体模型逐一选择训练方式。
Select the training mode for each model selected in Step 1.

Selection order / 选择顺序: **Traditional ML → Deep Learning → Transformer**.

> **🚫 Blocking point — You MUST call AskUserQuestion for each selected model to ask baseline/tune/both. Do not skip. This includes models selected via the all-153 path: even though model architecture mode (full_ft/LoRA etc.) is encoded in the number, baseline/tune/both MUST still be asked per-model here.**

每个模型一个 question（multiSelect），超过 4 个模型时分多轮 AskUserQuestion。

**每条 AskUserQuestion 开头用文本列出所有已选模型：**

> **ZH**: "当前已选模型：\n- [传统ML] SVM + TF-IDF bigram\n- [DL] TextCNN + GloVe（冻结/微调）\n- [TF] BERT base（全参数微调/LoRA）\n..."
> **EN**: "Currently selected models:\n- [Trad ML] SVM + TF-IDF bigram\n- [DL] TextCNN + GloVe (frozen/fine-tuned)\n- [TF] BERT base (full_ft/LoRA)\n..."

> **ZH Question**: "{模型名称} 的训练方式？"
> **EN Question**: "Training mode for {model_name}?"

| Value | ZH Label | EN Label | Description |
|-------|----------|----------|-------------|
| baseline | "默认参数基线" | "Baseline (default params)" | "使用预设默认参数运行基线评估，不调参。快速获得参照指标。" / "Run baseline with default params. No tuning. Quick reference." |
| tune | "Optuna 调参" | "Tune with Optuna" | "超参数自动调优（Optuna + MedianPruner 早停）。" / "Auto hyperparameter tuning with Optuna + MedianPruner early stopping." |
| both | "基线 + 调参" | "Baseline + Tune" | "先跑基线，再用 Optuna 调参，对比提升幅度。推荐。" / "Baseline first, then tune. Compare improvement. Recommended." |

---

### 第三步：Optuna 调参配置 / Step 3: Optuna Tuning Configuration

**仅对第二步中选择了"Optuna 调参"或"基线 + 调参"的模型进行。**
**Only for models where "Tune with Optuna" or "Baseline + Tune" was selected in Step 2.**

每个需要调参的模型一个 question，选择 Optuna 试验次数。
One question per model needing tuning: select number of Optuna trials.

> **🚫 Blocking point — You MUST call AskUserQuestion for each tune/both model to ask trial count. Do not skip.**

超过 4 个模型时分多轮 AskUserQuestion。 / Split into multiple rounds if more than 4 models.

**Each AskUserQuestion starts with a list of all models being tuned.**

> **ZH Question**: "{模型名称} 的 Optuna 试验次数？"
> **EN Question**: "Number of Optuna trials for {model_name}?"

每个 option 的 description 需根据模型类型说明预估时间。
Each option's description should include estimated time based on model type:

| Value | ZH Label | EN Label |
|-------|----------|----------|
| 20 | "20 次（快速）" | "20 trials (quick)" |
| 30 | "30 次（推荐）" | "30 trials (recommended)" |
| 50 | "50 次（彻底）" | "50 trials (thorough)" |

**各模型调参次数与预估时间参考 / Estimated Trial Time Reference:**

> Calibrated on 5K samples, ~200 word average text length, 5-fold CV, RTX 5060 8GB.
> Long texts (>150 words/sample) increase time 1.5-3×. Short texts (<50 words/sample) reduce time 30-50%.

| Model Type / 模型类型 | 20 trials | 30 trials (recommended) | 50 trials |
|----------|-------|-------------|-------|
| Traditional ML + TF-IDF / 传统 ML + TF-IDF | ~2 min | ~3 min | ~5 min |
| Traditional ML + embeddings / 传统 ML + 嵌入 | ~5 min | ~8 min | ~15 min |
| DL + frozen embeddings / DL + 固定嵌入 | ~15 min | ~25 min | ~40 min |
| DL + fine-tuned embeddings / DL + 微调嵌入 | ~30 min | ~45 min | ~75 min |
| Transformer (DistilBERT/ALBERT) | ~30 min | ~45 min | ~75 min |
| Transformer (BERT/RoBERTa/ELECTRA base) | ~60 min | ~90 min | ~150 min |
| Transformer (DeBERTa/XLNet/large) | ~90 min | ~135 min | ~220 min |

> **Note**: Times above are tuning phase only (not including baseline). `both` mode adds baseline time (~1× per-trial).
> Baseline reference: Traditional ML <1 min, DL ~3-8 min, BERT/XLNet base ~15-40 min (varies with text length and CV folds).

**Early stopping / 早停机制**: All Optuna tuning enables MedianPruner:
- First 5 trials not pruned (establish baseline)
- First 3 steps per trial (CV folds or epochs) not pruned
- Split mode: DL/Transformer report intermediate values each epoch; bad trials terminated early
- Traditional ML: reports after each CV fold; no early stopping within 3 folds (single-fold training is fast, low impact)

---

### 第四步：确认训练计划 / Step 4: Confirm Training Plan

Summarize all choices from the three steps above, output the training plan table / 汇总以上三步的所有选择，输出训练计划表：

**ZH template:**

```
## 训练计划汇总

| # | 模型 | 类别 | 嵌入/训练方式 | 训练模式 | 调参次数 | 预估时间 |
|---|------|------|-------------|---------|---------|---------|
| 1 | SVM + TF-IDF bigram | 传统ML | — | 基线 + 调参 | 30 | ~5 分钟 |
| 2 | 逻辑回归 + GloVe 300d | 传统ML | — | 基线 + 调参 | 30 | ~10 分钟 |
| 3 | TextCNN + GloVe | 深度学习 | 冻结 + 微调 | 基线 + 调参 | 30 | ~50 分钟 |
| 4 | BERT base | Transformer | 全参数微调 + LoRA | 基线 + 调参 | 30 | ~120 分钟 |
| ... | ... | ... | ... | ... | ... | ... |

总预估时间：约 X 小时 Y 分钟（串行）/ Z 分钟（并行 GPU 允许时）
预计磁盘占用：~Z GB（含模型产物和嵌入文件）

早停机制：已启用 MedianPruner（5 trial 冷启动 + 3 step 预热）
```

**EN template:**

```
## Training Plan Summary

| # | Model | Category | Embedding/Mode | Training | Trials | Est. Time |
|---|-------|----------|---------------|----------|--------|-----------|
| 1 | SVM + TF-IDF bigram | Trad ML | — | baseline + tune | 30 | ~5 min |
| 2 | Logistic Regression + GloVe 300d | Trad ML | — | baseline + tune | 30 | ~10 min |
| 3 | TextCNN + GloVe | DL | frozen + fine-tuned | baseline + tune | 30 | ~50 min |
| 4 | BERT base | TF | full_ft + LoRA | baseline + tune | 30 | ~120 min |
| ... | ... | ... | ... | ... | ... | ... |

Total estimated time: ~X hr Y min (serial) / Z min (parallel where GPU allows)
Estimated disk usage: ~Z GB (including model artifacts and embeddings)

Early stopping: MedianPruner enabled (5 trial startup + 3 step warmup)
```

> **🚫 阻断点 — 显示完训练计划汇总表后，你必须在此处调用 AskUserQuestion 确认是否开始训练，不得跳过。**

使用 **AskUserQuestion** 确认：

> **EN Question**: "Ready to start training with the plan above?"
> **ZH Question**: "确认以上训练计划，开始训练？"

| Value | EN Label | ZH Label |
|-------|----------|----------|
| proceed | "Yes, start training" | "确认，开始训练" |
| adjust | "No, let me adjust" | "否，我要调整" |

如果选择 **adjust**，回到第一步重新确认模型选择。
如果选择 **proceed**，进入 Stage 4（训练）。

---

## 模型描述参考

展示模型卡片时，根据交互语言使用以下描述：

**中文描述（用户使用中文时）：**

| 模型 | 描述 |
|------|------|
| SVM (LinearSVC) | 强线性分类器；通常是传统 ML 在文本分类上的性能上限。训练快，无需 GPU。 |
| 逻辑回归 (Logistic Regression) | 简单高效的线性模型；小数据集上不易过拟合，适合作为快速基线。 |
| 随机森林 (Random Forest) | 决策树集成；与线性模型有不同的归纳偏置，能捕获非线性特征交互。适合不平衡数据。 |
| 多项式朴素贝叶斯 (Multinomial NB) | 概率模型；极快，适合短文本和小数据集。常作为基线比较。 |
| SVM/ LR/ RF + GloVe/Word2Vec/fastText | 传统 ML + 稠密嵌入：用预训练词向量平均池化替代稀疏 BoW。RBF SVM 在稠密向量上显著优于稀疏 TF-IDF。适合 5K+ 样本。 |
| BiLSTM | 双向 LSTM；捕获长距离序列依赖。适合长文本和复杂句式。 |
| BiGRU | 双向 GRU；与 BiLSTM 类似但收敛快约 15%，性能相当。 |
| GRU + Attention | GRU 加注意力机制；学习哪些词对分类决策最重要。适合长文本。 |
| LSTM + Attention | LSTM 加注意力；与 GRU+Attention 类似但使用 LSTM 单元。大数据集上可能更有表达力。 |
| Stacked LSTM / GRU | 多层循环网络；对复杂模式有更强容量。小数据集上有过拟合风险。 |
| BERT base | 12 层 Transformer（110M 参数）；文本分类的标准参照点。综合表现好。 |
| RoBERTa base | BERT 优化版（125M 参数）；分类基准上通常优于 BERT。预训练数据更多。 |
| DeBERTa base/v3 | 解耦注意力（140M 参数）；文本分类上常达最优。v3 使用 ELECTRA 式预训练。 |
| DeBERTa large | 24 层 DeBERTa（400M 参数）；比 base 好约 2-3%。推荐 12GB+ 显存使用。 |
| DistilBERT | BERT 蒸馏版（66M 参数）；体积小约 40%，快约 60%，保留约 95% BERT 性能。适合部署和低显存场景。 |
| ALBERT base | 参数高效 BERT（12M 参数）；共享层权重 + 分解嵌入，显存占用低。适合显存有限的 GPU。 |
| ALBERT large | 18M 参数 large 版（hidden 1024）；比 base 好约 2%，参数共享保持低显存。 |
| ELECTRA small | 紧凑判别器（13M 参数）；hidden 256。极低显存场景下快速实验。适合 2-4GB 显存。 |
| ELECTRA base | 判别器预训练模型；中小数据集上常优于 BERT，训练效率高。 |
| ELECTRA large | 24 层判别器（335M 参数）；比 base 好约 2-3%。推荐 12GB+ 显存。 |
| XLNet base | 自回归 Transformer-XL（110M 参数）；无 [CLS] token，使用最后 token 表示。适合长文本中的长距离依赖。 |
| XLNet large | 24 层 XLNet（340M 参数）；比 base 好约 2-3%。推荐 12GB+ 显存。 |
| BERT/RoBERTa large | 24 层版本（340M/355M 参数）；比 base 好约 2-3%，但慢 3-4 倍。仅推荐 12GB+ 显存使用。 |

**English descriptions (when user communicates in English):**

| Model | Description |
|-------|-------------|
| SVM (LinearSVC) | Strong linear classifier; often the performance ceiling for traditional ML on text. Fast training, no GPU needed. |
| Logistic Regression | Simple, efficient linear model; less prone to overfitting on small datasets. Great quick baseline. |
| Random Forest | Decision tree ensemble; different inductive bias from linear models, captures non-linear feature interactions. Good for imbalanced data. |
| Multinomial NB | Probabilistic model; extremely fast, good for short texts and small datasets. Common baseline. |
| SVM / LR / RF + GloVe/Word2Vec/fastText | Traditional ML + dense embeddings: average pretrained word vectors instead of sparse BoW. RBF SVM significantly outperforms sparse TF-IDF on dense vectors. Best for 5K+ samples. |
| BiLSTM | Bidirectional LSTM; captures long-range sequential dependencies. Good for long texts and complex syntax. |
| BiGRU | Bidirectional GRU; similar to BiLSTM but converges ~15% faster with comparable performance. |
| GRU + Attention | GRU with attention mechanism; learns which words matter most for classification. Good for long texts. |
| LSTM + Attention | LSTM with attention; similar to GRU+Attention but uses LSTM cells. May be more expressive on large datasets. |
| Stacked LSTM / GRU | Multi-layer recurrent networks; higher capacity for complex patterns. Risk of overfitting on small datasets. |
| BERT base | 12-layer Transformer (110M params); standard reference point for text classification. Strong all-around performance. |
| RoBERTa base | Optimized BERT (125M params); typically outperforms BERT on classification benchmarks. More pretraining data. |
| DeBERTa base/v3 | Disentangled attention (140M params); often state-of-the-art on text classification. v3 uses ELECTRA-style pretraining. |
| DeBERTa large | 24-layer DeBERTa (400M params); ~2-3% better than base. Recommended with 12GB+ VRAM. |
| DistilBERT | Distilled BERT (66M params); ~40% smaller, ~60% faster, retains ~95% of BERT performance. Great for deployment and low-VRAM. |
| ALBERT base | Parameter-efficient BERT (12M params); shared layer weights + factorized embeddings, low VRAM usage. Good for limited GPUs. |
| ALBERT large | 18M param large variant (hidden 1024); ~2% better than base, shared params keep VRAM low. |
| ELECTRA small | Compact discriminator (13M params); hidden 256. Fast experiments on very low VRAM. Good for 2-4GB VRAM. |
| ELECTRA base | Discriminator-pretrained model; often outperforms BERT on small-medium datasets, efficient training. |
| ELECTRA large | 24-layer discriminator (335M params); ~2-3% better than base. Recommended with 12GB+ VRAM. |
| XLNet base | Autoregressive Transformer-XL (110M params); no [CLS] token, uses last token representation. Good for long-range dependencies in long texts. |
| XLNet large | 24-layer XLNet (340M params); ~2-3% better than base. Recommended with 12GB+ VRAM. |
| BERT/RoBERTa large | 24-layer versions (340M/355M params); ~2-3% better than base but 3-4x slower. Only recommended with 12GB+ VRAM. |

---

## Stage 4：模型训练 / Model Training

Run `scripts/step4_train.py`:

```bash
PYTHONIOENCODING=utf-8 python scripts/step4_train.py \
  --csv <path> --text-col <name> --label-col <name> \
  --scheme <project_dir>/model_scheme.json \
  [--split <project_dir>/split_info.json] \
  [--models <name1;name2>] \
  [--mode baseline|tune|both] \
  [--cv-folds 5] \
  [--tune-method cv|split] \
  [--tune-trials 50] \
  [--output-dir <project_dir>] \
  [--embedding-path <path>] \
  [--glove-path <path>] [--word2vec-path <path>] [--fasttext-path <path>] \
  [--epochs <N>] [--encoding <enc>] \
  [--no-mlflow] [--seed 42]
```

- `--mode` defaults to `both` (baseline + tuning).
- `--models` uses `;` to separate model names (display_name may contain commas).
- `--encoding` auto-detects (tries utf-8, utf-8-sig, gbk, latin-1, etc.); can manually specify.
- `--embedding-path` is legacy; prefer `--glove-path` / `--word2vec-path` / `--fasttext-path`.

**Training progress tracking / 训练进度日志**: Dual progress tracking enabled:
1. **Real-time log file / 实时日志文件**: `output/training.log` — line-buffered, tail -f capable
2. **Periodic progress reports / 定期进度报告**: Every 5 minutes prints summary (completed models, current model, remaining, elapsed) to console and log

**Baseline mode**: Default params, stratified K-fold CV per model. Display per-fold metrics.

**Tune mode**: Optuna hyperparameter optimization with MedianPruner early stopping. tqdm progress bars show trial progress and current best. Best params re-evaluated with full CV.

**Both mode (default)**: Baseline first, then tuning, then comparison. Shows absolute and relative improvement.

GPU auto-detection (CUDA > MPS > CPU). Conservative batch sizes for low-VRAM GPUs. Auto-fallback to CPU on OOM with warning.

### Training Time Estimates / 训练时间预估

Approximate times on a single GPU (e.g., RTX 3060/4060/5060 8GB). **Actual times highly depend on text length** — based on ~100 words/sample typical text. Long texts (>200 words) may double these times.

| Data Size / 数据量 | Traditional ML (all P1) | DL (1 model, 5-fold) | Transformer base (5-fold, 3 epoch) |
|--------|-------------------|-------------------|---------------------------|
| 5K samples | < 1 min | ~8 min | ~20-40 min |
| 50K samples | ~3 min | ~40 min | ~2-4 hr |
| 200K+ samples | ~8 min | ~2 hr | ~10+ hr (recommend --epochs 2) |

Large dataset tips / 大数据集建议:
- Use `--epochs 2` to reduce Transformer/DL training time
- Use `--tune-method split` for faster tuning (avoids CV per trial)
- Use `--cv-folds 3` instead of default 5-fold
- For long-text data, prefer traditional ML or lightweight DistilBERT/ALBERT
- Tuning total ≈ trials × baseline time (e.g. 20 trials × 8 min baseline ≈ 160 min)

### China / Restricted Network Environments / 中国 / 网络受限环境

If Hugging Face is unreachable (common in mainland China), set the mirror before Stage 3 or 4:

```bash
# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com

# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"

# Windows CMD
set HF_ENDPOINT=https://hf-mirror.com
```

Script auto-detects network errors and prompts to use the mirror on download timeout.

MLflow tracking enabled by default (SQLite backend, stored at `output/mlflow.db`). All params, metrics, and per-fold results are logged.

Generates `<project_dir>/training_results.json` and saves intermediate model files in `<project_dir>/models/`.

**After training completes**, display baseline vs tuned results in the user's language:

1. Model metrics summary table — **strictly follow the ⛔ Global Output Format Rule above**. Flat single-row headers: `{Dataset}_{Metric} (N={sample_count})`. Copy directly from `step4_train.py` output; do not reformat.
2. Absolute and relative improvement for tuned models (vs baseline)
3. Best model annotation (sorted by F1/Accuracy)
4. **Best model parameter table** — below the summary table, list the best model's params:
   - **Best model is tuned** → Show Optuna best params with default value comparison column (`Default → Best`)
   - **Best model is baseline** → Show that model's default parameter table
   Extract params from `training_results.json` for the corresponding model (`baseline.params` or `tuned.best_params`).
5. **Skipped models note** — Below the summary, list any models that were skipped:
   > ⚠️ The following models could not be trained:
   > - **{model name}**: {reason skipped}. Solution: {specific steps}
   - Reasons must be specific (embedding download failed / network unreachable / insufficient VRAM / user cancelled)
   - Solutions must be actionable (e.g., manually download file then pass `--xxx-path`, set `HF_ENDPOINT` mirror, free VRAM)
   - Omit if all models trained successfully
6. **Post-training diagnostic analysis** — After displaying the summary table, run `scripts/analyze_results.py`:

   ```bash
   PYTHONIOENCODING=utf-8 python scripts/analyze_results.py \
     --results <project_dir>/training_results.json \
     --analysis <project_dir>/analysis.json \
     --output-dir <project_dir>
   ```

   脚本输出以下维度的诊断分析：

   - **模型综合排名**：按 Test F1 降序排列，显示最佳模型及其领先幅度
   - **过拟合分析**：检查每个模型的 Train/Val 指标差距。gap > 15pp 标记为严重（🔴），gap > 5pp 标记为中度（🟡）。对严重过拟合给出简化结构/增加正则化的建议
   - **交叉验证稳定性**：检查各 CV 折之间的 Acc 极差。极差 > 5% 标记为不稳定
   - **Test 集异常检测**：检查 Test 指标是否显著高于所有 CV 折的最高值。若 Test Acc > max(折 Acc) + 2pp，标记为异常，提示可能原因（全量重训验证划分不一致 / 随机种子效应）
   - **模型-数据规模匹配**：基于数据样本量检查模型复杂度是否合理。如 Stacked 模型在 <10K 样本上的过拟合风险、Large 模型在 <20K 样本上不推荐等
   - **调参效率分析**：对做过 Optuna 调参的模型，评估提升是否值得计算开销。若超参数 ≤2 个且搜索次数 ≥10 但 ΔF1 < 0.01，标记为性价比低，建议直接用默认参数
   - **📋 诊断建议（Diagnostic Recommendations）**：综合以上所有诊断维度，自动生成具体、可操作的建议。每项建议包含：
     - **严重级别**：critical（严重）/ high（高优先）/ moderate（中等）/ info（参考）
     - **分类**：convergence_failure（收敛失败）/ cv_instability / severe_overfitting / test_anomaly / model_data_mismatch / tuning_inefficiency / seq_len_truncation
     - **具体问题描述**：包含相关指标数值
     - **操作建议**：按步骤给出可执行的修复动作（如"将 learning_rate 从 2e-5 提高到 5e-5"）
     - **原因说明**：解释为什么会出现该问题
     
     覆盖的诊断场景包括：
     - **CV 部分收敛失败**（部分折 Acc≈0.50 随机，其他折正常）→ 建议提高 LR、增加 warmup、使用 split 模式调参
     - **严重/中度过拟合** → 按模型类别（Transformer/DL/传统ML）给出针对性正则化建议
     - **Test 异常高于 CV** → 识别是否为 CV 不稳定导致的假象，给出验证建议
     - **文本截断**（平均长度 > max_seq_len）→ 建议增加 seq_len 或使用长文本模型
   - **总结建议**：综合以上各维度，给出最佳模型推荐和后续优化方向

   脚本生成的诊断结果保存至 `<project_dir>/post_analysis.json`。
   将全部诊断内容逐段展示给用户，不得省略任何维度。

   **⛔ 此分析步骤为强制步骤，每次 Stage 4 训练完成后必须执行。** 脚本已集成所有分析逻辑，无需手工计算。

> **🚫 阻断点 — 训练结果展示 + 诊断分析完毕后，你必须在此处调用 AskUserQuestion 询问用户下一步操作，不得跳过。**

然后询问用户下一步：

> **EN Question**: "All tuning trials are complete. What would you like to do next?"
> **ZH Question**: "所有模型调优已完成。接下来做什么？"

| Value | EN Label | ZH Label |
|-------|----------|----------|
| proceed | "Pick the best model → re-train on full dataset → save & deploy (Stage 5)" | "选择最佳模型 → 全量数据重新训练 → 保存并部署（Stage 5）" |
| retune | "Re-tune some models with different parameters" | "调整参数重新调优某些模型" |
| more | "Train additional models from the scheme" | "从方案中增加训练更多模型" |

**注意**：用户必须确认最佳模型后，才能在 Stage 5 中用全量数据重新训练。
这是因为调优阶段使用交叉验证/切分来公平评估，最终部署前应使用全部数据
（在 Stage 5 中）重新训练以获得最佳泛化性能。

---

## Stage 5：全量训练 + 保存最终模型 / Full Training + Save Final Model

Stage 4 中通过交叉验证/切分确定了最佳超参数。Stage 5 使用**全部数据**
重新训练最佳模型，以获得生产环境的最佳泛化性能。

用户确认最佳模型后，运行 `scripts/step5_save.py`：

```bash
PYTHONIOENCODING=utf-8 python scripts/step5_save.py \
  --csv <path> --text-col <name> --label-col <name> \
  --analysis <project_dir>/analysis.json \
  --training-results <project_dir>/training_results.json \
  --best-model "<模型展示名称>" \
  [--split <project_dir>/split_info.json] \
  [--retrain-on-full] \
  [--output-dir <project_dir>] \
  [--embedding-path <path>] \
  [--glove-path <path>] [--word2vec-path <path>] [--fasttext-path <path>] \
  [--encoding <enc>] \
  [--no-mlflow] [--seed 42]
```

- 若提供了 `--split`：先在测试集上评估，`--retrain-on-full` 则用全部数据重训
- `--encoding` 默认自动检测；`--embedding-path` 为兼容参数，推荐用 per-type 参数
- **全量重训前必须与用户确认**

`<project_dir>` 为 Stage 1 环境确认中用户指定的项目目录。

This script executes in order / 此脚本按以下顺序执行:
1. **Full data training / 全量数据训练**: Retrain on **all samples** using Stage 4 best hyperparameters
2. **Save** model artifacts to `<project_dir>/final_model/`:
   - sklearn: `.pkl` files (model + vectorizer) via joblib
   - PyTorch: `.pt` files (state_dict) + vocab `.pkl`
   - Transformers: `.pt` file + tokenizer directory
3. **Generate** `<project_dir>/training_report.html` — standalone HTML report:
   - Dataset statistics summary
   - Training results comparison table for all models
   - Best model details (hyperparameters, per-fold/split metrics)
   - Confusion matrix (from Stage 4 validation set predictions)
   - Deployment info (API endpoints, model paths, dependencies)
4. **Generate** deployment artifacts at `<project_dir>/deploy/`:
   - `api_server.py` (FastAPI with `/health`, `/predict`, `/predict_batch` endpoints)
   - `requirements.txt` (pinned dependency versions)
   - `Dockerfile` (multi-stage build with health check)
   - `monitoring.md` (Prometheus metrics, alert rules, drift detection)

⚠️ `step5_save.py` internally calls `webbrowser.open()` to open the HTML report. **Do NOT manually open it again** or two browser windows will pop up.
Print the output directory tree showing all generated files.

## Output Directory Structure / 输出目录结构

```
output/
├── analysis.json              # Stage 1 output
├── split_info.json            # Stage 2 output
├── model_scheme.json          # Stage 3 output
├── training_results.json      # Stage 4 output
├── post_analysis.json         # Stage 4 post-training diagnostic analysis
├── mlflow.db                  # MLflow tracking database
├── models/                    # Stage 4 intermediate models
│   ├── <name>_baseline.pkl
│   └── <name>_tuned.pkl
├── final_model/               # Final model artifacts (Stage 5)
│   ├── <name>_model.pkl/.pt
│   └── <name>_vectorizer.pkl / tokenizer/
├── training_report.html       # HTML report (Stage 4)
└── deploy/                    # Deployment artifacts (Stage 5)
    ├── api_server.py
    ├── requirements.txt
    ├── Dockerfile
    └── monitoring.md
```

## Encoding Strategy / 编码策略

- Python source code in English (ASCII-compatible); comments may use Chinese
- All file I/O uses `encoding='utf-8'` with `errors='replace'`
- JSON files use `ensure_ascii=False` for readable Unicode in reports
- Reports use `<meta charset="UTF-8">`
- **Critical / 关键**: All Bash calls must set `PYTHONIOENCODING=utf-8` before the python command

## Error Handling / 错误处理

Each script follows a unified pattern:
1. CLI arg validation — file existence, valid options
2. Data validation — column types, label values (0/1 only), language check
3. Training errors — GPU OOM fallback, convergence warnings, empty vocabulary
4. File I/O — atomic writes, directory creation (exist_ok)

| Error / 错误 | Message / 消息 | Handling / 处理 |
|------|------|------|
| File not found — Windows path format | `FileNotFoundError` from Python with `/c/Users/...` path | **Self-heal / 自愈**: Convert path to `C:\Users\...` format and retry |
| File not found | `[ERROR] File not found: <path>` | Exit code 1 |
| Column name error | `[ERROR] Column '<name>' not found. Available: [...]` | Exit code 1 |
| Labels not 0/1 | `[ERROR] Label column must contain only 0 and 1. Found: [...]` | Exit code 1 |
| Non-English > 30% | `[WARN] Data may contain significant non-English text` | Warn, continue |
| GPU OOM | `[WARN] GPU out of memory, falling back to CPU` | Auto fallback |
| MLflow unavailable | `[WARN] MLflow is not installed. Experiment tracking disabled.` | Continue without MLflow |
| Empty vocabulary | `[ERROR] All texts became empty after cleaning` | Exit code 1 |

## Dependencies / 依赖

Core / 核心: `numpy pandas scikit-learn nltk joblib tqdm psutil`
Deep Learning / 深度学习: `torch transformers tokenizers`
Optimization / 优化: `optuna`
Tracking / 追踪: `mlflow`
Web API: `fastapi uvicorn pydantic` (for deployment)
Optional / 可选: `langdetect` (better language detection)
Environment check / 环境检查: `pyyaml packaging`

One-liner install / 一键安装:
```bash
pip install numpy pandas scikit-learn nltk joblib tqdm psutil \
  torch transformers tokenizers optuna mlflow \
  fastapi uvicorn pydantic pyyaml packaging
```

## References / 参考资料

- `references/model_params.md` — Hyperparameter recommendations for all model types, compiled from literature and web research. Reference when user asks about specific parameter choices or defaults.
- `references/requirements.yaml` — Python dependency manifest; the reference baseline for Stage 1 environment checks. Update when adding new dependencies.
- `scripts/` — All Python modules. Read relevant files when debugging or extending functionality.
