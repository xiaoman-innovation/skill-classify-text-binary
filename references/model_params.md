# Model Hyperparameter Recommendations

Reference for recommended hyperparameters gathered from web search of best
practices in English text binary classification. These defaults feed into
`model_factory.py` and `step2_scheme.py`. Updated: 2026-05.

---

## A. Traditional Machine Learning

### SVM (LinearSVC)
| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| C | 1e-2 to 1e2 (log-uniform) | 1.0 | Practical sweet spot 0.1–10; higher = less regularization |
| max_iter | 1000, 2000, 5000 | 2000 | Increase for large datasets |
| loss | squared_hinge, hinge | squared_hinge | squared_hinge often more stable |
| dual | True, False | False | dual=False for n_features << n_samples |
| class_weight | None, balanced | None | Use 'balanced' for imbalanced classes |

**Empirical benchmark (2025 study on sentiment classification, TF-IDF features):**
LinearSVC reached 82.0% accuracy vs RBF SVM at 76.0% — linear kernels are
strongly preferred for high-dimensional sparse TF-IDF vectors. RBF may
outperform linear only when using dense embeddings (~100–300 dim).

**Dense embedding variants (GloVe / Word2Vec / fastText):**
Traditional ML classifiers can also use pretrained word embeddings as input
features by averaging word vectors per text (300-dim dense vector). Key points:

- RBF SVM + GloVe: RBF kernel benefits most from dense representations; can
  outperform LinearSVC + TF-IDF on datasets with rich semantic content.
- Logistic Regression + GloVe: Fast and strong baseline; often competitive
  with TF-IDF, especially for datasets > 5K samples.
- Random Forest + GloVe: Captures non-linear interactions in embedding space;
  heavier than SVM/LR but worth trying when linear models underperform.
- Multinomial NB is excluded from embedding variants (requires non-negative
  features; embeddings contain negative values).

Sequence of operations: raw text → tokenize → lookup word vectors → average
(fixed representation, no gradients) → sklearn classifier. Unlike DL models,
embeddings are NOT fine-tuned — they are used as static features.

**Large dataset alternative:** For >100k samples, use
`SGDClassifier(loss='hinge')` instead of `LinearSVC` for significantly
faster training with comparable results.

### SVM (SVC with RBF kernel)
| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| C | 1e-2 to 1e2 (log-uniform) | 1.0 | |
| gamma | 1e-4 to 1e1 (log-uniform) | scale | 'scale' = 1/(n_features * var) |
| kernel | rbf | rbf | Only consider when using dense embeddings, not sparse TF-IDF |
| class_weight | None, balanced | None | Use 'balanced' for imbalanced classes |

### Logistic Regression
| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| C | 1e-3 to 1e3 (log-uniform) | 1.0 | Inverse regularization strength |
| penalty | l1, l2, elasticnet | l2 | l2 is stable default; l1 for feature selection |
| solver | saga, lbfgs | saga | saga supports all penalties + elasticnet |
| max_iter | 100, 500, 1000, 2000 | 1000 | |
| class_weight | None, balanced | None | Use 'balanced' for imbalanced classes |

LR + TF-IDF is consistently the strongest "quick baseline" for text
classification — start here before trying more complex models. On most
datasets within 1-3% of the best tuned model.

### Random Forest
| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| n_estimators | 100, 200, 300, 500, 1000 | 200 | More trees = more stable; diminishing returns |
| max_depth | 10, 20, 30, 50, None | 30 | None = unlimited; can overfit text data |
| min_samples_split | 2, 5, 10, 20 | 5 | Higher = less overfitting |
| min_samples_leaf | 1, 2, 4, 8 | 2 | |
| max_features | sqrt, log2, None | sqrt | sqrt is good for text (many features) |
| class_weight | balanced, balanced_subsample, None | balanced | Use for imbalanced datasets |

### Multinomial Naive Bayes
| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| alpha | 1e-3 to 10.0 (log-uniform) | 1.0 | Smoothing parameter |
| fit_prior | True, False | True | Learn class priors from data |

---

## B. Deep Learning

### General Training Parameters
| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| embedding_dim | 100, 200, 300 | 300 | Match pretrained embedding dim if using GloVe |
| hidden_dim | 64, 128, 256, 512 | 128 | Larger for bigger datasets |
| num_layers | 1, 2, 3 | 2 | 2 is sweet spot for most tasks |
| batch_size | 16, 32, 64, 128 | 64 | Reduce for larger models / limited GPU |
| learning_rate | 1e-5 to 1e-2 (log-uniform) | 1e-3 | 1e-4 to 1e-3 typical; lower for fine-tuning |
| optimizer | Adam, AdamW, SGD | Adam | AdamW if using weight decay |
| dropout | 0.1, 0.2, 0.3, 0.5, 0.7 | 0.5 | 0.5 is standard for text; reduce if underfitting |
| weight_decay | 0.0, 1e-5, 1e-4, 1e-3 | 0.0 | Only used with AdamW |
| epochs | 5, 10, 15, 20, 30, 50 | 20 | With early stopping patience=5 |
| max_seq_len | 64, 128, 256, 512 | 128 | Based on text length distribution |

**Architecture selection by text characteristics (based on 2024-2025 benchmarks):**
- **Short texts (<50 words)**: TextCNN excels — CNN filters capture local
  n-gram patterns efficiently. Faster to train than RNNs.
- **Medium texts (50-200 words)**: BiLSTM or BiGRU are solid defaults.
  BiGRU converges ~15% faster with comparable performance.
- **Long texts (>200 words)**: GRU/LSTM + Attention learns which words
  matter. Attention mechanism provides interpretability for which spans
  drive the classification decision.
- **Very long texts (>500 words)**: Stacked LSTM/GRU (3 layers) or
  Transformers handle long-range dependencies better.
- **Small datasets (<5k)**: Prefer fixed embeddings (freeze_embeddings=True)
  to avoid overfitting. TextCNN is more parameter-efficient than RNNs.
- **Composite models**: TextCNN + BiLSTM hybrid architectures (CNN filters
  feeding into RNN) have shown strong results in recent benchmarks, but add
  complexity and training time — only pursue if simpler models underperform.

### Embedding Options

| Embedding | Dim | Vocab Size | Strengths |
|-----------|-----|-----------|-----------|
| Word2Vec (GoogleNews) | 300 | 3M | Good all-round, captures semantic similarity well |
| GloVe (6B/840B) | 300 | 2.2M | Co-occurrence based, good for analogy tasks |
| FastText (Common Crawl) | 300 | 2M | Subword (n-gram) aware, handles OOV and misspellings |

#### Word2Vec
- Google News pretrained (100B words): `GoogleNews-vectors-negative300.bin`
- Text format: one word+vector per line (same format as GloVe)
- Fixed mode: good baseline, less prone to overfitting on small datasets
- Fine-tuned mode: can adapt domain-specific semantics
- Recommendation: use fixed for <5K samples, fine-tuned for >=5K

#### GloVe
- 6B token version recommended: `glove.6B.300d.txt`
- Standard baseline for most NLP classification benchmarks
- Same recommendations as Word2Vec for fixed vs fine-tuned

#### FastText
- Common Crawl pretrained (600B tokens): `crawl-300d-2M-subword.vec`
- Subword information helps with rare words, misspellings, informal text
- Particularly effective for user-generated content (reviews, tweets)
- Recommendation: use fixed mode for noisy text; subword info provides
  good coverage even without fine-tuning

### LSTM / GRU (Unidirectional, Configurable Layers)

Single-direction RNNs with configurable `num_layers` (default 1):

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| hidden_dim | 64, 128, 256, 512 | 128 | Output dimension per direction |
| num_layers | 1, 2, 3 | 1 (LSTM), 1 (GRU) | Stack depth |
| dropout | 0.1-0.7 | 0.5 | Applied between layers (num_layers > 1) |

- LSTM: stronger long-range dependency modeling, more params
- GRU: ~15% faster convergence, similar performance to LSTM
- Single layer (num_layers=1): baseline, fast training
- Multi-layer (num_layers>=2): use StackedLSTM / StackedGRU defaults (3 layers)
- When to use: medium-length texts (50-200 words), CPU training is feasible

### TextCNN (Kim 2014)
| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| filter_sizes | [2,3,4], [3,4,5], [4,5,6] | [3,4,5] | Multi-scale feature extraction |
| num_filters | 64, 100, 128, 256 | 100 | Per filter size |
| use_batch_norm | True, False | True | Often helps convergence |

### BiLSTM
| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| hidden_dim | 64, 128, 256 | 128 | Bidirectional => output dim = 2 * hidden_dim |
| num_layers | 1, 2, 3 | 2 | Stack LSTM layers |
| attention_dim | 32, 64, 128 | 64 | If using attention variant |

### StackedLSTM
Same as BiLSTM but with `bidirectional=False` and 2-3 layers.

### BiGRU
Same ranges as BiLSTM. GRU converges slightly faster than LSTM for similar
performance on most text classification benchmarks. Use hidden_dim=128 for
datasets < 50k samples, 256 for larger datasets.

---

## C. Transformers

### General Fine-Tuning Recommendations
Based on Devlin et al. (BERT paper), Liu et al. (RoBERTa), He et al. (DeBERTa),
and 2024-2025 empirical benchmarks on binary text classification:

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| learning_rate | 1e-6 to 5e-5 (log-uniform) | 2e-5 | Best range for BERT-family fine-tuning |
| batch_size | 8, 16, 32 | 16 | 16 is safe for 8GB VRAM with max_len=256 |
| max_seq_len | 64, 128, 256, 512 | 256 | Truncate to this; 512 for long texts |
| epochs | 2, 3, 4, 5 | 3 | Transformers overfit quickly on small data |
| warmup_ratio | 0.0, 0.06, 0.1 | 0.06 | Linear warmup proportion of total steps |
| weight_decay | 0.0, 0.01, 0.1 | 0.01 | Regularization for fine-tuning |
| dropout | 0.1, 0.2, 0.3 | 0.1 | Classification head dropout; encoder uses default |
| gradient_accumulation_steps | 1, 2, 4, 8 | 1 | Simulate larger batch: effective_batch = batch * accum |

**Learning rate sensitivity by model family (from 2024-2025 community benchmarks):**
- **BERT base**: 2e-5 is the standard starting point. Range 1e-5 to 5e-5
  works well. Lower end (1e-5) for small datasets (<5k), higher end (3e-5
  to 5e-5) for larger datasets (>20k).
- **RoBERTa base**: Slightly more sensitive. Start at 1e-5. The optimal
  point is often 1e-5 to 2e-5 — using BERT's 2e-5 default may overshoot.
- **DeBERTa v3 base**: 1e-5 to 3e-5 works well. v3's ELECTRA-style
  pretraining makes it more robust to LR choice.
- **DistilBERT**: 2e-5 to 5e-5 — less sensitive due to fewer parameters.

**Batch size vs GPU VRAM quick reference:**
| VRAM | batch_size=8 safe | batch_size=16 safe | batch_size=32 safe |
|------|-------------------|--------------------|---------------------|
| 4 GB | Yes | With max_len=128 | No |
| 6 GB | Yes | Yes | With max_len=128 |
| 8 GB | Yes | Yes | Yes |
| 12+ GB | Yes | Yes | Yes (max_len=512 ok) |

For gradient accumulation: set `gradient_accumulation_steps=N` so that
`batch_size × N = 32` (the recommended effective batch size).

### Feature Extraction Mode (frozen encoder)
When the pretrained encoder is frozen and only the classification head is trained:
- learning_rate: 1e-4 to 1e-3 (higher since only the head is learned)
- epochs: 5-20 (more epochs needed since fewer params)
- dropout: 0.2-0.5 (more regularization on the head)

### Model-Specific Notes

#### BERT (bert-base-uncased / bert-large-uncased)
- 12 layers (base) / 24 layers (large)
- Hidden dim: 768 (base) / 1024 (large)
- Max position embeddings: 512
- Good general-purpose English text representation
- Fine-tune all layers for best results; freeze first 6 layers for partial fine-tuning

#### RoBERTa (roberta-base / roberta-large)
- Same architecture as BERT but trained on more data with dynamic masking
- Generally outperforms BERT on classification benchmarks
- Slightly more sensitive to learning rate (try 1e-5 as starting point)
- No token_type_ids (segment embeddings removed)

#### DeBERTa (deberta-base / deberta-large / deberta-v3-base / deberta-v3-large)
- Uses disentangled attention mechanism (separate content/position embeddings)
- Base: 12 layers, hidden 768, 140M params
- Large: 24 layers, hidden 1024, ~400M params
- v3 variants use ELECTRA-style pretraining (RTD objective)
- Often state-of-the-art on classification benchmarks
- Higher memory footprint than BERT/RoBERTa of same size
- Large variants: ~2-3% better than base, ~3-4x slower, need 12GB+ VRAM
- Recommended modes: full_ft for best results, peft for large on modest GPUs

#### DistilBERT (distilbert-base-uncased)
- 40% smaller, 60% faster than BERT-base
- Good for: quick experiments, CPU deployment, limited GPU memory
- Performance: ~95% of BERT-base on most GLUE tasks

#### ALBERT (albert-base-v2 / albert-large-v2)
- Parameter-sharing across all transformer layers (single weight group)
- Base: hidden 768, 11M total params (vs 110M for BERT-base)
- Large: hidden 1024, 18M total params
- Much smaller memory footprint — excellent for low-VRAM environments
- Performance: competitive with BERT-base on GLUE despite smaller size
- Partial fine-tuning: less meaningful due to parameter sharing
  - Code auto-detects ALBERT and limits partial_ft to embedding freeze only
- Recommended modes: full_ft for best performance, peft for rapid iteration

#### ELECTRA (electra-small-discriminator / electra-base-discriminator / electra-large-discriminator)
- Uses Replaced Token Detection (RTD) pretraining objective
- Small: hidden 256, 12 layers, 13M params — compact, fast experiments
- Base: hidden 768, 12 layers, 110M params
- Large: hidden 1024, 24 layers, 335M params
- Discriminator head is well-suited for classification tasks
- Training: more sample-efficient than BERT due to RTD objective
- Recommended modes: full_ft for best results, feature_extraction for baseline
- Small variant: ideal for 2-4GB VRAM, quick iteration

#### XLNet (xlnet-base-cased / xlnet-large-cased)
- Autoregressive Transformer-XL architecture with permutation language modeling
- Base: 12 layers, hidden 768, 110M params
- Large: 24 layers, hidden 1024, 340M params
- No [CLS] token — uses the last token's hidden state for classification
- No token_type_ids (segment embeddings) — architecture-specific
- Attention modules use short names (q, v, k, o, r) vs BERT-family (query, value)
- LoRA target_modules: use ["rel_attn.q", "rel_attn.v"] instead of ["query", "value"]
- Strengths: captures long-range dependencies better than BERT on long texts
  (>200 words); permutation LM objective provides bidirectional context
- Recommended modes: full_ft for best results, peft for large on limited VRAM
- Memory note: XLNet is more memory-intensive than BERT of similar size due
  to relative positional encoding computation

### Partial Fine-Tuning Recommendations

- Freeze bottom half of encoder layers by default (6 of 12 for base models,
  12 of 24 for large models)
- Embedding layer is always frozen in partial_ft mode
- Learning rate: 2e-5 to 5e-5 (same as full fine-tuning)
- Epochs: 3-5 (slightly more than full fine-tuning to compensate for fewer
  trainable params)
- Benefits:
  - ~40-50% faster training than full FT
  - Less overfitting on small datasets (< 5K samples)
  - Preserves general language knowledge in frozen bottom layers
- When to use: medium-sized datasets (5K-20K), or when full FT overfits

### PEFT (LoRA) Recommendations

- LoRA rank (r): 4-16, default 8 (higher = more capacity, more memory)
  - r=4: minimal memory, good for quick experiments
  - r=8: balanced (recommended default)
  - r=16: near full-FT quality on some tasks
- LoRA alpha: 8-32, default 16 (scaling factor; alpha/r = effective rank)
- LoRA dropout: 0.0-0.2, default 0.1
- Target modules: "query" and "value" projections (standard for all BERT-family
  models; peft auto-maps these to architecture-specific parameter names)
- Learning rate: 1e-4 to 5e-4 (10-25x higher than full fine-tuning;
  fewer params = can use higher LR)
- Epochs: 5-10 (more needed since fewer trainable params per step)
- Weight decay: 0.0-0.01 (lower than full FT since adapters are small)
- Memory savings:
  - Trainable params: ~0.1-0.5% of base model
  - VRAM: ~3-4 GB for base models (vs 6-8 GB for full FT)
  - Gradients + optimizer states are drastically smaller
- Adapter size on disk: ~1-5 MB (vs 400+ MB for full model)
- Benefits:
  - Trainable on 4GB VRAM GPUs
  - Fast iteration — swap adapters without reloading base model
  - Catastrophic forgetting is virtually eliminated
- When to use: low-VRAM GPUs, multi-task scenarios (one adapter per task),
  large model + small dataset combinations

---

## D. Quick Reference: When to Use What

| Scenario | Recommended Approach | Priority |
|----------|---------------------|----------|
| Tiny dataset (< 500) | Traditional ML (BoW + LR) | 1 |
| Small dataset (500-5k) | BoW + LR/SVM, TF-IDF + RF | 1 |
| Medium dataset (5k-50k) | TF-IDF + SVM/LR, DL with fixed embeddings (any type) | 1-2 |
| Large dataset (50k-500k) | DL (LSTM/GRU/CNN) with GloVe/Word2Vec (fine-tuned) | 1 |
| Noisy/user-generated text | FastText + TextCNN (subword-aware) | 1 |
| CPU-only, medium data | LSTM/GRU (single layer) with fixed embeddings | 2 |
| Very large dataset (> 500k) | Transformers (fine-tuned) | 1 |
| GPU available, any size | DL + Transformers become viable | 1 |
| CPU only, any size | Traditional ML first; DL only if needed | 1 |
| Short texts (< 50 words) | TextCNN, BoW approaches — CNN excels at local n-gram patterns | 1 |
| Medium texts (50-200 words) | BiLSTM/BiGRU are solid defaults; BiGRU ~15% faster | 1 |
| Long texts (> 200 words) | LSTM/GRU + Attention, Transformers — attention learns which spans matter | 1 |
| Imbalanced classes | Use class_weight='balanced' or weighted loss | All |
| Quick baseline needed | TF-IDF + Logistic Regression (within 1-3% of best model) | 1 |
| Best accuracy for benchmark | DeBERTa-v3 fine-tuned | 3 |
| Production / low latency | DistilBERT or BoW + fast model; SGDClassifier for >100k samples | 1-2 |
| Low-VRAM GPU (2-4GB) | ALBERT, ELECTRA small, DistilBERT + LoRA | 2 |
| Long texts (>200 words) | XLNet, LSTM/GRU + Attention, Transformers | 2 |
| Want interpretable predictions | GRU/LSTM + Attention (attention weights), or LR (coefficient ranking) | 2 |

---

## Sources
- Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers"
- Liu et al. (2019) "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- He et al. (2021) "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"
- Kim (2014) "Convolutional Neural Networks for Sentence Classification"
- Joulin et al. (2017) "Bag of Tricks for Efficient Text Classification" (fastText)
- Sanh et al. (2019) "DistilBERT, a distilled version of BERT"
- Scikit-learn documentation: text feature extraction best practices
- Optuna documentation: hyperparameter optimization for NLP
- ResearchGate (2025) "Optimized Text Classification Performance Using Support
  Vector Classifiers and Deep Neural Networks" — SVM Linear (82%) vs RBF (76%)
  on TF-IDF sentiment classification
- Nature Scientific Reports (2025) "Hyperparameter settings and rationale
  for models used in multi-stage sentiment analysis" — C=1.0 default, grid
  search [0.1, 1, 10] for SVM
- CSDN / community benchmarks (2025) "TextCNN-BiLSTM combined model
  hyperparameter optimization" — hybrid architectures and Optuna tuning
- HuggingFace community (2024-2025): BERT-family learning rate sensitivity
  benchmarks, batch size vs VRAM guidelines
