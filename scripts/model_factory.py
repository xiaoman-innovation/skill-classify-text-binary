"""
Model factory: all architectures for English binary text classification.

Categories:
  A. Traditional ML  -- SVM, LogisticRegression, RandomForest, MultinomialNB
  B. Deep Learning   -- TextCNN, BiLSTM, StackedLSTM, LSTMAttention,
                         BiGRU, StackedGRU, GRUAttention
  C. Transformers    -- BERT, RoBERTa, DeBERTa, DistilBERT
"""

from __future__ import annotations
from dataclasses import dataclass, field
import warnings

import numpy as np

import os as _os

# Disable progress bars by default (cleaner logs), but do NOT force offline mode —
# auto-fallback logic below handles network issues adaptively.
if not _os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"):
    _os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


# ---------------------------------------------------------------------------
# Network-aware HuggingFace download with multi-source fallback
# ---------------------------------------------------------------------------

# Cache the working endpoint across calls within the same process.
_WORKING_HF_ENDPOINT = None
_WORKING_MODELSCOPE = None  # True / False / None (untested)


def _classify_hf_error(exc: Exception) -> str:
    """Classify an HF download error into a human-readable type."""
    msg = str(exc).lower()
    if "getaddrinfo" in msg or "name or service not known" in msg or "nodename" in msg:
        return "DNS"
    if "timed out" in msg or "timeout" in msg or "timedout" in msg:
        return "timeout"
    if "403" in msg or "forbidden" in msg:
        return "403_forbidden"
    if "connection refused" in msg or "refused" in msg:
        return "connection_refused"
    if "connection reset" in msg:
        return "connection_reset"
    if "ssl" in msg or "certificate" in msg or "tls" in msg:
        return "SSL"
    if "offline" in msg and "cannot" in msg:
        return "offline_mode"
    return "unknown"


def _try_load_from_local_cache(loader, model_name, kwargs):
    """
    Try loading model from local caches BEFORE any network attempt.

    Checks (in order):
      1. HuggingFace cache  — ~/.cache/huggingface/hub/  (via local_files_only=True)
      2. ModelScope cache   — ~/.cache/modelscope/hub/models/<name>/

    Returns the loaded model on cache hit, or None if not found locally.
    """
    import os as _os_inner
    from pathlib import Path

    # ---- 1. HuggingFace cache (local_files_only, no network) ----
    # Save HF_ENDPOINT before modifying; restore on return
    saved_endpoint = _os_inner.environ.get("HF_ENDPOINT", None)
    try:
        # Temporarily clear HF_ENDPOINT so the loader uses the default
        # cache path (which may contain models from past successful downloads).
        _os_inner.environ.pop("HF_ENDPOINT", None)
        result = loader(model_name, local_files_only=True, **kwargs)
        print(f"  [CACHE] Model loaded from HuggingFace cache", flush=True)
        return result
    except Exception:
        pass
    finally:
        if saved_endpoint:
            _os_inner.environ["HF_ENDPOINT"] = saved_endpoint

    # ---- 2. ModelScope cache ----
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        modelscope_cache = Path.home() / ".cache" / "modelscope" / "hub" / "models" / model_name
        if modelscope_cache.exists():
            result = loader(str(modelscope_cache), local_files_only=True, **kwargs)
            print(f"  [CACHE] Model loaded from ModelScope cache → {modelscope_cache}", flush=True)
            return result
    except Exception:
        pass

    return None


def _hf_from_pretrained_with_fallback(loader, model_name, **kwargs):
    """
    Auto-adapting HuggingFace downloader with multi-source fallback.

    Load order (cache-first):
      0. Local cache     — HuggingFace cache + ModelScope cache (0 network)
      1. HF endpoint     — default huggingface.co or user-set HF_ENDPOINT
      2. hf-mirror.com   — mainland China community mirror (via HF_ENDPOINT)
      3. ModelScope      — Alibaba-hosted mirror (via modelscope package, if installed)

    Each network attempt has a 30s timeout. On first success, caches the working source
    so subsequent calls within the same process use it directly.

    When ALL sources fail, raises a detailed error with diagnosis hints instead
    of a generic "connection failed" message.
    """
    import os as _os_inner

    global _WORKING_HF_ENDPOINT, _WORKING_MODELSCOPE

    # ---- layer 0: local cache (fast path, 0 network) ----
    result = _try_load_from_local_cache(loader, model_name, kwargs)
    if result is not None:
        return result

    # ---- network path: build ordered candidate list ----
    current_endpoint = _os_inner.environ.get("HF_ENDPOINT", None)

    # ---- build ordered candidate list ----
    hf_endpoints = []
    if current_endpoint:
        hf_endpoints.append(current_endpoint)
    if _WORKING_HF_ENDPOINT and _WORKING_HF_ENDPOINT not in hf_endpoints:
        hf_endpoints.append(_WORKING_HF_ENDPOINT)
    if "https://hf-mirror.com" not in hf_endpoints:
        hf_endpoints.append("https://hf-mirror.com")
    if None not in hf_endpoints:
        hf_endpoints.append(None)  # default huggingface.co

    # ---- layer 1 & 2: try HF endpoints ----
    hf_errors = []
    last_error = None
    for ep in hf_endpoints:
        try:
            if ep is None:
                _os_inner.environ.pop("HF_ENDPOINT", None)
            else:
                _os_inner.environ["HF_ENDPOINT"] = ep
            result = loader(model_name, **kwargs)
            # success — persist
            if ep:
                _os_inner.environ["HF_ENDPOINT"] = ep
                _WORKING_HF_ENDPOINT = ep
            else:
                _os_inner.environ.pop("HF_ENDPOINT", None)
                _WORKING_HF_ENDPOINT = None
            if ep and ep != current_endpoint:
                print(f"  [ADAPT] HF endpoint auto-switched to {ep}", flush=True)
            return result
        except Exception as e:
            err_type = _classify_hf_error(e)
            source = ep or "huggingface.co"
            hf_errors.append(f"{source}: [{err_type}] {str(e)[:120]}")
            last_error = e
            continue

    # ---- layer 3: try ModelScope (if installed and not yet tried) ----
    if _WORKING_MODELSCOPE is not False and not _os_inner.environ.get("HF_HUB_OFFLINE"):
        modelscope_result = _try_modelscope_download(loader, model_name, kwargs)
        if modelscope_result is not None:
            _WORKING_MODELSCOPE = True
            # Restore original HF_ENDPOINT so later HF calls work normally
            if current_endpoint:
                _os_inner.environ["HF_ENDPOINT"] = current_endpoint
            else:
                _os_inner.environ.pop("HF_ENDPOINT", None)
            return modelscope_result
        _WORKING_MODELSCOPE = False

    # ---- all layers failed — restore env and raise detailed error ----
    if current_endpoint:
        _os_inner.environ["HF_ENDPOINT"] = current_endpoint
    else:
        _os_inner.environ.pop("HF_ENDPOINT", None)

    # Build a detailed diagnostic message
    error_lines = [f"All download sources failed for model '{model_name}'."]
    error_lines.append("Attempts:")
    for i, err in enumerate(hf_errors, 1):
        error_lines.append(f"  {i}. {err}")
    if _WORKING_MODELSCOPE is False:
        error_lines.append("  —  modelscope: unavailable (package not installed or download failed)")

    # Add actionable hints
    error_lines.append("")
    error_lines.append("Troubleshooting:")
    error_lines.append("  1. Check network: curl -I https://huggingface.co")
    error_lines.append("  2. Use mirror: set HF_ENDPOINT=https://hf-mirror.com")
    error_lines.append("  3. Use modelscope: pip install modelscope")
    error_lines.append("  4. Or skip Transformers: re-run with only traditional ML / DL models")
    raise RuntimeError("\n".join(error_lines)) from last_error


def _try_modelscope_download(loader, model_name, kwargs):
    """
    Attempt to download via ModelScope (Alibaba mirror).

    ModelScope mirrors many HuggingFace models with the same model IDs.
    Uses snapshot_download() to fetch the full repo, then calls the loader
    from the local cache.

    Returns the loaded model on success, or None if ModelScope is unavailable
    or the download fails.
    """
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        return None  # modelscope not installed — not a hard error

    import os as _os_inner
    import shutil
    from pathlib import Path

    try:
        # snapshot_download caches to ~/.cache/modelscope/hub/
        local_dir = snapshot_download(model_name)
        if not local_dir or not Path(local_dir).exists():
            return None
        print(f"  [ADAPT] Model downloaded via ModelScope → {local_dir}", flush=True)

        # Point the loader to the local directory
        result = loader(local_dir, local_files_only=True, **kwargs)
        print(f"  [ADAPT] Model loaded from ModelScope cache", flush=True)
        return result
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Model Spec
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Immutable specification for a model to train."""
    name: str                    # e.g. "lr_tfidf_unigram"
    category: str                # "traditional_ml" | "deep_learning" | "transformer"
    display_name: str            # e.g. "Logistic Regression + TF-IDF (1-gram)"
    priority: int = 2            # 1=baseline(must), 2=recommended, 3=exploratory
    params: dict = field(default_factory=dict)
    vectorizer: str = "tfidf"    # For traditional ML only
    ngram_range: tuple = (1, 1)
    use_embedding: str = "none"  # "glove" | "fasttext" | "none"
    freeze_embeddings: bool = False
    training_mode: str = "full_ft"  # "feature_extraction"|"full_ft"|"partial_ft"|"peft"
    id: int = 0                  # Global model number (1-153), assigned before printing


# ---------------------------------------------------------------------------
# A. Traditional ML Models
# ---------------------------------------------------------------------------

def create_traditional_ml_model(name: str, **params) -> tuple:
    """
    Returns (model, param_search_space).
    param_search_space is a callable: fn(trial) -> dict of Optuna suggestions.
    """
    from sklearn.svm import LinearSVC, SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB

    _defaults = TRADITIONAL_ML_PARAMS.get(name, {})
    merged = {**_defaults, **params}

    if name == "svm_linear":
        model = LinearSVC(
            C=merged.get("C", 1.0),
            max_iter=merged.get("max_iter", 2000),
            loss=merged.get("loss", "squared_hinge"),
            dual=merged.get("dual", False),
            penalty=merged.get("penalty", "l2"),
            class_weight=merged.get("class_weight"),
            random_state=42,
        )
        return model, _svm_linear_suggest

    elif name == "svm_rbf":
        model = SVC(
            C=merged.get("C", 1.0),
            gamma=merged.get("gamma", "scale"),
            kernel="rbf",
            class_weight=merged.get("class_weight"),
            probability=True,
            random_state=42,
        )
        return model, _svm_rbf_suggest

    elif name == "logistic_regression":
        model = LogisticRegression(
            C=merged.get("C", 1.0),
            penalty=merged.get("penalty", "l2"),
            solver=merged.get("solver", "saga"),
            max_iter=merged.get("max_iter", 1000),
            l1_ratio=merged.get("l1_ratio", None),
            class_weight=merged.get("class_weight"),
            random_state=42,
        )
        return model, _lr_suggest

    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=merged.get("n_estimators", 200),
            max_depth=merged.get("max_depth", 30),
            min_samples_split=merged.get("min_samples_split", 5),
            min_samples_leaf=merged.get("min_samples_leaf", 2),
            max_features=merged.get("max_features", "sqrt"),
            class_weight=merged.get("class_weight", "balanced"),
            random_state=42,
            n_jobs=-1,
        )
        return model, _rf_suggest

    elif name == "multinomial_nb":
        model = MultinomialNB(
            alpha=merged.get("alpha", 1.0),
            fit_prior=merged.get("fit_prior", True),
        )
        return model, _nb_suggest

    else:
        raise ValueError(f"Unknown traditional ML model: {name}")


TRADITIONAL_ML_PARAMS = {
    "svm_linear": {"C": 1.0, "max_iter": 2000, "loss": "squared_hinge", "dual": False, "penalty": "l2", "class_weight": None},
    "svm_rbf": {"C": 1.0, "gamma": "scale", "class_weight": None},
    "logistic_regression": {"C": 1.0, "penalty": "l2", "solver": "saga", "max_iter": 1000, "class_weight": None},
    "random_forest": {"n_estimators": 200, "max_depth": 30, "min_samples_split": 5,
                      "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced"},
    "multinomial_nb": {"alpha": 1.0, "fit_prior": True},
}


def _svm_linear_suggest(trial):
    # Use a single categorical to enforce valid (penalty, loss, dual) combos.
    # Valid LinearSVC combos:
    #   penalty=l1 -> loss=squared_hinge, dual=False
    #   penalty=l2, loss=hinge -> dual=True
    #   penalty=l2, loss=squared_hinge -> dual=True or dual=False
    combo = trial.suggest_categorical("solver_combo", [
        "l2_hinge_dual",
        "l2_sqhinge_dual",
        "l2_sqhinge_nodual",
        "l1_sqhinge_nodual",
    ])
    penalty_map = {
        "l2_hinge_dual": "l2",
        "l2_sqhinge_dual": "l2",
        "l2_sqhinge_nodual": "l2",
        "l1_sqhinge_nodual": "l1",
    }
    loss_map = {
        "l2_hinge_dual": "hinge",
        "l2_sqhinge_dual": "squared_hinge",
        "l2_sqhinge_nodual": "squared_hinge",
        "l1_sqhinge_nodual": "squared_hinge",
    }
    dual_map = {
        "l2_hinge_dual": True,
        "l2_sqhinge_dual": True,
        "l2_sqhinge_nodual": False,
        "l1_sqhinge_nodual": False,
    }
    # Decode penalty/loss/dual directly from the combo so they appear
    # in best_params without re-registering (avoids Optuna dynamic-space error).
    # C range 1e-2~1e2 is the practical sweet spot (2025 benchmarks);
    # the broader 1e-3~1e3 rarely yields improvements at the extremes.
    return {
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "max_iter": trial.suggest_categorical("max_iter", [1000, 2000, 5000]),
        "loss": loss_map[combo],
        "penalty": penalty_map[combo],
        "dual": dual_map[combo],
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _svm_rbf_suggest(trial):
    return {
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "probability": True,
    }


def _lr_suggest(trial):
    return {
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
        "solver": "saga",
        "max_iter": trial.suggest_categorical("max_iter", [500, 1000, 2000]),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _rf_suggest(trial):
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
        "max_depth": trial.suggest_categorical("max_depth", [10, 20, 30, 50, None]),
        "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10, 20]),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4, 8]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
    }


def _nb_suggest(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "fit_prior": trial.suggest_categorical("fit_prior", [True, False]),
    }


# ---------------------------------------------------------------------------
# B. Deep Learning Models (PyTorch)
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Multi-filter TextCNN (Kim 2014)."""
    def __init__(self, embedding_dim=300, num_filters=100, filter_sizes=(3, 4, 5),
                 dropout=0.5, num_classes=2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len, embedding_dim)
        x = x.unsqueeze(1)  # (batch, 1, seq_len, embedding_dim)
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(x)).squeeze(3)   # (batch, num_filters, seq_len-fs+1)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # (batch, num_filters)
            conv_outs.append(c)
        x = torch.cat(conv_outs, dim=1)  # (batch, num_filters * len(filter_sizes))
        x = self.dropout(x)
        return self.fc(x)


class BiLSTM(nn.Module):
    """Bidirectional LSTM with optional multi-layer stacking."""
    def __init__(self, embedding_dim=300, hidden_dim=128, num_layers=2,
                 dropout=0.5, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout
                            if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Last timestep
        x = self.dropout(x)
        return self.fc(x)


class StackedLSTM(nn.Module):
    """Multi-layer unidirectional LSTM."""
    def __init__(self, embedding_dim=300, hidden_dim=128, num_layers=3,
                 dropout=0.5, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=False, dropout=dropout
                            if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)


class LSTMAttention(nn.Module):
    """Bidirectional LSTM with additive (Bahdanau) attention."""
    def __init__(self, embedding_dim=300, hidden_dim=128, attention_dim=64,
                 dropout=0.5, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                            bidirectional=True)
        self.attn_W = nn.Linear(hidden_dim * 2, attention_dim)
        self.attn_v = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        # Attention scores
        u = torch.tanh(self.attn_W(lstm_out))  # (batch, seq_len, attn_dim)
        scores = self.attn_v(u).squeeze(-1)     # (batch, seq_len)
        weights = F.softmax(scores, dim=1)       # (batch, seq_len)
        # Weighted sum
        x = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        x = self.dropout(x)
        return self.fc(x)


class BiGRU(nn.Module):
    """Bidirectional GRU."""
    def __init__(self, embedding_dim=300, hidden_dim=128, num_layers=2,
                 dropout=0.5, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout
                          if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        x = gru_out[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)


class StackedGRU(nn.Module):
    """Multi-layer unidirectional GRU."""
    def __init__(self, embedding_dim=300, hidden_dim=128, num_layers=3,
                 dropout=0.5, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=False, dropout=dropout
                          if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        x = gru_out[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)


class LSTM(nn.Module):
    """Unidirectional LSTM with configurable layers."""
    def __init__(self, embedding_dim=300, hidden_dim=128, num_layers=1,
                 dropout=0.5, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=False,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)


class GRU(nn.Module):
    """Unidirectional GRU with configurable layers."""
    def __init__(self, embedding_dim=300, hidden_dim=128, num_layers=1,
                 dropout=0.5, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=False,
                          dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        x = gru_out[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)


class GRUAttention(nn.Module):
    """Bidirectional GRU with additive attention."""
    def __init__(self, embedding_dim=300, hidden_dim=128, attention_dim=64,
                 dropout=0.5, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True,
                          bidirectional=True)
        self.attn_W = nn.Linear(hidden_dim * 2, attention_dim)
        self.attn_v = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        u = torch.tanh(self.attn_W(gru_out))
        scores = self.attn_v(u).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        x = torch.bmm(weights.unsqueeze(1), gru_out).squeeze(1)
        x = self.dropout(x)
        return self.fc(x)


# Map encoder type -> class
_ENCODER_CLASSES = {
    "textcnn": TextCNN,
    "bilstm": BiLSTM,
    "lstm": LSTM,
    "stacked_lstm": StackedLSTM,
    "lstm_attention": LSTMAttention,
    "bigru": BiGRU,
    "gru": GRU,
    "stacked_gru": StackedGRU,
    "gru_attention": GRUAttention,
}

_ENCODER_DISPLAY_NAMES = {
    "textcnn": "TextCNN (Kim 2014)",
    "bilstm": "BiLSTM",
    "lstm": "LSTM",
    "stacked_lstm": "Stacked LSTM",
    "lstm_attention": "LSTM + Attention",
    "bigru": "BiGRU",
    "gru": "GRU",
    "stacked_gru": "Stacked GRU",
    "gru_attention": "GRU + Attention",
}


class TextClassifier(nn.Module):
    """
    Wrapper combining: Embedding -> Encoder (CNN/LSTM/GRU) -> Classifier head.
    """
    def __init__(self, vocab_size, embedding_dim=300, encoder_type="bilstm",
                 num_classes=2, pretrained_embeddings=None, freeze_embeddings=False,
                 **encoder_kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        cls = _ENCODER_CLASSES[encoder_type]
        self.encoder = cls(embedding_dim=embedding_dim, num_classes=num_classes,
                           **encoder_kwargs)

    def forward(self, x):
        # x: (batch, seq_len) -- integer token ids
        emb = self.embedding(x)  # (batch, seq_len, embedding_dim)
        return self.encoder(emb)


class TextDataset(torch.utils.data.Dataset):
    """Map tokenized texts + labels to tensors."""
    def __init__(self, X, y, max_len=128):
        self.X = torch.LongTensor(X[:, :max_len])
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dl_model(
    encoder_type: str,
    vocab_size: int,
    embedding_dim: int = 300,
    pretrained_embeddings: np.ndarray = None,
    freeze_embeddings: bool = False,
    **encoder_kwargs,
) -> TextClassifier:
    """Factory for all deep learning models."""
    if encoder_type not in _ENCODER_CLASSES:
        raise ValueError(f"Unknown encoder type '{encoder_type}'. "
                         f"Available: {list(_ENCODER_CLASSES.keys())}")
    return TextClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        encoder_type=encoder_type,
        num_classes=2,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=freeze_embeddings,
        **encoder_kwargs,
    )


def get_dl_param_space(encoder_type: str) -> callable:
    """Returns callable fn(trial) -> dict of Optuna hyperparameter suggestions."""

    def _general_dl_suggest(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "dropout": trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.5, 0.7]),
            "epochs": trial.suggest_categorical("epochs", [5, 10, 15, 20, 30]),
            "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        }
        return params

    def _cnn_suggest(trial):
        params = _general_dl_suggest(trial)
        params.update({
            "num_filters": trial.suggest_categorical("num_filters", [64, 100, 128, 256]),
            "filter_sizes": trial.suggest_categorical("filter_sizes",
                                                      [[2, 3, 4], [3, 4, 5], [4, 5, 6]]),
        })
        return params

    def _rnn_suggest(trial):
        params = _general_dl_suggest(trial)
        params.update({
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
            "num_layers": trial.suggest_categorical("num_layers", [1, 2, 3]),
        })
        return params

    def _attn_suggest(trial):
        params = _general_dl_suggest(trial)
        params.update({
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "attention_dim": trial.suggest_categorical("attention_dim", [32, 64, 128]),
        })
        return params

    if encoder_type == "textcnn":
        return _cnn_suggest
    elif encoder_type in ("lstm_attention", "gru_attention"):
        return _attn_suggest
    else:
        return _rnn_suggest


DL_DEFAULT_PARAMS = {
    "textcnn": {"num_filters": 100, "filter_sizes": [3, 4, 5], "dropout": 0.5},
    "bilstm": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.5},
    "lstm": {"hidden_dim": 128, "num_layers": 1, "dropout": 0.5},
    "stacked_lstm": {"hidden_dim": 128, "num_layers": 3, "dropout": 0.5},
    "lstm_attention": {"hidden_dim": 128, "attention_dim": 64, "dropout": 0.5},
    "bigru": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.5},
    "gru": {"hidden_dim": 128, "num_layers": 1, "dropout": 0.5},
    "stacked_gru": {"hidden_dim": 128, "num_layers": 3, "dropout": 0.5},
    "gru_attention": {"hidden_dim": 128, "attention_dim": 64, "dropout": 0.5},
}


# ---------------------------------------------------------------------------
# C. Transformer Models (HuggingFace)
# ---------------------------------------------------------------------------

TRANSFORMER_VARIANTS = {
    "bert-base-uncased":      "google-bert/bert-base-uncased",
    "bert-large-uncased":     "google-bert/bert-large-uncased",
    "roberta-base":           "FacebookAI/roberta-base",
    "roberta-large":          "FacebookAI/roberta-large",
    "deberta-v3-base":        "microsoft/deberta-v3-base",
    "deberta-v3-large":       "microsoft/deberta-v3-large",
    "deberta-base":           "microsoft/deberta-base",
    "deberta-large":          "microsoft/deberta-large",
    "distilbert-base-uncased": "distilbert/distilbert-base-uncased",
    "albert-base-v2":         "albert/albert-base-v2",
    "albert-large-v2":        "albert/albert-large-v2",
    "electra-small-discriminator": "google/electra-small-discriminator",
    "electra-base-discriminator":  "google/electra-base-discriminator",
    "electra-large-discriminator": "google/electra-large-discriminator",
    "xlnet-base-cased":       "xlnet/xlnet-base-cased",
    "xlnet-large-cased":      "xlnet/xlnet-large-cased",
}

TRANSFORMER_DISPLAY_NAMES = {
    "bert-base-uncased":      "BERT base (uncased)",
    "bert-large-uncased":     "BERT large (uncased)",
    "roberta-base":           "RoBERTa base",
    "roberta-large":          "RoBERTa large",
    "deberta-v3-base":        "DeBERTa v3 base",
    "deberta-v3-large":       "DeBERTa v3 large",
    "deberta-base":           "DeBERTa base",
    "deberta-large":          "DeBERTa large",
    "distilbert-base-uncased": "DistilBERT base",
    "albert-base-v2":         "ALBERT base v2",
    "albert-large-v2":        "ALBERT large v2",
    "electra-small-discriminator": "ELECTRA small",
    "electra-base-discriminator":  "ELECTRA base",
    "electra-large-discriminator": "ELECTRA large",
    "xlnet-base-cased":       "XLNet base (cased)",
    "xlnet-large-cased":      "XLNet large (cased)",
}


class TransformerClassifier(nn.Module):
    """
    HuggingFace AutoModel + classification head.

    Supports four training modes:
      - "full_ft"            : all encoder params trainable
      - "feature_extraction" : freeze all encoder params
      - "partial_ft"         : freeze bottom N layers + embeddings
      - "peft"               : LoRA adapters (requires peft library)
    """
    def __init__(self, model_name: str, num_classes: int = 2,
                 training_mode: str = "full_ft",
                 freeze_encoder_layers: int = None,
                 lora_r: int = 8, lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 dropout: float = 0.1, max_seq_len: int = 256):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.training_mode = training_mode
        self._is_xlnet = "xlnet" in model_name.lower()

        config = _hf_from_pretrained_with_fallback(
            AutoConfig.from_pretrained, model_name
        )
        # BERT-family (BERT, RoBERTa, DeBERTa, ELECTRA, ALBERT) uses
        # hidden_dropout_prob / attention_probs_dropout_prob.
        # DistilBERT and XLNet use dropout / attention_dropout.
        _name_lower = model_name.lower()
        if "distilbert" in _name_lower or "xlnet" in _name_lower:
            if hasattr(config, "dropout"):
                config.dropout = dropout
            if hasattr(config, "attention_dropout"):
                config.attention_dropout = dropout
        else:
            if hasattr(config, "hidden_dropout_prob"):
                config.hidden_dropout_prob = dropout
            if hasattr(config, "attention_probs_dropout_prob"):
                config.attention_probs_dropout_prob = dropout

        self.encoder = _hf_from_pretrained_with_fallback(
            AutoModel.from_pretrained, model_name, config=config
        )
        self.hidden_size = config.hidden_size
        self.dropout = dropout

        self._apply_training_mode(training_mode, model_name,
                                  freeze_encoder_layers, lora_r,
                                  lora_alpha, lora_dropout, num_classes)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_classes),
        )

        # ── per-device weight-precision strategy ──────────────────
        # FP32 is the safe default: all ops & optimizer states use the
        # same dtype, no GradScaler needed, compatible with every GPU.
        #   VRAM < 4 GB → match encoder native dtype (may be FP16) to
        #                 save ~50 % parameter memory.
        #   VRAM ≥ 4 GB → force FP32.  Use autocast (applied in the
        #                 training loop) for compute speed instead of
        #                 lowering weight precision.
        #   CPU / MPS   → force FP32 (AMP not beneficial).
        # ──────────────────────────────────────────────────────────
        try:
            import torch as _torch
            _vram_gb = 0.0
            if _torch.cuda.is_available():
                _vram_gb = _torch.cuda.get_device_properties(0).total_memory / (1024**3)
            _low_vram = 0 < _vram_gb < 4.0
        except Exception:
            _low_vram = False

        encoder_dtype = next(self.encoder.parameters()).dtype
        cls_dtype = next(self.classifier.parameters()).dtype

        if _low_vram:
            # Scarce VRAM: let the encoder stay in its native dtype and
            # match the classifier to it.
            if encoder_dtype != cls_dtype:
                self.classifier = self.classifier.to(encoder_dtype)
        else:
            # Sufficient VRAM (or CPU): force FP32 for both.
            if encoder_dtype != _torch.float32:
                self.encoder = self.encoder.to(_torch.float32)
            if cls_dtype != _torch.float32:
                self.classifier = self.classifier.to(_torch.float32)

    # ------------------------------------------------------------------
    # Training mode routing
    # ------------------------------------------------------------------

    def _apply_training_mode(self, mode, model_name, freeze_layers,
                             lora_r, lora_alpha, lora_dropout, num_classes):
        if mode == "feature_extraction":
            self._freeze_all_encoder()
        elif mode == "partial_ft":
            n = freeze_layers if freeze_layers is not None else self._default_frozen_layers()
            self._freeze_bottom_layers(n)
        elif mode == "peft":
            self._apply_lora(model_name, lora_r, lora_alpha, lora_dropout, num_classes)
        # "full_ft": nothing to do

    def _freeze_all_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _default_frozen_layers(self) -> int:
        """Return half the encoder layers (floor)."""
        import re
        max_layer = -1
        for name, _ in self.encoder.named_parameters():
            m = re.search(r'(?:encoder|transformer|electra|albert)\.layer\.(\d+)', name)
            if m:
                max_layer = max(max_layer, int(m.group(1)))
        if max_layer < 0:
            return 6  # fallback: freeze 6 of 12
        # ALBERT shares parameters across layer groups — partial freezing
        # of individual layers is meaningless, so only freeze embeddings.
        if "albert" in self.model_name.lower():
            return -1  # signal: freeze embeddings only, not layers
        return (max_layer + 1) // 2

    def _freeze_bottom_layers(self, num_layers_to_freeze: int):
        """Freeze bottom N encoder layers + embeddings.
        num_layers_to_freeze=0 means freeze no layers (embeddings remain trainable
        unless this is ALBERT, where layer freezing is meaningless due to shared params)."""
        import re
        if num_layers_to_freeze < 0:
            # Negative: freeze embeddings only (ALBERT path)
            for name, param in self.encoder.named_parameters():
                if 'embedding' in name or 'embeddings' in name:
                    param.requires_grad = False
            return
        if num_layers_to_freeze == 0:
            return  # freeze nothing
        for name, param in self.encoder.named_parameters():
            if 'embedding' in name or 'embeddings' in name:
                param.requires_grad = False
                continue
            m = re.search(r'(?:encoder|transformer|electra|deberta)\.layer\.(\d+)', name)
            if m and int(m.group(1)) < num_layers_to_freeze:
                param.requires_grad = False

    def _apply_lora(self, model_name, r, alpha, dropout, num_classes):
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            import warnings
            warnings.warn(
                "peft library not installed. Install with: pip install peft. "
                "Falling back to feature extraction mode."
            )
            self._freeze_all_encoder()
            self.training_mode = "feature_extraction"
            return

        # XLNet uses short attention module names (rel_attn.q, rel_attn.v)
        # vs BERT-family (query, value).  Use the full sub-module prefix to avoid
        # false matches on single-letter substrings like "q" or "v".
        target_modules = (["rel_attn.q", "rel_attn.v"] if self._is_xlnet
                          else ["query", "value"])

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        # Update hidden_size in case peft wrapping changed it
        try:
            self.hidden_size = self.encoder.config.hidden_size
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if not self._is_xlnet:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        if self._is_xlnet:
            # XLNet uses the last token's representation for classification
            # (no [CLS] token — autoregressive architecture)
            pooled = outputs.last_hidden_state[:, -1, :]
        else:
            # BERT-family: use [CLS] token (position 0)
            pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)


class TransformerDataset(torch.utils.data.Dataset):
    """HuggingFace tokenized dataset."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def create_transformer_model(
    model_name: str,
    num_classes: int = 2,
    training_mode: str = "full_ft",
    freeze_encoder_layers: int = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    dropout: float = 0.1,
    max_seq_len: int = 256,
) -> TransformerClassifier:
    """Load a pretrained HF transformer model with classification head."""
    if model_name not in TRANSFORMER_VARIANTS:
        raise ValueError(f"Unknown transformer model '{model_name}'. "
                         f"Available: {list(TRANSFORMER_VARIANTS.keys())}")

    hf_name = TRANSFORMER_VARIANTS[model_name]
    return TransformerClassifier(
        model_name=hf_name,
        num_classes=num_classes,
        training_mode=training_mode,
        freeze_encoder_layers=freeze_encoder_layers,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )


def get_transformer_param_space(model_name: str = "", training_mode: str = "full_ft") -> callable:
    """Returns callable fn(trial) -> dict of Optuna suggestions for transformers."""
    def suggest(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "epochs": trial.suggest_categorical("epochs", [2, 3, 4, 5]),
            "dropout": trial.suggest_categorical("dropout", [0.1, 0.2, 0.3]),
            "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.0, 0.06, 0.1]),
            "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.1]),
            "max_seq_len": trial.suggest_categorical("max_seq_len", [128, 256, 512]),
        }
        if training_mode == "peft":
            params.update({
                "lora_r": trial.suggest_categorical("lora_r", [4, 8, 16]),
                "lora_alpha": trial.suggest_categorical("lora_alpha", [8, 16, 32]),
                "lora_dropout": trial.suggest_categorical("lora_dropout", [0.0, 0.1, 0.2]),
            })
        return params
    return suggest


TRANSFORMER_DEFAULT_PARAMS = {
    "learning_rate": 2e-5,
    "batch_size": 16,
    "epochs": 3,
    "dropout": 0.1,
    "warmup_ratio": 0.06,
    "weight_decay": 0.01,
    "max_seq_len": 256,
}


# ---------------------------------------------------------------------------
# Unified Factory
# ---------------------------------------------------------------------------

def create_model(spec: ModelSpec, vocab_size: int = None, **overrides):
    """Single entry point. Returns (model, param_search_space)."""
    if spec.category == "traditional_ml":
        return create_traditional_ml_model(spec.name, **(spec.params | overrides))
    elif spec.category == "deep_learning":
        if vocab_size is None:
            raise ValueError("vocab_size is required for deep learning models")
        encoder_params = {**DL_DEFAULT_PARAMS.get(spec.name, {}), **spec.params, **overrides}
        embedding_dim = encoder_params.pop("embedding_dim", 300)
        # Strip training-only params — they don't belong in the encoder constructor
        for _train_key in ("learning_rate", "batch_size", "epochs",
                           "optimizer", "weight_decay", "max_seq_len"):
            encoder_params.pop(_train_key, None)
        model = create_dl_model(
            encoder_type=spec.name,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            freeze_embeddings=spec.freeze_embeddings,
            **encoder_params,
        )
        return model, get_dl_param_space(spec.name)
    elif spec.category == "transformer":
        params = TRANSFORMER_DEFAULT_PARAMS | spec.params | overrides
        model = create_transformer_model(
            model_name=spec.name,
            training_mode=spec.training_mode,
            freeze_encoder_layers=params.get("freeze_encoder_layers", None),
            lora_r=params.get("lora_r", 8),
            lora_alpha=params.get("lora_alpha", 16),
            lora_dropout=params.get("lora_dropout", 0.1),
            dropout=params.get("dropout", 0.1),
            max_seq_len=params.get("max_seq_len", 256),
        )
        return model, get_transformer_param_space(spec.name, spec.training_mode)
    else:
        raise ValueError(f"Unknown model category: {spec.category}")


# ---------------------------------------------------------------------------
# Default Model Specs
# ---------------------------------------------------------------------------

def get_default_specs() -> list[ModelSpec]:
    """
    Return all 153 ModelSpec entries across all three categories with sensible
    default hyperparameters and priorities.
    """
    specs = []

    # --- A. Traditional ML ---
    ml_defs = [
        ("svm_linear", "SVM (LinearSVC)", 1),
        ("svm_rbf", "SVM (RBF kernel)", 3),
        ("logistic_regression", "Logistic Regression", 1),
        ("random_forest", "Random Forest", 2),
        ("multinomial_nb", "Multinomial NB", 2),
    ]
    vectorizers = [
        ("count", "Count (1-gram)", (1, 1), 1),
        ("count_bigram", "Count (1,2-gram)", (1, 2), 2),
        ("onehot", "OneHot (1-gram)", (1, 1), 3),
        ("tfidf", "TF-IDF (1-gram)", (1, 1), 1),
        ("tfidf_bigram", "TF-IDF (1,2-gram)", (1, 2), 2),
    ]

    for vec_key, vec_name, ngram, vec_priority in vectorizers:
        for ml_key, ml_name, ml_priority in ml_defs:
            # Skip impractical combinations
            if vec_key == "onehot" and ml_key in ("svm_rbf", "random_forest"):
                continue
            priority = min(vec_priority, ml_priority)  # take the more important
            specs.append(ModelSpec(
                name=ml_key,
                category="traditional_ml",
                display_name=f"{ml_name} + {vec_name}",
                priority=priority,
                vectorizer=vec_key,
                ngram_range=ngram,
                params=TRADITIONAL_ML_PARAMS[ml_key].copy(),
            ))

    # --- A. Traditional ML (cont'd) -- Dense Embedding Variants ---
    # Embedding-based traditional ML: average pretrained word vectors →
    # dense features → sklearn classifiers. RBF SVM benefits most from dense
    # representations (~100-300 dim vs sparse TF-IDF). Multinomial NB is
    # excluded because it requires non-negative features (embeddings contain
    # negative values).
    ml_embedding_defs = [
        ("svm_linear", "SVM (LinearSVC)", 2),
        ("svm_rbf", "SVM (RBF kernel)", 2),
        ("logistic_regression", "Logistic Regression", 2),
        ("random_forest", "Random Forest", 3),
    ]
    for emb_key, emb_name, emb_type in [
        ("glove", "GloVe 300d", "glove"),
        ("word2vec", "Word2Vec 300d", "word2vec"),
        ("fasttext", "FastText 300d", "fasttext"),
    ]:
        for ml_key, ml_name, ml_priority in ml_embedding_defs:
            specs.append(ModelSpec(
                name=ml_key,
                category="traditional_ml",
                display_name=f"{ml_name} + {emb_name}",
                priority=ml_priority,
                vectorizer="embedding",
                ngram_range=(1, 1),  # Not used for embeddings
                use_embedding=emb_type,
                params=TRADITIONAL_ML_PARAMS[ml_key].copy(),
            ))

    # --- B. Deep Learning ---
    dl_models = [
        ("textcnn", "TextCNN", 1),
        ("bilstm", "BiLSTM", 1),
        ("lstm", "LSTM", 1),
        ("stacked_lstm", "Stacked LSTM", 3),
        ("lstm_attention", "LSTM + Attention", 2),
        ("bigru", "BiGRU", 2),
        ("gru", "GRU", 2),
        ("stacked_gru", "Stacked GRU", 3),
        ("gru_attention", "GRU + Attention", 3),
    ]
    # (key, display_label, priority, freeze, embedding_type)
    dl_embeddings = [
        ("glove_fixed",    "GloVe 300d",              1, True,  "glove"),
        ("glove_ft",       "GloVe 300d (fine-tuned)",  2, False, "glove"),
        ("word2vec_fixed", "Word2Vec 300d",            1, True,  "word2vec"),
        ("word2vec_ft",    "Word2Vec 300d (fine-tuned)",2, False, "word2vec"),
        ("fasttext_fixed", "FastText 300d",            1, True,  "fasttext"),
        ("fasttext_ft",    "FastText 300d (fine-tuned)",2, False, "fasttext"),
    ]

    for emb_key, emb_name, emb_priority, freeze, emb_type in dl_embeddings:
        for dl_key, dl_name, dl_priority in dl_models:
            specs.append(ModelSpec(
                name=dl_key,
                category="deep_learning",
                display_name=f"{dl_name} + {emb_name}",
                priority=min(emb_priority, dl_priority),
                vectorizer="tokenize",
                use_embedding=emb_type,
                freeze_embeddings=freeze,
                params=DL_DEFAULT_PARAMS[dl_key].copy(),
            ))

    # --- C. Transformers ---
    transformer_defs = [
        ("bert-base-uncased", 1),
        ("bert-large-uncased", 2),
        ("roberta-base", 1),
        ("roberta-large", 2),
        ("deberta-base", 2),
        ("deberta-large", 3),
        ("deberta-v3-base", 2),
        ("deberta-v3-large", 3),
        ("distilbert-base-uncased", 2),
        ("albert-base-v2", 2),
        ("albert-large-v2", 3),
        ("electra-small-discriminator", 2),
        ("electra-base-discriminator", 2),
        ("electra-large-discriminator", 3),
        ("xlnet-base-cased", 2),
        ("xlnet-large-cased", 3),
    ]
    # mode_key, mode_label, mode_priority, training_mode
    tf_modes = [
        ("full_ft", "full fine-tuning", 1, "full_ft"),
        ("feature_extraction", "feature extraction", 2, "feature_extraction"),
        ("partial_ft", "partial fine-tuning", 2, "partial_ft"),
        ("peft", "PEFT (LoRA)", 2, "peft"),
    ]

    for mode_key, mode_label, mode_priority, training_mode in tf_modes:
        for tf_key, tf_priority in transformer_defs:
            specs.append(ModelSpec(
                name=tf_key,
                category="transformer",
                display_name=f"{TRANSFORMER_DISPLAY_NAMES[tf_key]} ({mode_label})",
                priority=min(mode_priority, tf_priority),
                vectorizer="tokenize_subword",
                use_embedding="pretrained",
                training_mode=training_mode,
                params=TRANSFORMER_DEFAULT_PARAMS.copy(),
            ))

    return specs


def filter_specs_by_priority(specs: list[ModelSpec], max_priority: int) -> list[ModelSpec]:
    """Filter specs to only include those at or below the given priority."""
    return [s for s in specs if s.priority <= max_priority]


def get_specs_by_category(specs: list[ModelSpec]) -> dict:
    """Group specs by category. Returns {category: [ModelSpec, ...]}."""
    groups = {"traditional_ml": [], "deep_learning": [], "transformer": []}
    for s in specs:
        groups[s.category].append(s)
    return groups


def dict_to_spec(d: dict) -> ModelSpec:
    """Convert dict back to ModelSpec."""
    return ModelSpec(
        name=d["name"],
        category=d["category"],
        display_name=d["display_name"],
        priority=d["priority"],
        params=d.get("params", {}),
        vectorizer=d.get("vectorizer", "tfidf"),
        ngram_range=tuple(d.get("ngram_range", (1, 1))),
        use_embedding=d.get("use_embedding", "none"),
        freeze_embeddings=d.get("freeze_embeddings", False),
        training_mode=d.get("training_mode", "full_ft"),
    )
