"""
Stage 4: Model Training (Baseline + Optuna Hyperparameter Tuning)

Usage:
    python step4_train.py --csv <path> --text-col <name> --label-col <name>
                          --scheme <path/to/model_scheme.json>
                          [--models <name1,name2>]   # specific models to train
                          [--mode baseline|tune|both] # default: both
                          [--cv-folds N]              # default: 5
                          [--tune-method cv|split]    # default: cv
                          [--tune-trials N]           # default: 50
                          [--output-dir <dir>]
                          [--no-mlflow]
                          [--seed N]

Output:
    output/training_results.json
    output/models/<name>_baseline.pkl
    output/models/<name>_tuned.pkl
    MLflow tracking data (if enabled)
"""

# Suppress progress bars for cleaner logs. We do NOT force offline mode here —
# _hf_from_pretrained_with_fallback in model_factory.py handles network issues
# adaptively with multi-source fallback and selective degradation.
import os as _os
if not _os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"):
    _os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import atexit
import sys
import os
import time
import json
import pathlib
import warnings
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pandas as pd

_script_dir = pathlib.Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from utils import (
    read_csv_safe, load_json, save_json, print_header, print_subheader,
    print_table, print_model_header, detect_device, set_seed,
    create_progress_bar, format_duration, setup_logging_and_warnings,
    setup_training_log, ensure_output_dir,
    resolve_embedding_path, ProgressReporter, get_amp_config,
)
from preprocessing import (
    clean_text, build_vocab, encode_texts_as_ids,
    build_embedding_matrix, get_vectorizer,
    load_embeddings, create_embedding_vectorizer,
)
from model_factory import (
    create_model, ModelSpec, dict_to_spec, TextDataset, TransformerDataset,
    _hf_from_pretrained_with_fallback,
)
from mlflow_utils import (
    setup_mlflow, mlflow_run, log_params, log_metrics,
    log_cv_results,
    log_optimization_history,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_auc_ks(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """Compute AUC and KS statistic from true labels and positive-class probabilities."""
    try:
        auc = float(roc_auc_score(y_true, y_prob))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ks = float(np.max(tpr - fpr))
        return auc, ks
    except ValueError:
        return (0.5, 0.0)


class EarlyStopping:
    """
    Simple early stopping with best-model checkpointing for DL/Transformer training.
    Monitors validation loss; stops after ``patience`` epochs without improvement.
    ``restore_best`` reloads the best state_dict on stop.
    """
    def __init__(self, patience: int = 3, min_delta: float = 1e-4,
                 restore_best: bool = True, min_epochs: int = 3):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.min_epochs = min_epochs
        self.best_score = None
        self.best_epoch = -1
        self.counter = 0
        self.early_stop = False
        self._best_state = None

    def step(self, val_loss, model, epoch: int) -> bool:
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best:
                import copy as _copy
                import torch as _torch
                self._best_state = _copy.deepcopy(
                    {k: v.cpu() for k, v in model.state_dict().items()}
                )
            return True  # improved
        self.counter += 1
        # Do not trigger early stop before min_epochs, even if patience is
        # exhausted — the model needs a minimum warmup before deciding.
        if self.counter >= self.patience and epoch >= self.min_epochs - 1:
            self.early_stop = True
        return False  # no improvement

    def load_best(self, model):
        if self._best_state is not None:
            model.load_state_dict(self._best_state)

    def summary(self) -> str:
        return f"best_epoch={self.best_epoch + 1}"


def _metric_dict(y_true, y_pred, y_prob=None):
    """Return a standard metrics dict with optional AUC/KS from y_prob."""
    d = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        d["auc"], d["ks"] = _compute_auc_ks(y_true, y_prob)
    else:
        d["auc"], d["ks"] = 0.5, 0.0
    return d


# ---------------------------------------------------------------------------
# Baseline Training
# ---------------------------------------------------------------------------



def train_baseline(
    X_train: list,
    y_train: np.ndarray,
    spec: ModelSpec,
    cv_folds: int = 5,
    random_state: int = 42,
    device_str: str = "cpu",
    embedding_path: str = None,
    output_dir: str = "output",
    X_valid: list = None,
    y_valid: np.ndarray = None,
    reporter: "ProgressReporter" = None,
) -> dict:
    """
    Train a single model with default parameters.

    When X_valid is provided (3-way split): train on full X_train,
    evaluate on X_valid once.  No CV.

    When X_valid is None (2-way split): stratified K-fold CV on X_train.
    """
    set_seed(random_state)
    start_time = time.time()

    if X_valid is not None and len(X_valid) > 0:
        # ---- 3-way: train on full train, evaluate on valid ----
        split_method = "split"
        print_subheader("Baseline: default params · train → valid (3-way)")
        fold_start = time.time()

        if spec.category == "traditional_ml":
            metrics = _train_ml_fold(X_train, y_train, X_valid, y_valid, spec,
                                     embedding_path=embedding_path)
        elif spec.category == "deep_learning":
            metrics = _train_dl_fold(X_train, y_train, X_valid, y_valid, spec,
                                     device_str, embedding_path)
        elif spec.category == "transformer":
            metrics = _train_transformer_fold(X_train, y_train, X_valid, y_valid, spec,
                                              device_str)

        metrics["fit_time"] = round(time.time() - fold_start, 2)
        f_acc = metrics['accuracy']
        f_f1 = metrics['f1']
        f_auc = metrics.get('auc', 0)
        print(f"     valid  █  acc={f_acc:.4f}  f1={f_f1:.4f}  "
              f"auc={f_auc:.4f}  \033[90m{metrics['fit_time']:.0f}s\033[0m",
              flush=True)

        # Train metrics already computed internally by _train_*_fold
        train_metrics = {
            "accuracy": metrics["train_accuracy"],
            "precision": metrics["train_precision"],
            "recall": metrics["train_recall"],
            "f1": metrics["train_f1"],
            "auc": metrics["train_auc"],
            "ks": 0.0,
        }

        total_time = time.time() - start_time
        result = {
            "name": spec.name,
            "model_name": spec.display_name,
            "category": spec.category,
            "vectorizer": spec.vectorizer,
            "ngram_range": list(spec.ngram_range),
            "training_mode": spec.training_mode,
            "use_embedding": spec.use_embedding,
            "freeze_embeddings": spec.freeze_embeddings,
            "mode": "baseline",
            "params": spec.params,
            "split_method": split_method,
            "train_metrics": train_metrics,
            "valid_metrics": metrics,
            "mean_accuracy": round(float(metrics['accuracy']), 4),
            "std_accuracy": 0.0,
            "mean_precision": round(float(metrics['precision']), 4),
            "mean_recall": round(float(metrics['recall']), 4),
            "mean_f1": round(float(metrics['f1']), 4),
            "mean_auc": round(float(metrics.get('auc', 0)), 4),
            "mean_ks": round(float(metrics.get('ks', 0)), 4),
            "total_fit_time": round(total_time, 2),
            "device_used": device_str,
        }

        print(f"     \033[1mresult\033[0m  █  acc=\033[1m{result['mean_accuracy']:.4f}\033[0m  "
              f"f1=\033[1m{result['mean_f1']:.4f}\033[0m  "
              f"auc=\033[1m{result['mean_auc']:.4f}\033[0m  "
              f"\033[90m{format_duration(total_time)}\033[0m", flush=True)

    else:
        # ---- 2-way: K-fold CV on train ----
        split_method = "cv"
        print_subheader(f"Baseline: default params · {cv_folds}-fold CV (2-way)")
        fold_results = []

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
                if reporter is not None:
                    reporter.update(phase_progress=f"fold {fold + 1}/{cv_folds}")
                fold_start = time.time()
                X_tr = [X_train[i] for i in tr_idx]
                X_va = [X_train[i] for i in va_idx]
                y_tr = y_train[tr_idx]
                y_va = y_train[va_idx]

                if spec.category == "traditional_ml":
                    metrics = _train_ml_fold(X_tr, y_tr, X_va, y_va, spec,
                                             embedding_path=embedding_path)
                elif spec.category == "deep_learning":
                    metrics = _train_dl_fold(X_tr, y_tr, X_va, y_va, spec,
                                             device_str, embedding_path)
                elif spec.category == "transformer":
                    metrics = _train_transformer_fold(X_tr, y_tr, X_va, y_va, spec,
                                                      device_str)

                metrics["fold"] = fold
                metrics["fit_time"] = round(time.time() - fold_start, 2)
                fold_results.append(metrics)
                tr_acc = metrics.get('train_accuracy', 0)
                f_acc = metrics['accuracy']
                f_f1 = metrics['f1']
                f_auc = metrics.get('auc', 0)
                bar_chars = "▁▂▃▄▅▆▇█"
                # Scale: 0.5 (random baseline) -> bar[0], 1.0 (perfect) -> bar[-1]
                bar_idx = int((f_acc - 0.5) / 0.5 * (len(bar_chars) - 1))
                bar_idx = max(0, min(bar_idx, len(bar_chars) - 1))
                bar = bar_chars[bar_idx]
                print(f"     fold {fold + 1}/{cv_folds}  {bar}  train_acc={tr_acc:.4f}  "
                      f"val_acc={f_acc:.4f}  val_f1={f_f1:.4f}  val_auc={f_auc:.4f}  "
                      f"\033[90m{metrics['fit_time']:.0f}s\033[0m",
                      flush=True)

        accs = [f["accuracy"] for f in fold_results]
        precs = [f["precision"] for f in fold_results]
        recs = [f["recall"] for f in fold_results]
        f1s = [f["f1"] for f in fold_results]
        aucs = [f["auc"] for f in fold_results]
        kss = [f["ks"] for f in fold_results]
        tr_accs = [f.get("train_accuracy", 0) for f in fold_results]
        tr_f1s = [f.get("train_f1", 0) for f in fold_results]
        tr_aucs = [f.get("train_auc", 0) for f in fold_results]

        total_time = time.time() - start_time

        result = {
            "name": spec.name,
            "model_name": spec.display_name,
            "category": spec.category,
            "vectorizer": spec.vectorizer,
            "ngram_range": list(spec.ngram_range),
            "training_mode": spec.training_mode,
            "use_embedding": spec.use_embedding,
            "freeze_embeddings": spec.freeze_embeddings,
            "mode": "baseline",
            "params": spec.params,
            "split_method": split_method,
            "cv_folds": fold_results,
            "mean_train_accuracy": round(float(np.mean(tr_accs)), 4),
            "mean_train_f1": round(float(np.mean(tr_f1s)), 4),
            "mean_train_auc": round(float(np.mean(tr_aucs)), 4),
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std_accuracy": round(float(np.std(accs)), 4),
            "mean_precision": round(float(np.mean(precs)), 4),
            "mean_recall": round(float(np.mean(recs)), 4),
            "mean_f1": round(float(np.mean(f1s)), 4),
            "mean_auc": round(float(np.mean(aucs)), 4),
            "mean_ks": round(float(np.mean(kss)), 4),
            "total_fit_time": round(total_time, 2),
            "device_used": device_str,
        }

        print(f"     \033[1mavg\033[0m  █  train_acc=\033[1m{result['mean_train_accuracy']:.4f}\033[0m  "
              f"val_acc=\033[1m{result['mean_accuracy']:.4f}\033[0m"
              f" ± {result['std_accuracy']:.4f}  "
              f"val_f1=\033[1m{result['mean_f1']:.4f}\033[0m  "
              f"val_auc=\033[1m{result['mean_auc']:.4f}\033[0m  "
              f"\033[90m{format_duration(total_time)}\033[0m", flush=True)

    # Save baseline model (trained on full training set)
    _save_baseline_model(X_train, y_train, spec, device_str, embedding_path, output_dir)

    return result


def _train_ml_fold(X_tr, y_tr, X_val, y_val, spec, embedding_path=None):
    """Train + evaluate a traditional ML model on one fold."""
    if spec.use_embedding != "none":
        ep = resolve_embedding_path(embedding_path, spec.use_embedding)
        if ep is None:
            raise RuntimeError(
                f"Embedding '{spec.use_embedding}' is required for "
                f"{spec.display_name} but could not be resolved."
            )
        embeddings = load_embeddings(ep, spec.use_embedding)
        vectorizer = create_embedding_vectorizer(embeddings, embedding_dim=300)
    else:
        vectorizer = get_vectorizer(spec.vectorizer)

    X_tr_vec = vectorizer.fit_transform(X_tr)
    X_val_vec = vectorizer.transform(X_val)

    model, _ = create_model(spec)
    model.fit(X_tr_vec, y_tr)

    # Train metrics
    y_tr_pred = model.predict(X_tr_vec)
    try:
        y_tr_prob = model.predict_proba(X_tr_vec)[:, 1]
    except (AttributeError, NotImplementedError):
        try:
            y_tr_prob = model.decision_function(X_tr_vec)
        except (AttributeError, NotImplementedError):
            y_tr_prob = None
    train_metrics = _metric_dict(y_tr, y_tr_pred, y_tr_prob)

    # Val metrics
    y_pred = model.predict(X_val_vec)
    try:
        y_prob = model.predict_proba(X_val_vec)[:, 1]
    except (AttributeError, NotImplementedError):
        try:
            y_prob = model.decision_function(X_val_vec)
        except (AttributeError, NotImplementedError):
            y_prob = None
    val_metrics = _metric_dict(y_val, y_pred, y_prob)

    return {
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "train_auc": train_metrics["auc"],
        "accuracy": val_metrics["accuracy"],
        "precision": val_metrics["precision"],
        "recall": val_metrics["recall"],
        "f1": val_metrics["f1"],
        "auc": val_metrics["auc"],
        "ks": val_metrics["ks"],
    }


def _train_dl_fold(X_tr, y_tr, X_val, y_val, spec, device_str, embedding_path):
    """Train + evaluate a DL model on one fold."""
    import torch
    from torch.utils.data import DataLoader

    # Build vocabulary
    word2idx = build_vocab(X_tr)
    max_len = spec.params.get("max_seq_len", 128)

    # Load embeddings if specified
    pretrained = None
    ep = resolve_embedding_path(embedding_path, spec.use_embedding)
    if spec.use_embedding in ("glove", "word2vec", "fasttext") and ep:
        embeddings = load_embeddings(ep, spec.use_embedding)
        pretrained = build_embedding_matrix(word2idx, embeddings, embedding_dim=300)

    # Create model
    model, _ = create_model(spec, vocab_size=len(word2idx),
                            pretrained_embeddings=pretrained)
    model = model.to(device_str)

    # Data
    X_tr_ids = encode_texts_as_ids(X_tr, word2idx, max_len)
    X_val_ids = encode_texts_as_ids(X_val, word2idx, max_len)
    train_ds = TextDataset(X_tr_ids, y_tr, max_len)
    val_ds = TextDataset(X_val_ids, y_val, max_len)

    batch_size = spec.params.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Train
    opt_name = spec.params.get("optimizer", "Adam")
    opt_class = torch.optim.Adam if opt_name == "Adam" else torch.optim.AdamW
    optimizer = opt_class(model.parameters(),
                          lr=spec.params.get("learning_rate", 1e-3),
                          weight_decay=spec.params.get("weight_decay", 0.0))
    criterion = torch.nn.CrossEntropyLoss()
    epochs = spec.params.get("epochs", 10)

    autocast_ctx, scaler = get_amp_config(device_str)

    es = EarlyStopping(patience=5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    import time as _time
    _t0 = _time.time()
    for ep in range(epochs):
        model.train()
        _ep_loss = 0.0
        _ep_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device_str), yb.to(device_str)
            optimizer.zero_grad()
            with autocast_ctx:
                loss = criterion(model(xb), yb)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            _ep_loss += loss.item()
            _ep_batches += 1

        # Validation loss for early stopping + LR scheduling
        model.eval()
        _val_loss = 0.0
        _val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device_str), yb.to(device_str)
                _val_loss += criterion(model(xb), yb).item()
                _val_batches += 1
        val_loss = _val_loss / max(_val_batches, 1)
        scheduler.step(val_loss)

        improved = es.step(val_loss, model, ep)
        _marker = " +" if improved else ""
        _elapsed = _time.time() - _t0
        _eta = (_elapsed / (ep + 1)) * (epochs - ep - 1) if ep < epochs - 1 else 0
        print(f"     epoch {ep+1:2d}/{epochs}  loss={_ep_loss/_ep_batches:.4f}  "
              f"val_loss={val_loss:.4f}{_marker}  "
              f"\033[90m{_elapsed:.0f}s  ETA {_eta:.0f}s\033[0m", flush=True)

        lr_now = optimizer.param_groups[0]['lr']
        if es.early_stop or ep == epochs - 1:
            print(f"     \033[90mEarlyStopping: {es.summary()} (lr={lr_now:.2e})\033[0m",
                  flush=True)
            if es.early_stop:
                es.load_best(model)
            break

    # Evaluate on train
    model.eval()
    all_tr_preds, all_tr_labels, all_tr_probs = [], [], []
    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(device_str)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            all_tr_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_tr_probs.extend(probs[:, 1].cpu().tolist())
            all_tr_labels.extend(yb.tolist())
    train_metrics = _metric_dict(np.array(all_tr_labels), np.array(all_tr_preds),
                                 np.array(all_tr_probs))

    # Evaluate on val
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device_str)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())
            all_labels.extend(yb.tolist())
    val_metrics = _metric_dict(np.array(all_labels), np.array(all_preds),
                               np.array(all_probs))

    if device_str == "cuda":
        torch.cuda.empty_cache()

    return {
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "train_auc": train_metrics["auc"],
        "accuracy": val_metrics["accuracy"],
        "precision": val_metrics["precision"],
        "recall": val_metrics["recall"],
        "f1": val_metrics["f1"],
        "auc": val_metrics["auc"],
        "ks": val_metrics["ks"],
    }
def _train_transformer_fold(X_tr, y_tr, X_val, y_val, spec, device_str):
    """Train + evaluate a transformer model on one fold."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    model, _ = create_model(spec)
    model = model.to(device_str)

    tokenizer = _hf_from_pretrained_with_fallback(
        AutoTokenizer.from_pretrained,
        model.model_name if hasattr(model, 'model_name') else spec.name
    )
    max_len = spec.params.get("max_seq_len", 256)

    def tokenize(texts):
        return tokenizer(texts, padding=True, truncation=True,
                         max_length=max_len, return_tensors="pt")

    train_enc = tokenize(X_tr)
    val_enc = tokenize(X_val)
    train_ds = TransformerDataset(train_enc, y_tr)
    val_ds = TransformerDataset(val_enc, y_val)

    batch_size = spec.params.get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    epochs = spec.params.get("epochs", 3)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.params.get("learning_rate", 2e-5),
        weight_decay=spec.params.get("weight_decay", 0.01),
    )
    total_steps = epochs * len(train_loader)
    warmup_steps = int(spec.params.get("warmup_ratio", 0.06) * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = torch.nn.CrossEntropyLoss()

    autocast_ctx, scaler = get_amp_config(device_str)

    es = EarlyStopping(patience=5)

    import time as _time
    _t0 = _time.time()
    for ep in range(epochs):
        model.train()
        _ep_loss = 0.0
        _ep_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device_str)
            attention_mask = batch["attention_mask"].to(device_str)
            labels = batch["labels"].to(device_str)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device_str)
            with autocast_ctx:
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            _ep_loss += loss.item()
            _ep_batches += 1

        # Validation loss for early stopping
        model.eval()
        _val_loss = 0.0
        _val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device_str)
                attention_mask = batch["attention_mask"].to(device_str)
                labels = batch["labels"].to(device_str)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device_str)
                logits = model(input_ids, attention_mask, token_type_ids)
                _val_loss += criterion(logits, labels).item()
                _val_batches += 1
        val_loss = _val_loss / max(_val_batches, 1)

        improved = es.step(val_loss, model, ep)
        _marker = " +" if improved else ""
        _elapsed = _time.time() - _t0
        _eta = (_elapsed / (ep + 1)) * (epochs - ep - 1) if ep < epochs - 1 else 0
        print(f"     epoch {ep+1:2d}/{epochs}  loss={_ep_loss/_ep_batches:.4f}  "
              f"val_loss={val_loss:.4f}{_marker}  "
              f"\033[90m{_elapsed:.0f}s  ETA {_eta:.0f}s\033[0m", flush=True)

        if es.early_stop or ep == epochs - 1:
            print(f"     \033[90mEarlyStopping: {es.summary()} (lr={optimizer.param_groups[0]['lr']:.2e})\033[0m",
                  flush=True)
            if es.early_stop:
                es.load_best(model)
            break

    # Evaluate on train
    model.eval()
    all_tr_preds, all_tr_labels, all_tr_probs = [], [], []
    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device_str)
            attention_mask = batch["attention_mask"].to(device_str)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device_str)
            logits = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=1)
            all_tr_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_tr_probs.extend(probs[:, 1].cpu().tolist())
            all_tr_labels.extend(batch["labels"].tolist())
    train_metrics = _metric_dict(np.array(all_tr_labels), np.array(all_tr_preds),
                                 np.array(all_tr_probs))

    # Evaluate on val
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device_str)
            attention_mask = batch["attention_mask"].to(device_str)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device_str)
            logits = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())
            all_labels.extend(batch["labels"].tolist())
    val_metrics = _metric_dict(np.array(all_labels), np.array(all_preds),
                               np.array(all_probs))

    if device_str == "cuda":
        torch.cuda.empty_cache()

    return {
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "train_auc": train_metrics["auc"],
        "accuracy": val_metrics["accuracy"],
        "precision": val_metrics["precision"],
        "recall": val_metrics["recall"],
        "f1": val_metrics["f1"],
        "auc": val_metrics["auc"],
        "ks": val_metrics["ks"],
    }


def _save_baseline_model(X, y, spec, device_str, embedding_path, output_dir):
    """Save a baseline model trained on full data."""
    import joblib
    models_dir = pathlib.Path(output_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Clean texts
    X_clean = clean_text(X)

    if spec.category == "traditional_ml":
        if spec.use_embedding != "none":
            ep = resolve_embedding_path(embedding_path, spec.use_embedding)
            if ep is None:
                raise RuntimeError(
                    f"Embedding '{spec.use_embedding}' is required for "
                    f"{spec.display_name} but could not be resolved."
                )
            embeddings = load_embeddings(ep, spec.use_embedding)
            vectorizer = create_embedding_vectorizer(embeddings, embedding_dim=300)
        else:
            vectorizer = get_vectorizer(spec.vectorizer)
        X_vec = vectorizer.fit_transform(X_clean)
        model, _ = create_model(spec)
        model.fit(X_vec, y)
        key = _spec_file_key(spec)
        joblib.dump(model, models_dir / f"{key}_baseline.pkl")
        joblib.dump(vectorizer, models_dir / f"{key}_baseline_vectorizer.pkl")

    elif spec.category == "deep_learning":
        import torch
        word2idx = build_vocab(X_clean)
        joblib.dump(word2idx, models_dir / f"{spec.name}_baseline_vocab.pkl")
        pretrained = None
        ep = resolve_embedding_path(embedding_path, spec.use_embedding)
        if spec.use_embedding in ("glove", "word2vec", "fasttext") and ep:
            embeddings = load_embeddings(ep, spec.use_embedding)
            pretrained = build_embedding_matrix(word2idx, embeddings, embedding_dim=300)
        model, _ = create_model(spec, vocab_size=len(word2idx),
                                pretrained_embeddings=pretrained)
        model = model.to(device_str)
        # Quick full fit
        max_len = spec.params.get("max_seq_len", 128)
        X_ids = encode_texts_as_ids(X_clean, word2idx, max_len)
        ds = TextDataset(X_ids, y, max_len)
        loader = torch.utils.data.DataLoader(ds, batch_size=spec.params.get("batch_size", 64),
                                             shuffle=True)
        opt_name = spec.params.get("optimizer", "Adam")
        opt_class = torch.optim.Adam if opt_name == "Adam" else torch.optim.AdamW
        optimizer = opt_class(model.parameters(),
                              lr=spec.params.get("learning_rate", 1e-3),
                              weight_decay=spec.params.get("weight_decay", 0.0))
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        epochs = spec.params.get("epochs", 10)
        autocast_ctx, scaler = get_amp_config(device_str)
        import time as _time
        _t0 = _time.time()
        for ep in range(epochs):
            _ep_loss = 0.0
            _ep_batches = 0
            for xb, yb in loader:
                xb, yb = xb.to(device_str), yb.to(device_str)
                optimizer.zero_grad()
                with autocast_ctx:
                    loss = criterion(model(xb), yb)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                _ep_loss += loss.item()
                _ep_batches += 1
            _elapsed = _time.time() - _t0
            _eta = (_elapsed / (ep + 1)) * (epochs - ep - 1) if ep < epochs - 1 else 0
            print(f"     epoch {ep+1:2d}/{epochs}  loss={_ep_loss/_ep_batches:.4f}  "
                  f"\033[90m{_elapsed:.0f}s  ETA {_eta:.0f}s\033[0m", flush=True)
        torch.save(model.state_dict(), models_dir / f"{spec.name}_baseline.pt")

    elif spec.category == "transformer":
        import torch
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
        model, _ = create_model(spec)
        model = model.to(device_str)
        tokenizer = AutoTokenizer.from_pretrained(
            model.model_name if hasattr(model, 'model_name') else spec.name
        )
        tokenizer.save_pretrained(str(models_dir / f"{spec.name}_baseline_tokenizer"))
        max_len = spec.params.get("max_seq_len", 256)
        enc = tokenizer(X_clean, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        ds = TransformerDataset(enc, y)
        epochs = spec.params.get("epochs", 3)
        batch_size = spec.params.get("batch_size", 16)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=spec.params.get("learning_rate", 2e-5),
                                      weight_decay=spec.params.get("weight_decay", 0.01))
        total_steps = epochs * len(loader)
        warmup_steps = int(spec.params.get("warmup_ratio", 0.06) * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        autocast_ctx, scaler = get_amp_config(device_str)
        import time as _time
        _t0 = _time.time()
        for ep in range(epochs):
            _ep_loss = 0.0
            _ep_batches = 0
            for batch in loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device_str)
                attention_mask = batch["attention_mask"].to(device_str)
                labels = batch["labels"].to(device_str)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device_str)
                with autocast_ctx:
                    logits = model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(logits, labels)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                _ep_loss += loss.item()
                _ep_batches += 1
            _elapsed = _time.time() - _t0
            _eta = (_elapsed / (ep + 1)) * (epochs - ep - 1) if ep < epochs - 1 else 0
            print(f"     epoch {ep+1:2d}/{epochs}  loss={_ep_loss/_ep_batches:.4f}  "
                  f"\033[90m{_elapsed:.0f}s  ETA {_eta:.0f}s\033[0m", flush=True)
        actual_mode = getattr(model, 'training_mode', spec.training_mode)
        if actual_mode == "peft":
            try:
                model.encoder.save_pretrained(str(models_dir / f"{spec.name}_baseline_adapter"))
            except AttributeError:
                torch.save(model.state_dict(), models_dir / f"{spec.name}_baseline.pt")
        else:
            torch.save(model.state_dict(), models_dir / f"{spec.name}_baseline.pt")


# ---------------------------------------------------------------------------
# Tuned Training (Optuna)
# ---------------------------------------------------------------------------

def train_tuned(
    X_train: list,
    y_train: np.ndarray,
    spec: ModelSpec,
    tune_method: str = "cv",
    cv_folds: int = 5,
    n_trials: int = 50,
    random_state: int = 42,
    device_str: str = "cpu",
    embedding_path: str = None,
    output_dir: str = "output",
    epochs_override: int = None,
    X_valid: list = None,
    y_valid: np.ndarray = None,
    reporter: "ProgressReporter" = None,
) -> dict:
    """
    Optuna hyperparameter tuning for a single model.

    When X_valid is provided (3-way split): force split method using the
    pre-split valid set for evaluation.

    When X_valid is None (2-way split): use tune_method as specified
    (cv for cross-validation, split for internal 80/20 split).
    """
    import optuna

    # 3-way split with valid set: always use split mode with external valid
    if X_valid is not None and len(X_valid) > 0:
        effective_method = "split"
        print_subheader(f"Tuning: Optuna {n_trials} trials · split (3-way, pre-split valid)")
    else:
        effective_method = tune_method
        print_subheader(f"Tuning: Optuna {n_trials} trials · {effective_method}")
    set_seed(random_state)

    models_dir = pathlib.Path(output_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Get param search space
    _, suggest_fn = create_model(spec, vocab_size=10000)

    # Use MedianPruner for early stopping.
    # n_startup_trials=5: first 5 trials never pruned (establishes baseline).
    # n_warmup_steps=1: only the first reported step per trial is warmup.
    #   CV mode (3 folds): folds 1-2 can be pruned after fold 0 establishes baseline.
    #   Split mode DL (epochs): epochs 1+ can be pruned after epoch 0.
    #   Split mode Transformer: epochs 1+ can be pruned.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name=f"tune_{spec.name}",
    )

    def objective(trial):
        params = suggest_fn(trial)
        # Enforce fast-validation overrides during tuning
        if epochs_override is not None and spec.category in ("deep_learning", "transformer"):
            params["epochs"] = epochs_override
        # Cap max_seq_len and floor batch_size for fast transformer validation
        if spec.category == "transformer":
            params["max_seq_len"] = min(params.get("max_seq_len", 256), 128)
            params["batch_size"] = max(params.get("batch_size", 16), 16)  # min 16 for throughput
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if effective_method == "cv":
                return _evaluate_cv(X_train, y_train, spec, params, cv_folds,
                                    random_state, device_str, embedding_path,
                                    trial=trial)
            else:
                return _evaluate_split(X_train, y_train, spec, params,
                                       random_state, device_str, embedding_path,
                                       trial=trial,
                                       X_val=X_valid, y_val=y_valid)

    # Progress bar with nicer format
    pbar = create_progress_bar(n_trials, desc=f"     tuning", unit="trial")

    def callback(study, trial):
        pbar.update(1)
        postfix = {}
        if study.best_value is not None:
            postfix["best"] = f"{study.best_value:.4f}"
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        if n_pruned > 0:
            postfix["pruned"] = str(n_pruned)
        if postfix:
            pbar.set_postfix(postfix)
        # Update periodic progress reporter
        if reporter is not None:
            n_done = len(study.trials)
            reporter.update(phase_progress=f"trial {n_done}/{n_trials}")

    study.optimize(objective, n_trials=n_trials, callbacks=[callback],
                   show_progress_bar=False)
    pbar.close()

    total_time = time.time() - start_time

    # Collect optimization history
    history = [
        {"trial": t.number, "value": t.value, "state": str(t.state)}
        for t in study.trials if t.value is not None
    ]

    # Retrain with best params and evaluate with CV
    # Keep Optuna-searched params for reporting (not polluted with defaults)
    try:
        _tuned_best_params = dict(study.best_params)
    except ValueError:
        print(f"     \033[33m[WARN] All {n_trials} trials pruned/failed — "
              f"falling back to default params.\033[0m", flush=True)
        _tuned_best_params = {}
    # Merge with spec defaults so hardcoded (non-Optuna) suggest values
    # (e.g. solver="saga" in _lr_suggest) are included in the final evaluation.
    best_params = deepcopy(spec.params)
    best_params.update(_tuned_best_params)
    # Ensure epochs_override takes precedence over Optuna-suggested epochs
    if epochs_override is not None and spec.category in ("deep_learning", "transformer"):
        best_params["epochs"] = epochs_override
    # Decode solver_combo → individual loss/penalty/dual for LinearSVC.
    # The _svm_linear_suggest function derives these from a combo key to
    # guarantee valid (penalty, loss, dual) triples, but the derived values
    # are not Optuna trial parameters — only solver_combo is stored.
    if "solver_combo" in best_params:
        _SC_PENALTY = {"l2_hinge_dual": "l2", "l2_sqhinge_dual": "l2",
                       "l2_sqhinge_nodual": "l2", "l1_sqhinge_nodual": "l1"}
        _SC_LOSS = {"l2_hinge_dual": "hinge", "l2_sqhinge_dual": "squared_hinge",
                    "l2_sqhinge_nodual": "squared_hinge",
                    "l1_sqhinge_nodual": "squared_hinge"}
        _SC_DUAL = {"l2_hinge_dual": True, "l2_sqhinge_dual": True,
                    "l2_sqhinge_nodual": False, "l1_sqhinge_nodual": False}
        combo = best_params.pop("solver_combo")
        best_params["penalty"] = _SC_PENALTY[combo]
        best_params["loss"] = _SC_LOSS[combo]
        best_params["dual"] = _SC_DUAL[combo]
        # Also update the reporting params
        _tuned_best_params.pop("solver_combo", None)
        _tuned_best_params["penalty"] = _SC_PENALTY[combo]
        _tuned_best_params["loss"] = _SC_LOSS[combo]
        _tuned_best_params["dual"] = _SC_DUAL[combo]
    # Filter out internal params for display
    display_params = {k: v for k, v in best_params.items()
                      if k not in ("solver_combo",)}
    try:
        best_trial_num = study.best_trial.number
        best_val = study.best_value
    except ValueError:
        best_trial_num = -1
        best_val = float('nan')
    print(f"     best trial #{best_trial_num}  "
          f"val={best_val:.4f}  "
          f"\033[90m{display_params}\033[0m", flush=True)

    # Re-evaluate with best params
    # 3-way: train on full train, evaluate on valid
    # 2-way: full CV on train
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if X_valid is not None and len(X_valid) > 0:
            # 3-way: single train→valid evaluation
            valid_metrics = _quick_fit_eval(
                X_train, y_train, X_valid, y_valid, spec, best_params,
                device_str, embedding_path)
            final_result = {
                "folds": [valid_metrics],
                "train_metrics": {
                    "accuracy": valid_metrics.get("train_accuracy", 0),
                    "precision": valid_metrics.get("train_precision", 0),
                    "recall": valid_metrics.get("train_recall", 0),
                    "f1": valid_metrics.get("train_f1", 0),
                    "auc": valid_metrics.get("train_auc", 0),
                },
                "mean_train_accuracy": valid_metrics.get("train_accuracy", 0),
                "mean_train_f1": valid_metrics.get("train_f1", 0),
                "mean_train_auc": valid_metrics.get("train_auc", 0),
                "mean_accuracy": valid_metrics["accuracy"],
                "std_accuracy": 0.0,
                "mean_precision": valid_metrics["precision"],
                "mean_recall": valid_metrics["recall"],
                "mean_f1": valid_metrics["f1"],
                "mean_auc": valid_metrics.get("auc", 0),
                "mean_ks": valid_metrics.get("ks", 0),
            }
            split_method = "split"
        else:
            final_result = _evaluate_cv_full(X_train, y_train, spec, best_params,
                                             cv_folds, random_state, device_str,
                                             embedding_path)
            split_method = "cv"

    result = {
        "name": spec.name,
        "model_name": spec.display_name,
        "category": spec.category,
        "vectorizer": spec.vectorizer,
        "ngram_range": list(spec.ngram_range),
        "training_mode": spec.training_mode,
        "mode": "tuned",
        "split_method": split_method,
        "best_params": _tuned_best_params,
        "best_trial_number": best_trial_num,
        "best_value": best_val,
        "cv_folds": final_result["folds"],
        "train_metrics": final_result.get("train_metrics", {}),
        "mean_train_accuracy": final_result.get("mean_train_accuracy", 0),
        "mean_train_f1": final_result.get("mean_train_f1", 0),
        "mean_train_auc": final_result.get("mean_train_auc", 0),
        "mean_accuracy": final_result["mean_accuracy"],
        "std_accuracy": final_result["std_accuracy"],
        "mean_precision": final_result["mean_precision"],
        "mean_recall": final_result["mean_recall"],
        "mean_f1": final_result["mean_f1"],
        "mean_auc": final_result["mean_auc"],
        "mean_ks": final_result["mean_ks"],
        "optimization_history": history,
        "total_trials": n_trials,
        "total_fit_time": round(total_time, 2),
        "device_used": device_str,
    }

    print(f"     \033[1mtuned\033[0m  █  train_acc=\033[1m{result['mean_train_accuracy']:.4f}\033[0m  "
          f"val_acc=\033[1m{result['mean_accuracy']:.4f}\033[0m"
          f" ± {result['std_accuracy']:.4f}  "
          f"val_f1=\033[1m{result['mean_f1']:.4f}\033[0m  "
          f"val_auc=\033[1m{result['mean_auc']:.4f}\033[0m  "
          f"\033[90m{format_duration(total_time)}\033[0m", flush=True)

    # Save tuned model
    _save_tuned_model(X_train, y_train, spec, best_params, device_str,
                      embedding_path, output_dir)

    return result


def _evaluate_cv(X, y, spec, params, cv_folds, random_state, device_str, embedding_path,
                 trial=None):
    """Evaluate params using a lighter-weight inner CV (3-fold). Reports per-fold
    accuracy to Optuna for pruning when a trial object is provided."""
    inner_folds = min(3, cv_folds)
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr = [X[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_tr = y[train_idx]
        y_val = y[val_idx]
        metrics = _quick_fit_eval(X_tr, y_tr, X_val, y_val, spec, params,
                                  device_str, embedding_path, trial=trial)
        fold_acc = metrics["accuracy"]
        scores.append(fold_acc)
        if trial is not None:
            trial.report(fold_acc, step=fold)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned(
                    f"Pruned at fold {fold} with accuracy={fold_acc:.4f}"
                )
    return float(np.mean(scores))


def _evaluate_split(X, y, spec, params, random_state, device_str, embedding_path,
                    trial=None, X_val=None, y_val=None):
    """Evaluate params using a train/val split.

    If X_val is provided, use it directly as the validation set.
    Otherwise, create an internal 80/20 split from X.
    Passes trial to _quick_fit_eval for per-epoch pruning on DL/Transformer models.
    """
    if X_val is not None and len(X_val) > 0:
        X_tr, y_tr = X, y
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_state,
        )
    metrics = _quick_fit_eval(X_tr, y_tr, X_val, y_val, spec, params,
                              device_str, embedding_path, trial=trial)
    return metrics["accuracy"]


def _quick_fit_eval(X_tr, y_tr, X_val, y_val, spec, params, device_str, embedding_path,
                    trial=None):
    """Quick single-pass fit and evaluate with given params.
    When trial is provided (Optuna tuning), reports per-epoch validation
    accuracy for DL/Transformer models, enabling MedianPruner early stopping."""
    if spec.category == "traditional_ml":
        from sklearn.svm import LinearSVC, SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import MultinomialNB

        # Merge spec defaults so tuning evaluations use the right base params
        # (e.g. solver="saga", max_iter=1000) instead of sklearn raw defaults.
        _eval_params = deepcopy(spec.params)
        _eval_params.update(params)

        if spec.use_embedding != "none":
            ep = resolve_embedding_path(embedding_path, spec.use_embedding)
            if ep is None:
                raise RuntimeError(
                    f"Embedding '{spec.use_embedding}' is required for "
                    f"{spec.display_name} but could not be resolved."
                )
            embeddings = load_embeddings(ep, spec.use_embedding)
            vectorizer = create_embedding_vectorizer(embeddings, embedding_dim=300)
        else:
            vectorizer = get_vectorizer(spec.vectorizer)

        X_tr_vec = vectorizer.fit_transform(X_tr)
        X_val_vec = vectorizer.transform(X_val)

        model_map = {
            "svm_linear": LinearSVC,
            "svm_rbf": SVC,
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "multinomial_nb": MultinomialNB,
        }
        model_cls = model_map[spec.name]
        # Filter params to those valid for the model
        valid_params = _filter_model_params(_eval_params, spec.name)
        # Ensure SVC always has probability=True for AUC computation
        if spec.name == "svm_rbf" and "probability" not in valid_params:
            valid_params["probability"] = True
        # Ensure RandomForest uses parallel training
        if spec.name == "random_forest" and "n_jobs" not in valid_params:
            valid_params["n_jobs"] = -1
        # MultinomialNB does not accept random_state
        if spec.name == "multinomial_nb":
            model = model_cls(**valid_params)
        else:
            model = model_cls(random_state=42, **valid_params)
        model.fit(X_tr_vec, y_tr)

        # Train metrics
        y_tr_pred = model.predict(X_tr_vec)
        try:
            y_tr_prob = model.predict_proba(X_tr_vec)[:, 1]
        except (AttributeError, NotImplementedError):
            try:
                y_tr_prob = model.decision_function(X_tr_vec)
            except (AttributeError, NotImplementedError):
                y_tr_prob = None
        train_metrics = _metric_dict(y_tr, y_tr_pred, y_tr_prob)

        y_pred = model.predict(X_val_vec)
        try:
            y_prob = model.predict_proba(X_val_vec)[:, 1]
        except (AttributeError, NotImplementedError):
            try:
                y_prob = model.decision_function(X_val_vec)
            except (AttributeError, NotImplementedError):
                y_prob = None
        val_metrics = _metric_dict(y_val, y_pred, y_prob)
        return {
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
            "auc": val_metrics["auc"],
            "ks": val_metrics["ks"],
        }

    elif spec.category == "deep_learning":
        import torch
        from torch.utils.data import DataLoader

        word2idx = build_vocab(X_tr)
        max_len = params.get("max_seq_len", 128)
        pretrained = None
        ep = resolve_embedding_path(embedding_path, spec.use_embedding)
        if spec.use_embedding in ("glove", "word2vec", "fasttext") and ep:
            embeddings = load_embeddings(ep, spec.use_embedding)
            pretrained = build_embedding_matrix(word2idx, embeddings, embedding_dim=300)

        model, _ = create_model(spec, vocab_size=len(word2idx),
                                pretrained_embeddings=pretrained, **params)
        model = model.to(device_str)

        X_tr_ids = encode_texts_as_ids(X_tr, word2idx, max_len)
        X_val_ids = encode_texts_as_ids(X_val, word2idx, max_len)
        train_ds = TextDataset(X_tr_ids, y_tr, max_len)
        val_ds = TextDataset(X_val_ids, y_val, max_len)
        train_loader = DataLoader(train_ds, batch_size=params.get("batch_size", 64),
                                  shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=params.get("batch_size", 64))

        opt_name = params.get("optimizer", "Adam")
        opt_class = torch.optim.Adam if opt_name == "Adam" else torch.optim.AdamW
        optimizer = opt_class(model.parameters(),
                              lr=params.get("learning_rate", 1e-3),
                              weight_decay=params.get("weight_decay", 0.0))
        criterion = torch.nn.CrossEntropyLoss()
        epochs = params.get("epochs", 5)
        autocast_ctx, scaler = get_amp_config(device_str)

        # Early stopping only for final evaluation (no Optuna trial pruning).
        # During tuning, MedianPruner handles trial-level early stopping.
        es = EarlyStopping(patience=5) if trial is None else None
        if es is not None:
            scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
            )

        import time as _time
        _t0 = _time.time()
        for epoch in range(epochs):
            model.train()
            _ep_loss = 0.0
            _ep_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device_str), yb.to(device_str)
                optimizer.zero_grad()
                with autocast_ctx:
                    loss = criterion(model(xb), yb)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                _ep_loss += loss.item()
                _ep_batches += 1
            _elapsed = _time.time() - _t0
            _eta = (_elapsed / (epoch + 1)) * (epochs - epoch - 1) if epoch < epochs - 1 else 0

            # Validation metrics for early stopping / trial pruning
            model.eval()
            _val_loss = 0.0
            _val_batches = 0
            ep_preds, ep_probs = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device_str), yb.to(device_str)
                    logits = model(xb)
                    probs = torch.softmax(logits, dim=1)
                    _val_loss += criterion(logits, yb).item()
                    _val_batches += 1
                    ep_preds.extend(logits.argmax(dim=1).cpu().tolist())
                    ep_probs.extend(probs[:, 1].cpu().tolist())
            val_loss = _val_loss / max(_val_batches, 1)

            if trial is not None:
                ep_acc = float(accuracy_score(y_val, np.array(ep_preds)))
                trial.report(ep_acc, step=epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned(
                        f"Pruned at epoch {epoch} with accuracy={ep_acc:.4f}"
                    )

            _improved = es.step(val_loss, model, epoch) if es is not None else False
            _marker = " +" if _improved else ""
            print(f"     epoch {epoch+1:2d}/{epochs}  loss={_ep_loss/_ep_batches:.4f}  "
                  f"val_loss={val_loss:.4f}{_marker}  "
                  f"\033[90m{_elapsed:.0f}s  ETA {_eta:.0f}s\033[0m", flush=True)

            if es is not None:
                scheduler_lr.step(val_loss)
                if es.early_stop or epoch == epochs - 1:
                    print(f"     \033[90mEarlyStopping: {es.summary()}\033[0m", flush=True)
                    if es.early_stop:
                        es.load_best(model)
                    break

        # Train metrics — collect labels alongside predictions
        # (train_loader uses shuffle=True, so y_tr order ≠ prediction order)
        model.eval()
        all_tr_preds, all_tr_probs, all_tr_labels = [], [], []
        with torch.no_grad():
            for xb, yb in train_loader:
                logits = model(xb.to(device_str))
                probs = torch.softmax(logits, dim=1)
                all_tr_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_tr_probs.extend(probs[:, 1].cpu().tolist())
                all_tr_labels.extend(yb.tolist())
        train_metrics = _metric_dict(np.array(all_tr_labels), np.array(all_tr_preds),
                                     np.array(all_tr_probs))

        model.eval()
        all_preds, all_probs = [], []
        with torch.no_grad():
            for xb, _ in val_loader:
                logits = model(xb.to(device_str))
                probs = torch.softmax(logits, dim=1)
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_probs.extend(probs[:, 1].cpu().tolist())
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        val_metrics = _metric_dict(y_val, y_pred, y_prob)
        return {
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
            "auc": val_metrics["auc"],
            "ks": val_metrics["ks"],
        }

    elif spec.category == "transformer":
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup

        model, _ = create_model(spec, **params)
        model = model.to(device_str)
        tokenizer = AutoTokenizer.from_pretrained(
            model.model_name if hasattr(model, 'model_name') else spec.name
        )
        max_len = params.get("max_seq_len", 256)

        def tokenize(texts):
            return tokenizer(texts, padding=True, truncation=True,
                             max_length=max_len, return_tensors="pt")

        train_enc = tokenize(X_tr)
        val_enc = tokenize(X_val)
        train_ds = TransformerDataset(train_enc, y_tr)
        val_ds = TransformerDataset(val_enc, y_val)
        train_loader = DataLoader(train_ds, batch_size=params.get("batch_size", 16),
                                  shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=params.get("batch_size", 16))

        epochs = params.get("epochs", 2)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.get("learning_rate", 2e-5),
            weight_decay=params.get("weight_decay", 0.01),
        )
        total_steps = epochs * len(train_loader)
        warmup_steps = int(params.get("warmup_ratio", 0.06) * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        criterion = torch.nn.CrossEntropyLoss()
        autocast_ctx, scaler = get_amp_config(device_str)

        # Early stopping only for final evaluation (no Optuna trial pruning)
        es = EarlyStopping(patience=5) if trial is None else None

        import time as _time
        _t0 = _time.time()
        for epoch in range(epochs):
            model.train()
            _ep_loss = 0.0
            _ep_batches = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device_str)
                attention_mask = batch["attention_mask"].to(device_str)
                labels = batch["labels"].to(device_str)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device_str)
                with autocast_ctx:
                    logits = model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(logits, labels)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                _ep_loss += loss.item()
                _ep_batches += 1

            # Validation metrics for early stopping / trial pruning
            model.eval()
            _val_loss = 0.0
            _val_batches = 0
            ep_preds, ep_probs = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device_str)
                    attention_mask = batch["attention_mask"].to(device_str)
                    labels = batch["labels"].to(device_str)
                    token_type_ids = batch.get("token_type_ids")
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(device_str)
                    logits = model(input_ids, attention_mask, token_type_ids)
                    probs = torch.softmax(logits, dim=1)
                    _val_loss += criterion(logits, labels).item()
                    _val_batches += 1
                    ep_preds.extend(logits.argmax(dim=1).cpu().tolist())
                    ep_probs.extend(probs[:, 1].cpu().tolist())
            val_loss = _val_loss / max(_val_batches, 1)

            if trial is not None:
                ep_acc = float(accuracy_score(y_val, np.array(ep_preds)))
                trial.report(ep_acc, step=epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned(
                        f"Pruned at epoch {epoch} with accuracy={ep_acc:.4f}"
                    )

            _improved = es.step(val_loss, model, epoch) if es is not None else False
            _marker = " +" if _improved else ""
            _elapsed = _time.time() - _t0
            _eta = (_elapsed / (epoch + 1)) * (epochs - epoch - 1) if epoch < epochs - 1 else 0
            print(f"     epoch {epoch+1:2d}/{epochs}  loss={_ep_loss/_ep_batches:.4f}  "
                  f"val_loss={val_loss:.4f}{_marker}  "
                  f"\033[90m{_elapsed:.0f}s  ETA {_eta:.0f}s\033[0m", flush=True)

            if es is not None:
                if es.early_stop or epoch == epochs - 1:
                    print(f"     \033[90mEarlyStopping: {es.summary()}\033[0m", flush=True)
                    if es.early_stop:
                        es.load_best(model)
                    break

        # Train metrics — collect labels alongside predictions
        # (train_loader uses shuffle=True, so y_tr order ≠ prediction order)
        model.eval()
        all_tr_preds, all_tr_probs, all_tr_labels = [], [], []
        with torch.no_grad():
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device_str)
                attention_mask = batch["attention_mask"].to(device_str)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device_str)
                logits = model(input_ids, attention_mask, token_type_ids)
                probs = torch.softmax(logits, dim=1)
                all_tr_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_tr_probs.extend(probs[:, 1].cpu().tolist())
                all_tr_labels.extend(batch["labels"].tolist())
        train_metrics = _metric_dict(np.array(all_tr_labels), np.array(all_tr_preds),
                                     np.array(all_tr_probs))

        model.eval()
        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device_str)
                attention_mask = batch["attention_mask"].to(device_str)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device_str)
                logits = model(input_ids, attention_mask, token_type_ids)
                probs = torch.softmax(logits, dim=1)
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_probs.extend(probs[:, 1].cpu().tolist())
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        val_metrics = _metric_dict(y_val, y_pred, y_prob)
        return {
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
            "auc": val_metrics["auc"],
            "ks": val_metrics["ks"],
        }

    # Fallback (should not reach here)
    raise ValueError(f"Unknown model category: {spec.category}")


def _evaluate_cv_full(X, y, spec, best_params, cv_folds, random_state,
                      device_str, embedding_path):
    """Full CV evaluation with best params."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr = [X[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_tr = y[train_idx]
        y_val = y[val_idx]
        metrics = _quick_fit_eval(X_tr, y_tr, X_val, y_val, spec, best_params,
                                  device_str, embedding_path)
        metrics["fold"] = fold
        fold_results.append(metrics)

    accs = [f["accuracy"] for f in fold_results]
    aucs = [f["auc"] for f in fold_results]
    kss = [f["ks"] for f in fold_results]
    tr_accs = [f.get("train_accuracy", 0) for f in fold_results]
    tr_f1s = [f.get("train_f1", 0) for f in fold_results]
    tr_aucs = [f.get("train_auc", 0) for f in fold_results]
    return {
        "folds": fold_results,
        "mean_train_accuracy": round(float(np.mean(tr_accs)), 4),
        "mean_train_f1": round(float(np.mean(tr_f1s)), 4),
        "mean_train_auc": round(float(np.mean(tr_aucs)), 4),
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "std_accuracy": round(float(np.std(accs)), 4),
        "mean_precision": round(float(np.mean([f["precision"] for f in fold_results])), 4),
        "mean_recall": round(float(np.mean([f["recall"] for f in fold_results])), 4),
        "mean_f1": round(float(np.mean([f["f1"] for f in fold_results])), 4),
        "mean_auc": round(float(np.mean(aucs)), 4),
        "mean_ks": round(float(np.mean(kss)), 4),
    }


def _save_tuned_model(X, y, spec, best_params, device_str, embedding_path, output_dir):
    """Save the best tuned model trained on full data."""
    import joblib
    models_dir = pathlib.Path(output_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X_clean = clean_text(X)
    merged_params = deepcopy(spec.params)
    merged_params.update(best_params)

    if spec.category == "traditional_ml":
        if spec.use_embedding != "none":
            ep = resolve_embedding_path(embedding_path, spec.use_embedding)
            if ep is None:
                raise RuntimeError(
                    f"Embedding '{spec.use_embedding}' is required for "
                    f"{spec.display_name} but could not be resolved."
                )
            embeddings = load_embeddings(ep, spec.use_embedding)
            vectorizer = create_embedding_vectorizer(embeddings, embedding_dim=300)
        else:
            vectorizer = get_vectorizer(spec.vectorizer)
        X_vec = vectorizer.fit_transform(X_clean)
        model, _ = create_model(spec, **merged_params)
        model.fit(X_vec, y)
        key = _spec_file_key(spec)
        joblib.dump(model, models_dir / f"{key}_tuned.pkl")
        joblib.dump(vectorizer, models_dir / f"{key}_tuned_vectorizer.pkl")
    elif spec.category == "deep_learning":
        import torch
        word2idx = build_vocab(X_clean)
        joblib.dump(word2idx, models_dir / f"{spec.name}_tuned_vocab.pkl")
        pretrained = None
        ep = resolve_embedding_path(embedding_path, spec.use_embedding)
        if spec.use_embedding in ("glove", "word2vec", "fasttext") and ep:
            embeddings = load_embeddings(ep, spec.use_embedding)
            pretrained = build_embedding_matrix(word2idx, embeddings, embedding_dim=300)
        model, _ = create_model(spec, vocab_size=len(word2idx),
                                pretrained_embeddings=pretrained, **merged_params)
        model = model.to(device_str)
        max_len = merged_params.get("max_seq_len", 128)
        X_ids = encode_texts_as_ids(X_clean, word2idx, max_len)
        ds = TextDataset(X_ids, y, max_len)
        loader = torch.utils.data.DataLoader(ds, batch_size=merged_params.get("batch_size", 64),
                                             shuffle=True)
        opt_name = merged_params.get("optimizer", "Adam")
        opt_class = torch.optim.Adam if opt_name == "Adam" else torch.optim.AdamW
        optimizer = opt_class(model.parameters(),
                              lr=merged_params.get("learning_rate", 1e-3),
                              weight_decay=merged_params.get("weight_decay", 0.0))
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        epochs_full = merged_params.get("epochs", 10)
        autocast_ctx, scaler = get_amp_config(device_str)
        import time as _time
        _t0 = _time.time()
        for ep in range(epochs_full):
            _ep_loss = 0.0
            _ep_batches = 0
            for xb, yb in loader:
                xb, yb = xb.to(device_str), yb.to(device_str)
                optimizer.zero_grad()
                with autocast_ctx:
                    loss = criterion(model(xb), yb)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                _ep_loss += loss.item()
                _ep_batches += 1
            _elapsed = _time.time() - _t0
            _eta = (_elapsed / (ep + 1)) * (epochs_full - ep - 1) if ep < epochs_full - 1 else 0
            print(f"     epoch {ep+1:2d}/{epochs_full}  loss={_ep_loss/_ep_batches:.4f}  "
                  f"\033[90m{_elapsed:.0f}s  ETA {_eta:.0f}s\033[0m", flush=True)
        torch.save(model.state_dict(), models_dir / f"{spec.name}_tuned.pt")
        # Save config
        save_json(merged_params, str(models_dir / f"{spec.name}_tuned_config.json"))
    elif spec.category == "transformer":
        import torch
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
        model, _ = create_model(spec, **merged_params)
        model = model.to(device_str)
        tokenizer = AutoTokenizer.from_pretrained(
            model.model_name if hasattr(model, 'model_name') else spec.name
        )
        tokenizer.save_pretrained(str(models_dir / f"{spec.name}_tuned_tokenizer"))
        max_len = merged_params.get("max_seq_len", 256)
        enc = tokenizer(X_clean, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        ds = TransformerDataset(enc, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=merged_params.get("batch_size", 16),
                                             shuffle=True)
        epochs = merged_params.get("epochs", 3)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=merged_params.get("learning_rate", 2e-5),
            weight_decay=merged_params.get("weight_decay", 0.01),
        )
        total_steps = epochs * len(loader)
        warmup_steps = int(merged_params.get("warmup_ratio", 0.06) * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        epochs_full = epochs
        autocast_ctx, scaler = get_amp_config(device_str)
        import time as _time
        _t0 = _time.time()
        for ep in range(epochs_full):
            _ep_loss = 0.0
            _ep_batches = 0
            for batch in loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device_str)
                attention_mask = batch["attention_mask"].to(device_str)
                labels = batch["labels"].to(device_str)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device_str)
                with autocast_ctx:
                    logits = model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(logits, labels)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                _ep_loss += loss.item()
                _ep_batches += 1
            _elapsed = _time.time() - _t0
            _eta = (_elapsed / (ep + 1)) * (epochs_full - ep - 1) if ep < epochs_full - 1 else 0
            print(f"     epoch {ep+1:2d}/{epochs_full}  loss={_ep_loss/_ep_batches:.4f}  "
                  f"\033[90m{_elapsed:.0f}s  ETA {_eta:.0f}s\033[0m", flush=True)
        actual_mode = getattr(model, 'training_mode', spec.training_mode)
        if actual_mode == "peft":
            try:
                model.encoder.save_pretrained(str(models_dir / f"{spec.name}_tuned_adapter"))
            except AttributeError:
                torch.save(model.state_dict(), models_dir / f"{spec.name}_tuned.pt")
        else:
            torch.save(model.state_dict(), models_dir / f"{spec.name}_tuned.pt")
        save_json(merged_params, str(models_dir / f"{spec.name}_tuned_config.json"))


def _spec_file_key(spec: ModelSpec) -> str:
    """Unique file-safe key for a given spec (name + optional embedding)."""
    if spec.use_embedding != "none":
        return f"{spec.name}_{spec.use_embedding}"
    return spec.name


def _filter_model_params(params: dict, model_name: str) -> dict:
    """Keep only valid params for the specified model."""
    valid_keys = {
        "svm_linear": {"C", "max_iter", "loss", "dual", "penalty", "class_weight"},
        "svm_rbf": {"C", "gamma", "kernel", "probability", "class_weight"},
        "logistic_regression": {"C", "penalty", "solver", "max_iter", "l1_ratio",
                                "class_weight"},
        "random_forest": {"n_estimators", "max_depth", "min_samples_split",
                          "min_samples_leaf", "max_features", "class_weight", "n_jobs"},
        "multinomial_nb": {"alpha", "fit_prior"},
    }
    allowed = valid_keys.get(model_name, set(params.keys()))
    return {k: v for k, v in params.items() if k in allowed}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_models(baseline: dict, tuned: dict) -> dict:
    """Compare baseline vs tuned model performance."""
    b_acc = baseline.get("mean_accuracy", 0)
    t_acc = tuned.get("mean_accuracy", 0)
    b_f1 = baseline.get("mean_f1", 0)
    t_f1 = tuned.get("mean_f1", 0)
    b_auc = baseline.get("mean_auc", 0.5)
    t_auc = tuned.get("mean_auc", 0.5)
    abs_imp = t_acc - b_acc
    rel_imp = abs_imp / max(b_acc, 1e-6)
    abs_f1 = t_f1 - b_f1
    abs_auc = t_auc - b_auc
    return {
        "baseline_accuracy": b_acc,
        "tuned_accuracy": t_acc,
        "absolute_improvement": round(abs_imp, 4),
        "relative_improvement": round(rel_imp, 4),
        "baseline_f1": b_f1,
        "tuned_f1": t_f1,
        "f1_improvement": round(abs_f1, 4),
        "baseline_auc": b_auc,
        "tuned_auc": t_auc,
        "auc_improvement": round(abs_auc, 4),
    }


# ---------------------------------------------------------------------------
# Master Train Function
# ---------------------------------------------------------------------------

def train_all(
    csv_path: str,
    text_col: str,
    label_col: str,
    scheme_path: str,
    models: list = None,
    mode: str = "both",
    cv_folds: int = 5,
    tune_method: str = "cv",
    n_trials: int = 50,
    output_dir: str = "output",
    use_mlflow: bool = True,
    seed: int = 42,
    embedding_path: str = None,
    epochs_override: int = None,
    encoding: str = None,
    split_path: str = None,
    log_file: str = None,
) -> dict:
    """
    Master training function. Trains selected models from the scheme.

    If split_path is provided (from Stage 2), training uses only the
    train set; CV/tuning is performed on train only.  The test set is
    never touched during training.

    Returns: {model_name: {"baseline": ..., "tuned": ..., "comparison": ...}}
    """
    setup_logging_and_warnings()
    set_seed(seed)

    # Set up training log so user can tail the file for real-time progress.
    log_filename = log_file or "training.log"
    tee = setup_training_log(output_dir, log_filename)
    print(f"  Training log: {os.path.join(output_dir, log_filename)}")

    # Load data
    print_header("Stage 4: Model Training")
    df, _ = read_csv_safe(csv_path, encoding=encoding)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    X_full = clean_text(texts)
    y_full = np.array(labels)

    # Validate training preconditions before expensive work
    if cv_folds < 2:
        raise ValueError(f"cv_folds must be >= 2 for stratified CV, got {cv_folds}")
    unique_classes = np.unique(y_full)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Training requires at least 2 classes. "
            f"Found only class(es): {unique_classes.tolist()}"
        )

    # Apply split if provided
    X_valid, y_valid = None, None
    X_test, y_test = None, None
    test_idx = None
    split_type = None  # "3way" or "2way" or None
    if split_path:
        split_info = load_json(split_path)
        train_idx = split_info["splits"]["train"]
        valid_idx = split_info["splits"].get("valid")  # may be None
        test_idx = split_info["splits"].get("test")    # may be None
        X = [X_full[i] for i in train_idx]
        y = y_full[train_idx]
        print(f"  Using pre-split data from: {split_path}")
        print(f"  Train: {len(train_idx):,}  |  "
              f"class 0: {(y==0).sum():,}  "
              f"class 1: {(y==1).sum():,}")
        if valid_idx:
            X_valid = [X_full[i] for i in valid_idx]
            y_valid = y_full[valid_idx]
            split_type = "3way"
            print(f"  Valid: {len(valid_idx):,}  |  "
                  f"class 0: {(y_valid==0).sum():,}  "
                  f"class 1: {(y_valid==1).sum():,}")
        else:
            split_type = "2way"
        if test_idx:
            X_test = [X_full[i] for i in test_idx]
            y_test = y_full[test_idx]
            print(f"  Test:  {len(test_idx):,} (\033[33mheld out\033[0m)")
    else:
        X, y = X_full, y_full
        cat_counts = dict(zip(*np.unique(y, return_counts=True)))
        print(f"  Loaded {len(X):,} samples  │  "
              f"class 0: {cat_counts.get(0, 0):,}  "
              f"class 1: {cat_counts.get(1, 0):,}", flush=True)

    # Load scheme
    scheme_data = load_json(scheme_path)
    all_specs = [dict_to_spec(m) for m in scheme_data.get("models", [])]

    # Apply epochs override
    if epochs_override is not None:
        for s in all_specs:
            if s.category in ("deep_learning", "transformer"):
                s.params["epochs"] = epochs_override
        print(f"  Epochs override: {epochs_override}")

    # Filter to selected models.
    # Display names match exactly (preferred).  Base names match all variants
    # that share that name, but when multiple variants have the same name only
    # the single highest-priority variant is kept (first in scheme order breaks
    # ties).  This prevents "logistic_regression" from pulling in all 5 variants
    # while still letting "TextCNN + GloVe 300d (fine-tuned)" target exactly one.
    if models:
        filtered = []
        matched_names = set()
        for s in all_specs:
            # exact display_name match — highest precision, always accept
            if s.display_name in models:
                filtered.append(s)
                matched_names.add(s.name)
        for s in all_specs:
            if s in filtered:
                continue
            # name match: only accept if no display_name match already captured
            # a same-named model AND this is the single best-priority variant
            if s.name in models and s.name not in matched_names:
                same_name_specs = [x for x in all_specs if x.name == s.name]
                best_pri = min(x.priority for x in same_name_specs)
                if s.priority == best_pri:
                    # tie-break: first spec in scheme order wins
                    already = any(f.name == s.name for f in filtered)
                    if not already:
                        filtered.append(s)
        all_specs = filtered

    if not all_specs:
        print("[ERROR] No models selected for training.")
        return {}

    # Pre-check: ensure embeddings are available before training starts.
    # This avoids mid-training failures when auto-download is slow or fails.
    _needed_embeddings = set()
    for s in all_specs:
        if s.use_embedding not in ("none", "pretrained"):
            _needed_embeddings.add(s.use_embedding)
    # Track embedding types that failed to download (module-level cache
    # prevents repeated download attempts across folds and models).
    _failed_embeddings = set()

    if _needed_embeddings:
        from preprocessing import ensure_embeddings
        print(f"  Checking embeddings: {', '.join(sorted(_needed_embeddings))}")
        for emb_type in sorted(_needed_embeddings):
            path = ensure_embeddings(emb_type)
            if path:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"    {emb_type}: OK ({size_mb:.0f} MB)")
            else:
                _failed_embeddings.add(emb_type)
                print(f"    {emb_type}: \033[31mNOT FOUND\033[0m — "
                      f"all 3 mirrors exhausted, download failed")
        if _failed_embeddings:
            _skipped_models = [s.display_name for s in all_specs
                               if s.use_embedding in _failed_embeddings]
            if _skipped_models:
                print(f"\n  \033[33m[SKIP] The following models require embeddings "
                      f"that failed to download:\033[0m")
                for m in _skipped_models:
                    print(f"    - {m}")
                print(f"  \033[90mCause: network restricted — HuggingFace, hf-mirror, "
                      f"and official mirrors all unreachable.\033[0m")
                print(f"  \033[90mRandom initialization would produce poor results "
                      f"(e.g. Val F1 ~0.7 vs ~0.88 with embeddings).\033[0m")
                print(f"  \033[90mThese models will be skipped. To use them, manually "
                      f"download the embedding file and pass --<type>-path.\033[0m")
                # Remove specs that require failed embeddings
                all_specs = [s for s in all_specs
                             if s.use_embedding not in _failed_embeddings]
                if not all_specs:
                    print("\n  \033[31m[ERROR] All selected models skipped — no "
                          "embeddings available.\033[0m")
                    return {}

    # Pre-check: if any Transformer models are selected, test HF connectivity first.
    # This avoids a 2-5 minute timeout wait per model when the network is restricted.
    _transformer_specs = [s for s in all_specs if s.category == "transformer"]
    if _transformer_specs:
        from utils import run_network_diagnostics
        net = run_network_diagnostics()
        if not net.get("any_reachable"):
            _tf_names = [s.display_name for s in _transformer_specs]
            print()
            print(f"  \033[33m[SKIP] All HF download sources unreachable. "
                  f"Skipping {len(_tf_names)} Transformer model(s):\033[0m")
            for m in _tf_names:
                print(f"    - {m}")
            print(f"  \033[90mCause: huggingface.co, hf-mirror.com, and modelscope.cn "
                  f"all failed.\033[0m")
            print(f"  \033[90mTraditional ML and DL models (no HF dependency) will "
                  f"proceed normally.\033[0m")
            # Record skipped models in results for later reporting
            for s in _transformer_specs:
                results[s.display_name] = {
                    "error": "network_restricted: all HF sources unreachable",
                    "skipped": True,
                    "split_type": split_type,
                }
            save_json(results, str(result_path))
            all_specs = [s for s in all_specs if s.category != "transformer"]
            if not all_specs:
                print("\n  \033[31m[ERROR] All selected models skipped — "
                      "no network for Transformers, no other models selected.\033[0m")
                return results

    # Device
    device_info = detect_device()
    device_str = device_info["recommended_device"]
    print(f"  Using device: {device_str}")

    # MLflow setup
    if use_mlflow:
        setup_mlflow()

    out_dir = ensure_output_dir(output_dir)
    result_path = out_dir / "training_results.json"
    # Load existing results for the models we're about to train (or all if
    # training everything).  This allows resuming interrupted runs for the
    # same model set but prevents stale results from other sessions leaking in.
    if result_path.exists():
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            # Always keep all cached results; new/updated results for the
            # current model(s) will be merged in below.  This preserves
            # results from previous runs when training models separately.
            results = cached
        except (json.JSONDecodeError, OSError):
            results = {}
    else:
        results = {}

    # Start periodic progress reporter (every 5 minutes)
    reporter = ProgressReporter(interval_minutes=5)
    reporter.update(
        total_models=len(all_specs),
        start_time=time.time(),
    )
    reporter.start()
    atexit.register(reporter.stop)  # safety net: stop reporter on crash

    for i, spec in enumerate(all_specs):
        try:
            print_model_header(i + 1, len(all_specs), spec.display_name, spec.category)

            reporter.update(
                current_idx=i + 1,
                current_name=spec.display_name,
                phase="preparing",
                phase_progress="",
            )

            model_results = {}

            _ctx = mlflow_run(run_name=f"{spec.name}_{spec.category}") if use_mlflow else nullcontext()
            with _ctx:
                if use_mlflow:
                    log_params({"category": spec.category, "vectorizer": spec.vectorizer,
                                "model": spec.display_name})

                # Determine what to run based on mode
                # baseline/tune: run as requested for all models
                # both: run baseline first, then tuning, for every selected model
                run_baseline = (mode == "baseline") or (mode == "both")
                run_tuned = (mode == "tune") or (mode == "both")

                # Baseline
                if run_baseline:
                    reporter.update(phase="baseline", phase_progress="starting...")
                    baseline = train_baseline(
                        X, y, spec, cv_folds=cv_folds, random_state=seed,
                        device_str=device_str, embedding_path=embedding_path,
                        output_dir=output_dir,
                        X_valid=X_valid, y_valid=y_valid,
                        reporter=reporter,
                    )
                    reporter.update(phase="baseline", phase_progress="done")
                    # Evaluate on test set
                    if X_test is not None:
                        baseline["test_metrics"] = _evaluate_on_test(
                            X, y, X_test, y_test, spec, baseline.get("best_params", spec.params),
                            device_str, embedding_path)
                    model_results["baseline"] = baseline
                    model_results["split_type"] = split_type
                    if use_mlflow:
                        log_cv_results(baseline)
                        log_metrics({
                            "baseline_mean_accuracy": baseline["mean_accuracy"],
                            "baseline_mean_f1": baseline["mean_f1"],
                            "baseline_mean_auc": baseline["mean_auc"],
                            "baseline_mean_ks": baseline["mean_ks"],
                        })

                # Tuned
                if run_tuned:
                    reporter.update(phase="tuning", phase_progress="starting...")
                    tuned = train_tuned(
                        X, y, spec, tune_method=tune_method, cv_folds=cv_folds,
                        n_trials=n_trials, random_state=seed, device_str=device_str,
                        embedding_path=embedding_path, output_dir=output_dir,
                        epochs_override=epochs_override,
                        X_valid=X_valid, y_valid=y_valid,
                        reporter=reporter,
                    )
                    # Evaluate on test set
                    if X_test is not None:
                        tuned["test_metrics"] = _evaluate_on_test(
                            X, y, X_test, y_test, spec, tuned.get("best_params", spec.params),
                            device_str, embedding_path)
                    model_results["tuned"] = tuned
                    if use_mlflow:
                        log_cv_results(tuned)
                        log_metrics({
                            "tuned_mean_accuracy": tuned["mean_accuracy"],
                            "tuned_mean_f1": tuned["mean_f1"],
                            "tuned_mean_auc": tuned["mean_auc"],
                            "tuned_mean_ks": tuned["mean_ks"],
                            "best_trial_value": tuned["best_value"],
                        })
                        log_params({"best_params": str(tuned.get("best_params", {}))})
                        log_optimization_history(tuned.get("optimization_history", []))

                # Comparison (only when both baseline and tuned were run)
                if model_results.get("baseline") and model_results.get("tuned"):
                    comparison = compare_models(model_results.get("baseline", {}),
                                                model_results.get("tuned", {}))
                    model_results["comparison"] = comparison
                    arrow = "\033[32m↑\033[0m" if comparison['absolute_improvement'] > 0 else "\033[31m↓\033[0m"
                    print(f"     {arrow} "
                          f"Δacc=\033[1m{comparison['absolute_improvement']:+.4f}\033[0m "
                          f"Δf1=\033[1m{comparison['f1_improvement']:+.4f}\033[0m  "
                          f"\033[90m({comparison['relative_improvement']:+.1%})\033[0m",
                          flush=True)

            # Merge into existing results for this model (preserving prior runs)
            existing = results.setdefault(spec.display_name, {})
            existing.update(model_results)
            if "split_type" not in existing:
                existing["split_type"] = split_type

            # Free GPU memory before next model to prevent OOM accumulation
            if device_str == "cuda":
                import torch
                torch.cuda.empty_cache()
                import gc
                gc.collect()

            # Incremental save after each model completes
            save_json(results, str(result_path))

            # Mark model as completed for progress reporter
            has_baseline = "baseline" in model_results
            has_tuned = "tuned" in model_results
            reporter.add_completed(spec.display_name, baseline_ok=has_baseline, tuned_ok=has_tuned)

        except Exception as e:
            print(f"\n  \033[31m[ERROR] Model '{spec.display_name}' failed: {e}\033[0m\n")
            results[spec.display_name] = {"error": str(e), "split_type": split_type}
            save_json(results, str(result_path))
            reporter.add_completed(spec.display_name, baseline_ok=False, tuned_ok=False)

    # Stop the periodic progress reporter
    reporter.stop()

    # Final summary table
    _print_final_summary(results, mode, split_info if split_path else None)

    print(f"\n  \033[90mResults saved to: {out_dir / 'training_results.json'}\033[0m", flush=True)

    return results


def _evaluate_on_test(X_train, y_train, X_test, y_test, spec, params, device_str,
                      embedding_path):
    """Train on full train set with given params, evaluate on test set."""
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        if spec.category == "traditional_ml":
            metrics = _quick_fit_eval(X_train, y_train, X_test, y_test, spec,
                                      params, device_str, embedding_path)
        elif spec.category == "deep_learning":
            metrics = _quick_fit_eval(X_train, y_train, X_test, y_test, spec,
                                      params, device_str, embedding_path)
        elif spec.category == "transformer":
            metrics = _quick_fit_eval(X_train, y_train, X_test, y_test, spec,
                                      params, device_str, embedding_path)
    return metrics


def _print_final_summary(results: dict, mode: str, split_info: dict = None) -> None:
    """Print a compact summary table of all trained models."""
    if not results:
        return

    # Compute sample counts from split_info
    n_train, n_valid, n_test = 0, 0, 0
    if split_info:
        splits = split_info.get("splits", {})
        n_train = len(splits.get("train", []))
        n_valid = len(splits.get("valid") or [])
        n_test = len(splits.get("test") or [])

    # Detect split type from first model result
    split_type = None
    for name, mr in results.items():
        if isinstance(mr, dict) and "split_type" in mr:
            split_type = mr["split_type"]
        elif isinstance(mr, dict):
            for sub in mr.values():
                if isinstance(sub, dict) and sub.get("split_method"):
                    split_type = "3way" if sub["split_method"] == "split" else "2way"
                    break
        if split_type:
            break

    if split_type == "3way":
        headers = ["Model", "Mode",
                   f"CV_Tr_Acc (N={n_train})", f"CV_Tr_F1 (N={n_train})", f"CV_Tr_AUC (N={n_train})",
                   f"Valid_Acc (N={n_valid})", f"Valid_F1 (N={n_valid})", f"Valid_AUC (N={n_valid})",
                   f"Test_Acc (N={n_test})", f"Test_F1 (N={n_test})", f"Test_AUC (N={n_test})",
                   "Time"]
    else:
        headers = ["Model", "Mode",
                   f"CV-Tr_Acc (N={n_train})", f"CV-Tr_F1 (N={n_train})", f"CV-Tr_AUC (N={n_train})",
                   f"Val_Acc (N=CV)", f"Val_F1 (N=CV)", f"Val_AUC (N=CV)",
                   f"Test_Acc (N={n_test})", f"Test_F1 (N={n_test})", f"Test_AUC (N={n_test})",
                   "Time"]

    rows = []
    for name, model_results in results.items():
        if not isinstance(model_results, dict):
            continue
        mr_split_type = model_results.get("split_type", split_type)

        if "baseline" in model_results:
            b = model_results["baseline"]
            test = b.get("test_metrics", {})
            if mr_split_type == "3way":
                tm = b.get("train_metrics", {})
                tr_acc = tm.get("accuracy", 0)
                tr_f1 = tm.get("f1", 0)
                tr_auc = tm.get("auc", 0)
            else:
                tr_acc = b.get('mean_train_accuracy', 0)
                tr_f1 = b.get('mean_train_f1', 0)
                tr_auc = b.get('mean_train_auc', 0)
            rows.append([
                name[:40],
                "\033[37mbaseline\033[0m",
                f"{tr_acc:.4f}" if tr_acc else "—",
                f"{tr_f1:.4f}" if tr_f1 else "—",
                f"{tr_auc:.4f}" if tr_auc else "—",
                f"{b['mean_accuracy']:.4f}",
                f"{b['mean_f1']:.4f}",
                f"{b.get('mean_auc', 0):.4f}",
                f"{test.get('accuracy', 0):.4f}" if test else "—",
                f"{test.get('f1', 0):.4f}" if test else "—",
                f"{test.get('auc', 0):.4f}" if test else "—",
                format_duration(b.get('total_fit_time') or b.get('total_time', 0)),
            ])
        if "tuned" in model_results:
            t = model_results["tuned"]
            test = t.get("test_metrics", {})
            if mr_split_type == "3way":
                tm = t.get("train_metrics", {})
                tr_acc = tm.get("accuracy", 0)
                tr_f1 = tm.get("f1", 0)
                tr_auc = tm.get("auc", 0)
            else:
                tr_acc = t.get('mean_train_accuracy', 0)
                tr_f1 = t.get('mean_train_f1', 0)
                tr_auc = t.get('mean_train_auc', 0)
            rows.append([
                name[:40],
                "\033[33mtuned\033[0m",
                f"{tr_acc:.4f}" if tr_acc else "—",
                f"{tr_f1:.4f}" if tr_f1 else "—",
                f"{tr_auc:.4f}" if tr_auc else "—",
                f"{t['mean_accuracy']:.4f}",
                f"{t['mean_f1']:.4f}",
                f"{t.get('mean_auc', 0):.4f}",
                f"{test.get('accuracy', 0):.4f}" if test else "—",
                f"{test.get('f1', 0):.4f}" if test else "—",
                f"{test.get('auc', 0):.4f}" if test else "—",
                format_duration(t.get('total_fit_time') or t.get('total_time', 0)),
            ])

    # Sort by validation accuracy descending (column index 5 = Val Acc)
    rows.sort(key=lambda r: float(r[5]) if r[5] != "—" else 0, reverse=True)

    if rows:
        print_header("Training Results Summary")
        print_table(headers, rows)

        # ── Best Model Parameters (selected by Validation F1, not Test!) ──
        def _get_val_f1(mr):
            """Extract validation F1 from a model result dict (baseline or tuned)."""
            return mr.get("mean_f1", -1)

        best_name = None
        best_mode = None
        best_val_f1 = -1.0
        best_test_f1 = -1.0
        best_params = {}
        best_default_params = {}

        for name, model_results in results.items():
            if not isinstance(model_results, dict):
                continue
            for mode_key in ["tuned", "baseline"]:
                if mode_key not in model_results:
                    continue
                mr = model_results[mode_key]
                val_f1 = _get_val_f1(mr)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_name = name
                    best_mode = mode_key
                    best_test_f1 = mr.get("test_metrics", {}).get("f1", -1)
                    if mode_key == "tuned":
                        best_params = mr.get("best_params", {})
                        best_default_params = model_results.get("baseline", {}).get("params", {})
                    else:
                        best_params = mr.get("params", {})
                        best_default_params = {}

        if best_name:
            print_header(f"Best Model: {best_name}")
            mode_label = "tuned (Optuna)" if best_mode == "tuned" else "baseline (default params)"
            print(f"  Selected by: highest Valid F1 = {best_val_f1:.4f}  [{mode_label}]")
            if best_test_f1 >= 0:
                print(f"  Corresponding Test F1 = {best_test_f1:.4f}")
            print()

            if best_default_params:
                print(f"  {'Parameter':<30s}  {'Default':<20s}  {'Best (Optuna)':<20s}")
                print(f"  {'-'*30}  {'-'*20}  {'-'*20}")
                for k in best_params:
                    def_v = str(best_default_params.get(k, "—"))
                    best_v = str(best_params.get(k, "—"))
                    print(f"  {str(k):<30s}  {def_v:<20s}  {best_v:<20s}")
            else:
                print(f"  {'Parameter':<30s}  {'Value':<20s}")
                print(f"  {'-'*30}  {'-'*20}")
                for k, v in best_params.items():
                    print(f"  {str(k):<30s}  {str(v):<20s}")
            print()

        # ── All model parameters ──
        for name, model_results in results.items():
            if not isinstance(model_results, dict):
                continue
            if name == best_name:
                continue
            for mode_key in ["tuned", "baseline"]:
                if mode_key not in model_results:
                    continue
                mr = model_results[mode_key]
                val_f1 = _get_val_f1(mr)
                test_metrics = mr.get("test_metrics", {})
                test_f1 = test_metrics.get("f1", -1)
                if mode_key == "tuned":
                    params = mr.get("best_params", {})
                else:
                    params = mr.get("params", {})
                if not params:
                    continue
                mode_label = "tuned (Optuna)" if mode_key == "tuned" else "baseline (default params)"
                print(f"  [{name}] {mode_label} → Valid F1={val_f1:.4f}  Test F1={test_f1:.4f}")
                print(f"  {'Parameter':<30s}  {'Value':<20s}")
                print(f"  {'-'*30}  {'-'*20}")
                for k, v in params.items():
                    print(f"  {str(k):<30s}  {str(v):<20s}")
                print()


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Model Training (Baseline + Optuna Tuning)"
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--text-col", required=True, help="Text column name")
    parser.add_argument("--label-col", required=True, help="Label column name (0/1)")
    parser.add_argument("--scheme", required=True,
                        help="Path to model_scheme.json from Stage 3")
    parser.add_argument("--models", default=None,
                        help="Semicolon-separated model names to train, e.g. 'svm_linear;logistic_regression' (default: all). Display names with commas inside parentheses must use semicolons.")
    parser.add_argument("--mode", default="both",
                        choices=["baseline", "tune", "both"],
                        help="Training mode")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--tune-method", default="cv", choices=["cv", "split"],
                        help="Tuning evaluation method")
    parser.add_argument("--tune-trials", type=int, default=50,
                        help="Number of Optuna trials")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow tracking")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--embedding-path", default=None,
                        help="Path to GloVe/fastText/Word2Vec embedding file (for DL models, backward-compat)")
    parser.add_argument("--glove-path", default=None,
                        help="Path to GloVe embedding file (overrides --embedding-path for glove)")
    parser.add_argument("--word2vec-path", default=None,
                        help="Path to Word2Vec embedding file")
    parser.add_argument("--fasttext-path", default=None,
                        help="Path to fastText embedding file")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for all models (baseline and tuning)")
    parser.add_argument("--encoding", default=None,
                        help="CSV file encoding (e.g. utf-8, latin-1). If not specified, auto-detected via fallback chain.")
    parser.add_argument("--log-file", default=None,
                        help="Training log filename (default: training.log in output dir)")
    parser.add_argument("--split", default=None,
                        help="Path to split_info.json from Stage 2 (optional). "
                             "If provided, training uses only train set indices; "
                             "test set is held out completely.")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] File not found: {args.csv}")
        sys.exit(1)
    if not os.path.exists(args.scheme):
        print(f"[ERROR] Scheme file not found: {args.scheme}")
        sys.exit(1)
    if args.split and not os.path.exists(args.split):
        print(f"[ERROR] Split file not found: {args.split}")
        sys.exit(1)

    # Use semicolon as the primary delimiter because display names like
    # "TF-IDF (1,2-gram)" contain commas that would be mistakenly split.
    # Fall back to the raw string as a single model name when no semicolon
    # is present (handles the single-model-with-commas case).  Comma
    # splitting is only used when the raw string contains commas but no
    # semicolons AND the split produces fragments that look like valid
    # model references (all fragments appear in scheme names/display_names).
    if args.models:
        raw = args.models.strip()
        if ";" in raw:
            model_list = [x.strip() for x in raw.split(";") if x.strip()]
        else:
            model_list = [raw]  # single model, may contain commas in display name
    else:
        model_list = None

    # Build embedding path map: type -> path.
    # --embedding-path is a legacy convenience; only apply as fallback when no
    # per-type path (--glove-path / --word2vec-path / --fasttext-path) is given.
    _has_specific = any([args.glove_path, args.word2vec_path, args.fasttext_path])
    _legacy_path = args.embedding_path if not _has_specific else None
    embedding_paths = {
        "glove": args.glove_path or _legacy_path,
        "word2vec": args.word2vec_path or _legacy_path,
        "fasttext": args.fasttext_path or _legacy_path,
    }

    try:
        train_all(
            csv_path=args.csv,
            text_col=args.text_col,
            label_col=args.label_col,
            scheme_path=args.scheme,
            models=model_list,
            mode=args.mode,
            cv_folds=args.cv_folds,
            tune_method=args.tune_method,
            n_trials=args.tune_trials,
            output_dir=args.output_dir,
            use_mlflow=not args.no_mlflow,
            seed=args.seed,
            embedding_path=embedding_paths,
            epochs_override=args.epochs,
            encoding=args.encoding,
            split_path=args.split,
            log_file=args.log_file,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
