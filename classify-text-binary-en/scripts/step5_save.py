"""
Stage 5: Save final model, generate report, and produce deployment artifacts.

Usage:
    python step5_save.py --csv <path> --text-col <name> --label-col <name>
                         --training-results <path/to/training_results.json>
                         --analysis <path/to/analysis.json>
                         --best-model <name>
                         [--output-dir <dir>]
                         [--no-mlflow]
                         [--seed N]

Output:
    output/final_model/           -- final model artifacts
    output/training_report.html   -- comprehensive HTML report

Prevents transformers safetensors conversion background thread crash.
"""

import os as _os
if not _os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"):
    _os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import argparse
import sys
import os
import pathlib
import time
import webbrowser

import numpy as np
import joblib

_script_dir = pathlib.Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from utils import (
    read_csv_safe, load_json, save_json, print_header, print_subheader,
    detect_device, set_seed, format_duration,
    resolve_embedding_path,
    setup_logging_and_warnings,
)
from preprocessing import (
    clean_text, build_vocab, encode_texts_as_ids,
    build_embedding_matrix, load_embeddings, get_vectorizer,
    create_embedding_vectorizer,
)
from model_factory import (
    create_model, ModelSpec, dict_to_spec, TextDataset, TransformerDataset,
    _hf_from_pretrained_with_fallback,
)
from mlflow_utils import (
    setup_mlflow, mlflow_run, log_params, log_metrics, log_artifact,
)
from report import generate_html_report
from deploy import generate_deployment
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve,
)



def retrain_full(
    X: list,
    y: np.ndarray,
    spec: ModelSpec,
    best_params: dict,
    device_str: str = "cpu",
    embedding_path: str = None,
    seed: int = 42,
    X_test: list = None,
    y_test: np.ndarray = None,
) -> tuple:
    """
    Retrain the best model on the full dataset with best hyperparameters.
    Evaluates on an independent hold-out: either a user-provided X_test (which
    must NOT overlap with X), or by splitting 20% from X internally.  When
    retraining on full data for deployment, X_test should be None so that
    evaluation does not leak training data.

    Returns (model, vectorizer_or_tokenizer, metrics_dict).
    """
    import torch
    from torch.utils.data import DataLoader

    set_seed(seed)
    X_clean = clean_text(X)
    y = np.array(y)
    merged_params = {**spec.params, **best_params}

    # Determine evaluation set
    do_eval = True
    if X_test is not None and y_test is not None and len(X_test) > 0:
        X_eval = clean_text(X_test)
        y_eval = np.array(y_test)
        eval_label = "test set"
    elif X_test is None and y_test is None:
        # Caller explicitly passed None for both — train on all data, skip eval
        do_eval = False
        eval_label = ""
    else:
        X_clean, X_eval, y, y_eval = train_test_split(
            X_clean, y, test_size=0.2, stratify=y, random_state=seed,
        )
        eval_label = "20% hold-out"
    if do_eval:
        y_eval = np.array(y_eval)

    if do_eval:
        print(f"  Retraining {spec.display_name} on {len(X_clean)} samples "
              f"(evaluating on {len(X_eval)} {eval_label} samples)...")
    else:
        print(f"  Retraining {spec.display_name} on all {len(X_clean)} samples...")

    if spec.category == "traditional_ml":
        if spec.use_embedding != "none":
            ep = resolve_embedding_path(embedding_path, spec.use_embedding)
            if ep is None:
                raise RuntimeError(
                    f"Embedding '{spec.use_embedding}' is required for "
                    f"{spec.display_name} but could not be resolved."
                )
            embeddings = load_embeddings(ep, spec.use_embedding)
            # embeddings is a dict {word: np.ndarray}, infer dim from first vector
            embedding_dim = next(iter(embeddings.values())).shape[0] if embeddings else 300
            vectorizer = create_embedding_vectorizer(embeddings, embedding_dim=embedding_dim)
        else:
            vectorizer = get_vectorizer(spec.vectorizer)
        X_vec = vectorizer.fit_transform(X_clean)
        model, _ = create_model(spec, **merged_params)
        model.fit(X_vec, y)
        if do_eval:
            X_eval_vec = vectorizer.transform(X_eval)
            y_pred = model.predict(X_eval_vec)
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_eval_vec)[:, 1]
            else:
                y_prob = model.decision_function(X_eval_vec)
            fpr, tpr, _ = roc_curve(y_eval, y_prob)
            metrics = {
                "accuracy": float(accuracy_score(y_eval, y_pred)),
                "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
                "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
                "f1": float(f1_score(y_eval, y_pred, zero_division=0)),
                "auc": float(roc_auc_score(y_eval, y_prob)),
                "ks": float(np.max(tpr - fpr)),
                "confusion_matrix": confusion_matrix(y_eval, y_pred).tolist(),
            }
        else:
            metrics = {}
        return model, vectorizer, metrics

    elif spec.category == "deep_learning":
        word2idx = build_vocab(X_clean)
        max_len = merged_params.get("max_seq_len", 128)
        pretrained = None
        ep = resolve_embedding_path(embedding_path, spec.use_embedding)
        if spec.use_embedding in ("glove", "word2vec", "fasttext") and ep:
            embeddings = load_embeddings(ep, spec.use_embedding)
            # embeddings is a dict {word: np.ndarray}, infer dim from first vector
            embedding_dim = next(iter(embeddings.values())).shape[0] if embeddings else 300
            pretrained = build_embedding_matrix(word2idx, embeddings, embedding_dim=embedding_dim)

        model, _ = create_model(spec, vocab_size=len(word2idx),
                                pretrained_embeddings=pretrained, **merged_params)
        model = model.to(device_str)

        X_ids = encode_texts_as_ids(X_clean, word2idx, max_len)
        ds = TextDataset(X_ids, y, max_len)
        batch_size = merged_params.get("batch_size", 64)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        if do_eval:
            X_eval_ids = encode_texts_as_ids(X_eval, word2idx, max_len)
            eval_ds = TextDataset(X_eval_ids, y_eval, max_len)
            eval_loader = DataLoader(eval_ds, batch_size=batch_size)
        opt_name = merged_params.get("optimizer", "Adam")
        opt_class = torch.optim.Adam if opt_name == "Adam" else torch.optim.AdamW
        optimizer = opt_class(model.parameters(),
                              lr=merged_params.get("learning_rate", 1e-3),
                              weight_decay=merged_params.get("weight_decay", 0.0))
        criterion = torch.nn.CrossEntropyLoss()
        epochs_dl = merged_params.get("epochs", 10)
        import time as _time
        _t0 = _time.time()
        model.train()
        for ep in range(epochs_dl):
            _ep_loss = 0.0
            _ep_batches = 0
            for xb, yb in loader:
                xb, yb = xb.to(device_str), yb.to(device_str)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                _ep_loss += loss.item()
                _ep_batches += 1
            _elapsed = _time.time() - _t0
            _eta = (_elapsed / (ep + 1)) * (epochs_dl - ep - 1) if ep < epochs_dl - 1 else 0
            print(f"     epoch {ep+1:2d}/{epochs_dl}  loss={_ep_loss/_ep_batches:.4f}  "
                  f"\033[90m{_elapsed:.0f}s  ETA {_eta:.0f}s\033[0m", flush=True)

        if do_eval:
            model.eval()
            all_preds, all_probs = [], []
            with torch.no_grad():
                for xb, _ in eval_loader:
                    logits = model(xb.to(device_str))
                    probs = torch.softmax(logits, dim=1)
                    all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                    all_probs.extend(probs[:, 1].cpu().tolist())
            y_pred = np.array(all_preds)
            y_prob = np.array(all_probs)
            fpr, tpr, _ = roc_curve(y_eval, y_prob)
            metrics = {
                "accuracy": float(accuracy_score(y_eval, y_pred)),
                "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
                "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
                "f1": float(f1_score(y_eval, y_pred, zero_division=0)),
                "auc": float(roc_auc_score(y_eval, y_prob)),
                "ks": float(np.max(tpr - fpr)),
                "confusion_matrix": confusion_matrix(y_eval, y_pred).tolist(),
            }
        else:
            metrics = {}
        return model, word2idx, metrics

    elif spec.category == "transformer":
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup

        model, _ = create_model(spec, **merged_params)
        model = model.to(device_str)
        tokenizer = _hf_from_pretrained_with_fallback(
            AutoTokenizer.from_pretrained,
            model.model_name if hasattr(model, 'model_name') else spec.name
        )
        max_len = merged_params.get("max_seq_len", 256)

        enc = tokenizer(X_clean, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        ds = TransformerDataset(enc, y)
        batch_size = merged_params.get("batch_size", 16)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
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

        import time as _time
        _t0 = _time.time()
        model.train()
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
                loss = criterion(model(input_ids, attention_mask, token_type_ids), labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                _ep_loss += loss.item()
                _ep_batches += 1
            _elapsed = _time.time() - _t0
            _eta = (_elapsed / (ep + 1)) * (epochs - ep - 1) if ep < epochs - 1 else 0
            print(f"     epoch {ep+1:2d}/{epochs}  loss={_ep_loss/_ep_batches:.4f}  "
                  f"\033[90m{_elapsed:.0f}s  ETA {_eta:.0f}s\033[0m", flush=True)

        if do_eval:
            eval_enc = tokenizer(X_eval, padding=True, truncation=True,
                                max_length=max_len, return_tensors="pt")
            eval_ds = TransformerDataset(eval_enc, y_eval)
            eval_loader = DataLoader(eval_ds, batch_size=batch_size)
            model.eval()
            all_preds, all_probs = [], []
            with torch.no_grad():
                for batch in eval_loader:
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
            fpr, tpr, _ = roc_curve(y_eval, y_prob)
            metrics = {
                "accuracy": float(accuracy_score(y_eval, y_pred)),
                "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
                "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
                "f1": float(f1_score(y_eval, y_pred, zero_division=0)),
                "auc": float(roc_auc_score(y_eval, y_prob)),
                "ks": float(np.max(tpr - fpr)),
                "confusion_matrix": confusion_matrix(y_eval, y_pred).tolist(),
            }
        else:
            metrics = {}
        return model, tokenizer, metrics


def save_model_artifacts(
    model,
    vectorizer,
    output_dir: str,
    model_name: str,
    model_type: str,
    training_mode: str = "full_ft",
) -> dict:
    """
    Save model files to output/final_model/.
    Returns dict mapping component -> file path.
    """
    import torch
    final_dir = pathlib.Path(output_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    safe_name = model_name.replace(" ", "_").replace("/", "_").lower()

    if model_type == "sklearn":
        model_path = final_dir / f"{safe_name}_model.pkl"
        vec_path = final_dir / f"{safe_name}_vectorizer.pkl"
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vec_path)
        files["model"] = str(model_path)
        files["vectorizer"] = str(vec_path)

    elif model_type in ("pytorch", "transformers"):
        if model_type == "transformers" and training_mode == "peft":
            # PEFT: save adapter weights + base model config
            try:
                adapter_dir = final_dir / f"{safe_name}_adapter"
                model.encoder.save_pretrained(str(adapter_dir))
                files["adapter"] = str(adapter_dir)
                files["model"] = str(adapter_dir)  # deployment key fallback
            except AttributeError:
                model_path = final_dir / f"{safe_name}_model.pt"
                torch.save(model.state_dict(), model_path)
                files["model"] = str(model_path)
            # Save config for deployment reconstruction
            config = {
                "base_model_name": getattr(model, "model_name", model_name),
                "hidden_size": getattr(model, "hidden_size", 768),
                "training_mode": "peft",
                "num_classes": 2,
                "dropout": getattr(model, "dropout", 0.1),
            }
            config_path = final_dir / "config.json"
            save_json(config, str(config_path))
            files["config"] = str(config_path)
        else:
            model_path = final_dir / f"{safe_name}_model.pt"
            torch.save(model.state_dict(), model_path)
            # Save config for deployment reconstruction
            config = {
                "base_model_name": getattr(model, "model_name", model_name),
                "hidden_size": getattr(model, "hidden_size", 768),
                "training_mode": training_mode,
                "num_classes": 2,
                "dropout": getattr(model, "dropout", 0.1),
            }
            config_path = final_dir / "config.json"
            save_json(config, str(config_path))
            files["config"] = str(config_path)
            files["model"] = str(model_path)
        if model_type == "pytorch":
            vec_path = final_dir / f"{safe_name}_vocab.pkl"
            joblib.dump(vectorizer, vec_path)
            files["vectorizer"] = str(vec_path)
        else:
            # For transformers, save tokenizer
            tok_dir = final_dir / f"{safe_name}_tokenizer"
            vectorizer.save_pretrained(str(tok_dir))
            files["tokenizer"] = str(tok_dir)

    return files


def generate_all_artifacts(
    csv_path: str,
    text_col: str,
    label_col: str,
    analysis_path: str,
    training_results_path: str,
    best_model_name: str,
    output_dir: str = "output",
    use_mlflow: bool = True,
    seed: int = 42,
    embedding_path: str = None,
    encoding: str = None,
    split_path: str = None,
    retrain_on_full: bool = False,
) -> dict:
    """
    Stage 5 master function: retrain, save, report, deploy.

    If split_path is provided and retrain_on_full is True, retrains
    on train+valid+test combined. Otherwise uses the full CSV as-is.
    """
    setup_logging_and_warnings()
    print_header("Stage 5: Save Model & Generate Artifacts")
    set_seed(seed)

    # Load data
    df, _ = read_csv_safe(csv_path, encoding=encoding)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()

    # Load split info if provided
    split_info = None
    if split_path:
        split_info = load_json(split_path)
        print(f"  Split: {split_info['meta']['split_type']}  "
              f"|  train={split_info['counts']['train']:,}"
              f"  test={split_info['counts']['test']:,}")

    # Load analysis
    analysis = load_json(analysis_path)

    # Load training results
    training_results = load_json(training_results_path)

    # Find best model config
    best_config = _find_best_model(training_results, best_model_name)
    if best_config is None:
        raise ValueError(f"Best model '{best_model_name}' not found in training results")

    spec = dict_to_spec(best_config["spec"])
    best_params = best_config.get("best_params", {})

    print(f"  Best model: {spec.display_name}")

    # If split available, determine train data and hold-out test data
    X_test, y_test = None, None
    full_texts, full_labels = None, None  # for retrain_on_full phase 2
    if split_info and retrain_on_full:
        # Phase 1: evaluate on original Stage 2 test set (train on train only)
        # Phase 2: retrain on full data (train+test) for final deployment model
        train_idx = split_info["splits"]["train"]
        valid_idx = split_info["splits"].get("valid")
        test_idx = split_info["splits"]["test"]
        # Phase 1: train data (train set only, for fair eval)
        phase1_idx = train_idx + (valid_idx if valid_idx else [])
        # Phase 2: full data (for final model)
        full_idx = phase1_idx + test_idx
        full_texts = [texts[i] for i in full_idx]
        full_labels = [labels[i] for i in full_idx]
        # Evaluation on original test set
        X_test = [texts[i] for i in test_idx]
        y_test = [labels[i] for i in test_idx]
        # Phase 1 training uses train-only data
        texts = [texts[i] for i in phase1_idx]
        labels = [labels[i] for i in phase1_idx]
        print(f"  Phase 1: train on {len(texts):,} samples, eval on Stage 2 test set ({len(X_test):,} samples)")
        print(f"  Phase 2: retrain on FULL data ({len(full_texts):,} samples) for final model")
    elif split_info:
        train_idx = split_info["splits"]["train"]
        valid_idx = split_info["splits"].get("valid")
        test_idx = split_info["splits"]["test"]
        use_idx = train_idx + (valid_idx if valid_idx else [])
        X_test = [texts[i] for i in test_idx]
        y_test = [labels[i] for i in test_idx]
        texts = [texts[i] for i in use_idx]
        labels = [labels[i] for i in use_idx]
        print(f"  Retraining on train set only ({len(use_idx):,} samples)")
        print(f"  Test set for evaluation: {len(X_test):,} samples (held out from split)")
    print(f"  Best params: {best_params}")

    # Device
    device_info = detect_device()
    device_str = device_info["recommended_device"]

    # MLflow
    if use_mlflow:
        setup_mlflow()

    from contextlib import nullcontext
    _ctx = mlflow_run(run_name=f"final_{best_model_name.replace(' ', '_')}") if use_mlflow else nullcontext()
    with _ctx:
        if use_mlflow:
            log_params({"model": spec.display_name, "category": spec.category,
                        "best_params": str(best_params)})

        # Phase 1: Train on train-only data, evaluate on Stage 2 test set
        start_time = time.time()
        model, vectorizer, metrics = retrain_full(
            texts, labels, spec, best_params, device_str, embedding_path, seed,
            X_test=X_test, y_test=y_test,
        )
        phase1_time = time.time() - start_time
        print(f"  Phase 1 completed in {format_duration(phase1_time)}")
        if metrics:
            print(f"  Test eval (Stage 2 test set): accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")

        # Phase 2 (retrain_on_full only): retrain on all data for final deployment model
        if full_texts is not None:
            print(f"  Phase 2: retraining on full data ({len(full_texts):,} samples)...")
            start_time = time.time()
            model, vectorizer, _ = retrain_full(
                full_texts, full_labels, spec, best_params, device_str, embedding_path, seed,
                X_test=None, y_test=None,  # no eval needed, just train on all data
            )
            phase2_time = time.time() - start_time
            print(f"  Phase 2 completed in {format_duration(phase2_time)}")
            total_time = phase1_time + phase2_time
        else:
            total_time = phase1_time
        print(f"  Total time: {format_duration(total_time)}")

        # Save model artifacts
        model_type_map = {
            "traditional_ml": "sklearn",
            "deep_learning": "pytorch",
            "transformer": "transformers",
        }
        model_type = model_type_map[spec.category]
        saved_files = save_model_artifacts(
            model, vectorizer, output_dir, spec.display_name, model_type,
            training_mode=getattr(model, 'training_mode', spec.training_mode),
        )
        print(f"  Model saved to: {output_dir}/final_model/")

        if use_mlflow and metrics:
            log_metrics({
                "final_accuracy": metrics.get("accuracy", 0),
                "final_f1": metrics.get("f1", 0),
                "final_precision": metrics.get("precision", 0),
                "final_recall": metrics.get("recall", 0),
            })
            for name, path in saved_files.items():
                log_artifact(path)

        # Generate HTML report
        print_subheader("Generating Training Report")
        report_data = _build_report_data(analysis, training_results, spec,
                                         best_params, metrics, saved_files)
        report_path = generate_html_report(report_data, output_dir)
        print(f"  Report: {report_path}")
        try:
            _abs_report = str(pathlib.Path(report_path).resolve())
            webbrowser.open(f"file:///{_abs_report.replace(chr(92), '/')}")
        except Exception:
            pass  # headless / no-browser environments

        if use_mlflow:
            log_artifact(report_path)

        # Generate deployment artifacts
        print_subheader("Generating Deployment Artifacts")
        relative_model_path = f"final_model/{pathlib.Path(saved_files['model']).name}"
        _vec_abs = saved_files.get("vectorizer") or saved_files.get("tokenizer")
        relative_vectorizer_path = f"final_model/{pathlib.Path(_vec_abs).name}" if _vec_abs else None
        deploy_files = generate_deployment(
            model_path=relative_model_path,
            vectorizer_path=relative_vectorizer_path,
            model_type=model_type,
            output_dir=f"{output_dir}/deploy",
        )
        for name, path in deploy_files.items():
            print(f"  {name}: {path}")

        if use_mlflow:
            for name, path in deploy_files.items():
                log_artifact(path)

    # Final summary
    print_header("Pipeline Complete!")
    print(f"  Final model: {output_dir}/final_model/")
    print(f"  Report:      {output_dir}/training_report.html")
    print(f"  Deployment:  {output_dir}/deploy/")
    if use_mlflow:
        print(f"  MLflow:      {output_dir}/mlflow.db")
        print(f"  Run: mlflow ui --backend-store-uri sqlite:///{output_dir}/mlflow.db")

    return {
        "model_path": str(pathlib.Path(output_dir) / "final_model"),
        "report_path": report_path,
        "deploy_path": str(pathlib.Path(output_dir) / "deploy"),
        "metrics": metrics,
    }


def _find_best_model(training_results: dict, best_model_name: str) -> dict:
    """Find the best model config from training results.

    Matching order:
      1. Exact display_name match
      2. Exact match after stripping whitespace
      3. Case-insensitive exact match
      4. best_model_name is a substring of the display_name (one-way only)
    """
    # Build lookup
    names = list(training_results.keys())

    # Tier 1: exact
    if best_model_name in training_results:
        candidates = [best_model_name]
    else:
        # Tier 2: strip both sides
        stripped_match = next(
            (n for n in names if n.strip() == best_model_name.strip()), None
        )
        if stripped_match:
            candidates = [stripped_match]
        else:
            # Tier 3: case-insensitive
            lower_match = next(
                (n for n in names if n.lower() == best_model_name.lower()), None
            )
            if lower_match:
                candidates = [lower_match]
            else:
                # Tier 4: case-insensitive substring match.
                # Prefer shorter names (more specific) and matches at the start.
                _lower = best_model_name.lower()
                _all = [n for n in names if _lower in n.lower()]
                _all.sort(key=lambda n: (len(n), not n.lower().startswith(_lower)))
                candidates = _all

    if not candidates:
        return None

    model_name = candidates[0]
    model_data = training_results[model_name]

    # Extract spec info and best params
    tuned = model_data.get("tuned", {})
    baseline = model_data.get("baseline", {})
    source = tuned if tuned else baseline
    if not source:
        raise ValueError(
            f"Model '{model_name}' has no training data in training_results.json. "
            f"The file may be corrupted or from an incomplete run."
        )
    spec_dict = {
        "name": source.get("name", model_name),
        "category": source.get("category", "traditional_ml"),
        "display_name": model_name,
        "priority": 1,
        "params": source.get("params", {}),
        "vectorizer": source.get("vectorizer", "tfidf"),
        "ngram_range": source.get("ngram_range", (1, 1)),
        "use_embedding": source.get("use_embedding", "none"),
        "freeze_embeddings": source.get("freeze_embeddings", False),
        "training_mode": source.get("training_mode", "full_ft"),
    }
    return {
        "spec": spec_dict,
        "best_params": tuned.get("best_params", {}),
        "baseline_metrics": baseline,
        "tuned_metrics": tuned,
    }


def _build_report_data(analysis: dict, training_results: dict, spec: ModelSpec,
                       best_params: dict, metrics: dict,
                       saved_files: dict) -> dict:
    """Assemble the results dict for report generation."""
    dataset = analysis.get("dataset", {})
    return {
        "dataset": {
            "total_samples": dataset.get("total_samples"),
            "class_0_count": dataset.get("class_0_count"),
            "class_1_count": dataset.get("class_1_count"),
            "class_ratio": dataset.get("class_ratio"),
            "is_english": dataset.get("is_english"),
            "text_stats": dataset.get("text_stats", {}),
        },
        "system": {
            "platform": analysis.get("system", {}).get("platform"),
            "python_version": analysis.get("system", {}).get("python_version"),
            "device": analysis.get("device", {}),
            "libraries": analysis.get("python_env", {}),
        },
        "model_scheme": [
            {"name": spec.name, "category": spec.category,
             "display_name": spec.display_name, "priority": 1},
        ],
        "training": training_results,
        "final_model": {
            "model_name": spec.display_name,
            "params": {**spec.params, **best_params},
            "test_accuracy": metrics.get("accuracy"),
            "test_precision": metrics.get("precision"),
            "test_recall": metrics.get("recall"),
            "test_f1": metrics.get("f1"),
            "confusion_matrix": metrics.get("confusion_matrix", [[0, 0], [0, 0]]),
            "saved_path": f"output/final_model/",
        },
        "deployment": {
            "files": saved_files,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Save Model, Generate Report & Deployment"
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--text-col", required=True, help="Text column name")
    parser.add_argument("--label-col", required=True, help="Label column name (0/1)")
    parser.add_argument("--analysis", required=True,
                        help="Path to analysis.json from Stage 1")
    parser.add_argument("--training-results", required=True,
                        help="Path to training_results.json from Stage 4")
    parser.add_argument("--best-model", required=True,
                        help="Display name of the best model to save")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow tracking")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--embedding-path", default=None,
                        help="Path to GloVe/fastText/Word2Vec embedding file (backward-compat)")
    parser.add_argument("--glove-path", default=None,
                        help="Path to GloVe embedding file")
    parser.add_argument("--word2vec-path", default=None,
                        help="Path to Word2Vec embedding file")
    parser.add_argument("--fasttext-path", default=None,
                        help="Path to fastText embedding file")
    parser.add_argument("--encoding", default=None,
                        help="CSV file encoding (e.g. utf-8, latin-1). If not specified, auto-detected via fallback chain.")
    parser.add_argument("--split", default=None,
                        help="Path to split_info.json from Stage 2 (optional). "
                             "If provided, evaluates on test set and prompts "
                             "for full-data retrain confirmation.")
    parser.add_argument("--retrain-on-full", action="store_true",
                        help="Use with --split. Retrain on full data "
                             "(train+valid+test combined) for deployment.")
    args = parser.parse_args()

    for f in [args.csv, args.analysis, args.training_results]:
        if not os.path.exists(f):
            print(f"[ERROR] File not found: {f}")
            sys.exit(1)

    # Build embedding path map from CLI args.
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
        generate_all_artifacts(
            csv_path=args.csv,
            text_col=args.text_col,
            label_col=args.label_col,
            analysis_path=args.analysis,
            training_results_path=args.training_results,
            best_model_name=args.best_model,
            output_dir=args.output_dir,
            use_mlflow=not args.no_mlflow,
            seed=args.seed,
            embedding_path=embedding_paths,
            encoding=args.encoding,
            split_path=args.split,
            retrain_on_full=args.retrain_on_full,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
