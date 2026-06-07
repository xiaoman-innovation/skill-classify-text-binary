"""
MLflow experiment tracking with graceful degradation.
If MLflow is not installed, all functions become safe no-ops.
"""

import os
import warnings
from contextlib import contextmanager

_MLFLOW_AVAILABLE = False
_MLFLOW_WARNED = False

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    pass


def _warn_once():
    global _MLFLOW_WARNED
    if not _MLFLOW_WARNED:
        warnings.warn("MLflow is not installed. Experiment tracking disabled. "
                       "Install with: pip install mlflow")
        _MLFLOW_WARNED = True


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_mlflow(
    tracking_uri: str = "sqlite:///output/mlflow.db",
    experiment_name: str = "text_binary_classification",
    artifact_location: str = "output/mlflow_artifacts",
) -> str:
    """
    Initialize MLflow with SQLite backend (no server needed).
    Returns experiment ID string, or None if MLflow unavailable.
    """
    if not _MLFLOW_AVAILABLE:
        _warn_once()
        return None

    mlflow.set_tracking_uri(tracking_uri)

    # Get or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            exp_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location,
            )
        else:
            exp_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
        return exp_id
    except Exception as e:
        warnings.warn(f"MLflow setup failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------

class _NoOpRun:
    """No-op replacement for mlflow.ActiveRun."""
    def __enter__(self): return self
    def __exit__(self, *args): pass


@contextmanager
def mlflow_run(run_name: str = "training", nested: bool = False):
    """
    Context manager for an MLflow run. Safe to use when MLflow is disabled.
    Usage:
        with mlflow_run("baseline_svm") as run:
            log_params({"C": 1.0})
            log_metrics({"accuracy": 0.92})
    """
    if not _MLFLOW_AVAILABLE:
        _warn_once()
        yield _NoOpRun()
        return

    try:
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            yield run
    except Exception as e:
        warnings.warn(f"MLflow run failed: {e}")
        # Re-raise so the caller can handle the actual error.
        # NOTE: a generator-based contextmanager MUST NOT yield again
        # after .throw() was called on it — doing so causes the
        # confusing "generator didn't stop after throw()" RuntimeError
        # that masks the original exception.
        raise


# ---------------------------------------------------------------------------
# Logging Functions
# ---------------------------------------------------------------------------

def log_params(params: dict) -> None:
    """Batch log parameters. Truncates long values."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        safe = {}
        for k, v in params.items():
            val = str(v)[:250] if not isinstance(v, (int, float, bool)) else v
            safe[k] = val
        mlflow.log_params(safe)
    except Exception:
        pass


def log_metrics(metrics: dict, step: int = None) -> None:
    """Batch log metrics."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        mlflow.log_metrics(metrics, step=step)
    except Exception:
        pass


def log_artifact(path: str) -> None:
    """Log a local file as an MLflow artifact."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        mlflow.log_artifact(path)
    except Exception:
        pass


def log_model(model, artifact_path: str = "model") -> None:
    """Log a trained model with MLflow. Supports sklearn, PyTorch, and HF models."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        # Try sklearn log first
        import sklearn.base
        if isinstance(model, sklearn.base.BaseEstimator):
            mlflow.sklearn.log_model(model, artifact_path)
            return
    except Exception:
        pass
    try:
        # Check for transformers BEFORE torch.nn.Module since
        # PreTrainedModel is a subclass of torch.nn.Module
        import transformers
        if isinstance(model, transformers.PreTrainedModel):
            mlflow.transformers.log_model(model, artifact_path)
            return
    except Exception:
        pass
    try:
        # Try PyTorch log
        import torch.nn
        if isinstance(model, torch.nn.Module):
            mlflow.pytorch.log_model(model, artifact_path)
            return
    except Exception:
        pass
    # No matching MLflow model flavour -- silently skip
    return


def log_dataset_summary(stats: dict) -> None:
    """Log dataset statistics as MLflow tags and params."""
    tags = {
        "total_samples": stats.get("total_samples"),
        "class_0_count": stats.get("class_0_count"),
        "class_1_count": stats.get("class_1_count"),
        "class_ratio": stats.get("class_ratio"),
    }
    params = {
        "vocab_size": stats.get("vocab_size"),
        "mean_text_length": stats.get("mean_text_length"),
        "missing_ratio": stats.get("missing_ratio"),
    }
    if _MLFLOW_AVAILABLE:
        try:
            for k, v in tags.items():
                if v is not None:
                    mlflow.set_tag(k, str(v))
            log_params(params)
        except Exception:
            pass


def log_cv_results(cv_results: dict) -> None:
    """Log cross-validation fold results and aggregate metrics."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        for fold in cv_results.get("folds", []):
            fold_num = fold.get("fold", 0)
            mlflow.log_metrics({
                f"fold_{fold_num}_accuracy": fold.get("accuracy"),
                f"fold_{fold_num}_precision": fold.get("precision"),
                f"fold_{fold_num}_recall": fold.get("recall"),
                f"fold_{fold_num}_f1": fold.get("f1"),
            })
        mlflow.log_metrics({
            "cv_mean_accuracy": cv_results.get("mean_accuracy"),
            "cv_std_accuracy": cv_results.get("std_accuracy"),
            "cv_mean_f1": cv_results.get("mean_f1"),
        })
    except Exception:
        pass


def log_confusion_matrix(y_true, y_pred, labels: list = None) -> None:
    """Log confusion matrix as an MLflow figure."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        cm = confusion_matrix(y_true, y_pred, labels=labels or [0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=labels or [0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
    except Exception:
        pass


def log_optimization_history(history: list) -> None:
    """Log Optuna optimization history as MLflow metrics over trials."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        for entry in history:
            trial = entry.get("trial", 0)
            value = entry.get("value")
            if value is not None:
                mlflow.log_metric("optuna_trial_value", value, step=trial)
    except Exception:
        pass
