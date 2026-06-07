"""
Shared utilities: hardware detection, system info, progress bars, seed setting,
English language detection, output directory management.
"""

from __future__ import annotations
import os
import sys
import time
import random
import platform
import socket
import json
import pathlib
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import threading as _threading


# ---------------------------------------------------------------------------
# Training Log File (tee stdout to file)
# ---------------------------------------------------------------------------

class TeeLogger:
    """Duplicate stdout/stderr to a log file for real-time progress monitoring."""

    def __init__(self, log_path: str, stream, auto_flush: bool = True):
        self.log_path = log_path
        self.stream = stream
        self.auto_flush = auto_flush
        self._file = None

    def open(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._file = open(self.log_path, "w", encoding="utf-8", buffering=1)

    def write(self, data):
        self.stream.write(data)
        if self._file is not None and not self._file.closed:
            self._file.write(data)
            if self.auto_flush:
                self._file.flush()

    def flush(self):
        self.stream.flush()
        if self._file is not None and not self._file.closed:
            self._file.flush()

    def close(self):
        if self._file is not None and not self._file.closed:
            self._file.close()

    def fileno(self):
        return self.stream.fileno()

    def isatty(self):
        return self.stream.isatty()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Periodic Progress Reporter (background thread)
# ---------------------------------------------------------------------------

class ProgressReporter:
    """Background thread that periodically prints training progress summary.

    Call ``update(**kwargs)`` from the main thread to refresh state,
    and the reporter will print a summary every `interval_minutes`.
    """

    def __init__(self, interval_minutes: int = 5):
        self._interval = max(interval_minutes, 1) * 60
        self._running = False
        self._thread = None
        self._lock = _threading.Lock()
        self._state = {
            "total_models": 0,
            "current_idx": 0,
            "current_name": "",
            "phase": "",                # "baseline" | "tuning" | ""
            "phase_progress": "",       # e.g. "fold 3/5" or "trial 7/20"
            "completed": [],            # list of (name, baseline_ok, tuned_ok)
            "start_time": None,
        }

    # -- public API -------------------------------------------------------

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = _threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)

    def update(self, **kwargs):
        with self._lock:
            self._state.update(kwargs)

    def add_completed(self, name: str, baseline_ok: bool = False,
                      tuned_ok: bool = False):
        """Append a model to the completed list."""
        with self._lock:
            self._state["completed"].append(
                {"name": name, "baseline": baseline_ok, "tuned": tuned_ok}
            )

    # -- internal ---------------------------------------------------------

    def _loop(self):
        # First report fires after one interval, not immediately.
        while self._running:
            time.sleep(self._interval)
            if not self._running:
                break
            self._report()

    def _report(self):
        with self._lock:
            s = dict(self._state)  # shallow copy under lock
            s['completed'] = list(self._state.get('completed', []))  # deep-copy mutable list
        total = s["total_models"]
        completed = len(s["completed"])
        remaining = total - completed

        lines = []
        lines.append("")
        lines.append("=" * 62)
        lines.append("  \033[1;36m[PROGRESS] Training Status Report\033[0m")
        lines.append("  " + "-" * 58)

        # Overall progress bar
        if total > 0:
            pct = completed / total
            bar_width = 40
            filled = int(bar_width * pct)
            bar = "█" * filled + "░" * (bar_width - filled)
            lines.append(f"  Models: [{bar}] {completed}/{total} ({pct:.0%})")

        # Current model
        if s["current_name"]:
            phase_str = s["phase"] or "preparing"
            detail = f" ({s['phase_progress']})" if s.get("phase_progress") else ""
            lines.append(f"  Current: {s['current_name']}  [{phase_str}]{detail}")

        # Completed models
        if s["completed"]:
            lines.append(f"  Completed ({completed}):")
            for m in s["completed"]:
                flags = []
                if m["baseline"]:
                    flags.append("baseline ✓")
                if m["tuned"]:
                    flags.append("tuned ✓")
                flag_str = " | ".join(flags) if flags else "pending"
                lines.append(f"    - {m['name']}  [{flag_str}]")

        # Remaining models (if we know them)
        if remaining > 0:
            lines.append(f"  Remaining: {remaining} model(s)")

        # Elapsed
        if s["start_time"] is not None:
            elapsed = time.time() - s["start_time"]
            lines.append(f"  Elapsed: {format_duration(elapsed)}")

        lines.append("=" * 62)
        lines.append("")
        # Write directly so it goes through the TeeLogger (stdout → console + file)
        for line in lines:
            print(line)


def setup_training_log(output_dir: str, log_filename: str = "training.log"):
    """
    Redirect stdout to both console and a log file under output_dir.
    Returns the TeeLogger instance so the caller can close it when done.
    """
    log_path = os.path.join(output_dir, log_filename)
    tee = TeeLogger(log_path, sys.stdout)
    tee.open()
    sys.stdout = tee
    return tee


# ---------------------------------------------------------------------------
# Warning & Logging Suppression
# ---------------------------------------------------------------------------

def setup_logging_and_warnings() -> None:
    """
    Suppress noisy third-party warnings and log messages during training.
    Call once at the start of step3_train.py (or any other stage).
    """
    # scikit-learn convergence warnings (common during quick tuning fits)
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings(
        "ignore", message=".*failed to converge.*", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore", message=".*Liblinear failed to converge.*",
    )

    # Optuna trial-level INFO logs (Trial N finished / Best is trial N)
    logging.getLogger("optuna").setLevel(logging.WARNING)

    # HuggingFace / transformers download progress bars
    logging.getLogger("transformers").setLevel(logging.WARNING)

    # MLflow noisy INFO messages
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    # PyTorch DDP / multiprocessing warnings
    warnings.filterwarnings("ignore", message=".*TypedStorage.*", category=UserWarning)


# ---------------------------------------------------------------------------
# Device / Hardware Detection
# ---------------------------------------------------------------------------

def detect_device() -> dict:
    """
    Auto-detect available compute device and hardware specs.
    Priority: CUDA > MPS > CPU.

    Returns dict with keys:
        cpu_cores_physical, cpu_cores_logical, cpu_freq_mhz,
        total_ram_gb, gpu_name, gpu_vram_gb, gpu_cuda_cores, gpu_count,
        gpu_type ('cuda'|'mps'|'none'), recommended_device (str),
        has_gpu (bool)
    """
    import psutil

    info = {
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": None,
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "gpu_name": None,
        "gpu_vram_gb": None,
        "gpu_cuda_cores": None,
        "gpu_count": 0,
        "gpu_type": "none",
        "recommended_device": "cpu",
        "has_gpu": False,
    }

    # CPU frequency
    try:
        freq = psutil.cpu_freq()
        if freq is not None:
            info["cpu_freq_mhz"] = round(freq.max, 1) if freq.max else round(freq.current, 1)
    except Exception:
        pass

    # Try CUDA
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_type"] = "cuda"
            info["recommended_device"] = "cuda"
            info["has_gpu"] = True
            name = torch.cuda.get_device_name(0)
            info["gpu_name"] = name
            try:
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                info["gpu_vram_gb"] = round(vram_bytes / (1024 ** 3), 1)
            except Exception:
                info["gpu_vram_gb"] = None
            # GPU CUDA cores: try pynvml first, then lookup table
            info["gpu_cuda_cores"] = _get_gpu_cuda_cores(name)
            return info
    except ImportError:
        pass

    # Try MPS (macOS Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu_count"] = 1
            info["gpu_type"] = "mps"
            info["gpu_name"] = "Apple MPS"
            info["gpu_vram_gb"] = None  # MPS shares unified memory
            info["gpu_cuda_cores"] = None  # MPS uses Apple GPU cores
            info["recommended_device"] = "mps"
            info["has_gpu"] = True
            return info
    except ImportError:
        pass

    return info


# GPU CUDA cores lookup table (common models)
_GPU_CUDA_CORES = {
    # RTX 50-series (Blackwell)
    "rtx 5090": 21760, "rtx 5080": 10752, "rtx 5070 ti": 8960,
    "rtx 5070": 6400, "rtx 5060 ti": 4608, "rtx 5060": 3840,
    # RTX 40-series (Ada Lovelace)
    "rtx 4090": 16384, "rtx 4080 super": 10240, "rtx 4080": 9728,
    "rtx 4070 ti super": 8448, "rtx 4070 ti": 7680, "rtx 4070 super": 7168,
    "rtx 4070": 5888, "rtx 4060 ti": 4352, "rtx 4060": 3840, "rtx 4050": 2560,
    # RTX 30-series (Ampere)
    "rtx 3090 ti": 10752, "rtx 3090": 10496, "rtx 3080 ti": 10240,
    "rtx 3080": 8704, "rtx 3070 ti": 6144, "rtx 3070": 5888,
    "rtx 3060 ti": 4864, "rtx 3060": 3584, "rtx 3050": 2560,
    # RTX 20-series (Turing)
    "rtx 2080 ti": 4352, "rtx 2080 super": 3072, "rtx 2080": 2944,
    "rtx 2070 super": 2560, "rtx 2070": 2304, "rtx 2060 super": 2176,
    "rtx 2060": 1920,
    # GTX 16-series / 10-series
    "gtx 1660 ti": 1536, "gtx 1660 super": 1408, "gtx 1660": 1408,
    "gtx 1650 super": 1280, "gtx 1650 ti": 1024, "gtx 1650": 896,
    "gtx 1080 ti": 3584, "gtx 1080": 2560, "gtx 1070 ti": 2432,
    "gtx 1070": 1920, "gtx 1060": 1280, "gtx 1050 ti": 768, "gtx 1050": 640,
    # Quadro / Professional
    "rtx a6000": 10752, "rtx a5000": 8192, "rtx a4000": 6144,
    "rtx a2000": 3328, "quadro rtx 8000": 4608, "quadro rtx 6000": 4608,
    "quadro rtx 5000": 3072, "quadro rtx 4000": 2304,
    # Data center
    "a100": 6912, "a40": 10752, "a30": 3584, "a10": 9216,
    "h100": 18432, "h200": 18432,
    "v100": 5120, "t4": 2560, "l40s": 18176, "l40": 18176, "l4": 7680,
    # Tesla
    "tesla t4": 2560, "tesla v100": 5120, "tesla p100": 3584,
    "tesla p40": 3840, "tesla p4": 2560, "tesla k80": 2496 * 2, "tesla m40": 3072,
}


def _get_gpu_cuda_cores(gpu_name: str) -> int | None:
    """Get CUDA core count via pynvml or lookup table."""
    # Try pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            cores = pynvml.nvmlDeviceGetNumCudaCores(handle)
            if cores:
                return cores
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    # Fallback: lookup table
    name_lower = gpu_name.lower()
    # Try longest match first
    for key, cores in sorted(_GPU_CUDA_CORES.items(), key=lambda x: -len(x[0])):
        if key in name_lower:
            return cores
    return None


# ---------------------------------------------------------------------------
# System / Library Info
# ---------------------------------------------------------------------------

def get_system_info() -> dict:
    """Collect OS, Python, and key library version information."""
    info = {
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "hostname": socket.gethostname(),
    }

    # Key library versions (tolerant of missing packages)
    _try_version(info, "numpy", "numpy_version")
    _try_version(info, "pandas", "pandas_version")
    _try_version(info, "scipy", "scipy_version")

    _try_package(info, "sklearn", "sklearn_version", lambda m: m.__version__)
    _try_package(info, "nltk", "nltk_version", lambda m: m.__version__)
    _try_package(info, "joblib", "joblib_version", lambda m: m.__version__)

    # PyTorch
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
    except ImportError:
        info["torch_version"] = None

    # Transformers
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = None

    # Optuna
    try:
        import optuna
        info["optuna_version"] = optuna.__version__
    except ImportError:
        info["optuna_version"] = None

    # MLflow
    try:
        import mlflow
        info["mlflow_version"] = mlflow.__version__
    except ImportError:
        info["mlflow_version"] = None

    # tqdm
    try:
        import tqdm
        info["tqdm_version"] = tqdm.__version__
    except ImportError:
        info["tqdm_version"] = None

    return info


def _try_version(info: dict, module_name: str, key: str):
    """Try to get __version__ from an already-importable module."""
    mod = sys.modules.get(module_name)
    if mod is not None and hasattr(mod, "__version__"):
        info[key] = mod.__version__
    else:
        info[key] = None


def _try_package(info: dict, import_name: str, key: str, get_version):
    """Try to import a package and extract its version."""
    try:
        mod = __import__(import_name)
        info[key] = get_version(mod)
    except ImportError:
        info[key] = None


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seed for Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Progress & Timing
# ---------------------------------------------------------------------------

def create_progress_bar(total: int, desc: str = "", unit: str = "it",
                        position: int = 0, leave: bool = True):
    """Standardized tqdm progress bar with clean compact styling."""
    from tqdm import tqdm
    return tqdm(
        total=total, desc=desc, unit=unit, position=position,
        leave=leave, ncols=80, bar_format=(
            "{desc:>12s} {percentage:3.0f}% {bar:20} {n:3d}/{total_fmt}"
            " [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        ),
    )


def format_duration(seconds: float) -> str:
    """Human-readable duration string. E.g. '2h 35m 12s'."""
    if seconds < 0:
        return "0s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def estimate_remaining(start_time: float, current: int, total: int) -> str:
    """Estimate remaining time based on linear progress."""
    if current == 0:
        return "calculating..."
    elapsed = time.time() - start_time
    rate = current / elapsed
    remaining = (total - current) / rate
    return format_duration(remaining)


# ---------------------------------------------------------------------------
# Output Directory
# ---------------------------------------------------------------------------

def ensure_output_dir(base: str = "output") -> pathlib.Path:
    """Create output/ directory (and subdirs) if they don't exist."""
    out = pathlib.Path(base)
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_subdirs(base: str, *subdirs: str) -> dict:
    """Create subdirectories under base. Returns {name: Path} mapping."""
    base_path = pathlib.Path(base)
    result = {}
    for sub in subdirs:
        p = base_path / sub
        p.mkdir(parents=True, exist_ok=True)
        result[sub] = p
    return result


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _warn_non_serializable(obj, path: str = "<root>") -> None:
    """Recursively check for types that will be coerced by json.dump default=str."""
    import collections.abc
    if isinstance(obj, dict):
        for k, v in obj.items():
            _warn_non_serializable(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _warn_non_serializable(v, f"{path}[{i}]")
    elif isinstance(obj, bytes):
        print(f"  [WARN] save_json: bytes value at {path} will be serialized via str() "
              f"(length={len(obj)}). Consider decoding to str before saving.")
    elif obj is not None and not isinstance(obj, (bool, int, float, str, collections.abc.Mapping,
                                                   collections.abc.Sequence)):
        print(f"  [WARN] save_json: non-serializable type {type(obj).__name__} at "
              f"{path} will be coerced via str(). Consider converting to a JSON-safe type.")


def save_json(data: dict, path: str, indent: int = 2) -> None:
    """Save dict to JSON file atomically with UTF-8 encoding."""
    import os
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    # Warn about non-serializable types that will be coerced by default=str
    _warn_non_serializable(data)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        os.replace(tmp, str(out))  # Atomic on same filesystem
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_json(path: str) -> dict:
    """Load JSON file with UTF-8 encoding. Returns dict; raises on error."""
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)
    if not isinstance(result, dict):
        raise TypeError(f"Expected JSON object at {path}, got {type(result).__name__}")
    return result


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_ENCODINGS_TO_TRY = [
    "utf-8", "utf-8-sig",
    "gbk", "gb2312", "gb18030",
    "latin-1", "cp1252", "ISO-8859-1",
]


def read_csv_safe(csv_path: str, encoding: str = None) -> tuple:
    """
    Read CSV with encoding fallback chain.
    If `encoding` is provided, it is tried first before falling back.
    Reports the encoding actually used.
    Returns (DataFrame, encoding_used).
    Raises ValueError if all encodings fail.
    """
    encodings = [encoding] + _ENCODINGS_TO_TRY if encoding else _ENCODINGS_TO_TRY
    seen = set()
    for enc in encodings:
        if enc is None or enc in seen:
            continue
        seen.add(enc)
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            if encoding is not None and enc != encoding:
                print(f"  [NOTE] File encoding is '{enc}', not '{encoding}'. "
                      f"Auto-detected and loaded correctly.")
            elif encoding is None and enc != "utf-8":
                print(f"  [NOTE] File encoding detected as '{enc}'.")
            return df, enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(
        f"Failed to read CSV with encodings: {encodings}. "
        "Please check the file encoding."
    )


# ---------------------------------------------------------------------------
# English Language Detection
# ---------------------------------------------------------------------------

def check_english(texts: list, sample_size: int = 200) -> dict:
    """
    Heuristic English detection using character-range analysis.
    Tries langdetect if installed; falls back to Latin-script ratio.

    Returns: {"is_english": bool, "non_english_ratio": float, "method": str,
              "details": str}
    """
    if len(texts) == 0:
        return {"is_english": True, "non_english_ratio": 0.0,
                "method": "empty", "details": "No texts to check"}

    # Try langdetect first
    try:
        import langdetect
        langdetect.DetectorFactory.seed = 0
        sample = _get_sample(texts, sample_size)
        non_en = 0
        for t in sample:
            t_clean = str(t).strip()
            if len(t_clean) < 10:
                continue
            try:
                if langdetect.detect(t_clean) != "en":
                    non_en += 1
            except langdetect.LangDetectException:
                pass
        ratio = non_en / max(len(sample), 1)
        return {
            "is_english": ratio < 0.30,
            "non_english_ratio": round(ratio, 4),
            "method": "langdetect",
            "details": f"{non_en}/{len(sample)} sampled texts detected as non-English",
        }
    except ImportError:
        pass

    # Fallback: Latin-script character ratio heuristic
    sample = _get_sample(texts, sample_size)
    non_latin_count = 0
    total = 0
    for t in sample:
        t_clean = str(t).strip()
        if len(t_clean) == 0:
            continue
        total += 1
        latin_chars = sum(1 for c in t_clean
                          if c.isascii() and (c.isalpha() or c.isspace()))
        if latin_chars / max(len(t_clean), 1) < 0.85:
            non_latin_count += 1

    ratio = non_latin_count / max(total, 1)
    return {
        "is_english": ratio < 0.30,
        "non_english_ratio": round(ratio, 4),
        "method": "latin_ratio_heuristic",
        "details": f"{non_latin_count}/{total} sampled texts have low Latin-script ratio",
    }


def detect_language_distribution(texts: list, sample_size: int = 500) -> dict:
    """
    Detect detailed language distribution using langdetect (or fallback).

    Returns dict:
        method (str), languages (dict of lang_code -> {count, ratio}),
        is_english_majority (bool), non_english_ratio (float)
    """
    if len(texts) == 0:
        return {
            "method": "empty",
            "languages": {},
            "is_english_majority": True,
            "non_english_ratio": 0.0,
        }

    sample = _get_sample(texts, sample_size)

    # Try langdetect
    try:
        import langdetect
        langdetect.DetectorFactory.seed = 0
        lang_counts = {}
        skipped = 0
        for t in sample:
            t_clean = str(t).strip()
            if len(t_clean) < 10:
                skipped += 1
                continue
            try:
                lang = langdetect.detect(t_clean)
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            except langdetect.LangDetectException:
                skipped += 1

        total_detected = sum(lang_counts.values())
        if total_detected == 0:
            return {
                "method": "langdetect",
                "languages": {},
                "is_english_majority": True,
                "non_english_ratio": 0.0,
                "sampled": len(sample),
                "detected": 0,
                "skipped": skipped,
            }

        non_en = sum(c for l, c in lang_counts.items() if l != "en")
        languages = {}
        for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1]):
            languages[lang] = {
                "count": cnt,
                "ratio": round(cnt / total_detected, 4),
            }

        return {
            "method": "langdetect",
            "languages": languages,
            "is_english_majority": (non_en / total_detected) < 0.30,
            "non_english_ratio": round(non_en / total_detected, 4),
            "sampled": len(sample),
            "detected": total_detected,
            "skipped": skipped,
        }
    except ImportError:
        pass

    # Fallback: Latin-script ratio (can't distinguish individual non-English languages)
    non_latin = 0
    total = 0
    for t in sample:
        t_clean = str(t).strip()
        if len(t_clean) == 0:
            continue
        total += 1
        latin_chars = sum(1 for c in t_clean
                          if c.isascii() and (c.isalpha() or c.isspace()))
        if latin_chars / max(len(t_clean), 1) < 0.85:
            non_latin += 1

    ratio = non_latin / max(total, 1)
    return {
        "method": "latin_ratio_heuristic",
        "languages": {"en": {"count": total - non_latin, "ratio": round(1 - ratio, 4)}}
        if total > 0 else {},
        "is_english_majority": ratio < 0.30,
        "non_english_ratio": round(ratio, 4),
        "sampled": len(sample),
        "detected": total,
        "skipped": len(sample) - total,
    }


def _get_sample(texts: list, sample_size: int) -> list:
    """Get a random sample (or all if fewer than sample_size)."""
    if len(texts) <= sample_size:
        return texts
    indices = np.random.RandomState(42).choice(len(texts), sample_size,
                                                replace=False)
    return [texts[i] for i in indices]


# ---------------------------------------------------------------------------
# Storage Detection
# ---------------------------------------------------------------------------

def detect_storage(output_dir: str = "output") -> dict:
    """
    Detect storage info: free space, disk type (SSD/HDD), read/write speed.

    Returns dict:
        free_space_gb, total_space_gb, disk_type, read_speed_mbs, write_speed_mbs,
        mount_point
    """
    import tempfile
    import time

    info = {
        "free_space_gb": None,
        "total_space_gb": None,
        "disk_type": "unknown",
        "read_speed_mbs": None,
        "write_speed_mbs": None,
        "mount_point": None,
    }

    # Free / total space
    try:
        target_dir = os.path.abspath(output_dir) if output_dir else os.getcwd()
        usage = __import__("psutil").disk_usage(target_dir)
        info["free_space_gb"] = round(usage.free / (1024 ** 3), 1)
        info["total_space_gb"] = round(usage.total / (1024 ** 3), 1)
        info["mount_point"] = target_dir
    except Exception:
        pass

    # Disk type (SSD / HDD)
    info["disk_type"] = _detect_disk_type()

    # Read/write speed benchmark (simple)
    try:
        read_speed, write_speed = _benchmark_disk_speed()
        info["read_speed_mbs"] = read_speed
        info["write_speed_mbs"] = write_speed
    except Exception:
        pass

    return info


def _detect_disk_type() -> str:
    """Detect if the current working drive is SSD or HDD."""
    system = platform.system()

    if system == "Linux":
        try:
            import subprocess
            result = subprocess.run(
                ["lsblk", "-d", "-o", "name,rota", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                for dev in data.get("blockdevices", []):
                    if dev.get("rota") == "0":
                        return "SSD"
                    elif dev.get("rota") == "1":
                        return "HDD"
                return "SSD"  # default assumption for modern systems
        except Exception:
            pass

    elif system == "Windows":
        try:
            import subprocess
            result = subprocess.run(
                ["powershell", "-Command",
                 "(Get-PhysicalDisk | Select-Object MediaType | "
                 "ForEach-Object { $_.MediaType }) -join ','"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                media_types = result.stdout.strip().lower()
                if "ssd" in media_types:
                    return "SSD"
                elif "hdd" in media_types:
                    return "HDD"
                elif "nvme" in media_types:
                    return "SSD (NVMe)"
        except Exception:
            pass

    elif system == "Darwin":  # macOS
        try:
            import subprocess
            result = subprocess.run(
                ["diskutil", "info", "/"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                output = result.stdout.lower()
                if "solid state" in output or "ssd" in output:
                    return "SSD"
                elif "rotational" in output:
                    return "HDD"
        except Exception:
            pass

    return "unknown"


def _benchmark_disk_speed(file_size_mb: int = 50) -> tuple:
    """Simple disk read/write speed benchmark. Returns (read_mbs, write_mbs)."""
    import tempfile
    import time

    data = b'\x00' * (file_size_mb * 1024 * 1024)

    # Write benchmark
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        t0 = time.perf_counter()
        tmp.write(data)
        tmp.flush()
        try:
            os.fsync(tmp.fileno())
        except (OSError, AttributeError):
            pass  # fsync may not be supported on all platforms / file types
        t1 = time.perf_counter()
        write_time = t1 - t0
        write_speed = round(file_size_mb / max(write_time, 0.001), 1)

        # Read benchmark
        t0 = time.perf_counter()
        with open(tmp.name, "rb") as f:
            _ = f.read()
        t1 = time.perf_counter()
        read_time = t1 - t0
        read_speed = round(file_size_mb / max(read_time, 0.001), 1)
    finally:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return read_speed, write_speed


# ---------------------------------------------------------------------------
# Python Environment Check (YAML-based)
# ---------------------------------------------------------------------------

def check_python_env(yaml_path: str) -> dict:
    """
    Check installed Python packages against a requirements YAML file.

    YAML format:
        packages:
          numpy: ">=1.24"
          pandas: ">=2.0"
          torch: ">=2.0"
          ...

    Returns dict:
        ok (bool), installed (dict of pkg -> version),
        missing (list of pkg), outdated (dict of pkg -> {required, installed})
    """
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.exists(yaml_path):
        return {
            "ok": False,
            "error": f"YAML file not found: {yaml_path}",
            "installed": {},
            "missing": [],
            "outdated": {},
        }

    # Parse YAML
    try:
        packages = _parse_requirements_yaml(yaml_path)
    except Exception as e:
        return {
            "ok": False,
            "error": f"Failed to parse YAML: {e}",
            "installed": {},
            "missing": [],
            "outdated": {},
        }

    installed_pkgs = {}
    missing_pkgs = []
    outdated_pkgs = {}

    for pkg_name, version_spec in packages.items():
        installed_ver = _get_package_version(pkg_name)
        if installed_ver is None:
            missing_pkgs.append(pkg_name)
        else:
            installed_pkgs[pkg_name] = installed_ver
            if version_spec and not _version_matches(installed_ver, version_spec):
                outdated_pkgs[pkg_name] = {
                    "required": version_spec,
                    "installed": installed_ver,
                }

    return {
        "ok": len(missing_pkgs) == 0 and len(outdated_pkgs) == 0,
        "installed": installed_pkgs,
        "missing": missing_pkgs,
        "outdated": outdated_pkgs,
        "total_required": len(packages),
        "total_installed": len(installed_pkgs),
        "total_missing": len(missing_pkgs),
        "total_outdated": len(outdated_pkgs),
    }


def _parse_requirements_yaml(path: str) -> dict:
    """Parse a simple requirements YAML file. Returns {pkg: version_spec}."""
    # Try PyYAML first, fallback to simple line parser
    try:
        import yaml as _yaml
    except ImportError:
        _yaml = None

    if _yaml is not None:
        with open(path, "r", encoding="utf-8") as f:
            data = _yaml.safe_load(f)
        return data.get("packages", {}) if isinstance(data, dict) else {}

    # Fallback: simple line-by-line parser for YAML subset
    packages = {}
    with open(path, "r", encoding="utf-8") as f:
        in_packages = False
        for line in f:
            line = line.rstrip()
            if line.strip().startswith("#"):
                continue
            if line.strip() == "packages:":
                in_packages = True
                continue
            if in_packages:
                if line.startswith("  ") or line.startswith("\t"):
                    stripped = line.strip()
                    if ":" in stripped:
                        parts = stripped.split(":", 1)
                        pkg = parts[0].strip()
                        ver = parts[1].strip().strip('"').strip("'") if len(parts) > 1 else ""
                        if pkg:
                            packages[pkg] = ver
                elif stripped := line.strip():
                    # End of packages section
                    if not line.startswith(" ") and not line.startswith("\t"):
                        break
    return packages


def _get_package_version(pkg_name: str) -> str | None:
    """Get installed package version. Returns None if not installed."""
    # Try importlib.metadata (Python 3.8+)
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(pkg_name)
        except PackageNotFoundError:
            pass
    except ImportError:
        pass

    # Try pkg_resources (fallback)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pkg_resources.*")
            import pkg_resources
        return pkg_resources.get_distribution(pkg_name).version
    except Exception:
        pass

    # Try __import__ + __version__
    _version_map = {
        "scikit-learn": "sklearn",
        "scikit_learn": "sklearn",
    }
    import_name = _version_map.get(pkg_name, pkg_name.replace("-", "_"))
    try:
        mod = __import__(import_name)
        if hasattr(mod, "__version__"):
            return str(mod.__version__)
        if hasattr(mod, "VERSION"):
            return str(mod.VERSION)
    except ImportError:
        pass

    return None


def _version_matches(installed: str, required: str) -> bool:
    """Check if installed version satisfies the requirement spec."""
    required = required.strip()
    if not required:
        return True
    try:
        from packaging.version import parse as parse_version
        from packaging.specifiers import SpecifierSet
        spec = SpecifierSet(required)
        return parse_version(installed) in spec
    except ImportError:
        # Fallback: simple comparison operators
        return _simple_version_match(installed, required)


def _simple_version_match(installed: str, required: str) -> bool:
    """Simple version matching without packaging library."""
    import re
    req = required.strip()
    if not req:
        return True

    def _parse(v):
        parts = re.findall(r'\d+', v)
        return tuple(int(p) for p in parts)

    try:
        iv = _parse(installed)
    except Exception:
        return True  # can't parse, assume OK

    # Handle >= X.Y
    m = re.match(r'>=\s*(.+)', req)
    if m:
        return iv >= _parse(m.group(1))

    # Handle > X.Y
    m = re.match(r'>\s*(.+)', req)
    if m:
        return iv > _parse(m.group(1))

    # Handle <= X.Y
    m = re.match(r'<=\s*(.+)', req)
    if m:
        return iv <= _parse(m.group(1))

    # Handle < X.Y
    m = re.match(r'<\s*(.+)', req)
    if m:
        return iv < _parse(m.group(1))

    # Handle == X.Y or exact version
    m = re.match(r'==\s*(.+)', req)
    if m:
        return iv == _parse(m.group(1))

    # Handle != X.Y
    m = re.match(r'!=\s*(.+)', req)
    if m:
        return iv != _parse(m.group(1))

    # Plain version number as minimum
    return iv >= _parse(req)


# ---------------------------------------------------------------------------
# Network Diagnostics — pre-check connectivity before training
# ---------------------------------------------------------------------------

_NET_CHECK_SOURCES = {
    "huggingface.co": "https://huggingface.co",
    "hf-mirror.com": "https://hf-mirror.com",
    "modelscope.cn": "https://modelscope.cn",
}

_NET_CHECK_TIMEOUT = 5  # seconds per source


def _classify_network_error(exc: Exception) -> str:
    """Classify a network error into a human-readable type."""
    msg = str(exc).lower()
    if "getaddrinfo" in msg or "name or service not known" in msg or "nodename nor servname" in msg:
        return "DNS"
    if "timed out" in msg or "timeout" in msg or "timedout" in msg:
        return "timeout"
    if "403" in msg or "forbidden" in msg:
        return "403_forbidden"
    if "connection refused" in msg or "refused" in msg:
        return "connection_refused"
    if "connection reset" in msg or "reset" in msg:
        return "connection_reset"
    if "ssl" in msg or "certificate" in msg or "tls" in msg:
        return "SSL"
    return "unknown"


def run_network_diagnostics() -> dict:
    """
    Test HTTP connectivity to HuggingFace and common mirrors.

    Returns a dict keyed by source name, each containing:
        reachable: bool
        latency_ms: int (rounded) or None
        error: str or None
        error_type: str or None (DNS/timeout/403_forbidden/...)

    Also returns top-level keys:
        any_reachable: bool
        proxy_detected: str or None
        dns_ok: bool
        recommendations: list[str]
    """
    import urllib.request
    import urllib.error
    import time as _time
    import socket as _socket

    results = {}
    any_ok = False

    # ---- per-source HTTP check ----
    for name, url in _NET_CHECK_SOURCES.items():
        try:
            req = urllib.request.Request(url, method="HEAD")
            t0 = _time.monotonic()
            resp = urllib.request.urlopen(req, timeout=_NET_CHECK_TIMEOUT)
            elapsed = round((_time.monotonic() - t0) * 1000)
            results[name] = {
                "reachable": True,
                "latency_ms": elapsed,
                "error": None,
                "error_type": None,
            }
            any_ok = True
        except Exception as e:
            results[name] = {
                "reachable": False,
                "latency_ms": None,
                "error": str(e)[:200],
                "error_type": _classify_network_error(e),
            }

    # ---- DNS check (resolve huggingface.co) ----
    dns_ok = False
    try:
        _socket.getaddrinfo("huggingface.co", 443, proto=_socket.IPPROTO_TCP)
        dns_ok = True
    except Exception:
        try:
            _socket.getaddrinfo("hf-mirror.com", 443, proto=_socket.IPPROTO_TCP)
            dns_ok = True
        except Exception:
            pass

    # ---- proxy detection ----
    proxy = None
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        p = os.environ.get(var)
        if p:
            proxy = f"{var}={p}"
            break

    # ---- recommendations ----
    recs = []
    if not any_ok:
        if not dns_ok:
            recs.append("DNS 解析失败：请检查网络连接或 DNS 配置")
        if proxy:
            recs.append(f"检测到代理 {proxy}，但所有源均不可达。请检查代理是否正常工作")
        else:
            recs.append("所有下载源不可达。当前处于离线环境，Transformers 模型将自动跳过")
        if any(r["error_type"] == "SSL" for r in results.values()):
            recs.append("SSL 证书错误：可能需要配置系统证书或设置 CURL_CA_BUNDLE")
    elif not results.get("huggingface.co", {}).get("reachable"):
        if results.get("hf-mirror.com", {}).get("reachable"):
            recs.append("huggingface.co 不可达但 hf-mirror.com 可用，训练时将自动切换镜像")
        elif results.get("modelscope.cn", {}).get("reachable"):
            recs.append("仅 modelscope.cn 可达，部分模型可通过 modelscope 下载")

    return {
        **results,
        "any_reachable": any_ok,
        "dns_ok": dns_ok,
        "proxy_detected": proxy,
        "recommendations": recs,
    }


def print_network_diagnostics(result: dict) -> None:
    """Print a formatted network diagnostic table to stdout."""
    print()
    print_subheader("Network Diagnostics")
    print()
    # summary table
    rows = []
    for name in _NET_CHECK_SOURCES:
        r = result.get(name, {})
        if r.get("reachable"):
            status = "\033[32m✓ Reachable\033[0m"
            detail = f"{r['latency_ms']}ms"
        else:
            status = "\033[31m✗ Unreachable\033[0m"
            detail = r.get("error_type", "?")
        rows.append([name, status, detail])

    rows.append(["DNS resolution", "\033[32m✓ OK\033[0m" if result.get("dns_ok") else "\033[31m✗ FAILED\033[0m", ""])
    rows.append(["HTTP proxy detected", f"\033[33m{result.get('proxy_detected')}\033[0m" if result.get("proxy_detected") else "\033[37mNone\033[0m", ""])

    print_table(["Source", "Status", "Detail"], rows)

    recs = result.get("recommendations", [])
    if recs:
        print()
        for r in recs:
            print(f"  \033[33m[INFO]\033[0m {r}")

    if not result.get("any_reachable"):
        print()
        print("  \033[33m[NOTE]\033[0m Transformers 模型需要联网下载。当前所有源不可达，"
              "Transformer 模型将自动跳过，传统 ML / DL 不受影响。")


# ---------------------------------------------------------------------------
# Environment Evaluation
# ---------------------------------------------------------------------------

def evaluate_environment(analysis: dict) -> dict:
    """
    Evaluate whether the environment meets minimum requirements for text classification.

    Checks:
      - Python version >= 3.8
      - RAM >= 4 GB
      - Free disk space >= 5 GB (traditional ML) / 20 GB (transformers)
      - GPU VRAM >= 8 GB recommended for transformers (not a hard blocker)

    Returns dict:
        can_proceed (bool), blockers (list of str), warnings (list of str),
        recommendations (list of str)
    """
    blockers = []
    warnings = []
    recommendations = []

    system = analysis.get("system", {})
    device = analysis.get("device", {})
    storage = analysis.get("storage", {})
    pyenv = analysis.get("python_env", {})
    dataset = analysis.get("dataset", {})

    # Python version check
    py_ver = system.get("python_version", "0.0")
    try:
        py_major, py_minor = map(int, py_ver.split(".")[:2])
        if py_major < 3 or (py_major == 3 and py_minor < 8):
            blockers.append(
                f"Python {py_ver} is too old. Minimum required: Python 3.8+."
            )
    except (ValueError, AttributeError):
        warnings.append("Could not determine Python version.")

    # RAM check
    ram_gb = device.get("total_ram_gb", 0)
    if ram_gb and ram_gb < 4:
        blockers.append(
            f"RAM ({ram_gb} GB) is below minimum (4 GB). "
            "Training may not be possible."
        )
    elif ram_gb and ram_gb < 8:
        warnings.append(
            f"RAM ({ram_gb} GB) is low. Transformer training may be very slow or fail. "
            "Traditional ML is still viable."
        )
        recommendations.append(
            "Use traditional ML models (Category A) — they work well with limited RAM."
        )

    # Free disk space check
    free_gb = storage.get("free_space_gb", 0)
    if free_gb and free_gb < 5:
        blockers.append(
            f"Free disk space ({free_gb} GB) is below minimum (5 GB). "
            "Cannot download models or save artifacts."
        )
    elif free_gb and free_gb < 20:
        warnings.append(
            f"Free disk space ({free_gb} GB) is limited. "
            "Transformer model downloads (~500MB-1.5GB each) may fill the disk. "
            "Prefer traditional ML or smaller transformer variants (DistilBERT)."
        )
        recommendations.append(
            "Use DistilBERT instead of BERT/RoBERTa base to reduce disk usage."
        )

    # GPU check (not a hard blocker, CPU fallback exists)
    if device.get("has_gpu"):
        vram = device.get("gpu_vram_gb", 0)
        if vram and vram < 4:
            warnings.append(
                f"GPU VRAM ({vram} GB) is low. "
                "Transformer fine-tuning with batch_size=1 may work, "
                "but larger models (BERT large) will run out of memory."
            )
            recommendations.append(
                "Use DistilBERT or traditional ML. Avoid BERT/RoBERTa large variants."
            )
        elif vram and vram < 8:
            warnings.append(
                f"GPU VRAM ({vram} GB) is moderate. "
                "BERT base fine-tuning is fine with small batch sizes. "
                "Avoid large model variants."
            )
    else:
        warnings.append(
            "No GPU detected. Deep learning and Transformer training will use CPU, "
            "which is significantly slower. Traditional ML is unaffected."
        )
        recommendations.append(
            "Traditional ML (Category A) is recommended for fast iteration. "
            "If using DL/Transformers, reduce epochs (--epochs 2) and use smaller models."
        )

    # Missing critical packages
    missing = pyenv.get("missing", [])
    critical_missing = [p for p in missing if p in (
        "numpy", "pandas", "scikit-learn", "torch", "transformers"
    )]
    if critical_missing:
        blockers.append(
            f"Critical packages missing: {', '.join(critical_missing)}. "
            "Install them before proceeding."
        )

    can_proceed = len(blockers) == 0

    return {
        "can_proceed": can_proceed,
        "blockers": blockers,
        "warnings": warnings,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# Package Installation
# ---------------------------------------------------------------------------

def install_missing_packages(missing_packages: list, yaml_path: str = None) -> dict:
    """
    Install missing Python packages via pip.

    Args:
        missing_packages: list of package names to install
        yaml_path: optional path to requirements.yaml for version specs

    Returns dict:
        success (bool), installed (list), failed (list of {package, error})
    """
    if not missing_packages:
        return {"success": True, "installed": [], "failed": []}

    # Build pip install specs with version constraints if YAML available
    version_specs = {}
    if yaml_path and os.path.exists(yaml_path):
        try:
            packages = _parse_requirements_yaml(yaml_path)
            version_specs = packages
        except Exception:
            pass

    specs = []
    for pkg in missing_packages:
        ver = version_specs.get(pkg, "")
        if ver:
            # Convert >=X.Y to pkg>=X.Y for pip
            specs.append(f"{pkg}{ver}" if ver.startswith((">=", ">", "<=", "<", "==", "!=")) else f"{pkg}>={ver}")
        else:
            specs.append(pkg)

    installed = []
    failed = []

    import subprocess
    for spec in specs:
        pkg_name = spec.split(">=")[0].split(">")[0].split("<=")[0].split("<")[0].split("==")[0].split("!=")[0].strip()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", spec, "--quiet"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                installed.append(pkg_name)
                print(f"    Installed: {pkg_name}")
            else:
                err_msg = result.stderr.strip().split("\n")[-1] if result.stderr else "Unknown error"
                failed.append({"package": pkg_name, "error": err_msg})
                print(f"    [FAILED] {pkg_name}: {err_msg}")
        except subprocess.TimeoutExpired:
            failed.append({"package": pkg_name, "error": "Timeout (5 min)"})
            print(f"    [FAILED] {pkg_name}: Timed out after 5 minutes")
        except Exception as e:
            failed.append({"package": pkg_name, "error": str(e)})
            print(f"    [FAILED] {pkg_name}: {e}")

    return {
        "success": len(failed) == 0,
        "installed": installed,
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Console Output Helpers
# ---------------------------------------------------------------------------

def print_header(title: str, width: int = 66) -> None:
    """Print a formatted main header (ASCII-safe)."""
    print(f"\n{'=' * width}", flush=True)
    print(f"  \033[1;32m{title}\033[0m", flush=True)
    print(f"{'=' * width}", flush=True)


def print_subheader(title: str, width: int = 66) -> None:
    """Print a formatted subsection header (ASCII-safe)."""
    print(f"\n  \033[1;36m{title}\033[0m", flush=True)
    print(f"  \033[90m{'-' * (width - 4)}\033[0m", flush=True)


def print_table(headers: list, rows: list[list], col_widths: list[int] = None,
                group_header: list = None) -> None:
    """Print a nicely formatted table with flat column headers.

    All column headers MUST be self-contained flat names (e.g.
    ``Train_Acc (N=3999)``).  ``group_header`` is intentionally unsupported —
    two-row grouped headers are forbidden (they break in Markdown and hide
    per-column sample counts).
    """
    if group_header is not None:
        raise ValueError(
            "group_header is no longer supported. "
            "Use flat column headers with dataset and sample count in each name, "
            "e.g. 'Train_Acc (N=3999)' instead of a grouped header row."
        )
    if not rows:
        return
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2
                      for i in range(len(headers))]

    def fmt_row(vals):
        return "|".join(str(v).ljust(w) for v, w in zip(vals, col_widths))

    sep = "+".join("-" * w for w in col_widths)

    print(f"  +{sep}+", flush=True)
    print(f"  |{fmt_row(headers)}|", flush=True)
    print(f"  +{sep}+", flush=True)
    for row in rows:
        print(f"  |{fmt_row(row)}|", flush=True)
    print(f"  +{sep}+", flush=True)


def print_metric(name: str, value, indent: int = 2, color: str = None) -> None:
    """Print a single metric line, optionally colored."""
    prefix = " " * indent
    val_str = str(value)
    if color:
        val_str = f"\033[{color}m{val_str}\033[0m"
    print(f"{prefix}\033[90m{name}\033[0m: {val_str}", flush=True)


def print_model_header(index: int, total: int, name: str, category: str = "") -> None:
    """Print a compact per-model header."""
    cat_color = {"traditional_ml": "33", "deep_learning": "35", "transformer": "34"}
    color = cat_color.get(category, "0")
    tag = {"traditional_ml": "Bow+ML", "deep_learning": "DL", "transformer": "TF"}
    t = tag.get(category, category)
    bar = "=" * 60
    print(f"\n\033[1m\033[{color}m{bar}\033[0m", flush=True)
    print(f"\033[1m [{index}/{total}] \033[{color}m{name}\033[0m "
          f"\033[90m[{t}]\033[0m", flush=True)
    print(f"\033[1m\033[{color}m{bar}\033[0m", flush=True)


def now_iso() -> str:
    """Return current time as ISO-8601 string."""
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Embedding path resolution
# ---------------------------------------------------------------------------

def resolve_embedding_path(embedding_path, emb_type: str,
                           auto_download: bool = True) -> str | None:
    """
    Resolve the file path for a given embedding type.

    Resolution order:
      1. User-provided path from CLI (dict of type->path or single string).
      2. Auto-download to local cache (~/.cache/text-binary-classification/).

    Returns the resolved file path, or None if unavailable.
    """
    # Step 1: user-provided path
    if embedding_path is not None:
        if isinstance(embedding_path, dict):
            p = embedding_path.get(emb_type)
        else:
            p = embedding_path
        if p and os.path.exists(p):
            return p
        if p:
            print(f"[WARN] Embedding file not found: {p}, "
                  f"falling back to auto-download")

    # Step 2: auto-download
    if auto_download:
        from preprocessing import ensure_embeddings
        return ensure_embeddings(emb_type)
    return None


def get_amp_config(device_str: str):
    """
    Return (autocast_ctx, scaler_or_none) for mixed-precision training.

    Strategy (device-aware):
      - CUDA Ampere+    (sm ≥ 8.0):  autocast(bfloat16), no scaler
             BF16 has the same exponent range as FP32 — no gradient
             underflow, so GradScaler is unnecessary.
      - CUDA Volta/Turing (sm 7.0–7.5): autocast(float16) + GradScaler
             FP16 needs loss scaling to preserve small gradients.
      - CUDA pre-Volta / CPU / MPS: nullcontext(), no scaler
             AMP not beneficial (or not supported).

    Returns
    -------
    autocast_ctx : context manager
        Call ``with autocast_ctx:`` before the forward pass.
    scaler : torch.cuda.amp.GradScaler or None
        If not None, call ``scaler.scale(loss).backward()`` and
        ``scaler.step(optimizer); scaler.update()``.
    """
    import contextlib

    if device_str != "cuda":
        return contextlib.nullcontext(), None

    try:
        import torch
        cap = torch.cuda.get_device_capability(0)
        sm = cap[0] + cap[1] / 10.0  # e.g. (8, 9) → 8.9
    except Exception:
        return contextlib.nullcontext(), None

    if sm >= 8.0:
        # Ampere / Ada / Hopper / Blackwell: BF16 safe, no scaler
        return torch.amp.autocast("cuda", dtype=torch.bfloat16), None
    elif sm >= 7.0:
        # Volta / Turing: FP16 + GradScaler
        scaler = torch.amp.GradScaler()
        return torch.amp.autocast("cuda", dtype=torch.float16), scaler
    else:
        # Maxwell / Pascal / Kepler: AMP not beneficial
        return contextlib.nullcontext(), None
