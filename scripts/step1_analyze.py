"""
Stage 1: Data Analysis
Load CSV, validate columns, compute comprehensive statistics, detect hardware.

Usage:
    python step1_analyze.py --csv <path> --text-col <name> --label-col <name>
                           [--output-dir <dir>] [--encoding <enc>]
Output:
    output/analysis.json
"""

import argparse
import sys
import os
import json
import pathlib

import pandas as pd

# Allow importing sibling modules from the skill scripts directory
_script_dir = pathlib.Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from utils import (
    read_csv_safe, save_json, print_header, print_subheader,
    detect_device, get_system_info, check_english,
    detect_language_distribution, detect_storage, check_python_env,
    ensure_output_dir, now_iso, setup_logging_and_warnings,
    run_network_diagnostics, print_network_diagnostics,
)
from preprocessing import (
    compute_text_stats, compute_vocabulary_richness,
    compute_syntactic_complexity,
)


def analyze_data(
    csv_path: str,
    text_col: str,
    label_col: str,
    output_dir: str = "output",
    encoding: str = None,
) -> dict:
    """
    Core analysis function. Returns comprehensive statistics dict.

    Steps:
      1. Load CSV with encoding fallback
      2. Validate text_col (string) and label_col (0/1 only)
      3. Compute sample stats (count, class distribution, class ratio)
      4. Compute text length stats (mean, median, percentiles)
      5. Compute vocab size, missing ratio, duplicate ratio
      6. Language detection (binary check + detailed per-language distribution)
      7. Vocabulary richness (TTR, hapax ratio, repeated word ratio)
      8. Syntactic complexity (sentence length, clause count heuristic)
      9. OS & hardware detection (OS, CPU cores/freq, RAM, GPU details + CUDA cores)
     10. Storage detection (free space, disk type SSD/HDD, r/w speed benchmark)
     11. Python environment check (validate against references/requirements.yaml)
     12. Save to output/analysis.json
    """
    out_dir = ensure_output_dir(output_dir)

    # --- Load CSV ---
    setup_logging_and_warnings()
    print_header("Stage 1: Data Analysis")
    print(f"Loading CSV: {csv_path}")
    df, used_encoding = read_csv_safe(csv_path, encoding=encoding)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns: {list(df.columns)}")

    # --- Validate columns ---
    if text_col not in df.columns:
        raise ValueError(
            f"Text column '{text_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    # --- Validate text column ---
    if not __import__("pandas").api.types.is_string_dtype(df[text_col]):
        print(f"  [WARN] text_col '{text_col}' is not string dtype; converting...")
        df[text_col] = df[text_col].astype(str)

    # --- Validate label column (with auto-conversion from string labels) ---
    temp = pd.to_numeric(df[label_col], errors='coerce')
    n_nan = int(temp.isna().sum())
    if n_nan > 0:
        raise ValueError(
            f"Label column '{label_col}' contains {n_nan} NaN/None values. "
            "Please remove or fix these rows."
        )
    if not temp.dropna().apply(lambda x: float(x).is_integer()).all():
        raise ValueError(
            f"Label column '{label_col}' contains non-integer values. "
            "Labels must be exactly 0 or 1."
        )
    df[label_col] = temp.fillna(-1).astype(int)
    unique_labels = sorted(df[label_col].dropna().unique())
    if set(unique_labels) != {0, 1}:
        raise ValueError(
            f"Label column '{label_col}' must contain only 0 and 1. "
            f"Found unique values: {unique_labels}"
        )
    print(f"  Label column '{label_col}' validated: 0/1 only.")

    # --- Sample statistics ---
    print_subheader("Sample Statistics")
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    n_total = len(texts)
    # Early check: empty dataset
    if n_total == 0:
        raise ValueError(
            f"CSV file '{csv_path}' contains no data rows after loading. "
            "Cannot proceed with analysis on an empty dataset."
        )
    n_class_0 = sum(1 for l in labels if l == 0)
    n_class_1 = sum(1 for l in labels if l == 1)
    class_ratio = n_class_0 / max(n_class_1, 1)

    # Handle missing labels
    n_missing_label = sum(1 for l in labels if l is None or (
        isinstance(l, float) and __import__("numpy").isnan(l)))

    print(f"  Total samples: {n_total}")
    print(f"  Class 0: {n_class_0} ({n_class_0 / max(n_total, 1) * 100:.1f}%)")
    print(f"  Class 1: {n_class_1} ({n_class_1 / max(n_total, 1) * 100:.1f}%)")
    print(f"  Class ratio (0:1): {class_ratio:.2f}:1")
    if n_missing_label:
        print(f"  [WARN] Missing labels: {n_missing_label}")

    # Imbalance warning
    min_class_pct = min(n_class_0, n_class_1) / max(n_total, 1)
    if min_class_pct < 0.05:
        print(f"  [WARN] Minority class is only {min_class_pct:.1%} of data. Class imbalance detected.")

    # --- Text statistics ---
    print_subheader("Text Statistics")
    text_stats = compute_text_stats(texts)
    for key, val in text_stats.items():
        print(f"  {key}: {val}")

    # --- Language detection ---
    print_subheader("Language Detection")
    lang_result = check_english(texts)
    print(f"  Method: {lang_result['method']}")
    print(f"  Is English: {lang_result['is_english']}")
    print(f"  Non-English ratio: {lang_result['non_english_ratio']}")
    print(f"  Details: {lang_result['details']}")
    if not lang_result["is_english"]:
        print(f"  [WARN] Data may contain significant non-English text. "
              f"Consider filtering or confirming language.")

    # --- Detailed language distribution ---
    print_subheader("Language Distribution")
    lang_dist = detect_language_distribution(texts)
    print(f"  Method: {lang_dist['method']}")
    print(f"  Sampled: {lang_dist.get('sampled', 'N/A')} | "
          f"Detected: {lang_dist.get('detected', 'N/A')} | "
          f"Skipped (too short): {lang_dist.get('skipped', 'N/A')}")
    languages = lang_dist.get("languages", {})
    if languages:
        print(f"  Language breakdown:")
        for lang_code, info in languages.items():
            pct = info["ratio"] * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {lang_code:>8s}: {info['count']:>5d} ({pct:5.1f}%) {bar}")
    else:
        print(f"  (no language data available)")

    # --- Vocabulary richness ---
    print_subheader("Vocabulary Richness")
    vocab_rich = compute_vocabulary_richness(texts)
    print(f"  Vocab size:        {vocab_rich['vocab_size']:>8,}")
    print(f"  Total tokens:      {vocab_rich['total_tokens']:>8,}")
    print(f"  Type-Token Ratio:  {vocab_rich['type_token_ratio']:>8.4f}")
    print(f"  Hapax Legomena:    {vocab_rich['hapax_legomena_ratio']:>8.4f} "
          f"(ratio of words appearing only once)")
    print(f"  Repeated Word:     {vocab_rich['repeated_word_ratio']:>8.4f} "
          f"(ratio of repeated tokens)")
    # Interpretation hints
    ttr = vocab_rich["type_token_ratio"]
    if ttr < 0.10:
        print(f"  [NOTE] Low TTR ({ttr:.4f}) — highly repetitive vocabulary, narrow domain likely.")
    elif ttr > 0.40:
        print(f"  [NOTE] High TTR ({ttr:.4f}) — very diverse vocabulary, broad-domain or noisy text.")
    hlr = vocab_rich["hapax_legomena_ratio"]
    if hlr > 0.50:
        print(f"  [NOTE] High hapax ratio ({hlr:.4f}) — many rare words, may indicate noisy/sparse data.")

    # --- Syntactic complexity ---
    print_subheader("Syntactic Complexity")
    syn_cpx = compute_syntactic_complexity(texts)
    print(f"  Avg sentence length:      {syn_cpx['avg_sentence_length']:>6.1f} words")
    print(f"  Avg sentences per text:   {syn_cpx['avg_sentences_per_text']:>6.1f}")
    print(f"  Avg clauses per sentence: {syn_cpx['avg_clauses_per_sentence']:>6.1f}")
    if syn_cpx['avg_sentence_length'] < 8:
        print(f"  [NOTE] Very short sentences — may be headlines, tweets, or fragmented text.")
    elif syn_cpx['avg_sentence_length'] > 35:
        print(f"  [NOTE] Very long sentences — may be academic/legal text or run-on sentences.")
    if syn_cpx['avg_clauses_per_sentence'] < 1.2:
        print(f"  [NOTE] Low clause count — text may be simple or informal (social media, reviews).")

    # --- Device & system info ---
    print_subheader("Environment: OS & Hardware")
    device_info = detect_device()
    system_info = get_system_info()

    # OS
    os_name = system_info.get("platform_system", "Unknown")
    print(f"  OS:       {system_info['platform']}")

    # CPU
    print(f"  CPU:      {device_info['cpu_cores_physical']} physical cores / "
          f"{device_info['cpu_cores_logical']} logical cores"
          + (f", {device_info['cpu_freq_mhz']:.0f} MHz" if device_info.get("cpu_freq_mhz") else ""))

    # RAM
    print(f"  RAM:      {device_info['total_ram_gb']} GB")

    # GPU
    if device_info["has_gpu"]:
        gpu_line = f"  GPU:      {device_info['gpu_name']} x{device_info['gpu_count']}"
        if device_info.get("gpu_vram_gb"):
            gpu_line += f" ({device_info['gpu_vram_gb']} GB VRAM)"
        if device_info.get("gpu_cuda_cores"):
            gpu_line += f", {device_info['gpu_cuda_cores']} CUDA cores"
        gpu_line += f", CUDA: Yes"
        print(gpu_line)
    else:
        print(f"  GPU:      None (CPU-only mode)")

    # --- Storage detection ---
    print_subheader("Environment: Storage")
    storage_info = detect_storage(output_dir)
    print(f"  Free space:   {storage_info['free_space_gb']} GB / "
          f"{storage_info['total_space_gb']} GB")
    if storage_info.get("disk_type") and storage_info["disk_type"] != "unknown":
        print(f"  Disk type:    {storage_info['disk_type']}")
    if storage_info.get("read_speed_mbs"):
        print(f"  Read speed:   {storage_info['read_speed_mbs']} MB/s")
    if storage_info.get("write_speed_mbs"):
        print(f"  Write speed:  {storage_info['write_speed_mbs']} MB/s")
    if storage_info.get("free_space_gb") is not None and storage_info["free_space_gb"] < 10:
        print(f"  [WARN] Less than 10 GB free disk space. Large model downloads may fail.")

    # --- Python environment check ---
    print_subheader("Environment: Python Dependencies")
    yaml_path = pathlib.Path(_script_dir).parent / "references" / "requirements.yaml"
    if yaml_path.exists():
        pyenv_info = check_python_env(str(yaml_path))
        if pyenv_info.get("error"):
            print(f"  [WARN] {pyenv_info['error']}")
        else:
            print(f"  Required packages:   {pyenv_info['total_required']}")
            print(f"  Installed (matched): {pyenv_info['total_installed']}")
            print(f"  Missing:             {pyenv_info['total_missing']}")
            print(f"  Outdated (version):  {pyenv_info['total_outdated']}")
            if pyenv_info["missing"]:
                print(f"  [WARN] Missing packages: {', '.join(pyenv_info['missing'])}")
            if pyenv_info["outdated"]:
                for pkg, ver_info in pyenv_info["outdated"].items():
                    print(f"  [WARN] {pkg}: required {ver_info['required']}, "
                          f"installed {ver_info['installed']}")
            if pyenv_info["ok"]:
                print(f"  All required packages are installed and up-to-date.")
    else:
        pyenv_info = {"error": f"requirements.yaml not found at {yaml_path}"}
        print(f"  [WARN] {pyenv_info['error']}")
        pyenv_info = {"ok": False, "installed": {}, "missing": [], "outdated": {},
                      "error": pyenv_info["error"]}

    # --- Network diagnostics (pre-check download sources) ---
    network_info = run_network_diagnostics()
    print_network_diagnostics(network_info)

    # --- Assemble result ---
    analysis = {
        "meta": {
            "csv_path": str(pathlib.Path(csv_path).resolve()),
            "text_col": text_col,
            "label_col": label_col,
            "encoding": used_encoding,
            "analyzed_at": now_iso(),
        },
        "dataset": {
            "total_samples": n_total,
            "class_0_count": n_class_0,
            "class_1_count": n_class_1,
            "class_ratio": round(class_ratio, 2),
            "missing_labels": n_missing_label,
            "is_english": lang_result["is_english"],
            "non_english_ratio": lang_result["non_english_ratio"],
            "language_method": lang_result["method"],
            "language_distribution": lang_dist,
            "text_stats": text_stats,
            "vocabulary_richness": vocab_rich,
            "syntactic_complexity": syn_cpx,
        },
        "device": device_info,
        "system": system_info,
        "storage": storage_info,
        "python_env": pyenv_info,
        "network": network_info,
    }

    # --- Save ---
    output_path = out_dir / "analysis.json"
    save_json(analysis, str(output_path))
    print(f"\n  Analysis saved to: {output_path}")

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Data Analysis for English Binary Text Classification"
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--text-col", required=True, help="Name of text column")
    parser.add_argument("--label-col", required=True, help="Name of label column (0/1)")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--encoding", default=None,
                        help="CSV encoding (auto-detect if not specified)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] File not found: {args.csv}")
        sys.exit(1)

    try:
        analyze_data(
            csv_path=args.csv,
            text_col=args.text_col,
            label_col=args.label_col,
            output_dir=args.output_dir,
            encoding=args.encoding,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] Data validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
