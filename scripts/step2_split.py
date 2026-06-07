"""
Stage 2: Train/Validation/Test Split

Usage:
    python step2_split.py --csv <path> --text-col <name> --label-col <name>
                          --split-type <random_2way|column_2way|random_3way|column_3way>
                          [--train-ratio 0.8] [--valid-ratio 0.1] [--test-ratio 0.1]
                          [--split-column <name>]
                          [--train-value <val>] [--valid-value <val>] [--test-value <val>]
                          [--output-dir <dir>] [--encoding <enc>] [--seed 42]

Output:
    <output_dir>/split_info.json
"""

import argparse
import sys
import os
import pathlib
import warnings

import numpy as np
import pandas as pd

_script_dir = pathlib.Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from utils import (
    read_csv_safe, save_json, print_header, print_subheader,
    print_table, set_seed, ensure_output_dir, now_iso,
)
from sklearn.model_selection import train_test_split

# Targeted warning suppression for sklearn convergence/deprecation warnings
# that are expected during parameter search (not in split stage, but set early).
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings(
    "ignore", message=".*failed to converge.*", category=UserWarning
)


def load_csv_for_split(csv_path: str, text_col: str, label_col: str,
                       encoding: str = None) -> pd.DataFrame:
    """Load and validate CSV. Returns DataFrame with integer index."""
    print_header("Stage 2: Data Split")
    print(f"  Loading: {csv_path}")

    df, _ = read_csv_safe(csv_path, encoding=encoding)

    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in CSV.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV.")

    # Validate label column (0/1 only, no NaN allowed)
    # Auto-convert string labels "0"/"1" to int 0/1
    temp = pd.to_numeric(df[label_col], errors='coerce')
    n_nan = int(temp.isna().sum())
    if n_nan > 0:
        raise ValueError(
            f"Label column '{label_col}' contains {n_nan} NaN/None values. "
            f"Please remove or fix these rows before splitting."
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
            f"Found: {unique_labels}"
        )

    # Ensure clean integer index for index-based splitting
    df = df.reset_index(drop=True)

    print(f"  {len(df)} samples  |  "
          f"class 0: {(df[label_col]==0).sum():,}  "
          f"class 1: {(df[label_col]==1).sum():,}")
    return df


def perform_split(df: pd.DataFrame, label_col: str, split_type: str,
                  **kwargs) -> dict:
    """
    Execute data split and return split_info dict.

    split_type:
      - "random_2way": random train/test split
      - "column_2way": split based on a CSV column value
      - "random_3way": random train/valid/test split
      - "column_3way": split based on a CSV column value (3-way)
    """
    seed = kwargs.get("random_state", 42)
    n_total = len(df)
    labels = df[label_col].values

    train_idx, valid_idx, test_idx = [], [], []

    if split_type == "random_2way":
        test_ratio = kwargs.get("test_ratio", 0.2)
        try:
            train_df, test_df = train_test_split(
                df, test_size=test_ratio, stratify=labels,
                random_state=seed,
            )
        except ValueError:
            warnings.warn(
                "Stratified split failed (likely a class has too few samples). "
                "Falling back to non-stratified random split."
            )
            train_df, test_df = train_test_split(
                df, test_size=test_ratio,
                random_state=seed,
            )
        train_idx = sorted(train_df.index.tolist())
        test_idx = sorted(test_df.index.tolist())

    elif split_type == "column_2way":
        split_col = kwargs["split_column"]
        train_val = kwargs["train_value"]
        test_val = kwargs["test_value"]
        train_idx = sorted(df[df[split_col] == train_val].index.tolist())
        test_idx = sorted(df[df[split_col] == test_val].index.tolist())
        overlap = set(train_idx) & set(test_idx)
        if overlap:
            raise ValueError(
                f"Overlap detected between train and test sets: {len(overlap)} rows"
            )
        # Check for rows dropped because they don't match any split value
        assigned = len(train_idx) + len(test_idx)
        if assigned < len(df):
            warnings.warn(
                f"Column split assigned {assigned}/{len(df)} rows. "
                f"{len(df) - assigned} rows do not match "
                f"'{train_val}' or '{test_val}' in column '{split_col}' "
                f"and will be excluded."
            )

    elif split_type == "random_3way":
        test_ratio = kwargs.get("test_ratio", 0.1)
        valid_ratio = kwargs.get("valid_ratio", 0.1)
        # Step 1: hold out test
        try:
            df_temp, test_df = train_test_split(
                df, test_size=test_ratio, stratify=labels,
                random_state=seed,
            )
        except ValueError:
            warnings.warn(
                "Stratified test split failed (likely a class has too few samples). "
                "Falling back to non-stratified random split."
            )
            df_temp, test_df = train_test_split(
                df, test_size=test_ratio,
                random_state=seed,
            )
        # Step 2: split remaining into train/valid
        adjusted_valid = valid_ratio / (1 - test_ratio)
        try:
            train_df, valid_df = train_test_split(
                df_temp, test_size=adjusted_valid,
                stratify=df_temp[label_col].values,
                random_state=seed + 1,
            )
        except ValueError:
            warnings.warn(
                "Stratified valid split failed (likely a class has too few samples). "
                "Falling back to non-stratified random split."
            )
            train_df, valid_df = train_test_split(
                df_temp, test_size=adjusted_valid,
                random_state=seed + 1,
            )
        train_idx = sorted(train_df.index.tolist())
        valid_idx = sorted(valid_df.index.tolist())
        test_idx = sorted(test_df.index.tolist())

    elif split_type == "column_3way":
        split_col = kwargs["split_column"]
        train_val = kwargs["train_value"]
        valid_val = kwargs["valid_value"]
        test_val = kwargs["test_value"]
        train_idx = sorted(df[df[split_col] == train_val].index.tolist())
        valid_idx = sorted(df[df[split_col] == valid_val].index.tolist())
        test_idx = sorted(df[df[split_col] == test_val].index.tolist())
        # Check for overlaps
        _splits = {"train": train_idx, "valid": valid_idx, "test": test_idx}
        for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
            overlap = set(_splits[a]) & set(_splits[b])
            if overlap:
                raise ValueError(
                    f"Overlap between {a} and {b} sets: {len(overlap)} rows"
                )
        # Check for rows dropped because they don't match any split value
        assigned = len(train_idx) + len(valid_idx) + len(test_idx)
        if assigned < len(df):
            warnings.warn(
                f"Column split assigned {assigned}/{len(df)} rows. "
                f"{len(df) - assigned} rows do not match "
                f"'{train_val}', '{valid_val}', or '{test_val}' in column '{split_col}' "
                f"and will be excluded."
            )

    else:
        raise ValueError(
            f"Unknown split_type '{split_type}'. "
            f"Use: random_2way | column_2way | random_3way | column_3way"
        )

    n_train = len(train_idx)
    n_valid = len(valid_idx)
    n_test = len(test_idx)
    if n_train <= 0:
        raise ValueError(
            f"Train set is empty (0 rows). Split_type='{split_type}' "
            "produced no training samples. Check split parameters."
        )
    if n_test <= 0:
        raise ValueError(
            f"Test set is empty (0 rows). Split_type='{split_type}' "
            "produced no test samples. Check split parameters."
        )
    if "3way" in split_type and n_valid <= 0:
        raise ValueError(
            f"Valid set is empty (0 rows). Split_type='{split_type}' "
            "requires a non-empty validation set. Check split parameters."
        )

    # Label distribution per split
    def _class_dist(indices):
        subset_labels = df[label_col].iloc[indices]
        c0 = int((subset_labels == 0).sum())
        c1 = int((subset_labels == 1).sum())
        return c0, c1

    info = {
        "meta": {
            "csv_path": str(pathlib.Path(kwargs.get("csv_path", "")).resolve()),
            "text_col": kwargs.get("text_col", ""),
            "label_col": label_col,
            "split_type": split_type,
            "random_state": seed,
            "total_samples": n_total,
            "created_at": now_iso(),
        },
        "splits": {
            "train": train_idx,
            "valid": valid_idx if valid_idx else None,
            "test": test_idx,
        },
        "counts": {
            "train": n_train,
            "valid": n_valid if valid_idx else None,
            "test": n_test,
        },
        "class_distribution": {
            "train": {"class_0": _class_dist(train_idx)[0],
                       "class_1": _class_dist(train_idx)[1]},
            "valid": {"class_0": _class_dist(valid_idx)[0],
                       "class_1": _class_dist(valid_idx)[1]} if valid_idx else None,
            "test": {"class_0": _class_dist(test_idx)[0],
                      "class_1": _class_dist(test_idx)[1]},
        },
    }
    return info


def print_split_summary(info: dict) -> None:
    """Print a formatted summary of the split."""
    print_subheader("Split Summary")
    print(f"  Method: {info['meta']['split_type']}")
    print(f"  Total samples: {info['meta']['total_samples']:,}")

    rows = []
    for subset in ("train", "valid", "test"):
        counts = info["counts"].get(subset)
        if counts is None:
            continue
        dist = info["class_distribution"][subset]
        pct = counts / info["meta"]["total_samples"] * 100
        rows.append([
            subset.upper(),
            f"{counts:,}",
            f"{pct:.1f}%",
            f"{dist['class_0']:,}",
            f"{dist['class_1']:,}",
        ])

    print_table(
        ["Set", "Count", "Ratio", "Class 0", "Class 1"],
        rows,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Train/Validation/Test Split"
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--text-col", required=True, help="Text column name")
    parser.add_argument("--label-col", required=True, help="Label column name (0/1)")
    parser.add_argument("--split-type", required=True,
                        choices=["random_2way", "column_2way",
                                 "random_3way", "column_3way"],
                        help="Split method")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train ratio (for random splits)")
    parser.add_argument("--valid-ratio", type=float, default=0.1,
                        help="Valid ratio (for random_3way, default 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Test ratio (for random_3way, default 0.1; for random_2way, use 0.2)")
    parser.add_argument("--split-column", default=None,
                        help="CSV column containing split labels")
    parser.add_argument("--train-value", default="train",
                        help="Value in split-column for train set")
    parser.add_argument("--valid-value", default="valid",
                        help="Value in split-column for valid set")
    parser.add_argument("--test-value", default="test",
                        help="Value in split-column for test set")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--encoding", default=None, help="CSV encoding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] File not found: {args.csv}")
        sys.exit(1)

    # Validate ratios for random splits
    if args.split_type in ("random_2way", "random_3way"):
        if args.split_type == "random_2way":
            t = args.test_ratio
            if not (0.05 <= t <= 0.5):
                print(f"[ERROR] test_ratio must be 0.05-0.5, got {t}")
                sys.exit(1)
            tr = 1.0 - t
        else:
            tr = args.train_ratio
            v = args.valid_ratio
            t = args.test_ratio
            total = tr + v + t
            if abs(total - 1.0) > 0.01:
                print(f"[ERROR] Ratios must sum to 1.0. "
                      f"Got train={tr} + valid={v} + test={t} = {total}")
                sys.exit(1)

    # Required for column-based splits
    if args.split_type in ("column_2way", "column_3way"):
        if not args.split_column:
            print(f"[ERROR] --split-column is required for {args.split_type}")
            sys.exit(1)

    set_seed(args.seed)

    try:
        df = load_csv_for_split(
            args.csv, args.text_col, args.label_col, args.encoding,
        )

        kwargs = {
            "csv_path": args.csv,
            "text_col": args.text_col,
            "random_state": args.seed,
        }
        if args.split_type in ("random_2way", "random_3way"):
            kwargs["test_ratio"] = args.test_ratio
            if args.split_type == "random_3way":
                kwargs["valid_ratio"] = args.valid_ratio
        if args.split_type in ("column_2way", "column_3way"):
            kwargs["split_column"] = args.split_column
            kwargs["train_value"] = args.train_value
            kwargs["test_value"] = args.test_value
            if args.split_type == "column_3way":
                kwargs["valid_value"] = args.valid_value

        info = perform_split(df, args.label_col, args.split_type, **kwargs)
        print_split_summary(info)

        # Save
        out_dir = ensure_output_dir(args.output_dir)
        out_path = out_dir / "split_info.json"
        save_json(info, str(out_path))
        print(f"\n  Split info saved to: {out_path}")

        return info

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
