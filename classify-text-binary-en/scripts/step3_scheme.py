"""
Stage 3: Model Scheme Generation
Based on dataset analysis, recommends model approaches with priorities.

Usage:
    python step3_scheme.py --analysis <path/to/analysis.json>
                           [--output-dir <dir>]
Output:
    output/model_scheme.json

Note: Web search for hyperparameter recommendations is performed by Claude
(the orchestrating LLM) before calling this script; results are passed via
the --web-results argument or loaded from model_params.md reference.
"""

import argparse
import sys
import os
import pathlib

_script_dir = pathlib.Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from utils import (
    load_json, save_json, print_header, print_subheader,
    print_table, ensure_output_dir,
)
from model_factory import (
    ModelSpec, get_default_specs,
    get_specs_by_category,
)


def _pick_best_baseline(analysis: dict) -> list:
    """
    Pick mandatory baseline model(s) based on dataset characteristics.
    Returns a list of (model_name, vectorizer_key) tuples.

    Per Rule 2 of the skill, SVM + LR are always mandatory baselines
    (except tiny datasets where only MNB is included).

    Heuristics:
      1. Tiny dataset (<1K)      → MultinomialNB + Count (only one needed)
      2. Small+ (>=1K)           → LogisticRegression + LinearSVC, both with
                                    TF-IDF (1,2-gram if mean text > 200)

    Both baselines are fast, reliable, and essential reference points
    before trying more expensive approaches.
    """
    dataset = analysis.get("dataset", {})
    n_samples = dataset.get("total_samples", 0)
    text_stats = dataset.get("text_stats", {})
    mean_len = text_stats.get("mean_length", 0)

    vec = "tfidf_bigram" if mean_len > 200 else "tfidf"

    if n_samples < 1000:
        return [("multinomial_nb", "count")]

    return [
        ("logistic_regression", vec),
        ("svm_linear", vec),
    ]


def _collapse_to_single_baseline(
    specs: list,
    baseline_models: list,
) -> list:
    """
    Demote all P1 specs to P2, then promote only the chosen baseline models
    (matched by name + vectorizer) to P1.

    baseline_models: list of (model_name, vectorizer_key) tuples.
    """
    import warnings

    # Demote all existing P1 -> P2
    for s in specs:
        if s.priority == 1:
            s.priority = 2

    # Promote each mandatory baseline to P1
    for model_name, vectorizer in baseline_models:
        found = False
        for s in specs:
            if s.name == model_name and s.vectorizer == vectorizer:
                s.priority = 1
                found = True
                break

        if not found:
            # Fallback: match only by name
            warnings.warn(
                f"No exact match for baseline '{model_name}/{vectorizer}'. "
                f"Falling back to name-only match."
            )
            for s in specs:
                if s.name == model_name:
                    s.priority = 1
                    found = True
                    break

    return specs


# ---------------------------------------------------------------------------
# Dynamic model scoring — replaces hardcoded curation with data + hardware
# aware continuous scoring.  Each model gets a suitability score; top-N
# become P2 (recommended), the rest P3 (exploratory).
# ---------------------------------------------------------------------------

# Architecture groups for diversity enforcement: each group should have at
# least one representative in the P2 set regardless of raw score.
_ARCH_GROUPS = [
    ("linear_sparse",   lambda s: s.category == "traditional_ml" and s.use_embedding == "none" and s.name in ("svm_linear", "logistic_regression")),
    ("tree_sparse",     lambda s: s.category == "traditional_ml" and s.use_embedding == "none" and s.name == "random_forest"),
    ("nb_sparse",       lambda s: s.category == "traditional_ml" and s.use_embedding == "none" and s.name == "multinomial_nb"),
    ("rbf_dense",       lambda s: s.category == "traditional_ml" and s.use_embedding != "none" and s.name == "svm_rbf"),
    ("linear_dense",    lambda s: s.category == "traditional_ml" and s.use_embedding != "none" and s.name in ("svm_linear", "logistic_regression")),
    ("cnn",             lambda s: s.category == "deep_learning" and s.name == "textcnn"),
    ("rnn",             lambda s: s.category == "deep_learning" and s.name in ("lstm", "bilstm", "gru", "bigru")),
    ("rnn_attention",   lambda s: s.category == "deep_learning" and s.name in ("lstm_attention", "gru_attention")),
    # Group transformers by size, not training mode — so BERT, RoBERTa,
    # DeBERTa etc. compete within the same group and the top few win.
    ("tf_base",         lambda s: s.category == "transformer" and "large" not in s.name),
    ("tf_large",        lambda s: s.category == "transformer" and "large" in s.name),
]

# Intrinsic model capability scores (0-10), sourced from 2024-2025 benchmarks.
# These reflect expected peak performance independent of dataset fit.
_TF_CAPABILITY = {
    "deberta-v3-base": 9.0, "deberta-v3-large": 9.5,
    "roberta-base": 8.8, "roberta-large": 9.3,
    "bert-base-uncased": 8.5, "bert-large-uncased": 9.0,
    "deberta-base": 8.3, "deberta-large": 8.8,
    "electra-base-discriminator": 8.0, "electra-large-discriminator": 8.5,
    "xlnet-base-cased": 7.8, "xlnet-large-cased": 8.3,
    "distilbert-base-uncased": 7.5,
    "albert-base-v2": 7.0, "albert-large-v2": 7.5,
    "electra-small-discriminator": 5.5,
}
_TF_MODE_DELTA = {"full_ft": 0, "partial_ft": -0.8, "peft": -1.2, "feature_extraction": -2.5}

_ML_CAPABILITY = {
    "svm_linear": 7.0, "svm_rbf": 6.0, "logistic_regression": 6.5,
    "random_forest": 5.5, "multinomial_nb": 4.5,
}
_VEC_DELTA = {"tfidf_bigram": 1.0, "tfidf": 0.5, "count_bigram": 0.5,
              "count": 0, "onehot": -1.0, "embedding": 1.5}

_DL_CAPABILITY = {
    "lstm_attention": 7.5, "gru_attention": 7.3,
    "bilstm": 7.0, "stacked_lstm": 7.0,
    "bigru": 6.8, "stacked_gru": 6.8,
    "textcnn": 6.5, "lstm": 6.5, "gru": 6.3,
}
_EMB_DELTA = {"glove": 0.5, "fasttext": 0.5, "word2vec": 0}


def _base_capability(spec: ModelSpec) -> float:
    """Intrinsic model quality (0-10), independent of dataset."""
    if spec.category == "transformer":
        return _TF_CAPABILITY.get(spec.name, 6.0) + _TF_MODE_DELTA.get(spec.training_mode, 0)
    if spec.category == "traditional_ml":
        return _ML_CAPABILITY.get(spec.name, 5.0) + _VEC_DELTA.get(spec.vectorizer, 0)
    # deep_learning
    ft_bonus = 0.0 if spec.freeze_embeddings else 1.0
    return _DL_CAPABILITY.get(spec.name, 6.0) + _EMB_DELTA.get(spec.use_embedding, 0) + ft_bonus


def _score_and_assign_priorities(specs: list, analysis: dict) -> list:
    """
    Score every non-baseline model on data fit + hardware fit + capability,
    then assign P2 (top-N) / P3 (rest).  Architecture diversity is enforced
    so every major group has at least one representative in P2.
    """
    dataset = analysis.get("dataset", {})
    device = analysis.get("device", {})
    text_stats = dataset.get("text_stats", {})
    richness = dataset.get("vocabulary_richness", {})

    n_samples = dataset.get("total_samples", 0)
    mean_len = text_stats.get("mean_length", 0)
    ttr = richness.get("type_token_ratio", 0)
    hapax = richness.get("hapax_legomena_ratio", 0)
    non_en_ratio = dataset.get("non_english_ratio", 0)
    class_0 = dataset.get("class_0_count", 0)
    class_1 = dataset.get("class_1_count", 0)
    minority_pct = min(class_0, class_1) / max(n_samples, 1)
    is_imbalanced = minority_pct < 0.20
    has_gpu = device.get("has_gpu", False)
    gpu_vram = device.get("gpu_vram_gb") or 0
    # MPS (Apple Silicon) shares unified memory — VRAM reports None but
    # most Apple GPUs have at least 8 GB available for compute.
    if gpu_vram == 0 and device.get("gpu_type") == "mps":
        gpu_vram = 8.0

    # ------------------------------------------------------------------
    # Phase 1: hardware filter — eliminate impossible models
    # ------------------------------------------------------------------
    for s in specs:
        if s.priority == 1:
            continue
        if s.category in ("deep_learning", "transformer") and not has_gpu:
            s.priority = 3
        elif s.category == "transformer":
            if gpu_vram < 2:
                s.priority = 3
            elif "large" in s.name and gpu_vram < 12:
                s.priority = 3
            elif s.training_mode == "full_ft" and gpu_vram < 4 and "distilbert" not in s.name and "albert" not in s.name and "electra-small" not in s.name:
                s.priority = 3  # full_ft needs 4+ GB for most models

    # ------------------------------------------------------------------
    # Phase 2: score every viable candidate
    # ------------------------------------------------------------------
    scored = []
    for s in specs:
        if s.priority in (1, 3):  # baseline or already filtered
            continue

        score = _base_capability(s)

        # --- Data fit adjustments (-3 to +3) ---

        # Sample size: small → trad-ML, large → transformer
        if n_samples < 1000:
            if s.category == "traditional_ml":
                score += 2.0
            elif s.category == "transformer":
                score -= 3.0
            elif s.category == "deep_learning":
                score -= 1.5
        elif n_samples < 5000:
            if s.category == "traditional_ml":
                score += 1.0
            elif s.category == "transformer":
                score -= 0.3  # viable with GPU, just slightly riskier
        elif n_samples > 50000:
            if s.category == "transformer":
                score += 1.5
                if "large" in s.name:
                    score += 0.5
            elif s.category == "traditional_ml":
                score -= 0.5  # trad-ML doesn't scale as well

        # Text length: short → CNN/MNB, long → RNN+Attn/XLNet
        if mean_len < 50:
            if s.name == "textcnn":
                score += 1.0
            if s.name == "multinomial_nb":
                score += 0.5
            if s.name in ("bilstm", "lstm_attention", "gru_attention") and not s.freeze_embeddings:
                score -= 0.5  # RNN overkill for short text
        elif mean_len > 150:
            if s.name in ("lstm_attention", "gru_attention"):
                score += 1.0
            if "xlnet" in s.name:
                score += 1.0
            if s.name == "textcnn":
                score -= 0.3

        # Vocabulary diversity: low TTR → sparse TF-IDF sufficient
        if ttr < 0.05:  # very repetitive vocabulary
            if s.category == "traditional_ml" and s.use_embedding == "none" and "tfidf" in s.vectorizer:
                score += 0.5  # sparse TF-IDF handles low-diversity text efficiently
            if s.use_embedding != "none" and s.category != "transformer":
                score -= 0.5  # dense embeddings less needed
        elif ttr > 0.5:  # very diverse vocabulary
            if s.use_embedding == "fasttext":
                score += 1.0  # subword coverage
            if s.category == "transformer":
                score += 0.5

        # Hapax ratio: many rare words → FastText subword advantage
        if hapax > 0.30:
            if s.use_embedding == "fasttext":
                score += 0.5  # subword coverage helps with rare/OOV words
            if s.category == "transformer":
                if s.training_mode in ("feature_extraction", "partial_ft"):
                    score += 0.3  # less risk of overfitting to hapax noise
            if s.category == "deep_learning" and s.freeze_embeddings:
                score += 0.3

        # Class imbalance: tree models + class_weight
        if is_imbalanced:
            if s.name == "random_forest":
                score += 1.0
            if s.category == "traditional_ml":
                score += 0.3  # class_weight='balanced' is set by _adjust_priorities

        # Non-English content: FastText subword advantage
        if non_en_ratio > 0.10:
            if s.use_embedding == "fasttext":
                score += 1.5

        # --- Hardware fit (-2 to +1) ---
        if s.category == "transformer":
            if gpu_vram >= 12:
                score += 0.5  # large models viable
            elif gpu_vram < 6:
                if "distilbert" in s.name or "albert" in s.name:
                    score += 1.0
                if s.training_mode == "peft":
                    score += 0.5
                if "large" in s.name:
                    score -= 2.0
        elif s.category == "deep_learning":
            if gpu_vram < 4 and not s.freeze_embeddings:
                score -= 1.0  # fine-tuning needs VRAM

        scored.append((s, score))

    # ------------------------------------------------------------------
    # Phase 3: diversity enforcement — each arch group guaranteed ≥1 rep
    # ------------------------------------------------------------------
    group_best: dict[str, tuple] = {}
    group_members: dict[str, list] = {}
    for s, score in scored:
        for gname, gfn in _ARCH_GROUPS:
            if gfn(s):
                group_members.setdefault(gname, []).append((s, score))
                if gname not in group_best or score > group_best[gname][1]:
                    group_best[gname] = (s, score)

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Cap same-model variants: keep top-2 training modes per transformer model
    _tf_model_variants: dict[str, list] = {}
    for s, sc in scored:
        if s.category == "transformer":
            _tf_model_variants.setdefault(s.name, []).append((s, sc))
    for _name, variants in _tf_model_variants.items():
        if len(variants) > 2:
            variants.sort(key=lambda x: x[1], reverse=True)
            for s, _sc in variants[2:]:
                s.priority = 3  # demote excess variants
                # Find and remove from scored
                for idx, (spec, score_val) in enumerate(scored):
                    if spec is s:
                        scored[idx] = (spec, -999.0)
                        break
    scored = [(s, sc) for s, sc in scored if sc > -999.0]

    # P2 count scales with data size
    if n_samples < 1000:
        n_p2 = 5
    elif n_samples < 5000:
        n_p2 = 10
    elif n_samples < 10000:
        n_p2 = 14
    elif n_samples < 50000:
        n_p2 = 17
    else:
        n_p2 = 17

    n_p2 = max(n_p2, len(group_best))

    # --- Group-level minimum quotas ---
    # Some architecture groups should have extra representation beyond the
    # default 1-per-group guarantee, especially when hardware allows it.
    _group_quotas: dict[str, int] = {}
    for gname in group_best:
        _group_quotas[gname] = 1  # every group gets at least 1

    # Transformer base: at least 3 when GPU and sufficient data
    if has_gpu and gpu_vram >= 4 and n_samples >= 3000:
        _group_quotas["tf_base"] = max(_group_quotas.get("tf_base", 1), 3)
    # RNN (non-attention): at least 2 when GPU available
    if has_gpu:
        _group_quotas["rnn"] = max(_group_quotas.get("rnn", 1), 2)
    # Linear sparse: at least 2 (SVM + LR complement each other)
    _group_quotas["linear_sparse"] = max(_group_quotas.get("linear_sparse", 1), 2)

    total_quota = sum(_group_quotas.values())
    n_p2 = max(n_p2, total_quota)

    # --- Greedy selection with quotas ---
    p2_specs: list = []
    _quota_remaining = dict(_group_quotas)
    remaining_slots = n_p2

    # Find which group each spec belongs to
    _spec_groups: dict = {}
    for s, _sc in scored:
        for gname, gfn in _ARCH_GROUPS:
            if gfn(s):
                _spec_groups[id(s)] = gname
                break

    # Build priority queue: (effective_score, spec)
    candidates = list(scored)
    while remaining_slots > 0 and candidates:
        best_idx = 0
        best_effective = -999.0
        for idx, (s, raw_score) in enumerate(candidates):
            gname = _spec_groups.get(id(s), "")
            quota_left = _quota_remaining.get(gname, 0)
            # Strong bonus when quota still needs filling
            quota_bonus = 3.0 if quota_left > 0 else 0.0
            # Penalty for exceeding group's fair share
            over_quota_penalty = 0.0
            if quota_left <= 0:
                already_in = sum(1 for ps in p2_specs if _spec_groups.get(id(ps)) == gname)
                over_quota_penalty = already_in * 1.5
            effective = raw_score + quota_bonus - over_quota_penalty
            if effective > best_effective:
                best_effective = effective
                best_idx = idx

        chosen_s, _ = candidates.pop(best_idx)
        p2_specs.append(chosen_s)
        remaining_slots -= 1
        gname = _spec_groups.get(id(chosen_s), "")
        if gname in _quota_remaining:
            _quota_remaining[gname] = max(0, _quota_remaining[gname] - 1)

    for s, _ in scored:
        s.priority = 2 if s in p2_specs else 3

    p2_count = len(p2_specs)
    quota_met = sum(1 for q in _quota_remaining.values() if q <= 0)
    print(f"\n  \033[1;33mDynamic scoring\033[0m: {p2_count} models → P2, "
          f"{len(scored) - p2_count} → P3  "
          f"(top-{n_p2}, {quota_met}/{len(_group_quotas)} group quotas filled)")

    return specs


def generate_scheme(
    analysis_path: str,
    output_dir: str = "output",
    web_recommendations: dict = None,
) -> list[ModelSpec]:
    """
    Generate a prioritized model scheme based on data analysis.

    Args:
        analysis_path: Path to analysis.json from Stage 1.
        output_dir: Output directory.
        web_recommendations: Optional dict of model->params from web search.

    Returns: list of ModelSpec with priorities assigned.
    """
    analysis = load_json(analysis_path)
    dataset = analysis.get("dataset", {})
    device = analysis.get("device", {})

    print_header("Stage 3: Model Scheme Generation")

    # Extract key metrics for decision-making
    n_samples = dataset.get("total_samples", 0)
    class_ratio = dataset.get("class_ratio", 1.0)
    is_imbalanced = min(
        dataset.get("class_0_count", 0),
        dataset.get("class_1_count", 0),
    ) / max(n_samples, 1) < 0.20
    text_stats = dataset.get("text_stats", {})
    mean_len = text_stats.get("mean_length", 0)
    vocab_size = text_stats.get("vocab_size", 0)
    richness = dataset.get("vocabulary_richness", {})
    ttr = richness.get("type_token_ratio", 0)
    hapax = richness.get("hapax_legomena_ratio", 0)
    non_en_ratio = dataset.get("non_english_ratio", 0)
    has_gpu = device.get("has_gpu", False)
    gpu_vram = device.get("gpu_vram_gb") or 0
    # MPS (Apple Silicon) shares unified memory — VRAM reports None but
    # most Apple GPUs have at least 8 GB available for compute.
    if gpu_vram == 0 and device.get("gpu_type") == "mps":
        gpu_vram = 8.0

    print(f"  Dataset size: {n_samples} samples")
    print(f"  Mean text length: {mean_len:.0f} words")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  TTR: {ttr:.4f}  |  Hapax: {hapax:.1%}  |  Non-EN: {non_en_ratio:.1%}")
    print(f"  Class ratio: {class_ratio:.2f}  |  Imbalanced: {is_imbalanced}")
    print(f"  GPU: {'yes' if has_gpu else 'no'} ({gpu_vram} GB VRAM)")

    # --- Determine which categories are viable ---
    categories = _determine_categories(n_samples, has_gpu, gpu_vram)

    # --- Get all specs ---
    all_specs = get_default_specs()

    # --- Adjust priorities based on data analysis ---
    specs = _adjust_priorities(all_specs, n_samples, mean_len, vocab_size,
                                is_imbalanced, has_gpu, gpu_vram)

    # --- Mandatory baselines: pick best model(s), demote the rest ---
    baseline_models = _pick_best_baseline(analysis)
    specs = _collapse_to_single_baseline(specs, baseline_models)
    names = [f"{n} + {v}" for n, v in baseline_models]
    print(f"\n  \033[1;32mBaseline model(s)\033[0m: {', '.join(names)}")
    print(f"    (selected based on {n_samples:,} samples, "
          f"mean_len={mean_len:.0f}, vocab={vocab_size:,}, "
          f"GPU={'yes' if has_gpu else 'no'})")

    # --- Curated tuning recommendations ---
    specs = _score_and_assign_priorities(specs, analysis)

    # --- Apply web search recommendations if provided ---
    if web_recommendations:
        specs = _apply_web_recommendations(specs, web_recommendations)

    # --- Filter by viable categories ---
    specs = [s for s in specs if s.category in categories]

    # --- Assign global IDs to all specs before printing ---
    for i, s in enumerate(specs, 1):
        s.id = i

    # --- Present scheme ---
    groups = get_specs_by_category(specs)

    print_subheader("A. Traditional Machine Learning (BoW + Classifier)")
    _print_spec_table(groups.get("traditional_ml", []))

    print_subheader("B. Deep Learning (CNN / LSTM / GRU + Pretrained Embeddings)")
    _print_spec_table(groups.get("deep_learning", []))

    print_subheader("C. Transformers (BERT / RoBERTa / DeBERTa)")
    _print_spec_table(groups.get("transformer", []))

    print_subheader("Summary")
    p1 = [s for s in specs if s.priority == 1]
    p2 = [s for s in specs if s.priority == 2]
    p3 = [s for s in specs if s.priority == 3]
    print(f"  Baseline model(s) (P1): {len(p1)}  ← mandatory baseline(s), fast & reliable")
    if p1:
        print(f"    → {p1[0].display_name}")
    print(f"  Recommended for tuning (P2): {len(p2)}")
    if p3:
        print(f"  Exploratory (P3): {len(p3)}")
    print(f"  Total models available: {len(specs)}")

    # Category-level P1+P2 counts for skill.md header verification
    print("\n  \033[1;33m═══ P1+P2 by category (use these counts for smart-rec headers) ═══\033[0m")
    cat_counts = {}
    for s in specs:
        if s.priority in (1, 2):
            cat = s.category
            if cat not in cat_counts:
                cat_counts[cat] = {"P1": 0, "P2": 0, "total": 0}
            key = "P1" if s.priority == 1 else "P2"
            cat_counts[cat][key] += 1
            cat_counts[cat]["total"] += 1
    for cname in ("traditional_ml", "deep_learning", "transformer"):
        cc = cat_counts.get(cname, {"P1": 0, "P2": 0, "total": 0})
        label_map = {
            "traditional_ml": "Traditional ML",
            "deep_learning": "Deep Learning",
            "transformer": "Transformer",
        }
        label = label_map.get(cname, cname)
        print(f"  {label}: P1={cc['P1']}, P2={cc['P2']} → header count = {cc['total']}")
    print("  \033[90m↑ Paste these numbers into smart-rec table headers: 传统机器学习（X 个）etc.\033[0m")

    # --- Save ---
    out_dir = ensure_output_dir(output_dir)
    output_path = out_dir / "model_scheme.json"
    # Convert ModelSpec objects to dicts for JSON serialization
    scheme_data = {
        "meta": {
            "analysis_path": str(pathlib.Path(analysis_path).resolve()),
            "categories_available": list(categories),
            "n_samples": n_samples,
            "has_gpu": has_gpu,
        },
        "models": [_spec_to_dict(s, s.id) for s in specs],
    }
    save_json(scheme_data, str(output_path))
    print(f"\n  Model scheme saved to: {output_path}")

    return specs


def _determine_categories(n_samples: int, has_gpu: bool, gpu_vram: float) -> set:
    """Determine which model categories are recommended based on data size and hardware."""
    cats = {"traditional_ml"}  # Always viable

    if has_gpu and gpu_vram >= 2:
        cats.add("deep_learning")
    elif n_samples >= 5000:
        cats.add("deep_learning")  # DL viable on CPU for larger datasets

    if has_gpu and gpu_vram >= 4:
        cats.add("transformer")
    elif has_gpu and gpu_vram >= 2 and n_samples <= 20000:
        cats.add("transformer")  # Small transformers on modest GPU

    return cats


def _adjust_priorities(
    specs: list[ModelSpec],
    n_samples: int,
    mean_len: float,
    vocab_size: int,
    is_imbalanced: bool,
    has_gpu: bool,
    gpu_vram: float,
) -> list[ModelSpec]:
    """Set model parameters based on data characteristics.
    Priority assignment is handled by _score_and_assign_priorities() later;
    this function only adjusts per-model hyperparameter defaults."""

    for s in specs:
        # Set class_weight for imbalanced datasets
        if is_imbalanced and s.category == "traditional_ml":
            s.params["class_weight"] = "balanced"

    return specs


def _apply_web_recommendations(specs: list[ModelSpec],
                                recommendations: dict) -> list[ModelSpec]:
    """Merge web search recommendations into model specs."""
    for s in specs:
        key = f"{s.category}/{s.name}"
        if key in recommendations:
            s.params.update(recommendations[key])
    return specs


def _print_spec_table(specs: list[ModelSpec]) -> None:
    """Print a formatted table of model specs with global IDs."""
    if not specs:
        print("  (none available for current hardware/data)")
        return
    headers = ["#", "Priority", "Model", "Vectorizer/Embedding"]
    rows = []
    mode_labels = {
        "feature_extraction": "feature extraction",
        "full_ft": "full fine-tuning",
        "partial_ft": "partial fine-tuning",
        "peft": "LoRA (PEFT)",
    }
    for s in sorted(specs, key=lambda x: (x.priority, x.display_name)):
        prio_label = {1: "[P1 BASELINE ★]", 2: "[P2 TUNE]", 3: "[P3 EXPLORE]"}.get(s.priority, "")
        vec_info = VEC_INFO.get(s.vectorizer, s.vectorizer)
        if s.use_embedding != "none":
            if s.category == "transformer":
                embedding_label = mode_labels.get(s.training_mode, s.training_mode)
            elif s.category == "traditional_ml":
                embedding_label = "averaged"
            else:
                embedding_label = "fixed" if s.freeze_embeddings else "fine-tuned"
            vec_info = f"{s.use_embedding} ({embedding_label})"
        rows.append([str(s.id), prio_label, s.display_name, vec_info])
    print_table(headers, rows)


VEC_INFO = {
    "count": "Count Vectorizer (1-gram)",
    "count_bigram": "Count Vectorizer (1,2-gram)",
    "onehot": "OneHot Encoding (1-gram)",
    "tfidf": "TF-IDF (1-gram)",
    "tfidf_bigram": "TF-IDF (1,2-gram)",
    "embedding": "Dense Embedding Avg",
    "tokenize": "Word Tokenization",
    "tokenize_subword": "BERT/Subword Tokenizer",
}


def _spec_to_dict(spec: ModelSpec, model_id: int = 0) -> dict:
    """Convert ModelSpec to JSON-serializable dict."""
    return {
        "id": model_id,
        "name": spec.name,
        "category": spec.category,
        "display_name": spec.display_name,
        "priority": spec.priority,
        "is_recommended": spec.priority in (1, 2),
        "params": spec.params,
        "vectorizer": spec.vectorizer,
        "ngram_range": list(spec.ngram_range),
        "use_embedding": spec.use_embedding,
        "freeze_embeddings": spec.freeze_embeddings,
        "training_mode": spec.training_mode,
    }



def _list_all_models_table(scheme_path: str, output_dir: str = None) -> None:
    """Print the complete 153-model table grouped by sub-module.

    Reads model_scheme.json and outputs 7 sub-tables (A–G), each row with
    global ID, recommendation marker, model name, vectorizer/embedding, and
    priority.  Recommended models (P1/P2) are marked with ✓ and bolded via
    Markdown **…** so the orchestrating LLM can display the output directly
    without manually transcribing JSON.

    When *output_dir* is provided, the full table is also written to
    ``<output_dir>/model_list.md`` so it can be read back without truncation.
    """
    import json as _json
    if not os.path.exists(scheme_path):
        print(f"[ERROR] Model scheme file not found: {scheme_path}")
        sys.exit(1)
    with open(scheme_path, "r", encoding="utf-8") as fh:
        data = _json.load(fh)

    models = data["models"]

    vec_labels = {
        "count": "Count (1-gram)", "count_bigram": "Count (1,2-gram)",
        "onehot": "OneHot (1-gram)", "tfidf": "TF-IDF (1-gram)",
        "tfidf_bigram": "TF-IDF (1,2-gram)",
    }
    enc_labels = {
        "textcnn": "TextCNN", "bilstm": "BiLSTM", "lstm": "LSTM",
        "stacked_lstm": "Stacked LSTM", "lstm_attention": "LSTM + Attention",
        "bigru": "BiGRU", "gru": "GRU", "stacked_gru": "Stacked GRU",
        "gru_attention": "GRU + Attention",
    }

    def _b(marker, text):
        """Wrap in ** if recommended, else return plain text."""
        return f"**{text}**" if marker == "✓" else text

    def _r(m):
        return "✓" if m.get("is_recommended") else "—"

    def _p(m):
        p = m["priority"]
        return f"P{p} ★" if (p == 1 and m.get("is_recommended")) else f"P{p}"

    # Partition models
    trad_sparse, trad_dense, dl_models = [], [], []
    tf_full, tf_extract, tf_partial, tf_peft = [], [], [], []

    for m in models:
        cat = m["category"]
        rec = _r(m)
        pid = m["id"]
        disp = m["display_name"]
        pri = _p(m)
        if cat == "traditional_ml":
            emb = m.get("use_embedding", "none")
            if emb != "none":
                trad_dense.append((pid, rec, disp, pri, emb))
            else:
                vec = vec_labels.get(m.get("vectorizer", ""), m.get("vectorizer", ""))
                trad_sparse.append((pid, rec, disp, pri, vec))
        elif cat == "deep_learning":
            mode = "微调" if m.get("freeze_embeddings") is False else "冻结"
            enc = enc_labels.get(m.get("name", ""), m.get("name", ""))
            dl_models.append((pid, rec, enc, m.get("use_embedding", ""), mode, pri))
        elif cat == "transformer":
            tm = m.get("training_mode", "")
            if tm == "full_ft":
                tf_full.append((pid, rec, disp, pri))
            elif tm == "feature_extraction":
                tf_extract.append((pid, rec, disp, pri))
            elif tm == "partial_ft":
                tf_partial.append((pid, rec, disp, pri))
            else:
                tf_peft.append((pid, rec, disp, pri))

    # Collect all lines into a buffer so we can write to file AND stdout
    buf = []
    def _emit(*args, **kwargs):
        line = " ".join(str(a) for a in args) if args else ""
        if kwargs:
            line = line + " ".join(f"{k}={v}" for k, v in kwargs.items())
        buf.append(line)
        print(line)

    _emit()
    _emit("<!-- FULL_MODEL_LIST_START -->")
    _emit("## 全部 153 个模型（按模块分类）")
    _emit()
    _emit("💡 直接输入编号选择模型，如 `19,21,90,92,96,98`。支持逗号分隔和范围选取。")
    _emit("💡 推荐标注说明：P1 ★ + P2 = 智能推荐（✓），P3 = 探索（—）。")
    _emit()

    # A
    _emit("### A. 传统机器学习 — 稀疏特征（{}个）".format(len(trad_sparse)))
    _emit("| 编号 | 推荐 | 模型 | 向量化器 | 优先级 |")
    _emit("|------|------|------|---------|--------|")
    for row in trad_sparse:
        pid, rec, disp, pri, vec = row
        _emit(f"| {_b(rec, str(pid))} | {_b(rec, rec)} | {_b(rec, disp)} | {_b(rec, vec)} | {_b(rec, pri)} |")
    _emit()

    # B
    _emit("### B. 传统机器学习 — 稠密嵌入（{}个）".format(len(trad_dense)))
    _emit("| 编号 | 推荐 | 模型 | 词嵌入 | 优先级 |")
    _emit("|------|------|------|--------|--------|")
    for row in trad_dense:
        pid, rec, disp, pri, emb = row
        _emit(f"| {_b(rec, str(pid))} | {_b(rec, rec)} | {_b(rec, disp)} | {_b(rec, emb)} | {_b(rec, pri)} |")
    _emit()

    # C
    _emit("### C. 深度学习（{}个）".format(len(dl_models)))
    _emit("| 编号 | 推荐 | 编码器 | 词嵌入 | 嵌入方式 | 优先级 |")
    _emit("|------|------|--------|--------|---------|--------|")
    for row in dl_models:
        pid, rec, enc, emb, mode, pri = row
        _emit(f"| {_b(rec, str(pid))} | {_b(rec, rec)} | {_b(rec, enc)} | {_b(rec, emb)} | {_b(rec, mode)} | {_b(rec, pri)} |")
    _emit()

    # D subsections
    for title, sublist in [
        ("Full Fine-tuning", tf_full),
        ("Feature Extraction", tf_extract),
        ("Partial Fine-tuning", tf_partial),
        ("LoRA / PEFT", tf_peft),
    ]:
        _emit(f"### D. Transformer — {title}（{len(sublist)}个）")
        _emit("| 编号 | 推荐 | 模型 | 优先级 |")
        _emit("|------|------|------|--------|")
        for row in sublist:
            pid, rec, disp, pri = row
            _emit(f"| {_b(rec, str(pid))} | {_b(rec, rec)} | {_b(rec, disp)} | {_b(rec, pri)} |")
        _emit()

    # Summary
    n_p1 = sum(1 for m in models if m["priority"] == 1 and m.get("is_recommended"))
    n_p2 = sum(1 for m in models if m["priority"] == 2 and m.get("is_recommended"))
    n_p3 = sum(1 for m in models if m["priority"] == 3)
    _emit(f'> **总计：{len(models)} 个模型**（P1 ★ = {n_p1}，P2 = {n_p2} — 推荐 ✓，P3 = {n_p3}）')
    _emit("<!-- FULL_MODEL_LIST_END -->")

    # Persist to file so the orchestrating LLM can read without truncation
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "model_list.md")
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(buf) + "\n")
        print()
        print(f"[OK] Full model list (all 153 rows) saved to: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Model Scheme Generation"
    )
    parser.add_argument("--analysis", default="",
                        help="Path to analysis.json from Stage 1")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory")
    parser.add_argument("--web-recommendations", default=None,
                        help="Optional JSON file with web search recommendations")
    parser.add_argument("--list-all", default=None,
                        help="Read existing model_scheme.json and print complete "
                             "153-model table (7 sub-tables, all rows). "
                             "Pass path to model_scheme.json.")
    args = parser.parse_args()

    if args.list_all:
        _list_all_models_table(args.list_all, output_dir=args.output_dir)
        return

    if not args.analysis:
        print("[ERROR] --analysis is required (or use --list-all to print the full model table)")
        sys.exit(1)

    if not os.path.exists(args.analysis):
        print(f"[ERROR] Analysis file not found: {args.analysis}")
        sys.exit(1)

    web_recs = None
    if args.web_recommendations and os.path.exists(args.web_recommendations):
        web_recs = load_json(args.web_recommendations)

    try:
        generate_scheme(
            analysis_path=args.analysis,
            output_dir=args.output_dir,
            web_recommendations=web_recs,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
