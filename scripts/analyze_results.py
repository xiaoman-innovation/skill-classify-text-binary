"""
Post-training analysis: overfitting detection, tuning efficiency,
cross-model comparison, and diagnostic recommendations.

Run standalone:
  PYTHONIOENCODING=utf-8 python scripts/analyze_results.py \
    --results output/training_results.json \
    --analysis output/analysis.json \
    --output-dir output

Or import:
  from analyze_results import run_post_analysis, print_analysis
"""

import json
import sys
from pathlib import Path
from typing import Optional


def load_results(training_results_path: str) -> dict:
    with open(training_results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_analysis(analysis_path: Optional[str]) -> Optional[dict]:
    if analysis_path is None:
        return None
    path = Path(analysis_path)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def _get_entries(results: dict):
    """Yield (model_name, mode, entry_dict) for every baseline/tuned entry."""
    for model_name, model_data in results.items():
        for mode in ['baseline', 'tuned']:
            if mode in model_data:
                yield model_name, mode, model_data[mode]


def _get_best_entry(results: dict):
    """Yield (model_name, best_mode, best_entry) preferring tuned over baseline."""
    for model_name, model_data in results.items():
        if 'tuned' in model_data:
            yield model_name, 'tuned', model_data['tuned']
        elif 'baseline' in model_data:
            yield model_name, 'baseline', model_data['baseline']


# ── 1. Overfitting Detection ──────────────────────────────────────────

def analyze_overfitting(results: dict) -> list:
    """
    Detect overfitting by comparing Train vs Validation metrics.
    Flags severe (>15pp gap) and moderate (>5pp gap) cases.
    """
    findings = []
    for model_name, mode, entry in _get_entries(results):
        train_acc = entry.get('mean_train_accuracy', entry.get('train_accuracy', 0))
        val_acc = entry.get('mean_accuracy', entry.get('accuracy', 0))
        train_f1 = entry.get('mean_train_f1', entry.get('train_f1', 0))
        val_f1 = entry.get('mean_f1', entry.get('f1', 0))

        gap_acc = train_acc - val_acc
        gap_f1 = train_f1 - val_f1

        severity = None
        if gap_acc > 0.15 or gap_f1 > 0.15:
            severity = 'severe'
        elif gap_acc > 0.05 or gap_f1 > 0.05:
            severity = 'moderate'

        if severity:
            findings.append({
                'model': model_name,
                'mode': mode,
                'train_acc': round(train_acc, 4),
                'val_acc': round(val_acc, 4),
                'gap_acc': round(gap_acc, 4),
                'train_f1': round(train_f1, 4),
                'val_f1': round(val_f1, 4),
                'gap_f1': round(gap_f1, 4),
                'severity': severity
            })
    return findings


# ── 2. Test Set Anomaly Detection ─────────────────────────────────────

def analyze_test_anomaly(results: dict) -> list:
    """
    Flag models whose Test metric exceeds ALL individual CV fold metrics.
    This suggests the final re-train procedure may differ from CV.
    """
    findings = []
    for model_name, mode, entry in _get_entries(results):
        folds = entry.get('cv_folds', [])
        if not folds:
            continue

        fold_accs = [f.get('accuracy', f.get('val_accuracy', 0)) for f in folds]
        fold_f1s = [f.get('f1', f.get('val_f1', 0)) for f in folds]

        mean_val_acc = entry.get('mean_accuracy', 0)
        std_val_acc = entry.get('std_accuracy', 0)

        test_metrics = entry.get('test_metrics', {})
        test_acc = test_metrics.get('accuracy', 0)
        test_f1 = test_metrics.get('f1', 0)

        if not fold_accs:
            continue

        max_fold_acc = max(fold_accs)

        if test_acc > max_fold_acc + 0.02:
            findings.append({
                'model': model_name,
                'mode': mode,
                'test_acc': round(test_acc, 4),
                'best_fold_acc': round(max_fold_acc, 4),
                'mean_val_acc': round(mean_val_acc, 4),
                'std_val_acc': round(std_val_acc, 4),
                'test_f1': round(test_f1, 4),
                'best_fold_f1': round(max(fold_f1s), 4) if fold_f1s else 0,
                'anomaly': 'test_above_all_folds',
                'gap_to_best_fold': round(test_acc - max_fold_acc, 4),
            })
    return findings


# ── 3. Model-Data Size Mismatch ───────────────────────────────────────

def analyze_model_data_mismatch(results: dict, analysis: Optional[dict] = None) -> list:
    """
    Check if model complexity is appropriate for dataset size.
    Uses heuristics from the skill's data-threshold rules.
    """
    findings = []
    n_samples = 5000
    if analysis:
        ds = analysis.get('dataset', {})
        n_samples = ds.get('total_samples', 5000)

    for model_name, mode, entry in _get_best_entry(results):
        category = entry.get('category', '')

        issue = None
        if 'Stacked' in model_name and n_samples < 10000:
            issue = f'Stacked 编码器在 {n_samples} 样本上容易过拟合（多层结构参数过多）。建议单层 BiLSTM/BiGRU'
        elif 'LSTM' in model_name and 'Attention' in model_name and n_samples < 8000:
            issue = f'LSTM+Attention 在 {n_samples} 样本上可能过于复杂。Attention 机制需要更多数据学习对齐'
        elif 'large' in model_name.lower() and n_samples < 20000:
            issue = f'Large 模型在 {n_samples} 样本上不推荐，参数量远超数据规模'
        elif 'GRU' in model_name and 'Attention' in model_name and n_samples < 8000:
            issue = f'GRU+Attention 在 {n_samples} 样本上可能过于复杂'

        if issue:
            findings.append({
                'model': model_name,
                'category': category,
                'n_samples': n_samples,
                'issue': issue
            })
    return findings


# ── 4. Tuning Efficiency ──────────────────────────────────────────────

def analyze_tuning_efficiency(results: dict) -> list:
    """
    Evaluate whether Optuna tuning produced meaningful improvement.
    Flags low-ROI tuning runs (few params, many trials, negligible gain).
    """
    findings = []
    for model_name, model_data in results.items():
        if 'tuned' not in model_data or 'baseline' not in model_data:
            continue

        baseline = model_data['baseline']
        tuned = model_data['tuned']

        baseline_f1 = baseline.get('mean_f1', 0)
        tuned_f1 = tuned.get('mean_f1', 0)
        improvement = tuned_f1 - baseline_f1

        best_params = tuned.get('best_params', {})
        n_params = len(best_params)
        n_trials = tuned.get('total_trials', 0)

        verdict = None
        if improvement < 0.003:
            verdict = '提升可忽略（ΔF1 < 0.003），默认参数已足够'
        elif improvement < 0.01:
            verdict = '提升有限（ΔF1 < 0.01），调参收益偏低'

        if n_params <= 2 and n_trials >= 10 and improvement < 0.01:
            verdict = (f'仅 {n_params} 个超参数用了 {n_trials} 次搜索，'
                       f'ΔF1={improvement:+.4f}，性价比极低。建议直接用默认参数')

        findings.append({
            'model': model_name,
            'baseline_f1': round(baseline_f1, 4),
            'tuned_f1': round(tuned_f1, 4),
            'improvement': round(improvement, 4),
            'n_params': n_params,
            'n_trials': n_trials,
            'verdict': verdict
        })
    return findings


# ── 5. Cross-Model Ranking ────────────────────────────────────────────

def rank_models(results: dict) -> list:
    """Rank all models by Test F1 (preferring tuned over baseline per model)."""
    rankings = []
    for model_name, mode, entry in _get_best_entry(results):
        test_metrics = entry.get('test_metrics', {})
        rankings.append({
            'model': model_name,
            'mode': mode,
            'category': entry.get('category', ''),
            'mean_train_acc': round(entry.get('mean_train_accuracy', 0), 4),
            'mean_train_f1': round(entry.get('mean_train_f1', 0), 4),
            'val_acc': round(entry.get('mean_accuracy', 0), 4),
            'val_f1': round(entry.get('mean_f1', 0), 4),
            'val_auc': round(entry.get('mean_auc', 0), 4),
            'val_std': round(entry.get('std_accuracy', 0), 4),
            'test_acc': round(test_metrics.get('accuracy', 0), 4),
            'test_f1': round(test_metrics.get('f1', 0), 4),
            'test_auc': round(test_metrics.get('auc', 0), 4),
            'fit_time': entry.get('total_fit_time', 0),
        })
    rankings.sort(key=lambda x: x['test_f1'], reverse=True)
    return rankings


# ── 6. Per-Fold Stability ─────────────────────────────────────────────

def analyze_fold_stability(results: dict) -> list:
    """
    Check per-fold variance — high variance indicates unstable model.
    CV fold Acc spread > 5% flagged.
    """
    findings = []
    for model_name, mode, entry in _get_entries(results):
        folds = entry.get('cv_folds', [])
        if not folds:
            continue
        fold_accs = [f.get('accuracy', f.get('val_accuracy', 0)) for f in folds]
        if not fold_accs:
            continue
        spread = max(fold_accs) - min(fold_accs)
        if spread > 0.05:
            findings.append({
                'model': model_name,
                'mode': mode,
                'fold_accs': [round(a, 4) for a in fold_accs],
                'spread': round(spread, 4),
                'severity': 'high' if spread > 0.08 else 'moderate'
            })
    return findings


# ── 7. Intelligent Recommendations ─────────────────────────────────────

def generate_recommendations(report: dict, analysis: Optional[dict] = None) -> list:
    """
    Generate specific, actionable recommendations from all diagnostic findings.
    Each recommendation includes: severity, category, issue, action, and rationale.
    """
    recommendations = []
    ds = (analysis or {}).get('dataset', {})
    n_samples = ds.get('total_samples', 0)
    mean_len = ds.get('text_statistics', {}).get('mean_length', 0)

    # --- Convergence failure: folds with near-random accuracy ---
    for fs in report.get('fold_stability', []):
        fold_accs = fs['fold_accs']
        n_random = sum(1 for a in fold_accs if 0.48 <= a <= 0.54)
        n_good = sum(1 for a in fold_accs if a > 0.70)

        if n_random >= 2 and n_good >= 1 and fs['spread'] > 0.15:
            # Partial convergence failure — some folds learned, some didn't
            category = fs.get('category', _infer_category(fs['model']))
            if category in ('transformer', 'deep_learning'):
                recommendations.append({
                    'priority': 1,
                    'severity': 'critical',
                    'category': 'convergence_failure',
                    'model': fs['model'],
                    'mode': fs['mode'],
                    'issue': (
                        f'{n_random}/{len(fold_accs)} CV folds failed to converge '
                        f'(accuracy ≈ random 0.50), while {n_good}/{len(fold_accs)} folds '
                        f'learned successfully (acc > 0.70). Spread = {fs["spread"]:.2f}. '
                        f'This means default hyperparameters are at the boundary of stability '
                        f'for this data.'
                    ),
                    'action': [
                        f'Increase learning_rate from default (2e-5) to 3e-5 or 5e-5',
                        f'Increase warmup_ratio from default (0.06) to 0.1',
                        f'Consider --tune-method split instead of cv to avoid per-fold instability',
                        f'If using fp16, switch to fp32 for more stable optimization',
                    ],
                    'rationale': (
                        f'The loss plateau at ~0.693 (binary random baseline) indicates '
                        f'optimizer stuck in flat region. A higher LR helps escape. '
                        f'Warmup gives optimizer time to find a good descent direction.'
                    ),
                })
            else:
                recommendations.append({
                    'priority': 2,
                    'severity': 'high',
                    'category': 'convergence_failure',
                    'model': fs['model'],
                    'mode': fs['mode'],
                    'issue': (
                        f'{n_random}/{len(fold_accs)} CV folds near random. '
                        f'Unusual for traditional ML — check data quality in those folds.'
                    ),
                    'action': [
                        'Examine per-fold class distribution for imbalance',
                        'Try a different random seed',
                        'Check if text preprocessing is consistent across folds',
                    ],
                    'rationale': 'Traditional ML models rarely show per-fold convergence failure.',
                })

        elif fs['spread'] > 0.08 and not (n_random >= 2 and n_good >= 1):
            recommendations.append({
                'priority': 2,
                'severity': 'high',
                'category': 'cv_instability',
                'model': fs['model'],
                'mode': fs['mode'],
                'issue': (
                    f'CV fold spread = {fs["spread"]:.4f} exceeds 8pp. '
                    f'Model performance varies significantly across data splits.'
                ),
                'action': [
                    'Use --tune-method split for more stable evaluation',
                    'Increase --cv-folds to 10 for more granular estimate',
                    'Check if dataset has hidden sub-domains causing fold imbalance',
                ],
                'rationale': (
                    f'High CV variance means the model is sensitive to data split. '
                    f'Split-based tuning (single train/valid split) avoids this issue.'
                ),
            })

    # --- Overfitting ---
    for o in report.get('overfitting', []):
        if o['severity'] == 'severe':
            category = o.get('category', _infer_category(o['model']))
            rec = {
                'priority': 1,
                'severity': 'high',
                'category': 'severe_overfitting',
                'model': o['model'],
                'mode': o['mode'],
                'issue': (
                    f'Train-Val gap = {o["gap_acc"]:.2%} (Acc) / {o["gap_f1"]:.2%} (F1). '
                    f'Model is memorizing training data.'
                ),
                'action': [],
                'rationale': '',
            }
            if category == 'transformer':
                rec['action'] = [
                    f'Increase dropout from 0.1 to 0.3-0.5',
                    f'Reduce epochs from 3 to 2',
                    f'Use LoRA (low-rank adaptation) instead of full_ft to limit capacity',
                    f'Increase weight_decay from 0.01 to 0.05',
                ]
                rec['rationale'] = (
                    f'Transformer full fine-tuning on {n_samples} samples with '
                    f'mean {mean_len:.0f}-word texts can memorize easily. '
                    f'Stronger regularization is needed.'
                )
            elif category in ('deep_learning',):
                rec['action'] = [
                    'Switch from fine-tuned embeddings to frozen embeddings',
                    'Reduce hidden_dim or number of layers',
                    'Add dropout between all layers (0.3-0.5)',
                ]
                rec['rationale'] = 'Deep learning models overfit quickly on small datasets.'
            else:
                rec['action'] = [
                    'Reduce max_features or increase min_df for TF-IDF vectorizer',
                    'Switch from bigram to unigram (fewer features)',
                    'Add stronger regularization (increase alpha for NB, C for SVM/LR)',
                ]
                rec['rationale'] = 'Traditional ML overfitting is usually caused by too many features.'
            recommendations.append(rec)

        elif o['severity'] == 'moderate':
            recommendations.append({
                'priority': 3,
                'severity': 'moderate',
                'category': 'moderate_overfitting',
                'model': o['model'],
                'mode': o['mode'],
                'issue': (
                    f'Train-Val gap = {o["gap_acc"]:.2%} (Acc) / {o["gap_f1"]:.2%} (F1). '
                    f'Within acceptable range but worth monitoring.'
                ),
                'action': [
                    'Consider light regularization tuning via Optuna',
                    'Monitor if gap widens with more training data',
                ],
                'rationale': '5-15pp gap is common for small datasets and not always a problem.',
            })

    # --- Test anomalies ---
    for a in report.get('test_anomalies', []):
        category = a.get('category', _infer_category(a['model']))
        rec = {
            'priority': 1,
            'severity': 'warning',
            'category': 'test_anomaly',
            'model': a['model'],
            'mode': a['mode'],
            'issue': (
                f'Test Acc ({a["test_acc"]:.4f}) exceeds best CV fold ({a["best_fold_acc"]:.4f}) '
                f'by {a.get("gap_to_best_fold", 0):+.4f}. CV mean = {a["mean_val_acc"]:.4f} '
                f'± {a["std_val_acc"]:.4f}.'
            ),
            'action': [],
            'rationale': '',
        }
        if category == 'transformer':
            n_random = 0
            for fs in report.get('fold_stability', []):
                if fs['model'] == a['model']:
                    n_random = sum(1 for acc in fs['fold_accs'] if 0.48 <= acc <= 0.54)
                    break
            if n_random > 0:
                rec['action'] = [
                    'The CV metric is unreliable due to partial convergence failure (see convergence warning above).',
                    'Trust the Test result but re-run with higher learning rate to verify CV stability.',
                    'For production: use the full-retrain model with default params (it worked on all data).',
                ]
                rec['rationale'] = (
                    f'Full retrain on all {n_samples} samples gave the model enough data '
                    f'to escape the optimization plateau. CV folds with only '
                    f'{n_samples * 0.8 * 0.8:.0f} train samples did not.'
                )
            else:
                rec['action'] = [
                    'Verify that training data and test data come from the same distribution',
                    'Re-run with a different --seed to check if the anomaly persists',
                    'Check if the split column creates a distribution shift between train and test',
                ]
                rec['rationale'] = 'Test significantly better than all CV folds is unusual and needs investigation.'
        else:
            rec['action'] = [
                'Check for data leakage between train and test sets',
                'Verify the split procedure is correct',
            ]
            rec['rationale'] = 'Traditional ML models rarely show this pattern without data issues.'
        recommendations.append(rec)

    # --- Model-data mismatch ---
    for m in report.get('model_data_mismatch', []):
        recommendations.append({
            'priority': 2,
            'severity': 'warning',
            'category': 'model_data_mismatch',
            'model': m['model'],
            'mode': m.get('mode', ''),
            'issue': m['issue'],
            'action': [
                'Switch to a simpler model variant (e.g. base instead of large, BiLSTM instead of Stacked)',
                'If simpler model achieves similar performance, prefer it for deployment',
            ],
            'rationale': f'With only {m["n_samples"]} samples, a simpler model is likely sufficient and more robust.',
        })

    # --- Tuning efficiency ---
    for t in report.get('tuning_efficiency', []):
        if t.get('verdict'):
            if t['improvement'] < 0.005:
                rec_action = [
                    'Skip Optuna tuning for this model in future runs — default params are sufficient',
                    f'The small ΔF1 ({t["improvement"]:+.4f}) does not justify {t["n_trials"]} trials',
                ]
            else:
                rec_action = [
                    f'Consider reducing trials from {t["n_trials"]} to 10 — the extra search found minimal gain',
                ]
            recommendations.append({
                'priority': 3,
                'severity': 'info',
                'category': 'tuning_inefficiency',
                'model': t['model'],
                'mode': 'tuned',
                'issue': (
                    f'Tuning ΔF1 = {t["improvement"]:+.4f} with {t["n_params"]} params '
                    f'over {t["n_trials"]} trials. {t["verdict"]}'
                ),
                'action': rec_action,
                'rationale': 'Optuna tuning has diminishing returns when the hyperparameter space is small.',
            })

    # --- Text length vs max_seq_len for Transformers ---
    if mean_len > 200:
        rankings = report.get('rankings', [])
        for r in rankings:
            if _infer_category(r['model']) == 'transformer':
                if r['test_f1'] > 0.85 and r['val_f1'] < r['test_f1'] - 0.05:
                    recommendations.append({
                        'priority': 2,
                        'severity': 'info',
                        'category': 'seq_len_truncation',
                        'model': r['model'],
                        'mode': r['mode'],
                        'issue': (
                            f'Mean text length = {mean_len:.0f} words exceeds the typical '
                            f'max_seq_len=256 for Transformers. Up to {mean_len - 256:.0f} '
                            f'words may be truncated per sample on average.'
                        ),
                        'action': [
                            'Increase max_seq_len to 512 to capture full text',
                            'If VRAM is limited, use a model with efficient attention (Longformer, BigBird)',
                            'Or truncate strategically (keep first + last N tokens)',
                        ],
                        'rationale': (
                            f'Truncation at 256 tokens for {mean_len:.0f}-word texts may discard '
                            f'important context. Longer sequences use more VRAM but may improve performance.'
                        ),
                    })
                    break

    # Sort by priority (1=critical → 3=info)
    recommendations.sort(key=lambda x: x['priority'])
    return recommendations


def _infer_category(model_name: str) -> str:
    """Infer model category from model name."""
    transformers = ['BERT', 'RoBERTa', 'DeBERTa', 'DistilBERT', 'ALBERT', 'ELECTRA', 'XLNet']
    dl_encoders = ['TextCNN', 'BiLSTM', 'LSTM', 'BiGRU', 'GRU', 'Stacked']
    for t in transformers:
        if t.lower() in model_name.lower():
            return 'transformer'
    for d in dl_encoders:
        if d.lower() in model_name.lower():
            return 'deep_learning'
    return 'traditional_ml'


# ── Orchestration ─────────────────────────────────────────────────────

def run_post_analysis(
    training_results_path: str,
    analysis_path: Optional[str] = None,
    split_info_path: Optional[str] = None,
    output_dir: str = 'output'
) -> dict:
    """Run all post-training analysis modules and return a consolidated report."""

    results = load_results(training_results_path)
    analysis = load_analysis(analysis_path) if analysis_path else None

    report = {
        'rankings': rank_models(results),
        'overfitting': analyze_overfitting(results),
        'test_anomalies': analyze_test_anomaly(results),
        'model_data_mismatch': analyze_model_data_mismatch(results, analysis),
        'tuning_efficiency': analyze_tuning_efficiency(results),
        'fold_stability': analyze_fold_stability(results),
    }
    report['recommendations'] = generate_recommendations(report, analysis)

    output_path = Path(output_dir) / 'post_analysis.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


# ── Text Output ───────────────────────────────────────────────────────

def print_analysis(report: dict):
    """Print the analysis report in readable Chinese format."""

    # ── Rankings ──
    rankings = report.get('rankings', [])
    if rankings:
        print()
        print('=' * 85)
        print('  模型综合排名（按 Test F1 降序）')
        print('=' * 85)
        header = (
            f"{'排名':<4} {'模型':<42} {'模式':<8} "
            f"{'Test F1':<8} {'Val F1':<8} {'Train F1':<8} {'耗时':<10}"
        )
        print(header)
        print('-' * 85)
        for i, r in enumerate(rankings, 1):
            flag = ' ★' if i == 1 else ''
            print(
                f"{i:<4} {r['model']:<42} {r['mode']:<8} "
                f"{r['test_f1']:<8.4f} {r['val_f1']:<8.4f} "
                f"{r['mean_train_f1']:<8.4f} {r['fit_time']:<10.0f}s{flag}"
            )

    # ── Overfitting ──
    overfitting = report.get('overfitting', [])
    if overfitting:
        print()
        print('=' * 85)
        print('  ⚠️  过拟合分析')
        print('=' * 85)
        for o in overfitting:
            icon = '🔴' if o['severity'] == 'severe' else '🟡'
            print(f"  {icon} [{o['severity']}] {o['model']} ({o['mode']})")
            print(f"     Train Acc = {o['train_acc']:.4f}  →  Val Acc = {o['val_acc']:.4f}"
                  f"  (gap = {o['gap_acc']:+.4f})")
            print(f"     Train F1  = {o['train_f1']:.4f}  →  Val F1  = {o['val_f1']:.4f}"
                  f"  (gap = {o['gap_f1']:+.4f})")
            if o['severity'] == 'severe':
                print(f"     建议：增加 dropout、减少层数、使用更轻量的编码器")
    else:
        print()
        print('  ✅ 过拟合检查：所有模型 Train/Val 差距在正常范围内')

    # ── Fold stability ──
    fold_stability = report.get('fold_stability', [])
    if fold_stability:
        print()
        print('=' * 85)
        print('  ⚠️  交叉验证稳定性')
        print('=' * 85)
        for f in fold_stability:
            icon = '🔴' if f['severity'] == 'high' else '🟡'
            print(f"  {icon} {f['model']} ({f['mode']})")
            print(f"     各折 Acc: {f['fold_accs']}")
            print(f"     Max-Min 极差: {f['spread']:.4f}")

    # ── Test anomalies ──
    anomalies = report.get('test_anomalies', [])
    if anomalies:
        print()
        print('=' * 85)
        print('  ⚠️  Test 集异常检测')
        print('=' * 85)
        for a in anomalies:
            print(f"  🔴 {a['model']} ({a['mode']})")
            print(f"     Test Acc = {a['test_acc']:.4f}  >  最佳折 Val Acc = {a['best_fold_acc']:.4f}"
                  f"  (超出 {a.get('gap_to_best_fold', 0):+.4f})")
            print(f"     CV Val 均值 = {a['mean_val_acc']:.4f} ± {a['std_val_acc']:.4f}")
            print(f"     可能原因：全量重训时内部验证集划分方式与 CV 不一致 / 随机种子效应 / 数据分布差异")

    # ── Model-data mismatch ──
    mismatches = report.get('model_data_mismatch', [])
    if mismatches:
        print()
        print('=' * 85)
        print('  ⚠️  模型-数据规模匹配分析')
        print('=' * 85)
        for m in mismatches:
            print(f"  🟡 {m['model']}")
            print(f"     {m['issue']}")

    # ── Tuning efficiency ──
    tuning = report.get('tuning_efficiency', [])
    if tuning:
        print()
        print('=' * 85)
        print('  ⚠️  调参效率分析')
        print('=' * 85)
        for t in tuning:
            if t.get('verdict'):
                print(f"  🟡 {t['model']}")
                print(f"     baseline F1 = {t['baseline_f1']:.4f}  →  "
                      f"tuned F1 = {t['tuned_f1']:.4f}  "
                      f"(Δ = {t['improvement']:+.4f})")
                print(f"     可调超参数: {t['n_params']} 个  |  搜索次数: {t['n_trials']} 次")
                print(f"     评估: {t['verdict']}")
            else:
                print(f"  ✅ {t['model']}")
                print(f"     baseline F1 = {t['baseline_f1']:.4f}  →  "
                      f"tuned F1 = {t['tuned_f1']:.4f}  "
                      f"(Δ = {t['improvement']:+.4f})")

    # ── Summary ──
    print()
    print('=' * 85)
    print('  总结建议')
    print('=' * 85)

    if rankings:
        best = rankings[0]
        print(f"  ★ 最佳模型：{best['model']}")
        print(f"     Test F1 = {best['test_f1']:.4f}  |  Val F1 = {best['val_f1']:.4f}"
              f"  |  AUC = {best['test_auc']:.4f}")

        if len(rankings) > 1:
            second = rankings[1]
            margin = best['test_f1'] - second['test_f1']
            print(f"     领先第二名 {second['model']}: ΔF1 = {margin:+.4f}")

    n_severe = len([o for o in overfitting if o['severity'] == 'severe'])
    if n_severe > 0:
        print(f"  ⚠  {n_severe} 个模型存在严重过拟合，建议在后续任务中简化结构或增加正则化")

    if anomalies:
        print(f"  ⚠  {len(anomalies)} 个模型 Test 指标异常高于 CV 各折，建议复核全量重训逻辑")

    n_inefficient = len([t for t in tuning if t.get('verdict')])
    if n_inefficient:
        print(f"  💡 {n_inefficient} 个模型的调参效率偏低，后续可跳过 Optuna 直接用默认参数")

    # ── Recommendations ──
    print_recommendations(report.get('recommendations', []))

    print()
    print('─' * 85)
    print('  以上分析基于 CV 每折指标、Test 集指标、模型参数量与数据规模的对比。')
    print('  分析结果已保存至: post_analysis.json')


def print_recommendations(recommendations: list):
    """Print specific, actionable recommendations from diagnostics."""
    if not recommendations:
        return

    print()
    print('=' * 85)
    print('  📋 诊断建议（Diagnostic Recommendations）')
    print('=' * 85)

    severity_icons = {
        'critical': '🔴',
        'high': '🟠',
        'moderate': '🟡',
        'warning': '🟡',
        'info': '🔵',
    }
    severity_labels = {
        'critical': '严重',
        'high': '高优先',
        'moderate': '中等',
        'warning': '注意',
        'info': '参考',
    }
    category_labels = {
        'convergence_failure': '收敛失败 — 部分 CV 折未学到任何信息',
        'cv_instability': 'CV 不稳定 — 各折指标差异过大',
        'severe_overfitting': '严重过拟合 — Train-Val 差距 > 15pp',
        'moderate_overfitting': '中度过拟合 — Train-Val 差距 5-15pp',
        'test_anomaly': 'Test 异常 — Test 指标显著高于所有 CV 折',
        'model_data_mismatch': '模型-数据不匹配 — 模型复杂度超过数据规模',
        'tuning_inefficiency': '调参效率低 — Optuna 搜索回报率不足',
        'seq_len_truncation': '文本截断 — 平均文本长度超过 max_seq_len',
    }

    for i, rec in enumerate(recommendations, 1):
        icon = severity_icons.get(rec['severity'], '⚪')
        sev = severity_labels.get(rec['severity'], rec['severity'])
        cat = category_labels.get(rec['category'], rec['category'])
        print()
        print(f"  [{i}] {icon} [{sev}] {cat}")
        print(f"      模型: {rec['model']} ({rec['mode']})")
        print(f"      问题: {rec['issue']}")
        print(f"      建议:")
        for j, action in enumerate(rec['action'], 1):
            print(f"        {j}. {action}")
        if rec.get('rationale'):
            print(f"      原因: {rec['rationale']}")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Post-training model analysis')
    parser.add_argument('--results', required=True, help='Path to training_results.json')
    parser.add_argument('--analysis', default=None, help='Path to analysis.json (optional)')
    parser.add_argument('--split-info', default=None, help='Path to split_info.json (optional)')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    args = parser.parse_args()

    report = run_post_analysis(args.results, args.analysis, args.split_info, args.output_dir)
    print_analysis(report)
