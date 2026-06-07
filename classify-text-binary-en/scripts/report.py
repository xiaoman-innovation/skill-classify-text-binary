"""
HTML report generation: produces a self-contained, responsive HTML training
report with Chart.js for interactive visualizations.
"""

import json
import pathlib
from datetime import datetime


def generate_html_report(
    results: dict,
    output_dir: str = "output",
    report_name: str = "training_report.html",
) -> str:
    """
    Generate a self-contained HTML training report.

    results dict top-level keys:
        dataset, system, model_scheme, training, final_model, deployment

    Returns absolute path to the generated HTML file.
    """
    sections = []
    sections.append(_build_dataset_section(results.get("dataset", {})))
    sections.append(_build_system_section(results.get("system", {})))
    sections.append(_build_scheme_section(results.get("model_scheme", [])))
    sections.append(_build_training_section(results.get("training", {})))
    sections.append(_build_final_model_section(results.get("final_model", {})))
    sections.append(_build_deployment_section(results.get("deployment", {})))

    html = _render_template("\n".join(sections), title="Training Report")

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / report_name
    out_path.write_text(html, encoding="utf-8")
    return str(out_path.resolve())


# ---------------------------------------------------------------------------
# HTML Section Builders
# ---------------------------------------------------------------------------

def _build_dataset_section(ds: dict) -> str:
    if not ds:
        return ""
    class_0 = ds.get("class_0_count", 0)
    class_1 = ds.get("class_1_count", 0)
    text_stats = ds.get("text_stats", {})

    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in [
            ("Total Samples", ds.get("total_samples", "-")),
            ("Class 0 Count", class_0),
            ("Class 1 Count", class_1),
            ("Class Ratio (0:1)", f"{class_0}:{class_1}"),
            ("Mean Text Length (words)", text_stats.get("mean_length", "-")),
            ("Median Text Length", text_stats.get("median_length", "-")),
            ("Max Text Length", text_stats.get("max_length", "-")),
            ("Vocabulary Size", text_stats.get("vocab_size", "-")),
            ("Missing Ratio", f"{text_stats.get('missing_ratio', 0):.2%}"),
            ("Duplicate Ratio", f"{text_stats.get('duplicate_ratio', 0):.2%}"),
            ("Is English", ds.get("is_english", "-")),
        ]
    )

    return f"""
    <section id="dataset">
        <h2>1. Dataset Overview</h2>
        <div class="grid-2">
            <div>
                <h3>Summary</h3>
                <table class="info-table">{rows}</table>
            </div>
            <div>
                <h3>Class Distribution</h3>
                <canvas id="classChart" height="250"></canvas>
            </div>
        </div>
        <script>
        new Chart(document.getElementById('classChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['Class 0', 'Class 1'],
                datasets: [{{ data: [{class_0}, {class_1}], backgroundColor: ['#4e79a7','#f28e2b'] }}]
            }},
            options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }} }}
        }});
        </script>
    </section>"""


def _build_system_section(sys: dict) -> str:
    if not sys:
        return ""
    device = sys.get("device", {})
    libs = sys.get("libraries", {})

    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in [
            ("OS", sys.get("platform", "-")),
            ("Python", sys.get("python_version", "-")),
            ("CPU Cores (physical)", device.get("cpu_cores_physical", "-")),
            ("CPU Cores (logical)", device.get("cpu_cores_logical", "-")),
            ("RAM (GB)", device.get("total_ram_gb", "-")),
            ("GPU", device.get("gpu_name") or "None"),
            ("GPU VRAM (GB)", device.get("gpu_vram_gb") if device.get("gpu_vram_gb") is not None else "-"),
            ("GPU Count", device.get("gpu_count", 0)),
            ("Compute Device", device.get("recommended_device", "cpu")),
        ]
    )
    lib_rows = "".join(
        f"<tr><td>{k}</td><td>{v or 'N/A'}</td></tr>"
        for k, v in sorted(libs.items())
    )

    return f"""
    <section id="system">
        <h2>2. System Information</h2>
        <div class="grid-2">
            <div>
                <h3>Hardware</h3>
                <table class="info-table">{rows}</table>
            </div>
            <div>
                <h3>Libraries</h3>
                <table class="info-table">{lib_rows}</table>
            </div>
        </div>
    </section>"""


def _build_scheme_section(scheme: list) -> str:
    if not scheme:
        return ""
    rows = ""
    for s in scheme:
        cat_emoji = {"traditional_ml": "[ML]", "deep_learning": "[DL]", "transformer": "[TF]"}.get(s.get("category", ""), "")
        rows += f"""
        <tr class="priority-{s.get('priority', 2)}">
            <td>{cat_emoji}</td>
            <td>{s.get('display_name', s.get('name', ''))}</td>
            <td><span class="badge">P{s.get('priority', 2)}</span></td>
        </tr>"""
    return f"""
    <section id="scheme">
        <h2>3. Model Scheme</h2>
        <table class="info-table">
            <thead><tr><th>Category</th><th>Model</th><th>Priority</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        <p class="note">P1 = Baseline (always included) | P2 = Recommended | P3 = Exploratory</p>
    </section>"""


def _build_training_section(training: dict) -> str:
    if not training:
        return ""
    parts = []
    for model_name, model_data in training.items():
        baseline = model_data.get("baseline", {})
        tuned = model_data.get("tuned", {})
        comparison = model_data.get("comparison", {})

        parts.append(f"<h3>{model_name}</h3>")

        # Metrics table
        metrics_rows = ""
        for metric in ["accuracy", "precision", "recall", "f1"]:
            b_val = baseline.get(f"mean_{metric}", "-")
            t_val = tuned.get(f"mean_{metric}", "-")
            if isinstance(b_val, float):
                b_val = f"{b_val:.4f}"
            if isinstance(t_val, float):
                t_val = f"{t_val:.4f}"
            metrics_rows += f"<tr><td>{metric.upper()}</td><td>{b_val}</td><td>{t_val}</td></tr>"

        parts.append(f"""
        <table class="info-table">
            <thead><tr><th>Metric</th><th>Baseline</th><th>Tuned</th></tr></thead>
            <tbody>{metrics_rows}</tbody>
        </table>""")

        if comparison:
            parts.append(f"""
            <p>Improvement: <strong>{comparison.get('absolute_improvement', 0):.4f}</strong> absolute
               ({comparison.get('relative_improvement', 0):.1%} relative)</p>""")

        if tuned.get("best_params"):
            params_rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>"
                for k, v in tuned["best_params"].items()
            )
            parts.append(f"""
            <details><summary>Best Hyperparameters</summary>
            <table class="info-table">{params_rows}</table></details>""")

        if baseline.get("cv_folds"):
            fold_labels = [f"Fold {f['fold']}" for f in baseline["cv_folds"]]
            fold_acc = [f["accuracy"] for f in baseline["cv_folds"]]
            # Strip special characters from model name for valid HTML ID
            _safe_name = model_name
            for _ch in ' +(),.()[]{}#:;=':
                _safe_name = _safe_name.replace(_ch, '_')
            _safe_name = _safe_name.replace('__', '_').strip('_')
            chart_id = f"cvChart_{_safe_name}"
            parts.append(f"""
            <canvas id="{chart_id}" height="200"></canvas>
            <script>
            new Chart(document.getElementById('{chart_id}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(fold_labels)},
                    datasets: [{{
                        label: 'Accuracy per Fold',
                        data: {json.dumps(fold_acc)},
                        backgroundColor: '#4e79a7'
                    }}]
                }},
                options: {{ responsive: true, scales: {{ y: {{ min: 0, max: 1 }} }} }}
            }});
            </script>""")

    return f"""
    <section id="training">
        <h2>4. Training Results</h2>
        {"".join(parts)}
    </section>"""


def _build_final_model_section(final: dict) -> str:
    if not final:
        return ""
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in [
            ("Model Name", final.get("model_name", "-")),
            ("Test Accuracy", f"{final.get('test_accuracy', 0):.4f}" if final.get("test_accuracy") is not None else "-"),
            ("Test Precision", f"{final.get('test_precision', 0):.4f}" if final.get("test_precision") is not None else "-"),
            ("Test Recall", f"{final.get('test_recall', 0):.4f}" if final.get("test_recall") is not None else "-"),
            ("Test F1", f"{final.get('test_f1', 0):.4f}" if final.get("test_f1") is not None else "-"),
            ("Model Path", final.get("saved_path", "-")),
        ]
    )
    cm = final.get("confusion_matrix", [])
    cm_script = ""
    if cm:
        cm_script = f"""
        <canvas id="cmChart" height="250"></canvas>
        <script>
        new Chart(document.getElementById('cmChart'), {{
            type: 'matrix',
            data: {{
                datasets: [{{
                    label: 'Confusion Matrix',
                    data: [
                        {{ x: 'Pred 0', y: 'True 0', v: {cm[0][0]} }},
                        {{ x: 'Pred 1', y: 'True 0', v: {cm[0][1]} }},
                        {{ x: 'Pred 0', y: 'True 1', v: {cm[1][0]} }},
                        {{ x: 'Pred 1', y: 'True 1', v: {cm[1][1]} }},
                    ],
                    backgroundColor: (ctx) => ctx.dataset.data[ctx.dataIndex].v > 0 ? '#4e79a7' : '#ddd'
                }}]
            }}
        }});
        </script>"""
    return f"""
    <section id="final-model">
        <h2>5. Final Model</h2>
        <table class="info-table">{rows}</table>
        {cm_script}
    </section>"""


def _build_deployment_section(deploy: dict) -> str:
    if not deploy:
        return ""
    files_rows = "".join(
        f"<tr><td>{name}</td><td><code>{path}</code></td></tr>"
        for name, path in deploy.get("files", {}).items()
    )
    return f"""
    <section id="deployment">
        <h2>6. Deployment</h2>
        <table class="info-table">
            <thead><tr><th>File</th><th>Path</th></tr></thead>
            <tbody>{files_rows}</tbody>
        </table>
    </section>"""


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

def _render_template(body: str, title: str = "Training Report") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js">
</script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@2.0.1/dist/chartjs-chart-matrix.min.js">
</script>
<style>
:root {{
    --bg: #ffffff; --fg: #1a1a2e; --accent: #4e79a7;
    --border: #e0e0e0; --code-bg: #f5f5f5; --header-bg: #f8f9fa;
}}
[data-theme="dark"] {{
    --bg: #1a1a2e; --fg: #e0e0e0; --accent: #f28e2b;
    --border: #333; --code-bg: #2a2a3e; --header-bg: #22223a;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--fg); line-height: 1.6;
}}
.container {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
header {{
    background: linear-gradient(135deg, #4e79a7, #2c3e50);
    color: white; padding: 30px 0; text-align: center;
}}
header h1 {{ font-size: 2em; }}
header p {{ opacity: 0.8; margin-top: 5px; }}
nav {{
    position: sticky; top: 0; background: var(--header-bg); border-bottom: 1px solid var(--border);
    padding: 10px 20px; z-index: 100; display: flex; gap: 15px; flex-wrap: wrap;
}}
nav a {{ color: var(--accent); text-decoration: none; font-weight: 500; }}
nav a:hover {{ text-decoration: underline; }}
section {{ margin: 30px 0; padding: 20px; border: 1px solid var(--border); border-radius: 8px; }}
section h2 {{ color: var(--accent); border-bottom: 2px solid var(--accent); padding-bottom: 8px; margin-bottom: 15px; }}
h3 {{ margin: 15px 0 8px 0; }}
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
.info-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
.info-table th, .info-table td {{
    padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border);
}}
.info-table th {{ background: var(--header-bg); font-weight: 600; }}
.badge {{
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 0.85em; font-weight: 600;
}}
.priority-1 .badge {{ background: #e74c3c; color: white; }}
.priority-2 .badge {{ background: #f39c12; color: white; }}
.priority-3 .badge {{ background: #3498db; color: white; }}
.note {{ font-size: 0.9em; color: #888; margin-top: 8px; }}
details {{ margin: 10px 0; }}
details summary {{ cursor: pointer; color: var(--accent); }}
code {{ font-family: "Fira Code", "Cascadia Code", monospace; background: var(--code-bg); padding: 2px 6px; border-radius: 3px; }}
footer {{ text-align: center; padding: 20px; color: #888; font-size: 0.9em; border-top: 1px solid var(--border); margin-top: 40px; }}
.theme-toggle {{ cursor: pointer; float: right; background: none; border: 1px solid var(--border); padding: 4px 10px; border-radius: 4px; color: var(--fg); }}
</style>
</head>
<body>
<header>
    <div class="container">
        <h1>{title}</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</header>
<nav>
    <a href="#dataset">1. Dataset</a>
    <a href="#system">2. System</a>
    <a href="#scheme">3. Model Scheme</a>
    <a href="#training">4. Training</a>
    <a href="#final-model">5. Final Model</a>
    <a href="#deployment">6. Deployment</a>
    <button class="theme-toggle" onclick="toggleTheme()">Dark / Light</button>
</nav>
<div class="container">
{body}
</div>
<footer>
    <p>Generated by text-binary-classification skill</p>
</footer>
<script>
function toggleTheme() {{
    const body = document.documentElement;
    const current = body.getAttribute('data-theme');
    body.setAttribute('data-theme', current === 'dark' ? '' : 'dark');
}}
</script>
</body>
</html>"""
