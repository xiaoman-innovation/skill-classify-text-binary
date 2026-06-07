"""
Deployment artifact generation: FastAPI/Flask server, requirements.txt,
Dockerfile, monitoring guide.
"""

import pathlib


def generate_deployment(
    model_path: str,
    vectorizer_path: str = None,
    model_type: str = "sklearn",
    output_dir: str = "output/deploy",
    api_framework: str = "fastapi",
) -> dict:
    """
    Generate all deployment files.

    Args:
        model_path: Path to the saved model file.
        vectorizer_path: Path to the saved vectorizer file (None for HF tokenizers).
        model_type: "sklearn" | "pytorch" | "transformers"
        output_dir: Directory to write deployment files.
        api_framework: "fastapi" | "flask"

    Returns: {filename: path} dict.
    """
    if vectorizer_path is None and model_type in ("sklearn", "pytorch"):
        raise ValueError(
            f"vectorizer_path is required for model_type='{model_type}'. "
            f"Provide the path to the saved vectorizer/tokenizer file."
        )

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = {}

    # API server
    if api_framework == "fastapi":
        content = _generate_fastapi_server(model_path, vectorizer_path, model_type)
    else:
        content = _generate_flask_server(model_path, vectorizer_path, model_type)
    api_file = out / "api_server.py"
    api_file.write_text(content, encoding="utf-8")
    files["api_server.py"] = str(api_file)

    # requirements.txt
    req_content = _generate_requirements(model_type)
    req_file = out / "requirements.txt"
    req_file.write_text(req_content, encoding="utf-8")
    files["requirements.txt"] = str(req_file)

    # Dockerfile
    docker_content = _generate_dockerfile(model_type, api_framework)
    docker_file = out / "Dockerfile"
    docker_file.write_text(docker_content, encoding="utf-8")
    files["Dockerfile"] = str(docker_file)

    # monitoring.md
    mon_content = _generate_monitoring_guide()
    mon_file = out / "monitoring.md"
    mon_file.write_text(mon_content, encoding="utf-8")
    files["monitoring.md"] = str(mon_file)

    return files


def _generate_fastapi_server(model_path: str, vectorizer_path: str,
                             model_type: str) -> str:
    pred_fn, load_code, imports = _get_prediction_code(model_path, vectorizer_path, model_type)
    return f'''"""
Production-ready FastAPI prediction server.
Run: uvicorn api_server:app --host 0.0.0.0 --port 8000
"""
import sys
import os
{imports}
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Text Binary Classifier", version="1.0.0")

MODEL_PATH = "{model_path}"
VECTORIZER_PATH = {repr(vectorizer_path)}

# Load model and vectorizer at startup
model = None
vectorizer = None

@app.on_event("startup")
def load_model():
    global model, vectorizer
{load_code}

class TextInput(BaseModel):
    text: str

class TextBatchInput(BaseModel):
    texts: list[str]

class PredictionOutput(BaseModel):
    prediction: int
    probability: float

class BatchPredictionOutput(BaseModel):
    predictions: list[int]
    probabilities: list[float]

@app.get("/health")
def health():
    return {{"status": "healthy", "model_loaded": model is not None}}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TextInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        pred, prob = _predict_single(input_data.text)
        return PredictionOutput(prediction=int(pred), probability=float(prob))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionOutput)
def predict_batch(input_data: TextBatchInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        preds, probs = _predict_batch(input_data.texts)
        return BatchPredictionOutput(
            predictions=[int(p) for p in preds],
            probabilities=[float(p) for p in probs],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

{pred_fn}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''


def _generate_flask_server(model_path: str, vectorizer_path: str,
                           model_type: str) -> str:
    pred_fn, load_code, imports = _get_prediction_code(model_path, vectorizer_path, model_type)
    return f'''"""
Production-ready Flask prediction server.
Run: gunicorn api_server:app -w 4 -b 0.0.0.0:8000
   or: python api_server.py
"""
import sys
import os
{imports}
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "{model_path}"
VECTORIZER_PATH = {repr(vectorizer_path)}

model = None
vectorizer = None

def init():
    global model, vectorizer
{load_code}

@app.route("/health")
def health():
    return jsonify({{"status": "healthy", "model_loaded": model is not None}})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({{"error": "Model not loaded"}}), 503
    data = request.get_json(force=True)
    if "text" in data:
        pred, prob = _predict_single(data["text"])
        return jsonify({{"prediction": int(pred), "probability": float(prob)}})
    elif "texts" in data:
        preds, probs = _predict_batch(data["texts"])
        return jsonify({{
            "predictions": [int(p) for p in preds],
            "probabilities": [float(p) for p in probs],
        }})
    return jsonify({{"error": "Provide 'text' or 'texts' field"}}), 400

# Load model at import time (works for both gunicorn and python direct)
try:
    init()
except Exception as e:
    import sys
    print(f"WARNING: Model loading failed: {e}", file=sys.stderr)

{pred_fn}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
'''


def _escape_path(p: str) -> str:
    """Escape backslashes and double quotes so the path is safe inside a Python string literal."""
    if p is None:
        return None
    return p.replace("\\", "\\\\").replace('"', '\\"')


def _get_prediction_code(model_path: str, vectorizer_path: str,
                         model_type: str) -> tuple:
    """Returns (prediction_functions, load_code, imports) based on model type."""
    # Escape paths to prevent broken string literals in generated code
    model_path = _escape_path(model_path)
    vectorizer_path = _escape_path(vectorizer_path)
    if model_type == "sklearn":
        imports = "import joblib\nimport numpy as np"
        # Security note: joblib uses pickle internally; ensure model files are trusted
        load_code = f"""    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)"""
        pred_fn = '''
def _preprocess(texts):
    cleaned = [str(t).lower().strip() for t in texts]
    return vectorizer.transform(cleaned)

def _predict_single(text):
    X = _preprocess([text])
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        return int(proba.argmax()), float(proba.max())
    else:
        score = float(model.decision_function(X)[0])
        pred = 1 if score > 0 else 0
        conf = 1.0 / (1.0 + np.exp(-abs(score)))
        return pred, conf

def _predict_batch(texts):
    X = _preprocess(texts)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        return proba.argmax(axis=1).tolist(), proba.max(axis=1).tolist()
    else:
        scores = model.decision_function(X)
        preds = (scores > 0).astype(int).tolist()
        confs = (1.0 / (1.0 + np.exp(-np.abs(scores)))).tolist()
        return preds, confs
'''
    elif model_type == "pytorch":
        imports = '''import torch
import torch.nn.functional as F
import numpy as np
import joblib'''
        load_code = f"""    # weights_only=True (default) for security -- only load trusted model files
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()
    # Security note: joblib uses pickle internally; ensure model files are trusted
    vectorizer = joblib.load(VECTORIZER_PATH)  # word2idx dict"""
        pred_fn = '''
def _preprocess(texts):
    cleaned = [str(t).lower().strip().split() for t in texts]
    max_len = 128
    result = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, tokens in enumerate(cleaned):
        for j, tok in enumerate(tokens[:max_len]):
            result[i, j] = vectorizer.get(tok, 1)
    return torch.LongTensor(result)

def _predict_single(text):
    model.eval()
    with torch.no_grad():
        X = _preprocess([text])
        logits = model(X)
        probs = F.softmax(logits, dim=1)
    return int(probs.argmax(dim=1)[0]), float(probs.max(dim=1)[0])

def _predict_batch(texts):
    model.eval()
    with torch.no_grad():
        X = _preprocess(texts)
        logits = model(X)
        probs = F.softmax(logits, dim=1)
    return probs.argmax(dim=1).tolist(), probs.max(dim=1)[0].tolist()
'''
    else:  # transformers
        imports = '''import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from transformers import AutoTokenizer, AutoModel'''
        load_code = f'''    config_path = os.path.join(os.path.dirname(MODEL_PATH), "config.json")
    with open(config_path) as f:
        config = json.load(f)
    training_mode = config.get("training_mode", "full_ft")
    if training_mode == "peft":
        try:
            from peft import PeftModel
            base_model = AutoModel.from_pretrained(config["base_model_name"])
            adapter_path = MODEL_PATH  # MODEL_PATH is the adapter directory for PEFT
            model = PeftModel.from_pretrained(base_model, adapter_path)
        except ImportError:
            raise ImportError("PEFT model requires peft. Install: pip install peft")
    else:
        encoder = AutoModel.from_pretrained(config["base_model_name"])
        model = DeployClassifier(
            encoder=encoder,
            hidden_size=config.get("hidden_size", 768),
            num_classes=config.get("num_classes", 2),
            dropout=config.get("dropout", 0.1),
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
    vectorizer = AutoTokenizer.from_pretrained(VECTORIZER_PATH)'''
        pred_fn = '''
class DeployClassifier(nn.Module):
    """Standalone classifier for deployment (no dependency on model_factory)."""
    def __init__(self, encoder, hidden_size=768, num_classes=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )
        enc_name = type(encoder).__name__.lower()
        self._is_xlnet = "xlnet" in enc_name

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if not self._is_xlnet and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        if self._is_xlnet:
            pooled = outputs.last_hidden_state[:, -1, :]
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)

def _preprocess(texts):
    cleaned = [str(t).strip() for t in texts]
    return vectorizer(cleaned, padding=True, truncation=True, max_length=256,
                      return_tensors="pt")

def _predict_single(text):
    model.eval()
    with torch.no_grad():
        enc = _preprocess([text])
        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = F.softmax(logits, dim=1)
    return int(probs.argmax(dim=1)[0]), float(probs.max(dim=1)[0])

def _predict_batch(texts):
    model.eval()
    with torch.no_grad():
        enc = _preprocess(texts)
        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = F.softmax(logits, dim=1)
    return probs.argmax(dim=1).tolist(), probs.max(dim=1)[0].tolist()
'''
    return pred_fn, load_code, imports


def _generate_requirements(model_type: str) -> str:
    base = [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "nltk>=3.8.0",
    ]
    dl_common = [
        "torch>=2.0.0",
        "tokenizers>=0.13.0",
    ]
    dl_transformers = [
        "transformers>=4.30.0",
        "peft>=0.7.0",
    ]
    web = [
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "pydantic>=2.0.0",
        "gunicorn>=21.0.0",
    ]
    monitor = [
        "prometheus-client>=0.17.0",
    ]

    packages = base[:]
    if model_type in ("pytorch", "transformers"):
        packages.extend(dl_common)
    if model_type == "transformers":
        packages.extend(dl_transformers)
    packages.extend(web)
    packages.extend(monitor)
    return "\n".join(packages)


def _generate_dockerfile(model_type: str, api_framework: str) -> str:
    command = "uvicorn api_server:app --host 0.0.0.0 --port 8000" if api_framework == "fastapi" \
              else "gunicorn api_server:app -w 4 -b 0.0.0.0:8000"
    base_image = "python:3.11-slim" if model_type == "sklearn" \
                 else "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"

    return f'''FROM {base_image}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_server.py .
COPY final_model/ ./final_model/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD {command}
'''


def _generate_monitoring_guide() -> str:
    return """# Model Monitoring Guide

## Metrics to Track

### Prediction Metrics
- **Prediction latency (p50, p95, p99)**: Monitor via Prometheus histogram
- **Prediction throughput**: Requests per second
- **Prediction distribution**: Ratio of class 0 vs class 1 predictions over time

### Data Drift Metrics
- **Input text length distribution**: Track mean and std over time
- **Vocabulary drift**: Compare token frequency distribution vs training data
- **Missing/empty text rate**: Monitor rate of null or empty inputs

### Model Performance Metrics
- **Business KPIs tied to predictions**: Track downstream outcomes
- **Human review accuracy**: If manual review exists, track agreement rate

## Prometheus Integration (FastAPI)

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

PREDICT_COUNT = Counter("predict_requests_total", "Total predictions", ["class"])
PREDICT_LATENCY = Histogram("predict_latency_seconds", "Prediction latency")

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

## Alerting Rules (Prometheus)

```yaml
groups:
  - name: model_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, predict_latency_seconds) > 1.0
        for: 5m
        annotations:
          summary: "p95 prediction latency > 1 second"

      - alert: PredictionDistributionShift
        expr: |
          abs(rate(predict_requests_total{class="0"}[1h]) -
              rate(predict_requests_total{class="1"}[1h])) > 0.3
        for: 1h
        annotations:
          summary: "Prediction class distribution has shifted significantly"
```

## Logging

- Log every prediction with: timestamp, input hash, prediction, confidence
- Log model version and config hash at startup
- Use structured logging (JSON format) for easy parsing

## Model Retraining Trigger

Consider retraining when:
1. Prediction distribution shifts more than 20% from training distribution
2. Business KPIs tied to predictions degrade
3. Human review finds >10% error rate in sampled predictions

## Quick Start

```bash
# Build and run
docker build -t text-classifier .
docker run -p 8000:8000 text-classifier

# Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test message"}'
```
"""
