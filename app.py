import logging
import os
import sys
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Step 4: Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("student-result-api")

# Step 5: Define Application Metrics
HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Student Result Prediction API",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)
PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total Student Result Prediction attempts",
    ["result"],
)
MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Student Result Prediction model is loaded",
)

# Step 6: Connect Model-Loaded Gauge
MODEL_PATH = Path("student_result_model.pkl")

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        MODEL_LOADED.set(1)
        logger.info("event=model_loaded model_file=%s", MODEL_PATH)
        return model
    except Exception:
        MODEL_LOADED.set(0)
        logger.exception("event=model_load_failed model_file=%s", MODEL_PATH)
        raise

model = load_model()

app = FastAPI(title="Student Result Prediction API")

# Step 7: Request/Response Logging & Metrics Middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    path = request.url.path
    method = request.method
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        duration_ms = duration * 1000

        # Prometheus metrics (exclude /metrics scrape traffic)
        if path != "/metrics":
            HTTP_REQUESTS.labels(
                method=method,
                path=path,
                status=str(response.status_code),
            ).inc()
            HTTP_LATENCY.labels(
                method=method,
                path=path,
            ).observe(duration)

        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            level,
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            method,
            path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        duration = time.perf_counter() - start_time
        duration_ms = duration * 1000
        if path != "/metrics":
            HTTP_REQUESTS.labels(
                method=method,
                path=path,
                status="500",
            ).inc()
            HTTP_LATENCY.labels(
                method=method,
                path=path,
            ).observe(duration)
        logger.exception(
            "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            method,
            path,
            duration_ms,
        )
        raise

# Validation Handler
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    fields = [
        ".".join(str(part) for part in error.get("loc", []))
        for error in exc.errors()
    ]
    logger.warning(
        "event=validation_error path=%s field_locations=%s error_count=%s",
        request.url.path,
        fields,
        len(exc.errors()),
    )
    return await request_validation_exception_handler(request, exc)

# Prediction Schema
class StudentResultInput(BaseModel):
    attendance: float
    internal_marks: float
    assignment_marks: float
    previous_score: float

SAFE_INPUT_FIELDS = set()

# Step 8: Count Prediction Successes and Runtime Failures
@app.post("/predict")
def predict(data: StudentResultInput):
    try:
        payload = data.model_dump()
        safe_input = {
            key: value
            for key, value in payload.items()
            if key in SAFE_INPUT_FIELDS
        }
        logger.info(
            "event=prediction_request feature_names=%s safe_input=%s",
            sorted(payload.keys()),
            safe_input,
        )

        features = ["attendance", "internal_marks", "assignment_marks", "previous_score"]
        sample = pd.DataFrame([{f: payload[f] for f in features}])

        prediction_code = int(model.predict(sample)[0])
        prediction = "PASS" if prediction_code == 1 else "FAIL"

        PREDICTIONS.labels(result="success").inc()
        logger.info(
            "event=prediction_success prediction=%s",
            prediction,
        )
        return {
            "prediction": prediction,
            "prediction_code": prediction_code,
        }
    except Exception:
        PREDICTIONS.labels(result="error").inc()
        logger.exception("event=prediction_failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Step 9: Add /metrics Endpoint
@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "student-result-prediction",
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
