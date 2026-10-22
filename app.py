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
from pydantic import BaseModel

# Step 4: Configure the Application Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("student-result-api")

# Step 5: Log Model Loading
MODEL_PATH = Path("student_result_model.pkl")

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("event=model_loaded model_file=%s", MODEL_PATH)
        return model
    except Exception:
        logger.exception("event=model_load_failed model_file=%s", MODEL_PATH)
        raise

model = load_model()

app = FastAPI(title="Student Result Prediction API")

# Step 6: Add Request/Response Logging Middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            level,
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

# Step 7: Log FastAPI Validation Failures
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

# Step 8: Add Safe Prediction Logging
class StudentResultInput(BaseModel):
    attendance: float
    internal_marks: float
    assignment_marks: float
    previous_score: float

SAFE_INPUT_FIELDS = set()

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
        
        # Order columns as trained: attendance, internal_marks, assignment_marks, previous_score
        features = ["attendance", "internal_marks", "assignment_marks", "previous_score"]
        sample = pd.DataFrame([{f: payload[f] for f in features}])
        
        prediction_code = int(model.predict(sample)[0])
        prediction = "PASS" if prediction_code == 1 else "FAIL"
        
        logger.info(
            "event=prediction_success prediction=%s",
            prediction,
        )
        return {
            "prediction": prediction,
            "prediction_code": prediction_code
        }
    except Exception:
        logger.exception("event=prediction_failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Step 9: Verify Health Endpoint
@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "student-result-prediction"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
