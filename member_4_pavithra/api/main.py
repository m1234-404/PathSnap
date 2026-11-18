import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pathsnap.config import MODEL_PATH, DATA_ROOT
from src.pathsnap.io import read_trip

app = FastAPI(title="PathSnap API", version="0.1.0")
INFERENCE_PARTITION = os.getenv("PATHSNAP_INFERENCE_PARTITION", "evaluator_test")

class PredictRequest(BaseModel):
    trip_id: str

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.get("/trips")
def trips():
    # Default to evaluator_test. Never silently substitute student_test.
    root = DATA_ROOT / INFERENCE_PARTITION
    if not root.exists():
        return {"trip_ids": [], "partition_available": False}
    return {"trip_ids": sorted(p.name for p in root.iterdir() if p.is_dir()),
            "partition_available": True}

@app.post("/predict")
def predict(req: PredictRequest):
    # Prediction path deliberately does not read .route reference labels.
    if not req.trip_id.isdigit() or len(req.trip_id) > 16:
        raise HTTPException(400, "Invalid trip id")
    trip_dir = DATA_ROOT / INFERENCE_PARTITION / req.trip_id
    if not trip_dir.is_dir():
        raise HTTPException(404, "Trip not found")
    data = read_trip(trip_dir, include_route=False)
    # TODO: integrate candidate generation -> trained classifier -> HMM/Viterbi.
    return {"trip_id": req.trip_id,
            "gps_points": data["track"].to_dict(orient="records"),
            "predicted_arc_ids": [],
            "status": "pipeline_not_integrated"}
