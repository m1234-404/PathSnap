from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("PATHSNAP_DATA_ROOT", ROOT / "data/raw/Student folder"))
ARTIFACT_ROOT = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_ROOT / "models/final_model.joblib"
FEATURE_COLUMNS_PATH = ARTIFACT_ROOT / "config/feature_columns.json"
