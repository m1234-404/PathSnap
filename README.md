# PathSnap — Distributed Map-Matching & Route Reconstruction

Team: PES2UG24CS332–346 (see `TEAM.md`).

## Important source requirements
Read the five supplied PDFs before implementation. This scaffold follows the supplied deliverables: data processing/EDA, at least five model experiments, group-aware validation over the 80 training trips, Bayesian optimisation, HMM/Viterbi, Folium maps, FastAPI, evaluator UI, reproducible logs, and one consolidated report.

**Do not invent results.** All metrics, plots, tables and report conclusions must be generated from actual runs. `TODO` markers identify decisions or implementation work the team must complete and defend.

## Environment
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset
The supplied archive is extracted under `data/raw/Student folder/` and contains `train/` (80 trips) and `student_test/` (10 trips). The supplied package does **not** contain the evaluator-testing partition (10 trips); do not fabricate it or substitute student-test trips.

Raw formats (no headers; whitespace separated):
- `.track`: longitude, latitude, timestamp seconds
- `.nodes`: longitude, latitude; row index is node ID
- `.arcs`: from-node ID, to-node ID; row index is arc ID
- `.route`: chronological reference arc IDs

Never assume a GPS row corresponds to a node/arc row. Keep reference routes out of model input and UI. Evaluator ground-truth labels must remain server-side.

## Run
```bash
python scripts/build_inventory.py
python scripts/run_eda.py --partition train --trip 00000001
python -m uvicorn member_4_pavithra.api.main:app --host 0.0.0.0 --port 8000
streamlit run member_4_pavithra/ui/app.py
```
Training scripts and model implementations are starter templates; complete the TODOs before running end-to-end. The five baseline model slots are configured in `member_1_medha/model_experiments/train_models.py`.

## Two-laptop demo
Laptop 1: run FastAPI on port 8000; allow the required network access and show its request logs.
Laptop 2: set `PATHSNAP_API_URL=http://<laptop-1-ip>:8000`, then run Streamlit. The UI sends only a trip ID and displays API output. Do not send labels or route files to the UI.

## Team workflow
See `TEAM.md` and each member's `README.md`. Each member should implement, test, commit and explain their own contribution. Do not share trained weights/configurations with other teams. Keep seeds, configs and experiment logs.

## Dataset partition caveat
The uploaded archive has 80 training + 10 student-testing trips. It does not include evaluator-testing trips. The API defaults to `evaluator_test` and will list no trips until the official evaluator partition is placed on Laptop 1. Do not point the evaluator UI at `student_test` as a substitute.
