
"""
Candidate-ranking experiments for PathSnap.

One row represents one GPS observation / candidate road arc.
The target is whether the candidate is the reference-route arc.

Integrates with Dhanush's build_trip_table() without modifying
any of his source files.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    log_loss,
)

from sklearn.model_selection import GroupKFold

from member_2_dhanush.features.build_features import (
    FEATURE_COLS as FEATURE_COLUMNS,
    build_trip_table,
)

from src.pathsnap.config import DATA_ROOT, ARTIFACT_ROOT

# ==================================================
# FIVE MACHINE LEARNING MODELS
# ==================================================

MODELS = {

    "logistic_regression": make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            random_state=42
        )
    ),

    # Optimized Random Forest
    "random_forest": make_pipeline(
        SimpleImputer(),
        RandomForestClassifier(
            n_estimators=80,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=2,
            random_state=42
        )
    ),

    # Optimized Extra Trees
    "extra_trees": make_pipeline(
        SimpleImputer(),
        ExtraTreesClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=2,
            random_state=42
        )
    ),

    "hist_gradient_boosting": make_pipeline(
        SimpleImputer(),
        HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=42
        )
    ),

    "gaussian_nb": make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        GaussianNB()
    ),
}

# ==================================================
# DATA COLLECTION AND INTEGRATION
# ==================================================

def collect(partition, k=12):

    rows = []

    base = DATA_ROOT / partition

    if not base.exists():
        raise FileNotFoundError(
            f"Dataset partition not found: {base}"
        )

    trip_dirs = sorted(
        x for x in base.iterdir() if x.is_dir()
    )

    print(
        f"\nCollecting {len(trip_dirs)} trips from {partition}...",
        flush=True
    )

    from src.pathsnap.io import read_trip

    for i, d in enumerate(trip_dirs, 1):

        print(
            f"Processing {partition} trip {i}/{len(trip_dirs)}: {d.name}",
            flush=True
        )

        data = read_trip(d, include_route=True)

        # Convert Pandas DataFrames into NumPy arrays.
        # Dhanush's LocalProjector expects array indexing.

        data["track"] = data["track"].to_numpy(dtype=float)
        data["nodes"] = data["nodes"].to_numpy(dtype=float)
        data["arcs"] = data["arcs"].to_numpy(dtype=np.int64)

        if data["route"] is not None:
            data["route"] = np.asarray(
                data["route"],
                dtype=np.int64
            )

        # Dhanush's function returns three objects.
        cand, obs_df, stats = build_trip_table(
            data,
            with_labels=True,
            k=k
        )

        if "true_arc" not in obs_df.columns:
            raise ValueError(
                f"Reference labels missing for trip {d.name}"
            )

        # Attach reference arc to every candidate row.
        frame = cand.merge(
            obs_df[["obs_idx", "true_arc"]],
            on="obs_idx",
            how="left",
            validate="many_to_one"
        )

        frame = frame.rename(
            columns={
                "obs_idx": "observation_id",
                "arc_id": "candidate_arc_id",
                "true_arc": "target_arc_id",
                "label": "is_target",
            }
        )

        frame["trip_id"] = d.name

        required = (
            [
                "trip_id",
                "observation_id",
                "candidate_arc_id",
                "target_arc_id",
                "is_target"
            ]
            + FEATURE_COLUMNS
        )

        missing = [
            col for col in required
            if col not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns in trip {d.name}: {missing}"
            )

        frame = frame[required].copy()

        frame["is_target"] = frame["is_target"].astype(int)

        if len(frame):
            rows.append(frame)

        print(
            f"  Candidate rows: {len(frame)} | "
            f"Miss rate: {stats.get('candidate_miss_rate', float('nan')):.4f}",
            flush=True
        )

    if not rows:
        raise RuntimeError(
            f"No labelled candidate rows found in {base}"
        )

    result = pd.concat(
        rows,
        ignore_index=True
    )

    print(
        f"\nCompleted {partition}: {result.shape[0]} candidate rows",
        flush=True
    )

    return result

# ==================================================
# TOP-1 CANDIDATE PREDICTION
# ==================================================

def _top1(frame, proba, classes):

    if 1 not in classes:
        scores = np.zeros(len(frame))
    else:
        pos_idx = list(classes).index(1)
        scores = proba[:, pos_idx]

    z = frame[
        [
            "trip_id",
            "observation_id",
            "candidate_arc_id",
            "target_arc_id",
        ]
    ].copy()

    z["score"] = scores

    # Select the highest-scoring candidate per observation.
    z = (
        z.sort_values(
            "score",
            ascending=False
        )
        .drop_duplicates(
            ["trip_id", "observation_id"]
        )
    )

    return z

# ==================================================
# MODEL EVALUATION
# ==================================================

def evaluate(model, frame):

    frame = frame.copy()

    X = frame[FEATURE_COLUMNS]

    y = frame["is_target"].to_numpy()

    probabilities = model.predict_proba(X)

    classes = model.classes_

    predictions = model.predict(X)

    top = _top1(
        frame,
        probabilities,
        classes
    )

    truth = top["target_arc_id"].to_numpy()

    predicted_arcs = top["candidate_arc_id"].to_numpy()

    top1_correct = (predicted_arcs == truth).astype(int)

    # Exclude observations where the true route arc was
    # absent from the generated candidate set.
    valid_top1 = truth != -1

    metrics = {

        "observation_top1_accuracy": float(
            np.mean(top1_correct[valid_top1])
        ) if valid_top1.any() else 0.0,

        # Binary F1 for whether the selected candidate is correct.
        "observation_top1_f1": float(
            f1_score(
                top1_correct[valid_top1],
                np.ones(valid_top1.sum(), dtype=int),
                zero_division=0
            )
        ) if valid_top1.any() else 0.0,

        "candidate_miss_rate": float(
            (
                frame.groupby(
                    ["trip_id", "observation_id"]
                )["is_target"].sum() == 0
            ).mean()
        ),

        "candidate_precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0
            )
        ),

        "candidate_recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0
            )
        ),

        "candidate_f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0
            )
        ),

        "candidate_log_loss": (
            float(
                log_loss(
                    y,
                    probabilities,
                    labels=classes
                )
            )
            if len(classes) > 1
            else None
        ),
    }

    return metrics

# ==================================================
# TRAINING AND CROSS-VALIDATION
# ==================================================

def run_experiments(
    train,
    student,
    output_dir="results/tables"
):

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    model_dir = ARTIFACT_ROOT / "models"

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    X = train[FEATURE_COLUMNS]

    y = train["is_target"].to_numpy()

    groups = train["trip_id"].to_numpy()

    unique_groups = np.unique(groups)

    if len(unique_groups) < 5:
        raise ValueError(
            "At least five training trips are required "
            "for 5-fold GroupKFold."
        )

    cv = GroupKFold(n_splits=5)

    records = []

    for name, base_model in MODELS.items():

        print(
            f"\n{'=' * 50}\n"
            f"Training model: {name}\n"
            f"{'=' * 50}",
            flush=True
        )

        fold_scores = []

        start = time.perf_counter()

        # Five-fold cross-validation grouped by trip.
        for fold_no, (tr, va) in enumerate(
            cv.split(X, y, groups),
            start=1
        ):

            model = clone(base_model)

            print(
                f"Starting {name} fold {fold_no}/5...",
                flush=True
            )

            model.fit(
                X.iloc[tr],
                y[tr]
            )

            fold_predictions = model.predict(
                X.iloc[va]
            )

            score = f1_score(
                y[va],
                fold_predictions,
                average="binary",
                zero_division=0
            )

            fold_scores.append(score)

            print(
                f"Completed fold {fold_no}/5 | "
                f"Candidate F1: {score:.4f}",
                flush=True
            )

        # Train final model on all training data.
        print(
            f"Training final {name} model...",
            flush=True
        )

        final_model = clone(base_model)

        final_model.fit(X, y)

        train_seconds = (
            time.perf_counter() - start
        )

        # Evaluate using student_test.
        print(
            f"Evaluating {name} on student_test...",
            flush=True
        )

        inference_start = time.perf_counter()

        metrics = evaluate(
            final_model,
            student
        )

        inference_seconds = (
            time.perf_counter() - inference_start
        )

        # Save trained model.
        model_file = model_dir / f"{name}.joblib"

        joblib.dump(
            final_model,
            model_file
        )

        records.append({

            "model": name,

            "cv_candidate_f1_mean": float(
                np.mean(fold_scores)
            ),

            "cv_candidate_f1_std": float(
                np.std(fold_scores)
            ),

            **metrics,

            "train_seconds": train_seconds,

            "student_inference_seconds": inference_seconds,
        })

        print(
            f"\n{name} completed.",
            flush=True
        )

        print(
            f"Cross-validation F1: {np.mean(fold_scores):.4f}"
        )

        print(
            f"Student top-1 accuracy: "
            f"{metrics['observation_top1_accuracy']:.4f}"
        )

    # Create comparison table.
    results = pd.DataFrame(records)

    results = results.sort_values(
        "cv_candidate_f1_mean",
        ascending=False
    )

    csv_path = output_path / "model_comparison.csv"

    results.to_csv(
        csv_path,
        index=False
    )

    return results

# ==================================================
# MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    print("\nLoading permitted train and student_test trips...")

    train = collect("train")

    student = collect("student_test")

    print("\n========== DATASET INFORMATION ==========")

    print("Training shape:", train.shape)

    print("Student test shape:", student.shape)

    print(
        "Training trips:",
        train["trip_id"].nunique()
    )

    print(
        "Student test trips:",
        student["trip_id"].nunique()
    )

    print(
        "\n========== STARTING MODEL EXPERIMENTS =========="
    )

    results = run_experiments(
        train,
        student
    )

    print(
        "\n========== FINAL MODEL COMPARISON =========="
    )

    print(
        results.to_string(index=False)
    )

    print(
        "\nComparison saved to:"
        " results/tables/model_comparison.csv"
    )

    print(
        "\nTrained models saved to:"
        f" {ARTIFACT_ROOT / 'models'}"
    )

    print("\nAll five model experiments completed.")