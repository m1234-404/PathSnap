"""Five baseline classifier families. Complete target/candidate dataset assembly first."""
import json, time
from pathlib import Path
import joblib, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold, cross_val_score

MODELS = {
 "logistic_regression": make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced")),
 "random_forest": RandomForestClassifier(n_estimators=300,class_weight="balanced",n_jobs=-1,random_state=42),
 "extra_trees": ExtraTreesClassifier(n_estimators=300,class_weight="balanced",n_jobs=-1,random_state=42),
 "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150,random_state=42),
 "rbf_svm": make_pipeline(SimpleImputer(),StandardScaler(),SVC(C=1.0,kernel="rbf",probability=True,class_weight="balanced",random_state=42)),
}

def run_experiments(X: pd.DataFrame, y, groups, X_student, y_student, output_dir="results/tables"):
    """TODO: verify each row is a candidate and target is candidate's arc_id.

    IMPORTANT: use trip_id groups; no random row split. Ensure each observation's
    candidates are evaluated using observation-level top-1, not merely candidate
    row accuracy. Student-test labels are for development comparison only.
    """
    Path(output_dir).mkdir(parents=True,exist_ok=True)
    records=[]
    cv=GroupKFold(n_splits=5)
    for name, model in MODELS.items():
        start=time.perf_counter()
        scores=cross_val_score(model,X,y,groups=groups,cv=cv,scoring="f1_macro",n_jobs=None)
        model.fit(X,y); train_s=time.perf_counter()-start
        t0=time.perf_counter(); pred=model.predict(X_student); infer_s=time.perf_counter()-t0
        proba=model.predict_proba(X_student) if hasattr(model,"predict_proba") else None
        records.append(dict(model=name,cv_macro_f1_mean=scores.mean(),cv_macro_f1_std=scores.std(),
            student_candidate_accuracy=accuracy_score(y_student,pred),
            student_candidate_macro_f1=f1_score(y_student,pred,average="macro",zero_division=0),
            student_log_loss=log_loss(y_student,proba,labels=model.classes_) if proba is not None else None,
            train_seconds=train_s,inference_seconds=infer_s))
        joblib.dump(model,Path("artifacts/models")/f"{name}.joblib")
    pd.DataFrame(records).to_csv(Path(output_dir)/"model_comparison.csv",index=False)
    return pd.DataFrame(records)

if __name__ == "__main__":
    raise SystemExit("Build candidate-level train/student datasets first; see TODOs.")
