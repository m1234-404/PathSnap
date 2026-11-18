"""Bayesian optimisation starter (Optuna). Use GroupKFold and training trips only."""
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

def optimise(X, y, groups, trials=30):
    cv=GroupKFold(n_splits=5)
    def objective(trial):
        params={
          "n_estimators":trial.suggest_int("n_estimators",150,600),
          "max_depth":trial.suggest_int("max_depth",5,35),
          "min_samples_leaf":trial.suggest_int("min_samples_leaf",1,8),
          "max_features":trial.suggest_float("max_features",0.3,1.0),
          "class_weight":"balanced", "n_jobs":-1, "random_state":42}
        scores=[]
        for tr,va in cv.split(X,y,groups):
            m=RandomForestClassifier(**params).fit(X.iloc[tr],y.iloc[tr])
            scores.append(f1_score(y.iloc[va],m.predict(X.iloc[va]),average="macro",zero_division=0))
        return sum(scores)/len(scores)
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective,n_trials=trials)
    study.trials_dataframe().to_csv("results/tables/optuna_trials.csv",index=False)
    return study.best_params, study.best_value
