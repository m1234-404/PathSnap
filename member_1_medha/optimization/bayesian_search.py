"""Optuna TPE Bayesian optimisation using GroupKFold over training trips only."""
import optuna,numpy as np,pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from member_2_dhanush.features.build_features import FEATURE_COLUMNS
from src.pathsnap.config import ARTIFACT_ROOT
import joblib

def optimise(train,trials=25):
 X=train[FEATURE_COLUMNS];y=train.is_target.to_numpy();groups=train.trip_id.to_numpy();cv=GroupKFold(n_splits=5)
 def objective(trial):
  params={'n_estimators':trial.suggest_int('n_estimators',100,400),'max_depth':trial.suggest_int('max_depth',6,32),
   'min_samples_leaf':trial.suggest_int('min_samples_leaf',1,6),'max_features':trial.suggest_float('max_features',.4,1.0),
   'class_weight':'balanced','n_jobs':-1,'random_state':42}
  scores=[]
  for a,b in cv.split(X,y,groups):
   m=RandomForestClassifier(**params).fit(X.iloc[a],y[a]);scores.append(f1_score(y[b],m.predict(X.iloc[b]),zero_division=0))
  return float(np.mean(scores))
 study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42));study.optimize(objective,n_trials=trials)
 Path('results/tables').mkdir(parents=True,exist_ok=True);study.trials_dataframe().to_csv('results/tables/optuna_trials.csv',index=False)
 best=RandomForestClassifier(**study.best_params,class_weight='balanced',n_jobs=-1,random_state=42).fit(X,y)
 (ARTIFACT_ROOT/'models').mkdir(parents=True,exist_ok=True);joblib.dump(best,ARTIFACT_ROOT/'models/final_model.joblib')
 pd.Series(study.best_params).to_json(ARTIFACT_ROOT/'config/best_params.json',indent=2)
 return study.best_params,study.best_value,best
