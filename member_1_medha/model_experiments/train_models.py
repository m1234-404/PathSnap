"""Candidate-ranking experiments. One row is one GPS-observation/road-arc candidate.
Target is binary: whether this candidate is the nearest reference-route arc for that GPS point.
"""
import json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier,RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score,precision_score,recall_score,log_loss,accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from member_2_dhanush.features.build_features import FEATURE_COLUMNS,build_trip_dataset
from src.pathsnap.config import DATA_ROOT,ARTIFACT_ROOT
MODELS={
 'logistic_regression':make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=1200,class_weight='balanced')),
 'random_forest':RandomForestClassifier(n_estimators=250,min_samples_leaf=2,class_weight='balanced',n_jobs=-1,random_state=42),
 'extra_trees':ExtraTreesClassifier(n_estimators=250,min_samples_leaf=2,class_weight='balanced',n_jobs=-1,random_state=42),
 'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=120,learning_rate=.08,l2_regularization=1.0,random_state=42),
 'gaussian_nb':make_pipeline(SimpleImputer(),StandardScaler(),GaussianNB())}

def collect(partition,k=12):
 rows=[]; base=DATA_ROOT/partition
 for d in sorted(x for x in base.iterdir() if x.is_dir()):
  from src.pathsnap.io import read_trip
  data=read_trip(d,include_route=True); frame=build_trip_dataset(data,k)
  if len(frame): frame['trip_id']=d.name; rows.append(frame)
 if not rows: raise RuntimeError(f'No labelled candidate rows found in {base}')
 return pd.concat(rows,ignore_index=True)

def _top1(frame,proba,classes):
 pos_idx=list(classes).index(1) if 1 in classes else None
 scores=proba[:,pos_idx] if pos_idx is not None else np.zeros(len(frame))
 z=frame.copy()
 if 'trip_id' not in z: z['trip_id']='single_trip'
 z=z[['trip_id','observation_id','candidate_arc_id','target_arc_id','is_target']].copy();z['score']=scores
 z=z.sort_values('score',ascending=False).drop_duplicates(['trip_id','observation_id'])
 return z

def evaluate(model,frame):
 frame=frame.copy()
 if 'trip_id' not in frame: frame['trip_id']='single_trip'
 X=frame[FEATURE_COLUMNS]; y=frame.is_target.to_numpy(); p=model.predict_proba(X); classes=model.classes_
 top=_top1(frame,p,classes); truth=top.target_arc_id.to_numpy(); pred=top.candidate_arc_id.to_numpy()
 return {'observation_top1_accuracy':float(np.mean(pred==truth)),'observation_macro_f1':float(f1_score(truth,pred,average='macro',zero_division=0)),
         'candidate_miss_rate':float((frame.groupby(['trip_id','observation_id']).is_target.sum()==0).mean()),
         'candidate_precision':float(precision_score(y,model.predict(X),zero_division=0)),
         'candidate_recall':float(recall_score(y,model.predict(X),zero_division=0)),
         'candidate_f1':float(f1_score(y,model.predict(X),zero_division=0)),
         'candidate_log_loss':float(log_loss(y,p,labels=classes)) if len(classes)>1 else None}

def run_experiments(train,student,output_dir='results/tables'):
 out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);(ARTIFACT_ROOT/'models').mkdir(parents=True,exist_ok=True)
 X=train[FEATURE_COLUMNS]; y=train.is_target.to_numpy(); groups=train.trip_id.to_numpy(); cv=GroupKFold(n_splits=5); records=[]
 for name,base in MODELS.items():
  fold=[]; t0=time.perf_counter()
  for tr,va in cv.split(X,y,groups):
   m=clone(base);m.fit(X.iloc[tr],y[tr]); fold.append(f1_score(y[va],m.predict(X.iloc[va]),average='binary',zero_division=0))
  model=clone(base);model.fit(X,y);train_s=time.perf_counter()-t0
  t1=time.perf_counter(); metrics=evaluate(model,student);infer=time.perf_counter()-t1
  joblib.dump(model,ARTIFACT_ROOT/'models'/f'{name}.joblib')
  records.append({'model':name,'cv_candidate_f1_mean':float(np.mean(fold)),'cv_candidate_f1_std':float(np.std(fold)),**metrics,'train_seconds':train_s,'student_inference_seconds':infer})
 pd.DataFrame(records).to_csv(out/'model_comparison.csv',index=False)
 return pd.DataFrame(records)

if __name__=='__main__':
 print('Loading permitted train and student_test trips...')
 train=collect('train');student=collect('student_test');print(train.shape,student.shape)
 print(run_experiments(train,student).to_string(index=False))
