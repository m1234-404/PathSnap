import os
from pathlib import Path
import joblib,numpy as np
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from src.pathsnap.config import MODEL_PATH,DATA_ROOT
from src.pathsnap.io import read_trip
from member_2_dhanush.features.build_features import build_trip_dataset,FEATURE_COLUMNS
from member_2_dhanush.features.candidates import label_observations
from member_3_pavankumar.hmm.viterbi import decode_candidate_sequence
from member_3_pavankumar.evaluation.continuity import route_continuity
from member_4_pavithra.visualizations.folium_map import create_trip_map
app=FastAPI(title='PathSnap API',version='1.0.0')
INFERENCE_PARTITION=os.getenv('PATHSNAP_INFERENCE_PARTITION','evaluator_test')
class PredictRequest(BaseModel): trip_id:str

def load_model():
 if not MODEL_PATH.exists(): raise RuntimeError(f'Model not found: {MODEL_PATH}. Train and save final_model.joblib first.')
 return joblib.load(MODEL_PATH)
@app.get('/health')
def health(): return {'status':'ok','model_loaded':MODEL_PATH.exists(),'partition':INFERENCE_PARTITION}
@app.get('/trips')
def trips():
 root=DATA_ROOT/INFERENCE_PARTITION
 if not root.exists(): return {'trip_ids':[],'partition_available':False,'message':'Evaluator partition is not present in this local dataset.'}
 return {'trip_ids':sorted(p.name for p in root.iterdir() if p.is_dir()),'partition_available':True}
@app.post('/predict')
def predict(req:PredictRequest):
 if not req.trip_id.isdigit() or len(req.trip_id)>16: raise HTTPException(400,'Invalid trip ID')
 trip_dir=(DATA_ROOT/INFERENCE_PARTITION/req.trip_id).resolve()
 if not trip_dir.is_relative_to((DATA_ROOT/INFERENCE_PARTITION).resolve()) or not trip_dir.is_dir(): raise HTTPException(404,'Trip not found in configured inference partition')
 try:
  data=read_trip(trip_dir,include_route=False); model=load_model(); frame=build_trip_dataset(data,with_labels=False)
  if frame.empty: raise ValueError('No road candidates generated')
  prob=model.predict_proba(frame[FEATURE_COLUMNS]); classes=list(model.classes_); pos=classes.index(1) if 1 in classes else 0
  frame['score']=prob[:,pos]; steps=[]
  for oid,g in frame.groupby('observation_id',sort=True):
   g=g.sort_values('score',ascending=False).head(8); p=g.score.to_numpy(float);p=p/max(p.sum(),1e-12)
   steps.append({'arc_ids':g.candidate_arc_id.astype(int).tolist(),'probabilities':p.tolist()})
  predicted=decode_candidate_sequence(steps,data['arcs'])
  html_path=Path('results/figures')/f'prediction_{req.trip_id}.html';html_path.parent.mkdir(parents=True,exist_ok=True)
  create_trip_map(data['track'],data['nodes'],data['arcs'],predicted_arc_ids=predicted,output=str(html_path))
  # Optional server-side scoring: reference labels are read only AFTER prediction and
  # are never passed to candidate generation, classifier, decoder, or the UI as labels.
  score_data=read_trip(trip_dir,include_route=True); point_accuracy=None
  if score_data.get('route'):
   reference=label_observations(score_data['track'],score_data['nodes'],score_data['arcs'],score_data['route'])
   if len(reference)==len(predicted): point_accuracy=float(np.mean(np.asarray(predicted)==reference))
  return {'trip_id':req.trip_id,'status':'ok','gps_points':len(data['track']),'predicted_arc_ids':predicted,
          'observation_top1_accuracy':point_accuracy,'route_continuity':route_continuity(predicted,data['arcs']),'map_file':str(html_path)}
 except RuntimeError as e: raise HTTPException(503,str(e))
 except Exception as e: raise HTTPException(500,f'Prediction failed: {e}')
