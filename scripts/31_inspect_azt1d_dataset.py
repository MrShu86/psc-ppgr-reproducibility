import argparse
from pathlib import Path
import pandas as pd

TIMESTAMP=['timestamp','time','datetime','date_time','device_time','start_time','created_at','event_time','date']
SUBJECT=['subject_id','subject','patient_id','patient','participant_id','participant','person_id','user_id','id']
GLUCOSE=['glucose','cgm','sensor_glucose','sensor glucose','sgv','bg','blood_glucose','blood glucose','glucose_value','mg/dl']
CARB=['carbs','carb','carbohydrate','carbohydrates','cho','meal_carbs','carb_input','grams_carbs','food_carbs']
INSULIN=['insulin','bolus','basal','dose','units','insulin_units','total_dose','correction','meal_bolus']
MODE=['mode','device_mode','activity','state','sleep','exercise','automation_mode']

def norm(x): return str(x).strip().lower().replace('-','_')
def matches(cols, cands):
    out=[]
    for c in cols:
        nc=norm(c)
        if any(norm(k)==nc or norm(k) in nc for k in cands): out.append(c)
    return list(dict.fromkeys(out))

def read_head(p,n=10):
    s=p.suffix.lower()
    if s=='.csv': return pd.read_csv(p,nrows=n)
    if s in ['.tsv','.txt']:
        try: return pd.read_csv(p,sep='\t',nrows=n)
        except Exception: return pd.read_csv(p,nrows=n)
    if s=='.parquet': return pd.read_parquet(p).head(n)
    if s in ['.xlsx','.xls']: return pd.read_excel(p,nrows=n)
    raise ValueError(s)

def role(p, cols):
    text=str(p).lower(); r=[]
    if matches(cols,GLUCOSE) or any(k in text for k in ['cgm','glucose','sgv']): r.append('cgm_like')
    if matches(cols,CARB) or any(k in text for k in ['carb','meal','food','nutrition']): r.append('carb_or_meal_like')
    if matches(cols,INSULIN) or any(k in text for k in ['insulin','bolus','basal']): r.append('insulin_like')
    if matches(cols,MODE) or any(k in text for k in ['mode','sleep','exercise']): r.append('mode_like')
    return ';'.join(r) if r else 'unknown'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--output_dir',default='outputs_azt1d/inspection')
    ap.add_argument('--max_files',type=int,default=1000)
    a=ap.parse_args(); root=Path(a.root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    files=[]
    for ext in ['*.csv','*.tsv','*.txt','*.parquet','*.xlsx','*.xls']: files += list(root.rglob(ext))
    rows=[]; samples=[]
    for p in sorted(files)[:a.max_files]:
        try:
            df=read_head(p); cols=list(df.columns)
            rows.append({'file':str(p),'relative_file':str(p.relative_to(root)),'suffix':p.suffix.lower(),'n_columns':len(cols),'columns':' | '.join(map(str,cols)),'timestamp_candidates':' | '.join(matches(cols,TIMESTAMP)),'subject_candidates':' | '.join(matches(cols,SUBJECT)),'glucose_candidates':' | '.join(matches(cols,GLUCOSE)),'carb_candidates':' | '.join(matches(cols,CARB)),'insulin_candidates':' | '.join(matches(cols,INSULIN)),'mode_candidates':' | '.join(matches(cols,MODE)),'inferred_role':role(p,cols)})
            ss=df.head(3).copy(); ss.insert(0,'__source_file__',str(p.relative_to(root))); samples.append(ss)
        except Exception as e: rows.append({'file':str(p),'relative_file':str(p.relative_to(root)),'error':str(e)})
    summ=pd.DataFrame(rows); summ.to_csv(out/'azt1d_file_inspection_summary.csv',index=False)
    if samples: pd.concat(samples,ignore_index=True,sort=False).to_csv(out/'azt1d_sample_rows.csv',index=False)
    print(f'[OK] saved {out}/azt1d_file_inspection_summary.csv')
    if 'inferred_role' in summ: print(summ['inferred_role'].value_counts(dropna=False).to_string())
if __name__=='__main__': main()
