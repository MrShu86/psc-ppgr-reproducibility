import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np

TIMESTAMP=['timestamp','time','datetime','date_time','device_time','start_time','created_at','event_time','date']
SUBJECT=['subject_id','subject','patient_id','patient','participant_id','participant','person_id','user_id']
GLUCOSE=['glucose','cgm','sensor_glucose','sensor glucose','sgv','bg','blood_glucose','glucose_value']
CARB=['carbs','carb','carbohydrate','carbohydrates','cho','meal_carbs','carb_input','grams_carbs']
INSULIN=['insulin','bolus','dose','units','insulin_units','total_dose','correction','meal_bolus']
MODE=['mode','device_mode','activity','state','sleep','exercise','automation_mode']

def norm(x): return str(x).strip().lower().replace('-','_')
def find_col(cols,cands):
    ncols=[(norm(c),c) for c in cols]
    for k in cands:
        nk=norm(k)
        for nc,c in ncols:
            if nk==nc: return c
    for k in cands:
        nk=norm(k)
        for nc,c in ncols:
            if nk in nc: return c
    return None

def load(p):
    s=p.suffix.lower()
    if s=='.csv': return pd.read_csv(p,low_memory=False)
    if s in ['.tsv','.txt']:
        try: return pd.read_csv(p,sep='\t',low_memory=False)
        except Exception: return pd.read_csv(p,low_memory=False)
    if s=='.parquet': return pd.read_parquet(p)
    if s in ['.xlsx','.xls']: return pd.read_excel(p)
    raise ValueError(s)

def subject_from_path(p):
    text=str(p)
    for pat in [r'(?:subject|patient|participant|person|user|pt|subj)[_\-\s]*([A-Za-z0-9]+)',r'([Pp]\d+)',r'(\d{2,})']:
        m=re.search(pat,text)
        if m: return m.group(1)
    return p.stem

def files(root):
    out=[]
    for ext in ['*.csv','*.tsv','*.txt','*.parquet','*.xlsx','*.xls']: out += list(Path(root).rglob(ext))
    return sorted(out)

def extract(root, carb_threshold):
    cgm=[]; carbs=[]; insulin=[]; mode=[]; logs=[]
    for p in files(root):
        try:
            df=load(p)
            if df.empty: continue
            cols=list(df.columns); tcol=find_col(cols,TIMESTAMP)
            if tcol is None: logs.append((str(p),'no_time_col')); continue
            scol=find_col(cols,SUBJECT); gcol=find_col(cols,GLUCOSE); ccol=find_col(cols,CARB); icol=find_col(cols,INSULIN); mcol=find_col(cols,MODE)
            sid=df[scol].astype(str) if scol else pd.Series([subject_from_path(p)]*len(df),index=df.index)
            ts=pd.to_datetime(df[tcol],errors='coerce')
            if gcol:
                x=pd.DataFrame({'subject_id':sid,'timestamp':ts,'glucose':pd.to_numeric(df[gcol],errors='coerce'),'source_file':str(p)})
                x=x.dropna(subset=['timestamp','glucose']); x=x[(x.glucose>=30)&(x.glucose<=500)]
                if len(x): cgm.append(x)
            if ccol:
                x=pd.DataFrame({'subject_id':sid,'meal_time':ts,'carbs':pd.to_numeric(df[ccol],errors='coerce'),'source_file':str(p)})
                x=x.dropna(subset=['meal_time','carbs']); x=x[x.carbs>=carb_threshold]
                if len(x): carbs.append(x)
            if not ccol:
                ecols=[c for c in cols if any(k in norm(c) for k in ['event','type','description','name'])]
                vcols=[c for c in cols if norm(c) in ['value','amount','grams','quantity']]
                if ecols and vcols:
                    mask=df[ecols[0]].astype(str).str.lower().str.contains('carb|meal|food|cho',regex=True,na=False)
                    x=pd.DataFrame({'subject_id':sid[mask],'meal_time':ts[mask],'carbs':pd.to_numeric(df.loc[mask,vcols[0]],errors='coerce'),'source_file':str(p)})
                    x=x.dropna(subset=['meal_time','carbs']); x=x[x.carbs>=carb_threshold]
                    if len(x): carbs.append(x)
            if icol:
                x=pd.DataFrame({'subject_id':sid,'timestamp':ts,'insulin':pd.to_numeric(df[icol],errors='coerce'),'source_file':str(p)})
                x=x.dropna(subset=['timestamp','insulin']); x=x[(x.insulin>0)&(x.insulin<100)]
                if len(x): insulin.append(x)
            if mcol:
                x=pd.DataFrame({'subject_id':sid,'timestamp':ts,'mode':df[mcol].astype(str),'source_file':str(p)}).dropna(subset=['timestamp'])
                if len(x): mode.append(x)
        except Exception as e: logs.append((str(p),f'error:{e}'))
    return (pd.concat(cgm,ignore_index=True) if cgm else pd.DataFrame(), pd.concat(carbs,ignore_index=True) if carbs else pd.DataFrame(), pd.concat(insulin,ignore_index=True) if insulin else pd.DataFrame(), pd.concat(mode,ignore_index=True) if mode else pd.DataFrame(), pd.DataFrame(logs,columns=['file','status']))

def meal_type(h):
    if 5<=h<10: return 'breakfast'
    if 10<=h<15: return 'lunch'
    if 15<=h<21: return 'dinner'
    return 'snack'

def trapz(t,v):
    if len(t)<2: return np.nan
    o=np.argsort(t); return float(np.trapz(np.asarray(v)[o],np.asarray(t)[o]))

def nearest_mode(mode,sid,t):
    if mode.empty: return 'unknown'
    sub=mode[mode.subject_id.astype(str).eq(str(sid))]
    sub=sub[sub.timestamp<=t].sort_values('timestamp')
    if sub.empty: return 'unknown'
    if (t-sub.iloc[-1].timestamp).total_seconds()/3600>6: return 'unknown'
    return str(sub.iloc[-1].mode)

def build(cgm,carbs,insulin,mode,pre_min,post_min,min_pre,min_post,clip_positive):
    if cgm.empty or carbs.empty: raise RuntimeError('Empty CGM or carbs table. Check inspection summary and column names.')
    for d in [cgm,carbs,insulin,mode]:
        if not d.empty and 'subject_id' in d: d['subject_id']=d.subject_id.astype(str)
    cgm=cgm.sort_values(['subject_id','timestamp']); carbs=carbs.sort_values(['subject_id','meal_time']); rows=[]; eid=0
    for sid, meals in carbs.groupby('subject_id'):
        cg=cgm[cgm.subject_id.eq(sid)].sort_values('timestamp')
        if cg.empty: continue
        ins=insulin[insulin.subject_id.eq(sid)] if not insulin.empty else pd.DataFrame()
        for _,m in meals.iterrows():
            t=m.meal_time
            pre=cg[(cg.timestamp>=t-pd.Timedelta(minutes=pre_min))&(cg.timestamp<=t)]
            post=cg[(cg.timestamp>=t)&(cg.timestamp<=t+pd.Timedelta(minutes=post_min))]
            if len(pre)<min_pre or len(post)<min_post: continue
            base=float(pre.glucose.iloc[-1])
            px=(pre.timestamp-pre.timestamp.iloc[0]).dt.total_seconds().to_numpy()/60
            pv=pre.glucose.to_numpy(float); slope=float(np.polyfit(px,pv,1)[0]) if len(px)>=2 and np.std(px)>0 else 0.0
            tt=(post.timestamp-t).dt.total_seconds().to_numpy()/60; gv=post.glucose.to_numpy(float); delta=gv-base
            iauc=trapz(tt,np.maximum(delta,0) if clip_positive else delta); aucd=trapz(tt,delta)
            bol=np.nan; ins30=np.nan
            if not ins.empty:
                bol=float(ins[(ins.timestamp>=t-pd.Timedelta(minutes=15))&(ins.timestamp<=t+pd.Timedelta(minutes=15))].insulin.sum())
                ins30=float(ins[(ins.timestamp>=t-pd.Timedelta(minutes=pre_min))&(ins.timestamp<=t)].insulin.sum())
            eid+=1; h=int(t.hour)
            rows.append({'event_id':f'azt1d_evt_{eid:06d}','subject_id':sid,'meal_time':t,'carbs':float(m.carbs),'meal_type':meal_type(h),'hour':h,'hour_sin':np.sin(2*np.pi*h/24),'hour_cos':np.cos(2*np.pi*h/24),'device_mode':nearest_mode(mode,sid,t),'baseline_glucose':base,'pre_mean':float(pre.glucose.mean()),'pre_std':float(pre.glucose.std(ddof=1)) if len(pre)>1 else 0.0,'pre_min':float(pre.glucose.min()),'pre_max':float(pre.glucose.max()),'pre_last':base,'pre_slope':slope,'bolus_around_meal':bol,'insulin_pre30':ins30,'iauc_pos_2h':iauc,'auc_delta_2h':aucd,'post_mean_delta_2h':float(np.mean(delta)),'post_max_delta_2h':float(np.max(delta)),'post_min_delta_2h':float(np.min(delta)),'n_pre_points':len(pre),'n_post_points':len(post),'source_file':m.get('source_file','')})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--output_dir',default='outputs_azt1d'); ap.add_argument('--carb_threshold',type=float,default=5); ap.add_argument('--pre_min',type=int,default=30); ap.add_argument('--post_min',type=int,default=120); ap.add_argument('--min_pre_points',type=int,default=3); ap.add_argument('--min_post_points',type=int,default=12); ap.add_argument('--allow_negative_iauc',action='store_true')
    a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    cgm,carbs,ins,mode,logs=extract(a.root,a.carb_threshold)
    cgm.to_csv(out/'azt1d_extracted_cgm.csv',index=False); carbs.to_csv(out/'azt1d_extracted_carbs.csv',index=False); ins.to_csv(out/'azt1d_extracted_insulin.csv',index=False); mode.to_csv(out/'azt1d_extracted_mode.csv',index=False); logs.to_csv(out/'azt1d_extraction_logs.csv',index=False)
    print(f'[INFO] extracted: cgm={len(cgm)}, carbs={len(carbs)}, insulin={len(ins)}, mode={len(mode)}')
    ev=build(cgm,carbs,ins,mode,a.pre_min,a.post_min,a.min_pre_points,a.min_post_points,clip_positive=not a.allow_negative_iauc)
    ev.to_csv(out/'azt1d_meal_events.csv',index=False)
    summ={'n_events':len(ev),'n_subjects':ev.subject_id.nunique() if len(ev) else 0,'target':'iauc_pos_2h' if not a.allow_negative_iauc else 'auc_delta_2h','pre_min':a.pre_min,'post_min':a.post_min,'carb_threshold':a.carb_threshold}
    pd.DataFrame([summ]).to_csv(out/'azt1d_event_build_summary.csv',index=False)
    if len(ev): ev.meal_type.value_counts().rename_axis('meal_type').reset_index(name='count').to_csv(out/'azt1d_meal_type_counts.csv',index=False)
    print('[OK] saved events:',out/'azt1d_meal_events.csv'); print(pd.DataFrame([summ]).to_string(index=False))
if __name__=='__main__': main()
