import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def pearson(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); ok=np.isfinite(y)&np.isfinite(p)
    return np.nan if ok.sum()<3 else float(np.corrcoef(y[ok],p[ok])[0,1])
def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); ok=np.isfinite(y)&np.isfinite(p); y=y[ok]; p=p[ok]
    if len(y)==0: return {'MAE':np.nan,'RMSE':np.nan,'R2':np.nan,'Pearson':np.nan,'bias':np.nan}
    return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p)) if len(y)>1 else np.nan,'Pearson':pearson(y,p),'bias':float(np.mean(p-y))}
def get_models(seed,names):
    allm={'Ridge':Ridge(alpha=1.0),'ElasticNet':ElasticNet(alpha=0.01,l1_ratio=0.25,random_state=seed,max_iter=10000),'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,random_state=seed),'RandomForest':RandomForestRegressor(n_estimators=400,min_samples_leaf=3,random_state=seed,n_jobs=-1)}
    try:
        from xgboost import XGBRegressor
        allm['XGBoost']=XGBRegressor(n_estimators=600,max_depth=3,learning_rate=0.03,subsample=0.9,colsample_bytree=0.9,reg_lambda=1.0,objective='reg:squarederror',random_state=seed,n_jobs=-1)
    except Exception as e: print('[WARN] XGBoost unavailable:',e)
    if names==['all']: return allm
    out={m:allm[m] for m in names if m in allm}
    if not out: raise RuntimeError('No selected model available')
    return out
def preprocess(X):
    num=[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]; cat=[c for c in X.columns if c not in num]
    pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
    return pre
def residual(ps,ys,pq):
    d=0.0 if len(ys)==0 else float(np.nanmean(np.asarray(ys)-np.asarray(ps)))
    return np.asarray(pq)+d,d
def affine(ps,ys,pq):
    ps=np.asarray(ps,float); ys=np.asarray(ys,float); pq=np.asarray(pq,float); ok=np.isfinite(ps)&np.isfinite(ys)
    if ok.sum()<3 or np.nanstd(ps[ok])<1e-8:
        pred,d=residual(ps,ys,pq); return pred,np.nan,d
    a,b=np.polyfit(ps[ok],ys[ok],1); return a*pq+b,float(a),float(b)
def features(events,target):
    drop={'event_id','subject_id','meal_time','source_file','iauc_pos_2h','auc_delta_2h','post_mean_delta_2h','post_max_delta_2h','post_min_delta_2h'}
    cols=[c for c in events.columns if c not in drop]
    return events[cols].copy(), pd.to_numeric(events[target],errors='coerce'), cols
def evaluate(events,target,models,shots,min_query):
    events=events.copy(); events['meal_time']=pd.to_datetime(events.meal_time,errors='coerce')
    events=events.dropna(subset=['meal_time',target,'subject_id']).sort_values(['subject_id','meal_time'])
    X,y,_=features(events,target); pre=preprocess(X)
    subjects=sorted(events.subject_id.astype(str).unique()); detail=[]; preds=[]
    print(f'[INFO] LOSO subjects={len(subjects)}, events={len(events)}, features={X.shape[1]}')
    for mn,model in models.items():
        for sid in subjects:
            test=events.subject_id.astype(str).eq(str(sid)); train=~test
            if train.sum()<20 or test.sum()<max(shots)+min_query: continue
            pipe=Pipeline([('preprocess',pre),('model',model)]); pipe.fit(X.loc[train],y.loc[train])
            te=events.loc[test].copy().sort_values('meal_time'); te['y_true']=y.loc[test].to_numpy(float); te['y_pred']=pipe.predict(X.loc[test]); te['model']=mn
            for k in shots:
                if len(te)<=k+min_query: continue
                sup=te.iloc[:k].copy(); q=te.iloc[k:].copy();
                if len(q)<min_query: continue
                yq=q.y_true.to_numpy(float); pq=q.y_pred.to_numpy(float)
                for pers,pred,extra in [('global_0shot' if k==0 else 'global_no_update',pq,{})]:
                    m=metrics(yq,pred); detail.append({'dataset':'AZT1D','setting':'external_loso_cross_subject','model':mn,'subject_id':sid,'shot':k,'personalization':pers,'n_support':len(sup),'n_query':len(q),**m}); qq=q.copy(); qq['dataset']='AZT1D'; qq['setting']='external_loso_cross_subject'; qq['shot']=k; qq['personalization']=pers; qq['y_pred_calibrated']=pred; preds.append(qq)
                if k>0 and len(sup)>=1:
                    ps=sup.y_pred.to_numpy(float); ys=sup.y_true.to_numpy(float)
                    pr,d=residual(ps,ys,pq); m=metrics(yq,pr); detail.append({'dataset':'AZT1D','setting':'external_loso_cross_subject','model':mn,'subject_id':sid,'shot':k,'personalization':'support_residual_calibration','n_support':len(sup),'n_query':len(q),**m}); qq=q.copy(); qq['dataset']='AZT1D'; qq['setting']='external_loso_cross_subject'; qq['shot']=k; qq['personalization']='support_residual_calibration'; qq['y_pred_calibrated']=pr; qq['support_delta']=d; preds.append(qq)
                    pa,a,b=affine(ps,ys,pq); m=metrics(yq,pa); detail.append({'dataset':'AZT1D','setting':'external_loso_cross_subject','model':mn,'subject_id':sid,'shot':k,'personalization':'support_affine_calibration','n_support':len(sup),'n_query':len(q),**m}); qq=q.copy(); qq['dataset']='AZT1D'; qq['setting']='external_loso_cross_subject'; qq['shot']=k; qq['personalization']='support_affine_calibration'; qq['y_pred_calibrated']=pa; qq['affine_a']=a; qq['affine_b']=b; preds.append(qq)
    return pd.DataFrame(detail), pd.concat(preds,ignore_index=True) if preds else pd.DataFrame()
def summarize(detail):
    rows=[]
    for keys,sub in detail.groupby(['dataset','setting','model','personalization','shot']):
        row=dict(zip(['dataset','setting','model','personalization','shot'],keys)); row['n_subjects']=sub.subject_id.nunique(); row['n_query_total']=int(sub.n_query.sum())
        for m in ['MAE','RMSE','R2','Pearson','bias']:
            row[f'{m}_mean']=sub[m].mean(); row[f'{m}_std']=sub[m].std(); row[f'{m}_median']=sub[m].median()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['model','shot','personalization'])
def paired_tests(detail):
    try: from scipy import stats
    except Exception: stats=None
    rows=[]
    for keys,sub in detail.groupby(['dataset','setting','model','shot']):
        dataset,setting,model,shot=keys
        if int(shot)==0: continue
        base=sub[sub.personalization.eq('global_no_update')]
        for pers in ['support_residual_calibration','support_affine_calibration']:
            cal=sub[sub.personalization.eq(pers)]
            if base.empty or cal.empty: continue
            mer=base[['subject_id','RMSE']].merge(cal[['subject_id','RMSE']],on='subject_id',suffixes=('_no_update','_cal'))
            if len(mer)<3: continue
            diff=mer.RMSE_no_update.to_numpy(float)-mer.RMSE_cal.to_numpy(float)
            row={'dataset':dataset,'setting':setting,'model':model,'shot':shot,'personalization':pers,'n_subjects':len(mer),'mean_delta_RMSE':float(np.mean(diff)),'median_delta_RMSE':float(np.median(diff)),'relative_RMSE_reduction_%':float((mer.RMSE_no_update.mean()-mer.RMSE_cal.mean())/mer.RMSE_no_update.mean()*100),'improvement_rate_%':float(np.mean(diff>0)*100),'wilcoxon_p':np.nan,'paired_t_p':np.nan}
            if stats is not None and np.any(diff!=0):
                try: row['wilcoxon_p']=float(stats.wilcoxon(diff,zero_method='wilcox').pvalue)
                except Exception: pass
                try: row['paired_t_p']=float(stats.ttest_1samp(diff,0).pvalue)
                except Exception: pass
            rows.append(row)
    return pd.DataFrame(rows)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--events',default='outputs_azt1d/azt1d_meal_events.csv'); ap.add_argument('--output_dir',default='outputs_azt1d/psc_external_check'); ap.add_argument('--target',default='iauc_pos_2h'); ap.add_argument('--models',default='HistGradientBoosting,XGBoost,Ridge,RandomForest'); ap.add_argument('--shots',default='0,1,3,5,10'); ap.add_argument('--min_query',type=int,default=3); ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    ev=pd.read_csv(a.events); models=get_models(a.seed,[x.strip() for x in a.models.split(',') if x.strip()]); shots=tuple(int(x) for x in a.shots.split(',') if x.strip())
    detail,pred=evaluate(ev,a.target,models,shots,a.min_query)
    if detail.empty: raise RuntimeError('No evaluation rows. Check per-subject event counts or min_query/shots.')
    summ=summarize(detail); tests=paired_tests(detail); best=summ.loc[summ.groupby('shot').RMSE_mean.idxmin()].copy()
    detail.to_csv(out/'azt1d_external_psc_subject_detail.csv',index=False); pred.to_csv(out/'azt1d_external_psc_predictions.csv',index=False); summ.to_csv(out/'azt1d_external_psc_summary.csv',index=False); tests.to_csv(out/'azt1d_external_psc_paired_tests.csv',index=False); best.to_csv(out/'azt1d_external_psc_best_by_shot.csv',index=False)
    print('[OK] saved outputs to',out); print(summ[['model','personalization','shot','n_subjects','RMSE_mean','MAE_mean','bias_mean','Pearson_mean']].to_string(index=False))
if __name__=='__main__': main()
