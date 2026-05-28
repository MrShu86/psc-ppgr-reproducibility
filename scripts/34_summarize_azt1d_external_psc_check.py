import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--psc_dir',default='outputs_azt1d/psc_external_check'); ap.add_argument('--output_dir',default='outputs_azt1d/psc_external_check/figures')
    a=ap.parse_args(); pdir=Path(a.psc_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(pdir/'azt1d_external_psc_summary.csv')
    tests=pd.read_csv(pdir/'azt1d_external_psc_paired_tests.csv') if (pdir/'azt1d_external_psc_paired_tests.csv').exists() else pd.DataFrame()
    pred=pd.read_csv(pdir/'azt1d_external_psc_predictions.csv') if (pdir/'azt1d_external_psc_predictions.csv').exists() else pd.DataFrame()
    plt.figure(figsize=(9,5))
    for (model,pers),sub in summary.groupby(['model','personalization']):
        if pers in ['global_0shot','global_no_update','support_residual_calibration','support_affine_calibration']:
            sub=sub.sort_values('shot'); plt.plot(sub.shot,sub.RMSE_mean,marker='o',label=f'{model}/{pers.replace("_"," ")}')
    plt.xlabel('Number of support meal events'); plt.ylabel('Subject-level mean RMSE'); plt.title('AZT1D external PSC sanity check: RMSE shot curves'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/'fig_azt1d_external_psc_shot_curve_rmse.png',dpi=300); plt.close()
    plt.figure(figsize=(9,5))
    for (model,pers),sub in summary.groupby(['model','personalization']):
        if pers in ['global_0shot','global_no_update','support_residual_calibration']:
            sub=sub.sort_values('shot'); plt.plot(sub.shot,sub.bias_mean,marker='o',label=f'{model}/{pers.replace("_"," ")}')
    plt.axhline(0,linestyle='--'); plt.xlabel('Number of support meal events'); plt.ylabel('Mean bias'); plt.title('AZT1D external PSC sanity check: bias'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/'fig_azt1d_external_psc_bias_curve.png',dpi=300); plt.close()
    sub5=summary[(summary.shot.eq(5)) & (summary.personalization.isin(['global_no_update','support_residual_calibration']))]
    if not sub5.empty:
        pivot=sub5.pivot_table(index='model',columns='personalization',values='RMSE_mean',aggfunc='min').reset_index()
        if 'global_no_update' in pivot and 'support_residual_calibration' in pivot:
            pivot['RMSE_reduction_%']=(pivot.global_no_update-pivot.support_residual_calibration)/pivot.global_no_update*100; pivot.to_csv(pdir/'azt1d_external_psc_5shot_reduction_table.csv',index=False)
            x=np.arange(len(pivot)); w=.38; plt.figure(figsize=(7,4)); plt.bar(x-w/2,pivot.global_no_update,w,label='No update'); plt.bar(x+w/2,pivot.support_residual_calibration,w,label='5-shot residual PSC'); plt.xticks(x,pivot.model,rotation=25,ha='right'); plt.ylabel('Subject-level mean RMSE'); plt.title('AZT1D: 5-shot PSC vs no-update'); plt.legend(); plt.tight_layout(); plt.savefig(out/'fig_azt1d_external_psc_5shot_bar.png',dpi=300); plt.close()
    if not pred.empty:
        pred5=pred[pred.shot.eq(5) & pred.personalization.isin(['global_no_update','support_residual_calibration'])]
        for model,sub in pred5.groupby('model'):
            b=sub[sub.personalization.eq('global_no_update')]; aft=sub[sub.personalization.eq('support_residual_calibration')]
            if b.empty or aft.empty: continue
            plt.figure(figsize=(8,5)); plt.hist(b.y_pred-b.y_true,bins=30,alpha=.5,label='No update'); plt.hist(aft.y_pred_calibrated-aft.y_true,bins=30,alpha=.5,label='5-shot residual PSC'); plt.axvline(0,linestyle='--'); plt.xlabel('Prediction error'); plt.ylabel('Count'); plt.title(f'AZT1D residual distribution: {model}'); plt.legend(); plt.tight_layout(); plt.savefig(out/f'fig_azt1d_external_psc_residual_hist_{str(model).replace(" ","_")}.png',dpi=300); plt.close()
    print('[OK] saved figures to',out)
if __name__=='__main__': main()
