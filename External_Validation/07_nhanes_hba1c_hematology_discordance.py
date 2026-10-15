#!/usr/bin/env python3
"""SCRIPT 07 | Is HbA1c discordance with fasting glucose linked to hematology?

Run at repository root:
  python External_Validation/07_nhanes_hba1c_hematology_discordance.py

Required genuine input (same file used by Scripts 05, 98):
  EDA/Results/73_frozen_phenotype_map_input.csv

DESIGN
- Estimate HbA1c from fasting glucose, log-insulin and covariates in NHANES 2005-08.
- Compare that baseline with a matched-sample model adding Hb, RBC, MCV, RDW.
- Freeze both models and evaluate 2009-18, 2021-23 and individual survey cycles.
- Report weighted mean squared error (MSE) differences and descriptive participant
  bootstrap intervals. Outcome and weights are never used to construct PCA components.
- If suitable CRP/lipid raw columns exist in the frozen source, perform a clearly
  marked optional contextual sensitivity, separately and on matched rows.

IMPORTANT LIMITATIONS
- This is NOT a causal analysis. Single fasting glucose is an imperfect reference
  for multimonth glycemia. Renal function, iron status, drugs, hemoglobin variants,
  red-cell lifespan, diabetes and differential laboratory methods may confound results.
- Survey weights are used for weighted predictions, NOT design-based inference;
  descriptive bootstrap intervals ignore NHANES PSUs/strata.
- MIDJA does NOT contain glucose in our staged files, so it CANNOT validate
  glucose--HbA1c discordance directly. No MIDJA subjects are read/exported.
- No source data or participant-level data are modified or exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'EDA' / 'Results' / '73_frozen_phenotype_map_input.csv'
OUT = ROOT / 'External_Validation' / 'Results'
SEED = 20261015
ALPHA = 20.0
NBOOT = 600
SMALL = 'SUPPRESSED_0_TO_10'

HEM = ['LBXHGB','LBXRBCSI','LBXMCVSI','LBXRDW']
BASE = ['LBXGLU','LOG_IN','GLUCOSE_SQUARED','RIDAGEYR','BMXBMI','EGFR_2021','SEX_FEMALE']
Y = 'LBXGH'
NEEDED = ['CYCLE','SURVEY_WT','ELIGIBLE_ADULT_NONPREG',Y,
          'LBXGLU','LOG_IN','RIDAGEYR','RIAGENDR','BMXBMI','EGFR_2021', *HEM]
# Optional, only if *exactly* present in Script 73 source. Do not confuse
# serum CRP with hs-CRP without codebook harmonization.
OPTIONAL = {'CRP':'LBXCRP','CREATININE':'LBXSCR', 'TOTAL_CHOLESTEROL':'LBXTC',
            'HDL':'LBDHDD','TRIGLYCERIDES':'LBXTR'}
CYCS = ['0506','0708','0910','1112','1314','1516','1718','2123']
PERIODS = {'0506':'discovery_2005_08','0708':'discovery_2005_08',
           **{c:'holdout_2009_18' for c in CYCS[2:-1]},
           '2123':'holdout_2021_23'}
BOUNDS = {
    'LBXGH':(2.5,20.0),'LBXGLU':(40,600), 'LOG_IN':(-2,7),
    'LBXHGB':(5,22), 'LBXRBCSI':(2,9), 'LBXMCVSI':(50,130), 'LBXRDW':(8,40),
    'RIDAGEYR':(18,110),'BMXBMI':(10,80),'EGFR_2021':(5,200),
}


def safe_n(n):
    return str(int(n)) if n > 10 else SMALL


def num(s):
    return pd.to_numeric(s,errors='coerce').replace([np.inf,-np.inf],np.nan)


def weighted_mse(y,p,w):
    return float(np.average((y-p)**2,weights=w))


def metrics(y,p,w):
    mean = float(np.average(y,weights=w))
    denom = float(np.average((y-mean)**2,weights=w))
    mse = weighted_mse(y,p,w)
    return dict(mse=mse,rmse=float(np.sqrt(mse)),
                r2=(1-mse/denom if denom>0 else np.nan),
                mean_error=float(np.average(p-y,weights=w)))


def paired_bootstrap(y,p0,p1,w,reps,seed):
    """Descriptive case bootstrap; NOT a design-correct NHANES interval."""
    n=len(y)
    if reps<=0 or n<=10: return (np.nan,np.nan)
    rng=np.random.default_rng(seed)
    diff=(y-p0)**2-(y-p1)**2
    out=np.empty(reps)
    for i in range(reps):
        idx=rng.integers(0,n,size=n)
        out[i]=float(np.average(diff[idx],weights=w[idx]))
    return tuple(float(v) for v in np.quantile(out,[0.025,0.975]))


def load_data(source):
    if not source.is_file():
        raise FileNotFoundError(f'Missing genuine input {source}. This script never substitutes synthetic data.')
    cols=list(pd.read_csv(source,nrows=0).columns)
    absent=[c for c in NEEDED if c not in cols]
    if absent:
        raise ValueError('Genuine frozen source is missing required fields: '+', '.join(absent))
    opt_present={name:col for name,col in OPTIONAL.items() if col in cols}
    read=list(dict.fromkeys(NEEDED+list(opt_present.values())+(['SEQN'] if 'SEQN' in cols else [])))
    d=pd.read_csv(source,usecols=read,low_memory=False)
    d['CYCLE']=(d.CYCLE.astype(str).str.strip()
                .str.replace(r'\.0$','',regex=True).str.zfill(4))
    d=d.loc[d.CYCLE.isin(CYCS)].copy()
    if 'SEQN' in d.columns:
        if d.SEQN.isna().any() or d.SEQN.duplicated().any():
            raise ValueError('Missing/duplicate SEQN; study cohort integrity failed')
    for c in read:
        if c not in ('CYCLE',): d[c]=num(d[c])
    # No PHQ outcome is required: physiology-only analysis.
    d=d.loc[d.ELIGIBLE_ADULT_NONPREG.eq(1) & d.SURVEY_WT.gt(0)].copy()
    pre_n=len(d)
    good=d.RIAGENDR.isin([1.,2.])
    for c,(low,high) in BOUNDS.items(): good &= d[c].between(low,high)
    d=d.loc[good].copy()
    # The raw physiological variables are preserved, never normalized in saved rows.
    g=d.LBXGLU
    # Squared glucose after fixed physical centering, to reduce misspecification.
    d['GLUCOSE_SQUARED']=((g-100)/100)**2
    d['SEX_FEMALE']=d.RIAGENDR.eq(2.).astype(float)
    d['PERIOD']=d.CYCLE.map(PERIODS)
    if (d['PERIOD'].isna()).any(): raise RuntimeError('Cycle not mapped')
    counts={p:int((d.PERIOD==p).sum()) for p in sorted(set(PERIODS.values()))}
    if min(counts.values())<100:
        raise ValueError(f'Too few valid real observations per period: {counts}')
    return d,opt_present,pre_n,counts


class FrozenRidge:
    def __init__(self,cols,alpha=ALPHA):
        self.cols=list(cols);self.alpha=alpha
        self.scaler=StandardScaler()
        self.model=Ridge(alpha=alpha,solver='lsqr')
    def fit(self,d):
        x=d[self.cols].to_numpy(dtype=float)
        y=d[Y].to_numpy(dtype=float)
        w=d.SURVEY_WT.to_numpy(dtype=float)
        w=w/w.mean()  # match ridge penalty scale across sample sizes
        self.scaler.fit(x,sample_weight=w)
        self.model.fit(self.scaler.transform(x),y,sample_weight=w)
        return self
    def predict(self,d):
        x=d[self.cols].to_numpy(dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError('Nonfinite features in frozen prediction')
        return self.model.predict(self.scaler.transform(x))


def discovery_cv(d):
    ids=np.arange(len(d));splitter=KFold(n_splits=5,shuffle=True,random_state=SEED)
    preds0=np.full(len(d),np.nan);preds1=np.full(len(d),np.nan)
    for train,test in splitter.split(ids):
        a,b=d.iloc[train],d.iloc[test]
        preds0[test]=FrozenRidge(BASE).fit(a).predict(b)
        preds1[test]=FrozenRidge(BASE+HEM).fit(a).predict(b)
    y=d[Y].to_numpy(float);w=d.SURVEY_WT.to_numpy(float)
    if not (np.isfinite(preds0).all() and np.isfinite(preds1).all()):
        raise RuntimeError('Cross-validation predictions incomplete')
    return metrics(y,preds0,w),metrics(y,preds1,w)


def evaluate(d,pred0,pred1,label,seed,reps):
    y=d[Y].to_numpy(float);w=d.SURVEY_WT.to_numpy(float)
    m0=metrics(y,pred0,w);m1=metrics(y,pred1,w)
    ci=paired_bootstrap(y,pred0,pred1,w,reps,seed)
    return dict(period=label,n=safe_n(len(d)),
                baseline_mse=m0['mse'],hematology_mse=m1['mse'],
                hematology_delta_mse=m0['mse']-m1['mse'],
                delta_mse_ci_low=ci[0],delta_mse_ci_high=ci[1],
                baseline_r2=m0['r2'],hematology_r2=m1['r2'],
                baseline_mean_error=m0['mean_error'],hematology_mean_error=m1['mean_error'])


def output_run(source,out,bootstrap):
    d,opt,initial,counts=load_data(source)
    train=d.loc[d.PERIOD.eq('discovery_2005_08')]
    cv0,cv1=discovery_cv(train)
    a=FrozenRidge(BASE).fit(train)
    b=FrozenRidge(BASE+HEM).fit(train)

    rows=[]
    for i,label in enumerate(['discovery_2005_08','holdout_2009_18','holdout_2021_23']+CYCS):
        sel = (d.PERIOD==label) if label.startswith(('discovery','holdout')) else (d.CYCLE==label)
        sample=d.loc[sel]
        if len(sample)<=10: continue
        # Discovery row is in-sample; discovery CV reported separately.
        row=evaluate(sample,a.predict(sample),b.predict(sample),label,SEED+i,bootstrap)
        row['is_training_sample']=int(label=='discovery_2005_08' or label in ('0506','0708'))
        rows.append(row)
        if label in ('discovery_2005_08','holdout_2009_18','holdout_2021_23'):
            print(f'  {label:<19} | N={len(sample):>6} | frozen hematology ΔMSE={row["hematology_delta_mse"]:+.6f}')

    # Scalar, conditional standardized hematology coefficients: descriptive associations.
    coeff=[]
    for j,c in enumerate(b.cols):
        if c in HEM:
            coeff.append(dict(variable=c,coefficient_per_training_sd=float(b.model.coef_[j]),
                              discovery_mean=float(b.scaler.mean_[j]),
                              discovery_sd=float(b.scaler.scale_[j]),
                              interpretation='Conditional association, NOT causal or mechanism'))

    # Contextual marker audit: do NOT silently merge mismatched biochemical assay files.
    checks=[];sens=[]
    for name,col in OPTIONAL.items():
        if name not in opt:
            checks.append(dict(marker=name,column=col,status='ABSENT_FROM_SCRIPT73',n_discovery=SMALL))
            continue
        if name=='CRP': valid=d[col].between(0.01,100)
        elif name=='CREATININE': valid=d[col].between(0.1,20)
        else: valid=d[col].between(1,1500)
        sub=d.loc[valid].copy()
        tr=sub.loc[sub.PERIOD.eq('discovery_2005_08')]
        va=sub.loc[sub.PERIOD.eq('holdout_2009_18')]
        modern=sub.loc[sub.PERIOD.eq('holdout_2021_23')]
        if min(len(tr),len(va),len(modern))<100:
            checks.append(dict(marker=name,column=col,status='PRESENT_BUT_INSUFFICIENT_COVERAGE',n_discovery=safe_n(len(tr))))
            continue
        checks.append(dict(marker=name,column=col,status='EXPLORATORY_NOT_CODEBOOK_VERIFIED',n_discovery=safe_n(len(tr))))
        # Exploratory adjustment on SAME complete-case subset; only incremental H effect.
        # This is not a fair comparison against primary cohort due to reduced sample.
        c0=FrozenRidge(BASE+[col]).fit(tr)
        c1=FrozenRidge(BASE+[col]+HEM).fit(tr)
        for k,p in enumerate(('holdout_2009_18','holdout_2021_23')):
            q=sub.loc[sub.PERIOD.eq(p)]
            z=evaluate(q,c0.predict(q),c1.predict(q),p,SEED+100+k,bootstrap)
            sens.append(dict(marker=name,column=col,**z,warning='EXPLORATORY; verify assay and available-cohort shift'))

    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'07_frozen_transfer_metrics.csv',index=False)
    pd.DataFrame(coeff).to_csv(out/'07_hematology_coefficients.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'07_context_marker_audit.csv',index=False)
    # File present with a clear skip status even when no optional variables available.
    if sens:
        pd.DataFrame(sens).to_csv(out/'07_context_sensitivity.csv',index=False)
    else:
        pd.DataFrame([{'status':'NOT_RUN: no optional contextual assay with all-period coverage'}]).to_csv(out/'07_context_sensitivity.csv',index=False)

    main=[r for r in rows if r['period'].startswith(('discovery','holdout'))]
    lines=[
        '# Script 07 — Hematology-associated HbA1c/glucose discordance', '',
        '**Question:** Do Hb/RBC/MCV/RDW add transferable information for measured HbA1c after conditioning on fasting glucose, log-insulin and basic covariates?',
        '',
        '**Not causal identification.** Residual HbA1c after fasting-glucose adjustment is a proxy for *discordance*, not verified glucose-independent HbA1c bias.',
        '', '## Locked experiment', '',
        '- Discovery: NHANES 2005–08; outcomes not used to choose variables.',
        '- Baseline: fasting glucose, centered glucose squared, log-insulin, age, sex, BMI and estimated GFR.',
        '- Augmented: same variables plus Hb, RBC count, MCV and RDW.',
        f'- Fixed Ridge alpha={ALPHA:g}; weighted fitting; five-fold discovery-only CV; no holdout refitting.',
        '- All comparisons are matched-complete-case and use frozen source `73_frozen_phenotype_map_input.csv`.',
        '', '## Discovery five-fold CV (within same participants)', '',
        '| Model | Weighted CV MSE | Weighted CV R² |', '|---|---:|---:|',
        f'| Baseline | {cv0["mse"]:.6f} | {cv0["r2"]:.5f} |',
        f'| + Hematology | {cv1["mse"]:.6f} | {cv1["r2"]:.5f} |',
        '', '## Frozen temporal tests', '',
        'Positive ΔMSE favors adding hematology. 95% intervals are **descriptive case bootstraps only**, not NHANES survey-design inferential intervals.',
        '', '| Period | Complete N | Baseline MSE | +H MSE | ΔMSE | Descriptive 95% CI |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for r in main:
        lines.append(f'| {r["period"]} | {r["n"]} | {r["baseline_mse"]:.5f} | {r["hematology_mse"]:.5f} | {r["hematology_delta_mse"]:+.5f} | [{r["delta_mse_ci_low"]:+.5f}, {r["delta_mse_ci_high"]:+.5f}] |')
    lines += ['', '## Contextual expansion', '',
              'Optional labs are evaluated only when the **exact source columns** exist with sufficient data in ALL periods. Otherwise the experiment is explicitly skipped.', '']
    for r in checks:
        lines.append(f'- {r["marker"]} ({r["column"]}): {r["status"]}')
    lines += ['', '## Critical limitations', '',
              '1. **Fasting glucose is not mean glucose over the preceding 2–3 months**; HbA1c discrepancy cannot be attributed to RBC lifespan or an HbA1c assay effect on this evidence alone.',
              '2. Associations can arise from iron deficiency, kidney disease, medications, diabetes, assay drift, inflammation, age and selection; physiology is observational.',
              '3. Survey-weighted point metrics and naive participant-bootstrap intervals **do not** account for PSU/stratum sampling. Use design-based methods before publication-level inference.',
              '4. Age, eGFR and BMI adjustment cannot remove all confounding. Hematological indices are correlated and non-interventional.',
              '5. The source file may be a selected analytical extract rather than the full fasting NHANES population; sample restriction needs external audit.',
              '6. MIDJA in this batch has HbA1c/CBC but no matched fasting glucose; it cannot confirm glucose–HbA1c discordance. Unit and assay documentation remains outstanding.',
              '7. This is a new diagnostic study, not a reconstruction of the original frozen 7D NHANES physiological decomposition.',
              '', '## Outputs', '',
              '- `07_frozen_transfer_metrics.csv` (including per-cycle diagnostics)',
              '- `07_hematology_coefficients.csv` (conditional, ridge-shrunk associations)',
              '- `07_context_marker_audit.csv`, `07_context_sensitivity.csv`',
              '- `07_manifest.json`', '',
              '**Scientific gate:** If the frozen hematology increment fails in later periods, report non-transfer and reassess assays/phenotype before adding model complexity.'
              ]
    (out/'07_readout.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest={'script':'07_nhanes_hba1c_hematology_discordance.py',
              'input':str(source),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'discovery_period':'0506,0708','holdouts':['0910','1112','1314','1516','1718','2123'],
              'eligible_before_complete_screen':initial, 'complete_by_period':counts,
              'seed':SEED,'fixed_alpha':ALPHA,'bootstrap_reps':bootstrap,
              'target':Y,'baseline_features':BASE,'hematology_features':HEM,
              'causal_claim':False,'complex_survey_design_se_accounted_for':False,
              'midja_direct_discordance_test':False,'patient_records_exported':False}
    (out/'07_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('SCRIPT 07 | frozen NHANES HbA1c/glucose conditional hematology audit | complete')
    print('OUTPUT:',out/'07_readout.md')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=SOURCE)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--bootstrap',type=int,default=NBOOT)
    args=p.parse_args()
    if not (0<=args.bootstrap<=5000):
        p.error('--bootstrap must be in 0..5000')
    output_run(args.input,args.out,args.bootstrap)

if __name__=='__main__': main()
