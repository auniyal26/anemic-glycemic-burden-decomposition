#!/usr/bin/env python3
"""Design-aware paired evaluation of FROZEN candidate representations.

This requires subject-level out-of-period predictions and reconstructions
exported by the existing research pipeline. It DOES NOT fit models.
It assumes sample weights, biomarkers and methods have been frozen without
using later-period outcomes. Resulting percentile intervals condition on the
trained models; they do not include training-sample or model-selection error.

Full design-frame CSV (required for real data):
  period,cycle,SDMVSTRA,SDMVPSU
Include EVERY masked PSU/stratum from the survey frame, even when no
complete-case participant contributes a model score. These are public NHANES
design identifiers, not participant-level clinical data.

Input prediction long CSV (required):
  period,cycle,SEQN,SDMVSTRA,SDMVPSU,weight,method,rank,outcome,y,yhat
Optional physiology-reconstruction long CSV:
  period,cycle,SEQN,SDMVSTRA,SDMVPSU,weight,method,rank,squared_error,squared_reference
Here squared_error is the sum of squared reconstruction errors in discovery-
standardised biomarker units; squared_reference is the corresponding sum of
squares about the FROZEN DISCOVERY mean, common across candidate methods.
Each method must score exactly the same cohort for a given period/outcome.

Example:
 python tools/111_matched_psu_benchmark.py --predictions results/prediction_long.csv \
   --reconstruction results/reconstruction_long.csv --design-frame results/full_design.csv --out results/111 --reps 999
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

METHODS=('PCA5','FrozenZ5_trainX','JIVE5_rjive','AJIVE5_pyjive')
PERIODS=('test_2009-2018','validate_2021-2023')
DIMENSION=5
KEY=('period','cycle','SEQN','SDMVSTRA','SDMVPSU')
DESIGN=('cycle','SDMVSTRA')

def _validate(df,kind):
    shared=set(KEY)|{'weight','method','rank'}
    required= shared|({'outcome','y','yhat'} if kind=='pred' else {'squared_error','squared_reference'})
    miss=required-set(df.columns)
    if miss: raise ValueError(f'{kind}: missing columns {sorted(miss)}')
    df=df.copy()
    df=df[df['period'].isin(PERIODS)].copy()
    if set(df.method)!=set(METHODS):raise ValueError(f'Expected only {METHODS}, got {sorted(set(df.method))}')
    if (pd.to_numeric(df['rank'])!=DIMENSION).any():raise ValueError('Unmatched model rank')
    for name in ['weight','y','yhat'] if kind=='pred' else ['weight','squared_error','squared_reference']:
        df[name]=pd.to_numeric(df[name],errors='raise')
        if not np.isfinite(df[name]).all():raise ValueError(f'Nonfinite {name}')
    if (df.weight<=0).any():raise ValueError('Nonpositive weights')
    if kind=='rec' and ((df.squared_error<0)|(df.squared_reference<0)).any():raise ValueError('Negative squares')
    dup=list(KEY)+(['outcome'] if kind=='pred' else [])+['method']
    if df.duplicated(dup).any():raise ValueError('Duplicate subject-method rows')
    groups=['period']+(['outcome'] if kind=='pred' else [])
    for g,sub in df.groupby(groups,dropna=False):
        reference=None
        for method in METHODS:
            x=sub[sub.method==method].sort_values(list(KEY))
            if len(x)==0:raise ValueError(f'{g}: missing {method}')
            static=x[list(KEY)+['weight']+(['y'] if kind=='pred' else ['squared_reference'])].reset_index(drop=True)
            if reference is None: reference=static
            elif not static.equals(reference):
                raise ValueError(f'{g}: subject, weight, target, or reference mismatch for {method}')
        design=reference[list(KEY)].drop_duplicates()
        if design.SEQN.duplicated().any():raise ValueError(f'{g}: SEQN duplicated across cycles')
    return df

def _psu_multiplier_table(ref,rng):
    """m_h-1 PSU Rao-Wu-style multiplicity, separately within cycle x stratum.
    Approximate conditional bootstrap diagnostic, not universally valid NCHS CI.
    """
    tbl=[]
    for (cycle,stratum),rows in ref.groupby(list(DESIGN),dropna=False):
        psus=rows.SDMVPSU.drop_duplicates().to_numpy()
        m=len(psus)
        if m<2:raise ValueError(f'Only {m} sampled PSU in {cycle}/{stratum}; cannot rescale')
        draws=rng.choice(psus,size=m-1,replace=True)
        counts=pd.Series(draws).value_counts().to_dict()
        for psu in psus:
            tbl.append((cycle,stratum,psu,counts.get(psu,0)*m/(m-1)))
    return pd.DataFrame(tbl,columns=['cycle','SDMVSTRA','SDMVPSU','factor'])

def _eval_one(sub,kind,reps,rng,full_frame):
    # One period/outcome, four representations, exact same subjects.
    baseline=sub[sub.method=='PCA5'].sort_values(list(KEY)).reset_index(drop=True)
    n=len(baseline)
    vals={}
    for method in METHODS:
        s=sub[sub.method==method].sort_values(list(KEY)).reset_index(drop=True)
        vals[method]= (s.y.to_numpy()-s.yhat.to_numpy())**2 if kind=='pred' else s.squared_error.to_numpy()
    denominator=baseline.squared_reference.to_numpy() if kind=='rec' else None
    w=baseline.weight.to_numpy(float)
    def metric(z,ww):
        if kind=='pred':return float(np.average(z,weights=ww))
        den=np.dot(ww,denominator)
        if den<=0: raise ValueError('Zero reconstruction reference sum of squares')
        return float(1-np.dot(ww,z)/den)
    point={m:metric(v,w) for m,v in vals.items()}
    diffs={m:[] for m in METHODS if m!='PCA5'}
    for j in range(reps):
        mult=_psu_multiplier_table(full_frame,rng)
        joined=baseline[list(DESIGN)+['SDMVPSU']].merge(mult,on=list(DESIGN)+['SDMVPSU'],how='left',validate='many_to_one')
        ww=w*joined.factor.to_numpy(float)
        if ww.sum()<=0:raise ValueError('No bootstrap weight')
        bootstrap={m:metric(v,ww) for m,v in vals.items()}
        for m in diffs:
            # Prediction: positive means lower MSE than PCA; reconstruction:
            # positive means higher R2 than PCA.
            diff=(bootstrap['PCA5']-bootstrap[m]) if kind=='pred' else (bootstrap[m]-bootstrap['PCA5'])
            diffs[m].append(diff)
    rows=[]
    for m in diffs:
        d=(point['PCA5']-point[m]) if kind=='pred' else point[m]-point['PCA5']
        ci=np.quantile(diffs[m],[.025,.975]);
        rows.append({'method':m,'reference':'PCA5','n':n,
                     'metric':'MSE' if kind=='pred' else 'reconstruction_R2',
                     'method_metric':point[m],'baseline_metric':point['PCA5'],
                     'paired_gain_positive_better':d,
                     'approx_psu_low':ci[0],'approx_psu_high':ci[1],
                     'direction':'supports_gain' if ci[0]>0 else ('supports_lower' if ci[1]<0 else 'uncertain'),
                     'reps':reps,'bootstrap_scope':'frozen_holdout_only'})
    return rows

def run(predictions,reconstruction,design_frame,out,reps,seed):
    if reps<19:raise ValueError('Use >=19 resamples, recommended 999+')
    out.mkdir(parents=True,exist_ok=True)
    full=pd.read_csv(design_frame)
    frame_columns=['period','cycle','SDMVSTRA','SDMVPSU']
    if set(frame_columns)-set(full.columns):raise ValueError('Missing required full design-frame columns')
    full=full[frame_columns].drop_duplicates()
    frames=[];rng=np.random.default_rng(seed)
    for kind,path in [('pred',predictions),('rec',reconstruction)]:
        if path is None:continue
        df=_validate(pd.read_csv(path),kind)
        keys=['period','outcome'] if kind=='pred' else ['period']
        for g,sub in df.groupby(keys,dropna=False):
            frame=full[full.period==g[0] if isinstance(g,tuple) else full.period==g]
            eligible=sub[list(DESIGN)+['SDMVPSU']].drop_duplicates()
            chk=eligible.merge(frame,on=list(DESIGN)+['SDMVPSU'],how='left',indicator=True)
            if (chk._merge!='both').any():raise ValueError('Design frame excludes analytic PSUs')
            stats=_eval_one(sub,kind,reps,rng,frame)
            g=g if isinstance(g,tuple) else (g,)
            for s in stats:
                s.update({'analysis':'prediction' if kind=='pred' else 'reconstruction',
                          'period':g[0],'outcome':g[1] if kind=='pred' else 'physiology'})
            frames.extend(stats)
    result=pd.DataFrame(frames)
    result.to_csv(out/'111_paired_psu_comparisons.csv',index=False)
    return result

def self_test():
    rng=np.random.default_rng(111)
    rows=[];rec=[]
    for period in PERIODS:
        for stratum in [1,2,3]:
            for psu in [1,2,3]:
                for q in range(5):
                    seq=(1 if period==PERIODS[0] else 2)*100000+stratum*1000+psu*100+q
                    y=float(rng.normal());weight=float(rng.uniform(.5,2.))
                    for method in METHODS:
                        bias={'PCA5':0.08,'FrozenZ5_trainX':0.35,'JIVE5_rjive':0.09,'AJIVE5_pyjive':0.25}[method]
                        rows.append((period,'0506' if period==PERIODS[0] else '2123',seq,stratum,psu,weight,method,5,'PHQ_total',y,y+bias))
                        err={'PCA5':0.05,'FrozenZ5_trainX':0.2,'JIVE5_rjive':0.06,'AJIVE5_pyjive':0.3}[method]
                        rec.append((period,'0506' if period==PERIODS[0] else '2123',seq,stratum,psu,weight,method,5,err,1.))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d=Path(d);p=d/'p.csv';r=d/'r.csv'
        pd.DataFrame(rows,columns=list(KEY)+['weight','method','rank','outcome','y','yhat']).to_csv(p,index=False)
        pd.DataFrame(rec,columns=list(KEY)+['weight','method','rank','squared_error','squared_reference']).to_csv(r,index=False)
        frame=d/'full.csv'
        pd.DataFrame(rows,columns=list(KEY)+['weight','method','rank','outcome','y','yhat'])[['period','cycle','SDMVSTRA','SDMVPSU']].drop_duplicates().to_csv(frame,index=False)
        out=run(p,r,frame,d/'out',29,111)
        assert len(out)==12
        assert (out[out.method=='FrozenZ5_trainX'].paired_gain_positive_better<0).all()
        print('Self-test PASS: joined same-participant data, paired differences, cycle/stratum PSU resampling, 12 rows. Synthetic data ONLY.')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',type=Path);p.add_argument('--reconstruction',type=Path)
    p.add_argument('--design-frame',type=Path,help='Complete PSU/stratum survey design frame, not just analytic complete cases')
    p.add_argument('--out',type=Path,default=Path('111_outputs'));p.add_argument('--reps',type=int,default=999)
    p.add_argument('--seed',type=int,default=111);p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test();return
    if a.predictions is None or a.design_frame is None:raise SystemExit('Provide --predictions and --design-frame (full survey PSUs) or --self-test')
    s=run(a.predictions,a.reconstruction,a.design_frame,a.out,a.reps,a.seed)
    print(s.to_string(index=False))

if __name__=='__main__':main()
