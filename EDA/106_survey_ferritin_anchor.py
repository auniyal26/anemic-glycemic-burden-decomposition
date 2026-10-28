#!/usr/bin/env python3
"""106 | Survey-aware rediscovery + independent ferritin anchoring (EXPLORATORY).

Place this file beside EDA/103_jive_ajive_benchmark.py and
EDA/104_representation_validation_gate.py in the original NHANES repository.

From repository root:
    python EDA/106_survey_ferritin_anchor.py --self-test
    python EDA/106_survey_ferritin_anchor.py --fetch-missing --replicates 199

This script downloads ONLY public NHANES DEMO/GLU/FERTIN XPTs when the explicit
--fetch-missing flag is used. Never downloads or creates patient-level data.

Study design: independent, outcome-blind H/G CCA refits in 2005-06, 2007-08,
and 2021-23; 2005-06 discovery-only supervised ferritin prediction, tested
frozen in 2007-08 and 2021-23 for women aged 20-49. Serum ferritin (LBXFER)
is not among the seven physiological inputs. PHQ-9 is NOT an outcome here.

NHANES masked SDMVSTRA/SDMVPSU and fasting-subsample WTSAF2YR are recovered
from the original public data, not guessed. Repeated PSU-level rescaled
bootstrap (m_h-1 with replacement per design stratum, factor m_h/(m_h-1))
quantifies structural rediscovery and held-out anchor-prediction differences.
Intervals are approximate design-aware resampling diagnostics, NOT a claim of
nominal 95% design-based confidence coverage; additional NCHS review needed.
Do NOT interpret as identified pathways, causal associations, or independent
out-of-survey validation. The later NHANES periods were previously examined.

2021-23 has changed design, laboratory context and phlebotomy weights; fasting
WTSAF2YR is used for all analyses involving fasting glucose and insulin.
Ferritin methods and target populations differ; 2021-23 transport is an
assay-sensitive exploratory test, not an assay-harmonised validation.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DEFAULT = HERE / 'Results' / '106_survey_ferritin_anchor'
SEED = 106
CYCLES = {'0506':('2005','D','NHANES'), '0708':('2007','E','NHANES'),
          '2123':('2021','L','NHANES_Transfer')}
GROUPS = {'Shared2':[0,1], 'Discordant2':[2,3],
          'H_Isolated2':[4,5], 'G_Isolated1':[6]}
BASE_COV = ['RIDAGEYR','BMXBMI','INDFMPIR']


def load_module(filename, name):
    p=HERE/filename
    if not p.exists(): raise FileNotFoundError(f'Missing required script: {p}')
    spec=importlib.util.spec_from_file_location(name,p)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def get_xpt(cycle, kind, fetch):
    year,suffix,rootpart=CYCLES[cycle]
    local=ROOT/'Data'/rootpart/cycle/f'{kind}_{suffix}.XPT'
    for path in [local,local.with_suffix('.xpt')]:
        if path.is_file():return path
    if not fetch:
        raise FileNotFoundError(f'Public NHANES input missing: {local}\n'
                                'Rerun with --fetch-missing to download official XPTs.')
    url=f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{kind}_{suffix}.XPT'
    local.parent.mkdir(parents=True,exist_ok=True)
    print('Fetching public NHANES file:',url, flush=True)
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'NHANES-research/1.0'})
        with urllib.request.urlopen(request,timeout=70) as response,local.open('wb') as writer:
            while True:
                block=response.read(1<<20)
                if not block: break
                writer.write(block)
        return local
    except Exception as err:
        local.unlink(missing_ok=True)
        raise RuntimeError(f'Could not fetch {url}: {err}') from err


def read_xpt(cycle,kind,fetch,cols):
    p=get_xpt(cycle,kind,fetch)
    q=pd.read_sas(p,format='xport')
    missing=set(cols)-set(q.columns)
    if missing:raise RuntimeError(f'{p}: missing columns {sorted(missing)}')
    q=q[list(cols)].copy()
    for c in cols:q[c]=pd.to_numeric(q[c],errors='coerce')
    if q.SEQN.isna().any() or q.SEQN.duplicated().any():
        raise RuntimeError(f'Duplicate/invalid SEQN in {p}')
    return q


def read_inputs(fetch):
    s103=load_module('103_jive_ajive_benchmark.py','s103_in_106')
    s104=load_module('104_representation_validation_gate.py','s104_in_106')
    d,fullcov=s103.load_matched()
    d=d.copy();d['CYCLE']=d['CYCLE'].astype(str).str.zfill(4)
    fullcov_cols=[f'_COV_FULL_{i}' for i in range(fullcov.shape[1])]
    for i,c in enumerate(fullcov_cols):d[c]=fullcov[:,i]
    cohorts={};frames={};weight_audit=[]
    for cy in CYCLES:
        observed=d[d.CYCLE==cy].copy()
        if len(observed)<150:raise RuntimeError(f'Insufficient matched 98 data: {cy}, n={len(observed)}')
        demo=read_xpt(cy,'DEMO',fetch,['SEQN','SDMVSTRA','SDMVPSU'])
        glu=read_xpt(cy,'GLU',fetch,['SEQN','WTSAF2YR'])
        fer=read_xpt(cy,'FERTIN',fetch,['SEQN','LBXFER'])
        frame=demo.dropna().copy()
        frame[['SDMVSTRA','SDMVPSU']]=frame[['SDMVSTRA','SDMVPSU']].astype(int)
        frames[cy]=frame
        observed=observed.merge(demo,on='SEQN',how='left',validate='one_to_one')
        observed=observed.merge(glu,on='SEQN',how='left',validate='one_to_one')
        observed=observed.merge(fer,on='SEQN',how='left',validate='one_to_one')
        for k in ['SDMVSTRA','SDMVPSU','WTSAF2YR']:
            if observed[k].isna().any():
                raise RuntimeError(f'Missing {k} in {cy}: cannot silently substitute')
        observed[['SDMVSTRA','SDMVPSU']]=observed[['SDMVSTRA','SDMVPSU']].astype(int)
        if not observed.WTSAF2YR.gt(0).all():
            raise RuntimeError(f'Nonpositive fasting weights in matched data {cy}')
        ratio=(observed.SURVEY_WT/observed.WTSAF2YR).to_numpy(float)
        # Record instead of silently treating normalized/pooled weights as equal.
        weight_audit.append({'cohort':cy,'n_matched':len(observed),
             'median_precomputed_to_WTSAF':float(np.median(ratio)),
             'max_relative_ratio_deviation':float(np.max(np.abs(ratio/np.median(ratio)-1))),
             'weight_source_used':'official_GLUXPT_WTSAF2YR'})
        for st, group in frame.groupby('SDMVSTRA'):
            if group.SDMVPSU.nunique()<2:
                raise RuntimeError(f'{cy} DEMO survey stratum {st} has <2 PSUs')
        cohorts[cy]=observed.reset_index(drop=True)
    return cohorts,frames,s103,s104,weight_audit,fullcov_cols


def rescaled_psu_weights(sample, full_frame, rng):
    """Rao-Wu-style rescaled PSU bootstrap, using *entire* DEMO PSU frame.

    Fasting/complete-case participant records absent from a PSU remain absent:
    the sampling frame is nevertheless the full masked survey design.
    """
    factor={}
    for st, sf in full_frame.groupby('SDMVSTRA'):
        psu=sorted(sf.SDMVPSU.unique().tolist())
        m=len(psu)
        if m<2:raise RuntimeError(f'Survey stratum {st} has fewer than 2 PSUs')
        picks=rng.choice(m,size=m-1,replace=True)
        counts=np.bincount(picks,minlength=m)
        for j,p in enumerate(psu):factor[(int(st),int(p))]=float(m/(m-1)*counts[j])
    scales=np.fromiter((factor[(int(st),int(p))] for st,p in
                       zip(sample.SDMVSTRA,sample.SDMVPSU)),dtype=float,count=len(sample))
    result=sample.WTSAF2YR.to_numpy(float)*scales
    if result.sum()<=0:raise RuntimeError('Empty PSU bootstrap replicates')
    return result


def wmean(a,w): return np.average(a,axis=0,weights=w)

def mse(y,yp,w):return float(np.average((np.asarray(y)-np.asarray(yp))**2,weights=w))

def r2(y,yp,w):
    v=np.average((y-wmean(y,w))**2,weights=w)
    return float(1-mse(y,yp,w)/v) if v>0 else float('nan')


def fit_ridge(a,y,w):
    # Ridge penalty fixed ex ante, not tuned using heldout ferritin.
    model=Ridge(alpha=10.0)
    positive=np.asarray(w)>0
    model.fit(np.asarray(a)[positive],np.asarray(y)[positive],
              sample_weight=np.asarray(w)[positive]/np.mean(np.asarray(w)[positive]))
    return model


def main_analysis(fetch, reps, out, seed):
    cohorts,frames,s103,s104,weight_audit,fullcov_cols=read_inputs(fetch)
    rng=np.random.default_rng(seed)
    xcols=s103.X_COLS
    scaler=StandardScaler().fit(cohorts['0506'][xcols].to_numpy(float))
    x={k:scaler.transform(df[xcols].to_numpy(float)) for k,df in cohorts.items()}
    w={k:df.WTSAF2YR.to_numpy(float) for k,df in cohorts.items()}
    W0,mu0,rho=s104.fit_hg_cca(x['0506'],w['0506'])
    fits={k:s104.fit_hg_cca(x[k],w[k]) for k in CYCLES}
    anchor=x['0506']
    z0=(anchor-mu0)@W0
    struct_rows=[];replicate_rows=[]
    for target in ('0708','2123'):
        wt,mt,_=fits[target]
        base={(group):s104.angle_stats(z0[:,ix],((anchor-mt)@wt)[:,ix])
              ['mean_squared_principal_cosine'] for group,ix in GROUPS.items()}
        draws={group:[] for group in GROUPS}
        for j in range(reps):
            bw0=rescaled_psu_weights(cohorts['0506'],frames['0506'],rng)
            bwt=rescaled_psu_weights(cohorts[target],frames[target],rng)
            rw,rm,_=s104.fit_hg_cca(x['0506'][bw0>0],bw0[bw0>0])
            tw,tm,_=s104.fit_hg_cca(x[target][bwt>0],bwt[bwt>0])
            a=(anchor-rm)@rw;b=(anchor-tm)@tw
            for group,idx in GROUPS.items():
                draws[group].append(s104.angle_stats(a[:,idx],b[:,idx])
                                    ['mean_squared_principal_cosine'])
        for group in GROUPS:
            vals=np.asarray(draws[group],float)
            struct_rows.append({'reference':'0506','target':target,'group':group,
                'observed_similarity':base[group],
                'psu_rescaled_low_2p5':float(np.percentile(vals,2.5)),
                'psu_rescaled_high_97p5':float(np.percentile(vals,97.5)),
                'bootstrap_sd':float(np.std(vals,ddof=1)),
                'bootstrap_reps':reps,'stratum_psu_design_used':True})
    # External-to-seven-markers ASSAY anchor (not external population): ferritin.
    # Restrict to the consistently eligible adult WRA subgroup across periods.
    anchor_rows=[];comparison_rows=[];matrices={};labels={};afw={}
    candidates=['Covariates','H_Isolated2','Shared2','Discordant2',
                'G_Isolated1','H4','G3','PCA2','RawX7']
    for cy in CYCLES:
        df=cohorts[cy]
        elig=(df.RIAGENDR.eq(2) & df.RIDAGEYR.ge(20) & df.RIDAGEYR.le(49)
              & pd.to_numeric(df.LBXFER,errors='coerce').gt(0))
        elig &= df[BASE_COV].notna().all(axis=1)
        q=df.loc[elig].copy()
        if len(q)<45:
            raise RuntimeError(f'{cy}: Only {len(q)} adult women with complete ferritin. '
                               'Cannot responsibly run the predefined anchoring analysis.')
        ix=np.flatnonzero(elig.to_numpy())
        X=x[cy][ix]
        Z=(X-mu0)@W0  # fully frozen 0506 CCA; no later refitting.
        cov=q[BASE_COV].to_numpy(float)
        matrices[cy]={'Covariates':cov,'H_Isolated2':np.c_[cov,Z[:,4:6]],
           'Shared2':np.c_[cov,Z[:,:2]],'Discordant2':np.c_[cov,Z[:,2:4]],
           'G_Isolated1':np.c_[cov,Z[:,6]],'H4':np.c_[cov,X[:,:4]],
           'G3':np.c_[cov,X[:,4:]],'PCA2':None,'RawX7':np.c_[cov,X]}
        labels[cy]=np.log(q.LBXFER.to_numpy(float))
        afw[cy]=q.WTSAF2YR.to_numpy(float)
        anchor_rows.append({'cohort':cy,'eligible_women_20_49':len(q),
            'weighted_mean_log_ferritin':float(wmean(labels[cy],afw[cy])),
            'effective_weighted_n':float(afw[cy].sum()**2/np.sum(afw[cy]**2)),
            'represented_masked_psus':int(q[['SDMVSTRA','SDMVPSU']].drop_duplicates().shape[0]),
            'subgroup':'women 20-49, fasting complete-case with ferritin'})
        cohorts[cy+'_anchor']=q
    # PCA fits on 0506 full X, not on the ferritin outcome.
    from sklearn.decomposition import PCA
    pca=PCA(n_components=2,svd_solver='full').fit(x['0506'])
    for cy in CYCLES:
        q=cohorts[cy+'_anchor']
        # Preserve association between q and rows by matching key SEQN.
        loc=cohorts[cy].SEQN.astype(int).reset_index(drop=True)
        inds=loc.index[loc.isin(q.SEQN.astype(int))].to_numpy()
        X=x[cy][inds]
        matrices[cy]['PCA2']=np.c_[q[BASE_COV].to_numpy(float),pca.transform(X)]
    train='0506'
    # Fit standardizers on anchor-labelled TRAIN only; never look at later ferritin.
    fitted={};preds={name:{} for name in candidates}
    for name in candidates:
        std=StandardScaler().fit(matrices[train][name])
        xx={cy:std.transform(matrices[cy][name]) for cy in CYCLES}
        model=fit_ridge(xx[train],labels[train],afw[train])
        fitted[name]=(std,model)
        for cy in CYCLES: preds[name][cy]=model.predict(xx[cy])
    for cy in CYCLES:
        yy=labels[cy]; ww=afw[cy];
        baseline=mse(yy,preds['Covariates'][cy],ww)
        for name in candidates:
            point=mse(yy,preds[name][cy],ww)
            comparison_rows.append({'cohort':cy,'method':name,'n':len(yy),
                'weighted_log_ferritin_mse':point,
                'delta_mse_vs_covariates':float(baseline-point),
                'weighted_r2':r2(yy,preds[name][cy],ww),
                'frozen_0506_model':True,'ferritin_used_in_representation':False})
    anchor_uncertainty=[]
    for cy in ('0708','2123'):
        df=cohorts[cy+'_anchor']
        y=labels[cy];dw={n:(y-preds[n][cy])**2 for n in candidates}
        point={n:float(np.average(dw['Covariates']-dw[n],weights=afw[cy]))
               for n in candidates[1:]}
        draws={n:[] for n in candidates[1:]}
        for k in range(reps):
            bw=rescaled_psu_weights(df,frames[cy],rng)
            for n in draws:
                draws[n].append(float(np.average(dw['Covariates']-dw[n],weights=bw)))
        for n,ds in draws.items():
            vals=np.asarray(ds,float)
            anchor_uncertainty.append({'cohort':cy,'method':n,
                'delta_mse_vs_covariates':point[n],
                'psu_rescaled_low_2p5':float(np.percentile(vals,2.5)),
                'psu_rescaled_high_97p5':float(np.percentile(vals,97.5)),
                'bootstrap_sd':float(np.std(vals,ddof=1)),
                'bootstrap_reps':reps,
                'note':'Holdout PSU uncertainty only; frozen discovery fit fixed'})
    # Survey-aware PHQ transfer: 0506-only ridge on full-covariate encoding
    # inherited from Script 103/98. Later PHQ data never used for fitting.
    # Test-cohort resampling uncertainty conditions on the frozen training model.
    phq_preds={}
    phq_rows=[];phq_uncertainty=[]
    for method in ('Covariates','RawX7','H_G_CCA7'):
        f0=cohorts['0506'][fullcov_cols].to_numpy(float)
        train_x={'Covariates':f0,'RawX7':np.c_[f0,x['0506']],
                 'H_G_CCA7':np.c_[f0,(x['0506']-mu0)@W0]}[method]
        std=StandardScaler().fit(train_x)
        y0=cohorts['0506']['PHQ9_TOTAL'].to_numpy(float)
        model=fit_ridge(std.transform(train_x),y0,w['0506'])
        for cy in CYCLES:
            fc=cohorts[cy][fullcov_cols].to_numpy(float)
            f={'Covariates':fc,'RawX7':np.c_[fc,x[cy]],
               'H_G_CCA7':np.c_[fc,(x[cy]-mu0)@W0]}[method]
            yp=model.predict(std.transform(f))
            yy=cohorts[cy]['PHQ9_TOTAL'].to_numpy(float)
            phq_preds[(cy,method)]=yp
            phq_rows.append({'cohort':cy,'method':method,'n':len(yy),
               'frozen_phq_weighted_r2':r2(yy,yp,w[cy]),
               'frozen_phq_mse':mse(yy,yp,w[cy]),
               'model_fit_only_0506':True})
    for cy in ('0708','2123'):
        yy=cohorts[cy]['PHQ9_TOTAL'].to_numpy(float)
        draws={n:[] for n in ('Covariates','RawX7','H_G_CCA7')}
        for j in range(reps):
            wb=rescaled_psu_weights(cohorts[cy],frames[cy],rng)
            for name in draws:draws[name].append(r2(yy,phq_preds[(cy,name)],wb))
        for name,ds in draws.items():
            vals=np.asarray(ds)
            phq_uncertainty.append({'cohort':cy,'method':name,
                'point_frozen_phq_r2':r2(yy,phq_preds[(cy,name)],w[cy]),
                'psu_rescaled_low_2p5':float(np.percentile(vals,2.5)),
                'psu_rescaled_high_97p5':float(np.percentile(vals,97.5)),
                'bootstrap_reps':reps,
                'note':'Holdout PSU resampling; reference map and coefficients fixed'})

    out.mkdir(parents=True,exist_ok=True)
    for stem,table in [('106_phq_frozen_transfer',phq_rows),
        ('106_phq_psu_uncertainty',phq_uncertainty),
        ('106_weight_audit',weight_audit),
        ('106_design_rediscovery',struct_rows),('106_ferritin_coverage',anchor_rows),
        ('106_ferritin_frozen_transfer',comparison_rows),
        ('106_ferritin_psu_uncertainty',anchor_uncertainty)]:
        pd.DataFrame(table).to_csv(out/(stem+'.csv'),index=False)
    manifest={'script':Path(__file__).name,'seed':seed,'psu_bootstrap_reps':reps,
        'source':'Script98 common complete genuine NHANES + original public DEMO/GLU/FERTIN XPT',
        'periods':list(CYCLES),'physiological_reference':'New 0506-only fitted H/G CCA',
        'survey_weights':'WTSAF2YR from GLU XPT, period-specific (not pooled)',
        'variance':'Rao-Wu-style rescaled PSU bootstrap; approximate sensitivity, not formal CI',
        'ferritin_eligibility':'women 20-49, LBXFER>0, fasting marker complete cases',
        'ferritin_reference_assay':'LBXFER not in seven input markers',
        'ferritin_model':'Ridge alpha=10 fixed, trained 0506 only, log ferritin',
        'phq_model':'Full covariate encoding from Script103; ridge alpha=10 frozen from 0506',
        'all_cohorts_previously_inspected':True,'independent_external_dataset':False,
        'causal_or_clinical_identification':False,'assay_harmonized':False,
        'full_phq_sampling_nonresponse_addressed':False,
        'covariate_category_encoding_inherited_from_Script98_200508':True}
    (out/'106_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    ss=pd.DataFrame(struct_rows).round(4)
    ff=pd.DataFrame(comparison_rows).pivot(index='method',columns='cohort',values='delta_mse_vs_covariates').round(4)
    uu=pd.DataFrame(anchor_uncertainty).round(4)
    pp=pd.DataFrame(phq_uncertainty).round(4)
    txt=['# Script 106 — Survey-aware structural uncertainty and ferritin anchoring','',
      '## Reproducibility: shared and discordant subspaces','',
      'Full NHANES masked PSU/stratum frame, fasting subsample weights, and '+
      'rescaled PSU bootstrap; intervals approximate and exploratory.','',
      '```text',ss.to_string(index=False),'```','',
      '## Ferritin anchor: external assay, internal NHANES dataset','',
      'Frozen 0506-only ridge predictor (not a ferritin-trained physiological decomposition).',
      'Positive delta MSE indicates benefit over covariates-only on log ferritin.','',
      '```text',ff.to_string(),'```','',
      '## PSU resampling of ferritin transfer (heldout uncertainty only)','',
      '```text',uu.to_string(index=False),'```','',
      '## Frozen PHQ transfer with masked PSU resampling','',
      '```text',pp.to_string(index=False),'```','',
      '## Claim boundaries','',
      '- Ferritin is an independent laboratory anchor, but the same NHANES survey system; NOT external-population validation.',
      '- Early ferritin sampling targets primarily girls/women and children; adult comparison is women 20-49.',
      '- Ferritin reflects iron storage and inflammation; a link is not unique biological source identification.',
      '- Model transfer across 2021-23 is sensitive to assay change and population/weighting differences.',
      '- Full-rank RawX7 and CCA7 contain the same linear information. CCA gains in ridge are regularization/conditioning dependent.',
      '- Complete-case restriction and earlier NHANES inspection persist; results exploratory and not clinical validation.',
      '- Rescaled PSU intervals are approximate; survey methodologist should review NCHS weights and variance approach.',
      '- The 0506 training and 0708 test inherit earlier 2005-08 categorical encoding from Script98; 0708 is NOT fully untouched preprocessing.',
      '- An independent *population* with seven harmonized biomarkers remains an unmet publication requirement for strong transport claims.']
    (out/'106_readout.md').write_text('\n'.join(txt)+'\n',encoding='utf8')
    print('Wrote:',out)
    print('Read: 106_readout.md, 106_design_rediscovery.csv, 106_ferritin_frozen_transfer.csv, 106_phq_psu_uncertainty.csv')


def self_test():
    rng=np.random.default_rng(SEED)
    full=pd.DataFrame({'SDMVSTRA':np.repeat([1,2],200),
                       'SDMVPSU':np.tile(np.repeat([1,2],100),2)})
    small=full.copy();small['WTSAF2YR']=np.full(len(small),2.,float)
    a=rescaled_psu_weights(small,full,rng)
    assert a.min()==0 and a.max()==4 and a.sum()>0
    assert abs(np.mean([rescaled_psu_weights(small,full,rng).sum()
                       for k in range(200)])-small.WTSAF2YR.sum())<80
    y=rng.normal(size=400);yp=y+.1
    assert mse(y,yp,np.ones(400))>0
    assert r2(y,y,np.ones(400))>.999999
    print('PASS: masked-PSU rescaled bootstrap, weight expectation, weighted loss metrics')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--fetch-missing',action='store_true',help='Download missing official NHANES DEMO/GLU/FERTIN XPTs')
    p.add_argument('--replicates',type=int,default=199)
    p.add_argument('--out',type=Path,default=OUT_DEFAULT)
    p.add_argument('--seed',type=int,default=SEED)
    args=p.parse_args()
    if args.self_test:return self_test()
    if args.replicates<39:p.error('Use at least 39 PSU-bootstrap replicates')
    main_analysis(args.fetch_missing,args.replicates,args.out,args.seed)

if __name__=='__main__': main()
