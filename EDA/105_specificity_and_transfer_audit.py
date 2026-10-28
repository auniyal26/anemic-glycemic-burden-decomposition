#!/usr/bin/env python3
"""Script 105 — Falsification of representation rediscovery + phenotype transport.

Place in <repository>/EDA/ beside Scripts 98, 103, 104.

Run from repository root:
    python EDA/105_specificity_and_transfer_audit.py --self-test
    python EDA/105_specificity_and_transfer_audit.py --bootstrap 199 --null 199
    python EDA/105_specificity_and_transfer_audit.py --bootstrap 399 --null 399

THIS IS EXPLORATORY. Existing NHANES cohorts have already been explored. This
script cannot prove physiological source identity, causality, survey-design
statistical significance, or clinical utility. It specifically attempts to
falsify the claim that high subspace rediscovery is more than a trivial
statistical artefact, while separately auditing frozen phenotype transport.

Required: numpy, scipy, pandas, scikit-learn and the existing Script 98/103/104
and their genuine NHANES input data. The external raw NHANES data are not
bundled with this script. No native R JIVE/AJIVE refits here: they were
benchmarked previously (103/104); do not falsely label surrogates as native.

Important:
- Same original 7-marker set (Hb, RBC, MCV, RDW, HbA1c, glucose,
  log-insulin), selected from Script 98 common-complete cases.
- Independently fit 0506, 0708, 0918, 2123; common 0506 participant anchor.
- Refit CCA exactly as Script 104; preserve same 4H/3G partition.
- Null permutes whole glycaemic block WITHIN 2 sex x 4 age strata, retaining
  individual G-block covariance and coarse age/sex structure, but breaking
  within-stratum H/G pairing. Coarse conditioning is imperfect; 104 already
  used an unconditional null. P-values are exploratory, not design-corrected.
- Bootstrap resamples INDIVIDUALS within a cohort, not survey PSUs. Intervals
  are descriptive stability diagnostics and NOT NHANES confidence intervals.
- Outcome models use ONLY 0506 labels, fit linear Ridge using 0506 CV, and are
  applied frozen to disjoint later participants. Covariate encoding from
  Script 98/103 is defined using 2005-08 physiology/data; do not claim 0708
  is a completely untouched preprocessing holdout. Covariate values and the
  representation map are re-scaled here on 0506 only.
- PCA7, RawX7 and CCA7 are invertible linear coordinates; similar prediction
  is expected. The evidence of the diagnostic is the FAILURE of cross-period
  prediction, not a claim of higher CCA prediction accuracy.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE / 'Results' / '105_specificity_and_transfer_audit'
SEED = 105
COHORTS = {'0506':['0506'],'0708':['0708'],
           '0918':['0910','1112','1314','1516','1718'],'2123':['2123']}
GROUPS = {'Shared2':[0,1], 'Discordant2':[2,3],
          'H_Isolated2':[4,5], 'G_Isolated1':[6]}
OUTCOMES = {'PHQ_total':'PHQ9_TOTAL','Somatic':'SOMATIC_SCORE',
            'CogAff':'COGAFF_SUM'}
ALPHAS = [1.,10.,100.,1000.]


def import_s104():
    path = HERE / '104_representation_validation_gate.py'
    if not path.is_file():
        raise FileNotFoundError(f'Place script beside {path.name}: {path}')
    spec = importlib.util.spec_from_file_location('research104_s105',path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = obj
    spec.loader.exec_module(obj)
    return obj


def age_sex_strata(df):
    """Predefined broad bins; ensures age/sex alignment in conditioned null."""
    if not {'RIDAGEYR','RIAGENDR'}.issubset(df.columns):
        raise ValueError('Age and sex required; do not silently use unconditional null')
    age = df['RIDAGEYR'].to_numpy(dtype=float)
    sex = df['RIAGENDR'].to_numpy(dtype=float)
    if not np.all(np.isfinite(age)) or not np.all(np.isfinite(sex)):
        raise ValueError('Nonfinite age/sex for conditional permutation')
    return sex.astype(int)*10 + np.digitize(age,[35,50,65])


def permute_g_conditional(x, strata, rng):
    """Shuffle entire G row, never individual G markers, within age/sex cell."""
    result=x.copy()
    for label in np.unique(strata):
        ix=np.flatnonzero(strata == label)
        if len(ix)<10:
            raise ValueError(f'Stratum {label} has <10 observations; need coarser bins')
        result[ix,4:]=x[rng.permutation(ix),4:]
    return result


def weighted_r2(y, yp, w):
    w=np.asarray(w,float); y=np.asarray(y,float);yp=np.asarray(yp,float)
    denom=np.average((y-np.average(y,weights=w))**2,weights=w)
    return float(1-np.average((y-yp)**2,weights=w)/denom) if denom>0 else float('nan')


def weighted_mse(y,yp,w):
    return float(np.average((np.asarray(y)-np.asarray(yp))**2, weights=w))


def cv_ridge_alpha(x,y,w,nfold=5):
    folds=KFold(n_splits=nfold,shuffle=True,random_state=SEED)
    vals=[]
    for alpha in ALPHAS:
        losses=[]
        for tr,va in folds.split(x):
            m=Ridge(alpha=alpha).fit(x[tr],y[tr],sample_weight=w[tr]/np.mean(w[tr]))
            losses.append(weighted_mse(y[va],m.predict(x[va]),w[va]))
        vals.append((float(np.mean(losses)),alpha))
    return min(vals)


def orientation_rows(s104, base, alt, anchor, base_w=None, alt_w=None):
    wb,mb,_=s104.fit_hg_cca(base,base_w)
    wa,ma,_=s104.fit_hg_cca(alt,alt_w)
    p=(anchor-mb)@wb;q=(anchor-ma)@wa
    return {group:s104.angle_stats(p[:,idx],q[:,idx])['mean_squared_principal_cosine']
            for group,idx in GROUPS.items()}


def evaluate(d, cov, s104, out, boot, nnull, cap, seed):
    rng=np.random.default_rng(seed)
    d=d.copy(); d['CYCLE']=d.CYCLE.astype(str).str.zfill(4)
    by={};cov_by={}
    for label,cycles in COHORTS.items():
        ids=np.flatnonzero(d.CYCLE.isin(cycles).to_numpy())
        if cap and len(ids)>cap: ids=np.sort(rng.choice(ids,cap,replace=False))
        if len(ids)<150: raise ValueError(f'Insufficient real rows for {label}: {len(ids)}')
        by[label]=d.iloc[ids].reset_index(drop=True)
        cov_by[label]=cov[ids]
    # Every map uses ONE training-only measurement scaler.
    cols=s104.get_103().X_COLS
    scaler=StandardScaler().fit(by['0506'][cols].to_numpy(float))
    x={name:scaler.transform(dd[cols].to_numpy(float)) for name,dd in by.items()}
    w={name:dd.SURVEY_WT.to_numpy(float) for name,dd in by.items()}
    for name in by:
        if not np.isfinite(x[name]).all() or (w[name]<=0).any():
            raise ValueError(f'Invalid {name} data or weights')

    anchor=x['0506']
    ref_fit=s104.fit_hg_cca(anchor,w['0506'])
    wb,mb,_=ref_fit
    anchor_z=(anchor-mb)@wb
    obs_rows=[]; bootstrap_rows=[]; null_rows=[]
    for name in ['0708','0918','2123']:
        xx=x[name]; ww=w[name]
        wa,ma,_=s104.fit_hg_cca(xx,ww)
        true_z=(anchor-ma)@wa
        obs={g:s104.angle_stats(anchor_z[:,idx],true_z[:,idx])['mean_squared_principal_cosine']
             for g,idx in GROUPS.items()}
        strata=age_sex_strata(by[name]);draws={g:[] for g in GROUPS}
        for k in range(boot):
            sample=rng.integers(0,len(xx),size=len(xx))
            bsfit,bmu,_=s104.fit_hg_cca(xx[sample],ww[sample])
            zz=(anchor-bmu)@bsfit
            for g,idx in GROUPS.items():
                draws[g].append(s104.angle_stats(anchor_z[:,idx],zz[:,idx])
                                ['mean_squared_principal_cosine'])
        nulls={g:[] for g in GROUPS}
        for k in range(nnull):
            perm=permute_g_conditional(xx,strata,rng)
            wm,mu,_=s104.fit_hg_cca(perm,ww)
            zz=(anchor-mu)@wm
            for g,idx in GROUPS.items():
                nulls[g].append(s104.angle_stats(anchor_z[:,idx],zz[:,idx])
                                ['mean_squared_principal_cosine'])
        for g in GROUPS:
            point=obs[g]
            bs=np.asarray(draws[g]); nn=np.asarray(nulls[g]);
            obs_rows.append({'reference':'0506','cohort':name,'group':g,
                             'observed_principal_cosine_squared':point})
            bootstrap_rows.append({'cohort':name,'group':g,'observed_similarity':point,
                'bootstrap_low_2p5':float(np.percentile(bs,2.5)),
                'bootstrap_high_97p5':float(np.percentile(bs,97.5)),
                'bootstrap_n':boot, 'design_corrected':False})
            null_rows.append({'cohort':name,'group':g,'observed_similarity':point,
                'conditioned_null_mean':float(np.mean(nn)),
                'conditioned_null_high95':float(np.percentile(nn,95)),
                'specificity_margin_vs_null95':float(point-np.percentile(nn,95)),
                'exploratory_permutation_tail_p':float((1+np.sum(nn>=point))/(nnull+1)),
                'null_n':nnull,'conditioning':'sex x age(<35,35-49,50-64,>=65)',
                'survey_design_corrected':False})

    # Separate predictive task: frozen 0506 reference map and 0506 outcome model.
    # All 7D models are intentionally full-rank controls; no claim of gain by rotation.
    cpca=PCA(n_components=7,svd_solver='full').fit(anchor)
    models={'Covariates':{},'RawX7':{},'H_G_CCA7':{},'PCA7':{}}
    cscale=StandardScaler().fit(cov_by['0506'])
    for name in COHORTS:
        cs=cscale.transform(cov_by[name])
        # Keep covariates, but avoid direct outcome usage for physiologic map.
        models['Covariates'][name]=cs
        models['RawX7'][name]=np.column_stack([cs,x[name]])
        models['H_G_CCA7'][name]=np.column_stack([cs,(x[name]-mb)@wb])
        models['PCA7'][name]=np.column_stack([cs,cpca.transform(x[name])])

    pred_rows=[]
    for outcome, col in OUTCOMES.items():
        for method, feats in models.items():
            y0=by['0506'][col].to_numpy(float)
            wc=w['0506']; cv_loss,alpha=cv_ridge_alpha(feats['0506'],y0,wc)
            fit=Ridge(alpha=alpha).fit(feats['0506'],y0,sample_weight=wc/np.mean(wc))
            for name in COHORTS:
                yy=by[name][col].to_numpy(float)
                yp=fit.predict(feats[name]); wei=w[name]
                pred_rows.append({'outcome':outcome,'method':method,'cohort':name,
                   'n':len(yy),'frozen_discovery_cv_mse':cv_loss,'chosen_alpha':alpha,
                   'frozen_weighted_mse':weighted_mse(yy,yp,wei),
                   'frozen_weighted_r2':weighted_r2(yy,yp,wei),
                   'mean_prediction_error':float(np.average(yp-yy,weights=wei))})

    out.mkdir(parents=True,exist_ok=True)
    def save(name,rows):pd.DataFrame(rows).to_csv(out/name,index=False)
    save('105_observed_stability.csv',obs_rows)
    save('105_descriptive_bootstrap.csv',bootstrap_rows)
    save('105_conditional_null.csv',null_rows)
    save('105_frozen_phenotype.csv',pred_rows)
    manifest={'script':'105_specificity_and_transfer_audit.py','seed':seed,
       'cohort_sizes':{k:len(v) for k,v in by.items()},
       'bootstrap':boot,'conditional_null_reps':nnull,'row_cap':cap,
       'representations':'original Script104 weighted full-rank H/G CCA recipe',
       'supervised_training':'0506 only; Ridge CV 0506 only',
       'null':'whole G-block within sex x four broad age bands',
       'true_NHANES_design_variance':False,
       'negative_controls_pre_registered_before_new_results':False,
       'posthoc_existing_cohorts':True,'biological_anchor_tested':False,
       'clinical_utility_tested':False,'independent_external_replication':False,
       'same_cohort_between_structure_and_prediction':True,
       'native_JIVE_AJIVE_rerun':False}
    (out/'105_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    nulltab=pd.DataFrame(null_rows)
    perftab=pd.DataFrame(pred_rows)
    lines=['# Script 105 — Falsification and frozen-transfer audit','',
       'This script does NOT demonstrate biological source identifiability or superior prediction.','',
       '## Conditional null — observed rediscovery versus age/sex-preserving placebo','',
       'The crucial comparison is observed similarity vs. the 95th percentile of conditioned-null similarity.',
       'A small/null margin means high absolute similarity by itself is not informative.','',
       '```text',nulltab[['cohort','group','observed_similarity','conditioned_null_high95','specificity_margin_vs_null95','exploratory_permutation_tail_p']].round(5).to_string(index=False),'```','',
       '## Frozen outcome models (NOT untouched test cohorts)','',
       'Every model is fitted on 0506 only. Compare covariate-only vs full-rank RawX7/PCA7/CCA7.',
       '```text',perftab.pivot_table(index=['outcome','method'],columns='cohort',values='frozen_weighted_r2').round(4).to_string(),'```','',
       '## Interpretation rules','',
       '- Observed stability exceeding an age/sex-conditioned placebo is evidence of a nontrivial STATISTICAL grouping, not biological identification.',
       '- Null preserving coarse age and sex does not remove continuous age, metabolic, renal or measurement confounding.',
       '- Participant resampling is NOT NHANES design-based uncertainty. Obtain strata/PSU for publication-grade confidence intervals.',
       '- If all 7D models deteriorate together, phenotype transfer failure is not unique to the CCA representation.',
       '- Score rotations of equal full dimension cannot increase information; regularized model comparisons may vary numerically.',
       '- A fully independent cohort and independent physiological anchors are still required to establish biological relevance.',
       '- Previously inspected future cohorts make these analyses exploratory; no confirmatory p-value claims.',
       '', '## Scientific outcome','',
       'The method is useful only if this falsification test adds discriminatory evidence beyond an unconditional correlation or a trivial full-rank reconstruction.']
    (out/'105_readout.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote:',out)
    print('Read 105_readout.md, 105_conditional_null.csv and 105_frozen_phenotype.csv')


def self_test():
    mod=import_s104()
    rng=np.random.default_rng(SEED)
    n=500
    h=rng.normal(size=(n,2))
    x=np.column_stack([h+rng.normal(size=(n,2))*.4,rng.normal(size=(n,2)),
                       h+rng.normal(size=(n,2))*.5,rng.normal(size=n)])
    w=rng.uniform(.5,2.,size=n)
    a,b,_=mod.fit_hg_cca(x,w)
    assert np.max(np.abs(((x-b)@a)@np.linalg.inv(a)+b-x))<1e-9
    labels=np.repeat([10,11,12,13],n//4)
    shuffled=permute_g_conditional(x,labels,rng)
    assert np.allclose(x[:,:4],shuffled[:,:4]);
    assert np.allclose(np.sort(x[:,4],axis=0),np.sort(shuffled[:,4],axis=0))
    assert not np.array_equal(x[:,4:],shuffled[:,4:])
    sim=mod.angle_stats((x-b)@a[:,[0,1]],(x-b)@a[:,[0,1]])
    assert sim['mean_squared_principal_cosine']>.999999
    assert weighted_r2(np.arange(n),np.arange(n),w)>.999999
    c=rng.normal(size=(n,4)); y=c[:,0]+rng.normal(size=n)
    cv,alpha=cv_ridge_alpha(c,y,w)
    assert np.isfinite(cv) and alpha in ALPHAS
    print('PASS: invertibility, stratum-constrained whole-block shuffling, subspaces, weighted metrics, discovery CV')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--bootstrap',type=int,default=199)
    p.add_argument('--null',type=int,default=199)
    p.add_argument('--cap',type=int,default=1800,
                   help='Maximum rows per cohort for both outcome and unsupervised tests; 0=all')
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--seed',type=int,default=SEED)
    args=p.parse_args()
    if args.self_test: return self_test()
    if min(args.bootstrap,args.null)<19: p.error('Bootstrap and null >=19')
    if args.cap<0:p.error('--cap must be nonnegative')
    s104=import_s104(); s103=s104.get_103()
    d,cov=s103.load_matched()
    evaluate(d,cov,s104,args.out,args.bootstrap,args.null,args.cap,args.seed)

if __name__=='__main__': main()
