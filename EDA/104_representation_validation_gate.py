#!/usr/bin/env python3
"""104 | Physiology representation validation gate (exploratory).

Place in <repository>/EDA/ alongside:
    98_regularized_phenotype_equation.py
    103_jive_ajive_benchmark.py

Run (project root):
    python EDA/104_representation_validation_gate.py --self-test
    python EDA/104_representation_validation_gate.py
    python EDA/104_representation_validation_gate.py --cap 0   # all eligible rows; slower

Requires the SAME environment that successfully ran Script 103:
  numpy, pandas, scipy, scikit-learn, Rscript + r.jive, Python idc9 jive.

Purpose: independent unsupervised re-fitting across disjoint NHANES cycles,
fixed-reference group/direction rediscovery, CCA coupling permutation control,
and an honest claim ledger combining Script 103. DO NOT claim biological
identifiability, causal effect, or superiority from this output.

Key design caveats:
 - Original frozen 72/73 Z was learned on 2005-08; thus 2007-08 is not
   independent *of that original frozen map*. This script fits an entirely NEW
   CCA-type map separately for each of 0506, 0708, 0918, 2123 for a genuine
   cross-cohort independent-refit analysis. It does not refit original 72 map.
 - The SAME CCA recipe as Script 72 is refitted, on a DIFFERENT matched
   complete-case sample than the original 72 discovery fit.
 - H/G semantics are design constraints. Group similarity proves statistical
   rediscovery only, NOT identifiable physiology or biological mechanism.
 - JIVE/AJIVE use official methods and training-only affine adapters to score
   a *fixed reference sample*; out-of-sample scores are NOT native outputs.
 - Each unsupervised fit is unweighted, all using the SAME discovery scaling.
 - Matched fully observed participant selection inherits 98's sample definition;
   PHQ completeness is a selection condition, NOT used to fit representations.
 - NHANES 2009-18 / 2021-23 have been analyzed previously: EXPLORATORY.
 - No survey-design adjusted inference. Permutation null is association-only.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=ROOT/'EDA'/'Results'/'104_representation_validation_gate'
SEED=104
COHORTS={'0506':['0506'],'0708':['0708'],'0918':['0910','1112','1314','1516','1718'],'2123':['2123']}
TARGET_GROUPS={'CoupledShared2':[0,1], 'Discordant2':[2,3],
               'H_Isolated2':[4,5], 'G_Isolated1':[6]}
METHOD_GROUPS={
   'HG_CCA7':TARGET_GROUPS,
   'JIVE5':{'Joint2':[0,1], 'H_Individual2':[2,3], 'G_Individual1':[4]},
   'AJIVE5':{'Joint2':[0,1], 'H_Individual2':[2,3], 'G_Individual1':[4]},
   'PCA5':{'Leading2':[0,1], 'Residual2':[2,3], 'Residual1':[4]}}


def whiten(cov):
    vals,vecs=np.linalg.eigh(cov)
    if min(vals)<1e-10: raise RuntimeError('Nearly singular biomarker covariance')
    return (vecs*(vals**-0.5))@vecs.T


def fit_hg_cca(x, w=None):
    """Independently refit Script-72 weighted CCA construction, rank2 shared.

    Uses the SAME weighted block standardization, covariance-whitening,
    canonical SVD and 4th H nullspace completion as official Script 72.
    Only the eligible sample differs (here we use Script 98 complete cases).
    If w=None, fits unweighted for mathematical self-tests.
    Returned W maps discovery-standardized X directly into 7 Z coordinates.
    """
    x=np.asarray(x,float)
    if w is None: w=np.ones(len(x))
    w=np.asarray(w,float)
    if not np.isfinite(x).all() or not np.isfinite(w).all() or (w<=0).any():
        raise ValueError('Nonfinite X or nonpositive NHANES weights')
    wn=w/w.sum()
    mu=(x*wn[:,None]).sum(0)
    xc=x-mu
    sd=np.sqrt((xc*xc*wn[:,None]).sum(0))
    if min(sd)<1e-9: raise RuntimeError('Zero-variance biomarker')
    h=xc[:,:4]/sd[:4];g=xc[:,4:]/sd[4:]
    A=(h*wn[:,None]).T@h
    B=(g*wn[:,None]).T@g
    C=(h*wn[:,None]).T@g
    ah=whiten(A);bg=whiten(B)
    p,rho,vt=np.linalg.svd(ah@C@bg,full_matrices=False)
    # Script 72's H-null canonical 4th direction is the null of P.T.
    _,_,vh=np.linalg.svd(p.T,full_matrices=True)
    w_h=ah@np.column_stack([p,vh[-1,:]])
    w_g=bg@vt.T
    r=np.clip(rho[:2],0,.999999)
    W=np.zeros((7,7),float)
    for j in range(2):
        W[:4,j]=w_h[:,j]/np.sqrt(2*(1+r[j]))/sd[:4]
        W[4:,j]=w_g[:,j]/np.sqrt(2*(1+r[j]))/sd[4:]
        W[:4,2+j]=w_h[:,j]/np.sqrt(2*(1-r[j]))/sd[:4]
        W[4:,2+j]=-w_g[:,j]/np.sqrt(2*(1-r[j]))/sd[4:]
    W[:4,4]=w_h[:,2]/sd[:4]
    W[:4,5]=w_h[:,3]/sd[:4]
    W[4:,6]=w_g[:,2]/sd[4:]
    if np.linalg.matrix_rank(W)!=7:raise RuntimeError('CCA fit is not full rank')
    return W,mu,rho


def angle_stats(a,b):
    """Compare subspaces in a COMMON participant reference (mean centered)."""
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)!=len(b) or a.shape[1]!=b.shape[1]:raise ValueError('Unequal anchor/group size')
    a=a-a.mean(0);b=b-b.mean(0)
    qa,ra=np.linalg.qr(a,mode='reduced');qb,rb=np.linalg.qr(b,mode='reduced')
    if min(np.abs(np.diag(ra)))<1e-10 or min(np.abs(np.diag(rb)))<1e-10:
        raise ValueError('Rank deficient group scores')
    s=np.clip(np.linalg.svd(qa.T@qb,compute_uv=False),0,1)
    return {'mean_squared_principal_cosine':float(np.mean(s*s)),
            'min_principal_cosine':float(np.min(s)),
            'max_principal_angle_deg':float(np.degrees(np.arccos(np.min(s))))}


def axis_match(a,b):
    """Absolute correlation Hungarian matching: score axes can permute/sign flip."""
    a=np.asarray(a,float);b=np.asarray(b,float)
    ac=(a-a.mean(0))/a.std(0,ddof=1)
    bc=(b-b.mean(0))/b.std(0,ddof=1)
    corr=np.abs(ac.T@bc/(len(a)-1))
    ii,jj=linear_sum_assignment(-corr)
    return float(np.mean(corr[ii,jj])),float(np.min(corr[ii,jj]))


def perm_ccap(x,reps,rng,w=None):
    obs=fit_hg_cca(x,w)[2][:2]
    null=np.empty((reps,2))
    for k in range(reps):
        xx=x.copy();xx[:,4:]=x[rng.permutation(len(x)),4:]
        null[k]=fit_hg_cca(xx,w)[2][:2]
    p=(1+(null>=obs[None,:]).sum(axis=0))/(1+reps)
    return [{'axis':i+1,'rho':float(obs[i]),'permutation_p':float(p[i]),
             'null95_rho':float(np.percentile(null[:,i],95))} for i in range(2)]


def self_test():
    rng=np.random.default_rng(SEED)
    n=850
    h=rng.normal(size=(n,2))
    A=np.column_stack([h+rng.normal(size=(n,2))*.4,
                       rng.normal(size=(n,2))])
    G=np.column_stack([h+rng.normal(size=(n,2))*.4,
                       rng.normal(size=n)])
    x=np.column_stack([A,G]);x=StandardScaler().fit_transform(x)
    W,mu,rho=fit_hg_cca(x)
    recon=(x-mu)@W@np.linalg.inv(W)+mu
    assert np.max(np.abs(x-recon))<1e-9
    assert np.allclose(W[4:,:][:,[4,5]],0)
    assert np.allclose(W[:4,6],0)
    a=(x-mu)@W[:,TARGET_GROUPS['CoupledShared2']]
    assert angle_stats(a,a)['min_principal_cosine']>0.999999
    am,bm=axis_match(a,a)
    assert min(am,bm)>0.999999
    null=perm_ccap(x,49,rng)
    assert null[0]['rho']>null[0]['null95_rho']
    # Null cross-block independence must not be made 'significant' by algorithm.
    random_x=rng.normal(size=(n,7))
    null2=perm_ccap(random_x,49,rng)
    assert np.isfinite([i['permutation_p'] for i in null2]).all()
    print('PASS: invertible map, structural block constraints, subspace matching, permutation controls')
    print('Synthetic positive-control rho:',np.round(rho[:2],3))


def get_103():
    src=HERE/'103_jive_ajive_benchmark.py'
    if not src.is_file():raise FileNotFoundError(f'Script 103 required: {src}')
    import importlib.util
    spec=importlib.util.spec_from_file_location('original103validation',src)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


def fit_native(mod,name,xx,out,rscript):
    if name=='JIVE5':return mod.jive_native(xx,out,rscript)
    if name=='AJIVE5':return mod.ajive_native(xx)
    raise KeyError(name)


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--self-test',action='store_true')
    pa.add_argument('--out',type=Path,default=OUT)
    pa.add_argument('--cap',type=int,default=1800,
                    help='Max fit rows per period; 0 uses all. Same cap for every method.')
    pa.add_argument('--permutations',type=int,default=199,
                    help='Cross-block permutation null reps (min 19)')
    pa.add_argument('--rscript',default='Rscript')
    pa.add_argument('--cca-unweighted',action='store_true',
                    help='Sensitivity analysis: remove Script-72 survey weighting for CCA')
    pa.add_argument('--no-native',action='store_true',
                    help='Quick CCA/PCA only; not a JIVE/AJIVE closure test')
    args=pa.parse_args()
    if args.self_test:return self_test()
    if args.cap<0:pa.error('--cap must be >= 0')
    if args.permutations<19:pa.error('--permutations must be >=19')
    mod=get_103();d,_=mod.load_matched()
    if 'CYCLE' not in d:raise RuntimeError('Missing CYCLE')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    d=d.copy();d['CYCLE']=d.CYCLE.astype(str).str.zfill(4)
    data={}
    for name,cycles in COHORTS.items():
        temp=d[d.CYCLE.isin(cycles)].copy()
        if len(temp)<100:raise RuntimeError(f'Too few complete patients {name}: {len(temp)}')
        if args.cap and len(temp)>args.cap:
            select=np.sort(rng.choice(len(temp),args.cap,replace=False))
            temp=temp.iloc[select].copy()
        data[name]=temp
    names=list(COHORTS)
    # Global discovery-only measurement scaling, then DO NOT refit scaler.
    # Baseline discovery is 0506 only: each independently fitted model is unseen
    # to all other cohorts. Original Script-72 frozen map isn't refit here.
    scaler=StandardScaler().fit(data['0506'][mod.X_COLS].to_numpy(float))
    x={n:scaler.transform(data[n][mod.X_COLS].to_numpy(float)) for n in names}
    anchor=x['0506'] # identical evaluation individuals for EVERY independent model
    print('Independent unsupervised cohort fit sizes:',{n:len(data[n]) for n in names})
    print('CCA refit: exact Script-72 weighting recipe', 'UNWEIGHTED' if args.cca_unweighted else 'SURVEY')
    fits={};diagnostics=[];ccas=[];natives=[];errors=[]
    for cohort in names:
        wc=None if args.cca_unweighted else data[cohort]['SURVEY_WT'].to_numpy(float)
        W,mu,rho=fit_hg_cca(x[cohort],wc)
        sc={'HG_CCA7':(anchor-mu)@W}
        pi=PCA(n_components=5,svd_solver='full').fit(x[cohort]);sc['PCA5']=pi.transform(anchor)
        ccas.extend({'cohort':cohort,**rr} for rr in perm_ccap(x[cohort],args.permutations,rng,wc))
        if not args.no_native:
            for native in ['JIVE5','AJIVE5']:
                try:
                    ns=fit_native(mod,native,x[cohort],out,args.rscript)
                    adapter,diag=mod.learn_projection(x[cohort],ns,min_fit_r2=.90)
                    sc[native]=mod.affine_predict(anchor,adapter)
                    natives.append({'cohort':cohort,'method':native,**diag})
                    print(f'{cohort} {native} train adapter R2={diag["train_score_r2"]:.5f}')
                except Exception as ex:
                    errors.append({'cohort':cohort,'method':native,'error':str(ex)})
                    print(f'ERROR {cohort} {native}: {ex}',file=sys.stderr)
        fits[cohort]=sc
    if errors:raise RuntimeError('Native methods failed; no misleading partial paper readout. '+json.dumps(errors))
    for method,groups in METHOD_GROUPS.items():
        if args.no_native and method in ('JIVE5','AJIVE5'):continue
        for cohort in names[1:]:
            for group,idx in groups.items():
                base=fits['0506'][method][:,idx]
                rec=fits[cohort][method][:,idx]
                row={'method':method,'cohort_ref':'0506','cohort_rediscovered':cohort,
                     'group':group,'dimension':len(idx)}
                row.update(angle_stats(base,rec))
                axis_mean,axis_min=axis_match(base,rec)
                row.update({'matched_axis_mean_abs_corr':axis_mean,
                            'matched_axis_min_abs_corr':axis_min})
                diagnostics.append(row)
    pd.DataFrame(diagnostics).to_csv(out/'104_independent_rediscovery.csv',index=False)
    pd.DataFrame(ccas).to_csv(out/'104_crossblock_permutation.csv',index=False)
    pd.DataFrame(natives).to_csv(out/'104_native_adapter_diagnostics.csv',index=False)

    # Stronger benchmark claims require externally established clinical anchors,
    # alternative ranks, confidence intervals for methodological differences.
    claim=[
      ('Exact 7D reconstruction','SUPPORTED ALREADY','Scripts 72-73 and 103; full-rank linear map. Mathematical, not novel.'),
      ('Cross-block nonrandom correlation','EVALUATE 104','See 104_crossblock_permutation.csv. Noncausal coupling.'),
      ('Independent rediscovery by cohort','EVALUATE 104','See 104_independent_rediscovery.csv. Statistical subspace/axis, not biological identification.'),
      ('Superior 5D reconstruction to PCA/JIVE','NOT SUPPORTED','103: Z5 around .80-.82 vs PCA/JIVE around .97-.98.'),
      ('Superior 7D phenotype prediction','NOT SUPPORTED','103: practically tied with PCA7/RawX7.'),
      ('More biological meaning than JIVE/AJIVE','OPEN','Needs independent physiological anchors, expert review, falsification.'),
      ('Longitudinal subject-level evidence','OPEN','NHANES time transfer is repeated cross-sectional; use true repeated-person data.'),
      ('Generalisable validation framework','PROPOSED','Only credible after independent multicohort demonstrations + reproducible criteria.')]
    claim_df=pd.DataFrame(claim,columns=['proposed_claim','status','grounds'])
    claim_df.to_csv(out/'104_claim_ledger.csv',index=False)
    manifest={'script':'104_representation_validation_gate.py','seed':SEED,
       'data':'Script-98 matched real NHANES complete cases','cohorts':{n:len(data[n]) for n in names},
       'no_native':args.no_native,'cap':args.cap,'permutations':args.permutations,
       'cca_fit_recipe':'Script 72 weighted block standardization, covariance whitening, CCA, H nullspace',
       'cca_unweighted_sensitivity':args.cca_unweighted,
       'source_72_frozen_Z_unchanged':True,
       'new_HG_CCA_refits_original_72_recipe_on_selected_98_complete_cases':True,
       'ranks':{'HG_CCA7':7,'PCA5':5,'JIVE5':5,'AJIVE5':5},
       'adapter_native_out_of_sample':False,
       'comparison_anchor':'same 0506 discovery participants for ALL independently fitted maps',
       'future_outcomes_used_to_select_maps':False,'survey_design_correct_inference':False,
       'temporal_holdout_untouched':False,'biological_identifiability_proven':False,
       'errors':errors}
    (out/'104_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    scores=pd.DataFrame(diagnostics)
    tests=pd.DataFrame(ccas)
    rpt=['# Script 104 — Representation validation gate','',
       '## What was actually run','',
       f'- Cohort sizes (sampled without outcome information): `{manifest["cohorts"]}`.',
       '- Original Script-72 map was **not** changed. Its CCA recipe was **refit independently** per cohort, on the matched Script-98 sample.',
       ('- JIVE/AJIVE were NOT run (--no-native): THIS IS NOT THE FULL METHODS BENCHMARK.'
        if args.no_native else
        '- JIVE/AJIVE were fit using official packages; training-only linear projections allow fixed-anchor comparison.'),
       '- Every subspace was evaluated on exactly the same reference participants (0506).',
       '- Whole 7D subspace stability deliberately **not** scored (trivially identical for invertible maps).',
       '- Original Script 103 showed our 5D reduction underperformed JIVE/PCA compression.',
       '', '## Independent group rediscovery (principal subspace similarity; 1=perfect)','',
       '```text',scores.pivot_table(index=['method','group'],columns='cohort_rediscovered',
           values='mean_squared_principal_cosine').round(5).to_string(),'```','',
       '## Individual axis rediscovery (Hungarian matched absolute correlation)','',
       '```text',scores.pivot_table(index=['method','group'],columns='cohort_rediscovered',
           values='matched_axis_mean_abs_corr').round(5).to_string(),'```','',
       '## Cross-block correlations versus G-row permutation null','',
       '```text',tests.round(5).to_string(index=False),'```','',
       '## Claim ledger (read before drafting conclusions)','',
       '| Proposed claim | Status | Grounds |',
       '|---|---|---|',
       *[f'| {k} | {v} | {g} |' for k,v,g in claim],
       '', '## What this can and cannot establish','',
       '- Statistical subspace reproducibility is a valuable diagnostic, **not** evidence of unique biological source identification.',
       '- Shared/isolated partitions are imposed by construction. Compare JIVE and AJIVE as distinct models, not identical biological decompositions.',
       '- Comparisons depend on fixed ranks (JIVE/AJIVE 2 joint, 2 H individual, 1 G individual) and their affine score adapters.',
       '- Positive cross-block association can be caused by shared confounders or biomarker measurement effects; not causal.',
       '- Disjoint periods mean different people; within-person longitudinal claims remain unsupported.',
       '- 0506 and 0708 are independent patient groups here, but the ORIGINAL 72 map had been learned using 2005–08 data.',
       '- Do not turn high similarities into significance claims: these estimates are descriptive and lack survey-design uncertainty.',
       '- The strong paper angle is a transparently reported **validation protocol plus empirical case study**, not automatic algorithm superiority.',
       '', '## Closure decision','',
       'If component rediscovery is strong across 0918/2123, report it as an exploratory robustness result. ',
       'If it fails, report lack of generality. In either case, proceed with the review/framework draft while marking',
       'external biological anchoring, independent dataset and true longitudinal validation as open gates.']
    (out/'104_readout.md').write_text('\n'.join(rpt)+'\n',encoding='utf-8')
    print('\nSaved:',out)
    print('Read:',out/'104_readout.md')
    print('CLAIM: no demonstrated biological or methodological superiority merely from this benchmark.')

if __name__=='__main__':main()
