#!/usr/bin/env python3
"""Script 103 — Official JIVE/AJIVE versus frozen A/G decomposition.

From the existing NHANES repository root:
  python EDA/103_jive_ajive_benchmark.py
  python EDA/103_jive_ajive_benchmark.py --self-test
  python EDA/103_jive_ajive_benchmark.py --native-smoke

Required:
  pip install numpy pandas scipy scikit-learn jive
  Rscript available in PATH, and in R: install.packages('r.jive')

This is an exploratory, honest method benchmark, not a proof of clinical
validity or novelty. JIVE is CRAN r.jive, AJIVE is idc9/py_jive (pip 'jive').
Zero surrogate implementations or silently fabricated method results.

Inputs are exactly Script 98's real common-complete cohort and frozen Z.
Train representations and outcome predictors on 2005-08 only. Frozen tests
2009-18 and 2021-23. Later outcomes have already been inspected in research;
these are NOT untouched confirmatory holdouts. Distinct NHANES subjects.

Native JIVE/AJIVE do not offer a common survey-weighted training API or a
standard cross-period transform. We (a) fit them on discovery only with
unweighted standardized X, (b) export native fitted subject scores, then (c)
learn a training-only linear score projection for temporal holdouts. Thus
future scores are an OUT-OF-SAMPLE PROJECTION ADAPTER, not native method output.
Raw X, PCA and frozen Z receive the same outcome-fitting and scoring protocol.
All 5D comparisons use 5 scores; FrozenZ7 and RawX7 are 7D controls.
JIVE and AJIVE: fixed joint rank 2, A-individual 2, G-individual 1 (5D),
not ranks selected to maximize phenotype scores. Alternative rank choices
must be run and reported separately before making a general superiority claim.

Primary measures: target-free heldout reconstruction, weighted phenotype
prediction R²/MSE, paired descriptive uncertainty (NOT survey-design CIs),
training-only score adapter fit and source-block dependence. No automatically
constructed combined 'winner' score; inspect all the metrics.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
EDA = ROOT / 'EDA'
OUT_DEFAULT = EDA / 'Results' / '103_jive_ajive_benchmark'
SOURCE98 = EDA / '98_regularized_phenotype_equation.py'
SEED = 103
X_COLS = ['LBXHGB','LBXRBCSI','LBXMCVSI','LBXRDW','LBXGH','LBXGLU','LOG_IN']
Z_COLS = ['SP_SHARED1','SP_SHARED2','SP_DISCORD1','SP_DISCORD2',
          'SP_A_PRIVATE3','SP_A_PRIVATE4','SP_G_PRIVATE3']
# Z5 axes chosen outcome-free: both shared, one discord, one A-private, one G-private.
# Choices of discord and A-private are scored solely by discovery X reconstruction.
PHENOTYPES = {'PHQ_total':'PHQ9_TOTAL','Somatic':'SOMATIC_SCORE','CogAff':'COGAFF_SUM'}
PERIODS = ['train_2005-2008','test_2009-2018','validate_2021-2023']
ALPHAS = (1., 10., 100., 1000.)

# The R bridge calls original r.jive; never substitutes an informal imitation.
R_BRIDGE = r'''
args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(r.jive))
d <- as.matrix(read.csv(args[1], check.names=FALSE))
if (ncol(d)!=7 || any(!is.finite(d))) stop('Expected finite n by 7 matrix')
set.seed(103)
A <- t(d[,1:4,drop=FALSE]); G <- t(d[,5:7,drop=FALSE])
# Data already standardized on training X; disable internal centering and
# Frobenius scaling to match the PCA/AJIVE training input.
m <- r.jive::jive(list(A=A,G=G),method='given',rankJ=2,rankA=c(2,1),
                  scale=FALSE,center=FALSE,est=TRUE,orthIndiv=TRUE,
                  maxiter=1000,showProgress=FALSE)
score_svd <- function(m, rank) {
  if (rank == 0) return(matrix(numeric(),nrow=ncol(m),ncol=0))
  s <- svd(t(m),nu=rank,nv=0)
  sweep(s$u[,seq_len(rank),drop=FALSE],2,s$d[seq_len(rank)],`*`)
}
# r.jive's returned joint/individual lists are position-indexed in some
# versions even when input blocks were named. $A/$G may return NULL.
if (length(m$joint)!=2 || length(m$individual)!=2)
  stop("Unexpected official r.jive result: expected two blocks")
J <- rbind(as.matrix(m$joint[[1]]), as.matrix(m$joint[[2]]))
IA <- as.matrix(m$individual[[1]])
IG <- as.matrix(m$individual[[2]])
if (ncol(J)!=nrow(d) || ncol(IA)!=nrow(d) || ncol(IG)!=nrow(d))
  stop("Official r.jive output has unexpected participant dimension")
joint <- score_svd(J,2)
ai <- score_svd(IA,2)
gi <- score_svd(IG,1)
scores <- cbind(joint,ai,gi)
if(nrow(scores)!=nrow(d)||ncol(scores)!=5||any(!is.finite(scores))) stop('Invalid JIVE scores')
colnames(scores)<-c('joint1','joint2','Aind1','Aind2','Gind1')
write.csv(scores,args[2],row.names=FALSE)
'''


def weighted_mean(a, w):
    return np.average(a, weights=w, axis=0)


def weighted_mse(y, yp, w):
    return float(np.average((y-yp)**2, weights=w))


def weighted_r2(y, yp, w):
    m = np.average(y, weights=w)
    var = np.average((y-m)**2, weights=w)
    return float(1-weighted_mse(y,yp,w)/var) if var>0 else np.nan


def norm_sample_weights(w):
    return np.asarray(w,float)/np.mean(w)


def run_self_test():
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(200,7)); X[:,1] += .4*X[:,0]
    w = rng.uniform(.5,2,200)
    native = X @ rng.normal(size=(7,5))
    P, diagnostics = learn_projection(X,native)
    actual = affine_predict(X,P)
    assert np.max(np.abs(actual-native)) < 1.e-10
    assert diagnostics['train_score_r2'] > .999999
    a = np.linspace(-1,1,200)
    b = a + .1
    assert weighted_mse(a,b,w) > 0
    m = WeightedPCA(5).fit(X)
    assert m.transform(X).shape == (200,5)
    print('PASS: adapter recovery, score representation, metric sanity')


def load_matched():
    if not SOURCE98.is_file():
        raise FileNotFoundError(f'Script 98 not found at {SOURCE98}')
    spec = importlib.util.spec_from_file_location('s98_for_103',SOURCE98)
    s98 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = s98
    spec.loader.exec_module(s98)
    for name, val in [('yield_msg',[]),('normalizer',{}),
                      ('categories_reference',{}),('categories_feature_names',[])]:
        setattr(s98,name,val)
    d = s98.load_real()
    ff = s98.build_features(d)
    s98.categories_feature_names = [a for a in ff.columns if '=' in a]
    cov = ff[s98.covariates('full')].to_numpy(float)
    if not set(X_COLS+Z_COLS).issubset(d.columns):
        raise ValueError('Missing real X/Z columns in frozen cohort')
    return d,cov


class WeightedPCA:
    def __init__(self,k): self.k=k
    def fit(self,x):
        self.mu = np.mean(x,axis=0)
        _,_,vt = np.linalg.svd(x-self.mu,full_matrices=False)
        self.vt = vt[:self.k]
        return self
    def transform(self,x): return (x-self.mu)@self.vt.T


def learn_projection(x, target, min_fit_r2=.90):
    """Outcome-free affine frozen X -> native in-sample scores adapter.
    This may be approximate (quantified in diagnostics). Not native OOS JIVE/AJIVE.
    """
    x = np.asarray(x,float); target = np.asarray(target,float)
    if len(x)!=len(target): raise ValueError('Projection samples differ')
    A = np.column_stack((np.ones(len(x)),x))
    coef = np.linalg.lstsq(A,target,rcond=1e-12)[0]
    pred=A@coef
    residual=np.sum((pred-target)**2)
    total=np.sum((target-target.mean(0))**2)
    fit_r2=float(1-residual/total) if total>0 else np.nan
    if fit_r2 < min_fit_r2: raise RuntimeError(f'Unreliable projection adapter: R2={fit_r2:.6f}')
    return coef, {'train_score_r2':fit_r2,
                  'native_max_abs_error':float(np.max(np.abs(pred-target)))}


def affine_predict(x,coef):
    return np.column_stack((np.ones(len(x)),x))@coef


def select_z5_from_physiology(z_train, x_train, w_train):
    """Constrained train-only reconstruction selection, never phenotype-driven."""
    candidates=[]
    for discord in [2,3]:
        for aprivate in [4,5]:
            chosen=[0,1,discord,aprivate,6]
            decoder,_=learn_projection(z_train[:,chosen],x_train,min_fit_r2=-np.inf)
            rec=affine_predict(z_train[:,chosen],decoder)
            error=float(np.average(np.sum((x_train-rec)**2,axis=1),weights=w_train))
            candidates.append((error,chosen))
    return min(candidates,key=lambda x:(x[0],x[1]))[1]


def jive_native(xtrain, outdir, rscript):
    if not shutil.which(rscript) and not Path(rscript).is_file():
        raise RuntimeError(f'Rscript not found: {rscript}. Install R and CRAN r.jive.')
    with tempfile.TemporaryDirectory(prefix='benchmark103_r_') as td:
        xfile=Path(td)/'train.csv'; sfile=Path(td)/'jive_scores.csv'; rfile=Path(td)/'jive.R'
        pd.DataFrame(xtrain).to_csv(xfile,index=False)
        rfile.write_text(R_BRIDGE,encoding='utf-8')
        cmd=[rscript,'--vanilla',str(rfile),str(xfile),str(sfile)]
        run=subprocess.run(cmd,check=False,capture_output=True,text=True,timeout=300)
        if run.returncode:
            raise RuntimeError('Official r.jive failed: '+run.stderr[-3000:]+run.stdout[-1000:])
        if not sfile.is_file(): raise RuntimeError('r.jive did not produce native scores')
        scores=pd.read_csv(sfile).to_numpy(float)
    if scores.shape!=(len(xtrain),5):raise RuntimeError(f'JIVE wrong scores shape: {scores.shape}')
    return scores


def ajive_native(xtrain):
    try:
        # py_jive imports isshape/isintlike from scipy.sparse.sputils.
        # Modern SciPy moved them to scipy.sparse._sputils. Restore only
        # these legacy attributes in memory; keep the official AJIVE algorithm.
        import scipy.sparse.sputils as legacy_sputils
        from scipy.sparse import _sputils as modern_sputils
        for attr in ('isshape', 'isintlike'):
            if not hasattr(legacy_sputils, attr):
                setattr(legacy_sputils, attr, getattr(modern_sputils, attr))
        from jive.AJIVE import AJIVE
    except Exception as exc:
        raise RuntimeError('Official idc9 AJIVE import failed after SciPy '
                           f'compatibility shim. Original exception: {exc}') from exc
    # Respect package API. Fix ranks before seeing targets; never tune on later outcomes.
    a=AJIVE(init_signal_ranks={'A':3,'G':2}, joint_rank=2,
            indiv_ranks={'A':2,'G':1},center=False,
            reconsider_joint_components=False,n_jobs=1,store_full=False)
    a.fit({'A':xtrain[:,:4],'G':xtrain[:,4:]})
    native=a.results_dict()
    def as_scores(x,k,what):
        arr=np.asarray(x,dtype=float)
        if arr.shape!=(len(xtrain),k):
            raise RuntimeError(f'AJIVE native {what} scores expected {(len(xtrain),k)}, got {arr.shape}')
        return arr
    # Native PCA 'scores' = orthonormal U. Each scores block is a score axis.
    joint=as_scores(native['common']['scores'],2,'joint')
    ai=as_scores(native['A']['individual']['scores'],2,'A-private')
    gi=as_scores(native['G']['individual']['scores'],1,'G-private')
    return np.column_stack((joint,ai,gi))


def reconstruction_metric(x,true,train_scores,scores,train_x,wtrain,w):
    # Outcome-free linear decoder trained on discovery only.
    decoder,_=learn_projection(train_scores,train_x,min_fit_r2=-np.inf)
    re=affine_predict(scores,decoder)
    numerator=float(np.average(((true-re)**2).sum(1),weights=w))
    # Denominator is relative to the FROZEN discovery mean, not the future mean.
    anchor=weighted_mean(train_x,wtrain)
    denom=float(np.average(((true-anchor)**2).sum(1),weights=w))
    return float(1-numerator/denom) if denom>0 else np.nan


def fit_ridge_and_pred(f,y,w,train,cv,alpha):
    sc=StandardScaler()
    t=sc.fit_transform(f[train],sample_weight=w[train])
    m=Ridge(alpha=alpha)
    m.fit(t,y[train],sample_weight=norm_sample_weights(w[train]))
    return m.predict(sc.transform(f))


def cv_alpha(f,y,w,train,cv):
    best=None
    for alpha in ALPHAS:
        folds=[]
        for a,b in cv:
            it,iv=train[a],train[b]
            sc=StandardScaler().fit(f[it],sample_weight=w[it])
            m=Ridge(alpha=alpha)
            m.fit(sc.transform(f[it]),y[it],sample_weight=norm_sample_weights(w[it]))
            yp=m.predict(sc.transform(f[iv]))
            folds.append(weighted_mse(y[iv],yp,w[iv]))
        key=(float(np.mean(folds)),alpha)
        if best is None or key<best: best=key
    return best


def bootstrap_paired_mse(y, baseline, candidate, w, nrep, seed):
    rng=np.random.default_rng(seed); n=len(y)
    # Fixed model predictions, paired resampling of participants.
    # Resampling probabilities uniform: these are DESCRIPTIVE intervals,
    # NOT NHANES cluster/stratum-aware uncertainty and not a hypothesis test.
    vals=[]
    for _ in range(nrep):
        indices=rng.integers(0,n,size=n)
        vals.append(weighted_mse(y[indices],baseline[indices],w[indices])
                    -weighted_mse(y[indices],candidate[indices],w[indices]))
    return tuple(np.percentile(vals,[2.5,97.5]).tolist())


def weighted_correlation_profiles(x, scores, w):
    """Biomarker correlation fingerprints per fixed component (7 x k)."""
    x = np.asarray(x, float); scores = np.asarray(scores, float)
    w = np.asarray(w, float); w = w/w.sum()
    xc = x - np.sum(w[:,None]*x, axis=0)
    zc = scores - np.sum(w[:,None]*scores, axis=0)
    xs = np.sqrt(np.maximum(np.sum(w[:,None]*xc**2,axis=0),1e-14))
    zs = np.sqrt(np.maximum(np.sum(w[:,None]*zc**2,axis=0),1e-14))
    return ((xc*w[:,None]).T @ zc) / xs[:,None] / zs[None,:]


def fingerprint_transfer(x, scores, splits, w, name):
    """Descriptive transfer of frozen biomarker fingerprints, not rediscovery.
    Scores / directions are held fixed; later-period correlations may drift.
    """
    base = splits == PERIODS[0]
    ref = weighted_correlation_profiles(x[base],scores[base],w[base])
    rows = []
    for period in PERIODS[1:]:
        idx = splits == period
        later=weighted_correlation_profiles(x[idx],scores[idx],w[idx])
        for j in range(scores.shape[1]):
            v0=ref[:,j];v1=later[:,j]
            denom=np.linalg.norm(v0)*np.linalg.norm(v1)
            cosine=float(v0@v1/denom) if denom>1e-12 else np.nan
            rows.append({'method':name,'period':period,'coordinate':j+1,
                         'marker_fingerprint_cosine':cosine,
                         'dominant_train_biomarker':X_COLS[int(np.argmax(np.abs(v0)))],
                         'dominant_later_biomarker':X_COLS[int(np.argmax(np.abs(v1)))],
                         'same_top_biomarker':bool(np.argmax(np.abs(v0)) == np.argmax(np.abs(v1)))})
    return rows


def source_jacobian(x, scores):
    coef,_=learn_projection(x,scores)
    A=coef[1:,:]
    out=[]
    for j in range(scores.shape[1]):
        a=float(np.linalg.norm(A[:4,j])); g=float(np.linalg.norm(A[4:,j]));
        out.append({'coordinate':j+1,'A_raw_sensitivity_norm':a,
                    'G_raw_sensitivity_norm':g,
                    'G_share_of_squared_norm':float(g*g/(a*a+g*g)) if a*a+g*g>0 else np.nan})
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT_DEFAULT)
    parser.add_argument('--rscript',default='Rscript')
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--bootstraps',type=int,default=250)
    parser.add_argument('--skip-jive',action='store_true',help='Explicitly omit real JIVE (records omission)')
    parser.add_argument('--skip-ajive',action='store_true',help='Explicitly omit real AJIVE (records omission)')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--native-smoke',action='store_true',
                        help='Fit official JIVE and AJIVE on tiny synthetic data; '
                             'tests installation without reading NHANES')
    args=parser.parse_args()
    if args.self_test: return run_self_test()
    if args.native_smoke:
        rng = np.random.default_rng(SEED)
        n=90
        h=rng.normal(size=(n,2))
        a=h@rng.normal(size=(2,4))+rng.normal(size=(n,4))*.8
        g=h@rng.normal(size=(2,3))+rng.normal(size=(n,3))*.8
        x=StandardScaler().fit_transform(np.column_stack([a,g]))
        if not args.skip_jive:
            sj=jive_native(x,args.out,args.rscript)
            assert sj.shape==(n,5) and np.isfinite(sj).all()
            print('JIVE official native smoke: PASS')
        if not args.skip_ajive:
            sa=ajive_native(x)
            assert sa.shape==(n,5) and np.isfinite(sa).all()
            print('AJIVE official native smoke: PASS')
        return
    if not 3<=args.folds<=10:parser.error('--folds must be 3..10')
    if args.bootstraps<0:parser.error('--bootstraps must be >= 0')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    d,cov=load_matched()
    splits=d.SPLIT.astype(str).to_numpy();cycles=d.CYCLE.astype(str).to_numpy()
    w=d.SURVEY_WT.to_numpy(float)
    tr=np.flatnonzero(splits==PERIODS[0])
    if len(tr)<100 or not all((splits==s).sum()>=100 for s in PERIODS):
        raise RuntimeError('Insufficient complete real observations in discovery or later periods')
    raw=d[X_COLS].to_numpy(float)
    z=d[Z_COLS].to_numpy(float)
    # COMMON, unweighted training scale for native AJIVE/JIVE/PCA; no future fitting.
    scaler=StandardScaler().fit(raw[tr]); X=scaler.transform(raw)
    xtr=X[tr]
    print('Matched real cohort:',{s:int((splits==s).sum()) for s in PERIODS})
    print('Fitting official unsupervised JIVE/AJIVE on discovery physiology ONLY')
    z5_idx=select_z5_from_physiology(z[tr],xtr,w[tr])
    print('Frozen Z5 indices chosen by discovery physiological reconstruction only:',z5_idx)
    methods={
      'RawX7':X,
      'FrozenZ7':z,
      'FrozenZ5_trainX':z[:,z5_idx],
      'PCA5':WeightedPCA(5).fit(xtr).transform(X),
      'PCA7':WeightedPCA(7).fit(xtr).transform(X),
    }
    adapter_records=[];errors={}; versions={}
    for name,enabled,fn in [
        ('JIVE5_rjive',not args.skip_jive,lambda:jive_native(xtr,out,args.rscript)),
        ('AJIVE5_pyjive',not args.skip_ajive,lambda:ajive_native(xtr))]:
        if not enabled:
            errors[name]='SKIPPED BY USER FLAG';continue
        try:
            native=fn()
            mapping,diag=learn_projection(xtr,native)
            future=affine_predict(X,mapping)
            if future.shape[1]!=5:raise RuntimeError('Expected 5 native axes')
            methods[name]=future
            adapter_records.append({'method':name,**diag})
            print(f'{name}: SUCCESS; train adapter R2={diag["train_score_r2"]:.7f}')
        except Exception as exc:
            errors[name]=str(exc)
            print(f'{name}: FAILED — {exc}',file=sys.stderr)
    if errors and not (args.skip_jive or args.skip_ajive):
        raise RuntimeError('A required official method failed. '+json.dumps(errors)+
           '. Fix installation; or explicitly --skip-jive/--skip-ajive (incomplete comparison).')
    if not args.skip_jive and 'JIVE5_rjive' not in methods:raise RuntimeError('Missing JIVE output')
    if not args.skip_ajive and 'AJIVE5_pyjive' not in methods:raise RuntimeError('Missing AJIVE output')
    for name,score in methods.items():
        if not np.isfinite(score).all():raise ValueError(f'Nonfinite {name} scores')

    # PHENOTYPE-FREE TEST: out-of-period ability to reconstruct biomarkers from
    # a training-frozen low-dimensional representation. Rank-matched at 5D.
    recon=[];purity=[];fingerprints=[]
    for name,score in methods.items():
        for period in PERIODS:
            ix=splits==period
            rec=reconstruction_metric(X[ix],X[ix],score[tr],score[ix],X[tr],w[tr],w[ix])
            recon.append({'method':name,'dimension':score.shape[1],
                          'period':period,'n':int(ix.sum()),'reconstruction_R2':rec})
        fingerprints.extend(fingerprint_transfer(X, score, splits, w, name))
        for row in source_jacobian(xtr,score[tr]):
            label=(('joint' if row['coordinate']<=2 else
                    'A_individual' if row['coordinate']<=4 else 'G_individual')
                   if name.startswith(('JIVE5','AJIVE5')) else 'unordered_or_prespecified')
            purity.append({'method':name,'coordinate_label':label,**row})

    folds=list(StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=SEED)
               .split(tr,cycles[tr]))
    predictions={};cv_rows=[];perf=[];bs=[]
    for yname,ycol in PHENOTYPES.items():
        y=d[ycol].to_numpy(float)
        for name,score in methods.items():
            feat=np.column_stack((cov,score))
            cv_mse,alpha=cv_alpha(feat,y,w,tr,folds)
            yp=fit_ridge_and_pred(feat,y,w,tr,folds,alpha)
            predictions[(yname,name)]=yp
            cv_rows.append({'outcome':yname,'method':name,'dimension':score.shape[1],
                           'discovery_CV_MSE':cv_mse,'ridge_alpha':alpha})
            for period in PERIODS:
                ix=splits==period
                perf.append({'outcome':yname,'method':name,'dimension':score.shape[1],
                             'period':period,'n':int(ix.sum()),
                             'weighted_MSE':weighted_mse(y[ix],yp[ix],w[ix]),
                             'weighted_R2':weighted_r2(y[ix],yp[ix],w[ix]),
                             'weighted_MAE':float(np.average(abs(y[ix]-yp[ix]),weights=w[ix]))})
            print(f'{yname:9} {name:19} CV_MSE={cv_mse:.5f} alpha={alpha:.0f}')
        # Compare only equal-dimensional methods against PCA5 and report 7D separately.
        for name in methods:
            if name in ('PCA5','PCA7','RawX7','FrozenZ7'):continue
            comparator='PCA5' if methods[name].shape[1]==5 else 'PCA7'
            for period in PERIODS[1:]:
                ix=splits==period
                candidate=predictions[(yname,name)][ix]
                baseline=predictions[(yname,comparator)][ix]
                dMSE=weighted_mse(y[ix],baseline,w[ix])-weighted_mse(y[ix],candidate,w[ix])
                lo,hi=(np.nan,np.nan) if not args.bootstraps else bootstrap_paired_mse(
                    y[ix],baseline,candidate,w[ix],args.bootstraps,SEED+len(bs))
                bs.append({'outcome':yname,'method':name,'reference':comparator,
                           'period':period,'delta_MSE_positive_better':dMSE,
                           'descriptive_CI_low':lo,'descriptive_CI_high':hi,
                           'interval_includes_zero':bool(lo<=0<=hi) if np.isfinite(lo) else None})

    def save(name,rows):pd.DataFrame(rows).to_csv(out/name,index=False)
    save('103_reconstruction.csv',recon)
    save('103_training_cv.csv',cv_rows)
    save('103_frozen_phenotype_performance.csv',perf)
    save('103_paired_bootstrap.csv',bs)
    save('103_source_sensitivity.csv',purity)
    save('103_marker_fingerprint_transfer.csv',fingerprints)
    save('103_adapters.csv',adapter_records)
    pv=pd.DataFrame(perf)
    rec=pd.DataFrame(recon)
    paired=pd.DataFrame(bs)
    fp=pd.DataFrame(fingerprints)
    cvtab=pd.DataFrame(cv_rows)
    manifest={'script':'103_jive_ajive_benchmark.py','seed':SEED,
              'same_participants_per_method':True,'original_Z_frozen':True,
              'n_by_period':{s:int((splits==s).sum()) for s in PERIODS},
              'methods_run':list(methods),'omissions':errors,
              'jive_ajive_fixed_ranks':{'joint':2,'A_individual':2,'G_individual':1},
              'Z5_outcome_free_indices':z5_idx,
              'outcome_fit_and_tuning':'discovery_only',
              'unweighted_unsupervised_fit':'same common X scaling; JIVE/AJIVE do not accept NHANES design weights',
              'out_of_sample_joint_individual_scores':'training-only affine projection adapter (not native method output)',
              'survey_variance_correct':False,'external_confirmatory':False,
              'revisited_future_periods':True,'bootstrap_reps':args.bootstraps,
              'package_details':'official R r.jive and Python idc9 jive; package versions should be recorded manually'}
    (out/'103_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    report=['# Script 103 — Head-to-head decomposition comparison','',
            '## Method integrity',
            '- Official CRAN r.jive and original Python AJIVE (`jive`). Neither is approximated or relabeled.',
            '- JIVE/AJIVE fixed to 2 joint + 2 A-individual + 1 G-individual = 5 factors.',
            '- Frozen Z5 retains both shared, one discord, one A-private and G-private; choose discord and A-private by discovery-only X reconstruction (not phenotype). See manifest for indices.',
            '- 7D RawX/FrozenZ/PCA are controls; not fair compression competitors for 5D.',
            '- Common 2005–08 unweighted standardization for all unsupervised methods; later rows unchanged.',
            '- Official JIVE/AJIVE native scores are regressed on discovery X to form an OUT-OF-SAMPLE adapter.',
            '- Adapter R² is measured: a poor adapter invalidates external comparison.',
            '- Training-only CV selects Ridge alpha; PHQ outcomes are never used to learn unsupervised scores.',
            '- MSE differences computed on same patients and weighted point estimates.',
            '- Later NHANES periods previously explored; evidence is exploratory, not untouched confirmation.',
            '',f'Failures/omissions: `{json.dumps(errors)}`.','',
            '## Rank-matched physiology reconstruction (5D versus 5D)', '```text',
            rec[rec.dimension==5].pivot(index='method',columns='period',values='reconstruction_R2').round(5).to_string(),
            '```','',
            '## All-7D reconstruction controls', '```text',
            rec[rec.dimension==7].pivot(index='method',columns='period',values='reconstruction_R2').round(5).to_string(),
            '```','',
            '## Frozen biomarker-fingerprint transport (higher cosine = more similar)', '```text',
            fp.groupby(['method','period'])['marker_fingerprint_cosine'].median().unstack().round(5).to_string(),
            '```','',
            'Note: This measures similarity of biomarker correlations for FROZEN axes. It is not independent re-discovery or biological identifiability.',
            '',
            '## Training-CV phenotype weighted MSE (lower better)', '```text',
            cvtab.pivot(index='method',columns='outcome',values='discovery_CV_MSE').round(5).to_string(),
            '```','',
            '## Later phenotype weighted R² (higher better)', '```text',
            pv[pv.period!=PERIODS[0]].pivot_table(index=['outcome','method'],columns='period',values='weighted_R2').round(5).to_string(),
            '```','',
            '## Paired MSE gain over equal-dimension PCA baseline (positive = better)', '```text',
            paired.round(5).to_string(index=False) if len(paired) else 'No comparisons',
            '```','',
            '## Interpretation guardrails',
            '- Do not equate an enforced private loading zero with a discovered independent biological factor.',
            '- Shared/private scores differ by method definition; raw cross-block sensitivity is descriptive, not a mechanistic proof.',
            '- Rank selection is fixed here; test alternative prespecified ranks and official automatic rank methods next.',
            '- AJIVE/JIVE adapters are linear extensions of discovery scores. Refit ranks only in discovery in future analyses.',
            '- Reconstruction R² is relative to discovery mean and normalized by frozen raw biomarker standardization.',
            '- Bootstrap intervals resample participants rather than NHANES PSUs/strata; not design-based inference.',
            '- Equal or better performance on these metrics alone cannot prove methodological novelty or clinical relevance.',
            '- The best case is consistent improvement in matched-rank reconstruction and temporal prediction WITH credible physiological interpretability, then fully independent data.',
            '- The null is also scientifically useful: JIVE/AJIVE may perform equally well or better, narrowing our actual contribution.',
            '', '## Required next step for defensible novelty',
            'Repeat JIVE/AJIVE rank settings, add population-shift stress tests and external MIDUS/MIDJA, and compare physiological interpretations with domain experts.']
    (out/'103_readout.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\nSaved:',out)
    print('Read:',out/'103_readout.md')


if __name__=='__main__': main()
