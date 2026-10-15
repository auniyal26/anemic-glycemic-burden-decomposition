#!/usr/bin/env python3
"""SCRIPT 05 | Outcome-blind NHANES glycemic reference -> MIDUS prospective transfer.

Run from repository root:
    python External_Validation/05_midus_frozen_glycemic_transfer.py

Reads, locally:
    EDA/Results/73_frozen_phenotype_map_input.csv  (2005-08 discovery ONLY)
    data/MIDUS/raw/  (8 existing Core/Refresher biomarker/cognitive/survey files)

This is a NEW, NHANES-discovery-frozen G3 PCA benchmark, NOT the original
7-dimensional shared/discordant NHANES transform from Script 72/73. The original
models and files are never changed. No MIDUS outcome is used to learn physiology.

Primary: BTACT composite. Secondary: memory/executive. Chronological rule:
    baseline BTACT month < biomarker month < follow-up BTACT month.

Models (same cohort/domain rows): covariates; + HbA1c; + HOMA-IR;
+ raw G3; + NHANES PCA1/PCA2/PCA3. 5-fold CV in Core only.
Locked Core models are evaluated without recalibration in Refresher.
All record-level data stay in memory; only aggregate reports are written.
MIDUS data use subject IDs locally; never export them or participant-level values.
Counts of 0..10 are suppressed, following portal policy.

Dependencies: numpy, pandas, scikit-learn, pyreadstat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "MIDUS" / "raw"
SOURCE = ROOT / "EDA" / "Results" / "73_frozen_phenotype_map_input.csv"
RESULTS = ROOT / "External_Validation" / "Results"
SEED, FOLDS, ALPHA, BOOT = 20261015, 5, 10., 1000
LAB = ("HbA1c", "Glucose", "logInsulin")
DOMAIN = ("comp", "memory", "executive")
SMALL = "SUPPRESSED_0_TO_10"

FILES = {
    "M2_BIO": r"^M2_BIO_AGGREGATE_.*\.sav$",
    "M2_COG": r"^M2_P3_BTACT_.*\.sav$",
    "M2_SURVEY": r"^M2_P1_SURVEY_.*\.sav$",
    "M3_COG": r"^M3_P3_BTACT_.*\.sav$",
    "MR1_BIO": r"^MR1_BIO_AGGREGATE_.*\.sav$",
    "MR1_COG": r"^MR1_P3_BTACT_.*\.sav$",
    "MR1_SURVEY": r"^MR1_P1_SURVEY_.*\.sav$",
    "MR2_COG": r"^MR2_P3_BTACT.*\.sav$",
}
FIELDS = {
    "M2_BIO": {"HbA1c":"B4BHA1C","Glucose":"B4BGLUC","Insulin":"B4BINSLN", "Age":"B4ZAGE", "Sex":"B1PRSEX", "year":"B4ZCOMPY","month":"B4ZCOMPM"},
    "M2_COG": {"comp":"B3TCOMPZ1","memory":"B3TEMZ1","executive":"B3TEFZ1", "year":"B3PIDATE_YR","month":"B3PIDATE_MO"},
    "M2_SURVEY": {"Education":"B1PB1","year":"B1PIDATE_YR","month":"B1PIDATE_MO"},
    "M3_COG": {"comp":"C3TCOMP","memory":"C3TEM","executive":"C3TEF", "year":"C3IDATE_YR","month":"C3IDATE_MO"},
    "MR1_BIO": {"HbA1c":"RA4BHA1C","Glucose":"RA4BGLUC","Insulin":"RA4BINSLN", "Age":"RA4ZAGE","Sex":"RA1PRSEX", "year":"RA4ZCOMPY","month":"RA4ZCOMPM"},
    "MR1_COG": {"comp":"RA3TCOMPZ1","memory":"RA3TEMZ1","executive":"RA3TEFZ1", "year":"RA3IDATE_YR","month":"RA3IDATE_MO"},
    "MR1_SURVEY": {"Education":"RA1PB1","year":"RA1PIDATE_YR","month":"RA1PIDATE_MO"},
    "MR2_COG": {"comp":"RB3TCOMPZ_Long","memory":"RB3TEMZ_Long","executive":"RB3TEFZC_Long", "year":"RB3IDATE_YR","month":"RB3IDATE_MO"},
}
COHORTS = {
    "CORE": ("M2_BIO", "M2_COG", "M3_COG", "M2_SURVEY"),
    "REFRESHER": ("MR1_BIO", "MR1_COG", "MR2_COG", "MR1_SURVEY"),
}
SPECS = ("covariates", "HbA1c", "HOMA_IR", "raw_G3", "NHANES_PC1", "NHANES_PC2", "NHANES_PC3")


def disclose(n: int) -> str:
    return str(int(n)) if int(n) > 10 else SMALL


def numeric(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf],np.nan)


def date_index(df, prefix):
    y, m = numeric(df[prefix+"year"]),numeric(df[prefix+"month"])
    valid = y.between(1990,2030) & m.between(1,12) & y.eq(np.floor(y)) & m.eq(np.floor(m))
    return (12*y + m).where(valid)


def source_file(key):
    matches = [p for p in RAW.rglob('*') if p.is_file() and re.fullmatch(FILES[key], p.name, re.I)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one file for {key}; found {[p.name for p in matches]}")
    return matches[0]


def load_midus(key):
    try:
        import pyreadstat
    except ImportError as e:
        raise RuntimeError('Install pyreadstat: python -m pip install pyreadstat') from e
    p = source_file(key)
    _, meta = pyreadstat.read_sav(str(p), metadataonly=True)
    lookup = {n.upper():n for n in meta.column_names}
    id_col = 'MRID' if key.startswith('MR') else 'M2ID'
    required = [id_col,*FIELDS[key].values()]
    missing = [s for s in required if s.upper() not in lookup]
    if missing:
        raise RuntimeError(f"Missing source fields in {key}: {missing}; no fallback or guessing")
    cols = list(dict.fromkeys(lookup[x.upper()] for x in required))
    frame, _ = pyreadstat.read_sav(str(p), usecols=cols, user_missing=False, apply_value_formats=False)
    frame = frame.rename(columns={lookup[src.upper()]:alias for alias,src in FIELDS[key].items()})
    index_name = lookup[id_col.upper()]
    if frame[index_name].isna().any() or frame[index_name].duplicated().any():
        raise RuntimeError(f"Invalid, missing or duplicate person identifiers in {key}")
    frame = frame.set_index(index_name)
    assert frame.index.is_unique
    return frame, p


def frozen_nhanes():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing genuine NHANES reference: {SOURCE}. Run prior Script 73 locally; no substitute is used.")
    need = ['CYCLE','ELIGIBLE_ADULT_NONPREG','SURVEY_WT','LBXGH','LBXGLU','LOG_IN']
    columns = pd.read_csv(SOURCE, nrows=0).columns
    if not set(need).issubset(columns):
        raise RuntimeError(f"NHANES source missing columns {sorted(set(need)-set(columns))}")
    d = pd.read_csv(SOURCE, usecols=need, low_memory=False)
    cyc = d.CYCLE.astype(str).str.strip().str.replace(r'\.0$','',regex=True).str.zfill(4)
    d = d.loc[cyc.isin(['0506','0708']) & numeric(d.ELIGIBLE_ADULT_NONPREG).eq(1)].copy()
    w = numeric(d.SURVEY_WT)
    x = np.column_stack([numeric(d['LBXGH']),numeric(d['LBXGLU']),numeric(d['LOG_IN'])])
    ok = (np.all(np.isfinite(x),axis=1) & np.isfinite(w.to_numpy()) & (w.to_numpy()>0)
          & (x[:,0]>=2.5)&(x[:,0]<=20)&(x[:,1]>=40)&(x[:,1]<=600))
    x = x[ok]; w = w.to_numpy(dtype=float)[ok]
    if len(x) < 200:
        raise RuntimeError('Too few valid NHANES discovery observations; frozen PCA not fit')
    w = w/w.sum()
    mu = np.sum(w[:,None]*x,axis=0)
    sd = np.sqrt(np.sum(w[:,None]*(x-mu)**2,axis=0))
    if np.any(sd<=1e-9):
        raise RuntimeError('NHANES G3 variance degenerate')
    z = (x-mu)/sd
    cov = z.T@(w[:,None]*z)
    vals, vecs = np.linalg.eigh(cov)
    ix = np.argsort(vals)[::-1]
    vals, vecs = vals[ix], vecs[:,ix]
    # Deterministic sign orientation; does not use phenotype information.
    for j in range(3):
        k = int(np.argmax(np.abs(vecs[:,j])))
        if vecs[k,j]<0: vecs[:,j]*=-1
    if not np.allclose(vecs.T@vecs,np.eye(3),atol=1e-10):
        raise RuntimeError('Frozen PCA basis is nonorthogonal')
    return dict(mean=mu,sd=sd,basis=vecs,evals=vals,n=len(x),src=str(SOURCE.relative_to(ROOT)),
                sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())


def panel(df):
    h, g, i = [numeric(df['bio_'+v]).to_numpy(float) for v in ['HbA1c','Glucose','Insulin']]
    valid = np.isfinite(h)&np.isfinite(g)&np.isfinite(i)&(h>=2.5)&(h<=20)&(g>=40)&(g<=600)&(i>0)&(i<=500)
    with np.errstate(invalid='ignore',divide='ignore'):
        logi = np.where(i>0,np.log(i),np.nan)
    out = np.column_stack([h,g,logi])
    return valid,out


def prepare_cohort(cohort, data):
    bi,ba,fu,su = COHORTS[cohort]
    b = data[bi].add_prefix('bio_')
    a = data[ba].add_prefix('base_')
    f = data[fu].add_prefix('follow_')
    s = data[su].add_prefix('survey_')
    # inner join 3 required files; survey left joined then checked against specimen date.
    x = b.join(a, how='inner').join(f,how='inner').join(s,how='left')
    tb,t0,t1,ts = [date_index(x,p) for p in ('bio_','base_','follow_','survey_')]
    valid_lab, g3 = panel(x)
    good = (valid_lab & t0.lt(tb) & tb.lt(t1) & ts.lt(tb)
            & numeric(x.bio_Age).between(18,110)
            & numeric(x.bio_Sex).isin([1,2])
            & numeric(x.survey_Education).between(1,20)
            & np.isfinite(g3).all(axis=1))
    q = pd.DataFrame(index=x.index)
    q['Age'] = numeric(x.bio_Age)
    q['Sex'] = numeric(x.bio_Sex)
    q['Education'] = numeric(x.survey_Education)
    q['gap_baseline_to_blood'] = tb-t0
    q['gap_blood_to_later'] = t1-tb
    q['HbA1c'] = g3[:,0]
    q['Glucose'] = g3[:,1]
    q['logInsulin'] = g3[:,2]
    insulin = numeric(x.bio_Insulin).to_numpy(float)
    glucose = numeric(x.bio_Glucose).to_numpy(float)
    q['HOMA_IR'] = glucose*insulin/405.0  # Candidate fasting HOMA-IR; confirmation pending
    for d in DOMAIN:
        q['base_'+d] = numeric(x['base_'+d])
        q['follow_'+d] = numeric(x['follow_'+d])
    q = q.loc[good].copy()
    if q.index.duplicated().any():
        raise RuntimeError(f'{cohort}: duplicate merged participant ID')
    counts = dict(linked=len(x),lab_ordered_cov=len(q))
    return q, counts


def cohort_domain(q, domain):
    sel = q['base_'+domain].between(-10,10) & q['follow_'+domain].between(-10,10)
    return q.loc[sel].copy()


def add_pcs(data, f):
    g = data.loc[:, LAB].to_numpy(float)
    z = (g-f['mean'])/f['sd']
    p = z @ f['basis']
    for k in range(3): data['G_PC'+str(k+1)] = p[:,k]
    return data


def covariance_audit(cohort, d, f):
    g = d.loc[:, LAB].to_numpy(float)
    if len(g)<30: raise RuntimeError(f'{cohort}: too few G3 measurements for covariance audit')
    z = (g-f['mean'])/f['sd']
    p = z@f['basis']
    res=[]
    for k in (1,2,3):
        zhat = p[:,:k]@f['basis'][:,:k].T
        mse = float(np.mean((z-zhat)**2))
        scale = float(np.mean(z**2))
        res.append(dict(cohort=cohort, n=disclose(len(z)), k=k,
                        standardized_rMSE=math.sqrt(mse),
                        reference_energy_retained=(1-mse/scale if scale>0 else np.nan)))
    local_z = z-z.mean(axis=0)
    _, _, vh = np.linalg.svd(local_z, full_matrices=False)
    local_pc = vh[0]
    cosine = abs(float(f['basis'][:,0]@local_pc))
    diag=dict(cohort=cohort,n=disclose(len(g)),first_PC_abs_cosine=cosine)
    for j,c in enumerate(LAB):
        diag['frozen_z_mean_'+c] = float(z[:,j].mean())
        diag['frozen_z_sd_'+c] = float(z[:,j].std(ddof=0))
    return res,diag


def feature_frame(data):
    """Education category basis fixed by Core, never inferred from Refresher."""
    base = pd.DataFrame(index=data.index)
    base['base_BTACT'] = data['base_score'].astype(float)
    base['Age'] = data['Age'].astype(float)
    base['gap_baseline_to_blood_years'] = data['gap_baseline_to_blood'].astype(float)/12.0
    base['gap_blood_to_later_years'] = data['gap_blood_to_later'].astype(float)/12.0
    return base


def encode_covariates(core, refresh):
    x0=feature_frame(core)
    x1=feature_frame(refresh)
    levels={}
    for var in ('Sex','Education'):
        if not np.isclose(core[var].dropna().to_numpy(float), np.round(core[var].dropna().to_numpy(float))).all():
            raise RuntimeError(f'{var} Core coding is not integral; audit categories before modeling')
        if not np.isclose(refresh[var].dropna().to_numpy(float), np.round(refresh[var].dropna().to_numpy(float))).all():
            raise RuntimeError(f'{var} Refresher coding is not integral; audit categories before modeling')
        lv=sorted(set(int(i) for i in core[var].dropna().unique()))
        levels[var]=lv
        if len(lv)<2: raise RuntimeError(f'{var} has only one Core level')
        for v in lv[1:]:
            name=f'{var}={v}'
            x0[name]=core[var].eq(v).astype(float)
            x1[name]=refresh[var].eq(v).astype(float)
    unseen = {var: sorted(set(refresh[var].dropna().astype(int)) - set(lv)) for var,lv in levels.items()}
    if any(unseen.values()):
        raise RuntimeError(f'Unseen Refresher category codes {unseen}; verify cross-cohort education/sex harmonization, no silent reference encoding')
    return x0,x1,levels,unseen


def model_features(spec):
    if spec=='covariates': return []
    if spec=='HbA1c': return ['HbA1c']
    if spec=='HOMA_IR': return ['HOMA_IR']
    if spec=='raw_G3': return list(LAB)
    if spec.startswith('NHANES_PC'):
        return ['G_PC'+str(k) for k in range(1,int(spec[-1])+1)]
    raise ValueError(spec)


def fit_model(X, y):
    sc=StandardScaler()
    sx=sc.fit_transform(X)
    model=Ridge(alpha=ALPHA)
    model.fit(sx,y)
    return sc,model


def pred(sc, model, X):
    return np.asarray(model.predict(sc.transform(X)),dtype=float)


def stats(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if len(y)!=len(p) or len(y)<=10 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise RuntimeError('Bad or suppressed metric input')
    mse=float(np.mean((y-p)**2)); rmse=math.sqrt(mse)
    v=float(np.mean((y-y.mean())**2))
    return dict(MSE=mse, RMSE=rmse, MAE=float(np.mean(np.abs(y-p))), R2=(1-mse/v if v>0 else None), bias=float(np.mean(p-y)))


def paired_ci(y,p_base,p_alt,n_boot,seed):
    """Paired bootstrap external improvement: + means lower MSE than baseline."""
    e_base=(y-p_base)**2; e_alt=(y-p_alt)**2
    delta=e_base-e_alt
    if len(delta)<=10:return (None,None,None)
    gen=np.random.default_rng(seed)
    boots=np.empty(n_boot,dtype=float)
    for b in range(n_boot):
        take=gen.integers(0,len(delta),len(delta))
        boots[b]=delta[take].mean()
    return float(delta.mean()),float(np.quantile(boots,.025)),float(np.quantile(boots,.975))


def run_models(core, refresher, d):
    train=cohort_domain(core,d)
    test=cohort_domain(refresher,d)
    if len(train)<100 or len(test)<100:
        raise RuntimeError(f'{d}: too few analyzable participants in either cohort')
    train['base_score']=train['base_'+d]
    test['base_score']=test['base_'+d]
    c0,c1,levels,unseen=encode_covariates(train,test)
    y0=train['follow_'+d].to_numpy(float)
    y1=test['follow_'+d].to_numpy(float)
    cv=KFold(n_splits=FOLDS,shuffle=True,random_state=SEED)
    rows=[]; external_preds={}
    for spec in SPECS:
        cols=model_features(spec)
        a=pd.concat([c0,train[cols]],axis=1).to_numpy(float)
        b=pd.concat([c1,test[cols]],axis=1).to_numpy(float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f'Nonfinite model matrix {d}:{spec}')
        pred_cv=np.empty(len(y0),dtype=float)
        for tri,vi in cv.split(a):
            s,m=fit_model(a[tri],y0[tri])
            pred_cv[vi]=pred(s,m,a[vi])
        st_cv=stats(y0,pred_cv)
        s,m=fit_model(a,y0)
        pt=pred(s,m,b)
        external_preds[spec]=pt
        st_ext=stats(y1,pt)
        rows.extend([dict(domain=d, model=spec, cohort='CORE',evaluation='five_fold_CV',n=disclose(len(y0)),**st_cv),
                     dict(domain=d, model=spec, cohort='REFRESHER',evaluation='locked_external',n=disclose(len(y1)),**st_ext)])
    gains=[]
    for j,spec in enumerate(SPECS[1:]):
        gain,low,high=paired_ci(y1,external_preds['covariates'],external_preds[spec],BOOT,SEED+100*j+DOMAIN.index(d))
        gains.append(dict(domain=d,model=spec,reference='covariates',n=disclose(len(y1)),
                          external_MSE_reduction=gain,ci95_low=low,ci95_high=high,
                          interpretation='descriptive paired bootstrap, not formal transport or causal proof'))
    return rows,gains,dict(domain=d,core_n=disclose(len(train)),refresher_n=disclose(len(test)),
                           core_education_levels=levels['Education'],unseen_refresh_levels=unseen)


def fmt(v):
    if v is None or (isinstance(v,(float,np.floating)) and not math.isfinite(v)):
        return 'NA'
    return f'{float(v):.4f}'


def write_report(f, samples, cov_audits, drifts, metrics, gains, metadata, args):
    out=['# Script 05 — Frozen NHANES G3 representation and prospective MIDUS transfer','',
         '**Status:** Research analysis; source-codebook verification, survey selection and score invariance remain open.',
         '**Scope:** A NEW NHANES-2005–08 outcome-blind **G3-only** PCA benchmark, NOT original NHANES 7D shared/discordant decomposition.',
         '**Privacy:** No participant-level data, IDs, fitted individual predictions or per-person scores exported. Cells 0–10 suppressed.','',
         '## Reference physiology and lock','',
         f"- Training data: `{f['src']}`; discovery 2005–08 eligible NHANES only; N = {disclose(f['n'])}.",
         '- Variables, in order: HbA1c (%), fasting glucose (mg/dL), natural log fasting insulin (uIU/mL).',
         '- Sample weights used ONLY to learn NHANES discovery means/SDs/covariance/eigenvectors.',
         '- No MIDUS physiology or phenotype used to fit the NHANES reference; no NHANES outcome used.',
         '- PC2/PC3 are compressions with possible reconstruction loss; full 3-PC invertibility is algebraic, not discovery of extra information.',
         '- NHANES original frozen 7D transforms, outcomes and files: unchanged.',
         '', '## Cohort selection','',
         '| Cohort | linked bio+earlier/later cognition | ordered, lab and basic covariates |',
         '|---|---:|---:|']
    for k,v in samples.items():out.append(f"| {k} | {disclose(v['linked'])} | {disclose(v['lab_ordered_cov'])} |")
    out += ['', '## Reconstruction and physiology stability', '',
            'Reconstruction rMSE is in *NHANES-frozen standardized G3 units*. "Energy retained" uses NHANES origin, not MIDUS-centered variance.','','| Cohort | Retained PCs | Standardized rMSE | NHANES-origin energy retained |',
            '|---|---:|---:|---:|']
    for a in cov_audits:out.append(f"| {a['cohort']} | {a['k']} | {fmt(a['standardized_rMSE'])} | {fmt(a['reference_energy_retained'])} |")
    out += ['', '| Cohort | Abs. cosine: NHANES vs local first PC |', '|---|---:|']
    for x in drifts:out.append(f"| {x['cohort']} | {fmt(x['first_PC_abs_cosine'])} |")
    out += ['', '## Locked cognitive mapping: summary', '',
            'Five-fold shuffled *Core-only* CV; same participants per domain/model, fixed Ridge alpha=10 (after fold-training standardization).',
            'All Refresher predictions use the **unchanged Core-trained models** and Core-trained feature scalers.',
            'Each outcome is a follow-up BTACT reference-standardized score; baseline BTACT, age at blood, sex, earlier education and both visit gaps are baseline features.',
            'Refresher and Core BTACT use different national reference populations: absolute transport metrics are **comparability-conditional**, not proof of phenotype invariance.',
            '', '| Domain | Model | Core CV R2 | Refresher frozen R2 | Refresher ΔMSE vs baseline | 95% bootstrap CI |',
            '|---|---|---:|---:|---:|---|']
    for d in DOMAIN:
        for sp in SPECS:
            cv=next(v for v in metrics if v['domain']==d and v['model']==sp and v['cohort']=='CORE')
            ex=next(v for v in metrics if v['domain']==d and v['model']==sp and v['cohort']=='REFRESHER')
            g=next((v for v in gains if v['domain']==d and v['model']==sp),None)
            ci = f"[{fmt(g['ci95_low'])}, {fmt(g['ci95_high'])}]" if g else 'NA'
            out.append(f"| {d} | {sp} | {fmt(cv['R2'])} | {fmt(ex['R2'])} | {fmt(g['external_MSE_reduction']) if g else 'NA'} | {ci} |")
    out += ['', '## Caveats and stopping rules','',
            '1. The *only* physiological transfer tested is a **G3-only** representation: joint A/G, shared/discordant axes and phenotype→physiology reconstruction are NOT tested.',
            '2. HbA1c, glucose and insulin have source-label units, but assay, fasting compliance, clinical exclusions and cross-study harmonization remain unverified.',
            '3. HOMA-IR is `glucose(mg/dL) × insulin(uIU/mL) / 405`, an unvalidated clinical-index benchmark in this sample.',
            '4. Standardization and predictor fits are trained on NHANES discovery (physiology) or Core (outcomes) only; Refresher is untouched until external evaluation.',
            '5. Equal-dimension raw-G3 and 3-PC OLS models contain equivalent linear information. Ridge penalties may differ under scaling/rotation; no information gain should be claimed.',
            '6. BTACT score comparability, practice effects, survivor/biomarker-selection bias, survey weighting and other confounders require a separate validation. These are unweighted sample metrics.',
            '7. Baseline cognition occurred before the biomarker draw; prospective is not causal, and no model can use blood collected after the outcome.',
            '8. Analyses are hypothesis-generating for cognition and not an exact replication of NHANES depression-PHQ phenotype associations.',
            '9. Check metadata codebooks and category crosswalks, especially education; unfamiliar codes trigger an explicit error instead of being silently mapped.',
            '', '## Output files','',
            '- `05_readout.md` (this interpretation with explicit limitations)',
            '- `05_structure_transfer.csv` (aggregate reconstruction)',
            '- `05_distribution_shift.csv` (aggregate means, SD, first-PC alignment)',
            '- `05_cognitive_performance.csv` (Core CV and frozen Refresher model metrics)',
            '- `05_external_mse_improvement.csv` (paired-bootstrap CI)',
            '- `05_manifest.json` (input provenance, frozen basis and configuration; aggregate parameters only)', '',
            '**Decision gate:** Review result directions and source manuals before claiming stable physiological or cognitive transport.']
    (RESULTS/'05_readout.md').write_text('\n'.join(out)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test',action='store_true',help='Test numerical algebra only; no real datasets')
    args=parser.parse_args()
    if args.smoke_test:
        gen=np.random.default_rng(SEED)
        x=gen.normal(size=(100,3))
        v=np.linalg.qr(gen.normal(size=(3,3)))[0]
        for k in (1,2,3):
            y=x@v[:,:k]
            if k==3: assert np.allclose(y@v.T,x,atol=1e-9)
        y=gen.normal(size=100);b=gen.normal(size=100);z=gen.normal(size=100)
        assert np.isfinite(paired_ci(y,b,z,30,SEED)).all()
        print('SCRIPT 05 SMOKE TEST PASS (numerical invariants; no real data)')
        return
    if not RAW.is_dir():raise FileNotFoundError(f'Missing MIDUS raw folder: {RAW}')
    RESULTS.mkdir(parents=True,exist_ok=True)
    f=frozen_nhanes()
    print(f"  NHANES discovery-only frozen G3 reference | N={f['n']} | source: {f['src']}")
    data={}; files={}
    for role in FILES:
        data[role],p=load_midus(role)
        files[role]={'path':str(p.relative_to(ROOT)),'n':len(data[role])}
        print(f'  {role:11s} | {len(data[role]):5d} records')
    cohorts={}; samples={}
    for c in COHORTS:
        cohorts[c],samples[c]=prepare_cohort(c,data)
        cohorts[c]=add_pcs(cohorts[c],f)
    struct=[];drift=[]
    # Exact analysis-sample anchors: composite-score participants with documented chronology.
    for c in COHORTS:
        d=cohort_domain(cohorts[c],'comp')
        cov,di=covariance_audit(c,d,f)
        struct.extend(cov);drift.append(di)
    outputs=[];gains=[];meta=[]
    for dom in DOMAIN:
        a,b,c=run_models(cohorts['CORE'],cohorts['REFRESHER'],dom)
        outputs.extend(a);gains.extend(b);meta.append(c)
        mainrow=next(x for x in b if x['model']=='NHANES_PC2')
        print(f"  {dom:9s} | Core N={c['core_n']}; Refresher N={c['refresher_n']}; frozen PC2 ΔMSE vs baseline={mainrow['external_MSE_reduction']:.5f}")
    pd.DataFrame(struct).to_csv(RESULTS/'05_structure_transfer.csv',index=False)
    pd.DataFrame(drift).to_csv(RESULTS/'05_distribution_shift.csv',index=False)
    pd.DataFrame(outputs).to_csv(RESULTS/'05_cognitive_performance.csv',index=False)
    pd.DataFrame(gains).to_csv(RESULTS/'05_external_mse_improvement.csv',index=False)
    manifest=dict(script=Path(__file__).name,seed=SEED,folds=FOLDS,ridge_alpha=ALPHA,bootstrap_reps=BOOT,
                  nhanes_source=f['src'],nhanes_sha256=f['sha256'],nhanes_discovery_n=f['n'],
                  g3_order=list(LAB),nhanes_g3_weighted_mean=f['mean'].tolist(),
                  nhanes_g3_weighted_sd=f['sd'].tolist(),frozen_g3_basis_columns=f['basis'].tolist(),
                  frozen_g3_eigenvalues=f['evals'].tolist(),midus_files=files,cohort_counts=samples,
                  domain_counts=meta, models=list(SPECS),outputs=['05_readout.md','05_structure_transfer.csv',
                  '05_distribution_shift.csv','05_cognitive_performance.csv','05_external_mse_improvement.csv','05_manifest.json'],
                  original_7D_map_modified=False, note='NEW G3-only PCA learned from genuine NHANES 05–08 before any MIDUS outcomes; not original full 7D transform')
    (RESULTS/'05_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    write_report(f,samples,struct,drift,outputs,gains,meta,args)
    print('SCRIPT 05 | NHANES G3 frozen physiology + MIDUS prospective transfer | 2 cohorts | 0 model-selection leakage')
    print('OUTPUT: '+str(RESULTS/'05_readout.md'))

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('SCRIPT 05 FAILED: '+str(e),file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
