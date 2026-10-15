#!/usr/bin/env python3
"""Script 08 | Causal-identification stress test: low hemoglobin and HbA1c.

From repository root:
    python External_Validation/08_hba1c_causal_identification_stress_test.py

Uses the same NHANES selected-data extract as Script 07. Outputs ONLY aggregate
summaries. Does NOT claim that a causal effect was identified.

TARGET QUESTION (hypothetical; NOT identified from these data):
    Would modifying the hematological process responsible for low hemoglobin,
    while holding true long-term glucose exposure constant, alter HbA1c?

OBSERVABLE PROXY (estimable, non-causal):
    Weighted, cross-fitted doubly robust adjusted contrast in measured HbA1c
    between adults with versus without sex-threshold low hemoglobin.

The exposure is a *state*, not a well-defined intervention. Glycemia, anemia
etiology, lab values, and covariates are concurrent. Unmeasured red-cell lifespan,
iron status, medications, and glucose history invalidate any unconditional causal
claim. AIPW/double robustness protects against one of two *observed-data model*
misspecifications, NOT against unmeasured confounding or bad timing.

Includes: propensity overlap/positivity, diagnostics of model stability, same-row
covariate-set sensitivity, fasting-glucose specificity (NOT a negative control),
and explicit omitted-confounder bias scenarios. Weighted descriptive estimates
are NOT design-based SEs/CI; no clinical action or diagnostic adjustment should
be based on these results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'EDA' / 'Results' / '73_frozen_phenotype_map_input.csv'
OUTPUT = ROOT / 'External_Validation' / 'Results'
SEED = 20261015
N_FOLDS = 5
SMALL = 'SUPPRESSED_0_TO_10'
PERIODS = {
    '0506':'discovery_2005_08', '0708':'discovery_2005_08',
    '0910':'holdout_2009_18', '1112':'holdout_2009_18',
    '1314':'holdout_2009_18', '1516':'holdout_2009_18',
    '1718':'holdout_2009_18', '2123':'holdout_2021_23',
}
REQUIRED = ['CYCLE','ELIGIBLE_ADULT_NONPREG','SURVEY_WT','RIDAGEYR','RIAGENDR',
            'BMXBMI','EGFR_2021','LBXHGB','LBXGH','LBXGLU','LOG_IN']
OPTIONAL = ['INDFMPIR','RACE','EDUC3','SMOKING3']
BOUNDS = {
    'RIDAGEYR': (18, 110), 'RIAGENDR':(1,2), 'BMXBMI':(10,80),
    'EGFR_2021':(5,200), 'LBXHGB':(5,22), 'LBXGH':(2.5,20),
    'LBXGLU':(40,600), 'LOG_IN':(-2,7),
}
BASIC_NUM = ['RIDAGEYR','BMXBMI','EGFR_2021']
BASIC_CAT = ['RIAGENDR']
EXT_NUM = ['INDFMPIR']
EXT_CAT = ['RACE','EDUC3','SMOKING3']


def count(n: int) -> str:
    return str(int(n)) if n > 10 else SMALL


def get_data(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing real NHANES source: {path}. Never substitute simulated observations.')
    headers = list(pd.read_csv(path,nrows=0).columns)
    missing = [c for c in REQUIRED if c not in headers]
    if missing:
        raise ValueError('Missing required source columns: '+', '.join(missing))
    optional = [c for c in OPTIONAL if c in headers]
    names = REQUIRED + optional + ([c for c in ('SEQN','SDMVPSU','SDMVSTRA') if c in headers])
    d = pd.read_csv(path,usecols=list(dict.fromkeys(names)),low_memory=False)
    d['CYCLE'] = (d['CYCLE'].astype(str).str.strip()
                    .str.replace(r'\.0$', '', regex=True).str.zfill(4))
    d = d.loc[d['CYCLE'].isin(PERIODS)].copy()
    for c in names:
        if c != 'CYCLE':
            d[c] = pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    if 'SEQN' in d and (d.SEQN.isna().any() or d.SEQN.duplicated().any()):
        raise ValueError('SEQN missing or duplicated; cannot safely analyze')
    d = d.loc[d.ELIGIBLE_ADULT_NONPREG.eq(1) & d.SURVEY_WT.gt(0)].copy()
    before = d.groupby(d.CYCLE.map(PERIODS)).size().to_dict()
    mask = pd.Series(True,index=d.index)
    for c,(low,high) in BOUNDS.items():
        mask &= d[c].between(low,high)
    d = d.loc[mask].copy()
    d['PERIOD'] = d.CYCLE.map(PERIODS)
    d['LOW_HB'] = (((d.RIAGENDR==1)&(d.LBXHGB<13)) |
                   ((d.RIAGENDR==2)&(d.LBXHGB<12))).astype(int)
    # For contextual design, all covariates below are concurrent or earlier only
    # in unknown proportions; this is NOT an established pre-treatment set.
    return d, optional, before, ('SDMVPSU' in d and 'SDMVSTRA' in d)


def transformer(num, cat):
    return ColumnTransformer([
        ('numeric', Pipeline([('imputer',SimpleImputer(strategy='median')),
                              ('scale',StandardScaler())]), num),
        ('categorical', Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                                 ('onehot',OneHotEncoder(handle_unknown='ignore'))]), cat),
    ], sparse_threshold=0.0)


def aipw_crossfit(data, outcome, num, cat, seed=SEED, folds=N_FOLDS):
    """Cross-fitted, weighted augmented inverse-propensity adjusted contrast.

    Estimating equation: m1(W)-m0(W)+A*(Y-m1)/e - (1-A)*(Y-m0)/(1-e).
    This is an observational standardized contrast, causal only if consistency,
    exchangeability, positivity, timing and a coherent intervention hold.
    """
    a = data.LOW_HB.to_numpy(dtype=int)
    y = data[outcome].to_numpy(dtype=float)
    w = data.SURVEY_WT.to_numpy(dtype=float)
    n = len(data)
    if n < 200 or min(np.bincount(a,minlength=2)) < 40:
        return None
    X = data[num+cat].copy()
    pp = np.full(n,np.nan)
    mu1 = np.full(n,np.nan)
    mu0 = np.full(n,np.nan)
    f = StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    for tr,te in f.split(X,a):
        # Nuisance models use fold-training features only; no leakage.
        enc = transformer(num,cat)
        xtr = enc.fit_transform(X.iloc[tr])
        xte = enc.transform(X.iloc[te])
        wt = w[tr] / np.mean(w[tr])
        prop = LogisticRegression(C=0.25,max_iter=1500,solver='lbfgs')
        prop.fit(xtr,a[tr],sample_weight=wt)
        pp[te] = prop.predict_proba(xte)[:,1]
        for label,dest in [(1,mu1),(0,mu0)]:
            ix = a[tr]==label
            mod = Ridge(alpha=20,solver='lsqr')
            mod.fit(xtr[ix],y[tr][ix],sample_weight=wt[ix])
            dest[te] = mod.predict(xte)
    if not np.all(np.isfinite(pp+mu1+mu0)):
        raise AssertionError('Out-of-fold predictions incomplete')
    raw_min = float(pp.min()); raw_max = float(pp.max())
    clip = np.clip(pp,0.05,0.95)
    pseudo = mu1-mu0 + a*(y-mu1)/clip - (1-a)*(y-mu0)/(1-clip)
    estimate = float(np.average(pseudo,weights=w))
    denom = np.sum(w)
    eff_w = w * (a/clip + (1-a)/(1-clip))
    ess = float(eff_w.sum()**2 / (eff_w@eff_w))
    # Overlap-sensitive: remove out-of-support predicted propensity values,
    # recompute the same fixed pseudo equation on the trimmed subset.
    support = (pp>=0.05)&(pp<=0.95)
    tmean = float(np.average(pseudo[support],weights=w[support])) if support.sum()>30 else None
    stratified = {}
    for arm in (0,1):
        values = pp[a==arm]
        stratified[f'propensity_{arm}_p05'] = float(np.quantile(values,.05))
        stratified[f'propensity_{arm}_p50'] = float(np.quantile(values,.50))
        stratified[f'propensity_{arm}_p95'] = float(np.quantile(values,.95))
    return {
        'n':int(n),'n_low_hb':int(a.sum()),'n_not_low_hb':int(n-a.sum()),
        'exposure_rate_weighted':float(np.average(a,weights=w)),
        'contrast_aipw':estimate,'outcome_units':'HbA1c percentage points' if outcome=='LBXGH' else 'glucose mg/dL',
        'unadjusted_weighted_difference':float(np.average(y[a==1],weights=w[a==1])-
                                              np.average(y[a==0],weights=w[a==0])),
        'propensity_min':raw_min,'propensity_max':raw_max,
        'n_clipped':int((~support).sum()),'overlap_fraction':float(support.mean()),
        'overlap_trimmed_contrast':tmean,
        'effective_sample_size_weighted_pseudo':ess,
        **stratified,
    }


def safe_output_row(r):
    return {k:(count(v) if k.startswith('n_') and isinstance(v,(int,np.integer)) else v)
            for k,v in r.items()}


def run(source:Path, out:Path):
    d,avail,before,design_cols = get_data(source)
    rows=[]; biasrows=[]
    periods = ['discovery_2005_08','holdout_2009_18','holdout_2021_23']
    for i,period in enumerate(periods):
        q = d.loc[d.PERIOD.eq(period)].copy()
        if len(q)<200:
            raise ValueError(f'Insufficient genuine NHANES coverage in {period}')
        # Compare basic and enhanced adjustment on identical rows whenever feasible.
        expanded = all(c in q.columns for c in (EXT_NUM+EXT_CAT))
        basic_num = BASIC_NUM + ['LBXGLU','LOG_IN']
        cat = BASIC_CAT
        if expanded:
            # Restrict BOTH adjustment models to one matched, expanded complete cohort.
            for c in EXT_NUM:
                q = q.loc[q[c].between(0,8)].copy()
            for c in EXT_CAT:
                q = q.loc[q[c].notna()].copy()
        if len(q)<200 or min(q.LOW_HB.value_counts().reindex([0,1],fill_value=0))<40:
            raise ValueError(f'Insufficient valid low/normal Hb comparison in {period} after matched restriction; do not force an estimate')
        specs=[('glucose_adjusted_basic','LBXGH',basic_num,cat),
               ('glucose_proxy_specificity','LBXGLU',BASIC_NUM,cat)]
        if expanded:
            specs.append(('glucose_adjusted_social','LBXGH',basic_num+EXT_NUM,cat+EXT_CAT))
        for j,(spec,outcome,num,cat) in enumerate(specs):
            # For glucose proxy no HbA1c in W, since alternative outcome should be independent of adjustment.
            m=aipw_crossfit(q,outcome,num,cat,seed=SEED+i*10+j)
            if m is None: raise RuntimeError(f'No overlap for {period} {spec}')
            r={'period':period,'specification':spec,'n_total':count(m['n']),
               'n_low_hb':count(m['n_low_hb']),'n_not_low_hb':count(m['n_not_low_hb']),
               **{k:v for k,v in m.items() if k not in ('n','n_low_hb','n_not_low_hb','n_clipped')},
               'n_clipped':count(m['n_clipped'])}
            rows.append(r)
            print(f'  {period:<20} | {spec:<27} | N={len(q):>5} | contrast={m["contrast_aipw"]:+.4f} {m["outcome_units"]}')
            if outcome=='LBXGH':
                # Explanatory scenario, not formal robustness bound:
                # Bias = (unmeasured binary-U prevalence difference) * (U's
                # additive outcome association), no assumptions about joint U.
                for dprev in (.10,.20,.30):
                    for effect in (.10,.25,.50):
                        biasrows.append({'period':period,'specification':spec,
                             'hypothetical_U_prevalence_gap':dprev,
                             'hypothetical_U_HbA1c_effect_percentage_points':effect,
                             'assumed_bias_percentage_points':dprev*effect,
                             'contrast_minus_assumed_bias':m['contrast_aipw']-dprev*effect})
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'08_observational_causal_stress_test.csv',index=False)
    pd.DataFrame(biasrows).to_csv(out/'08_omitted_confounder_scenarios.csv',index=False)
    manifest = {
        'script':'08_hba1c_causal_identification_stress_test.py',
        'source':'EDA/Results/73_frozen_phenotype_map_input.csv',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'seed':SEED,'crossfit_folds':N_FOLDS,
        'source_has_PSU_and_strata_columns':design_cols,
        'matched_sample_restriction':'common expanded complete cohort when available',
        'exposure':'Low Hb: <13 g/dL men or <12 g/dL women; mixed anemia etiologies, NOT a well-defined intervention',
        'target':'Measured HbA1c; NOT directly observed assay bias or lifetime glycemia',
        'causal_effect_identified':False,'participants_exported':False,
        'extra_adjustment_variables_present':avail,
        'pre_qc_counts':{k:count(int(v)) for k,v in before.items()},
    }
    (out/'08_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    lines=[
        '# Script 08 — Causal-identification stress test: hematology and HbA1c','',
        '**Identification verdict: NOT ESTABLISHED.** AIPW is a causal *estimation method* only when its causal assumptions hold; these cross-sectional data cannot verify them.','',
        '## Explicit causal question','',
        'Would changing the red-cell/iron-deficiency process, without changing average glucose exposure, change the measured HbA1c? **Not directly identifiable here**. Low hemoglobin is a composite clinical state with multiple etiologies, not a specific treatment.',
        '', '## What the data can estimate','',
        'Cross-fitted, survey-weighted AIPW contrasts of sex-threshold low Hb vs other Hb in measured HbA1c; adjustment includes fasting glucose/log-insulin, age, sex, BMI, eGFR and optionally recorded socioeconomic/behavioral fields. Matching uses the same complete cohort within each period. All models are fitted separately by period.','',
        '| NHANES period | Specification | N | Low-Hb N | Adjusted contrast | Propensity clipped N |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(f'| {r["period"]} | {r["specification"]} | {r["n_total"]} | {r["n_low_hb"]} | {r["contrast_aipw"]:+.4f} | {r["n_clipped"]} |')
    lines += [
        '', 'Contrasts are in HbA1c **percentage points** except the `glucose_proxy_specificity` outcome, in glucose **mg/dL**. A null glucose contrast is suggestive of specificity at most: glucose is NOT a validated negative-control outcome.',
        '', '## Causal assumptions and threats','',
        '1. **Well-defined intervention / consistency:** low Hb can reflect iron deficiency, renal disease, inflammation, blood loss, hemoglobin variants, etc.; changing Hb alone is NOT defined by this comparison.',
        '2. **Conditional exchangeability:** unavailable treatment history, iron/ferritin status, RBC lifespan, hemoglobin variants, medication, and long-term glycemia create plausible unmeasured confounding.',
        '3. **Temporal ordering:** Hb, HbA1c, fasting glucose and insulin are measured concurrently; baseline covariates cannot be proven pre-exposure. Conditioning on glucose/insulin could also condition on mediators or colliders.',
        '4. **Reference measurement:** a single fasting glucose cannot establish true integrated multi-month glucose and therefore cannot isolate *HbA1c assay/biology distortion*.',
        '5. **Positivity / overlap:** inspect propensity distributions and clipped counts in the CSV; cross-fitting cannot repair missing support.',
        '6. **Sample/design:** the source is an analytical extract; weights were used for point summaries, but NO complete NHANES PSU/stratum-aware variance estimator was implemented. No publication-grade CI or causal p-value is reported.',
        '7. **Double robustness:** protects against one of the two nuisance models being misspecified *conditional on causal assumptions*; does not remove unmeasured confounding.',
        '', '## Sensitivity analyses','',
        '- The same-row basic vs expanded covariate set tests measured-confounder sensitivity, not identification.',
        '- `08_omitted_confounder_scenarios.csv` applies purely algebraic bias scenarios Δ_prevalence(U) × effect(U) to the contrast. These are NOT formal bounds or inferred strengths of an actual confounder.',
        '- `glucose_proxy_specificity` is a companion association. Fasting glucose is physiologically related to Hb status and does not qualify as a negative control.',
        '', '## What would establish a stronger causal claim?','',
        'An intervention or defensible natural experiment changing iron/red-cell biology with **repeated, appropriately timed HbA1c and CGM or repeated glucose**; account for glucose-affecting effects of treatment, lab assay, drugs and disease progression. A randomized iron-repletion trial among iron-deficient adults (with ethics/clinical supervision) is one concrete route. A rigorous instrument would require a justified exclusion restriction; none is identified here.',
        '', '**Scientific decision:** Treat these results as an identification and falsification audit, NOT as proof of cause, an HbA1c correction rule, or a diagnostic product.',
    ]
    (out/'08_readout.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('SCRIPT 08 | explicit causal estimand + falsification/assumption diagnostics | causal identification: NOT ESTABLISHED')
    print('OUTPUT:',out/'08_readout.md')


def smoke_test():
    rng=np.random.default_rng(42)
    n=1300
    a=rng.binomial(1,.23,n)
    age=rng.normal(48,11,n)
    glucose=rng.normal(100,18,n)
    y=5.6 + .018*(glucose-100) + .15*a + rng.normal(0,.6,n)
    d=pd.DataFrame({'LOW_HB':a,'LBXGH':y,'LBXGLU':glucose,
        'SURVEY_WT':rng.uniform(.5,2.5,n),'RIDAGEYR':age,
        'BMXBMI':rng.normal(26,4,n),'EGFR_2021':rng.normal(94,15,n),
        'RIAGENDR':rng.integers(1,3,n), 'LOG_IN':rng.normal(2.1,.7,n)})
    r=aipw_crossfit(d,'LBXGH',BASIC_NUM+['LBXGLU','LOG_IN'],BASIC_CAT,folds=5)
    assert r is not None and abs(r['contrast_aipw']-.15)<.13, r
    print('SMOKE PASS | numerical synthetic AIPW; no real-data result',r['contrast_aipw'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=SOURCE)
    p.add_argument('--out',type=Path,default=OUTPUT)
    p.add_argument('--smoke-test',action='store_true',help='synthetic numerical test only')
    a=p.parse_args()
    if a.smoke_test: smoke_test();return
    run(a.input,a.out)

if __name__=='__main__': main()
