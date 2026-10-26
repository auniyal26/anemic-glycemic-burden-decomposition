#!/usr/bin/env python3
"""Script 102 — Honest NHANES representation benchmark.

From the root of the existing repository:
    python EDA/102_representation_benchmark.py
    python EDA/102_representation_benchmark.py --folds 3

REQUIRES existing local inputs for Script 98, including DPQ XPTs and the
frozen Script-73 patient-level file. Will never download data, synthesize
participants, modify source data, or refit the user's frozen Z transform.

Compares, on a common matched cohort:
 - Covariates only and Hb/HbA1c scalar baseline
 - Raw seven biomarkers
 - PCA with four, five, or seven coordinates
 - Block PCA with three haematological and three glycaemic coordinates
 - Independently fitted seven-coordinate CCA shared/discord/private baseline
 - User's existing frozen seven-coordinate representation
 - Four-coordinate subset of user's frozen representation

Measures: survey-weighted point-estimate reconstruction, train-only CV,
held-out regression R²/RMSE, component block leakage, and descriptive drift.
JIVE/AJIVE/MOFA+ are *not* implemented or claimed by this script.
This is an exploratory benchmark: previous work inspected the later NHANES
periods repeatedly, and no untouched confirmatory holdout exists here.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / 'EDA' / 'Results'
OUTPUT = RESULTS / '102_representation_benchmark'
SCRIPT98 = HERE / '98_regularized_phenotype_equation.py'
SPLITS = ('train_2005-2008', 'test_2009-2018', 'validate_2021-2023')
X_VARS = ['LBXHGB', 'LBXRBCSI', 'LBXMCVSI', 'LBXRDW',
          'LBXGH', 'LBXGLU', 'LOG_IN']
Z_VARS = ['SP_SHARED1', 'SP_SHARED2', 'SP_DISCORD1', 'SP_DISCORD2',
          'SP_A_PRIVATE3', 'SP_A_PRIVATE4', 'SP_G_PRIVATE3']
Y_VARS = ['PHQ_total', 'Somatic', 'CogAff']
ALPHAS = [0.1, 1., 10., 100., 1000.]
SEED = 102


def weighted_avg(v, w):
    return np.average(v, weights=w, axis=0)


def weighted_cov(a, w):
    a = np.asarray(a, float)
    w = np.asarray(w, float)
    m = weighted_avg(a, w)
    x = a - m
    return (x.T * w) @ x / w.sum()


def invsqrt(mat, eps=1e-8):
    eig, vec = np.linalg.eigh((mat + mat.T)/2)
    return (vec * (1. / np.sqrt(np.maximum(eig, eps)))) @ vec.T


def ratio_r2(y, pred, weights):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    w = np.asarray(weights, float)
    mu = np.average(y, weights=w)
    den = np.average((y-mu)**2, weights=w)
    mse = np.average((y-pred)**2, weights=w)
    return float(1 - mse/den) if den > 0 else np.nan


def reconstruction_r2(xs, pred, w):
    """Equal weight per raw marker after scaling to discovery SDs."""
    xs = np.asarray(xs, float)
    pred = np.asarray(pred, float)
    mu = np.average(xs, axis=0, weights=w)
    den = float(np.average(np.sum((xs-mu)**2, axis=1), weights=w))
    num = float(np.average(np.sum((xs-pred)**2, axis=1), weights=w))
    return float(1-num/den) if den > 0 else np.nan


def encode_matrices():
    if not SCRIPT98.exists():
        raise FileNotFoundError(f'Missing {SCRIPT98}; place this script in EDA/')
    spec = importlib.util.spec_from_file_location('script98_for_benchmark102', SCRIPT98)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    # The module only initializes these globals inside its main().
    mod.yield_msg = []
    mod.normalizer = {}
    mod.categories_reference = {}
    mod.categories_feature_names = []
    data = mod.load_real()  # exact Script-98 complete-case + original DPQ item rules
    feats = mod.build_features(data)
    mod.categories_feature_names = [c for c in feats.columns if '=' in c]
    covars = mod.covariates('full')
    required = list(X_VARS) + list(Z_VARS) + ['CYCLE', 'SPLIT', 'SURVEY_WT']
    if not set(required).issubset(data.columns):
        raise ValueError('Missing required columns in Script-98 matched cohort')
    vals = data[X_VARS + Z_VARS].to_numpy(float)
    if not np.isfinite(vals).all():
        raise ValueError('Nonfinite complete physiological input')
    if not (data['SURVEY_WT'].to_numpy(float) > 0).all():
        raise ValueError('Invalid survey weights')
    for s in SPLITS:
        n = int((data.SPLIT == s).sum())
        if n < 100:
            raise ValueError(f'Insufficient real participants {s}: {n}')
    return mod, data, feats[covars].to_numpy(float), covars


class WeightedPCA:
    """SVD PCA with discovery survey weights; no outcome information."""
    def __init__(self, n_components):
        self.n_components = int(n_components)

    def fit(self, xx, w):
        xx = np.asarray(xx, float)
        w = np.asarray(w, float)
        self.mean_ = weighted_avg(xx, w)
        z = (xx-self.mean_) * np.sqrt(w[:, None]/w.mean())
        _, _, v = np.linalg.svd(z, full_matrices=False)
        self.components_ = v[:self.n_components]
        return self

    def transform(self, xx):
        return (np.asarray(xx, float)-self.mean_) @ self.components_.T


class Representation:
    def __init__(self, name, k, transform, coef=None, mean=None):
        self.name, self.k, self.transformer = name, k, transform
        self.coef, self.mean = coef, mean

    def project(self, x_scaled, z_frozen):
        if self.name.startswith('FrozenZ'):
            if self.name == 'FrozenZ7':
                return np.asarray(z_frozen, float)
            return np.asarray(z_frozen, float)[:, [0, 1, 2, 3]]
        return self.transformer(x_scaled)


def fit_cca_transform(x_scaled, w):
    """Fit whitened full-basis 4-by-3 CCA; return 7D shared/private transform.

    This is a true linear CCA calculation with an independently fitted basis.
    It is NOT a general JIVE or MOFA implementation.
    """
    mu = weighted_avg(x_scaled, w)
    xc = x_scaled-mu
    a, g = xc[:, :4], xc[:, 4:]
    c_aa, c_gg = weighted_cov(a, w), weighted_cov(g, w)
    c_ag = (a.T*w)@g/w.sum()
    w_a, w_g = invsqrt(c_aa), invsqrt(c_gg)
    u, sval, vt = np.linalg.svd(w_a@c_ag@w_g, full_matrices=True)
    Wa, Wg = w_a@u, w_g@vt.T
    rho = np.clip(sval[:2], -0.999999, 0.999999)

    def transform(xx):
        xx = np.asarray(xx, float)-mu
        ua = xx[:, :4] @ Wa
        ug = xx[:, 4:] @ Wg
        shared = (ua[:, :2]+ug[:, :2])/np.sqrt(2*(1+rho))
        discord = (ua[:, :2]-ug[:, :2])/np.sqrt(2*(1-rho))
        return np.column_stack([shared, discord, ua[:, 2], ua[:, 3], ug[:, 2]])

    return transform, np.asarray(sval[:3], float)


def fit_representations(x_train, z_train, w):
    ret = [Representation('RawX7', 7, lambda x: x),
           Representation('FrozenZ7', 7, None),
           Representation('FrozenZ4_shared_discord', 4, None)]
    for k in [4, 5, 7]:
        p = WeightedPCA(k).fit(x_train, w)
        ret.append(Representation(f'PCA{k}', k, p.transform))
    pa = WeightedPCA(3).fit(x_train[:, :4], w)
    pg = WeightedPCA(3).fit(x_train[:, 4:], w)
    ret.append(Representation('BlockPCA6', 6,
         lambda x: np.column_stack((pa.transform(x[:, :4]), pg.transform(x[:, 4:])))))
    cca, rho = fit_cca_transform(x_train, w)
    ret.append(Representation('IndependentCCA7', 7, cca))
    return ret, rho


def fit_decoder(rep_train, x_train, w):
    """Train-only affine least-squares decoder (no outcome leakage)."""
    m_r = weighted_avg(rep_train, w)
    m_x = weighted_avg(x_train, w)
    a = rep_train-m_r
    b = x_train-m_x
    coef = np.linalg.lstsq(a*np.sqrt(w[:, None]), b*np.sqrt(w[:, None]), rcond=1e-12)[0]
    return m_r, m_x, coef


def decode(r, decoder):
    mr, mx, coef = decoder
    return (r-mr)@coef+mx


def drift_score(train, later, wa, wb):
    """Descriptive, coordinate-dependent; not a validation of source stability."""
    mu = weighted_avg(train, wa)
    sds = np.sqrt(np.maximum(np.diag(weighted_cov(train, wa)), 1e-10))
    t = (train-mu)/sds
    v = (later-mu)/sds
    wm = float(np.sqrt(np.mean(weighted_avg(v, wb)**2)))
    cov_delta = float(np.linalg.norm(weighted_cov(v, wb)-weighted_cov(t, wa), ord='fro')/math.sqrt(train.shape[1]))
    return wm, cov_delta


def ridge_fit_predict(x_train, y_train, w_train, x_other, alpha):
    scaler = StandardScaler()
    f = scaler.fit_transform(x_train, sample_weight=w_train)
    ridge = Ridge(alpha=alpha)
    ridge.fit(f, y_train, sample_weight=w_train/w_train.mean())
    return ridge.predict(scaler.transform(x_other))


def evaluate_cv(x, y, w, tr_ids, folds):
    scores = []
    for a in ALPHAS:
        mses = []
        for aa, bb in folds:
            it, iv = tr_ids[aa], tr_ids[bb]
            pred = ridge_fit_predict(x[it], y[it], w[it], x[iv], a)
            mses.append(float(np.average((y[iv]-pred)**2, weights=w[iv])))
        scores.append((float(np.mean(mses)), a))
    return min(scores, key=lambda v: (v[0], v[1]))


def audit_leakage(x_standard, z, w, var_names):
    # Only audit explicitly designated private axes; identity is by construction.
    mx0 = weighted_avg(x_standard, w)
    mz = weighted_avg(z, w)
    xx = (x_standard-mx0)*np.sqrt(w[:, None])
    zz = (z-mz)*np.sqrt(w[:, None])
    jac = np.linalg.lstsq(xx, zz, rcond=1e-12)[0]  # 7 raw x 7 Z
    rows = []
    for j, name in enumerate(var_names):
        if not ('PRIVATE' in name):
            continue
        a = float(np.sum(jac[:4, j]**2))
        g = float(np.sum(jac[4:, j]**2))
        total = a+g
        unexpected = g if 'A_PRIVATE' in name else a
        rows.append({'component':name,'unexpected_block_loading_fraction':unexpected/total if total else np.nan})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5, help='3-10 training CV folds (default 5)')
    ap.add_argument('--out', type=Path, default=OUTPUT)
    args = ap.parse_args()
    if not 3 <= args.folds <= 10:
        ap.error('--folds must be between 3 and 10')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    mod, d, covariates, covnames = encode_matrices()
    splits = d.SPLIT.astype(str).to_numpy()
    cycles = d.CYCLE.astype(str).to_numpy()
    w = d.SURVEY_WT.to_numpy(float)
    original_x = d[X_VARS].to_numpy(float)
    z = d[Z_VARS].to_numpy(float)
    tr_idx = np.flatnonzero(splits == SPLITS[0])
    xnorm = StandardScaler().fit(original_x[tr_idx], sample_weight=w[tr_idx])
    x = xnorm.transform(original_x)
    reps, rho = fit_representations(x[tr_idx], z[tr_idx], w[tr_idx])
    transformed = {r.name:r.project(x, z) for r in reps}
    names = ['CovariatesOnly', 'Hb_HbA1c'] + [r.name for r in reps]
    preds = {'CovariatesOnly':covariates,
             'Hb_HbA1c': np.column_stack([covariates, x[:, [0, 4]]])}
    for r in reps:
        preds[r.name] = np.column_stack([covariates, transformed[r.name]])

    counts = {s:int((splits==s).sum()) for s in SPLITS}
    if counts != {'train_2005-2008':3815, 'test_2009-2018':9893, 'validate_2021-2023':2247}:
        print('NOTICE: cohort counts differ from published Script-98 result:', counts)
    print('Real matched cohort:', counts)
    print('Independently fitted CCA canonical correlations:', np.round(rho, 5))

    reconstruction = []
    drift = []
    for rep in reps:
        r = transformed[rep.name]
        decoder = fit_decoder(r[tr_idx], x[tr_idx], w[tr_idx])
        rec = decode(r, decoder)
        for s in SPLITS:
            idx = splits == s
            reconstruction.append(dict(method=rep.name, dimension=rep.k, period=s,
                 n=int(idx.sum()), weighted_reconstruction_R2=reconstruction_r2(x[idx],rec[idx],w[idx])))
            if s != SPLITS[0]:
                mshift, cshift = drift_score(r[tr_idx],r[idx],w[tr_idx],w[idx])
                drift.append(dict(method=rep.name, dimension=rep.k, period=s,
                     standardized_mean_shift=mshift, covariance_shift=cshift))

    cv_folds = list(StratifiedKFold(n_splits=args.folds, shuffle=True,random_state=SEED).split(tr_idx, cycles[tr_idx]))
    performance, cv_records = [], []
    ycols = {'PHQ_total':'PHQ9_TOTAL','Somatic':'SOMATIC_SCORE','CogAff':'COGAFF_SUM'}
    for yname, ycol in ycols.items():
        y = d[ycol].to_numpy(float)
        for method in names:
            features = np.asarray(preds[method], float)
            mse, alpha = evaluate_cv(features,y,w,tr_idx,cv_folds)
            cv_records.append({'outcome':yname,'method':method,'dimension':features.shape[1],
                  'train_cv_weighted_MSE':mse,'selected_ridge_alpha':alpha})
            pred = ridge_fit_predict(features[tr_idx],y[tr_idx],w[tr_idx],features,alpha)
            for s in SPLITS:
                idx = splits == s
                performance.append({'outcome':yname,'method':method,'period':s,'n':int(idx.sum()),
                    'selected_ridge_alpha':alpha,'weighted_R2':ratio_r2(y[idx],pred[idx],w[idx]),
                    'weighted_RMSE':float(np.sqrt(np.average((y[idx]-pred[idx])**2,weights=w[idx]))),
                    'weighted_MAE':float(np.average(abs(y[idx]-pred[idx]),weights=w[idx]))})
            print(f'{yname:9} {method:30} CV MSE {mse:.5f} alpha {alpha:g}')

    leakage = audit_leakage(x[tr_idx], z[tr_idx], w[tr_idx], Z_VARS)
    for filename, table in [
        ('102_reconstruction.csv',reconstruction),('102_coordinate_drift.csv',drift),
        ('102_training_cv.csv',cv_records),('102_frozen_phenotype_performance.csv',performance),
        ('102_private_component_leakage.csv',leakage)]:
        pd.DataFrame(table).to_csv(out/filename,index=False)

    p = pd.DataFrame(performance)
    re = pd.DataFrame(reconstruction)
    cv = pd.DataFrame(cv_records)
    rpt = ['# Script 102 — NHANES representation benchmark','',
        '## Protocol',
        '- Genuine Script-98 common-complete cohort; original PHQ item definitions and covariates.',
        '- Original outcome-independent frozen Z is NOT refitted or altered.',
        '- Baseline PCA/CCA fitted on 2005–08 physiology only; no target data used in fitting.',
        '- Penalization alpha chosen exclusively with training-cycle-stratified five-fold CV (or --folds).',
        '- Fixed representations and regression coefficients applied to 2009–18 and 2021–23.',
        '- These later periods have been studied before: exploratory temporal stress tests, NOT untouched confirmation.',
        '- NHANES cycles contain different individuals; results are NOT within-person longitudinal evidence.',
        '- R² values are survey-weighted point estimates, without complex survey design variance estimation.',
        '',f'Observed counts: `{json.dumps(counts)}`.',
        f'Independent discovery CCA correlations: `{[round(float(a),5) for a in rho]}`.',
        '', '## Reconstruction (test and validation)',
        '```text',re[re.period != SPLITS[0]].pivot(index='method',columns='period',values='weighted_reconstruction_R2').round(5).to_string(),'```',
        '', '## Phenotype prediction on temporally held-out periods',
        '```text',p[p.period != SPLITS[0]].pivot_table(index=['outcome','method'],columns='period',values='weighted_R2').round(5).to_string(),'```',
        '', '## Training-only cross-validation',
        '```text',cv.pivot(index='method',columns='outcome',values='train_cv_weighted_MSE').round(5).to_string(),'```',
        '', '## What the numbers cannot establish',
        '- A full rank 7->7 linear transform is not intrinsically more informative than the original 7 biomarkers.',
        '- Equal or slightly better held-out Ridge scores can reflect regularization geometry or model selection, not new information.',
        '- A high reconstruction R² for seven full-rank features is mathematically expected, not proof of biological identifiability.',
        '- Private block zeros are structural properties defined by the transformation, not independently verified physiological mechanisms.',
        '- Drift scores are coordinate dependent, descriptive, and must not be interpreted as formal subspace alignment.',
        '- This script does not benchmark true JIVE, AJIVE, MOFA+, external MIDUS/MIDJA, or clinical utility; they require distinct protocols.',
        '- No causal inference, mechanistic identification, untouched external replication, or established framework novelty follows from this run.',
        '', '## Decision rule',
        'Judge the method on matched 4D/5D compression, held-out phenotype metrics, and independent interpretation; '
        'treat 7D reconstruction as a sanity check. If IndependentCCA7 and FrozenZ7 behave nearly identically, '
        'the distinguishing methodological claim remains unproven.',
    ]
    (out/'102_readout.md').write_text('\n'.join(rpt)+'\n',encoding='utf-8')
    meta = {'script':'102_representation_benchmark.py','seed':SEED,'folds':args.folds,
            'cohort_counts':counts,'discovery_CCA_rho':rho.tolist(),
            'methods':names,'real_data_only':True,'frozen_Z':True,
            'original_98_matched_sample':True,'outcome_fitting_only_discovery':True,
            'no_mofa_jive_claim':True,'not_untouched_validation':True}
    (out/'102_manifest.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print('\nResults written:',out)
    print('Read:',out/'102_readout.md')

if __name__ == '__main__':
    main()
