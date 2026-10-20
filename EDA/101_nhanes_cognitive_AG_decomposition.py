#!/usr/bin/env python3
"""NHANES cognitive phenotypes: frozen A/G physiology and decomposition audits.

Run from the project root:
    python EDA/101_nhanes_cognitive_AG_decomposition.py

Needs the real Script 73 phenotype-map input and NHANES CFQ_G/H SAS XPT files.
CFQ XPT files are downloaded from CDC if they are not already present locally.
2011-12 determines every fitted equation and hyperparameter; 2013-14 is
an untouched cross-sectional period for evaluation. No participant-level
records are exported. No causal attribution is claimed.
"""
from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Data' / 'NHANES_Transfer'
OUT = ROOT / 'EDA' / 'Results'
SOURCE = OUT / '73_frozen_phenotype_map_input.csv'
TRANSFORM = OUT / '72_shared_private_frozen_transform.npz'
SEED = 101
A = ['LBXHGB', 'LBXRBCSI', 'LBXMCVSI', 'LBXRDW']
G = ['LBXGH', 'LBXGLU', 'LOG_IN']
Z = ['SP_SHARED1', 'SP_SHARED2', 'SP_DISCORD1', 'SP_DISCORD2',
     'SP_A_PRIVATE3', 'SP_A_PRIVATE4', 'SP_G_PRIVATE3']
GROUPS = {
    'Shared': Z[0:2], 'Discordant': Z[2:4],
    'A_private': Z[4:6], 'G_private': Z[6:7]
}
OUTCOMES = {
    'DSST_processing_speed': 'CFDDS',
    'Animal_verbal_fluency': 'CFDAST',
    'CERAD_delayed_memory': 'CFDCSR',
    'CERAD_learning_sum': 'CERAD_LEARNING'
}
TEST_COLUMNS = ['CFDDS', 'CFDAST', 'CFDCSR', 'CFDCST1', 'CFDCST2', 'CFDCST3']
REGULARIZATION = [0.1, 1.0, 10.0, 100.0]
FOLDS = 4
BOOTSTRAP = 1200


def get_cfq(cycle: str, suffix: str) -> pd.DataFrame:
    look = [
        DATA / cycle / f'CFQ_{suffix}.XPT',
        DATA / cycle / f'CFQ_{suffix}.xpt',
        ROOT / 'Data' / 'NHANES' / cycle / f'CFQ_{suffix}.XPT',
        ROOT / f'CFQ_{suffix}.XPT',
    ]
    file = next((p for p in look if p.is_file()), None)
    if file is None:
        url = f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{"2011" if cycle == "1112" else "2013"}/DataFiles/CFQ_{suffix}.XPT'
        file = DATA / cycle / f'CFQ_{suffix}.XPT'
        file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=45) as resp, file.open('wb') as handle:
                import shutil
                shutil.copyfileobj(resp, handle)
        except Exception as exc:
            file.unlink(missing_ok=True)
            raise FileNotFoundError(f'Missing CFQ file {file}; download from {url}. Error: {exc}') from exc
    c = pd.read_sas(file, format='xport')
    required = ['SEQN'] + TEST_COLUMNS
    missing = [x for x in required if x not in c]
    if missing:
        raise ValueError(f'{file}: required cognitive scores absent: {missing}')
    c = c[required].copy()
    for key in required:
        c[key] = pd.to_numeric(c[key], errors='coerce')
    c['SEQN'] = c['SEQN'].astype('Int64')
    if c['SEQN'].isna().any() or c['SEQN'].duplicated().any():
        raise ValueError(f'{file}: invalid duplicate respondent IDs')
    for key in TEST_COLUMNS:
        c[key] = c[key].mask(c[key].abs() < 1e-10, 0.0)
    c['CERAD_LEARNING'] = c[['CFDCST1', 'CFDCST2', 'CFDCST3']].sum(axis=1, min_count=3)
    c['CYCLE'] = cycle
    return c


def load_data():
    if not SOURCE.exists():
        raise FileNotFoundError(f'Run EDA/73_frozen_phenotype_map_time_travel.py first. Missing {SOURCE}')
    required = ['SEQN', 'CYCLE', 'RIDAGEYR', 'RIAGENDR', 'EDUC3', 'BMXBMI',
                'SURVEY_WT', 'ELIGIBLE_ADULT_NONPREG'] + A + G + Z
    available = pd.read_csv(SOURCE, nrows=0).columns.to_list()
    missing = [x for x in required if x not in available]
    if missing:
        raise ValueError(f'Script 73 source lacks columns {missing}')
    d = pd.read_csv(SOURCE, usecols=required, low_memory=False)
    d['CYCLE'] = d['CYCLE'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
    d = d[d['CYCLE'].isin(['1112', '1314'])].copy()
    for key in required:
        if key != 'CYCLE':
            d[key] = pd.to_numeric(d[key], errors='coerce')
    d = d[d['RIDAGEYR'].ge(60) & d['ELIGIBLE_ADULT_NONPREG'].eq(1)].copy()
    d['SEQN'] = d['SEQN'].astype('Int64')
    if d['SEQN'].isna().any() or d['SEQN'].duplicated().any():
        raise ValueError('Script 73 source contains duplicated or missing respondent IDs')
    cfq = pd.concat([get_cfq('1112', 'G'), get_cfq('1314', 'H')], ignore_index=True)
    d = d.merge(cfq, on=['SEQN', 'CYCLE'], how='left', validate='one_to_one')
    base = ['RIDAGEYR', 'BMXBMI', 'RIAGENDR', 'EDUC3']
    for k in base + A + G + Z + ['SURVEY_WT']:
        d[k] = pd.to_numeric(d[k], errors='coerce')
    d['Female'] = (d['RIAGENDR'] == 2).astype(float)
    # Discovery-defined education reference categories; no outcome supervision.
    for value in [2, 3]:
        d[f'EDUC3_{value}'] = (d['EDUC3'] == value).astype(float)
    cov = ['RIDAGEYR', 'Female', 'BMXBMI', 'EDUC3_2', 'EDUC3_3']
    models = {
        'covariates': cov,
        'A_only': cov + A,
        'G_only': cov + G,
        'raw_A_plus_G': cov + A + G,
        'Z_shared_private': cov + Z,
    }
    coverage = []
    for cycle in ['1112', '1314']:
        q = d[d.CYCLE.eq(cycle)]
        valid_w = q.SURVEY_WT.gt(0)
        allphys = q[A + G + Z].notna().all(axis=1)
        for name, y in OUTCOMES.items():
            coverage.append({
                'cycle': cycle, 'outcome': name, 'age60_eligible_in_source': len(q),
                'cognitive_observed': int(q[y].notna().sum()),
                'fully_observed_X_and_Z': int((allphys & valid_w).sum()),
                'common_complete': int((allphys & valid_w & q[y].notna() &
                                        q[['RIDAGEYR', 'RIAGENDR', 'EDUC3', 'BMXBMI']].notna().all(axis=1)).sum())
            })
    return d, models, pd.DataFrame(coverage)


def weighted_mean(a, w):
    return float(np.average(a, weights=w))


def w_mse(y, p, w):
    return float(np.average(np.square(y - p), weights=w))


def w_r2(y, p, w):
    den = np.average(np.square(y - np.average(y, weights=w)), weights=w)
    return float(1 - w_mse(y, p, w) / den) if den > 1e-12 else float('nan')


def scale_fit(a, w):
    mu = np.average(a, axis=0, weights=w)
    sd = np.sqrt(np.average(np.square(a - mu), axis=0, weights=w))
    sd = np.where(sd < 1e-9, 1.0, sd)
    return mu, sd


def fit_model(X, y, w, alpha):
    mu, sd = scale_fit(X, w)
    reg = Ridge(alpha=float(alpha))
    reg.fit((X - mu) / sd, y, sample_weight=w / w.mean())
    return reg, mu, sd


def predict(fitted, X):
    reg, mu, sd = fitted
    return reg.predict((X - mu) / sd)


def choose_alpha(X, y, w):
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for a in REGULARIZATION:
        vals = []
        for tr, va in kf.split(X):
            p = predict(fit_model(X[tr], y[tr], w[tr], a), X[va])
            vals.append(w_mse(y[va], p, w[va]))
        scores.append((float(np.mean(vals)), a))
    return min(scores)[1]


def delta_ci(y, p_base, p_alt, w, rng):
    diffs = (y - p_base) ** 2 - (y - p_alt) ** 2
    n = len(y)
    if n < 8:
        return float('nan'), float('nan')
    out = np.empty(BOOTSTRAP)
    for b in range(BOOTSTRAP):
        ii = rng.integers(0, n, size=n)
        out[b] = float(np.average(diffs[ii], weights=w[ii]))
    return map(float, np.percentile(out, [2.5, 97.5]))


def jacobian_x_to_z():
    if not TRANSFORM.exists():
        return None
    v = np.load(TRANSFORM)
    a_sd = np.asarray(v['A_sd'], float)
    g_sd = np.asarray(v['G_sd'], float)
    wa = np.asarray(v['WA'], float)
    wg = np.asarray(v['WG'], float)
    rho = np.asarray(v['rho'], float)
    if int(v['shared_rank'].reshape(-1)[0]) != 2:
        raise ValueError('Unexpected Script 72 shared rank')
    j0 = np.zeros((7, 7))
    j0[:4, :4] = wa.T @ np.diag(1 / a_sd)
    j0[4:, 4:] = wg.T @ np.diag(1 / g_sd)
    h = np.zeros((7, 7))
    for i in range(2):
        den_s = np.sqrt(2 * (1 + rho[i]))
        den_d = np.sqrt(2 * (1 - rho[i]))
        h[i, i] = 1 / den_s
        h[i, 4 + i] = 1 / den_s
        h[2 + i, i] = 1 / den_d
        h[2 + i, 4 + i] = -1 / den_d
    h[4, 2] = h[5, 3] = h[6, 6] = 1
    return h @ j0, np.r_[a_sd, g_sd]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    d, models, coverage = load_data()
    coverage.to_csv(OUT / '101_cognitive_coverage.csv', index=False)
    rng = np.random.default_rng(SEED)
    results, contributions, coefficients, shapley, paired, gradients = [], [], [], [], [], []
    audit_j = jacobian_x_to_z()
    for name, target in OUTCOMES.items():
        mask = (d[A + G + Z].notna().all(axis=1) & d[list(models['covariates'])].notna().all(axis=1)
                & d['EDUC3'].notna() & d[target].notna() & d.SURVEY_WT.gt(0))
        sample = d[mask].copy()
        tr = sample[sample.CYCLE.eq('1112')]
        te = sample[sample.CYCLE.eq('1314')]
        print(f'{name:24} train={len(tr):4} test={len(te):4}')
        if len(tr) < 60 or len(te) < 35:
            print('  Insufficient fully observed individuals: outcome skipped; inspect 101_cognitive_coverage.csv')
            continue
        yt = tr[target].to_numpy(float)
        wh = te.SURVEY_WT.to_numpy(float)
        yh = te[target].to_numpy(float)
        wt = tr.SURVEY_WT.to_numpy(float)
        fitted, pred = {}, {}
        for spec, cols in models.items():
            xt, xh = tr[cols].to_numpy(float), te[cols].to_numpy(float)
            a = choose_alpha(xt, yt, wt)
            fitted[spec] = fit_model(xt, yt, wt, a)
            pred[spec] = predict(fitted[spec], xh)
            p = pred[spec]
            results.append({'outcome': name, 'specification': spec, 'n_train': len(tr),
                            'n_test': len(te), 'selected_alpha': a,
                            'test_MSE': w_mse(yh, p, wh), 'test_R2': w_r2(yh, p, wh),
                            'test_bias': weighted_mean(p - yh, wh),
                            'delta_MSE_vs_covariates': w_mse(yh, pred['covariates'], wh) - w_mse(yh, p, wh) if 'covariates' in pred else 0.0})
            if spec != 'covariates':
                c0, c1 = delta_ci(yh, pred['covariates'], p, wh, rng)
                paired.append({'outcome': name, 'reference': 'covariates', 'candidate': spec,
                               'delta_MSE': w_mse(yh, pred['covariates'], wh) - w_mse(yh, p, wh),
                               'descriptive_ci_low': c0, 'descriptive_ci_high': c1})
            if spec == 'Z_shared_private':
                reg, mu, sd = fitted[spec]
                b = reg.coef_ / sd
                intercept = float(reg.intercept_ - np.dot(b, mu))
                coefficients.append({'outcome': name, 'term': 'intercept', 'coefficient_original_basis': intercept,
                                     'selected_alpha': a})
                for feature, val in zip(cols, b):
                    coefficients.append({'outcome': name, 'term': feature, 'coefficient_original_basis': float(val),
                                         'selected_alpha': a})
                for block, zcols in GROUPS.items():
                    ii = [cols.index(v) for v in zcols]
                    raw = ((xh[:, ii] - mu[ii]) * b[ii]).sum(axis=1)
                    contributions.append({'outcome': name, 'Z_block': block,
                                          'mean_absolute_predicted_deviation': weighted_mean(np.abs(raw), wh),
                                          'rms_predicted_deviation': float(np.sqrt(np.average(raw ** 2, weights=wh))),
                                          'mean_signed_predicted_deviation': weighted_mean(raw, wh),
                                          'n': len(te)})
                if audit_j is not None:
                    j, x_sd = audit_j
                    b_z = np.array([b[cols.index(z)] for z in Z])
                    gx = b_z @ j
                    for i, x in enumerate(A + G):
                        gradients.append({'outcome': name, 'raw_biomarker': x,
                                          'block': 'hematology' if i < 4 else 'glycemia',
                                          'derivative_per_raw_unit': float(gx[i]),
                                          'derivative_per_discovery_SD': float(gx[i] * x_sd[i])})
        # Cooperative split of *incremental predictive performance*, not causal A/G effects.
        m0 = w_mse(yh, pred['covariates'], wh)
        ma = w_mse(yh, pred['A_only'], wh)
        mg = w_mse(yh, pred['G_only'], wh)
        mx = w_mse(yh, pred['raw_A_plus_G'], wh)
        phi_a = ((m0 - ma) + (mg - mx)) / 2
        phi_g = ((m0 - mg) + (ma - mx)) / 2
        shapley += [
            {'outcome': name, 'block': 'A', 'shapley_delta_MSE': phi_a,
             'total_rawX_gain': m0-mx, 'note': 'predictive allocation, not individual impairment or causal fraction'},
            {'outcome': name, 'block': 'G', 'shapley_delta_MSE': phi_g,
             'total_rawX_gain': m0-mx, 'note': 'predictive allocation, not individual impairment or causal fraction'},
        ]
        if not np.isclose(phi_a + phi_g, m0 - mx, atol=1e-9):
            raise AssertionError('A/G Shapley sum must equal joint gain')
        print(f'  A/G incremental predictive value = {m0 - mx:+.5f} ΔMSE; '
              f'A={phi_a:+.5f}, G={phi_g:+.5f}')
    frames = {
        '101_cognitive_models.csv': results,
        '101_cognitive_predictive_AG_allocation.csv': shapley,
        '101_cognitive_Z_block_attribution.csv': contributions,
        '101_cognitive_equation_terms.csv': coefficients,
        '101_cognitive_phenotype_gradients.csv': gradients,
        '101_cognitive_paired_intervals.csv': paired,
    }
    for fn, rows in frames.items():
        pd.DataFrame(rows).to_csv(OUT / fn, index=False)
    coverage_s = coverage.to_markdown(index=False)
    report = [
        '# Script 101 — NHANES A/G decomposition of measured cognitive outcomes',
        '',
        'Training: NHANES 2011–2012, independently evaluated: 2013–2014.',
        'Cognition: DSST, animal fluency, CERAD delayed recall and three-trial learning sum; age ≥60.',
        'Physiology: real Script 73 outcome-independent frozen NHANES seven-coordinate transformation.',
        'All models for each outcome use identical matched fully observed participants.',
        'All hyperparameters and standardization fitted within 2011–12 only.',
        '', '# Cohort availability', coverage_s, '',
        '# Out-of-period performance',
        pd.DataFrame(results).to_markdown(index=False, floatfmt='.5f') if results else 'No outcomes reached minimum sample sizes.',
        '', '# Predictive A versus G allocation',
        pd.DataFrame(shapley).to_markdown(index=False, floatfmt='.5f') if shapley else 'Not estimable.',
        '', '# Frozen latent block attribution',
        pd.DataFrame(contributions).to_markdown(index=False, floatfmt='.5f') if contributions else 'Not estimable.',
        '', '# Descriptive participant-bootstrap intervals',
        pd.DataFrame(paired).to_markdown(index=False, floatfmt='.5f') if paired else 'Not estimable.',
        '', '# Limits',
        '- All A/G allocation figures are about a *fitted prediction*, not causal shares of cognitive impairment.',
        '- Shared/discordant coordinates mix A and G, so cannot be attributed uniquely to either physiology block.',
        '- Full-rank Z preserves seven-dimensional information; it cannot create additional information beyond raw X.',
        '- Outcomes are contemporaneous cognitive test performances, not observed cognitive decline.',
        '- 2011–12 and 2013–14 contain different NHANES participants, not individual follow-up.',
        '- Fasting glucose and insulin restrict sample size; selection and test nonresponse may bias associations.',
        '- SURVEY_WT is inherited from Script 73; verify its fasting-weight construction before design-based inference.',
        '- This analysis estimates weighted point metrics but not NHANES strata/PSU-adjusted uncertainty.',
        '- Bootstrap confidence bands resample people, not the complex survey design, and are descriptive.',
        '- Clinical assay harmonization and missingness assumptions require independent verification.',
    ]
    (OUT / '101_cognitive_readout.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    manifest = {
        'script': Path(__file__).name,
        'input': str(SOURCE.relative_to(ROOT)) if SOURCE.is_relative_to(ROOT) else str(SOURCE),
        'training_cycle': '1112', 'external_cycle': '1314',
        'outcomes': OUTCOMES, 'A': A, 'G': G, 'frozen_Z': Z,
        'cross_validation': {'folds': FOLDS, 'alpha_choices': REGULARIZATION, 'seed': SEED},
        'uses_NHANES2011_2014_real_cognitive_scores': True,
        'refits_frozen_transformation': False,
        'causal_attribution': False,
        'survey_design_inference': False,
        'trained_outcomes': [x['outcome'] for x in results if x['specification'] == 'Z_shared_private'],
        'outputs': ['101_cognitive_coverage.csv', '101_cognitive_readout.md'] + list(frames),
    }
    (OUT / '101_cognitive_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('SCRIPT 101 | measured NHANES cognition and A/G decomposition | complete')
    print('OUTPUT:', OUT / '101_cognitive_readout.md')


if __name__ == '__main__':
    run()
