#!/usr/bin/env python3
"""107 | Diagnosed-diabetes sensitivity of frozen H/G-phenotype mapping.

Copy to EDA/107_diabetes_status_sensitivity.py and run without arguments:
    python EDA/107_diabetes_status_sensitivity.py

Automatically retrieves missing *public* NHANES DIQ and GLU XPT files; does
not overwrite existing downloads. Requires existing local 103 and 98 scripts,
73_frozen_phenotype_map_input.csv, and the original PHQ-9 XPT data used by 98.

Question: does the observed performance/association of the frozen physiological
model change when participants reporting a diabetes diagnosis are excluded?
Diabetes type (T1D vs T2D) is NOT identifiable from NHANES DIQ alone. A person
reporting 'no diabetes' is not necessarily free of undiagnosed diabetes.

The original outcome-independent 2005-08 seven-component representation is
retained (no new latent model is trained). Covariates, RawX7, and FrozenZ7
receive the same Ridge(alpha=10) model fitting protocol, fitted ONLY in
2005-08. Frozen tests: 2009-18 and 2021-23. A second discovery-only fit
excludes people with reported diabetes and participants with borderline/unknown
status, to test training-population sensitivity.

No confirmatory p values or causal/clinical statements. Public survey DIQ files
are joined by SEQN; weights are the official GLU fasting-subsample WTSAF2YR.
Outputs are descriptive complete-case metrics, not survey-corrected confidence
intervals. The same later NHANES cohorts have been inspected previously.
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
OUT = HERE / 'Results' / '107_diabetes_status_sensitivity'
SEED = 107
CYCLES = {
    '0506': ('2005', 'D', 'NHANES'),
    '0708': ('2007', 'E', 'NHANES'),
    '0910': ('2009', 'F', 'NHANES_Transfer'),
    '1112': ('2011', 'G', 'NHANES_Transfer'),
    '1314': ('2013', 'H', 'NHANES_Transfer'),
    '1516': ('2015', 'I', 'NHANES_Transfer'),
    '1718': ('2017', 'J', 'NHANES_Transfer'),
    '2123': ('2021', 'L', 'NHANES_Transfer'),
}
PERIODS = {
    '0506': 'discovery_2005_08', '0708': 'discovery_2005_08',
    '0910': 'test_2009_18', '1112': 'test_2009_18',
    '1314': 'test_2009_18', '1516': 'test_2009_18', '1718': 'test_2009_18',
    '2123': 'recent_2021_23',
}
YCOLS = {'PHQ_total': 'PHQ9_TOTAL', 'Somatic': 'SOMATIC_SCORE',
         'CogAff': 'COGAFF_SUM'}
GROUPS = ['all', 'no_reported_diabetes', 'reported_diabetes',
          'borderline_in_DIq010', 'unknown']
MIN_N = 30


def load_script(filename, module_name):
    src = HERE / filename
    if not src.is_file():
        raise FileNotFoundError(f'Missing required earlier script: {src}')
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def xpt_path(cycle, kind, auto_fetch):
    year, suffix, folder = CYCLES[cycle]
    target = ROOT / 'Data' / folder / cycle / f'{kind}_{suffix}.XPT'
    for f in (target, target.with_suffix('.xpt')):
        if f.is_file() and f.stat().st_size > 100:
            return f
    if not auto_fetch:
        raise FileNotFoundError(f'Missing {target}; allow automatic download or supply file.')
    url = f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{kind}_{suffix}.XPT'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.partial')
    print('Downloading:', url, flush=True)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'NHANES-Scientific-Study/1.0'})
        with urllib.request.urlopen(req, timeout=90) as resp, temporary.open('wb') as fd:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fd.write(chunk)
        if temporary.stat().st_size < 100:
            raise RuntimeError('NHANES download suspiciously small')
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def diabetes_status(df):
    status = pd.Series('unknown', index=df.index, dtype='object')
    status.loc[df.DIQ010.eq(1)] = 'reported_diabetes'
    status.loc[df.DIQ010.eq(2)] = 'no_reported_diabetes'
    status.loc[df.DIQ010.eq(3)] = 'borderline_in_DIq010'
    return status


def read_diabetes(cyc, fetch):
    q = pd.read_sas(xpt_path(cyc, 'DIQ', fetch), format='xport')
    oral = 'DID070' if cyc in ('0506', '0708') else 'DIQ070'
    required = ['SEQN', 'DIQ010', 'DIQ160', 'DIQ050', 'DID040', oral]
    absent = sorted(set(required) - set(q.columns))
    if absent:
        raise ValueError(f'DIQ_{cyc}: required variables missing: {absent}. Stop; do not infer diabetes type.')
    q = q[required].rename(columns={oral: 'DIQ_ORAL'}).copy()
    for col in q.columns:
        q[col] = pd.to_numeric(q[col], errors='coerce')
    if q.SEQN.isna().any() or q.SEQN.duplicated().any():
        raise ValueError(f'DIQ_{cyc}: SEQN missing or duplicated')
    # Age at diagnosis can use 666=under age 1, 777/999=refused/unknown.
    q['AGE_AT_DIAGNOSIS'] = q.DID040.where(q.DID040.between(1, 90)).copy()
    q.loc[q.DID040.eq(666), 'AGE_AT_DIAGNOSIS'] = 0.5
    q['DIABETES_STATUS'] = diabetes_status(q)
    return q[['SEQN', 'DIABETES_STATUS', 'DIQ160', 'DIQ050', 'DIQ_ORAL', 'AGE_AT_DIAGNOSIS']]


def read_fast_weights(cyc, fetch):
    d = pd.read_sas(xpt_path(cyc, 'GLU', fetch), format='xport')
    required = {'SEQN', 'WTSAF2YR'}
    if not required.issubset(d.columns):
        raise ValueError(f'GLU_{cyc}: WTSAF2YR unavailable')
    d = d[['SEQN', 'WTSAF2YR']].copy()
    d.SEQN = pd.to_numeric(d.SEQN, errors='coerce')
    d.WTSAF2YR = pd.to_numeric(d.WTSAF2YR, errors='coerce')
    if d.SEQN.isna().any() or d.SEQN.duplicated().any():
        raise ValueError(f'GLU_{cyc}: invalid SEQN')
    return d


def wmean(a, w):
    return float(np.average(a, weights=w))


def wmse(y, yp, w):
    return wmean((np.asarray(y)-np.asarray(yp))**2, w)


def wr2(y, yp, w):
    mu = wmean(y, w)
    denominator = wmean((np.asarray(y)-mu)**2, w)
    return float(1 - wmse(y, yp, w)/denominator) if denominator > 0 else float('nan')


def fit_model(xtrain, ytrain, wtrain, pred):
    sc = StandardScaler().fit(xtrain)
    model = Ridge(alpha=10., solver='svd')
    model.fit(sc.transform(xtrain), ytrain,
              sample_weight=wtrain/np.mean(wtrain))
    return model.predict(sc.transform(pred))


def audit_counts(df):
    rows = []
    for period, q in df.groupby('PERIOD', sort=False):
        for group in GROUPS:
            g = q if group == 'all' else q[q.DIABETES_STATUS.eq(group)]
            diag = g[g.DIABETES_STATUS.eq('reported_diabetes')]
            ww = g.WTSAF2YR.to_numpy(float)
            rows.append({
                'period': period, 'group': group, 'n': len(g),
                'weighted_fraction_within_period': float(g.WTSAF2YR.sum()/q.WTSAF2YR.sum()),
                'reported_prediabetes_DIQ160_yes_n': int(g.DIQ160.eq(1).sum()),
                'diagnosed_insulin_yes_n': int(diag.DIQ050.eq(1).sum()),
                'diagnosed_insulin_no_n': int(diag.DIQ050.eq(2).sum()),
                'diagnosed_oral_pills_yes_n': int(diag.DIQ_ORAL.eq(1).sum()),
                'diagnosed_oral_pills_no_n': int(diag.DIQ_ORAL.eq(2).sum()),
                'diagnosis_age_known_n': int(diag.AGE_AT_DIAGNOSIS.notna().sum()),
                'median_age_at_diagnosis': float(diag.AGE_AT_DIAGNOSIS.median())
                    if diag.AGE_AT_DIAGNOSIS.notna().any() else float('nan'),
                'mean_phq_total': wmean(g.PHQ9_TOTAL.to_numpy(float), ww)
                    if len(g) else float('nan'),
            })
    return pd.DataFrame(rows)


def analyze(fetch=True, output=OUT):
    s103 = load_script('103_jive_ajive_benchmark.py', 's103_for_107')
    d, cov = s103.load_matched()
    d = d.copy().reset_index(drop=True)
    d['CYCLE'] = d.CYCLE.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
    if not set(d.CYCLE).issubset(CYCLES):
        raise ValueError(f'Unexpected cycles: {set(d.CYCLE)-set(CYCLES)}')
    d['PERIOD'] = d.CYCLE.map(PERIODS)
    covcols = [f'_FULL_COV_{i}' for i in range(cov.shape[1])]
    if len(d) != len(cov):
        raise RuntimeError('Covariates and participant rows do not match')
    for i, name in enumerate(covcols):
        d[name] = cov[:, i]
    diq = []
    glu = []
    for cyc in sorted(set(d.CYCLE)):
        a = read_diabetes(cyc, fetch)
        a['CYCLE'] = cyc
        diq.append(a)
        b = read_fast_weights(cyc, fetch)
        b['CYCLE'] = cyc
        glu.append(b)
    diq = pd.concat(diq, ignore_index=True)
    glu = pd.concat(glu, ignore_index=True)
    # Duplicate SEQN across 2005-23 is theoretically prohibited; use CYCLE as well.
    d = d.merge(diq, on=['SEQN', 'CYCLE'], how='left', validate='one_to_one')
    d = d.merge(glu, on=['SEQN', 'CYCLE'], how='left', validate='one_to_one')
    d['DIABETES_STATUS'] = d.DIABETES_STATUS.fillna('unknown')
    if not d.WTSAF2YR.gt(0).all() or d.WTSAF2YR.isna().any():
        raise RuntimeError('Missing/nonpositive official fasting subsample weight in the matched analytic data')
    # Keep original 7-component reference fixed. Neither PHQ nor DIQ is used to learn Z.
    features = {
        'Covariates': d[covcols].to_numpy(float),
        'RawX7': d[s103.X_COLS + covcols].to_numpy(float),
        'FrozenZ7': d[s103.Z_COLS + covcols].to_numpy(float),
    }
    disc = d.PERIOD.eq('discovery_2005_08').to_numpy()
    nodiab = d.DIABETES_STATUS.eq('no_reported_diabetes').to_numpy()
    if disc.sum() < 300 or (disc & nodiab).sum() < 300:
        raise ValueError('Insufficient real discovery sample for diagnosed-diabetes sensitivity.')
    outputs = []
    for yname, col in YCOLS.items():
        y = d[col].to_numpy(float)
        for training in ('all_discovery', 'without_reported_diabetes'):
            train = disc if training == 'all_discovery' else (disc & nodiab)
            wtrain = d.loc[train, 'WTSAF2YR'].to_numpy(float)
            for method, xx in features.items():
                yp = fit_model(xx[train], y[train], wtrain, xx)
                for period in ['discovery_2005_08', 'test_2009_18', 'recent_2021_23']:
                    inperiod = d.PERIOD.eq(period).to_numpy()
                    for group in GROUPS:
                        sel = inperiod if group == 'all' else (inperiod & d.DIABETES_STATUS.eq(group).to_numpy())
                        n = int(sel.sum())
                        if n < MIN_N:
                            continue
                        ww = d.loc[sel, 'WTSAF2YR'].to_numpy(float)
                        outputs.append({
                            'outcome': yname, 'training': training, 'method': method,
                            'period': period, 'group': group, 'n': n,
                            'weighted_mse': wmse(y[sel], yp[sel], ww),
                            'weighted_r2': wr2(y[sel], yp[sel], ww),
                            'evaluated_in_training_period': period == 'discovery_2005_08',
                            'train_n': int(train.sum()),
                        })
    perf = pd.DataFrame(outputs)
    if perf.empty:
        raise RuntimeError('No sensitivity results produced')
    baseline = perf.loc[perf.method.eq('Covariates'),
        ['outcome', 'training', 'period', 'group', 'weighted_mse']].rename(
        columns={'weighted_mse': 'covariates_weighted_mse'})
    perf = perf.merge(baseline, on=['outcome', 'training', 'period', 'group'],
                      how='left', validate='many_to_one')
    perf['delta_mse_vs_covariates'] = perf.covariates_weighted_mse - perf.weighted_mse
    c = audit_counts(d)
    output.mkdir(parents=True, exist_ok=True)
    perf.to_csv(output/'107_frozen_phenotype_by_diabetes.csv', index=False)
    c.to_csv(output/'107_diabetes_group_counts.csv', index=False)
    manifest = {
        'script': '107_diabetes_status_sensitivity.py',
        'classification': 'self-reported DIQ010 doctor diagnosis, separate DIQ160 prediabetes indicator, NOT Type 1/Type 2',
        'insulin': 'DIQ050 current use, only among reporting diabetes; NOT disease type',
        'oral_drugs': 'DID070 in 2005-08, DIQ070 afterward',
        'age_at_diagnosis': 'DID040, descriptive only',
        'cohorts': PERIODS,
        'physiology': 'frozen Script73 full-rank 7D Z, learned in 2005-08, NOT refitted',
        'training': 'Ridge alpha=10, full-covariate 2005-08 only, comparison excluding diagnosed diabetes',
        'weights': 'official fasting-subsample WTSAF2YR per cycle; no formal pooled population estimates',
        'unknown_diabetes': 'excluded from no-reported-diabetes sensitivity training, retained in full model',
        'all_later_cohorts_previously_explored': True,
        'outcomes': list(YCOLS),
        'limitations': [
            'self-reported status may miss undiagnosed diabetes',
            'NHANES DIQ does not reliably determine Type 1 vs Type 2',
            'treatment choices and physiology are confounded; no causal implications',
            'DIQ010=2 subgroup includes participants reporting prediabetes on DIQ160 and people with undiagnosed diabetes',
            'model MSE and R2 are descriptive; no masked PSU/strata design confidence intervals',
            'same-data earlier 2005-08 preprocessing may enter 2007-08 evaluation',
            '5-cycle 2009-18 aggregation is an exploratory fixed-era comparison',
            'differences in subgroup R2 are not a hypothesis test of effect modification',
            'analysis uses original complete-case cohort and does not correct selection/nonresponse',
        ],
    }
    (output/'107_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    focus = perf[(perf.outcome.eq('PHQ_total')) &
                 (perf.period.isin(['test_2009_18','recent_2021_23'])) &
                 (perf.group.isin(['all','no_reported_diabetes','reported_diabetes'])) &
                 (perf.method.isin(['Covariates','RawX7','FrozenZ7']))]
    txt = [
        '# Script 107 — Diabetes status sensitivity (descriptive)', '',
        'Doctor-reported diabetes status; **not** Type 1/Type 2 identification.',
        'Full 2005–08 training versus discovery training excluding reported diabetes.', '',
        '## Complete-case diabetes groups', '',
        '```text', c[['period','group','n','weighted_fraction_within_period',
                     'diagnosed_insulin_yes_n','diagnosed_oral_pills_yes_n',
                     'reported_prediabetes_DIQ160_yes_n']].round(3).to_string(index=False),
        '```', '',
        '## Frozen PHQ-total outcome transfer', '',
        'Positive delta MSE indicates incremental value beyond covariates.', '',
        '```text', focus[['training','period','group','method','n','weighted_r2',
                       'delta_mse_vs_covariates']].round(4).to_string(index=False), '```', '',
        '## Important boundaries', '',
        '- **Never** describe DIQ010/DIQ050 as confirmed Type 1 vs Type 2.',
        '- Comparing subgroups with different outcome variance can change R² without changing biology.',
        '- The key sensitivity is whether excluding diagnosed diabetes from discovery fitting materially changes the frozen **no-reported-diabetes** evaluation.',
        '- This experiment is descriptive and exploratory; it does not establish diabetes as the source of any component, or formal effect modification.',
        '- Earlier NHANES periods and the original full-sample representation have already been used; this is not an untouched confirmation.',
    ]
    (output/'107_readout.md').write_text('\n'.join(txt)+'\n', encoding='utf-8')
    print('Wrote:', output)
    print('Read: 107_readout.md, 107_diabetes_group_counts.csv, 107_frozen_phenotype_by_diabetes.csv')


def self_test():
    demo = pd.DataFrame({'DIQ010': [1,2,3,7,9,np.nan],
                         'DIQ050': [1,2,2,2,1,np.nan]})
    assert list(diabetes_status(demo)) == [
        'reported_diabetes', 'no_reported_diabetes', 'borderline_in_DIq010',
        'unknown', 'unknown', 'unknown']
    yy = np.array([1.,2.,3.,4.]); ww = np.array([1.,2.,3.,4.]); yp = yy+.25
    assert wmse(yy,yp,ww) > 0 and abs(wr2(yy,yy,ww)-1)<1e-12
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(120,5)); y = x[:,0] + rng.normal(scale=.1,size=120)
    yp = fit_model(x[:80],y[:80],np.ones(80),x[80:])
    assert np.isfinite(yp).all() and len(yp)==40
    print('PASS: diabetes code categories, frozen ridge, weighted metrics')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--fetch-missing', action=argparse.BooleanOptionalAction,
                        default=True, help='Automatically download missing public DIQ/GLU XPTs (default on)')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        self_test()  # fast integrity check on every normal run
        analyze(args.fetch_missing, args.out)


if __name__ == '__main__':
    main()
