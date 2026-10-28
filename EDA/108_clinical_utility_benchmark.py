#!/usr/bin/env python3
"""Script 109: honest application benchmark for physiological representation and risk flags.

Run without CLI arguments from the existing repository root:
    python EDA/109_clinical_utility_benchmark.py

Optional quick mathematical tests (no NHANES required):
    python EDA/109_clinical_utility_benchmark.py --self-test

Question: On matched NHANES adults, does the ORIGINAL frozen seven-component
H/G physiological representation (i) improve PHQ-9 >= 10 screening beyond
standard covariate/raw/PCA models or (ii) identify model misses more effectively
than existing simple outcome-independent shift detectors?

This is NOT a clinical depression diagnosis, source-attribution or causal test.
No image/vital inputs exist here; this is a blood-data application baseline for
future multimodal risk assessment, not proof of multimodal performance.

The original Z7 was learned from NHANES 2005-08, NOT refitted here. All supervised
training and model selection uses 2005-08 only; 2009-18 and 2021-23 are held out
from fitting, but have already been examined extensively in this research.
All test findings are therefore exploratory, NOT independent confirmation.

Physiology preprocessing on training data only. Same participants, covariates,
outcome, CV folds and classifier family for all methods. RawX7 and PCA7
MUST agree up to numerical precision for a rotation-invariant (unwhitened PCA)
L2-regularised logistic classifier under the same regularization parameter.

The fixed Z7 and raw/PCA7 models carry the same input information; predictive
differences reflect training/regularisation/conditioning, not new information.
Optional JIVE5/AJIVE5 are official native methods with training-only linear
out-of-sample score adapters from Script103; missing dependencies are explicitly
reported, never replaced with imitations.

Outputs: EDA/Results/109_clinical_utility_benchmark/ (CSV, readout, manifest).
Survey weights give descriptive point estimates only. Paired case bootstrap
intervals are NOT masked-PSU/stratum-adjusted NHANES design intervals.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE / "Results" / "109_clinical_utility_benchmark"
SEED = 109
CS = (0.001, 0.01, 0.1, 1., 10., 100.)
FOLDS = 5
BOOTSTRAPS = 199
FLAG_FRACTION = 0.20
SENSITIVITY_TARGET = 0.80
PERIODS = ("train_2005-2008", "test_2009-2018", "validate_2021-2023")
GROUPS = {"Shared2": (0, 1), "Discordant2": (2, 3),
          "H_isolated2": (4, 5), "G_isolated1": (6,)}


def import_s103():
    p = HERE / "103_jive_ajive_benchmark.py"
    if not p.is_file():
        raise FileNotFoundError(f"Missing existing source {p}; do not substitute generated data")
    spec = importlib.util.spec_from_file_location("original_s103_for_s109", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def normw(w):
    w = np.asarray(w, float)
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Survey weights must be finite and positive")
    return w / w.mean()


def wavg(x, w):
    return float(np.average(np.asarray(x, float), weights=np.asarray(w, float)))


def classifier(C):
    return LogisticRegression(C=float(C), solver="lbfgs", max_iter=3000,
                              tol=1e-8, random_state=SEED)


def transform_features(cov, physiology, train):
    """Standardise each feature within the discovery sample; no outcome uses."""
    sc_cov = StandardScaler().fit(cov[train])
    sc_p = StandardScaler().fit(physiology[train])
    return np.column_stack([sc_cov.transform(cov), sc_p.transform(physiology)])


def select_model(a, y, w, train_idx, folds=FOLDS):
    """C selected with discovery-only weighted out-of-fold log loss."""
    yy = y[train_idx]
    if np.unique(yy).size < 2 or np.min(np.bincount(yy)) < folds:
        raise RuntimeError("Insufficient discovery events for stratified CV")
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True,
                                random_state=SEED).split(np.arange(len(train_idx)), yy))
    best, record, oofs = None, [], {}
    for c in CS:
        oof = np.full(len(train_idx), np.nan)
        for ii, jj in splits:
            tr, va = train_idx[ii], train_idx[jj]
            fit = classifier(c).fit(a[tr], y[tr], sample_weight=normw(w[tr]))
            oof[jj] = fit.predict_proba(a[va])[:, 1]
        loss = log_loss(yy, oof, sample_weight=w[train_idx], labels=[0, 1])
        record.append({"C": float(c), "weighted_cv_logloss": float(loss)})
        oofs[c] = oof
        if best is None or loss < best[0]:
            best = (loss, c)
    c = best[1]
    final = classifier(c).fit(a[train_idx], yy,
                              sample_weight=normw(w[train_idx]))
    return final.predict_proba(a)[:, 1], float(c), oofs[c], record


def risk_threshold(y, p, w, desired=SENSITIVITY_TARGET):
    """Train-only decision threshold obtaining >= desired weighted sensitivity."""
    positive = (np.asarray(y) == 1)
    if positive.sum() < 3:
        raise ValueError("Too few positive labels")
    pred, ww = np.asarray(p)[positive], np.asarray(w)[positive]
    order = np.argsort(-pred, kind="mergesort")
    cutoff_index = np.searchsorted(np.cumsum(ww[order]), desired*ww.sum(), side="left")
    return float(pred[order[min(cutoff_index, len(order)-1)]])


def metrics(y, p, w, decision_threshold):
    y, p, w = np.asarray(y, int), np.asarray(p, float), np.asarray(w, float)
    if len(y)==0 or np.unique(y).size < 2:
        return dict(n=int(len(y)), prevalence=np.nan, auroc=np.nan, auprc=np.nan,
                    brier=np.nan, logloss=np.nan, sensitivity=np.nan,
                    specificity=np.nan, mean_prediction=np.nan,
                    calibration_gap=np.nan)
    predicted_positive = (p >= decision_threshold)
    pos, neg = y == 1, y == 0
    return {
        "n": int(len(y)), "prevalence": wavg(y, w),
        "auroc": float(roc_auc_score(y, p, sample_weight=w)),
        "auprc": float(average_precision_score(y, p, sample_weight=w)),
        "brier": wavg((p-y)**2, w),
        "logloss": float(log_loss(y, p, sample_weight=w, labels=[0,1])),
        "sensitivity": wavg(predicted_positive[pos], w[pos]),
        "specificity": wavg(~predicted_positive[neg], w[neg]),
        "mean_prediction": wavg(p, w),
        "calibration_gap": wavg(p-y, w),
    }


def fit_reference_distances(x, w, train):
    """Raw Mahalanobis with fixed 10% ridge shrinkage, trained only on X."""
    mu = np.average(x[train], axis=0, weights=w[train])
    xx = x[train]-mu
    cov = (xx.T * w[train]) @ xx / w[train].sum()
    cov = .90*cov + .10*np.diag(np.diag(cov))
    inv = np.linalg.pinv(cov, rcond=1e-10)
    z = x-mu
    return np.sqrt(np.maximum(0, np.einsum('ij,jk,ik->i', z, inv, z)))


def group_ood(z, w, train, groups=GROUPS):
    """Outcome-blind group-specific anomaly, converted to train percentiles.

    Max of the reference-distribution percentile ranks for the four proposed
    structural groups; unlike full-space Mahalanobis, sensitive to extremes
    within a specific group rather than to total distance alone.
    """
    percentiles=[]
    for idx in groups.values():
        idx = list(idx)
        mu = np.average(z[train][:, idx], weights=w[train], axis=0)
        delta = z[:, idx] - mu
        tr_delta = delta[train]
        cov = (tr_delta.T*w[train]) @ tr_delta / w[train].sum()
        cov = 0.95*cov + 0.05*np.eye(len(idx))*np.trace(cov)/len(idx)
        inv = np.linalg.pinv(cov, rcond=1e-10)
        q = np.einsum("ij,jk,ik->i", delta, inv, delta)
        # Weighted empirical CDF of fitted-score distribution (no outcome).
        order = np.argsort(q[train])
        qs = q[train][order]
        ws = w[train][order]
        cdf = np.cumsum(ws)/ws.sum()
        percentiles.append(np.interp(q, qs, cdf, left=0., right=1.))
    return np.max(np.column_stack(percentiles), axis=1)


def top_weighted_fraction(x, w, fraction=FLAG_FRACTION):
    """Choose top weighted fraction within a *descriptive comparison cohort*.

    Outcome labels are NEVER used to choose which patients are flagged.
    This fixed-budget comparison is not a deployable threshold; additionally
    report frozen discovery-based operational thresholds.
    """
    order = np.argsort(-np.asarray(x), kind="mergesort")
    csum = np.cumsum(w[order])
    k = int(np.searchsorted(csum, fraction*w.sum(), side="left"))
    mask = np.zeros(len(x), dtype=bool)
    mask[order[:k+1]] = True
    return mask


def frozen_flag_threshold(score, w, train, fraction=FLAG_FRACTION):
    tr_s, tr_w = np.asarray(score)[train], np.asarray(w)[train]
    order=np.argsort(-tr_s, kind="mergesort")
    cutoff=int(np.searchsorted(np.cumsum(tr_w[order]), fraction*tr_w.sum(), side="left"))
    return float(tr_s[order[min(cutoff,len(order)-1)]])


def gate_scores(raw7, pca7, z7, p_z, threshold, w, train):
    sx=StandardScaler().fit(raw7[train]).transform(raw7)
    sp=StandardScaler().fit(pca7[train]).transform(pca7)
    return {
        "Z_group_anomaly":group_ood(z7, w, train),
        "Raw_max_marker_z":np.max(abs(sx), axis=1),
        "Raw_Mahalanobis":fit_reference_distances(sx, w, train),
        "PCA_max_axis_z":np.max(abs(sp), axis=1),
        "Near_risk_threshold":-abs(p_z-threshold),
    }


def gate_eval(y, p, w, score, decision_threshold, budget=FLAG_FRACTION):
    """Among predicted low-risk patients: fraction of missed cases flagged."""
    low = (p < decision_threshold)
    if low.sum() < 10 or np.sum(w[low & (y==1)]) <= 0:
        return None
    flags = top_weighted_fraction(score[low], w[low], budget)
    missed = (y[low] == 1)
    # Fraction of total missed weighted positive cases that are flagged.
    caught = wavg(flags[missed], w[low][missed])
    return {"n_low_risk":int(low.sum()),
            "missed_case_weighted_n":float(w[low][missed].sum()),
            "missed_case_capture_at_20pct":caught,
            "weighted_flag_fraction":wavg(flags,w[low]),
            "missed_case_prevalence_flagged": wavg(missed[flags],w[low][flags]),
            "missed_case_prevalence_all_lowrisk": wavg(missed, w[low])}


def paired_case_ci(y, a, b, w, reps=BOOTSTRAPS, seed=SEED):
    """Paired case-bootstrap Δ Brier and Δ AUROC: candidate Z minus raw.
    These are NOT survey-design-adjusted confidence intervals.
    """
    rng=np.random.default_rng(seed)
    n=len(y)
    db=[]; da=[]
    for _ in range(reps):
        ids=rng.integers(0,n,n)
        yy=y[ids]; ww=w[ids]
        db.append(wavg((yy-a[ids])**2-(yy-b[ids])**2,ww)) # raw error - Z error; >0 Z better
        if np.unique(yy).size==2:
            da.append(roc_auc_score(yy,b[ids],sample_weight=ww)-
                      roc_auc_score(yy,a[ids],sample_weight=ww))
    return dict(delta_brier_raw_minus_z=float(np.mean(db)),
                delta_brier_low=float(np.quantile(db,.025)),
                delta_brier_high=float(np.quantile(db,.975)),
                delta_auroc_z_minus_raw=float(np.mean(da)) if da else np.nan,
                delta_auroc_low=float(np.quantile(da,.025)) if da else np.nan,
                delta_auroc_high=float(np.quantile(da,.975)) if da else np.nan,
                bootstrap_reps=int(reps))


def assemble_methods(d, cov, s103, out, include_native=True):
    split=d.SPLIT.astype(str).to_numpy()
    train = split == PERIODS[0]
    w=d.SURVEY_WT.to_numpy(float)
    raw=d[s103.X_COLS].to_numpy(float)
    z=d[s103.Z_COLS].to_numpy(float)
    if not np.isfinite(raw).all() or not np.isfinite(z).all():
        raise ValueError('Frozen physiological data contain nonfinite values')
    rawscaled=StandardScaler().fit(raw[train]).transform(raw)
    pca=PCA(n_components=7, svd_solver="full").fit(rawscaled[train])
    pca7=pca.transform(rawscaled)
    methods={
      "Covariates":StandardScaler().fit(cov[train]).transform(cov),
      "Hb_HbA1c":transform_features(cov,raw[:,[0,4]],train),
      # IMPORTANT: no PCA-axis standardisation for full-rank PCA7. Unwhitened
      # PCA is an orthogonal rotation of the *standardised* X7; preserving the
      # L2 penalty geometry makes RawX7 vs PCA7 a genuine equivalence check.
      "RawX7":np.column_stack([StandardScaler().fit(cov[train]).transform(cov),rawscaled]),
      "PCA7":np.column_stack([StandardScaler().fit(cov[train]).transform(cov),pca7]),
      "FrozenZ7":transform_features(cov,z,train),
      "PCA4":transform_features(cov,pca7[:,:4],train),
      "FrozenZ4":transform_features(cov,z[:,:4],train),
    }
    native_status=[]
    if include_native:
        for name, fun in [("JIVE5", "jive_native"),("AJIVE5", "ajive_native")]:
            try:
                xtrain=rawscaled[train]
                native=getattr(s103,fun)(xtrain,out,"Rscript") if name=="JIVE5" else getattr(s103,fun)(xtrain)
                adapter,diag=s103.learn_projection(xtrain,native,min_fit_r2=.90)
                projected=s103.affine_predict(rawscaled,adapter)
                methods[name]=transform_features(cov,projected,train)
                native_status.append({"method":name,"status":"complete",
                                      "train_adapter_R2":float(diag['train_score_r2']),
                                      "warning":"frozen out-of-sample adapter, NOT native prediction"})
            except Exception as exc:
                native_status.append({"method":name,"status":"not_run",
                                      "error":str(exc),"warning":"No surrogate substituted"})
    return methods, raw, pca7, z, w, split, native_status


def run(self_test=False, skip_native=False, reps=BOOTSTRAPS):
    if self_test:
        run_self_test(); return
    s103=import_s103()
    print('Loading real complete-case NHANES through original Script103/98...')
    d,cov=s103.load_matched()
    split=d.SPLIT.astype(str).to_numpy()
    y=(d.PHQ9_TOTAL.to_numpy(float)>=10).astype(int)
    train=split==PERIODS[0]
    ids=np.flatnonzero(train)
    if min(int((split==v).sum()) for v in PERIODS)<100:
        raise RuntimeError('Insufficient matched participant rows in one or more periods')
    if int(y[train].sum()) < 30:raise RuntimeError('Not enough training PHQ>=10 cases')
    OUT.mkdir(parents=True,exist_ok=True)
    methods,raw,pca7,z,w,split,native_status=assemble_methods(d,cov,s103,OUT,include_native=not skip_native)
    print('Matched participants:',{s:int((split==s).sum()) for s in PERIODS},
          'discovery PHQ>=10:',int(y[train].sum()))
    models={}; cv_rows=[];metric_rows=[]
    for name,mat in methods.items():
        print('Fitting:',name)
        pred,c,oof,cv=select_model(mat,y,w,ids)
        cutoff=risk_threshold(y[ids],oof,w[ids])
        models[name]={'pred':pred,'C':c,'threshold':cutoff,'oof':oof}
        for rr in cv:cv_rows.append({'method':name,**rr})
        for period in PERIODS:
            keep=split==period
            vals=metrics(y[keep],pred[keep],w[keep],cutoff)
            metric_rows.append({'method':name,'period':period,'C':c,
                                'training_decision_threshold':cutoff,**vals})
    pd.DataFrame(metric_rows).to_csv(OUT/'109_screening_metrics.csv',index=False)
    pd.DataFrame(cv_rows).to_csv(OUT/'109_train_cv.csv',index=False)
    pd.DataFrame(native_status).to_csv(OUT/'109_native_status.csv',index=False)
    # Raw7/PCA7 should be equal up to numerical solver tolerance once both get
    # identical feature standardization and L2 penalty. If selected C differs,
    # print and explain; if both same C then check direct numerical equality.
    rawC=models['RawX7']['C']; pcaC=models['PCA7']['C']
    p_diff=float(np.max(abs(models['RawX7']['pred']-models['PCA7']['pred'])))
    print(f'RawX7 vs PCA7: C={rawC:g}/{pcaC:g}, maximum predicted-probability difference={p_diff:.7g}')
    if rawC==pcaC and p_diff>2e-4:
        raise AssertionError('RawX7/PCA7 full-rank rotation equivalence unexpectedly failed; inspect preprocessing')
    # Outcome-independent shift scores + threshold proximity (same FrozenZ7 risk model)
    gate=gate_scores(raw,pca7,z,models['FrozenZ7']['pred'],
                     models['FrozenZ7']['threshold'],w,train)
    # Flagging at 20% of CURRENT predicted negative group is a fair-budget
    # *retrospective comparison*, not a deployable threshold. Report both.
    gate_rows=[]
    for period in PERIODS[1:]:
        k=split==period
        for name,score in gate.items():
            value=gate_eval(y[k],models['FrozenZ7']['pred'][k],w[k],score[k],
                            models['FrozenZ7']['threshold'])
            if value is None: continue
            frozen_cutoff=frozen_flag_threshold(score,w,train)
            low=(models['FrozenZ7']['pred'][k]<models['FrozenZ7']['threshold'])
            low_score=score[k][low]
            fixed_flag=low_score >= frozen_cutoff
            missed=(y[k][low]==1)
            dct={
               'period':period,'gate':name,**value,
               'frozen_threshold':frozen_cutoff,
               'frozen_threshold_flag_fraction':wavg(fixed_flag,w[k][low]),
               'frozen_threshold_missed_capture':wavg(fixed_flag[missed],w[k][low][missed])
            }
            gate_rows.append(dct)
    pd.DataFrame(gate_rows).to_csv(OUT/'109_missed_case_detection.csv',index=False)
    bootstrap_rows=[]
    for period in PERIODS[1:]:
        keep=split==period
        q=paired_case_ci(y[keep], models['RawX7']['pred'][keep],
                          models['FrozenZ7']['pred'][keep],w[keep],reps=reps)
        bootstrap_rows.append({'period':period,**q})
    pd.DataFrame(bootstrap_rows).to_csv(OUT/'109_paired_bootstrap.csv',index=False)
    # Predeclare success/failure criteria; numerical comparisons remain
    # descriptive due to prior repeated exploration of these same outcomes.
    score=pd.DataFrame(metric_rows)
    gate_df=pd.DataFrame(gate_rows)
    comparison=[]
    for period in PERIODS[1:]:
        t=score[score.period.eq(period)].set_index('method')
        g=gate_df[gate_df.period.eq(period)].set_index('gate')
        delta_auc=float(t.loc['FrozenZ7','auroc']-t.loc['RawX7','auroc'])
        delta_brier=float(t.loc['RawX7','brier']-t.loc['FrozenZ7','brier'])
        this_gate=float(g.loc['Z_group_anomaly','missed_case_capture_at_20pct'])
        best_baseline=float(g.loc[['Raw_max_marker_z','Raw_Mahalanobis','PCA_max_axis_z','Near_risk_threshold'],
                                  'missed_case_capture_at_20pct'].max())
        comparison.append({'period':period,'delta_auc_z_minus_raw':delta_auc,
                           'delta_brier_raw_minus_z':delta_brier,
                           'Z_gate_miss_capture':this_gate,
                           'best_simple_gate_miss_capture':best_baseline,
                           'gate_advantage':this_gate-best_baseline,
                           'Z_gate_beats_all_controls':bool(this_gate>best_baseline),
                           'prediction_better_auc_and_brier':bool(delta_auc>0 and delta_brier>0)})
    pd.DataFrame(comparison).to_csv(OUT/'109_application_criteria.csv',index=False)
    manifest={
      'script':'109_clinical_utility_benchmark.py','seed':SEED,
      'analysis':'risk flag PHQ-9 >=10 (self-reported symptom screen, not diagnosis)',
      'periods':{s:int((split==s).sum()) for s in PERIODS},
      'frozen_physio_representation':'Original Script73 seven-variable outcome-independent Z, 2005-08 fitting',
      'all_fit_and_C_selection':'2005-08 only, class-stratified five-fold CV',
      'survey_weighted_point_metrics':True,'nhanes_design_variance':False,
      'case_bootstraps':reps,'case_bootstraps_not_design_CIs':True,
      'future_cohorts_previously_inspected':True,
      'native_methods':native_status,
      'note':'Full-rank RawX7/PCA7/Z7 have equivalent information; only numerically different regularized fits.',
      'gate_comparison':'20% weighted flag budget within each later low-risk cohort is a retrospective capacity-normalized comparison, NOT deployable prevalence calibration.',
      'frozen_gate_thresholds':'Also report discovery-based gate score thresholds for operational coverage drift.',
      'not_proven':['causality','individual depression drivers','clinical deployment','multimodal advantage',
                    'independent confirmation','survey-design statistical significance']}
    (OUT/'109_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    summary=pd.DataFrame(comparison)
    md=['# Script 109 — Clinical utility and reliability comparison',
      '', '## Question',
      'Does the original frozen 7D H/G representation improve PHQ-9>=10 screening, or flag missed cases better than standard biomarker and shift detectors?',
      '', '## Key held-out point results',
      '```',score[score.period.ne(PERIODS[0])][['period','method','n','auroc','auprc','brier','sensitivity','specificity']].to_string(index=False), '```',
      '', '## Matched 20% weighted review capacity among predicted low-risk participants',
      '```',gate_df[['period','gate','missed_case_capture_at_20pct','weighted_flag_fraction','frozen_threshold_flag_fraction','frozen_threshold_missed_capture']].to_string(index=False),'```',
      '', '## Direct criterion comparisons (positive values favor our method)',
      '```',summary.to_string(index=False),'```',
      '', '## Paired case-bootstrap intervals (descriptive, NOT design-based)',
      '```',pd.DataFrame(bootstrap_rows).to_string(index=False),'```',
      '', '## What would constitute an application advantage?',
      '- Screening: a convincing held-out improvement over RawX7 and PCA7 in AUROC and Brier, robust to uncertainty; not merely improvement over HbA1c.',
      '- Reliability: at equal review capacity, Z group anomaly should capture more false negatives than raw marker, Mahalanobis, PCA and threshold-proximity controls in both later eras.',
      '- Values are descriptive and exploratory; the earlier cohorts have been repeatedly inspected. This cannot certify clinical utility without independent testing.',
      '', '## Interpretation boundaries',
      '- The original Z7 is a reversible transformation of RawX7. It cannot create new biomarkers or information.',
      '- Full 7D PCA7 and RawX7 are L2-rotation-equivalent when standardized consistently. Small Z7 differences can be caused by regularization and feature scaling.',
      '- The structural gate is outcome-blind. Its usefulness is assessed retrospectively against missed high-PHQ cases, not validated clinical events.',
      '- The 20% matched-budget rule uses current-cohort distributions, so it is a diagnostic comparison, not an externally calibrated review threshold.',
      '- Original PHQ-9 measurement and cohort selection are observational and cannot explain individual causal contributors.',
      '- Case bootstrap uncertainty does not account for PSU/stratum design, nonresponse or repeated exploratory testing.',
      '- JIVE/AJIVE, if available, use their native discovery fits followed by train-only score adapters; projected scores are NOT native out-of-sample outputs.',
      '- Future fundus/vitals modalities remain untested: these results only assess current blood-based application feasibility.']
    (OUT/'109_readout.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('Wrote:',OUT)
    print('Read: 109_readout.md, 109_application_criteria.csv, 109_screening_metrics.csv, 109_missed_case_detection.csv')


def run_self_test():
    rng=np.random.default_rng(SEED)
    n=370
    w=rng.uniform(.3,2.7,n)
    train=np.zeros(n,dtype=bool);train[:220]=True
    cov=rng.normal(size=(n,4))
    raw=rng.normal(size=(n,7))
    z=raw@rng.normal(size=(7,7))
    y=(rng.uniform(size=n)<(1/(1+np.exp(-(.6*raw[:,0]-.5*raw[:,3]-1.1))))).astype(int)
    ids=np.flatnonzero(train)
    x=transform_features(cov,raw,train)
    pred,c,oof,_=select_model(x,y,w,ids,folds=3)
    cutoff=risk_threshold(y[ids],oof,w[ids])
    m=metrics(y[~train],pred[~train],w[~train],cutoff)
    assert 0<=m['auroc']<=1 and m['brier']>=0
    score=group_ood(z,w,train)
    assert len(score)==n and np.isfinite(score).all()
    gates=gate_scores(raw,raw,z,pred,cutoff,w,train)
    for v in gates.values():
        assert np.isfinite(v).all() and len(v)==n
    ee=gate_eval(y[~train],pred[~train],w[~train],score[~train],cutoff)
    assert ee and 0<=ee['missed_case_capture_at_20pct']<=1
    ci=paired_case_ci(y[~train],pred[~train],pred[~train],w[~train],reps=19)
    assert abs(ci['delta_brier_raw_minus_z'])<1e-12
    # Exact raw / full unwhitened PCA equivalence under same selected L2 penalty.
    rawz=StandardScaler().fit(raw[train]).transform(raw)
    pc=PCA(n_components=7).fit(rawz[train]).transform(rawz)
    a=transform_features(cov,rawz,train);b=transform_features(cov,pc,train)
    # PCA axis *standardization* changes penalty geometry; equivalence only
    # holds without axis-wise rescaling of PC scores.
    ac=np.column_stack([StandardScaler().fit(cov[train]).transform(cov),rawz]);
    bc=np.column_stack([StandardScaler().fit(cov[train]).transform(cov),pc]);
    f1=classifier(.1).fit(ac[train],y[train],sample_weight=normw(w[train]))
    f2=classifier(.1).fit(bc[train],y[train],sample_weight=normw(w[train]))
    # Same p to within solver tolerance.
    gap=float(np.max(abs(f1.predict_proba(ac)[:,1]-f2.predict_proba(bc)[:,1])))
    assert gap<1e-4, f'Unwhitened PCA7 rotation sanity test failed: {gap}'
    print('PASS: discovery-only CV, risk metrics, weighted 20% gate, pairwise bootstrap, rotation sanity check')


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--skip-native',action='store_true',help='Explicitly omit JIVE/AJIVE for a fast run')
    ap.add_argument('--bootstraps',type=int,default=BOOTSTRAPS)
    args=ap.parse_args()
    if args.bootstraps<19:ap.error('--bootstraps must be >=19')
    run(self_test=args.self_test,skip_native=args.skip_native,reps=args.bootstraps)
