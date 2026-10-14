#!/usr/bin/env python3
"""Script 98 | Frozen longitudinal-cycle transfer of interpretable phenotype equations.

Run from repository root:
    python EDA/98_regularized_phenotype_equation.py

2005-06 and 2007-08: model training + cycle-stratified cross-validation only.
2009-18: test only. 2021-23: independent later-period validation only.
No leakage, synthetic observations, refit of Script-72/73 transforms, Git writes,
or tuning on held-out periods. NHANES cycles are repeated cross-sections, not
longitudinal observations of the same individuals.

Model: weighted ordinary least squares / ridge / lasso / elastic net.
Groups: covariates; X = A + G; Z = shared + discordance + private A/G;
plus optional pairwise interactions. Outputs machine-readable, concrete
unstandardized equations, training-CV selection, held-out metrics, and
noncausal additive model contribution accounting.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EDA = ROOT / "EDA"
RESULTS = EDA / "Results"
INPUT = RESULTS / "73_frozen_phenotype_map_input.csv"
SEED = 98

A = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"]
G = ["LBXGH", "LBXGLU", "LOG_IN"]
X = A + G
Z = ["SP_SHARED1", "SP_SHARED2", "SP_DISCORD1", "SP_DISCORD2",
     "SP_A_PRIVATE3", "SP_A_PRIVATE4", "SP_G_PRIVATE3"]
PHQ = [f"DPQ0{i}0" for i in range(1, 10)]
SOMATIC = ["DPQ030", "DPQ040", "DPQ050"]
Y_TARGETS = {"PHQ_total":"PHQ9_TOTAL", "Somatic":"SOMATIC_SCORE", "CogAff":"COGAFF_SUM"}
CYCS = {"0506":"D", "0708":"E", "0910":"F", "1112":"G", "1314":"H",
        "1516":"I", "1718":"J", "2123":"L"}
PERIOD = {"0506":"train_2005-2008", "0708":"train_2005-2008",
          "0910":"test_2009-2018", "1112":"test_2009-2018", "1314":"test_2009-2018",
          "1516":"test_2009-2018", "1718":"test_2009-2018",
          "2123":"validate_2021-2023"}
NUM_COV = ["RIDAGEYR", "BMXBMI", "INDFMPIR", "EGFR_2021"]
CAT_COV = ["RIAGENDR", "RACE", "EDUC3", "SMOKING3"]
CLEAN_COV = ["RIDAGEYR", "BMXBMI", "RIAGENDR"]
FULL_COV = NUM_COV + CAT_COV
OUTPUTS = ("98_model_selection_cv.csv", "98_frozen_holdout_performance.csv",
           "98_equation_terms.csv", "98_model_contributions.csv",
           "98_equations.md", "98_primary_equations.md", "98_model_derivatives.csv",
           "98_phenotype_coherence.csv", "98_readout.md", "98_manifest.json", "98_cohort_coverage.csv")


def dpq_path(cyc):
    folder = ROOT / "Data" / ("NHANES" if cyc in ("0506", "0708") else "NHANES_Transfer") / cyc
    for ext in (".XPT", ".xpt"):
        p = folder / ("DPQ_" + CYCS[cyc] + ext)
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing real DPQ XPT for {cyc} in {folder}")


def load_real():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing frozen real NHANES file: {INPUT}")
    headers = list(pd.read_csv(INPUT, nrows=0).columns)
    need = ["SEQN", "CYCLE", "SURVEY_WT", "ELIGIBLE_ADULT_NONPREG", "PHQ9_TOTAL"] + X + Z + FULL_COV
    absent = sorted(set(need) - set(headers))
    if absent:
        raise ValueError(f"Frozen source missing required columns {absent}; do not fabricate or silently substitute")
    d = pd.read_csv(INPUT, usecols=need, low_memory=False)
    d["CYCLE"] = d["CYCLE"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(4)
    d = d.loc[d.CYCLE.isin(CYCS)].copy()
    d["SEQN"] = pd.to_numeric(d.SEQN, errors="coerce")
    for c in ["SURVEY_WT", "PHQ9_TOTAL"] + X + Z + NUM_COV + CAT_COV:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.loc[d.ELIGIBLE_ADULT_NONPREG.eq(1) & d.SURVEY_WT.gt(0)].copy()
    if d.SEQN.isna().any() or d.SEQN.duplicated().any():
        raise ValueError("Invalid/duplicate participant IDs in source")
    parts = []
    for cyc in CYCS:
        q = pd.read_sas(dpq_path(cyc), format="xport")
        if not set(["SEQN"] + PHQ).issubset(q.columns):
            raise ValueError(f"Raw PHQ-9 items absent from {cyc}")
        q = q[["SEQN"] + PHQ].copy()
        q.SEQN = pd.to_numeric(q.SEQN, errors="coerce")
        for c in PHQ:
            v = pd.to_numeric(q[c], errors="coerce")
            v = v.mask(v.abs() < 1e-10, 0.0)  # NHANES XPT near-zero fix, Scripts 81/95/96.
            q[c] = v.where(v.isin([0., 1., 2., 3.]))
        parts.append(q)
    items = pd.concat(parts, ignore_index=True)
    if items.SEQN.isna().any() or items.SEQN.duplicated().any():
        raise ValueError("Invalid/duplicate IDs in source PHQ XPTs")
    d = d.merge(items, how="left", on="SEQN", validate="one_to_one")
    d["SPLIT"] = d.CYCLE.map(PERIOD)
    all_y = d[PHQ].notna().all(axis=1) & d.PHQ9_TOTAL.notna()
    mismatch = all_y & (~np.isclose(d[PHQ].sum(axis=1), d.PHQ9_TOTAL))
    if mismatch.any():
        raise AssertionError(f"PHQ9 item totals disagree with stored total for {int(mismatch.sum())} rows")
    d["SOMATIC_SCORE"] = d[SOMATIC].sum(axis=1, min_count=3)
    d["COGAFF_SUM"] = d.PHQ9_TOTAL - d.SOMATIC_SCORE
    base_good = all_y & d[X + Z].notna().all(axis=1)
    cov_good = d[FULL_COV].notna().all(axis=1)
    for p in ["train_2005-2008", "test_2009-2018", "validate_2021-2023"]:
        sub = d[d.SPLIT.eq(p)]
        msg = dict(period=p, eligible=len(sub), complete_Y_X_Z=int(base_good.loc[sub.index].sum()),
                   complete_full_covars=int(cov_good.loc[sub.index].sum()),
                   common_complete=int((base_good & cov_good).loc[sub.index].sum()))
        yield_msg.append(msg)
    used = d.loc[base_good & cov_good].copy()
    if any((used.SPLIT == p).sum() < 200 for p in ["train_2005-2008", "test_2009-2018", "validate_2021-2023"]):
        raise ValueError(f"Too few matched complete genuine observations. See coverage: {yield_msg}")
    return used


def wmetrics(y, yh, w):
    y, yh, w = map(lambda x: np.asarray(x, float), (y, yh, w))
    m = np.average(y, weights=w)
    den = np.sum(w * (y-m)**2)
    return (float(1.0 - np.sum(w * (y-yh)**2)/den) if den > 0 else None,
            float(np.sqrt(np.average((y-yh)**2, weights=w))),
            float(np.average(np.abs(y-yh), weights=w)),
            float(np.average(yh-y, weights=w)))


def make_category_string(s):
    # Preserve integer-coded categorical values consistently in all periods.
    return pd.to_numeric(s, errors="raise").astype(np.int64).astype(str)


def build_features(d):
    tr = d.SPLIT.eq("train_2005-2008")
    out = pd.DataFrame(index=d.index)
    for c in X + Z + NUM_COV:
        out[c] = d[c].to_numpy(float)
    for c in CAT_COV:
        v = make_category_string(d[c])
        allowed = sorted(set(v[tr]))
        baseline = allowed[0]
        for level in allowed[1:]:
            out[c + "=" + level] = (v == level).to_numpy(dtype=float)
        unseen = sorted(set(v[~tr]) - set(allowed))
        if unseen:
            print(f"WARNING: category {c} has unseen holdout levels {unseen}; encoded as reference all-zero.")
        categories_reference[c] = baseline
    # Interaction features are defined with discovery-period means and SDs only.
    w = d.loc[tr, "SURVEY_WT"].to_numpy(dtype=float, copy=True)
    w = w / w.mean()
    for c in X + Z:
        mu = float(np.average(out.loc[tr, c], weights=w))
        sd = float(np.sqrt(np.average((out.loc[tr, c]-mu)**2, weights=w)))
        if sd <= 1e-12:
            raise ValueError(f"Degenerate physiology predictor {c}")
        normalizer[c] = {"mean":mu, "sd":sd}
    for a, g in itertools.product(A, G):
        out[a + "*" + g] = ((out[a]-normalizer[a]["mean"])/normalizer[a]["sd"]
                             * (out[g]-normalizer[g]["mean"])/normalizer[g]["sd"])
    for a, b in itertools.combinations(Z, 2):
        out[a + "*" + b] = ((out[a]-normalizer[a]["mean"])/normalizer[a]["sd"]
                             * (out[b]-normalizer[b]["mean"])/normalizer[b]["sd"])
    if not np.isfinite(out.to_numpy(float)).all():
        raise ValueError("Nonfinite constructed model feature")
    return out


def covariates(scope):
    use = CLEAN_COV if scope == "lean" else FULL_COV
    ans = []
    for c in use:
        if c in CAT_COV:
            ans.extend(sorted(k for k in categories_feature_names if k.startswith(c + "=")))
        else:
            ans.append(c)
    return ans


def specifications():
    specs=[]
    for scope in ("lean", "full"):
        c = covariates(scope)
        specs += [(scope, "covariates", list(c)),
                  (scope, "Z_main", c + Z),
                  (scope, "X_main", c + X),
                  (scope, "X_AxG", c + X + [a+"*"+g for a,g in itertools.product(A,G)]),
                  (scope, "Z_pairwise", c + Z + [a+"*"+b for a,b in itertools.combinations(Z,2)])]
    return specs


def model_options():
    # These grids are set without inspecting later periods. Normalized weights mean ~1.
    yield ("OLS", 0.0, None)
    for a in (0.01, 0.1, 1., 10., 100., 1000.):
        yield ("Ridge", float(a), None)
    for a in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5):
        yield ("LASSO", float(a), None)
    for a, ratio in itertools.product((0.001, 0.01, 0.05, 0.1, 0.25), (0.2, 0.5, 0.8)):
        yield ("ElasticNet", float(a), float(ratio))


def instantiate(method, alpha, l1_ratio):
    if method == "OLS":return LinearRegression()
    if method == "Ridge":return Ridge(alpha=alpha, random_state=SEED)
    if method == "LASSO":return Lasso(alpha=alpha, max_iter=50000, tol=1e-4, selection="cyclic")
    if method == "ElasticNet":return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50000, tol=1e-4)
    raise KeyError(method)


def fit(train_x, train_y, train_w, method, alpha, ratio):
    sc = StandardScaler()
    z = sc.fit_transform(train_x, sample_weight=train_w)
    m = instantiate(method, alpha, ratio)
    m.fit(z, train_y, sample_weight=train_w/train_w.mean())
    return sc, m


def prediction(sc, m, new_x):
    return np.asarray(m.predict(sc.transform(new_x)),float)


def add_block(key):
    if "*" in key:
        return "A_by_G" if key.split("*")[0] in X else "Z_interactions"
    if key in A:return "A_private_raw"
    if key in G:return "G_private_raw"
    if key.startswith("SP_SHARED"):return "Shared"
    if key.startswith("SP_DISCORD"):return "Discordant"
    if key.startswith("SP_A_PRIVATE"):return "A_private"
    if key.startswith("SP_G_PRIVATE"):return "G_private"
    return "Covariates"


def format_num(x):
    return f"{float(x):+.8g}"


def markdown_table(df):
    try:
        return df.to_markdown(index=False, floatfmt=".5f")
    except ImportError:
        # tabulate is optional; never fail after model training just for formatting.
        return "```csv\n" + df.to_csv(index=False) + "\n```"


def write_equation(rec, ctab, yname, exact_features):
    headline = f"### {yname}: {rec['scope']} / {rec['design']} / {rec['method']}"
    eq = f"\\hat{{Y}} = {rec['intercept_at_zero']:.8g}"
    for _,r in ctab.iterrows():
        if abs(r.beta_raw) > 1e-11:
            feature_latex = str(r.feature).replace("_", r"\_")
            eq += rf" {format_num(r.beta_raw)} \cdot \mathrm{{{feature_latex}}}"
    s = [headline, "", "Model equation on ORIGINAL feature basis (not z-scored in this display):", "", "$$", eq, "$$", "",
         "Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.",
         "Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.",
         "The intercept is the value when all included original features equal 0; it is not a biological baseline.",
         ""]
    return "\n".join(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    if args.cv_folds < 3 or args.cv_folds > 10:
        raise ValueError("CV folds must be between 3 and 10")
    RESULTS.mkdir(parents=True, exist_ok=True)
    global yield_msg, normalizer, categories_reference, categories_feature_names
    yield_msg=[]; normalizer={}; categories_reference={}; categories_feature_names=[]
    print("SCRIPT 98 | Fitted, frozen regularized phenotype equations")
    d = load_real()
    cvg = pd.DataFrame(yield_msg)
    cvg.to_csv(RESULTS/"98_cohort_coverage.csv",index=False)
    print("\nCOHORT:\n",cvg.to_string(index=False))
    print("Matched rows:",d.groupby("SPLIT").size().to_dict())
    F = build_features(d)
    categories_feature_names = [c for c in F.columns if "=" in c]
    specs=specifications()
    split_train = d.SPLIT.eq("train_2005-2008").to_numpy()
    train_idx = np.flatnonzero(split_train)
    weights = d.SURVEY_WT.to_numpy(float)
    cycles = d.CYCLE.to_numpy(str)
    rows_cv=[]; rows_perf=[]; rows_terms=[]; rows_contrib=[]; rows_derivs=[]
    eq_sections=[]; eq_by_key={}; winners=[]
    options = list(model_options())
    cv = list(StratifiedKFold(n_splits=args.cv_folds,shuffle=True,random_state=SEED).split(
        np.arange(len(train_idx)), cycles[train_idx]))
    print(f"\n{len(specs)} design/scopes, {len(options)} candidates each, {args.cv_folds} folds, {len(Y_TARGETS)} outcomes")
    for yname, outcome in Y_TARGETS.items():
        y = d[outcome].to_numpy(float)
        for scope, design, features in specs:
            matrix=F[features].to_numpy(float)
            vals=[]
            for method,alpha,ratio in options:
                errors=[]; r2s=[]
                for i,j in cv:
                    it=train_idx[i]; iv=train_idx[j]
                    sc,m=fit(matrix[it],y[it],weights[it],method,alpha,ratio)
                    yhat=prediction(sc,m,matrix[iv])
                    r2,rmse,mae,bias=wmetrics(y[iv],yhat,weights[iv])
                    # Fold-relative RMSE is less additive than W-MSE; select W-MSE.
                    errors.append(rmse**2);r2s.append(r2)
                row=dict(outcome=yname,scope=scope,design=design,n_features=len(features),method=method,
                         alpha=alpha,l1_ratio=ratio,cv_mean_weighted_MSE=float(np.mean(errors)),
                         cv_sd_weighted_MSE=float(np.std(errors)),cv_mean_weighted_R2=float(np.mean(r2s)))
                vals.append(row); rows_cv.append(row)
            best=min(vals,key=lambda r:(r["cv_mean_weighted_MSE"],r["n_features"],r["method"]))
            winners.append(best)
            sc,m=fit(matrix[train_idx],y[train_idx],weights[train_idx],best["method"],best["alpha"],best["l1_ratio"])
            betas = np.atleast_1d(np.asarray(m.coef_,float))
            if betas.shape[0] != len(features):
                raise AssertionError("Unexpected coefficient shape")
            beta_raw=betas/np.asarray(sc.scale_)
            intercept=float(m.intercept_ - np.dot(beta_raw,sc.mean_))
            eqrec={**best,"intercept_at_zero":intercept,"n_train":len(train_idx)}
            full_tab=[]
            for col,b,braw,mu,sd in zip(features,betas,beta_raw,sc.mean_,sc.scale_):
                row=dict(**{k:eqrec[k] for k in ("outcome","scope","design","method","alpha","l1_ratio")},
                         feature=col,block=add_block(col),beta_standardized=float(b),beta_raw=float(braw),
                         discovery_mean=float(mu),discovery_sd=float(sd),intercept_raw=float(intercept))
                rows_terms.append(row);full_tab.append(row)
            # Equation written for EACH pre-selected design, not only winner by test-set performance.
            equation_text=write_equation(eqrec,pd.DataFrame(full_tab),yname,features)
            eq_sections.append(equation_text)
            eq_by_key[(yname,scope,design)]=equation_text
            basis = dict(zip(features,beta_raw))
            base_physiology = X if design.startswith("X") else Z if design.startswith("Z") else []
            for v in base_physiology:
                main_beta=float(basis.get(v,0.0))
                cross=[]
                for f,beta in basis.items():
                    if "*" not in f or v not in f.split("*"):
                        continue
                    f1,f2=f.split("*")
                    other=f2 if f1==v else f1
                    # The design's interactions are centered and scaled by
                    # discovery physiology SD, then standardized for fitting.
                    factor=float(beta/(normalizer[v]["sd"]*normalizer[other]["sd"]))
                    cross.append(dict(other=other,other_train_mean=normalizer[other]["mean"],
                                      coefficient=factor))
                expression=f"{main_beta:+.8g}"
                for term in cross:
                    expression += f" {term['coefficient']:+.8g}*({term['other']}-{term['other_train_mean']:.8g})"
                rows_derivs.append(dict(outcome=yname,scope=scope,design=design,
                    physiology_variable=v, derivative_at_discovery_means=main_beta,
                    derivative_expression=expression,
                    interaction_details_json=json.dumps(cross),
                    caveat="Derivative of fitted conditional association, NOT causal effect / PDE solution"))
            for p in ("train_2005-2008", "test_2009-2018", "validate_2021-2023"):
                idx=d.SPLIT.eq(p).to_numpy()
                pred=prediction(sc,m,matrix[idx]);truth=y[idx];w=weights[idx]
                r2,rmse,mae,bias=wmetrics(truth,pred,w)
                rows_perf.append(dict(outcome=yname,scope=scope,design=design,method=best["method"],
                    alpha=best["alpha"],l1_ratio=best["l1_ratio"],period=p,n=int(idx.sum()),
                    weighted_R2=r2,weighted_RMSE=rmse,weighted_MAE=mae,weighted_mean_prediction_bias=bias,
                    chosen_on="training_CV_only"))
                # Predictions = mean_train_prediction + sum(block deviations around train means).
                base=float(m.intercept_)
                centered=matrix[idx]-sc.mean_[None,:]
                contribution=centered*beta_raw[None,:]
                if not np.allclose(base + contribution.sum(axis=1),pred,rtol=1e-8,atol=1e-6):
                    raise AssertionError("Model contribution accounting failed")
                for block in sorted({add_block(q) for q in features}):
                    cc=contribution[:,[i for i,c in enumerate(features) if add_block(c)==block]].sum(axis=1)
                    rows_contrib.append(dict(outcome=yname,scope=scope,design=design,
                        period=p,block=block,n=int(idx.sum()),
                        weighted_mean_signed=float(np.average(cc,weights=w)),
                        weighted_mean_absolute=float(np.average(np.abs(cc),weights=w)),
                        weighted_RMS=float(np.sqrt(np.average(cc**2,weights=w))),
                        train_anchor_prediction=base,
                        caveat="Additive model term only; not causal A/G percentage or clinical contribution"))
            print(f"  {yname:10s} {scope:4s} {design:12s} {best['method']:10s} CV MSE={best['cv_mean_weighted_MSE']:.4f}")
    sel=pd.DataFrame(rows_cv);perf=pd.DataFrame(rows_perf);terms=pd.DataFrame(rows_terms);con=pd.DataFrame(rows_contrib)
    sel.to_csv(RESULTS/"98_model_selection_cv.csv",index=False)
    perf.to_csv(RESULTS/"98_frozen_holdout_performance.csv",index=False)
    terms.to_csv(RESULTS/"98_equation_terms.csv",index=False)
    con.to_csv(RESULTS/"98_model_contributions.csv",index=False)
    pd.DataFrame(rows_derivs).to_csv(RESULTS/"98_model_derivatives.csv",index=False)
    # Choose report winner based *only* on training CV; no holdout selection.
    selected={y:pd.DataFrame([w for w in winners if w["outcome"]==y]).sort_values(
        ["cv_mean_weighted_MSE","n_features"]).iloc[0].to_dict() for y in Y_TARGETS}
    # The directly optimized PHQ-total equation need not equal the sum of
    # independently optimized somatic and cognitive-affective equations.
    # Quantify that lack of phenotype-score coherence; don't hide it.
    coherent_predictions={}
    for yy, chosen in selected.items():
        spec = next(z for z in specs if z[0]==chosen["scope"] and z[1]==chosen["design"])
        xm=F[spec[2]].to_numpy(float)
        yy_actual=d[Y_TARGETS[yy]].to_numpy(float)
        sc, fitted=fit(xm[train_idx], yy_actual[train_idx],weights[train_idx],
                      chosen["method"],chosen["alpha"],chosen["l1_ratio"])
        coherent_predictions[yy]=prediction(sc,fitted,xm)
    coher=[]
    for p in ("train_2005-2008", "test_2009-2018", "validate_2021-2023"):
        mask=d.SPLIT.eq(p).to_numpy()
        direct=coherent_predictions["PHQ_total"][mask]
        summed=(coherent_predictions["Somatic"][mask]+coherent_predictions["CogAff"][mask])
        target=d.PHQ9_TOTAL.to_numpy(float)[mask]
        w=weights[mask]
        rd,_,_,_=wmetrics(target,direct,w)
        rs,_,_,_=wmetrics(target,summed,w)
        coher.append(dict(period=p,n=int(mask.sum()),direct_total_weighted_R2=rd,
            coherent_sum_weighted_R2=rs,
            weighted_RMSE_disagreement=float(np.sqrt(np.average((direct-summed)**2,weights=w))),
            note="Somatic+CogAff is algebraically coherent, but independently CV-selected models need not agree with separately fit PHQ-total model"))
    pd.DataFrame(coher).to_csv(RESULTS/"98_phenotype_coherence.csv",index=False)
    headline=[]
    for yname,s in selected.items():
        p=perf[(perf.outcome==yname)&(perf.scope==s["scope"])&(perf.design==s["design"])].copy()
        headline.append(dict(outcome=yname,selected_scope=s["scope"],selected_design=s["design"],
                             method=s["method"],alpha=s["alpha"],l1_ratio=s["l1_ratio"],
                             cv_MSE=s["cv_mean_weighted_MSE"],
                             test_R2=float(p.loc[p.period.eq("test_2009-2018"),"weighted_R2"].iloc[0]),
                             validation_R2=float(p.loc[p.period.eq("validate_2021-2023"),"weighted_R2"].iloc[0])))
    lines=["# Script 98 — Fitted phenotype equations","",
           "Frozen NHANES cycle protocol: train 2005–08 (including training-only 5-fold CV); test 2009–18; validate 2021–23.",
           "No tuning using test or validation outcomes. Discovery transformation Z is unchanged.",
           "All designs use the same fully observed X/Z/Y/covariate cohort for fair score comparison.",
           "", "## CV-selected overall design for each outcome", "",
           markdown_table(pd.DataFrame(headline)), "",
           "## Every fitted equation", "", *eq_sections,
           "## Interpretation", "",
           "- A model coefficient is an adjusted association/conditional model sensitivity, not a causal effect.",
           "- Model contribution magnitudes depend on centering, feature basis and penalty; do not label them biological percentages.",
           "- A combined Z representation and raw X include the same underlying information, not independent modalities.",
           "- Later NHANES cycles are distinct participants, not repeated measurements of the same individuals.",
           "- Complex NHANES survey-design standard errors and bootstrap confidence intervals are not supplied here.", ""]
    (RESULTS/"98_equations.md").write_text("\n".join(lines),encoding="utf-8")
    focused = ["# Script 98 — Primary numerical regression equations", "",
               "All coefficients fitted using NHANES 2005–08 exclusively; model choice from training CV.",
               "The label 'test 2009–18' refers to the five available 2-year NHANES cycles, not 2019 subjects.", ""]
    for yname,s in selected.items():
        focused += ["## "+yname,"", "### Overall training-CV winner", "",
                    eq_by_key[(yname,s["scope"],s["design"])],""]
        for preferred in ("Z_main","X_main"):
            ws = [w for w in winners if w["outcome"] == yname and w["design"] == preferred]
            pick=min(ws,key=lambda r:r["cv_mean_weighted_MSE"])
            focused += ["### Best "+preferred+" equation (CV selection)","",
                        eq_by_key[(yname,pick["scope"],preferred)],""]
    focused += ["## Caution", "", "These are predictive/associative equations, not PDEs or causal models.",
                "Centering, units, contrast baselines and raw interaction definitions are documented in the full equations file."]
    (RESULTS/"98_primary_equations.md").write_text("\n".join(focused),encoding="utf-8")
    readout=["# Script 98 — Regularized phenotype equation readout", "",
             "## Best designs selected entirely by 2005–08 training CV", "",
             markdown_table(pd.DataFrame(headline)),"",
             "## Phenotype score coherence: independently fitted versus component-summed total", "",
             markdown_table(pd.DataFrame(coher)),"",
             "## Holdout performance (all designs; selection made on train only)","",
             markdown_table(perf[perf.period.ne("train_2005-2008")]),"",
             "## Key boundaries", "",
             "The regression explains conditional association, not causal A/G influence.",
             "NHANES cycle holdouts are temporal-population transfer, not person-level longitudinal validation.",
             "A negative R2 indicates poor generalization relative to holdout weighted mean.",
             "All metrics are survey-weighted point estimates without full design variance inference."]
    (RESULTS/"98_readout.md").write_text("\n".join(readout),encoding="utf-8")
    manifest=dict(script=Path(__file__).name,input=str(INPUT.relative_to(ROOT)),
       source="genuine Script73 frozen Z and raw DPQ items",train_cycles=["0506","0708"],
       test_cycles=["0910","1112","1314","1516","1718"], validation_cycles=["2123"],
       selection="5-fold (configurable) training-only cycle-stratified CV, smallest weighted MSE",
       designs=len(specs),candidates_per_design=len(options),outcomes=list(Y_TARGETS),
       matched_cohort=d.groupby("SPLIT").size().to_dict(),
       full_covariates=FULL_COV,lean_covariates=CLEAN_COV,
       categorical_reference=categories_reference,interaction_raw_scalers=normalizer,
       outcome_winners=headline,outputs=list(OUTPUTS), frozen_transforms_modified=False,
       causality=False,within_subject_longitudinal=False,survey_weighted_point_metrics_only=True,
       note="Equations in 98_equations.md and 98_equation_terms.csv are fitted to genuine local NHANES on run; selection uses training CV only.")
    (RESULTS/"98_manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    print("\nCV-SELECTED OUTCOMES:")
    print(pd.DataFrame(headline).to_string(index=False))
    print("\nACTUAL FITTED EQUATIONS: EDA/Results/98_primary_equations.md")
    print("READOUT: EDA/Results/98_readout.md")
    print("DONE. Frozen Script-72/73 assets unchanged.")


if __name__=="__main__":
    main()
