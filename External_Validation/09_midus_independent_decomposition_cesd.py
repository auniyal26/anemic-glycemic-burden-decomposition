#!/usr/bin/env python3
"""MIDUS independent glycemic decomposition, reconstruction, phenotype mapping and reverse inference."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "MIDUS" / "raw"
OUT = ROOT / "External_Validation" / "Results"
SEED = 20261019
FOLDS = 5
ALPHA = 10.0
NAMES = ("HbA1c", "Glucose", "logInsulin")
TARGETS = ("CESD_total", "CESD_somatic", "CESD_affect")
SMALL = "SUPPRESSED_0_TO_10"
BOOT = 1200
FILES = {
    "CORE": r"^M2_BIO_AGGREGATE_.*\.sav$",
    "REFRESHER": r"^MR1_BIO_AGGREGATE_.*\.sav$",
}
FIELDS = {
    "CORE": {
        "HbA1c": "B4BHA1C", "Glucose": "B4BGLUC", "Insulin": "B4BINSLN",
        "CESD_total": "B4QCESD", "CESD_somatic": "B4QCESDSC",
        "CESD_affect": "B4QCESDDA", "Age": "B4ZAGE", "Sex": "B1PRSEX",
    },
    "REFRESHER": {
        "HbA1c": "RA4BHA1C", "Glucose": "RA4BGLUC", "Insulin": "RA4BINSLN",
        "CESD_total": "RA4QCESD", "CESD_somatic": "RA4QCESDSC",
        "CESD_affect": "RA4QCESDDA", "Age": "RA4ZAGE", "Sex": "RA1PRSEX",
    },
}
SPECS = ("covariates", "HbA1c", "raw_G3", "PC1", "PC1_2", "PC1_2_3")
REVERSE = ("covariates", "CESD_total", "CESD_subscales")


def disclose(n: int) -> str:
    return str(int(n)) if int(n) > 10 else SMALL


def finite(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def get_file(cohort):
    if not RAW.is_dir():
        raise FileNotFoundError(f"MIDUS raw directory not found: {RAW}")
    paths = [p for p in RAW.rglob("*") if p.is_file() and re.fullmatch(FILES[cohort], p.name, flags=re.I)]
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one {cohort} source; found {[p.name for p in paths]}")
    return paths[0]


def read_cohort(cohort):
    import pyreadstat
    path = get_file(cohort)
    _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
    lookup = {c.upper(): c for c in meta.column_names}
    id_name = "M2ID" if cohort == "CORE" else "MRID"
    expected = [id_name, *FIELDS[cohort].values()]
    absent = [x for x in expected if x.upper() not in lookup]
    if absent:
        raise RuntimeError(f"{cohort}: required fields missing: {absent}")
    cols = list(dict.fromkeys(lookup[x.upper()] for x in expected))
    d, _ = pyreadstat.read_sav(str(path), usecols=cols, user_missing=False, apply_value_formats=False)
    id_source = lookup[id_name.upper()]
    if d[id_source].isna().any() or d[id_source].duplicated().any():
        raise RuntimeError(f"{cohort}: missing/duplicate participant ID")
    d = d.set_index(id_source).rename(columns={lookup[src.upper()]: name for name, src in FIELDS[cohort].items()})
    return screen(d), {"filename": path.name, "sha256": file_digest(path)}


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def screen(df):
    d = df.copy()
    for col in ("HbA1c", "Glucose", "Insulin", *TARGETS, "Age", "Sex"):
        d[col] = finite(d[col])
    ok = (
        d["HbA1c"].between(2.5, 20)
        & d["Glucose"].between(40, 600)
        & d["Insulin"].between(0.01, 500)
        & d["Age"].between(18, 110)
        & d["Sex"].isin([1, 2])
    )
    for y in TARGETS:
        ok &= d[y].between(0, 60)
    d = d.loc[ok].copy()
    d["logInsulin"] = np.log(d["Insulin"])
    if d.index.duplicated().any() or len(d) < 100:
        raise RuntimeError(f"Insufficient independent complete observations (N={len(d)})")
    return d


def g3(df):
    return df.loc[:, NAMES].to_numpy(dtype=float)


def covars(df):
    return np.column_stack((df["Age"].to_numpy(float), (df["Sex"] == 2).to_numpy(float)))


def learn_basis(x):
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0, ddof=0)
    if np.min(sd) < 1e-12:
        raise RuntimeError("Degenerate physiological biomarker")
    xz = (x - mu) / sd
    pca = PCA(n_components=3, svd_solver="full")
    pca.fit(xz)
    V = pca.components_.T.copy()
    # Standardize signs across independently learned bases without observing outcomes.
    for k in range(V.shape[1]):
        i = int(np.argmax(np.abs(V[:, k])))
        if V[i, k] < 0:
            V[:, k] *= -1
    return {"mu": mu, "sd": sd, "V": V, "ratio": pca.explained_variance_ratio_}


def encode(x, b):
    return ((x - b["mu"]) / b["sd"]) @ b["V"]


def decode(z, b, k):
    return (z[:, :k] @ b["V"][:, :k].T) * b["sd"] + b["mu"]


def feature(df, b, spec):
    cv = covars(df)
    if spec == "covariates":
        return cv
    if spec == "HbA1c":
        return np.column_stack((cv, df["HbA1c"].to_numpy(float)))
    if spec == "raw_G3":
        return np.column_stack((cv, g3(df)))
    count = {"PC1": 1, "PC1_2": 2, "PC1_2_3": 3}[spec]
    return np.column_stack((cv, encode(g3(df), b)[:, :count]))


def reverse_feature(df, spec):
    cv = covars(df)
    if spec == "covariates":
        return cv
    if spec == "CESD_total":
        return np.column_stack((cv, df["CESD_total"].to_numpy(float)))
    return np.column_stack((cv, df["CESD_somatic"].to_numpy(float), df["CESD_affect"].to_numpy(float)))


def model_fit(x, y):
    scale = StandardScaler().fit(x)
    model = Ridge(alpha=ALPHA).fit(scale.transform(x), y)
    return scale, model


def model_predict(fit, x):
    return fit[1].predict(fit[0].transform(x))


def mse(y, p):
    return float(np.mean((y - p) ** 2))


def r2(y, p):
    den = float(np.sum((y - np.mean(y)) ** 2))
    return float(1 - np.sum((y - p) ** 2) / den) if den > 0 else np.nan


def metrics_row(stage, outcome, spec, y, pred, n, reference):
    return {"stage": stage, "outcome": outcome, "model": spec, "n": disclose(n),
            "mse": mse(y, pred), "r2": r2(y, pred), "mae": float(np.mean(np.abs(y-pred))),
            "baseline_mse": reference, "mse_gain_vs_covariates": reference - mse(y, pred)}


def discovery_cv(core):
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = {(y, m): np.full(len(core), np.nan) for y in TARGETS for m in SPECS}
    for it, iv in splitter.split(core):
        tr, va = core.iloc[it], core.iloc[iv]
        b = learn_basis(g3(tr))
        for sp in SPECS:
            xtr, xva = feature(tr, b, sp), feature(va, b, sp)
            for y in TARGETS:
                fit = model_fit(xtr, tr[y].to_numpy(float))
                oof[y, sp][iv] = model_predict(fit, xva)
    rows = []
    for y in TARGETS:
        truth = core[y].to_numpy(float)
        base = mse(truth, oof[y, "covariates"])
        for sp in SPECS:
            rows.append(metrics_row("CORE_5fold_CV", y, sp, truth, oof[y,sp], len(core), base))
    return rows


def paired_gain_ci(y, base, alt, seed):
    diff = (y-base)**2 - (y-alt)**2
    rng = np.random.default_rng(seed)
    n = len(y)
    # Participant-paired case bootstrap; not a survey/attrition-aware interval.
    sampled = rng.integers(0, n, (BOOT, n))
    values = np.mean(diff[sampled], axis=1)
    return float(np.mean(diff)), float(np.quantile(values,.025)),float(np.quantile(values,.975))


def frozen_forward(core, refresh, b):
    rows=[]; pairs=[]
    for y in TARGETS:
        truth = refresh[y].to_numpy(float)
        pred = {}
        for sp in SPECS:
            fit = model_fit(feature(core,b,sp), core[y].to_numpy(float))
            pred[sp] = model_predict(fit, feature(refresh,b,sp))
        base = mse(truth, pred["covariates"])
        for sp in SPECS:
            rows.append(metrics_row("REFRESHER_frozen_transfer", y, sp, truth, pred[sp],len(refresh),base))
        for j, (base_spec,alt_spec) in enumerate((("covariates","PC1_2"),("HbA1c","PC1_2"),("covariates","raw_G3"))):
            delta, lo, hi = paired_gain_ci(truth,pred[base_spec],pred[alt_spec],SEED+40+10*TARGETS.index(y)+j)
            pairs.append({"outcome":y,"comparator":base_spec,"candidate":alt_spec,
                          "n":disclose(len(refresh)),"mse_gain":delta,
                          "case_bootstrap_ci_low":lo,"case_bootstrap_ci_high":hi,
                          "descriptive_direction":"favors_candidate" if lo>0 else "favors_comparator" if hi<0 else "inconclusive"})
    return rows,pairs


def reconstruct(core, refresh, b):
    rows=[]
    for label, d in (("CORE_discovery",core),("REFRESHER_frozen",refresh)):
        x = g3(d)
        z=encode(x,b)
        norm = (x-b["mu"]) / b["sd"]
        denom=float(np.mean(norm**2))
        for k in (1,2,3):
            hat=decode(z,b,k)
            err=(x-hat)/b["sd"]
            rmse=float(np.sqrt(np.mean(err**2)))
            rows.append({"stage":label,"n":disclose(len(d)),"k":k,
                "standardized_rmse":rmse,"reference_energy_retained":1-np.mean(err**2)/denom,
                "max_abs_reconstruction_error":float(np.max(np.abs(x-hat))) if k==3 else np.nan})
    return rows


def rediscovery(core, refresh, b):
    other = learn_basis(g3(refresh))
    rows=[]
    for k in (1,2,3):
        similarity=abs(float(np.dot(b["V"][:,k-1],other["V"][:,k-1])))
        cross=np.linalg.svd(b["V"][:,:k].T@other["V"][:,:k],compute_uv=False)
        rows.append({"k":k,"matched_pc_abs_cosine":similarity,"min_subspace_cosine":float(np.min(cross)),
                     "CORE_var_ratio":float(b["ratio"][k-1]), "REFRESHER_var_ratio":float(other["ratio"][k-1])})
    return rows,other


def reverse_audit(core, refresh, basis):
    ztr, zva = encode(g3(core),basis),encode(g3(refresh),basis)
    preds={}; bounds={}; rows=[]; coverage=[]
    split=KFold(n_splits=FOLDS,shuffle=True,random_state=SEED+1)
    for spec in REVERSE:
        xin=reverse_feature(core,spec)
        oof=np.zeros_like(ztr)
        for it,iv in split.split(core):
            f=model_fit(xin[it],ztr[it]);oof[iv]=model_predict(f,xin[iv])
        # Empirical coordinate-wise OOF absolute error quantiles: approximate, not formal conformal guarantee.
        bounds[spec]=np.quantile(np.abs(ztr-oof),0.90,axis=0)
        full=model_fit(xin,ztr)
        preds[spec]=model_predict(full,reverse_feature(refresh,spec))
        for k in range(3):
            coverage.append({"reverse_model":spec,"component":f"PC{k+1}",
                             "n":disclose(len(refresh)),"calibration_q90_abs_error":float(bounds[spec][k]),
                             "REFRESHER_marginal_coverage":float(np.mean(np.abs(zva[:,k]-preds[spec][:,k])<=bounds[spec][k]))})
            rows.append({"reverse_model":spec,"component":f"PC{k+1}","n":disclose(len(refresh)),
                         "CORE_oof_mse":mse(ztr[:,k],oof[:,k]),"REFRESHER_mse":mse(zva[:,k],preds[spec][:,k])})
    by={(x["reverse_model"],x["component"]):x for x in rows}
    for row in rows:
        v=by[("covariates",row["component"])]["REFRESHER_mse"]
        row["REFRESHER_gain_vs_covariates"]=v-row["REFRESHER_mse"]
    return rows,coverage


def run(core, refresh, sources, output):
    b=learn_basis(g3(core))
    reconstruction=reconstruct(core,refresh,b)
    rediscover, other=rediscovery(core,refresh,b)
    cv=discovery_cv(core)
    transfer,pairs=frozen_forward(core,refresh,b)
    reverse,coverage=reverse_audit(core,refresh,b)
    if max(x["max_abs_reconstruction_error"] for x in reconstruction if x["k"]==3)>1e-8:
        raise RuntimeError("Failed full-rank reconstruction sanity gate")
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(reconstruction).to_csv(output/"09_reconstruction.csv",index=False)
    pd.DataFrame(rediscover).to_csv(output/"09_rediscovery.csv",index=False)
    pd.DataFrame(cv+transfer).to_csv(output/"09_forward_cesd.csv",index=False)
    pd.DataFrame(pairs).to_csv(output/"09_forward_paired_ci.csv",index=False)
    pd.DataFrame(reverse).to_csv(output/"09_reverse_inference.csv",index=False)
    pd.DataFrame(coverage).to_csv(output/"09_reverse_coverage.csv",index=False)
    manifest={"experiment":"MIDUS independent discovery G3 to CES-D", "script":"09_midus_independent_decomposition_cesd.py",
              "seed":SEED,"ridge_alpha":ALPHA,"discovery_cv_folds":FOLDS,"case_bootstrap_reps":BOOT,
              "inputs":sources,"core_n":disclose(len(core)),"refresher_n":disclose(len(refresh)),
              "source_features":list(NAMES),"target_fields":list(TARGETS),"covariates":["Age","Sex"],
              "frozen_CORE_component_mean":b["mu"].tolist(),"frozen_CORE_component_sd":b["sd"].tolist(),
              "frozen_CORE_loadings":b["V"].tolist(),"frozen_CORE_variance_ratios":b["ratio"].tolist(),
              "REFRESHER_independently_rediscovered_loadings":other["V"].tolist(),
              "REFRESHER_independently_rediscovered_variance_ratios":other["ratio"].tolist(),
              "NHANES_used_for_fit_or_variable_selection":False,"causal_identification":False,
              "laboratory_protocol_verified":False,"CESD_invariance_verified":False,
              "sampling_weights_or_attrition_adjusted":False,"individual_records_exported":False}
    (output/"09_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    ff=pd.DataFrame(cv+transfer)
    re=pd.DataFrame(rediscover)
    rev=pd.DataFrame(reverse)
    cov=pd.DataFrame(coverage)
    pairdf=pd.DataFrame(pairs)
    lines=["# Script 09 — Independent MIDUS physiological discovery and CES-D validation","",
      "## Independent discovery","",
      f"- MIDUS Core M2: {disclose(len(core))} complete participants; MIDUS Refresher MR1: {disclose(len(refresh))} complete participants.",
      "- Three glycemic markers were selected from MIDUS measurements without using NHANES coefficients or CES-D outcomes.",
      "- Core-only outcome-independent PCA; reconstruction; separately rediscovered Refresher PCA; frozen forward and reverse models.",
      "- This is a glycemic-only experiment, not the NHANES seven-variable hematology/glycemia decomposition.","",
      "## Structure and rediscovery","",
      "| PCs | Core explained variance | Refresher explained variance | Independent subspace cosine | Refresher frozen energy retained |","|---|---:|---:|---:|---:|"]
    for k in (1,2,3):
        rc=re.iloc[k-1]
        rec=next(x for x in reconstruction if x["stage"]=="REFRESHER_frozen" and x["k"]==k)
        lines.append(f"| {k} | {float(np.sum(b['ratio'][:k])):.4f} | {float(np.sum(other['ratio'][:k])):.4f} | {rc.min_subspace_cosine:.4f} | {rec['reference_energy_retained']:.4f} |")
    lines += ["", "## CES-D forward inference", "", "Positive ΔMSE favors the augmented specification. Refresher values are frozen Core-model transfer.", "",
              "| Outcome | Refresher covariate MSE | HbA1c ΔMSE | Raw G3 ΔMSE | PC1–2 ΔMSE | PC1–3 ΔMSE |", "|---|---:|---:|---:|---:|---:|"]
    for y in TARGETS:
        s=ff[(ff.stage=="REFRESHER_frozen_transfer")&(ff.outcome==y)].set_index("model")
        lines.append(f"| {y} | {s.loc['covariates','mse']:.4f} | {s.loc['HbA1c','mse_gain_vs_covariates']:+.4f} | {s.loc['raw_G3','mse_gain_vs_covariates']:+.4f} | {s.loc['PC1_2','mse_gain_vs_covariates']:+.4f} | {s.loc['PC1_2_3','mse_gain_vs_covariates']:+.4f} |")
    lines += ["", "## Paired frozen-transfer tests", "",
       "Positive ΔMSE favors the candidate. Intervals are descriptive participant-paired case bootstrap percentiles, not design-based confidence intervals.", "",
       "| Outcome | Comparator → candidate | ΔMSE | Descriptive 95% interval | Result |", "|---|---|---:|---|---|"]
    for _,row in pairdf.iterrows():
        lines.append(f"| {row['outcome']} | {row['comparator']} → {row['candidate']} | {row['mse_gain']:+.4f} | [{row['case_bootstrap_ci_low']:+.4f}, {row['case_bootstrap_ci_high']:+.4f}] | {row['descriptive_direction']} |")
    lines += ["", "## Reverse inference (CES-D → glycemic components)", "", "Positive ΔMSE indicates an improvement over the age/sex-only reverse model.","",
             "| Model | Mean component MSE | Mean ΔMSE versus covariates | Mean 90% marginal interval coverage |", "|---|---:|---:|---:|"]
    for spec in REVERSE:
        rr=rev[rev.reverse_model==spec]
        cc=cov[cov.reverse_model==spec]
        lines.append(f"| {spec} | {rr.REFRESHER_mse.mean():.4f} | {rr.REFRESHER_gain_vs_covariates.mean():+.4f} | {cc.REFRESHER_marginal_coverage.mean():.3f} |")
    lines += ["", "## Interpretation boundaries", "",
        "- The independently rediscovered axes are statistical patterns, not identified biological mechanisms.",
        "- Equal-dimensional G3 and 3PC regression retain the same linear information; rotation alone cannot add information.",
        "- Refresher is an independent cohort, but clinical assay harmonization, depression-scale invariance and attrition remain unaudited.",
        "- The reverse-model 90% widths come from discovery cross-validation residuals; coverage is descriptive, not finite-sample guaranteed.",
        "- Results are exploratory after testing several related outcomes and specifications; paired case-bootstraps do not adjust for sampling, selection or multiple tests.",
        "- These are concurrent biomarker–CES-D associations, not prospective or causal effects.",
        "- Outcome changes, instrument equivalence and the unavailable hematology block require separate experiments.",""]
    (output/"09_readout.md").write_text("\n".join(lines),encoding="utf-8")
    for y in TARGETS:
        s=ff[(ff.stage=="REFRESHER_frozen_transfer")&(ff.outcome==y)].set_index("model")
        print(f"  {y:14s} | N={len(refresh):5d} | frozen PC1–2 ΔMSE={s.loc['PC1_2','mse_gain_vs_covariates']:+.5f}")
    print("SCRIPT 09 | independent MIDUS G3 decomposition + CES-D transfer | complete")
    print(f"OUTPUT: {output/'09_readout.md'}")


def main():
    core, cs = read_cohort("CORE")
    refresher, rs = read_cohort("REFRESHER")
    run(core,refresher,{"CORE":cs,"REFRESHER":rs},OUT)


if __name__=="__main__":
    main()
