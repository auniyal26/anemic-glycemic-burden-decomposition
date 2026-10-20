#!/usr/bin/env python3
"""MIDJA wave-specific A/G discovery, reconstruction, CES-D mapping, and temporal audits."""
from __future__ import annotations

import hashlib
import json
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
BOOT = 1000
H = ("Hb", "RBC", "MCV")
FEATURES = (*H, "HbA1c")
TARGETS_W1 = ("CESD_total", "CESD_somatic", "CESD_affect")
SPECS = ("covariates", "HbA1c", "hematology_raw", "raw_X4", "PC1",
         "PC1_2", "PC1_2_3", "PC1_2_3_4", "A_PC1_2_plus_G", "A_PC1_2_3_plus_G")
REVERSE = ("covariates", "CESD_total")
PROSPECTIVE = ("prior_CESD", "prior_CESD_HbA1c", "prior_CESD_raw_X4",
               "prior_CESD_PC1_2", "prior_CESD_PC1_2_3_4")
SMALL = "SUPPRESSED_0_TO_10"
FILES = {
    "MIDJA1": r"^MIDJA_1_Biomarker_.*\.sav$",
    "MIDJA2": r"^MIDJA 2_Biomarker_.*\.sav$",
}
FIELDS = {
    "MIDJA1": {
        "Hb": "J2CHGB", "RBC": "J2CRBC", "MCV": "J2CMCV",
        "HbA1c_raw": "J2CHBA1C", "HbA1c_adjusted": "J2BHA1CA",
        "CESD_total": "J2QCESD", "CESD_somatic": "J2QCESDSC",
        "CESD_affect": "J2QCESDDA", "Age": "J2CAGE", "Sex": "J1SQ1",
        "Year": "J2CYEAR", "Month": "J2CMONTH",
    },
    "MIDJA2": {
        "Hb": "K2CHGB", "RBC": "K2CRBC", "MCV": "K2CMCV",
        "HbA1c_raw": "K2CHBA1C", "HbA1c_adjusted": "K2BHA1CA",
        "CESD_total": "K2QCESD", "Age": "K2CAGE", "Sex": "K2CSEX",
        "Year": "K2CYEAR", "Month": "K2CMONTH",
    },
}


def disclose(n):
    return str(int(n)) if int(n) > 10 else SMALL


def finite(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def find_file(wave):
    if not RAW.is_dir():
        raise FileNotFoundError(f"Source directory not found: {RAW}")
    pat = re.compile(FILES[wave], re.I)
    matches = [p for p in RAW.rglob("*") if p.is_file() and pat.fullmatch(p.name)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {wave} biomarker file, found: {[p.name for p in matches]}")
    return matches[0]


def canonical_id(x):
    if pd.isna(x):
        raise ValueError("Missing participant identifier")
    s = str(x).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", s):
        s = s.split(".", 1)[0]
    if not s:
        raise ValueError("Blank participant identifier")
    return s


def read_wave(wave):
    import pyreadstat

    path = find_file(wave)
    _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
    lookup = {c.upper(): c for c in meta.column_names}
    required = ["MIDJA_IDS", *FIELDS[wave].values()]
    absent = [field for field in required if field.upper() not in lookup]
    if absent:
        raise RuntimeError(f"{wave}: required variables absent: {absent}; no substitution attempted")
    cols = list(dict.fromkeys(lookup[c.upper()] for c in required))
    d, _ = pyreadstat.read_sav(str(path), usecols=cols, user_missing=False,
                                apply_value_formats=False)
    d.index = pd.Index([canonical_id(x) for x in d[lookup["MIDJA_IDS"]]], name="participant")
    if d.index.has_duplicates:
        raise RuntimeError(f"{wave}: duplicate participant IDs")
    d = d.drop(columns=lookup["MIDJA_IDS"])
    d = d.rename(columns={lookup[v.upper()]: k for k, v in FIELDS[wave].items()})
    for key in FIELDS[wave]:
        d[key] = finite(d[key])
    return d, {"filename": path.name, "sha256": digest(path)}, set(d.index)


def panel_rows(wave, panel):
    d = wave.copy()
    chosen = "HbA1c_raw" if panel == "raw" else "HbA1c_adjusted"
    d["HbA1c"] = d[chosen]
    ok = (d["Hb"].between(3, 25) & d["RBC"].between(150, 850)
          & d["MCV"].between(40, 140) & d["HbA1c"].between(2.5, 20)
          & d["Age"].between(18, 110) & d["Sex"].isin([1, 2])
          & d["CESD_total"].between(0, 60))
    d = d.loc[ok].copy()
    if len(d) < 80:
        raise RuntimeError(f"Insufficient complete rows for {panel}: n={len(d)}")
    for y in ("CESD_somatic", "CESD_affect"):
        if y in d:
            d[y] = d[y].where(d[y].between(0, 60))
    dates_ok = (d["Year"].between(1990, 2030)
                & d["Month"].between(1, 12)
                & d["Year"].eq(np.floor(d["Year"]))
                & d["Month"].eq(np.floor(d["Month"])))
    d["MonthIndex"] = (d["Year"] * 12 + d["Month"]).where(dates_ok)
    return d


def x4(d):
    return d.loc[:, FEATURES].to_numpy(dtype=float)


def age_sex(d):
    return np.column_stack((d["Age"].to_numpy(float), (d["Sex"] == 2).to_numpy(float)))


def basis(x):
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    if np.any(sd < 1e-10):
        raise RuntimeError("Degenerate physiological marker")
    p = PCA(n_components=x.shape[1], svd_solver="full").fit((x-mu)/sd)
    v = p.components_.T.copy()
    for j in range(v.shape[1]):
        i = int(np.argmax(np.abs(v[:, j])))
        if v[i, j] < 0:
            v[:, j] *= -1
    return {"mu": mu, "sd": sd, "v": v, "ratio": p.explained_variance_ratio_}


def fit_bases(d):
    return basis(x4(d)), basis(d.loc[:, H].to_numpy(float))


def project(x, b):
    return ((x-b["mu"])/b["sd"]) @ b["v"]


def reconstruct(z, b, k):
    return (z[:, :k] @ b["v"][:, :k].T) * b["sd"] + b["mu"]


def feature(d, b, a, spec):
    cv = age_sex(d)
    if spec == "covariates":
        return cv
    if spec == "HbA1c":
        return np.column_stack((cv, d["HbA1c"].to_numpy(float)))
    if spec == "hematology_raw":
        return np.column_stack((cv, d.loc[:, H].to_numpy(float)))
    if spec == "raw_X4":
        return np.column_stack((cv, x4(d)))
    if spec.startswith("A_PC"):
        k = 2 if spec == "A_PC1_2_plus_G" else 3
        az = project(d.loc[:, H].to_numpy(float), a)
        return np.column_stack((cv, az[:, :k], d["HbA1c"].to_numpy(float)))
    k = {"PC1":1, "PC1_2":2, "PC1_2_3":3, "PC1_2_3_4":4}[spec]
    return np.column_stack((cv, project(x4(d), b)[:, :k]))


def scaled_ridge(x, y):
    s = StandardScaler().fit(x)
    return s, Ridge(alpha=ALPHA).fit(s.transform(x), y)


def predict(f, x):
    return f[1].predict(f[0].transform(x))


def mse(y, p):
    return float(np.mean((y-p)**2))


def r2(y, p):
    den = np.sum((y-np.mean(y))**2)
    return float(1-np.sum((y-p)**2)/den) if den > 0 else np.nan


def perf(stage, yname, spec, y, p, baseline):
    return {"stage":stage, "outcome":yname, "model":spec, "n":disclose(len(y)),
            "mse":mse(y,p), "mae":float(np.mean(np.abs(y-p))), "r2":r2(y,p),
            "mse_gain_vs_covariates":baseline-mse(y,p)}


def ci_gain(y, reference, candidate, seed):
    dif = (y-reference)**2 - (y-candidate)**2
    rng = np.random.default_rng(seed)
    boot = np.empty(BOOT)
    for k in range(BOOT):
        boot[k] = dif[rng.integers(len(dif), size=len(dif))].mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(np.mean(dif)), float(lo), float(hi)


def make_cv(d, yname):
    d = d[d[yname].notna()]
    if len(d) < 70:
        return []
    truth = d[yname].to_numpy(float)
    pred = {spec:np.full(len(d),np.nan) for spec in SPECS}
    for it, iv in KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(d):
        train, valid = d.iloc[it], d.iloc[iv]
        b, a = fit_bases(train)
        for spec in SPECS:
            f = scaled_ridge(feature(train,b,a,spec), train[yname].to_numpy(float))
            pred[spec][iv] = predict(f, feature(valid,b,a,spec))
    base = mse(truth,pred["covariates"])
    return [perf("MIDJA1_5fold_CV",yname,spec,truth,pred[spec],base) for spec in SPECS]


def reconstruction_audit(first, second, first_b):
    rows = []
    for stage, d in (("MIDJA1_discovery",first),("MIDJA2_frozen",second)):
        xx = x4(d)
        z = project(xx,first_b)
        energy = np.mean(((xx-first_b["mu"])/first_b["sd"])**2)
        for k in range(1,5):
            xhat = reconstruct(z,first_b,k)
            e = (xx-xhat)/first_b["sd"]
            rows.append({"stage":stage,"n":disclose(len(d)),"k":k,
                "standardized_rmse":float(np.sqrt(np.mean(e**2))),
                "reference_energy_retained":float(1-np.mean(e**2)/energy),
                "max_abs_reconstruction_error":float(np.max(np.abs(xx-xhat))) if k == 4 else np.nan})
    if max(r["max_abs_reconstruction_error"] for r in rows if r["k"]==4)>1e-8:
        raise RuntimeError("Exact four-component round-trip sanity check failed")
    return rows


def rediscovery_audit(first_b, second_b):
    out=[]
    for k in range(1,5):
        similarity=abs(float(first_b["v"][:,k-1]@second_b["v"][:,k-1]))
        s=np.linalg.svd(first_b["v"][:,:k].T@second_b["v"][:,:k],compute_uv=False)
        out.append({"k":k,"abs_matched_pc_cosine":similarity,
                    "min_subspace_cosine":float(s.min()),
                    "MIDJA1_cumulative_variance":float(sum(first_b["ratio"][:k])),
                    "MIDJA2_cumulative_variance":float(sum(second_b["ratio"][:k]))})
    return out


def wave2_groups(second, all_first_ids):
    return {"MIDJA2_all":second,
            "MIDJA2_no_wave1_participation":second.loc[~second.index.isin(all_first_ids)],
            "MIDJA2_wave1_participants":second.loc[second.index.isin(all_first_ids)]}


def forward_transfer(first, second, b, a, all_first_ids):
    rows, pairs = [], []
    fits = {s:scaled_ridge(feature(first,b,a,s),first["CESD_total"].to_numpy(float)) for s in SPECS}
    for stage, d in wave2_groups(second,all_first_ids).items():
        if len(d) < 20:
            continue
        y=d["CESD_total"].to_numpy(float)
        preds={s:predict(fits[s],feature(d,b,a,s)) for s in SPECS}
        base=mse(y,preds["covariates"])
        for spec in SPECS:
            rows.append(perf(stage,"CESD_total",spec,y,preds[spec],base))
        for idx,(reference,candidate) in enumerate((("covariates","PC1_2"),
                          ("HbA1c","PC1_2"),("covariates","raw_X4"),
                          ("hematology_raw","A_PC1_2_plus_G"))):
            value,lo,hi=ci_gain(y,preds[reference],preds[candidate],SEED+idx+len(stage)*100)
            pairs.append({"stage":stage,"outcome":"CESD_total","reference":reference,
                          "candidate":candidate,"n":disclose(len(d)),"mse_gain":value,
                          "case_bootstrap_ci_low":lo,"case_bootstrap_ci_high":hi,
                          "direction":"favors_candidate" if lo>0 else "favors_reference" if hi<0 else "inconclusive"})
    return rows,pairs


def reverse_features(d, spec):
    cv=age_sex(d)
    return cv if spec=="covariates" else np.column_stack((cv,d["CESD_total"].to_numpy(float)))


def reverse_audit(first, second, b, all_first_ids):
    z1=project(x4(first),b)
    split=KFold(n_splits=FOLDS,shuffle=True,random_state=SEED+1)
    rows, cov = [], []
    for spec in REVERSE:
        xf=reverse_features(first,spec)
        oof=np.full(z1.shape,np.nan)
        for it,iv in split.split(first):
            model=scaled_ridge(xf[it],z1[it])
            oof[iv]=predict(model,xf[iv])
        q=np.quantile(np.abs(z1-oof),0.9,axis=0)
        model=scaled_ridge(xf,z1)
        for stage,d in wave2_groups(second,all_first_ids).items():
            if len(d)<20:
                continue
            true_z=project(x4(d),b)
            estimated=predict(model,reverse_features(d,spec))
            for j in range(4):
                rows.append({"stage":stage,"model":spec,"component":f"PC{j+1}",
                    "n":disclose(len(d)),"MIDJA1_oof_mse":mse(z1[:,j],oof[:,j]),
                    "MIDJA2_component_mse":mse(true_z[:,j],estimated[:,j])})
                cov.append({"stage":stage,"model":spec,"component":f"PC{j+1}",
                    "n":disclose(len(d)),"discovery_cv_q90_abs_error":float(q[j]),
                    "marginal_coverage":float(np.mean(np.abs(true_z[:,j]-estimated[:,j])<=q[j]))})
            xpred=reconstruct(estimated,b,4)
            normalized=(x4(d)-xpred)/b["sd"]
            rows.append({"stage":stage,"model":spec,"component":"X4_reconstructed_from_Y",
                "n":disclose(len(d)),"MIDJA1_oof_mse":np.nan,
                "MIDJA2_component_mse":float(np.mean(normalized**2))})
    tab=pd.DataFrame(rows)
    for i,row in tab.iterrows():
        baseline=tab[(tab.stage==row.stage)&(tab.model=="covariates")&(tab.component==row.component)]
        tab.loc[i,"gain_vs_covariates"]= (float(baseline.iloc[0].MIDJA2_component_mse)
                                             - float(row.MIDJA2_component_mse))
    return tab.to_dict("records"), cov


def paired_rows(first, second):
    ids=first.index.intersection(second.index)
    f=first.loc[ids]
    s=second.loc[ids]
    valid=f["MonthIndex"].notna()&s["MonthIndex"].notna()&(s["MonthIndex"]>f["MonthIndex"])
    return f.loc[valid],s.loc[valid],len(ids),int(valid.sum())


def prospective_features(d, b, a, spec):
    cv=np.column_stack((age_sex(d),d["CESD_total"].to_numpy(float)))
    if spec=="prior_CESD":
        return cv
    if spec=="prior_CESD_HbA1c":
        return np.column_stack((cv,d["HbA1c"].to_numpy(float)))
    if spec=="prior_CESD_raw_X4":
        return np.column_stack((cv,x4(d)))
    k=2 if spec=="prior_CESD_PC1_2" else 4
    return np.column_stack((cv,project(x4(d),b)[:,:k]))


def prospective_audit(first, second):
    f,s,total,dated=paired_rows(first,second)
    if len(f)<70:
        return [],[],[],{"matched_complete":disclose(total),"strictly_later_dated":disclose(dated),
                          "status":"insufficient_strictly_ordered_pairs"}
    y=s["CESD_total"].to_numpy(float)
    pred={spec:np.full(len(f),np.nan) for spec in PROSPECTIVE}
    for it,iv in KFold(n_splits=FOLDS,shuffle=True,random_state=SEED+2).split(f):
        train,valid=f.iloc[it],f.iloc[iv]
        b,a=fit_bases(train)
        for spec in PROSPECTIVE:
            fit=scaled_ridge(prospective_features(train,b,a,spec),y[it])
            pred[spec][iv]=predict(fit,prospective_features(valid,b,a,spec))
    base=mse(y,pred["prior_CESD"])
    rows=[perf("MIDJA1_to_MIDJA2_5fold_person_CV","followup_CESD_total",spec,
               y,pred[spec],base) for spec in PROSPECTIVE]
    pairs=[]
    for idx,spec in enumerate(PROSPECTIVE[1:]):
        value,lo,hi=ci_gain(y,pred["prior_CESD"],pred[spec],SEED+500+idx)
        pairs.append({"model":spec,"n":disclose(len(f)),"gain_vs_prior_CESD":value,
                      "case_bootstrap_ci_low":lo,"case_bootstrap_ci_high":hi,
                      "direction":"favors_candidate" if lo>0 else "favors_reference" if hi<0 else "inconclusive"})
    learned,_=fit_bases(f)
    dz=project(x4(s),learned)-project(x4(f),learned)
    dy=y-f["CESD_total"].to_numpy(float)
    changes=[]
    for j in range(4):
        valid=np.std(dz[:,j])>1e-10 and np.std(dy)>1e-10
        changes.append({"component":f"PC{j+1}","n":disclose(len(f)),
                        "mean_change_z":float(np.mean(dz[:,j])),
                        "delta_z_delta_cesd_correlation":float(np.corrcoef(dz[:,j],dy)[0,1]) if valid else np.nan})
    return rows,pairs,changes,{"matched_complete":disclose(total),"strictly_later_dated":disclose(dated),
                                 "status":"exploratory_observational"}


def report(panel, first, second, all_first_ids, b, other, recon, rediscover, ff, pairs, rev, coverage,
           prospect, ppairs, changes, chronological, sources):
    rdat=pd.DataFrame(rediscover)
    fdat=pd.DataFrame(ff)
    pdat=pd.DataFrame(pairs)
    vdat=pd.DataFrame(rev)
    cdat=pd.DataFrame(coverage)
    trans=fdat[fdat.stage=="MIDJA2_all"].set_index("model")
    lines=["# Script 10 — Independent MIDJA A/G decomposition and CES-D audits", "",
      "## Study design", "",
      f"- Panel: Hb, RBC, MCV and {'raw' if panel=='raw' else 'MIDUS-adjusted'} HbA1c.",
      f"- MIDJA1 complete N={disclose(len(first))}; MIDJA2 complete N={disclose(len(second))}.",
      f"- MIDJA2 with no MIDJA1 biomarker record: {disclose(sum(~second.index.isin(all_first_ids)))}; repeated MIDJA1 participants: {disclose(sum(second.index.isin(all_first_ids)))}.",
      "- Components are learned without NHANES/MIDUS loadings or CES-D supervision.",
      "- MIDJA waves contain overlapping participants; wave-to-wave structure is not independent-person replication.",
      "- Biomarker and CES-D measurement comparability, attrition, and assay harmonization are not established.","",
      "## Statistical structure", "",
      "| PCs | MIDJA1 cumulative variance | MIDJA2 cumulative variance | Subspace cosine | MIDJA2 frozen reconstruction energy |",
      "|---|---:|---:|---:|---:|"]
    for i in range(4):
        k=i+1
        rec=next(row for row in recon if row["stage"]=="MIDJA2_frozen" and row["k"]==k)
        lines.append(f"| {k} | {sum(b['ratio'][:k]):.4f} | {sum(other['ratio'][:k]):.4f} | {rdat.loc[i,'min_subspace_cosine']:.4f} | {rec['reference_energy_retained']:.4f} |")
    lines.extend(["","## CES-D forward transfer (MIDJA1 fitted, MIDJA2 evaluated)","",
      "Positive ΔMSE favors biomarkers added to age and sex.","",
      "| Model | MIDJA2 all ΔMSE | New-participant ΔMSE | Repeated-participant ΔMSE |",
      "|---|---:|---:|---:|"])
    for spec in SPECS:
        vals=[]
        for stage in ("MIDJA2_all","MIDJA2_no_wave1_participation","MIDJA2_wave1_participants"):
            df=fdat[(fdat.stage==stage)&(fdat.model==spec)]
            vals.append(f"{float(df.iloc[0].mse_gain_vs_covariates):+.4f}" if len(df) else "NA")
        lines.append(f"| {spec} | {' | '.join(vals)} |")
    lines.extend(["", "## Paired transfer comparisons", "",
      "Bootstrap intervals resample participants within each evaluated group; not survey-adjusted or multiplicity-adjusted.", "",
      "| Group | Comparator → candidate | ΔMSE | Descriptive 95% interval | Interpretation |",
      "|---|---|---:|---|---|"])
    for _,row in pdat.iterrows():
        lines.append(f"| {row.stage} | {row.reference} → {row.candidate} | {row.mse_gain:+.4f} | [{row.case_bootstrap_ci_low:+.4f}, {row.case_bootstrap_ci_high:+.4f}] | {row.direction} |")
    lines.extend(["", "## CES-D → Z → physiological reconstruction", "",
      "Predicting physiology from CES-D is an uncertain inverse problem, not an exact inverse.","",
      "| MIDJA2 group | Model | Average PC MSE | Gain vs age/sex | Average nominal 90% marginal coverage |",
      "|---|---|---:|---:|---:|"])
    for stage in ("MIDJA2_all","MIDJA2_no_wave1_participation","MIDJA2_wave1_participants"):
        for spec in REVERSE:
            d=vdat[(vdat.stage==stage)&(vdat.model==spec)&(vdat.component!="X4_reconstructed_from_Y")]
            c=cdat[(cdat.stage==stage)&(cdat.model==spec)]
            if len(d):
                lines.append(f"| {stage} | {spec} | {d.MIDJA2_component_mse.mean():.4f} | {d.gain_vs_covariates.mean():+.4f} | {c.marginal_coverage.mean():.3f} |")
    lines.extend(["", "## Within-person chronological audit", "",
        f"- Matched complete cases: {chronological['matched_complete']}; verified later follow-up: {chronological['strictly_later_dated']}.",
        f"- Status: {chronological['status']}."])
    if prospect:
        lines.extend(["", "| Baseline → later CES-D model | Participant-level five-fold CV ΔMSE vs prior CES-D + age/sex |",
                      "|---|---:|"])
        for row in prospect:
            lines.append(f"| {row['model']} | {row['mse_gain_vs_covariates']:+.4f} |")
        lines.append("- This is person-held-out *cross-validation on matched participants*, not external prospective validation. Baseline CES-D is included in every model.")
    lines.extend(["", "## Interpretation limits", "",
       "- Components are statistical axes; physiological interpretation needs independent testing.",
       "- A full-rank four-PC representation is a rotation of four standardized biomarkers, not an information gain.",
       "- HbA1c is the only glycemic measure here; full hematology–glycemia shared/private identification is limited.",
       "- RBC values have source-specific numerical scale; scaling and laboratory units must be verified from source documentation.",
       "- HbA1c adjusted versus raw is a sensitivity audit, not evidence that the assays are interchangeable.",
       "- CES-D scoring and invariance, laboratory protocols, sampling weights and attrition are unaudited.",
       "- The repeated-subject temporal analysis does not establish causality or eliminate unmeasured confounding.",
       "- Exact numerical reconstruction at full rank checks algebra, not physiological validity.",
       "- Interpret changes and cross-wave associations as exploratory and not proof of causality.", ""])
    return "\n".join(lines)


def run(wave1, wave2, all_first_ids, sources, output):
    output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    meta={"experiment":"Independent MIDJA hematology–HbA1c/CES-D framework validation",
          "script":"10_midja_independent_decomposition_cesd.py", "seed":SEED,
          "folds":FOLDS,"ridge_alpha":ALPHA,"bootstrap_reps":BOOT,
          "inputs":sources,"NHANES_or_MIDUS_fit_used":False,
          "causal_identification":False,"laboratory_harmonization_verified":False,
          "CESD_invariance_verified":False,"sampling_weights_or_attrition_adjusted":False,
          "participant_level_records_exported":False,"panels":{}}
    for panel in ("raw","adjusted"):
        first,second=panel_rows(wave1,panel),panel_rows(wave2,panel)
        b,a=fit_bases(first)
        other,other_a=fit_bases(second)
        recon=reconstruction_audit(first,second,b)
        rediscover=rediscovery_audit(b,other)
        forward_cv=[]
        for y in TARGETS_W1:
            if y in first:
                forward_cv.extend(make_cv(first,y))
        forward,paired=forward_transfer(first,second,b,a,all_first_ids)
        reverse,coverage=reverse_audit(first,second,b,all_first_ids)
        prospective,pro_pairs,changes,chronological=prospective_audit(first,second)
        stem=f"10_{panel}"
        for name,rows in (("reconstruction",recon),("rediscovery",rediscover),
                          ("forward_cesd",forward_cv+forward),("forward_paired_ci",paired),
                          ("reverse_inference",reverse),("reverse_coverage",coverage),
                          ("prospective_cesd",prospective),("prospective_paired_ci",pro_pairs),
                          ("longitudinal_component_changes",changes)):
            pd.DataFrame(rows).to_csv(output/f"{stem}_{name}.csv",index=False)
        (output/f"{stem}_readout.md").write_text(report(panel,first,second,all_first_ids,b,other,
          recon,rediscover,forward_cv+forward,paired,reverse,coverage,prospective,pro_pairs,changes,chronological,sources),
          encoding="utf-8")
        meta["panels"][panel]={"MIDJA1_n":disclose(len(first)),"MIDJA2_n":disclose(len(second)),
            "MIDJA2_no_wave1_record_n":disclose(sum(~second.index.isin(all_first_ids))),
            "MIDJA2_wave1_record_n":disclose(sum(second.index.isin(all_first_ids))),
            "chronological_pairs":chronological,
            "MIDJA1_mean":b["mu"].tolist(),"MIDJA1_sd":b["sd"].tolist(),
            "MIDJA1_loadings":b["v"].tolist(),"MIDJA1_variance_ratios":b["ratio"].tolist(),
            "MIDJA2_independent_loadings":other["v"].tolist(),
            "MIDJA2_variance_ratios":other["ratio"].tolist(),
            "MIDJA1_H3_loadings":a["v"].tolist(),"MIDJA2_H3_loadings":other_a["v"].tolist()}
        r=pd.DataFrame(forward)
        score=r[(r.stage=="MIDJA2_all")&(r.model=="PC1_2")]
        val=float(score.iloc[0].mse_gain_vs_covariates)
        summaries.append((panel,len(first),len(second),val,chronological["strictly_later_dated"]))
    (output/"10_manifest.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    for panel,n1,n2,delta,nlong in summaries:
        print(f"  {panel:8s} | MIDJA1 n={n1:3d}, MIDJA2 n={n2:3d} | PC1–2 CES-D ΔMSE={delta:+.5f} | chronological pairs={nlong}")
    print("SCRIPT 10 | independent MIDJA hematology/HbA1c + CES-D audits | complete")
    print(f"OUTPUT: {output/'10_raw_readout.md'}")
    print(f"        {output/'10_adjusted_readout.md'}")


def main():
    first,s1,all_first=read_wave("MIDJA1")
    second,s2,_=read_wave("MIDJA2")
    run(first,second,all_first,{"MIDJA1":s1,"MIDJA2":s2},OUT)


if __name__=="__main__":
    main()
