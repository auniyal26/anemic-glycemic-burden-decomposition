#!/usr/bin/env python3
"""Script 99 | NHANES temporal stability and lean/full adjustment audit.

Run from the repository root:
    python EDA/99_temporal_coefficient_stability.py

Requires the unchanged Script-98 implementation and its 98_model_selection_cv.csv,
98_frozen_holdout_performance.csv, 98_manifest.json, and the real Script-73
NHANES source files used by Script 98.

Protocol
--------
- Retain discovery/frozen Script-72/73 X->Z representation.
- Keep Script-98 best hyperparameters for each outcome x covariate-scope x design,
  as selected by 2005-08 CV only. Do NOT use later outcomes to choose models.
- Fit discovery equations using only 2005-08.
- Apply them without refitting to each NHANES cycle (2009-18 test and
  2021-23 validation), plus training-cycle diagnostics.
- Separately refit coefficients within each cycle, for descriptive drift ONLY.
  Refit models are never treated as validation predictions or model choices.
- Compare lean/full adjustments on exactly matched observations.

Later NHANES cycles are separate people, not participant-level longitudinal data.
This audit produces association and drift diagnostics, not causal inference.
Full complex survey design standard errors are NOT estimated.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPT98 = HERE / "98_regularized_phenotype_equation.py"
RESULTS = HERE / "Results"
SEED = 99
CYCLE_ORDER = ["0506", "0708", "0910", "1112", "1314", "1516", "1718", "2123"]
TRAIN = ("0506", "0708")
TEST = ("0910", "1112", "1314", "1516", "1718")
VALIDATION = ("2123",)
DESIGNS = ("covariates", "Z_main", "X_main", "X_AxG", "Z_pairwise")
SCOPES = ("lean", "full")
OUTPUTS = [
    "99_cycle_performance.csv",
    "99_scope_comparison.csv",
    "99_coefficient_drift.csv",
    "99_drift_summary.csv",
    "99_cycle_refit_diagnostics.csv",
    "99_manifest.json",
    "99_readout.md",
]


def load_98_module(path=SCRIPT98):
    if not path.is_file():
        raise FileNotFoundError(f"Script 98 not found alongside Script 99: {path}")
    spec = importlib.util.spec_from_file_location("script98_local_frozen", path)
    mod = importlib.util.module_from_spec(spec)
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev
    return mod


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wmse(y, yh, w):
    return float(np.average((np.asarray(y) - np.asarray(yh))**2, weights=np.asarray(w)))


def weighted_calibration(y, prediction, w):
    y = np.asarray(y, float)
    p = np.asarray(prediction, float)
    w = np.asarray(w, float)
    my, mp = np.average(y, weights=w), np.average(p, weights=w)
    den = np.sum(w * (p-mp)**2)
    if den < 1e-12:
        return float("nan"), float("nan")
    slope = float(np.sum(w * (p-mp) * (y-my)) / den)
    return float(my - slope*mp), slope


def choose_98_winners(cv):
    cols = {"outcome", "scope", "design", "method", "alpha", "l1_ratio", "cv_mean_weighted_MSE", "n_features"}
    missing = sorted(cols - set(cv.columns))
    if missing:
        raise ValueError(f"Script 98 CV output missing columns: {missing}")
    best = (cv.sort_values(["cv_mean_weighted_MSE", "n_features", "method"], kind="stable")
              .drop_duplicates(["outcome", "scope", "design"]))
    winners = {}
    for rec in best.to_dict(orient="records"):
        k = (str(rec["outcome"]), str(rec["scope"]), str(rec["design"]))
        winners[k] = rec
    return winners


def block(feature, mod):
    return mod.add_block(feature)


def model_fit_fixed_scale(mod, base_scaler, matrix, y, w, selection, n_discovery):
    # Same discovery feature scaling and regularization parameter in ALL cycles.
    # Ridge sklearn penalty is sum-based; equalize total weights to discovery n.
    transformed = base_scaler.transform(matrix)
    model = mod.instantiate(selection["method"], float(selection["alpha"]),
                            None if pd.isna(selection["l1_ratio"]) else float(selection["l1_ratio"]))
    w = np.asarray(w, float)
    weight = (w / w.sum()) * n_discovery
    model.fit(transformed, y, sample_weight=weight)
    return model


def stratum_group(cycle):
    if cycle in TRAIN:
        return "discovery_diagnostic"
    if cycle in TEST:
        return "test_2009-2018"
    if cycle in VALIDATION:
        return "validate_2021-2023"
    raise ValueError(cycle)


def write_table(df):
    if df.empty:
        return "(No observations)"
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        return "```csv\n" + df.to_csv(index=False) + "\n```"


def audit(mod, d, F, selections, old_perf, old_manifest, min_rows=100, check_98=True):
    expected = old_manifest["matched_cohort"]
    actual = d.groupby("SPLIT").size().to_dict()
    if check_98 and actual != expected:
        raise AssertionError(f"Script-98 matched cohort changed: old={expected}, now={actual}")
    train = d.SPLIT.eq("train_2005-2008").to_numpy()
    cycles = d.CYCLE.astype(str).to_numpy()
    w_all = d.SURVEY_WT.to_numpy(float)
    train_n = int(train.sum())
    if train_n < 200:
        raise ValueError("Insufficient training observations")
    data_by_cycle = {c: np.flatnonzero(cycles == c) for c in CYCLE_ORDER}
    if any(len(idx) < min_rows for idx in data_by_cycle.values()):
        raise ValueError("Cycle-specific observed cohort too small: " + str({c:len(idx) for c,idx in data_by_cycle.items()}))
    data_by_period = {
        "train_2005-2008": np.flatnonzero(train),
        "test_2009-2018": np.flatnonzero(d.SPLIT.eq("test_2009-2018")),
        "validate_2021-2023": np.flatnonzero(d.SPLIT.eq("validate_2021-2023")),
    }
    specs = {(s, des): f for s, des, f in mod.specifications()}
    out_perf, out_coef, out_refit = [], [], []
    deviations = []
    for target, outcol in mod.Y_TARGETS.items():
        y_all = d[outcol].to_numpy(float)
        ytrain = y_all[train]
        sy = float(np.sqrt(np.average((ytrain-np.average(ytrain,weights=w_all[train]))**2, weights=w_all[train])))
        if sy <= 1e-8:
            raise ValueError("Invalid discovery outcome SD")
        for scope in SCOPES:
            for design in DESIGNS:
                key = (target, scope, design)
                if key not in selections:
                    raise ValueError("No Script-98 CV result for " + repr(key))
                selection = selections[key]
                features = specs[(scope, design)]
                matrix = F[features].to_numpy(float)
                alpha = float(selection["alpha"])
                ratio = None if pd.isna(selection["l1_ratio"]) else float(selection["l1_ratio"])
                scaler, model = mod.fit(matrix[train],ytrain,w_all[train], selection["method"], alpha, ratio)
                pred_all = mod.prediction(scaler,model,matrix)
                beta_discovery = np.asarray(model.coef_, float)
                model_name = {"outcome":target,"scope":scope,"design":design,"method":selection["method"],
                              "alpha":alpha,"l1_ratio":ratio}
                # Confirm original Script-98 pooled frozen predictions BEFORE analysis.
                if check_98:
                    prior = old_perf.loc[(old_perf.outcome == target)&(old_perf.scope == scope)&
                        (old_perf.design == design)]
                    for pname, ii in data_by_period.items():
                        recorded = prior.loc[prior.period.eq(pname)]
                        if len(recorded) != 1:
                            raise AssertionError(f"Missing/duplicate 98 reference {key}, {pname}")
                        r2 = mod.wmetrics(y_all[ii], pred_all[ii], w_all[ii])[0]
                        err = abs(float(recorded.iloc[0].weighted_R2) - r2)
                        if err > 1e-5:
                            raise AssertionError(f"Script 98 mismatch {key} {pname} delta R2={err:.6g}")
                        deviations.append(err)
                for cycle, ii in data_by_cycle.items():
                    yy, ww, pp = y_all[ii], w_all[ii], pred_all[ii]
                    r2,rmse,mae,bias = mod.wmetrics(yy,pp,ww)
                    intercept,slope = weighted_calibration(yy,pp,ww)
                    out_perf.append({**model_name,"cycle":cycle,"status":stratum_group(cycle),"n":len(ii),
                        "R2":r2,"RMSE":rmse,"MAE":mae,"mean_prediction_bias":bias,
                        "MSE":wmse(yy,pp,ww),"calibration_intercept":intercept,"calibration_slope":slope,
                        "prediction_frozen_2005_2008":True,"selected_by":"training_CV_only"})
                    # Later-cycle coefficients are re-estimated solely to DESCRIBE coefficient drift.
                    # These are not out-of-time predictions, nor are they used to change 98 model.
                    diag = model_fit_fixed_scale(mod,scaler,matrix[ii],yy,ww,selection,train_n)
                    beta_cycle = np.asarray(diag.coef_,float)
                    naive_in_sample_r2 = mod.wmetrics(yy,mod.prediction(scaler,diag,matrix[ii]),ww)[0]
                    out_refit.append({**model_name,"cycle":cycle,"status":stratum_group(cycle),"n":len(ii),
                        "descriptive_refit_in_sample_R2":naive_in_sample_r2,
                        "frozen_out_of_cycle_R2":r2,
                        "refit_model_is_not_a_holdout_predictor":True})
                    for j, name in enumerate(features):
                        b0 = float(beta_discovery[j])/sy
                        b1 = float(beta_cycle[j])/sy
                        mag_threshold = 0.02
                        out_coef.append({**model_name,"cycle":cycle,"status":stratum_group(cycle),
                            "feature":name,"block":block(name,mod),
                            "beta_discovery_divided_by_Y_SD":b0,
                            "beta_cycle_refit_divided_by_Y_SD":b1,
                            "signed_shift":b1-b0,"absolute_shift":abs(b1-b0),
                            "meaningful_sign_flip":bool(abs(b0) >= mag_threshold and abs(b1) >= mag_threshold and np.sign(b0)!=np.sign(b1)),
                            "nonzero_discovery":bool(abs(b0) >= mag_threshold),
                            "nonzero_cycle":bool(abs(b1) >= mag_threshold),
                            "refits_descriptive_only":True})
                print(f"  {target:9s} {scope:4s} {design:12s}: "
                      f"frozen holdout 2009–18 R2={mod.wmetrics(y_all[data_by_period['test_2009-2018']],pred_all[data_by_period['test_2009-2018']],w_all[data_by_period['test_2009-2018']])[0]:+.4f}")
    perf = pd.DataFrame(out_perf)
    coef = pd.DataFrame(out_coef)
    refit = pd.DataFrame(out_refit)
    # Compare lean/full on exactly identical subjects, for each design/cycle/target.
    compare = []
    for (target,design,cyc), group in perf.groupby(["outcome","design","cycle"], sort=False):
        if set(group.scope) != {"lean", "full"}:
            raise AssertionError("Missing lean/full matched comparison")
        lean = group.loc[group.scope.eq("lean")].iloc[0]
        full = group.loc[group.scope.eq("full")].iloc[0]
        if int(lean.n) != int(full.n):
            raise AssertionError("Lean/full participant mismatch")
        compare.append(dict(outcome=target,design=design,cycle=cyc,status=lean.status,n=int(lean.n),
                            lean_R2=float(lean.R2),full_R2=float(full.R2),
                            full_minus_lean_R2=float(full.R2-lean.R2),
                            lean_MSE=float(lean.MSE),full_MSE=float(full.MSE),
                            full_minus_lean_MSE=float(full.MSE-lean.MSE),
                            lean_bias=float(lean.mean_prediction_bias),full_bias=float(full.mean_prediction_bias)))
    cmp = pd.DataFrame(compare)
    # Per-cycle shifts are descriptive; use models selected entirely by training CV.
    diagrows=[]
    for (target,scope,design,cyc,blk), group in coef.groupby(["outcome","scope","design","cycle","block"], sort=False):
        diagrows.append(dict(outcome=target,scope=scope,design=design,cycle=cyc,status=group.status.iloc[0],
                             block=blk,n_features=len(group),
                             rms_beta_shift=float(np.sqrt(np.mean(group.signed_shift.to_numpy(float)**2))),
                             max_abs_beta_shift=float(group.absolute_shift.max()),
                             meaningful_sign_flips=int(group.meaningful_sign_flip.sum()),
                             active_discovery_features=int(group.nonzero_discovery.sum()),
                             active_cycle_features=int(group.nonzero_cycle.sum())))
    summary=pd.DataFrame(diagrows)
    return perf, cmp, coef, summary, refit, max(deviations,default=0.)


def make_readout(perf, scope, drift, refits, old_manifest, maxdiff):
    main = perf[(perf.design=="Z_main") & perf.cycle.isin(TEST+VALIDATION)]
    zcomp=scope[(scope.design=="Z_main") & scope.cycle.isin(TEST+VALIDATION)]
    compact = main[["outcome","scope","cycle","n","R2","calibration_slope","mean_prediction_bias"]].copy()
    display_cycles = {"0506":"2005-06", "0708":"2007-08", "0910":"2009-10", "1112":"2011-12", "1314":"2013-14", "1516":"2015-16", "1718":"2017-18", "2123":"2021-23"}
    compact["cycle"] = compact["cycle"].map(display_cycles)
    zcomp = zcomp.copy()
    zcomp["cycle"] = zcomp["cycle"].map(display_cycles)
    blocks=drift[(drift.design=="Z_main") & drift.cycle.isin(TEST+VALIDATION) &
                 drift.block.isin(["A_private","G_private","Shared","Discordant"])].copy()
    blocks=blocks.sort_values("rms_beta_shift",ascending=False)
    blocks["cycle"] = blocks["cycle"].map(display_cycles)
    s=["# Script 99 — Temporal coefficient stability audit","",
       "Source: genuine frozen Script-73 NHANES, unchanged Script-98 CV configurations.",
       "2005–08 training; 2009–18 frozen test; 2021–23 frozen validation.",
       "Cycle-level refits below are diagnostic comparisons only, never test/validation predictions.",
       "", "## Frozen Z-main regression by cycle", "",write_table(compact),"",
       "## Full adjustment minus lean adjustment (Z-main; same observations)","",
       write_table(zcomp[["outcome","cycle","n","lean_R2","full_R2","full_minus_lean_R2"]]),"",
       "## Largest physiological-block coefficient shifts (Z-main refits)","",
       write_table(blocks[["outcome","scope","cycle","block","n_features","rms_beta_shift","meaningful_sign_flips"]].head(24)),"",
       "## Interpretation guardrails","",
       "- Every original model and regularization strength was selected on 2005–08 only.",
       "- Refit coefficients on later cycles are **descriptive** and use the original discovery feature scaling; their in-sample R² is not a generalization estimate.",
       "- Coefficients are divided by the discovery outcome SD; standardized differences are comparable within the same outcome and feature.",
       "- Sign flips are screened only when both magnitudes exceed 0.02 discovery-outcome SD; no significance test is implied.",
       "- Performance and coefficient drift may reflect sampling variability, phenotype changes, measurement issues, cohort composition or model misspecification.",
       "- Additional adjustment variables must not be interpreted as causes or reliable generalizable predictors solely from CV.",
       "- Later NHANES participants are different people, not longitudinal trajectories.",
       "- All scores are survey-weighted point metrics; no NHANES complex-design uncertainty or bootstrap CIs.",
       f"- Script-98 pooled frozen predictions numerically reproduced (maximum R² difference: {maxdiff:.4g}).",
       ""]
    return "\n".join(s)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-cycle-rows",type=int,default=100)
    parser.add_argument("--outdir",type=Path,default=RESULTS)
    args=parser.parse_args()
    if args.min_cycle_rows<50:
        raise ValueError("Require at least 50 cycle-level observations")
    mod=load_98_module()
    # Resolve relative filenames under actual repository, avoid cwd dependence.
    raw_cv=RESULTS/"98_model_selection_cv.csv"
    raw_perf=RESULTS/"98_frozen_holdout_performance.csv"
    raw_manifest=RESULTS/"98_manifest.json"
    for f in (raw_cv,raw_perf,raw_manifest):
        if not f.is_file():
            raise FileNotFoundError(f"Missing Script 98 result: {f}. Run 98 first")
    tracked=[SCRIPT98,raw_cv,raw_perf,raw_manifest]
    before={p.name:sha256(p) for p in tracked}
    cv=pd.read_csv(raw_cv)
    perf98=pd.read_csv(raw_perf)
    manifest98=json.loads(raw_manifest.read_text(encoding="utf-8"))
    winners=choose_98_winners(cv)
    mod.yield_msg=[]
    mod.normalizer={}
    mod.categories_reference={}
    mod.categories_feature_names=[]
    d=mod.load_real()
    F=mod.build_features(d)
    mod.categories_feature_names=[c for c in F.columns if "=" in c]
    print("SCRIPT 99 | FROZEN TEMPORAL STABILITY")
    print("Matched observations:",d.groupby("SPLIT").size().to_dict())
    per,cmp,coefs,drift,refit,maxdiff=audit(mod,d,F,winners,perf98,manifest98,min_rows=args.min_cycle_rows)
    out=args.outdir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    per.to_csv(out/OUTPUTS[0],index=False)
    cmp.to_csv(out/OUTPUTS[1],index=False)
    coefs.to_csv(out/OUTPUTS[2],index=False)
    drift.to_csv(out/OUTPUTS[3],index=False)
    refit.to_csv(out/OUTPUTS[4],index=False)
    md=make_readout(per,cmp,drift,refit,manifest98,maxdiff)
    (out/OUTPUTS[6]).write_text(md,encoding="utf-8")
    manifest={"script":Path(__file__).name,"discovery_cycles":list(TRAIN),"test_cycles":list(TEST),
              "validation_cycles":list(VALIDATION),"selection":"Script 98 original training-only CV winners",
              "inputs_used":[str(p) for p in tracked],"input_sha256":before,
              "full_covariates":mod.FULL_COV,"lean_covariates":mod.CLEAN_COV,
              "matched_cohorts":d.groupby("SPLIT").size().to_dict(),
              "models_audited":len(winners),"n_frozen_cycle_metrics":len(per),
              "n_refit_coefficients":len(coefs),"max_abs_98_R2_reproduction_error":maxdiff,
              "refits_used_to_select_predictors":False,"frozen_models_modified":False,
              "causal":False,"within_subject_longitudinal":False,"complex_survey_inference":False,
              "outputs":OUTPUTS}
    (out/OUTPUTS[5]).write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    after={p.name:sha256(p) for p in tracked}
    if before!=after:
        raise AssertionError("Script 98 or input result was changed during Script 99")
    print("\nSUMMARY — Z_main lean/full comparison (2021–23):")
    print(cmp.loc[(cmp.design=="Z_main") & (cmp.cycle=="2123"),
                  ["outcome","lean_R2","full_R2","full_minus_lean_R2"]].to_string(index=False))
    print("\nDone. Read EDA/Results/99_readout.md; frozen 72/73/98 files unchanged.")


if __name__=="__main__":
    main()
