#!/usr/bin/env python3
"""Test NHANES simulation hypotheses 84–85 against observed paired MIDJA physiology."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "External_Validation" / "Results"
SCRIPT10 = Path(__file__).resolve().with_name("10_midja_independent_decomposition_cesd.py")
PANELS = ("raw", "adjusted")
SEED = 20261019
FOLDS = 5
REPEATS = 10
ALPHA = 10.0
CAL_FRAC = 0.25
COVERAGE = 0.90
BOOT = 2000
RHOS = (0.0, 0.25, 0.50, 0.75, 0.90)
N_SIM = 300

BRIDGE84 = (
    "age_sex", "CESD2_only", "Z1_only", "Z1_plus_CESD2",
    "Z1_plus_CESD1", "Z1_plus_CESD1_plus_deltaCESD",
    "identity_persistence",
)
BRIDGE85 = ("age_sex", "deltaCESD_only", "Z1_only", "Z1_plus_deltaCESD", "Z1_plus_CESD1", "Z1_plus_CESD1_plus_deltaCESD")


def load_module():
    if not SCRIPT10.is_file():
        raise FileNotFoundError(f"Required Script 10 not found: {SCRIPT10}")
    spec = importlib.util.spec_from_file_location("midja_script10", SCRIPT10)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hashfile(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def conformal_q(errors, coverage=COVERAGE):
    s = np.sort(np.asarray(errors, dtype=float))
    if s.size < 15 or not np.all(np.isfinite(s)):
        raise RuntimeError("Insufficient valid calibration residuals")
    index = min(len(s), math.ceil((len(s) + 1) * coverage)) - 1
    return float(s[index])


def make_features(scope, model, baseline, later, z0, mod):
    age_sex = mod.age_sex(baseline)
    y0 = baseline["CESD_total"].to_numpy(float)
    y1 = later["CESD_total"].to_numpy(float)
    dy = y1 - y0
    if model == "identity_persistence":
        return None
    if model == "age_sex":
        return age_sex
    if scope == "84":
        if model == "CESD2_only":
            return np.column_stack((age_sex, y1))
        if model == "Z1_only":
            return np.column_stack((age_sex, z0))
        if model == "Z1_plus_CESD2":
            return np.column_stack((age_sex, z0, y1))
        if model == "Z1_plus_CESD1":
            return np.column_stack((age_sex, z0, y0))
        if model == "Z1_plus_CESD1_plus_deltaCESD":
            return np.column_stack((age_sex, z0, y0, dy))
    if scope == "85":
        if model == "deltaCESD_only":
            return np.column_stack((age_sex, dy))
        if model == "Z1_only":
            return np.column_stack((age_sex, z0))
        if model == "Z1_plus_deltaCESD":
            return np.column_stack((age_sex, z0, dy))
        if model == "Z1_plus_CESD1":
            return np.column_stack((age_sex, z0, y0))
        if model == "Z1_plus_CESD1_plus_deltaCESD":
            return np.column_stack((age_sex, z0, y0, dy))
    raise KeyError((scope, model))


def ridge_predict(x_train, x_cal, x_test, z_train, z_cal):
    scaler = StandardScaler().fit(x_train)
    xt = scaler.transform(x_train)
    xc = scaler.transform(x_cal)
    xv = scaler.transform(x_test)
    fitted = Ridge(alpha=ALPHA).fit(xt, z_train)
    pred_cal = fitted.predict(xc)
    pred_test = fitted.predict(xv)
    q = np.array([conformal_q(np.abs(z_cal[:, j] - pred_cal[:, j])) for j in range(z_train.shape[1])])
    return pred_test, q


def one_fold(mod, b0, b1, tr_ix, te_ix, split_seed):
    proper_ix, cal_ix = train_test_split(tr_ix, test_size=CAL_FRAC, random_state=split_seed)
    train = b0.iloc[proper_ix]
    basis = mod.basis(mod.x4(train))  # independent discovery, proper-training physiology only
    z = {}
    for role, ix in (("train", proper_ix), ("cal", cal_ix), ("test", te_ix)):
        za = mod.project(mod.x4(b0.iloc[ix]), basis)
        zb = mod.project(mod.x4(b1.iloc[ix]), basis)
        # Normalize by proper-training baseline PC SD to compare all four components.
        scale = np.std(mod.project(mod.x4(train), basis), axis=0)
        if np.any(scale <= 1e-9):
            raise RuntimeError("Degenerate within-training physiological axis")
        z[role] = (za / scale, zb / scale)
    target_test = {"84": z["test"][1], "85": z["test"][1] - z["test"][0]}
    pred = {}
    metric = {}
    for scope, models in (("84", BRIDGE84), ("85", BRIDGE85)):
        target = {role: zz[1] if scope == "84" else zz[1] - zz[0] for role, zz in z.items()}
        for model in models:
            if model == "identity_persistence":
                pred_test = z["test"][0]
                # Absolute residuals for identity require calibration values too.
                cal_residual = np.abs(z["cal"][1] - z["cal"][0])
                q = np.array([conformal_q(cal_residual[:, j]) for j in range(4)])
            else:
                fx = {}
                for role, ix in (("train", proper_ix), ("cal", cal_ix), ("test", te_ix)):
                    fx[role] = make_features(scope, model, b0.iloc[ix], b1.iloc[ix], z[role][0], mod)
                pred_test, q = ridge_predict(fx["train"], fx["cal"], fx["test"], target["train"], target["cal"])
            err = target_test[scope] - pred_test
            metric[(scope, model)] = {
                "sse": np.mean(err**2, axis=1),
                "coverage": np.mean(np.abs(err) <= q[None, :], axis=1),
                "halfwidth": float(np.mean(q)),
            }
            pred[(scope, model)] = pred_test
    return metric, z["test"], basis


def bootstrap_mean(diff, seed):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, float)
    picks = rng.integers(0, len(diff), size=(BOOT, len(diff)))
    return tuple(float(v) for v in np.quantile(diff[picks].mean(axis=1), (0.025, 0.975)))


def evaluate_bridges(mod, b0, b1):
    n = len(b0)
    records = []
    first_errors = {}
    all_models = [(scope, model) for scope, models in (("84", BRIDGE84), ("85", BRIDGE85)) for model in models]
    for rep in range(REPEATS):
        losses = {key: np.full(n, np.nan) for key in all_models}
        covers = {key: np.full(n, np.nan) for key in all_models}
        widths = {key: [] for key in all_models}
        for fold, (tr, te) in enumerate(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED+rep*97).split(b0)):
            metrics, _, _ = one_fold(mod, b0, b1, tr, te, SEED+rep*97+fold*7+500)
            for key in all_models:
                losses[key][te] = metrics[key]["sse"]
                covers[key][te] = metrics[key]["coverage"]
                widths[key].append(metrics[key]["halfwidth"] * len(te))
        if not all(np.all(np.isfinite(x)) for x in losses.values()):
            raise RuntimeError("CV did not evaluate every participant")
        if rep == 0:
            first_errors = {key: value.copy() for key, value in losses.items()}
        for key in all_models:
            scope, model = key
            base = (scope, "Z1_only")
            records.append({"simulation_origin": "NHANES_script_"+scope,
                            "repeat": rep, "model": model, "n": n,
                            "mean_pc_squared_error": float(np.mean(losses[key])),
                            "mean_90pct_marginal_coverage": float(np.mean(covers[key])),
                            "mean_90pct_halfwidth": float(sum(widths[key])/n),
                            "gain_vs_Z1_only": float(np.mean(losses[base] - losses[key]))})
    comp = [
        ("84", "Z1_only", "CESD2_only"),
        ("84", "Z1_only", "Z1_plus_CESD2"),
        ("84", "Z1_plus_CESD1", "Z1_plus_CESD1_plus_deltaCESD"),
        ("84", "identity_persistence", "Z1_only"),
        ("85", "Z1_only", "deltaCESD_only"),
        ("85", "Z1_only", "Z1_plus_deltaCESD"),
        ("85", "Z1_plus_CESD1", "Z1_plus_CESD1_plus_deltaCESD"),
    ]
    differences = []
    for k,(scope, reference, candidate) in enumerate(comp):
        d = first_errors[(scope, reference)]-first_errors[(scope, candidate)]
        lo,hi = bootstrap_mean(d, SEED+800+k)
        differences.append({"simulation_origin":"NHANES_script_"+scope,
                            "reference":reference,"candidate":candidate,
                            "gain_positive_is_better":float(np.mean(d)),
                            "descriptive_case_ci95_low":lo,"descriptive_case_ci95_high":hi,
                            "n":n})
    return pd.DataFrame(records), pd.DataFrame(differences)


def observed_vs_synthetic(mod, b0, b1):
    """Compare the rho grid of NHANES Script 84 to observed 4+ year MIDJA transitions."""
    discovery = mod.basis(mod.x4(b0))
    z0 = mod.project(mod.x4(b0), discovery)
    z1 = mod.project(mod.x4(b1), discovery)
    sd = np.std(z0,axis=0)
    z0,z1 = z0/sd,z1/sd
    dy = b1["CESD_total"].to_numpy(float)-b0["CESD_total"].to_numpy(float)
    rows=[]
    rng=np.random.default_rng(SEED+1200)
    for j in range(4):
        a,c=z0[:,j],z1[:,j]
        rho_observed=float(np.corrcoef(a,c)[0,1])
        change=c-a
        change_mean=float(change.mean())
        change_sd=float(change.std(ddof=0))
        delta_corr=float(np.corrcoef(change,dy)[0,1]) if np.std(dy)>0 else np.nan
        for rho in RHOS:
            correlations=[]
            delta_sds=[]
            shift_means=[]
            for _ in range(N_SIM):
                eps = rng.choice(a, size=len(a), replace=True)
                synthetic = rho*a + np.sqrt(1-rho*rho)*eps
                correlations.append(float(np.corrcoef(a,synthetic)[0,1]))
                delta_sds.append(float(np.std(synthetic-a,ddof=0)))
                shift_means.append(float(np.mean(synthetic-a)))
            rows.append({"component":f"PC{j+1}","n":len(a),
                         "observed_rho":rho_observed,"candidate_sim_rho":rho,
                         "sim_mean_rho":float(np.mean(correlations)),
                         "observed_delta_mean":change_mean,
                         "sim_mean_delta_mean":float(np.mean(shift_means)),
                         "observed_delta_sd":change_sd,
                         "sim_mean_delta_sd":float(np.mean(delta_sds)),
                         "delta_CESD_correlation":delta_corr,
                         "corr_absolute_discrepancy":abs(rho_observed-float(np.mean(correlations))),
                         "delta_sd_absolute_discrepancy":abs(change_sd-float(np.mean(delta_sds)))})
    return pd.DataFrame(rows)


def put_report(panel, basic, results, paired, transition):
    mean = (results.groupby(["simulation_origin","model"],sort=False)
            .agg(mse=("mean_pc_squared_error","mean"),
                 gain=("gain_vs_Z1_only","mean"),
                 coverage=("mean_90pct_marginal_coverage","mean"),
                 halfwidth=("mean_90pct_halfwidth","mean"))
            .reset_index())
    lines=["# Script 12 — MIDJA observed-person challenge of NHANES longitudinal simulations", "",
       "## Matched longitudinal design", "",
       f"- HbA1c panel: {panel}; {basic['n']} strictly chronological MIDJA1→MIDJA2 complete participants.",
       f"- Median follow-up: {basic['months_median']:.1f} months (range {basic['months_min']:.1f}–{basic['months_max']:.1f}).",
       "- Same people, two genuine measurement times; this is not independent-person replication.",
       "- Four markers: Hb, RBC, MCV, HbA1c. No glucose, insulin or RDW; NHANES seven-marker loadings cannot be reused.",
       "- All physiological PCA transforms learned within proper-training participants only and applied unchanged to their calibration and held-out participants.",
       f"- {REPEATS} repeated {FOLDS}-fold person-held-out splits with a {CAL_FRAC:.0%} calibration partition within each training fold.",
       "- Current CES-D and its change are measured at follow-up. Models using them are *same-time physiological inference*, not forecasts made at baseline.", "",
       "## Empirical test of Script 84 — recover current physiology", "",
       "| Model | Standardized mean PC MSE | Gain vs previous Z only | Nominal 90% coverage | Mean half-width |",
       "|---|---:|---:|---:|---:|"]
    for _,r in mean[mean.simulation_origin=="NHANES_script_84"].iterrows():
        lines.append(f"| {r.model} | {r.mse:.4f} | {r.gain:+.4f} | {r.coverage:.3f} | {r.halfwidth:.3f} |")
    lines += ["", "## Empirical test of Script 85 — recover physiological change", "",
       "| Model | Standardized mean ΔPC MSE | Gain vs previous Z only | Nominal 90% coverage | Mean half-width |",
       "|---|---:|---:|---:|---:|"]
    for _,r in mean[mean.simulation_origin=="NHANES_script_85"].iterrows():
        lines.append(f"| {r.model} | {r.mse:.4f} | {r.gain:+.4f} | {r.coverage:.3f} | {r.halfwidth:.3f} |")
    lines += ["", "## Paired uncertainty (first CV repeat)", "",
       "Positive MSE gain favors the candidate. Participant bootstrap intervals describe fixed held-out squared-error differences; they do not refit models.", "",
       "| Scenario | Reference → candidate | ΔMSE | Descriptive 95% interval |",
       "|---|---|---:|---|"]
    for _,r in paired.iterrows():
        lines.append(f"| {r.simulation_origin[-2:]} | {r.reference} → {r.candidate} | {r.gain_positive_is_better:+.4f} | [{r.descriptive_case_ci95_low:+.4f}, {r.descriptive_case_ci95_high:+.4f}] |")
    lines += ["", "## Physiological transition versus NHANES Script 84 simulation", "",
       "These five rho values are the original simulation grid, **not estimates derived from NHANES follow-up**. Synthetic transitions assume no population drift and are generated in the MIDJA baseline component space.", "",
       "| Component | Observed lag correlation | Observed ΔZ mean | Observed ΔZ SD | Best grid rho for lag correlation |",
       "|---|---:|---:|---:|---:|"]
    for comp,g in transition.groupby("component",sort=False):
        ref=g.iloc[0]
        best=g.loc[g.corr_absolute_discrepancy.idxmin()]
        lines.append(f"| {comp} | {ref.observed_rho:+.3f} | {ref.observed_delta_mean:+.3f} | {ref.observed_delta_sd:.3f} | {best.candidate_sim_rho:.2f} |")
    lines += ["", "## Limits", "",
       "- Similar persistence does not imply identical physiological mechanisms or validate a 15-year simulator.",
       "- With two visits per participant, the four Script 53 synthetic worlds (stable, mapping drift, A×G interaction, measurement drift) cannot be individually identified or distinguished reliably.",
       "- Only the Script 84–85 **hypotheses and rho grid** are challenged here, not a replay of the original seven-feature NHANES simulation or its fitted phenotype mapping.",
       "- Underlying Script 84–85 NHANES simulation outcomes differ (PHQ somatic/COGAFF vs MIDJA CES-D); do not compare their raw error magnitudes.",
       "- Selection/attrition, lab comparability, RBC units, CES-D scoring, survey weights and within-person confounding remain unverified.",
       "- Predicted uncertainty is split-calibrated within each outer training fold; exchangeability and population coverage are not guaranteed under selection/shift.",
       "- No causal conclusion follows from predictive gains or empirical lag correlations.", ""]
    (OUT/f"12_{panel}_readout.md").write_text("\n".join(lines),encoding="utf-8")


def run_panel(mod, wave1, wave2, panel):
    first=mod.panel_rows(wave1,panel)
    second=mod.panel_rows(wave2,panel)
    a,b,common,valid=mod.paired_rows(first,second)
    if len(a)<100:
        raise RuntimeError(f"Insufficient matched complete pairs: {len(a)}")
    if not a.index.equals(b.index):
        raise RuntimeError("Pair order mismatch")
    months=b["MonthIndex"].to_numpy(float)-a["MonthIndex"].to_numpy(float)
    if not np.all(months>0):
        raise RuntimeError("Chronological check failed")
    info={"n":len(a),"matched_complete":common,"dated_pairs":valid,
          "months_median":float(np.median(months)),
          "months_min":float(np.min(months)),"months_max":float(np.max(months)),
          "CESD_change_mean":float((b["CESD_total"]-a["CESD_total"]).mean())}
    cv, contrasts=evaluate_bridges(mod,a,b)
    temporal=observed_vs_synthetic(mod,a,b)
    cv.to_csv(OUT/f"12_{panel}_repeat_metrics.csv",index=False)
    contrasts.to_csv(OUT/f"12_{panel}_paired_comparisons.csv",index=False)
    temporal.to_csv(OUT/f"12_{panel}_transition_grid.csv",index=False)
    put_report(panel,info,cv,contrasts,temporal)
    print(f"  {panel:8s} | matched N={len(a)} | median lag={np.median(months):.1f} months | "
          f"Script84 joint ΔMSE={cv[(cv.simulation_origin=='NHANES_script_84') & (cv.model=='Z1_plus_CESD2')].gain_vs_Z1_only.mean():+.4f} | "
          f"Script85 joint ΔMSE={cv[(cv.simulation_origin=='NHANES_script_85') & (cv.model=='Z1_plus_deltaCESD')].gain_vs_Z1_only.mean():+.4f}")
    return info


def main():
    mod=load_module()
    first,src1,_=mod.read_wave("MIDJA1")
    second,src2,_=mod.read_wave("MIDJA2")
    OUT.mkdir(parents=True,exist_ok=True)
    info={panel:run_panel(mod,first,second,panel) for panel in PANELS}
    manifest={"script":Path(__file__).name,
       "tested_NHANES_simulations":["84_skip_informed_reverse_bridge_simulation","85_delta_state_informed_bridge_simulation"],
       "additional_simulation":"84 rho grid adapted to MIDJA baseline measurements",
       "not_identified":["53 stable/additive versus mapping drift versus A×G interaction versus assay drift", "15-year trajectories"],
       "seed":SEED,"folds":FOLDS,"repeats":REPEATS,"alpha":ALPHA,"calibration_fraction":CAL_FRAC,
       "coverage_nominal":COVERAGE,"rho_grid":RHOS,"sim_replicates_per_rho":N_SIM,
       "source_script10":{"file":SCRIPT10.name,"sha256":hashfile(SCRIPT10)},
       "source_files":{"MIDJA1":src1,"MIDJA2":src2},"panels":info,
       "nhanes_or_midus_fitted_parameters_used":False,"uses_followup_CESD_for_same_time_state_inference":True,
       "causal_identification":False,"external_independent_longitudinal_replication":False,
       "individual_level_outputs_exported":False}
    (OUT/"12_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("SCRIPT 12 | empirical challenge of NHANES simulations 84–85 | complete")
    print(f"OUTPUT: {OUT/'12_raw_readout.md'}")
    print(f"        {OUT/'12_adjusted_readout.md'}")


if __name__=="__main__":
    main()
