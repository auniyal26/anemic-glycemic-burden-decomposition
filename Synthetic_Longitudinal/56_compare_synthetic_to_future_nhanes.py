from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "Synthetic_Longitudinal"
SIM_RESULTS = SIM_DIR / "Results"
DATA_DISC = ROOT / "Data" / "NHANES"
DATA_TEMP = ROOT / "Data" / "NHANES_Transfer"
PARAMETERS = ROOT / "Parameters"
SIM_RESULTS.mkdir(parents=True, exist_ok=True)

SIM_FILE = SIM_DIR / "53_synthetic_longitudinal_observed.parquet"
SIM_MANIFEST = SIM_DIR / "53_synthetic_longitudinal_manifest.json"
A_PARAM = PARAMETERS / "46_outcome_independent_A_transform.json"
G_PARAM = PARAMETERS / "46_outcome_independent_G3_transform.json"

OUT_DETAIL = SIM_RESULTS / "56_realism_by_cycle.csv"
OUT_SUMMARY = SIM_RESULTS / "56_realism_summary.csv"
OUT_FLOW = SIM_RESULTS / "56_realism_actual_flow.csv"
OUT_MANIFEST = SIM_DIR / "56_realism_check_manifest.json"

# 2021-23 is deliberately NOT touched here.
CYCLE_MAP = {
    "0708": {"suffix": "E", "sim_year": 2,  "base": "disc"},
    "0910": {"suffix": "F", "sim_year": 4,  "base": "temp"},
    "1112": {"suffix": "G", "sim_year": 6,  "base": "temp"},
    "1314": {"suffix": "H", "sim_year": 8,  "base": "temp"},
    "1516": {"suffix": "I", "sim_year": 10, "base": "temp"},
    "1718": {"suffix": "J", "sim_year": 12, "base": "temp"},
}

A_VARS = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"]
RAW_G = ["LBXGH", "LBXGLU", "LOG_IN"]
COMPARE_RAW = [
    "LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW",
    "LBXGH", "LBXGLU", "LOG_IN", "BMXBMI", "SOMATIC_SCORE",
]

AGE_BINS = [20, 30, 40, 50, 60, 70, 80, 200]
AGE_LABELS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

print()
print("SYNTHETIC LONGITUDINAL — NHANES REALITY CHECK")
print("==============================================")
print("No simulator refitting or tuning occurs here.")
print("Synthetic years are compared with age/sex-poststratified repeated-cross-sectional NHANES.")
print("2021-23 remains untouched as a later stress test.")
print("Hidden synthetic truth is not read.")
print()

for p in [SIM_FILE, SIM_MANIFEST, A_PARAM, G_PARAM]:
    if not p.exists():
        raise FileNotFoundError(f"Missing prerequisite: {p}")

m53 = json.loads(SIM_MANIFEST.read_text(encoding="utf-8"))
sim_sha = hashlib.sha256(SIM_FILE.read_bytes()).hexdigest()
if sim_sha != m53["observed_sha256"]:
    raise RuntimeError("Synthetic observed SHA256 does not match Script 53 manifest.")

sim = pd.read_parquet(SIM_FILE)

if any(str(c).startswith("TRUE_") for c in sim.columns):
    raise RuntimeError("Leakage guard failed: hidden-truth columns found in observed synthetic data.")

def xpt_path(base: Path, stem: str) -> Path:
    for name in [f"{stem}.XPT", f"{stem}.xpt"]:
        p = base / name
        if p.exists():
            return p
    matches = list(base.glob(f"{stem}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve {stem} under {base}")

def cycle_base(cycle):
    return (DATA_DISC if CYCLE_MAP[cycle]["base"] == "disc" else DATA_TEMP) / cycle

def clean_phq(dpq):
    phq = [f"DPQ0{i}0" for i in range(1, 10)]
    needed = ["SEQN"] + phq
    missing = [c for c in needed if c not in dpq.columns]
    if missing:
        raise ValueError(f"DPQ missing {missing}")
    d = dpq[needed].copy()
    for c in phq:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[d[c].abs() < 1e-10, c] = 0
        d.loc[~d[c].isin([0, 1, 2, 3]), c] = np.nan
    d["SOMATIC_SCORE"] = d[["DPQ030", "DPQ040", "DPQ050"]].sum(axis=1, min_count=3)
    return d[["SEQN", "SOMATIC_SCORE"]]

def insulin_file(base, cycle, suffix):
    stem = f"GLU_{suffix}" if cycle in {"0708", "0910", "1112"} else f"INS_{suffix}"
    return xpt_path(base, stem)

def load_transform(path):
    x = json.loads(path.read_text(encoding="utf-8"))
    vars_ = x["variables"]
    mu = np.array([x["standardization"]["mean"][v] for v in vars_], dtype=float)
    sd = np.array([x["standardization"]["scale"][v] for v in vars_], dtype=float)
    L = np.array(x["pca"]["loadings_columns_are_components"], dtype=float)
    k = int(x["pca"]["retained_components"])
    return vars_, mu, sd, L, k

A_T_VARS, A_MU, A_SD, A_L, A_K = load_transform(A_PARAM)
G_T_VARS, G_MU, G_SD, G_L, G_K = load_transform(G_PARAM)

def add_frozen_scores(df):
    out = df.copy()
    ZA = (out[A_T_VARS].to_numpy(float) - A_MU) / A_SD
    ZG = (out[G_T_VARS].to_numpy(float) - G_MU) / G_SD
    SA = ZA @ A_L[:, :A_K]
    SG = ZG @ G_L[:, :G_K]
    for j in range(A_K):
        out[f"A_PC{j+1}"] = SA[:, j]
    for j in range(G_K):
        out[f"G_PC{j+1}"] = SG[:, j]
    return out

def build_actual_cycle(cycle):
    cfg = CYCLE_MAP[cycle]
    s = cfg["suffix"]
    base = cycle_base(cycle)

    demo = pd.read_sas(xpt_path(base, f"DEMO_{s}"), format="xport")
    cbc = pd.read_sas(xpt_path(base, f"CBC_{s}"), format="xport")
    ghb = pd.read_sas(xpt_path(base, f"GHB_{s}"), format="xport")
    glu = pd.read_sas(xpt_path(base, f"GLU_{s}"), format="xport")
    ins = pd.read_sas(insulin_file(base, cycle, s), format="xport")
    dpq = pd.read_sas(xpt_path(base, f"DPQ_{s}"), format="xport")
    bmx = pd.read_sas(xpt_path(base, f"BMX_{s}"), format="xport")

    demo_cols = ["SEQN", "RIDAGEYR", "RIAGENDR", "SDMVPSU", "SDMVSTRA"]
    if "RIDEXPRG" in demo.columns:
        demo_cols.append("RIDEXPRG")

    needed_cbc = ["SEQN"] + A_VARS
    missing = [c for c in needed_cbc if c not in cbc.columns]
    if missing:
        raise ValueError(f"{cycle} CBC missing {missing}")

    for name, frame, cols in [
        ("GHB", ghb, ["SEQN", "LBXGH"]),
        ("GLU", glu, ["SEQN", "LBXGLU", "WTSAF2YR"]),
        ("INS", ins, ["SEQN", "LBXIN"]),
        ("BMX", bmx, ["SEQN", "BMXBMI"]),
    ]:
        miss = [c for c in cols if c not in frame.columns]
        if miss:
            raise ValueError(f"{cycle} {name} missing {miss}")

    d = demo[demo_cols].copy()
    d = d.merge(glu[["SEQN", "LBXGLU", "WTSAF2YR"]], on="SEQN", how="inner", validate="one_to_one")
    d = d.merge(ghb[["SEQN", "LBXGH"]], on="SEQN", how="inner", validate="one_to_one")
    d = d.merge(ins[["SEQN", "LBXIN"]], on="SEQN", how="inner", validate="one_to_one")
    d = d.merge(cbc[needed_cbc], on="SEQN", how="inner", validate="one_to_one")
    d = d.merge(clean_phq(dpq), on="SEQN", how="inner", validate="one_to_one")
    d = d.merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="inner", validate="one_to_one")

    d["RIDAGEYR"] = pd.to_numeric(d["RIDAGEYR"], errors="coerce")
    d["WTSAF2YR"] = pd.to_numeric(d["WTSAF2YR"], errors="coerce")
    d["LBXIN"] = pd.to_numeric(d["LBXIN"], errors="coerce")
    d.loc[d["LBXIN"] <= 0, "LBXIN"] = np.nan
    d["LOG_IN"] = np.log(d["LBXIN"])

    age_ok = d["RIDAGEYR"] >= 20
    preg_ok = pd.Series(True, index=d.index)
    if "RIDEXPRG" in d.columns:
        preg_ok = d["RIDEXPRG"] != 1
    d = d[age_ok & preg_ok & (d["WTSAF2YR"] > 0)].copy()

    required = A_VARS + ["LBXGH", "LBXGLU", "LOG_IN", "BMXBMI", "SOMATIC_SCORE", "RIAGENDR"]
    d = d.dropna(subset=required).copy()

    d["AGE_BIN"] = pd.cut(
        d["RIDAGEYR"].clip(upper=199),
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
        include_lowest=True,
    )

    d["CYCLE"] = cycle
    d = add_frozen_scores(d)
    return d

# Add frozen scores and age bins to synthetic observed data.
sim = add_frozen_scores(sim)
sim["AGE_BIN"] = pd.cut(
    pd.to_numeric(sim["SIM_AGE"], errors="coerce").clip(upper=199),
    bins=AGE_BINS,
    labels=AGE_LABELS,
    right=False,
    include_lowest=True,
)

COMPARE = COMPARE_RAW + [f"A_PC{i}" for i in range(1, A_K + 1)] + [f"G_PC{i}" for i in range(1, G_K + 1)]

# Baseline scale is synthetic year 0, stable world. Used only to normalize discrepancies.
base0 = sim[(sim["SCENARIO"] == "stable_additive") & (sim["SIM_YEAR"] == 0)].copy()
baseline_sd = {c: float(pd.to_numeric(base0[c], errors="coerce").std(ddof=0)) for c in COMPARE}
for c, s in baseline_sd.items():
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError(f"Invalid baseline SD for {c}")

def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))

def weighted_sd(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    mu = np.sum(x[ok] * w[ok]) / np.sum(w[ok])
    return float(np.sqrt(np.sum(w[ok] * (x[ok] - mu) ** 2) / np.sum(w[ok])))

def poststrat_weights(actual, synthetic):
    # Target distribution is the synthetic cohort's age-sex composition at the mapped year.
    target = (
        synthetic.groupby(["AGE_BIN", "RIAGENDR"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    target["target_share"] = target["n"] / target["n"].sum()

    a = actual.copy()
    a["_base_w"] = pd.to_numeric(a["WTSAF2YR"], errors="coerce")
    cells = (
        a.groupby(["AGE_BIN", "RIAGENDR"], observed=True)["_base_w"]
        .sum()
        .rename("actual_w")
        .reset_index()
    )
    cells["actual_share"] = cells["actual_w"] / cells["actual_w"].sum()

    factors = target.merge(cells, on=["AGE_BIN", "RIAGENDR"], how="inner")
    factors["factor"] = factors["target_share"] / factors["actual_share"]

    a = a.merge(
        factors[["AGE_BIN", "RIAGENDR", "factor"]],
        on=["AGE_BIN", "RIAGENDR"],
        how="inner",
        validate="many_to_one",
    )
    a["POST_WT"] = a["_base_w"] * a["factor"]

    covered = float(factors["target_share"].sum())
    return a, covered

actual_frames = {}
flow_rows = []
for cycle in CYCLE_MAP:
    ac = build_actual_cycle(cycle)
    actual_frames[cycle] = ac
    flow_rows.append({
        "cycle": cycle,
        "sim_year": CYCLE_MAP[cycle]["sim_year"],
        "complete_joint_actual_n": len(ac),
        "psu_n": int(ac["SDMVPSU"].nunique()),
        "strata_n": int(ac["SDMVSTRA"].nunique()),
    })

pd.DataFrame(flow_rows).to_csv(OUT_FLOW, index=False)

detail_rows = []

for scenario in sorted(sim["SCENARIO"].unique()):
    for cycle, cfg in CYCLE_MAP.items():
        year = cfg["sim_year"]
        sy = sim[(sim["SCENARIO"] == scenario) & (sim["SIM_YEAR"] == year)].copy()
        ac = actual_frames[cycle]
        acw, coverage = poststrat_weights(ac, sy)

        for col in COMPARE:
            sx = pd.to_numeric(sy[col], errors="coerce").to_numpy(float)
            ax = pd.to_numeric(acw[col], errors="coerce").to_numpy(float)
            aw = acw["POST_WT"].to_numpy(float)

            smean = float(np.nanmean(sx))
            ssd = float(np.nanstd(sx, ddof=0))
            amean = weighted_mean(ax, aw)
            asd = weighted_sd(ax, aw)

            ok_a = np.isfinite(ax) & np.isfinite(aw) & (aw > 0)
            ok_s = np.isfinite(sx)
            wd = wasserstein_distance(
                sx[ok_s],
                ax[ok_a],
                u_weights=np.ones(ok_s.sum()),
                v_weights=aw[ok_a],
            )

            scale = baseline_sd[col]
            detail_rows.append({
                "scenario": scenario,
                "cycle": cycle,
                "sim_year": year,
                "variable": col,
                "synthetic_mean": smean,
                "actual_poststrat_mean": amean,
                "synthetic_sd": ssd,
                "actual_poststrat_sd": asd,
                "standardized_mean_difference": (smean - amean) / scale,
                "sd_ratio_synth_over_actual": ssd / asd if asd > 0 else np.nan,
                "standardized_wasserstein": wd / scale,
                "age_sex_target_coverage": coverage,
                "actual_n": len(acw),
                "synthetic_n": len(sy),
            })

detail = pd.DataFrame(detail_rows)
detail.to_csv(OUT_DETAIL, index=False)

summary = (
    detail.groupby(["scenario", "cycle", "sim_year"], as_index=False)
    .agg(
        median_abs_smd=("standardized_mean_difference", lambda x: float(np.median(np.abs(x)))),
        max_abs_smd=("standardized_mean_difference", lambda x: float(np.max(np.abs(x)))),
        median_std_wasserstein=("standardized_wasserstein", "median"),
        max_std_wasserstein=("standardized_wasserstein", "max"),
        age_sex_target_coverage=("age_sex_target_coverage", "min"),
    )
)
summary.to_csv(OUT_SUMMARY, index=False)

print("Reality-check summary (descriptive only; not for simulator tuning):")
for scenario in sorted(summary["scenario"].unique()):
    q = summary[summary["scenario"] == scenario]
    print(
        f"  {scenario}: "
        f"median |SMD|={q['median_abs_smd'].median():.3f}, "
        f"median std-WD={q['median_std_wasserstein'].median():.3f}, "
        f"worst |SMD|={q['max_abs_smd'].max():.3f}"
    )

manifest = {
    "script": "56_compare_synthetic_to_future_nhanes.py",
    "synthetic_input_sha256": sim_sha,
    "hidden_truth_read": False,
    "simulator_refit_or_tuned": False,
    "actual_cycles_read": list(CYCLE_MAP.keys()),
    "cycle_to_sim_year": {k: v["sim_year"] for k, v in CYCLE_MAP.items()},
    "nhanes_2021_2023_read": False,
    "comparison_sample": (
        "Adults age>=20, known pregnancy excluded, positive fasting weight, "
        "joint complete A raw + G3 raw + BMI + somatic score."
    ),
    "poststratification": "Actual NHANES reweighted to each synthetic year's age-sex composition.",
    "important_caution": (
        "NHANES is repeated cross-sectional, not longitudinal. These comparisons assess "
        "population-level plausibility only. Do not interpret them as validation of true "
        "individual trajectories."
    ),
    "future_use_rule": (
        "Do not tune Scripts 52-53 against these results without explicitly reclassifying "
        "2007-18 as development data. 2021-23 remains untouched."
    ),
    "detail_output": str(OUT_DETAIL.relative_to(ROOT)),
    "summary_output": str(OUT_SUMMARY.relative_to(ROOT)),
    "flow_output": str(OUT_FLOW.relative_to(ROOT)),
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print()
print(f"PASS detail: {OUT_DETAIL}")
print(f"PASS summary: {OUT_SUMMARY}")
print(f"PASS flow: {OUT_FLOW}")
print(f"PASS manifest: {OUT_MANIFEST}")
print("PASS 2021-23 untouched.")
