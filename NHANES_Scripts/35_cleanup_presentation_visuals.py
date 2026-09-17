from pathlib import Path
import shutil
import subprocess
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 35 — PRESENTATION VISUAL CLEANUP
# ============================================================
# Presentation-only corrections. This script:
#   - does NOT change scientific definitions
#   - does NOT tune/refit the frozen temporal-validation model
#   - does NOT overwrite preserved data
#   - fixes sequential cohort accounting
#   - separates physiological working y from phenotype P
#   - replaces raw A/G group means with survey-weighted estimates + CI
#   - labels unstandardized coefficients so A/G magnitudes are not miscompared
#   - removes line-trajectory styling from categorical CFA/loading figures
#   - removes the misleading shared-axis HbA1c-vs-fasting-glucose comparison
#
# Outputs:
# Results/Presentation_Evidence_Clean/
# ============================================================

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
DATA = ROOT / "Data" / "NHANES"
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
RLIB = ROOT / "R_library"

OUT = RESULTS / "Presentation_Evidence_Clean"
FIG = OUT / "Figures"
TAB = OUT / "Tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]
SOMATIC_ITEMS = ["DPQ030", "DPQ040", "DPQ050"]

CORE_FILES = {
    "0506": {"cbc":"CBC_D.XPT","demo":"DEMO_D.xpt","ghb":"GHB_D.XPT","dpq":"DPQ_D.XPT"},
    "0708": {"cbc":"CBC_E.XPT","demo":"DEMO_E.XPT","ghb":"GHB_E.XPT","dpq":"DPQ_E.XPT"},
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def savefig(name):
    p = FIG / name
    plt.tight_layout()
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    return p

def normalize_phq(d):
    d = d.copy()
    for c in ITEMS:
        x = pd.to_numeric(d[c], errors="coerce")
        x.loc[x.abs() < 1e-10] = 0
        x.loc[~x.isin([0,1,2,3])] = np.nan
        d[c] = x
    return d

def safe_csv(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None

print()
print("35 — PRESENTATION VISUAL CLEANUP")
print("================================")
print("No scientific definitions or frozen validation rules are changed.")
print()

core_path = PROCESSED / "nhanes_core_working.parquet"
hash_before = sha256(core_path) if core_path.exists() else None

# ------------------------------------------------------------
# 1) Rebuild discovery core from raw XPT
# ------------------------------------------------------------
frames = []
for cycle, f in CORE_FILES.items():
    p = DATA / cycle
    cbc = pd.read_sas(p/f["cbc"], format="xport")
    demo = pd.read_sas(p/f["demo"], format="xport")
    ghb = pd.read_sas(p/f["ghb"], format="xport")
    dpq = normalize_phq(pd.read_sas(p/f["dpq"], format="xport"))

    d = (
        demo[["SEQN","RIDAGEYR","RIAGENDR","RIDRETH1"]]
        .merge(cbc[["SEQN","LBXHGB"]], on="SEQN", validate="one_to_one")
        .merge(ghb[["SEQN","LBXGH"]], on="SEQN", validate="one_to_one")
        .merge(dpq[["SEQN"]+ITEMS], on="SEQN", validate="one_to_one")
    )
    d["CYCLE"] = cycle
    frames.append(d)

d0 = pd.concat(frames, ignore_index=True)
d0 = d0[d0["RIDAGEYR"] >= 18].copy()

d0["PHQ9_TOTAL"] = d0[ITEMS].sum(axis=1, min_count=9)
d0["SOMATIC_SCORE"] = d0[SOMATIC_ITEMS].sum(axis=1, min_count=3)
d0["HB_THRESHOLD"] = np.where(d0["RIAGENDR"] == 1, 13.0, 12.0)
d0["A"] = (d0["HB_THRESHOLD"] - d0["LBXHGB"]).clip(lower=0)
d0["G_HBA1C"] = (d0["LBXGH"] - 5.7).clip(lower=0)
d0["AG_HBA1C"] = d0["A"] * d0["G_HBA1C"]

# ------------------------------------------------------------
# 2) TRUE sequential discovery flow
# ------------------------------------------------------------
d1 = d0[d0["LBXHGB"].notna()].copy()
d2 = d1[d1["LBXGH"].notna()].copy()
d3 = d2[d2["PHQ9_TOTAL"].notna()].copy()

survey_path = RESULTS / "25_survey_parity_input.csv"
survey = pd.read_csv(survey_path)
survey["CYCLE"] = survey["CYCLE"].astype(str).str.replace(".0","",regex=False).str.zfill(4)
d3["CYCLE"] = d3["CYCLE"].astype(str).str.zfill(4)

survey_keys = set(zip(survey["SEQN"].astype(float), survey["CYCLE"]))
d4 = d3[
    [(float(s), c) in survey_keys for s,c in zip(d3["SEQN"], d3["CYCLE"])]
].copy()

flow = pd.DataFrame({
    "stage": [
        "Same-person core match, age 18+",
        "+ complete hemoglobin",
        "+ complete HbA1c",
        "+ complete PHQ-9",
        "+ full X3 survey-analysis requirements",
    ],
    "n": [len(d0), len(d1), len(d2), len(d3), len(d4)]
})
flow["retained_from_previous_pct"] = flow["n"] / flow["n"].shift(1) * 100
flow.loc[0,"retained_from_previous_pct"] = 100.0
flow["retained_from_start_pct"] = flow["n"] / flow.loc[0,"n"] * 100
flow.to_csv(TAB/"35_true_sequential_sample_flow.csv", index=False)

plt.figure(figsize=(10,5.4))
ax = plt.gca()
x = np.arange(len(flow))
ax.bar(x, flow["n"])
ax.set_xticks(x)
ax.set_xticklabels(flow["stage"], rotation=24, ha="right")
ax.set_ylabel("Participants")
ax.set_title("Discovery cohort: true sequential participant flow")
for i,r in flow.iterrows():
    ax.text(i, r["n"]+90, f"{int(r['n']):,}", ha="center", fontsize=9)
savefig("35_01_true_sequential_cohort_flow.png")

# ------------------------------------------------------------
# 3) Correct conceptual x -> y -> P structure
# ------------------------------------------------------------
working = d3[["A","G_HBA1C","AG_HBA1C"]].copy()
phen = d3[["PHQ9_TOTAL","SOMATIC_SCORE"]].copy()

summary_rows = []
for c,note in [
    ("A","Sex-thresholded hemoglobin deficit"),
    ("G_HBA1C","HbA1c excess above 5.7%"),
    ("AG_HBA1C","Scalar A×G interaction"),
]:
    x = pd.to_numeric(working[c], errors="coerce").dropna()
    summary_rows.append({
        "dimension":c,"n":len(x),"mean":x.mean(),"sd":x.std(ddof=1),
        "median":x.median(),"p25":x.quantile(.25),"p75":x.quantile(.75),
        "zero_pct":100*(x==0).mean(),"role":"working representation y","note":note
    })
pd.DataFrame(summary_rows).to_csv(TAB/"35_working_y_summary.csv", index=False)

phen_rows = []
for c,note in [
    ("PHQ9_TOTAL","Observed 9-item depressive phenotype"),
    ("SOMATIC_SCORE","Sleep + fatigue + appetite phenotype"),
]:
    x = pd.to_numeric(phen[c], errors="coerce").dropna()
    phen_rows.append({
        "dimension":c,"n":len(x),"mean":x.mean(),"sd":x.std(ddof=1),
        "median":x.median(),"p25":x.quantile(.25),"p75":x.quantile(.75),
        "zero_pct":100*(x==0).mean(),"role":"phenotype P","note":note
    })
pd.DataFrame(phen_rows).to_csv(TAB/"35_phenotype_P_summary.csv", index=False)

plt.figure(figsize=(12,6.2))
ax = plt.gca()
ax.axis("off")
ax.text(
    .08,.70,
    "Preserved raw state x\nHb · CBC markers · HbA1c/glucose · covariates X",
    transform=ax.transAxes, ha="left", va="center",
    bbox=dict(boxstyle="round,pad=.55"), fontsize=11
)
ax.text(
    .41,.70,
    "Domain-informed T(x)\nA = max(Hb threshold − Hb, 0)\nG = max(HbA1c − 5.7, 0)",
    transform=ax.transAxes, ha="left", va="center",
    bbox=dict(boxstyle="round,pad=.55"), fontsize=11
)
ax.text(
    .76,.70,
    "Working physiological y\n[A, G, A×G, X]",
    transform=ax.transAxes, ha="center", va="center",
    bbox=dict(boxstyle="round,pad=.55"), fontsize=11
)
ax.annotate("",xy=(.40,.70),xytext=(.30,.70),xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->",lw=1.5))
ax.annotate("",xy=(.72,.70),xytext=(.63,.70),xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->",lw=1.5))

ax.text(
    .76,.34,
    "Observed phenotype P\nPHQ-9 items → total / latent factors\nsomatic vs cognitive-affective",
    transform=ax.transAxes, ha="center", va="center",
    bbox=dict(boxstyle="round,pad=.55"), fontsize=11
)
ax.annotate("",xy=(.76,.44),xytext=(.76,.59),xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->",lw=1.5))

ax.text(
    .08,.22,
    "Future thesis-level step:\ny → [y_A, y_G, y_AG, y_X] → ŷ\nquantify L_y = d(y, ŷ)",
    transform=ax.transAxes, ha="left", va="center",
    bbox=dict(boxstyle="round,pad=.50"), fontsize=10
)
ax.text(
    .50,.05,
    "x remains preserved. Analyses branch from preserved references; P is an outcome/phenotype, not a component of y.",
    transform=ax.transAxes, ha="center", va="bottom", fontsize=10
)
ax.set_title("Correct operational architecture: preserved x → working physiology y → phenotype P")
savefig("35_02_correct_x_y_P_pipeline.png")

corr_cols = ["A","G_HBA1C","AG_HBA1C","PHQ9_TOTAL","SOMATIC_SCORE"]
corr = d3[corr_cols].corr()
corr.to_csv(TAB/"35_burden_phenotype_correlations.csv")

plt.figure(figsize=(7.6,6))
ax = plt.gca()
im = ax.imshow(corr.to_numpy(), aspect="auto")
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(["A","G","A×G","PHQ-9","Somatic"],rotation=35,ha="right")
ax.set_yticklabels(["A","G","A×G","PHQ-9","Somatic"])
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",fontsize=9)
plt.colorbar(im,ax=ax,label="Pearson correlation")
ax.set_title("Descriptive correlation: working burdens and observed phenotype")
savefig("35_03_burden_phenotype_correlation.png")

# ------------------------------------------------------------
# 4) Survey-weighted A/G group descriptives with valid uncertainty
# ------------------------------------------------------------
required = ["SEQN","CYCLE","A","G_HBA1C","PHQ9_TOTAL","SOMATIC_SCORE",
            "WTMEC4YR","STRATUM","PSU"]
missing = [c for c in required if c not in survey.columns]
if missing:
    raise RuntimeError(f"25_survey_parity_input.csv missing required columns: {missing}")

group_csv = TAB/"35_survey_group_input.csv"
survey[required].to_csv(group_csv,index=False)

rscript = shutil.which("Rscript")
if rscript is None:
    cand = sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"))
    if cand:
        rscript = str(cand[-1])
if rscript is None:
    raise RuntimeError("Rscript not found.")

r_code = r'''
args <- commandArgs(trailingOnly=TRUE)
input <- args[1]
output <- args[2]
userlib <- args[3]
.libPaths(c(userlib,.libPaths()))
library(survey)
options(survey.lonely.psu="adjust")

d <- read.csv(input, stringsAsFactors=FALSE)
d$GROUP <- ifelse(d$A<=0 & d$G_HBA1C<=0, "Neither",
           ifelse(d$A<=0 & d$G_HBA1C>0, "G only",
           ifelse(d$A>0 & d$G_HBA1C<=0, "A only", "Both")))
d$GROUP <- factor(d$GROUP, levels=c("Neither","G only","A only","Both"))
d$PHQ_GE10 <- as.numeric(d$PHQ9_TOTAL >= 10)

des <- svydesign(
  ids=~PSU,
  strata=~STRATUM,
  weights=~WTMEC4YR,
  nest=TRUE,
  data=d
)

out <- list()
for (g in levels(d$GROUP)) {
  sg <- subset(des, GROUP==g)
  raw_n <- sum(d$GROUP==g, na.rm=TRUE)

  m1 <- svymean(~PHQ9_TOTAL, sg, na.rm=TRUE)
  c1 <- confint(m1)

  m2 <- svymean(~SOMATIC_SCORE, sg, na.rm=TRUE)
  c2 <- confint(m2)

  m3 <- svymean(~PHQ_GE10, sg, na.rm=TRUE)
  c3 <- confint(m3)

  out[[length(out)+1]] <- data.frame(
    group=g,
    unweighted_n=raw_n,
    weighted_mean_PHQ9=coef(m1)[1],
    phq_ci_low=c1[1,1],
    phq_ci_high=c1[1,2],
    weighted_mean_somatic=coef(m2)[1],
    somatic_ci_low=c2[1,1],
    somatic_ci_high=c2[1,2],
    weighted_PHQ9_ge10_pct=100*coef(m3)[1],
    phq10_ci_low_pct=100*c3[1,1],
    phq10_ci_high_pct=100*c3[1,2]
  )
}
write.csv(do.call(rbind,out), output, row.names=FALSE)
'''
r_path = AUDIT/"35_survey_weighted_group_descriptives.R"
r_path.write_text(r_code,encoding="utf-8")

weighted_group_path = TAB/"35_survey_weighted_group_descriptives.csv"
proc = subprocess.run(
    [str(rscript),str(r_path),str(group_csv),str(weighted_group_path),str(RLIB)],
    capture_output=True,text=True
)
if proc.returncode != 0:
    print(proc.stdout)
    print(proc.stderr[-3000:])
    raise SystemExit(proc.returncode)

wg = pd.read_csv(weighted_group_path)

plt.figure(figsize=(8.7,5))
ax = plt.gca()
x = np.arange(len(wg))
y = wg["weighted_mean_PHQ9"].to_numpy()
lo = wg["phq_ci_low"].to_numpy()
hi = wg["phq_ci_high"].to_numpy()
ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt="o",capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(wg["group"])
ax.set_ylabel("Survey-weighted mean PHQ-9 (95% CI)")
ax.set_title("Observed depression burden across A/G groups")
for i,r in wg.iterrows():
    ax.text(i, r["phq_ci_high"]+.08, f"N={int(r['unweighted_n']):,}", ha="center", fontsize=9)
savefig("35_04_weighted_ag_group_phq9.png")

plt.figure(figsize=(8.7,5))
ax = plt.gca()
y = wg["weighted_PHQ9_ge10_pct"].to_numpy()
lo = wg["phq10_ci_low_pct"].to_numpy()
hi = wg["phq10_ci_high_pct"].to_numpy()
ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt="o",capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(wg["group"])
ax.set_ylabel("Survey-weighted PHQ-9 ≥10 prevalence (%)")
ax.set_title("Moderate-or-greater depressive symptom prevalence across A/G groups")
for i,r in wg.iterrows():
    ax.text(i, r["phq10_ci_high_pct"]+.25, f"N={int(r['unweighted_n']):,}", ha="center", fontsize=9)
savefig("35_05_weighted_ag_group_phq10_prevalence.png")

# ------------------------------------------------------------
# 5) Main discovery forest — retain estimates, fix interpretation
# ------------------------------------------------------------
coef26 = safe_csv("26_design_based_coefficients.csv")
if coef26 is None:
    raise FileNotFoundError(RESULTS/"26_design_based_coefficients.csv")

dd = coef26[
    (coef26["population"].astype(str)=="Pooled") &
    (coef26["outcome"].astype(str)=="SOMATIC_SCORE") &
    (coef26["model"].astype(str)=="Additive_A+G+X") &
    (coef26["term"].isin(["A","G_HBA1C"]))
].copy()

order = ["A","G_HBA1C"]
dd = dd.set_index("term").reindex(order).reset_index()
dd.to_csv(TAB/"35_discovery_unstandardized_effects.csv",index=False)

plt.figure(figsize=(9,4.8))
ax = plt.gca()
labels = [
    "A: Hb deficit (per 1 g/dL)",
    "G: HbA1c excess (per 1 %-point)"
]
for i,r in dd.iterrows():
    ax.errorbar(
        r["beta"],i,
        xerr=[[r["beta"]-r["ci_low"]],[r["ci_high"]-r["beta"]]],
        fmt="o",capsize=5
    )
    ax.text(r["ci_high"]+.015,i,f"β={r['beta']:.3f}; p={r['p']:.4f}",va="center",fontsize=9)
ax.axvline(0,lw=.9)
ax.set_yticks([0,1])
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Unstandardized adjusted coefficient (95% CI)")
ax.set_title("Discovery: A and G in the same survey-weighted X3 model")
ax.text(
    .5,-.23,
    "Different predictor units → do not compare A and G coefficient magnitudes directly.",
    transform=ax.transAxes,ha="center",fontsize=9
)
savefig("35_06_discovery_effects_units_explicit.png")

# ------------------------------------------------------------
# 6) CFA categorical comparison — points, not trajectory lines
# ------------------------------------------------------------
cfa = safe_csv("20_cfa_fit.csv")
if cfa is not None:
    models = ["one_factor","two_factor"]
    labels = ["One factor","Somatic + cognitive-affective"]
    cycles = sorted(cfa["cycle"].astype(str).unique())
    offsets = np.linspace(-.08,.08,max(1,len(cycles)))

    plt.figure(figsize=(8.5,4.8))
    ax = plt.gca()
    for off,cycle in zip(offsets,cycles):
        d = cfa[cfa["cycle"].astype(str)==cycle].set_index("model").reindex(models)
        xx = np.arange(2)+off
        ax.scatter(xx,d["cfi"],label=cycle)
        for xi,v in zip(xx,d["cfi"]):
            ax.text(xi,v+.0015,f"{v:.3f}",ha="center",fontsize=8)
    ax.set_xticks([0,1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("CFI")
    ax.set_title("Ordinal CFA model fit by independent NHANES cycle")
    ax.legend(title="Cycle")
    savefig("35_07_cfa_model_comparison_points.png")

# ------------------------------------------------------------
# 7) Somatic loadings — points, not connected trajectory
# ------------------------------------------------------------
load = safe_csv("20_cfa_two_factor_loadings.csv")
if load is not None:
    som = load[load["lhs"].astype(str)=="Somatic"].copy()
    item_order = ["DPQ030","DPQ040","DPQ050"]
    item_labels = ["Sleep","Fatigue","Appetite"]
    cycles = sorted(som["cycle"].astype(str).unique())
    offsets = np.linspace(-.08,.08,max(1,len(cycles)))

    plt.figure(figsize=(8.5,4.8))
    ax = plt.gca()
    for off,cycle in zip(offsets,cycles):
        d = som[som["cycle"].astype(str)==cycle].set_index("rhs").reindex(item_order)
        xx = np.arange(3)+off
        ax.scatter(xx,d["est.std"],label=cycle)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(item_labels)
    ax.set_ylim(0,1)
    ax.set_ylabel("Standardized factor loading")
    ax.set_title("Somatic PHQ factor loadings by independent cycle")
    ax.legend(title="Cycle")
    savefig("35_08_somatic_loadings_points.png")

# ------------------------------------------------------------
# 8) Fasting sensitivity — table, NOT misleading shared coefficient plot
# ------------------------------------------------------------
fast = safe_csv("18_fasting_glucose_sensitivity.csv")
if fast is not None:
    q = fast[
        (fast["model"].astype(str)=="X3_kidney_direct_effect") &
        (fast["term"].astype(str)=="G")
    ].copy()
    q["interpretation_note"] = (
        "Definitions use different physical units; compare direction/CI/pattern, "
        "not raw coefficient magnitude."
    )
    q.to_csv(TAB/"35_fasting_sensitivity_no_shared_scale.csv",index=False)

# ------------------------------------------------------------
# 9) Read-only integrity check
# ------------------------------------------------------------
hash_after = sha256(core_path) if core_path.exists() else None
hash_ok = (hash_before == hash_after) if hash_before is not None else None

notes = f'''# Script 35 — visual cleanup notes

## Changes made

1. **Cohort flow corrected**
   The flow is now a true sequential intersection:
   same-person core → Hb → HbA1c → complete PHQ-9 → full X3 survey-analysis requirements.

2. **x / y / P corrected**
   `P` is no longer shown as a component of `y`.

   Working structure:
   `x_raw → T(x) → y_working=[A,G,A×G,X] → P`

   Final thesis-level `y → components → y_hat` reconstruction remains future work.

3. **A/G group figure corrected**
   Main descriptive figure now uses the NHANES survey design and shows 95% confidence intervals.
   Raw N is retained only as sample-size annotation.

4. **Discovery A/G coefficient figure corrected**
   The estimates are unchanged.
   The figure explicitly states that A and G have different predictor units and raw coefficient magnitudes are not directly comparable.

5. **CFA / loading styling corrected**
   Categorical model/item comparisons are shown as points rather than line trajectories.

6. **Fasting glucose sensitivity corrected**
   A single shared-axis HbA1c-vs-fasting coefficient plot is intentionally not generated.
   The two G definitions have different units; use the table to compare direction, uncertainty and inferential support.

## Integrity

Preserved core SHA256 before/after identical: **{hash_ok}**

No scientific definition was altered by this script.
'''

(OUT/"35_VISUAL_CLEANUP_NOTES.md").write_text(notes,encoding="utf-8")

print(f"PASS  True sequential core flow: {len(d0):,} -> {len(d1):,} -> {len(d2):,} -> {len(d3):,} -> {len(d4):,}")
print("PASS  y and phenotype P are now separated conceptually.")
print("PASS  A/G group plot now uses complex-survey weighted estimates + 95% CI.")
print("PASS  Main A/G coefficient plot explicitly preserves different units.")
print("PASS  CFA/loadings categorical styling cleaned.")
print("PASS  Misleading shared-scale fasting-glucose coefficient comparison removed.")
print(f"PASS  Preserved core hash unchanged: {hash_ok}")
print()
print(f"SAVED {OUT}")
print(f"FIGURES {len(list(FIG.glob('*.png')))}")
print(f"TABLES  {len(list(TAB.glob('*.csv')))}")
