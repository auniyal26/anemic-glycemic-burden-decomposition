from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib

# ============================================================
# 36 — FINAL PRESENTATION CANDIDATE FIGURES
# ============================================================
# READ-ONLY presentation build from already-computed results.
# No model refitting, threshold changes, tuning, or protocol changes.
# ============================================================

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
RESULTS = ROOT / "Results"
PROCESSED = ROOT / "Data" / "Processed"
CLEAN = RESULTS / "Presentation_Evidence_Clean"
OUT = RESULTS / "Presentation_Final_Candidates"
FIG = OUT / "Figures"
TAB = OUT / "Tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

def csv(path):
    p = Path(path)
    if not p.is_absolute():
        p = RESULTS / p
    return pd.read_csv(p)

def savefig(name):
    p = FIG / name
    plt.tight_layout()
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    return p

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def weighted_corr(x, y, w):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    w = w / w.sum()
    mx, my = np.sum(w*x), np.sum(w*y)
    cov = np.sum(w*(x-mx)*(y-my))
    vx = np.sum(w*(x-mx)**2)
    vy = np.sum(w*(y-my)**2)
    return cov / np.sqrt(vx*vy)

print()
print("36 — FINAL PRESENTATION CANDIDATE FIGURES")
print("=========================================")
print("READ-ONLY: existing results only; no scientific refitting.")
print()

core = PROCESSED / "nhanes_core_working.parquet"
hash_before = sha256(core) if core.exists() else None

# ------------------------------------------------------------
# 1) Current work vs proposed thesis architecture
# ------------------------------------------------------------
plt.figure(figsize=(13, 7))
ax = plt.gca()
ax.axis("off")

ax.text(.02,.92,"WHAT HAS BEEN TESTED",transform=ax.transAxes,
        fontsize=12,fontweight="bold",va="top")

ax.text(.07,.69,
        "Preserved physiological measurements\n"
        "x_phys = [Hb, CBC markers, HbA1c / glucose]",
        transform=ax.transAxes,ha="left",va="center",
        bbox=dict(boxstyle="round,pad=.55"),fontsize=10.5)

ax.text(.38,.69,
        "Domain-informed first representation\n"
        "A = Hb deficit\nG = glycemic excess\nA×G = interaction",
        transform=ax.transAxes,ha="left",va="center",
        bbox=dict(boxstyle="round,pad=.55"),fontsize=10.5)

ax.text(.72,.69,
        "Observed depressive phenotype P\n"
        "PHQ-9 total + item structure\n"
        "somatic / cognitive-affective factors",
        transform=ax.transAxes,ha="left",va="center",
        bbox=dict(boxstyle="round,pad=.55"),fontsize=10.5)

ax.annotate("",xy=(.36,.69),xytext=(.27,.69),
            xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=1.5))
ax.annotate("",xy=(.70,.69),xytext=(.61,.69),
            xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=1.5))

ax.text(.42,.48,
        "Adjustment X enters the statistical model separately:\n"
        "age · sex · race · SES · education · smoking · BMI · kidney function",
        transform=ax.transAxes,ha="center",va="center",fontsize=9.5)

ax.text(.02,.33,"WHAT THE PhD ASKS NEXT",transform=ax.transAxes,
        fontsize=12,fontweight="bold",va="top")

ax.text(.09,.15,
        "Preserved x_phys",
        transform=ax.transAxes,ha="center",va="center",
        bbox=dict(boxstyle="round,pad=.45"),fontsize=10.5)
ax.text(.33,.15,
        "richer working representation y*",
        transform=ax.transAxes,ha="center",va="center",
        bbox=dict(boxstyle="round,pad=.45"),fontsize=10.5)
ax.text(.58,.15,
        "separable components\n[y_A, y_G, y_AG, ...]",
        transform=ax.transAxes,ha="center",va="center",
        bbox=dict(boxstyle="round,pad=.45"),fontsize=10.5)
ax.text(.82,.15,
        "reconstruction ŷ*\nquantify information loss",
        transform=ax.transAxes,ha="center",va="center",
        bbox=dict(boxstyle="round,pad=.45"),fontsize=10.5)

for x1,x2 in [(.17,.26),(.41,.49),(.67,.74)]:
    ax.annotate("",xy=(x2,.15),xytext=(x1,.15),
                xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=1.5))

ax.set_title("From the completed feasibility analysis to the proposed decomposition problem")
savefig("36_01_current_vs_proposed_architecture.png")

# ------------------------------------------------------------
# 2) True sequential cohort flow — polished
# ------------------------------------------------------------
flow = csv(CLEAN / "Tables" / "35_true_sequential_sample_flow.csv")
plt.figure(figsize=(10,5.2))
ax = plt.gca()
x = np.arange(len(flow))
ax.bar(x, flow["n"])
ax.set_xticks(x)
ax.set_xticklabels([
    "Core match\nage 18+",
    "+ Hb",
    "+ HbA1c",
    "+ PHQ-9",
    "+ full X3\nrequirements"
])
ax.set_ylabel("Participants")
ax.set_title("NHANES 2005–2008 discovery cohort")
for i,r in flow.iterrows():
    ax.text(i,r["n"]+100,f"{int(r['n']):,}\n({r['retained_from_start_pct']:.1f}%)",
            ha="center",fontsize=9)
savefig("36_02_discovery_cohort_flow.png")

# ------------------------------------------------------------
# 3) Survey-weighted descriptive somatic phenotype by A/G group
# ------------------------------------------------------------
wg = csv(CLEAN / "Tables" / "35_survey_weighted_group_descriptives.csv")
plt.figure(figsize=(8.8,5))
ax = plt.gca()
x = np.arange(len(wg))
y = wg["weighted_mean_somatic"].to_numpy()
lo = wg["somatic_ci_low"].to_numpy()
hi = wg["somatic_ci_high"].to_numpy()
ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt="o",capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(wg["group"])
ax.set_ylabel("Survey-weighted mean somatic score (95% CI)")
ax.set_title("Somatic depressive symptoms across first-pass A/G groups")
for i,r in wg.iterrows():
    ax.text(i,r["somatic_ci_high"]+.05,f"N={int(r['unweighted_n']):,}",ha="center",fontsize=9)
ax.text(.5,-.19,
        "Descriptive and survey-weighted, but unadjusted for X; group differences are not causal effects.",
        transform=ax.transAxes,ha="center",fontsize=9)
savefig("36_03_weighted_somatic_by_ag_group.png")

# Keep PHQ total descriptive table for appendix.
wg.to_csv(TAB/"36_weighted_ag_group_descriptives.csv",index=False)

# ------------------------------------------------------------
# 4) Primary discovery effects — units explicit
# ------------------------------------------------------------
coef = csv("26_design_based_coefficients.csv")
d = coef[
    (coef["population"].astype(str)=="Pooled") &
    (coef["outcome"].astype(str)=="SOMATIC_SCORE") &
    (coef["model"].astype(str)=="Additive_A+G+X") &
    (coef["term"].isin(["A","G_HBA1C"]))
].copy().set_index("term").reindex(["A","G_HBA1C"]).reset_index()

plt.figure(figsize=(9.2,4.8))
ax = plt.gca()
labels = ["A: Hb deficit\n(per 1 g/dL)","G: HbA1c excess\n(per 1 %-point)"]
for i,r in d.iterrows():
    ax.errorbar(r["beta"],i,
                xerr=[[r["beta"]-r["ci_low"]],[r["ci_high"]-r["beta"]]],
                fmt="o",capsize=5)
    ax.text(r["ci_high"]+.012,i,
            f"β={r['beta']:.3f}\n95% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]\np={r['p']:.4f}",
            va="center",fontsize=8.7)
ax.axvline(0,lw=.9)
ax.set_yticks([0,1])
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Unstandardized adjusted somatic-score coefficient")
ax.set_title("Discovery: A and G estimated jointly in the survey-weighted X3 model")
ax.text(.5,-.21,
        "Different predictor units: coefficient magnitudes are not directly comparable.",
        transform=ax.transAxes,ha="center",fontsize=9)
savefig("36_04_discovery_primary_effects.png")

# ------------------------------------------------------------
# 5) Separability evidence — weighted correlation + incremental R2
# ------------------------------------------------------------
survey = csv("25_survey_parity_input.csv")
rho = weighted_corr(survey["A"],survey["G_HBA1C"],survey["WTMEC4YR"])

inc = csv("27_incremental_information_tests.csv")
incd = inc[
    (inc["population"].astype(str)=="Pooled") &
    (inc["outcome"].astype(str)=="SOMATIC_SCORE") &
    (inc["test"].isin(["Add_A_to_G_plus_X","Add_G_to_A_plus_X"]))
].copy()
label_map = {
    "Add_A_to_G_plus_X":"Add A beyond G + X",
    "Add_G_to_A_plus_X":"Add G beyond A + X",
}
incd["label"] = incd["test"].map(label_map)
incd["delta_R2_pp"] = 100*incd["weighted_delta_R2"]

plt.figure(figsize=(8.7,4.8))
ax = plt.gca()
ax.bar(incd["label"],incd["delta_R2_pp"])
ax.set_ylabel("Incremental weighted R² (percentage points)")
ax.set_title("A and G contain non-identical information about somatic symptoms")
for i,r in incd.reset_index(drop=True).iterrows():
    ax.text(i,r["delta_R2_pp"]+.008,
            f"ΔR²={r['weighted_delta_R2']:.4f}\np={r['p']:.4f}",
            ha="center",fontsize=9)
ax.text(.5,.82,
        f"Survey-weighted corr(A, G) = {rho:.3f}",
        transform=ax.transAxes,ha="center",fontsize=10)
ax.text(.5,-.20,
        "ΔR² is small: this is evidence of incremental information, not large predictive power.",
        transform=ax.transAxes,ha="center",fontsize=9)
savefig("36_05_separability_incremental_information.png")

# ------------------------------------------------------------
# 6) CFA fit summary as a compact evidence table figure
# ------------------------------------------------------------
cfa = csv("20_cfa_fit.csv").copy()
cfa["cycle_label"] = cfa["cycle"].astype(str).map({
    "506":"2005–06","708":"2007–08","0506":"2005–06","0708":"2007–08"
}).fillna(cfa["cycle"].astype(str))
cfa["model_label"] = cfa["model"].map({
    "one_factor":"One factor",
    "two_factor":"Two factors"
})

rows = []
for cyc in ["2005–06","2007–08"]:
    for mod in ["One factor","Two factors"]:
        r = cfa[(cfa["cycle_label"]==cyc)&(cfa["model_label"]==mod)].iloc[0]
        rows.append([cyc,mod,f"{r['cfi']:.3f}",f"{r['rmsea']:.3f}",f"{r['srmr']:.3f}"])

plt.figure(figsize=(9.2,4.4))
ax = plt.gca()
ax.axis("off")
table = ax.table(
    cellText=rows,
    colLabels=["Cycle","PHQ structure","CFI ↑","RMSEA ↓","SRMR ↓"],
    loc="center",
    cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1,1.7)
ax.set_title("Ordinal CFA: two-factor PHQ structure improves fit in both cycles",pad=18)
ax.text(.5,.04,
        "The one-factor model already fits well; the evidence supports a reproducible refinement, not a uniquely proven ontology.",
        transform=ax.transAxes,ha="center",fontsize=9)
savefig("36_06_cfa_fit_summary.png")

# ------------------------------------------------------------
# 7) Somatic factor loadings with 95% CI
# ------------------------------------------------------------
load = csv("20_cfa_two_factor_loadings.csv")
som = load[load["lhs"].astype(str)=="Somatic"].copy()
som["cycle_label"] = som["cycle"].astype(str).map({
    "506":"2005–06","708":"2007–08","0506":"2005–06","0708":"2007–08"
}).fillna(som["cycle"].astype(str))
order = ["DPQ030","DPQ040","DPQ050"]
item_labels = ["Sleep","Fatigue","Appetite"]
offsets = {"2005–06":-.06,"2007–08":.06}

plt.figure(figsize=(8.8,4.8))
ax = plt.gca()
for cyc in ["2005–06","2007–08"]:
    dd = som[som["cycle_label"]==cyc].set_index("rhs").reindex(order)
    xx = np.arange(3)+offsets[cyc]
    est = dd["est.std"].to_numpy()
    se = dd["se"].to_numpy()
    ax.errorbar(xx,est,yerr=1.96*se,fmt="o",capsize=4,label=cyc)
ax.set_xticks([0,1,2])
ax.set_xticklabels(item_labels)
ax.set_ylim(.60,.86)
ax.set_ylabel("Standardized factor loading (approx. 95% CI)")
ax.set_title("Somatic PHQ factor reproduces across discovery cycles")
ax.legend(title="NHANES cycle")
savefig("36_07_somatic_factor_loadings.png")

# ------------------------------------------------------------
# 8) MIMIC somatic paths — standardized predictors
# ------------------------------------------------------------
mimic = csv("21_mimic_paths.csv")
ms = mimic[
    (mimic["lhs"].astype(str)=="Somatic") &
    (mimic["rhs"].isin(["A_Z","G_HBA1C_Z","AG_Z"]))
].copy()
ms["cycle_label"] = ms["cycle"].astype(str).map({
    "506":"2005–06","708":"2007–08","0506":"2005–06","0708":"2007–08"
}).fillna(ms["cycle"].astype(str))
terms = ["A_Z","G_HBA1C_Z","AG_Z"]
term_labels = ["A","G","A×G"]
offsets = {"2005–06":-.07,"2007–08":.07}

plt.figure(figsize=(8.8,4.9))
ax = plt.gca()
for cyc in ["2005–06","2007–08"]:
    dd = ms[ms["cycle_label"]==cyc].set_index("rhs").reindex(terms)
    xx = np.arange(3)+offsets[cyc]
    est = dd["est"].to_numpy()
    lo = dd["ci.lower"].to_numpy()
    hi = dd["ci.upper"].to_numpy()
    ax.errorbar(xx,est,yerr=[est-lo,hi-est],fmt="o",capsize=4,label=cyc)
ax.axhline(0,lw=.9)
ax.set_xticks([0,1,2])
ax.set_xticklabels(term_labels)
ax.set_ylabel("Latent somatic path estimate (95% CI)")
ax.set_title("Weighted MIMIC: A and G map consistently to the latent somatic phenotype")
ax.legend(title="NHANES cycle")
ax.text(.5,-.18,
        "Predictors are standardized; A×G is secondary because its path does not replicate.",
        transform=ax.transAxes,ha="center",fontsize=9)
savefig("36_08_mimic_somatic_paths.png")

# ------------------------------------------------------------
# 9) Reduced hematology PCA — multivariate representation evidence
# ------------------------------------------------------------
pca = csv("25_reduced_A_pca_sensitivity.csv")
x = np.arange(len(pca))
width=.34

plt.figure(figsize=(8.8,4.9))
ax = plt.gca()
ax.bar(x-width/2,100*pca["variance_0506"],width=width,label="2005–06")
ax.bar(x+width/2,100*pca["variance_0708"],width=width,label="2007–08")
ax.set_xticks(x)
ax.set_xticklabels([f"PC{int(c)}" for c in pca["component"]])
ax.set_ylabel("Explained standardized CBC variance (%)")
ax.set_title("Reduced CBC basis contains stable multivariate structure")
ax.legend(title="NHANES cycle")
ax.text(.5,-.18,
        f"Cross-cycle loading similarity: {pca['cross_cycle_similarity'].min():.3f}–{pca['cross_cycle_similarity'].max():.3f}. "
        "PCs are measurement axes, not named biological mechanisms.",
        transform=ax.transAxes,ha="center",fontsize=8.8)
savefig("36_09_reduced_hematology_pca.png")

# ------------------------------------------------------------
# 10) Frozen temporal transfer summary — table, not false shared scale
# ------------------------------------------------------------
tr = csv("32_transfer_design_based_coefficients.csv")
tr = tr[
    (tr["outcome"].astype(str)=="SOMATIC_SCORE") &
    (tr["model"].astype(str)=="Additive_A+G+X") &
    (tr["term"].isin(["A","G_HBA1C"]))
].copy()

rows=[]
for pop in ["2009-2018","2021-2023"]:
    for term in ["A","G_HBA1C"]:
        r = tr[(tr["population"]==pop)&(tr["term"]==term)].iloc[0]
        if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]):
            infer = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}], p={r['p']:.4f}"
        else:
            infer = "CI/p not estimable at X3 (residual survey df=0)"
        rows.append([
            pop,
            "A" if term=="A" else "G",
            f"{r['beta']:.3f}",
            infer
        ])

plt.figure(figsize=(11,4.4))
ax=plt.gca()
ax.axis("off")
table=ax.table(
    cellText=rows,
    colLabels=["Validation period","Term","Frozen X3 β","Design-based inference"],
    loc="center",cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1,1.7)
ax.set_title("Frozen temporal transfer: strong 2009–2018 replication, modern A non-transfer",pad=18)
ax.text(.5,.03,
        "Do not infer the 2021–23 A failure from the X3 NaN alone; the X0 diagnostic below already shows A near zero.",
        transform=ax.transAxes,ha="center",fontsize=9)
savefig("36_10_frozen_temporal_transfer_summary.png")

# ------------------------------------------------------------
# 11/12) 2021–23 diagnostic ladders — separate scales for A and G
# ------------------------------------------------------------
lad = csv("33_2123_x_ladder_coefficients.csv")
design = csv("33_2123_design_diagnostics.csv")
if "model" in design.columns and "residual_design_df" in design.columns:
    dfmap = dict(zip(design["model"].astype(str),design["residual_design_df"]))

for term,label,num in [("A","A: Hb-deficit representation",11),
                       ("G_HBA1C","G: HbA1c-excess representation",12)]:
    dd = lad[lad["term"].astype(str)==term].set_index("model").reindex(["X0","X1","X2","X3"]).reset_index()
    plt.figure(figsize=(8.8,4.9))
    ax=plt.gca()
    x=np.arange(4)
    ax.plot(x,dd["beta"],marker="o")
    ax.axhline(0,lw=.9)
    for i,r in dd.iterrows():
        if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]):
            ax.errorbar(i,r["beta"],
                        yerr=[[r["beta"]-r["ci_low"]],[r["ci_high"]-r["beta"]]],
                        fmt="none",capsize=4)
        rdf = dfmap.get(str(r["model"]),np.nan) if "dfmap" in locals() else np.nan
        txt=f"β={r['beta']:.3f}"
        if np.isfinite(rdf):
            txt+=f"\ndf={int(rdf)}"
        ax.text(i,r["beta"]+.018,txt,ha="center",fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["X0\nage/sex/race","+ SES/edu/\nsmoking","+ BMI","+ kidney"])
    ax.set_ylabel("Survey-weighted coefficient")
    ax.set_title(f"2021–2023 diagnostic ladder — {label}")
    if term=="A":
        ax.text(.5,-.19,
                "A is already near zero at X0 where design-based inference is valid; this is not only an X3 df problem.",
                transform=ax.transAxes,ha="center",fontsize=9)
    else:
        ax.text(.5,-.19,
                "G remains positive across the ladder; formal uncertainty weakens as residual survey df collapses.",
                transform=ax.transAxes,ha="center",fontsize=9)
    savefig(f"36_{num:02d}_2123_{'A' if term=='A' else 'G'}_diagnostic_ladder.png")

# ------------------------------------------------------------
# 13) Presentation decision map
# ------------------------------------------------------------
decisions = pd.DataFrame([
    ["36_01_current_vs_proposed_architecture.png","MAIN","Defines what was done versus the actual PhD problem."],
    ["36_02_discovery_cohort_flow.png","MAIN","Transparent participant accounting."],
    ["34_08_hb_to_A_transform.png + 34_09_hba1c_to_G_transform.png","MAIN","Show exactly how the first A/G representation was constructed."],
    ["36_03_weighted_somatic_by_ag_group.png","MAIN","Intuitive survey-weighted descriptive phenotype pattern."],
    ["36_04_discovery_primary_effects.png","MAIN","Primary adjusted discovery result."],
    ["36_05_separability_incremental_information.png","MAIN/APPENDIX","Use if supervisor asks what 'separable' means quantitatively."],
    ["36_06_cfa_fit_summary.png","MAIN","Why somatic vs cognitive-affective decomposition is justified."],
    ["36_07_somatic_factor_loadings.png","MAIN/APPENDIX","Shows reproducibility of somatic measurement structure."],
    ["36_08_mimic_somatic_paths.png","MAIN","Links A/G to the latent somatic phenotype."],
    ["36_09_reduced_hematology_pca.png","MAIN","Motivates why scalar Hb-deficit A may be insufficient."],
    ["36_10_frozen_temporal_transfer_summary.png","MAIN","Honest frozen transfer result."],
    ["36_11_2123_A_diagnostic_ladder.png","MAIN","Shows substantive A non-transfer."],
    ["36_12_2123_G_diagnostic_ladder.png","APPENDIX/MAIN","Shows G direction is more stable."],
    ["35_03_burden_phenotype_correlation.png","APPENDIX","Raw correlations are small; PHQ-total/somatic correlation is partly mechanical."],
    ["35_05_weighted_ag_group_phq10_prevalence.png","APPENDIX","Thresholded prevalence is not monotonic across groups."],
    ["34_15_incremental_weighted_R2.png","REPLACE","Use 36_05 instead."],
    ["34_19_fasting_glucose_sensitivity.png","REPLACE","Different units make shared coefficient axis misleading; use table/verbal sensitivity result."],
    ["34_21_G_pca_variance.png","APPENDIX","Only two glycemic markers; useful but not central."],
    ["34_25_2123_residual_design_df.png","APPENDIX","Technical explanation of modern inference limits."],
    ["34_30_reference_preserving_branch_architecture.png","APPENDIX","Integrity/methods backup; 36_01 is cleaner for the main story."],
],columns=["asset","decision","reason"])
decisions.to_csv(TAB/"36_presentation_asset_decisions.csv",index=False)

# ------------------------------------------------------------
# 14) Key numbers cheat sheet
# ------------------------------------------------------------
key = pd.DataFrame([
    ["Discovery same-person age18+ core","N",len(flow) and int(flow.iloc[0]["n"])],
    ["Complete A/G/PHQ discovery core","N",9743],
    ["Final discovery X3 survey cohort","N",7957],
    ["Discovery A","beta",float(d[d["term"]=="A"]["beta"].iloc[0])],
    ["Discovery A","p",float(d[d["term"]=="A"]["p"].iloc[0])],
    ["Discovery G","beta",float(d[d["term"]=="G_HBA1C"]["beta"].iloc[0])],
    ["Discovery G","p",float(d[d["term"]=="G_HBA1C"]["p"].iloc[0])],
    ["Weighted corr(A,G)","rho",rho],
    ["A incremental somatic weighted R2","delta_R2",float(incd[incd["test"]=="Add_A_to_G_plus_X"]["weighted_delta_R2"].iloc[0])],
    ["G incremental somatic weighted R2","delta_R2",float(incd[incd["test"]=="Add_G_to_A_plus_X"]["weighted_delta_R2"].iloc[0])],
],columns=["result","statistic","value"])
key.to_csv(TAB/"36_key_numbers_cheat_sheet.csv",index=False)

hash_after = sha256(core) if core.exists() else None
print(f"PASS  Preserved core hash unchanged: {hash_before == hash_after}")
print(f"PASS  Generated {len(list(FIG.glob('*.png')))} final candidate figures.")
print(f"PASS  Generated {len(list(TAB.glob('*.csv')))} supporting tables.")
print(f"SAVED {OUT}")
