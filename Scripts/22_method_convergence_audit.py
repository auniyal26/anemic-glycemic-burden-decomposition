from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
RESULTS = ROOT / "Results"
FIG = RESULTS / "Figures"
FIG.mkdir(parents=True, exist_ok=True)

survey = pd.read_csv(RESULTS / "16_staged_survey_results.csv")
mimic = pd.read_csv(RESULTS / "21_mimic_paths.csv")

survey = survey[
    (survey["outcome"] == "SOMATIC_SCORE")
    & (survey["model"] == "X3_kidney_direct_effect")
    & (survey["term"].isin(["A", "G", "AG"]))
].copy()

survey["method"] = "Survey-weighted observed somatic score"
survey["effect"] = survey["beta"]
survey["low"] = survey["ci_low"]
survey["high"] = survey["ci_high"]
survey["supported"] = (survey["low"] > 0) | (survey["high"] < 0)
survey["direction"] = np.sign(survey["effect"]).astype(int)

term_map = {
    "A_Z": "A",
    "G_HBA1C_Z": "G",
    "AG_Z": "AG"
}

latent = mimic[
    (mimic["lhs"] == "Somatic")
    & (mimic["rhs"].isin(term_map))
].copy()

latent["term"] = latent["rhs"].map(term_map)
latent["method"] = "Ordinal MIMIC latent somatic factor"
latent["effect"] = latent["est"]
latent["low"] = latent["ci.lower"]
latent["high"] = latent["ci.upper"]
latent["supported"] = (latent["low"] > 0) | (latent["high"] < 0)
latent["direction"] = np.sign(latent["effect"]).astype(int)

latent_summary = []

for term in ["A", "G", "AG"]:
    d = latent[latent["term"] == term].sort_values("cycle")

    same_direction = len(d) == 2 and d["direction"].nunique() == 1
    both_supported = len(d) == 2 and d["supported"].all()
    any_supported = len(d) == 2 and d["supported"].any()

    if same_direction and both_supported:
        verdict = "REPLICATES"
    elif same_direction and any_supported:
        verdict = "PARTIAL"
    elif same_direction:
        verdict = "SAME DIRECTION, UNCERTAIN"
    else:
        verdict = "DOES NOT REPLICATE"

    latent_summary.append({
        "term": term,
        "latent_cross_cycle_verdict": verdict,
        "latent_same_direction": same_direction,
        "latent_both_supported": both_supported
    })

latent_summary = pd.DataFrame(latent_summary)

rows = []

for term in ["A", "G", "AG"]:
    s = survey[survey["term"] == term].iloc[0]
    l = latent_summary[latent_summary["term"] == term].iloc[0]

    latent_term = latent[latent["term"] == term]
    latent_direction = int(np.sign(latent_term["effect"].mean()))

    concordant = (
        int(np.sign(s["effect"])) == latent_direction
    )

    rows.append({
        "term": term,
        "survey_direction": int(np.sign(s["effect"])),
        "survey_supported": bool(s["supported"]),
        "latent_direction": latent_direction,
        "latent_verdict": l["latent_cross_cycle_verdict"],
        "direction_concordant": concordant
    })

comparison = pd.DataFrame(rows)
comparison.to_csv(
    RESULTS / "22_method_convergence.csv",
    index=False
)

plot = comparison.copy()
plot["survey_score"] = plot["survey_supported"].astype(int)
plot["latent_score"] = plot["latent_verdict"].map({
    "REPLICATES": 2,
    "PARTIAL": 1,
    "SAME DIRECTION, UNCERTAIN": 0,
    "DOES NOT REPLICATE": -1
})

x = np.arange(len(plot))
w = 0.34

plt.figure(figsize=(8, 5))
plt.bar(x - w/2, plot["survey_score"], width=w, label="Survey observed score")
plt.bar(x + w/2, plot["latent_score"], width=w, label="Ordinal latent factor")
plt.xticks(x, ["A", "G", "A×G"])
plt.yticks(
    [-1, 0, 1, 2],
    ["Conflict", "Uncertain", "Supported / partial", "Replicates"]
)
plt.ylabel("Evidence status")
plt.title("Convergence across independent analysis routes")
plt.legend()
plt.tight_layout()
plt.savefig(
    FIG / "22_method_convergence.png",
    dpi=300
)
plt.close()

print()
print("METHOD CONVERGENCE AUDIT")
print("========================")

for row in comparison.itertuples():
    label = "A×G" if row.term == "AG" else row.term

    if (
        row.direction_concordant
        and row.survey_supported
        and row.latent_verdict == "REPLICATES"
    ):
        verdict = "STRONG CONVERGENCE"
    elif row.direction_concordant and row.survey_supported:
        verdict = "CONVERGENT, NOT FULLY REPLICATED"
    elif row.direction_concordant:
        verdict = "DIRECTIONALLY CONSISTENT"
    else:
        verdict = "METHOD CONFLICT"

    print(f"{label:<4}  {verdict}")

print()
print("INTERPRETATION")
print("A and G are the primary robust findings if both show strong convergence.")
print("A×G remains exploratory unless it independently replicates across representations and methods.")
print()
print("Comparison table and figure saved.")
