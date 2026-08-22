from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
RESULTS = ROOT / "Results"
FIG = RESULTS / "Figures"
FIG.mkdir(exist_ok=True)

labels = {
    "DPQ010": "Anhedonia",
    "DPQ020": "Depressed mood",
    "DPQ030": "Sleep",
    "DPQ040": "Fatigue",
    "DPQ050": "Appetite",
    "DPQ060": "Self-worth",
    "DPQ070": "Concentration",
    "DPQ080": "Psychomotor",
    "DPQ090": "Suicidal ideation"
}

groups = pd.read_csv(RESULTS / "04_clinical_groups.csv")
mapping = pd.read_csv(RESULTS / "11_cross_cycle_component_mapping.csv")
validation = pd.read_csv(RESULTS / "11_aligned_physiological_validation.csv")
l05 = pd.read_csv(RESULTS / "08_loadings_0506.csv", index_col=0)
l07 = pd.read_csv(RESULTS / "08_loadings_0708_aligned.csv", index_col=0)
stability = pd.read_csv(RESULTS / "10_bootstrap_structural_summary.csv")

group_names = {
    "0_0": "Neither",
    "0_1": "Glycemia only",
    "1_0": "Anemia only",
    "1_1": "Both"
}

groups["group_name"] = groups["group"].map(group_names)

plt.figure(figsize=(8, 5))
plt.bar(groups["group_name"], groups["mean_phq9"])
plt.ylabel("Weighted mean PHQ-9")
plt.title("Depressive symptom burden by physiological state")
plt.tight_layout()
plt.savefig(FIG / "12_clinical_groups_phq9.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.bar(mapping["aligned_component"], mapping["similarity"])
plt.axhline(0.8, linestyle="--", linewidth=1)
plt.ylim(0, 1)
plt.ylabel("Cross-cycle loading similarity")
plt.title("Independent latent-component replication")
plt.tight_layout()
plt.savefig(FIG / "12_cross_cycle_similarity.png", dpi=300)
plt.close()

ic = "IC3"

load = pd.DataFrame({
    "item": l05.index,
    "0506": l05[ic].values,
    "0708": l07[ic].values
})

load["label"] = load["item"].map(labels)

x = np.arange(len(load))
w = 0.38

plt.figure(figsize=(10, 5))
plt.bar(x - w/2, load["0506"], width=w, label="2005–06")
plt.bar(x + w/2, load["0708"], width=w, label="2007–08")
plt.axhline(0, linewidth=0.8)
plt.xticks(x, load["label"], rotation=40, ha="right")
plt.ylabel("ICA loading")
plt.title("Replicated latent depressive phenotype")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "12_ic3_item_loadings.png", dpi=300)
plt.close()

v = validation[
    (validation["component"] == "IC3") &
    (validation["variable"].isin(["A", "G"]))
].copy()

rows = []

for _, r in v.iterrows():
    rows.append({
        "variable": r["variable"],
        "cycle": "2005–06",
        "mean": r["mean_0506"],
        "low": r["ci05_low"],
        "high": r["ci05_high"]
    })
    rows.append({
        "variable": r["variable"],
        "cycle": "2007–08",
        "mean": r["mean_0708"],
        "low": r["ci07_low"],
        "high": r["ci07_high"]
    })

effects = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(8, 5))

positions = {
    ("A", "2005–06"): 0,
    ("A", "2007–08"): 1,
    ("G", "2005–06"): 3,
    ("G", "2007–08"): 4
}

for _, r in effects.iterrows():
    pos = positions[(r["variable"], r["cycle"])]
    ax.errorbar(
        pos,
        r["mean"],
        yerr=[[r["mean"] - r["low"]], [r["high"] - r["mean"]]],
        fmt="o",
        capsize=5
    )

ax.axhline(0, linewidth=0.8)
ax.set_xticks([0, 1, 3, 4])
ax.set_xticklabels([
    "Anemia\n05–06",
    "Anemia\n07–08",
    "Glycemia\n05–06",
    "Glycemia\n07–08"
])
ax.set_ylabel("Aligned latent-component association")
ax.set_title("Physiological associations with replicated IC3")
plt.tight_layout()
plt.savefig(FIG / "12_ic3_physiological_effects.png", dpi=300)
plt.close()

stab = stability[stability["component"] == "IC3"]

stable = (stab["median"] > 0.8).all()
cross = mapping.loc[
    mapping["aligned_component"] == "IC3",
    "similarity"
].iloc[0]

a = validation[
    (validation["component"] == "IC3") &
    (validation["variable"] == "A")
].iloc[0]

g = validation[
    (validation["component"] == "IC3") &
    (validation["variable"] == "G")
].iloc[0]

print()
print("THEORY VALIDATION STATUS")
print("------------------------")

if stable:
    print("PASS  Latent structure survives participant resampling.")
else:
    print("CAUTION  Latent structure is sampling-sensitive.")

if cross >= 0.9:
    print("PASS  The same latent phenotype independently appears in both cycles.")
elif cross >= 0.8:
    print("PARTIAL  Cross-cycle phenotype replication is reasonable.")
else:
    print("CAUTION  Cross-cycle phenotype replication is weak.")

if a["0506_nonzero"] and a["0708_nonzero"]:
    print("PASS  Anemia association independently replicates.")
else:
    print("PARTIAL  Anemia direction is consistent but not fully replicated.")

if g["same_direction"] and (g["0506_nonzero"] or g["0708_nonzero"]):
    print("PARTIAL  Glycemic association is directionally consistent but weaker.")
else:
    print("NO CLEAR SIGNAL  Glycemic latent association is not yet stable.")

print("NO SUPPORT  A×G interaction is not currently reproducible.")
print()
print("Figures saved to Results/Figures.")