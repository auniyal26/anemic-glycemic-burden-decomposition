from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
DATA = ROOT / "Data" / "NHANES"
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
FIG = RESULTS / "Figures"
AUDIT = ROOT / "Audit"

FIG.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)

CYCLES = {
    "0506": {
        "cbc": "CBC_D.XPT",
        "ghb": "GHB_D.XPT",
        "glu": "GLU_D.XPT"
    },
    "0708": {
        "cbc": "CBC_E.XPT",
        "ghb": "GHB_E.XPT",
        "glu": "GLU_E.XPT"
    }
}

A_VARS = [
    "LBXHGB",
    "LBXHCT",
    "LBXRBCSI",
    "LBXMCVSI",
    "LBXMCHSI",
    "LBXMC",
    "LBXRDW"
]

A_LABELS = {
    "LBXHGB": "Hemoglobin",
    "LBXHCT": "Hematocrit",
    "LBXRBCSI": "RBC count",
    "LBXMCVSI": "MCV",
    "LBXMCHSI": "MCH",
    "LBXMC": "MCHC",
    "LBXRDW": "RDW"
}

G_VARS = ["LBXGH", "LBXGLU"]

G_LABELS = {
    "LBXGH": "HbA1c",
    "LBXGLU": "Fasting glucose"
}

N_BOOT = 200
RNG = np.random.default_rng(42)

print()
print("INTERNAL A/G BURDEN DECOMPOSITION")
print("=================================")

for cycle, files in CYCLES.items():
    for key, filename in files.items():
        path = DATA / cycle / filename
        if not path.exists():
            print(f"FAIL  Missing {path}")
            raise SystemExit(1)

cohort = pd.read_parquet(PROCESSED / "nhanes_adjustment_cohort_20plus.parquet").copy()
cohort["CYCLE"] = cohort["CYCLE"].astype(str).replace({"506": "0506", "708": "0708"})

needed_proxy = ["SEQN", "CYCLE"]
for c in ["A", "G_HBA1C"]:
    if c in cohort.columns:
        needed_proxy.append(c)

cohort = cohort[needed_proxy].copy()

a_frames = []
g_frames = []

for cycle, files in CYCLES.items():
    cbc = pd.read_sas(DATA / cycle / files["cbc"], format="xport")
    ghb = pd.read_sas(DATA / cycle / files["ghb"], format="xport")
    glu = pd.read_sas(DATA / cycle / files["glu"], format="xport")

    missing_a = [c for c in ["SEQN"] + A_VARS if c not in cbc.columns]
    missing_g1 = [c for c in ["SEQN", "LBXGH"] if c not in ghb.columns]
    missing_g2 = [c for c in ["SEQN", "LBXGLU"] if c not in glu.columns]

    if missing_a or missing_g1 or missing_g2:
        print("FAIL  Required variables missing.")
        print("A:", missing_a)
        print("GHB:", missing_g1)
        print("GLU:", missing_g2)
        raise SystemExit(1)

    a = cbc[["SEQN"] + A_VARS].copy()
    a["CYCLE"] = cycle
    a_frames.append(a)

    g = ghb[["SEQN", "LBXGH"]].merge(
        glu[["SEQN", "LBXGLU"]],
        on="SEQN",
        how="inner",
        validate="one_to_one"
    )
    g["CYCLE"] = cycle
    g_frames.append(g)

a_raw = pd.concat(a_frames, ignore_index=True)
g_raw = pd.concat(g_frames, ignore_index=True)

a_raw = cohort.merge(a_raw, on=["SEQN", "CYCLE"], how="inner", validate="one_to_one")
g_raw = cohort.merge(g_raw, on=["SEQN", "CYCLE"], how="inner", validate="one_to_one")

a_raw = a_raw.dropna(subset=A_VARS).copy()
g_raw = g_raw.dropna(subset=G_VARS).copy()

def fit_cycle_pca(data, variables):
    scaler = StandardScaler()
    z = scaler.fit_transform(data[variables].astype(float))
    pca = PCA()
    scores = pca.fit_transform(z)
    return scaler, pca, z, scores

def align_to_reference(ref_loadings, target_loadings):
    sim = np.abs(ref_loadings.T @ target_loadings)
    row, col = linear_sum_assignment(-sim)
    order = col[np.argsort(row)]
    aligned = target_loadings[:, order].copy()
    signs = np.sign(np.sum(ref_loadings * aligned, axis=0))
    signs[signs == 0] = 1
    aligned *= signs
    similarities = np.abs(np.sum(ref_loadings * aligned, axis=0))
    return order, signs, aligned, similarities

def bootstrap_stability(data, variables, ref_loadings, n_boot, desc):
    sims = []
    loadings_acc = []

    with tqdm(total=n_boot, desc=desc, unit="boot", dynamic_ncols=True) as bar:
        for _ in range(n_boot):
            idx = RNG.integers(0, len(data), len(data))
            sample = data.iloc[idx]

            scaler = StandardScaler()
            z = scaler.fit_transform(sample[variables].astype(float))
            pca = PCA()
            pca.fit(z)

            boot_loadings = pca.components_.T
            order, signs, aligned, similarity = align_to_reference(ref_loadings, boot_loadings)
            sims.append(similarity)
            loadings_acc.append(aligned)
            bar.update(1)

    return np.vstack(sims), np.stack(loadings_acc)

def proxy_correlations(scores, data, proxy):
    out = []
    if proxy not in data.columns:
        return out

    valid_proxy = data[proxy].notna().values
    if valid_proxy.sum() < 10:
        return out

    y = data.loc[valid_proxy, proxy].astype(float).values

    for j in range(scores.shape[1]):
        x = scores[valid_proxy, j]
        r = np.corrcoef(x, y)[0, 1]
        out.append((j + 1, r))

    return out

a_models = {}
g_models = {}
alignment_rows = []
variance_rows = []
loading_rows = []
score_frames = []
stability_rows = []

for block, raw, variables, labels, proxy in [
    ("A", a_raw, A_VARS, A_LABELS, "A"),
    ("G", g_raw, G_VARS, G_LABELS, "G_HBA1C")
]:
    for cycle in ["0506", "0708"]:
        d = raw[raw["CYCLE"] == cycle].reset_index(drop=True)
        scaler, pca, z, scores = fit_cycle_pca(d, variables)

        model = {
            "data": d,
            "scaler": scaler,
            "pca": pca,
            "z": z,
            "scores": scores,
            "loadings": pca.components_.T.copy()
        }

        if block == "A":
            a_models[cycle] = model
        else:
            g_models[cycle] = model

        for j, evr in enumerate(pca.explained_variance_ratio_, start=1):
            variance_rows.append({
                "block": block,
                "cycle": cycle,
                "component": j,
                "explained_variance_ratio": evr,
                "cumulative_explained_variance": pca.explained_variance_ratio_[:j].sum()
            })

ref_sets = {
    "A": a_models,
    "G": g_models
}

for block, models in ref_sets.items():
    ref = models["0506"]["loadings"]
    target = models["0708"]["loadings"]

    order, signs, aligned, similarities = align_to_reference(ref, target)

    models["0708"]["aligned_loadings"] = aligned
    models["0708"]["aligned_scores"] = models["0708"]["scores"][:, order] * signs
    models["0506"]["aligned_loadings"] = ref
    models["0506"]["aligned_scores"] = models["0506"]["scores"]

    for j, sim in enumerate(similarities, start=1):
        alignment_rows.append({
            "block": block,
            "component": j,
            "cross_cycle_loading_similarity": sim,
            "0708_original_component": int(order[j - 1] + 1),
            "0708_sign": int(signs[j - 1])
        })

for block, models, variables, labels, proxy in [
    ("A", a_models, A_VARS, A_LABELS, "A"),
    ("G", g_models, G_VARS, G_LABELS, "G_HBA1C")
]:
    for cycle in ["0506", "0708"]:
        model = models[cycle]
        loadings = model["aligned_loadings"]
        scores = model["aligned_scores"]
        d = model["data"]

        for j in range(loadings.shape[1]):
            for i, var in enumerate(variables):
                loading_rows.append({
                    "block": block,
                    "cycle": cycle,
                    "component": j + 1,
                    "variable": var,
                    "label": labels[var],
                    "loading": loadings[i, j]
                })

        proxy_corrs = proxy_correlations(scores, d, proxy)
        proxy_map = {pc: r for pc, r in proxy_corrs}

        sf = d[["SEQN", "CYCLE"]].copy()
        for j in range(scores.shape[1]):
            sf[f"{block}_PC{j + 1}"] = scores[:, j]
            sf[f"{block}_PC{j + 1}_proxy_r"] = proxy_map.get(j + 1, np.nan)

        score_frames.append(sf)

        sims, boot_loadings = bootstrap_stability(
            d,
            variables,
            loadings,
            N_BOOT,
            f"{block} {cycle}"
        )

        for j in range(loadings.shape[1]):
            stability_rows.append({
                "block": block,
                "cycle": cycle,
                "component": j + 1,
                "bootstrap_mean_loading_similarity": sims[:, j].mean(),
                "bootstrap_p05_loading_similarity": np.quantile(sims[:, j], 0.05),
                "bootstrap_p95_loading_similarity": np.quantile(sims[:, j], 0.95)
            })

variance = pd.DataFrame(variance_rows)
loadings = pd.DataFrame(loading_rows)
alignment = pd.DataFrame(alignment_rows)
stability = pd.DataFrame(stability_rows)

variance.to_csv(RESULTS / "24_internal_burden_explained_variance.csv", index=False)
loadings.to_csv(RESULTS / "24_internal_burden_loadings.csv", index=False)
alignment.to_csv(RESULTS / "24_internal_burden_cross_cycle_alignment.csv", index=False)
stability.to_csv(RESULTS / "24_internal_burden_bootstrap_stability.csv", index=False)

a_scores = pd.concat([x for x in score_frames if any(c.startswith("A_PC") for c in x.columns)], ignore_index=True)
g_scores = pd.concat([x for x in score_frames if any(c.startswith("G_PC") for c in x.columns)], ignore_index=True)

scores = a_scores.merge(
    g_scores,
    on=["SEQN", "CYCLE"],
    how="outer"
)

scores.to_parquet(PROCESSED / "24_internal_burden_scores.parquet", index=False)

for block, variables, labels in [
    ("A", A_VARS, A_LABELS),
    ("G", G_VARS, G_LABELS)
]:
    subset = loadings[(loadings["block"] == block) & (loadings["component"] <= min(3, len(variables)))]

    for component in sorted(subset["component"].unique()):
        d = subset[subset["component"] == component]

        plt.figure(figsize=(9, 5))

        positions = np.arange(len(variables))
        width = 0.35

        for shift, cycle in [(-width / 2, "0506"), (width / 2, "0708")]:
            z = d[d["cycle"] == cycle].set_index("variable").reindex(variables)
            plt.bar(
                positions + shift,
                z["loading"],
                width=width,
                label=cycle
            )

        plt.axhline(0, linewidth=0.8)
        plt.xticks(
            positions,
            [labels[v] for v in variables],
            rotation=35,
            ha="right"
        )
        plt.ylabel("Aligned PCA loading")
        plt.title(f"{block} internal composition — PC{component}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            FIG / f"24_{block.lower()}_pc{component}_composition.png",
            dpi=300
        )
        plt.close()

summary = {
    "A_complete_n": {
        cycle: int(len(a_models[cycle]["data"]))
        for cycle in ["0506", "0708"]
    },
    "G_fasting_complete_n": {
        cycle: int(len(g_models[cycle]["data"]))
        for cycle in ["0506", "0708"]
    },
    "bootstrap_replicates_per_cycle": N_BOOT,
    "A_variables": A_VARS,
    "G_variables": G_VARS
}

with open(AUDIT / "24_internal_burden_decomposition.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

def print_block(block, models, proxy):
    print()
    print(f"{block} COMPOSITION")

    align = alignment[alignment["block"] == block]
    for row in align.itertuples():
        if row.component > 3:
            continue

        evr_0506 = variance[
            (variance["block"] == block)
            & (variance["cycle"] == "0506")
            & (variance["component"] == row.component)
        ]["explained_variance_ratio"].iloc[0]

        evr_0708 = variance[
            (variance["block"] == block)
            & (variance["cycle"] == "0708")
            & (variance["component"] == row.component)
        ]["explained_variance_ratio"].iloc[0]

        print(
            f"PC{row.component}: "
            f"variance {evr_0506:.3f}/{evr_0708:.3f}, "
            f"cross-cycle similarity {row.cross_cycle_loading_similarity:.3f}"
        )

    for cycle in ["0506", "0708"]:
        d = models[cycle]["data"]
        scores = models[cycle]["aligned_scores"]
        corrs = proxy_correlations(scores, d, proxy)

        if corrs:
            best_pc, best_r = max(corrs, key=lambda x: abs(x[1]))
            print(
                f"{cycle} closest to current {proxy} proxy: "
                f"PC{best_pc} r={best_r:.3f}"
            )

print(f"PASS  A complete CBC decomposition: n={len(a_raw):,}.")
print(f"PASS  G fasting biomarker decomposition: n={len(g_raw):,}.")
print(f"PASS  Independent-cycle PCA with cross-cycle component alignment.")
print(f"PASS  {N_BOOT} bootstrap stability checks per block and cycle.")

print_block("A", a_models, "A")
print_block("G", g_models, "G_HBA1C")

print()
print("INTERPRETATION RULE")
print("Do not call any PC the final A or G yet.")
print("A stable cross-cycle component that tracks the existing proxy is a candidate burden axis.")
print("Other stable components represent internal physiology that the current one-number proxy discards.")
print()
print("Scores, loadings, stability tables and figures saved.")
print("NEXT  Regress the stable internal A/G components on the validated somatic latent phenotype.")
