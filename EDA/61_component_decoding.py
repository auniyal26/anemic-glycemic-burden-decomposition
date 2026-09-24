#!/usr/bin/env python
# =============================================================================
# 61_component_decoding.py
#
# Goal:
#   Decode what the frozen haematology PCs represent physiologically.
#
# Uses:
#   EDA/Results/60_nonanaemic_model_input.csv
#   EDA/Results/60_nonanaemic_coefficients.csv
#
# No PCA refit. No locked outputs modified.
# =============================================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "EDA" / "Results"
FIG = ROOT / "EDA" / "Figures"
FIG.mkdir(parents=True, exist_ok=True)

INPUT = RES / "60_nonanaemic_model_input.csv"
COEF = RES / "60_nonanaemic_coefficients.csv"

PERIODS = ["2005-2008", "2009-2018", "2021-2023"]

PCS = ["A_OI_PC1_FZ", "A_OI_PC2_FZ", "A_OI_PC3_FZ"]
BIOMARKERS = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"]

DISPLAY = {
    "A_OI_PC1_FZ": "Haematology PC1",
    "A_OI_PC2_FZ": "Haematology PC2",
    "A_OI_PC3_FZ": "Haematology PC3",
    "LBXHGB": "Haemoglobin",
    "LBXRBCSI": "RBC count",
    "LBXMCVSI": "MCV",
    "LBXRDW": "RDW",
}


def weighted_mean(x, w):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    w = pd.to_numeric(pd.Series(w), errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    return np.sum(x[ok] * w[ok]) / np.sum(w[ok])


def weighted_corr(x, y, w):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    w = pd.to_numeric(pd.Series(w), errors="coerce").to_numpy(float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return np.nan

    x, y, w = x[ok], y[ok], w[ok]
    sw = w.sum()

    mx = np.sum(w * x) / sw
    my = np.sum(w * y) / sw

    cov = np.sum(w * (x - mx) * (y - my)) / sw
    vx = np.sum(w * (x - mx) ** 2) / sw
    vy = np.sum(w * (y - my) ** 2) / sw

    if vx <= 0 or vy <= 0:
        return np.nan

    return cov / np.sqrt(vx * vy)


def weighted_quantile(x, w, probs):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    w = pd.to_numeric(pd.Series(w), errors="coerce").to_numpy(float)

    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]

    if len(x) == 0:
        return np.repeat(np.nan, len(probs))

    idx = np.argsort(x)
    x, w = x[idx], w[idx]

    cdf = np.cumsum(w) / np.sum(w)
    return np.array([np.interp(p, cdf, x) for p in probs], dtype=float)


def weighted_quintile(x, w):
    cuts = weighted_quantile(x, w, [0.2, 0.4, 0.6, 0.8])

    # Protect against tied cutpoints.
    cuts = np.unique(cuts[np.isfinite(cuts)])

    if len(cuts) < 4:
        return pd.Series(
            pd.qcut(pd.Series(x), 5, labels=False, duplicates="drop") + 1,
            index=pd.Series(x).index
        )

    vals = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    q = np.digitize(vals, cuts, right=True) + 1
    q[~np.isfinite(vals)] = 0

    out = pd.Series(q, index=pd.Series(x).index)
    out = out.replace(0, np.nan)
    return out


if not INPUT.exists():
    raise FileNotFoundError(
        f"Missing {INPUT}. Run EDA/60_nonanaemic_signal_audit.py first."
    )

d = pd.read_csv(INPUT)

required = (
    ["PERIOD", "SURVEY_WT", "NONANAEMIC", "SOMATIC_SCORE", "PHQ9_TOTAL"]
    + PCS + BIOMARKERS
)

missing = [c for c in required if c not in d.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")


# -----------------------------------------------------------------------------
# 1. Weighted PC ↔ biomarker correlation profiles
# -----------------------------------------------------------------------------

rows = []

for period in PERIODS:
    q = d[
        d["PERIOD"].eq(period)
        & d["DOMAIN_SOMATIC_SCORE_X3"].eq(1)
    ].copy()

    for subgroup, z in [
        ("All analytic participants", q),
        ("Non-anaemic only (A = 0)", q[q["NONANAEMIC"].eq(1)].copy()),
    ]:
        for pc in PCS:
            for marker in BIOMARKERS:
                rows.append({
                    "period": period,
                    "subgroup": subgroup,
                    "component": pc,
                    "component_label": DISPLAY[pc],
                    "biomarker": marker,
                    "biomarker_label": DISPLAY[marker],
                    "n_complete": int(
                        z[[pc, marker, "SURVEY_WT"]].dropna().shape[0]
                    ),
                    "weighted_correlation": weighted_corr(
                        z[pc], z[marker], z["SURVEY_WT"]
                    ),
                })

profiles = pd.DataFrame(rows)
profiles.to_csv(
    RES / "61_component_biomarker_correlation_profiles.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# 2. Candidate physiological anchors
# -----------------------------------------------------------------------------

anchor_rows = []

for (period, subgroup, pc), q in profiles.groupby(
    ["period", "subgroup", "component"]
):
    q = q.copy()
    q["abs_r"] = q["weighted_correlation"].abs()
    q = q.sort_values("abs_r", ascending=False)

    top = q.iloc[0]
    second = q.iloc[1]

    anchor_rows.append({
        "period": period,
        "subgroup": subgroup,
        "component": pc,
        "component_label": DISPLAY[pc],
        "strongest_anchor": top["biomarker_label"],
        "strongest_anchor_r": top["weighted_correlation"],
        "second_anchor": second["biomarker_label"],
        "second_anchor_r": second["weighted_correlation"],
        "interpretation_status":
            "descriptive physiological anchor; not mechanistic identity",
    })

anchors = pd.DataFrame(anchor_rows)
anchors.to_csv(
    RES / "61_component_candidate_anchors.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# 3. Correlation-profile stability across time
# -----------------------------------------------------------------------------

stability_rows = []

for subgroup in profiles["subgroup"].unique():
    for pc in PCS:
        vectors = {}

        for period in PERIODS:
            q = profiles[
                profiles["period"].eq(period)
                & profiles["subgroup"].eq(subgroup)
                & profiles["component"].eq(pc)
            ].set_index("biomarker").reindex(BIOMARKERS)

            vectors[period] = q["weighted_correlation"].to_numpy(float)

        ref = vectors["2005-2008"]

        for period in ["2009-2018", "2021-2023"]:
            x = ref
            y = vectors[period]

            ok = np.isfinite(x) & np.isfinite(y)

            if ok.sum() < 2:
                similarity = np.nan
            else:
                denom = np.linalg.norm(x[ok]) * np.linalg.norm(y[ok])
                similarity = (
                    np.dot(x[ok], y[ok]) / denom
                    if denom > 0 else np.nan
                )

            stability_rows.append({
                "subgroup": subgroup,
                "component": pc,
                "component_label": DISPLAY[pc],
                "reference_period": "2005-2008",
                "comparison_period": period,
                "correlation_profile_cosine_similarity": similarity,
            })

stability = pd.DataFrame(stability_rows)
stability.to_csv(
    RES / "61_component_profile_stability.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# 4. Quintile decoding inside non-anaemic participants
# -----------------------------------------------------------------------------

quintile_rows = []

for period in PERIODS:
    q = d[
        d["PERIOD"].eq(period)
        & d["DOMAIN_SOMATIC_SCORE_X3"].eq(1)
        & d["NONANAEMIC"].eq(1)
    ].copy()

    for pc in PCS:
        q[f"{pc}_Q"] = weighted_quintile(q[pc], q["SURVEY_WT"])

        for quintile in sorted(q[f"{pc}_Q"].dropna().unique()):
            z = q[q[f"{pc}_Q"].eq(quintile)].copy()

            quintile_rows.append({
                "period": period,
                "component": pc,
                "component_label": DISPLAY[pc],
                "quintile": int(quintile),
                "n": len(z),
                "weighted_mean_component":
                    weighted_mean(z[pc], z["SURVEY_WT"]),
                "weighted_mean_Hb":
                    weighted_mean(z["LBXHGB"], z["SURVEY_WT"]),
                "weighted_mean_RBC":
                    weighted_mean(z["LBXRBCSI"], z["SURVEY_WT"]),
                "weighted_mean_MCV":
                    weighted_mean(z["LBXMCVSI"], z["SURVEY_WT"]),
                "weighted_mean_RDW":
                    weighted_mean(z["LBXRDW"], z["SURVEY_WT"]),
                "weighted_mean_somatic_PHQ":
                    weighted_mean(z["SOMATIC_SCORE"], z["SURVEY_WT"]),
                "weighted_mean_total_PHQ9":
                    weighted_mean(z["PHQ9_TOTAL"], z["SURVEY_WT"]),
            })

quintiles = pd.DataFrame(quintile_rows)
quintiles.to_csv(
    RES / "61_component_quintile_profiles.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# 5. Pull forward Script 60 phenotype coefficients for interpretation
# -----------------------------------------------------------------------------

if COEF.exists():
    coef = pd.read_csv(COEF)

    keep = coef[
        coef["model"].isin(
            [
                "A_PCs_nonanaemic",
                "raw_reduced_A_nonanaemic",
            ]
        )
    ].copy()

    keep.to_csv(
        RES / "61_component_outcome_links_from_script60.csv",
        index=False,
    )


# -----------------------------------------------------------------------------
# 6. Figures — explicit labels
# -----------------------------------------------------------------------------

for period in PERIODS:
    q = profiles[
        profiles["period"].eq(period)
        & profiles["subgroup"].eq("Non-anaemic only (A = 0)")
    ].copy()

    mat = (
        q.pivot(
            index="component_label",
            columns="biomarker_label",
            values="weighted_correlation",
        )
        .reindex(
            ["Haematology PC1", "Haematology PC2", "Haematology PC3"]
        )
        .reindex(
            columns=["Haemoglobin", "RBC count", "MCV", "RDW"]
        )
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(mat.to_numpy(float), vmin=-1, vmax=1)

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)

    ax.set_xlabel("Measured haematological variable")
    ax.set_ylabel("Frozen haematological component")
    ax.set_title(
        f"Component–biomarker correlations among non-anaemic participants: {period}"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Survey-weighted correlation")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if pd.notna(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                )

    fig.tight_layout()
    fig.savefig(
        FIG / f"61_component_biomarker_profile_{period}.png",
        dpi=300,
    )
    plt.close(fig)


# PC3 ↔ RDW gradient
q = quintiles[
    quintiles["component"].eq("A_OI_PC3_FZ")
].copy()

for period in PERIODS:
    z = q[q["period"].eq(period)].sort_values("quintile")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(
        z["quintile"],
        z["weighted_mean_RDW"],
        marker="o",
    )

    ax.set_xlabel("Haematology PC3 quintile among non-anaemic participants")
    ax.set_ylabel("Survey-weighted mean RDW (%)")
    ax.set_title(
        f"RDW across PC3 quintiles: {period}"
    )
    ax.set_xticks([1, 2, 3, 4, 5])

    fig.tight_layout()
    fig.savefig(
        FIG / f"61_pc3_quintile_RDW_{period}.png",
        dpi=300,
    )
    plt.close(fig)


# PC3 ↔ somatic symptom gradient
for period in PERIODS:
    z = q[q["period"].eq(period)].sort_values("quintile")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(
        z["quintile"],
        z["weighted_mean_somatic_PHQ"],
        marker="o",
    )

    ax.set_xlabel("Haematology PC3 quintile among non-anaemic participants")
    ax.set_ylabel("Survey-weighted mean somatic PHQ score (0–9)")
    ax.set_title(
        f"Somatic depressive symptoms across PC3 quintiles: {period}"
    )
    ax.set_xticks([1, 2, 3, 4, 5])

    fig.tight_layout()
    fig.savefig(
        FIG / f"61_pc3_quintile_somatic_PHQ_{period}.png",
        dpi=300,
    )
    plt.close(fig)


manifest = {
    "script": "61_component_decoding.py",
    "purpose":
        "Physiologically decode frozen haematology PC1-3 without refitting PCA.",
    "method":
        "Survey-weighted PC-to-Hb/RBC/MCV/RDW correlation profiles, "
        "cross-period profile stability, and weighted quintile gradients.",
    "primary_subgroup":
        "Non-anaemic participants with locked scalar A = 0",
    "interpretation_rule":
        "Correlation profiles are physiological anchors, not causal or mechanistic identities.",
    "locked_outputs_modified": False,
}

(RES / "61_component_decoding_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("PASS  Component decoding complete.")
print("PASS  PCA was NOT refit.")
print("PASS  PC1/PC2/PC3 mapped to Hb, RBC count, MCV and RDW.")
print("PASS  Non-anaemic decoding and cross-period stability quantified.")
print("PASS  PC3-RDW and PC3-somatic gradients generated.")
print("PASS  Outputs written only under EDA/Results and EDA/Figures.")
