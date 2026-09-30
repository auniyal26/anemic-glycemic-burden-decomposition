from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()

CANDIDATES = [
    ROOT / "EDA" / "Results" / "72_shared_private_model_input.csv",
    ROOT / "72_shared_private_model_input.csv",
]
INPUT = next((p for p in CANDIDATES if p.exists()), None)

if INPUT is None:
    raise FileNotFoundError(
        "Could not find 72_shared_private_model_input.csv. "
        "Run this from the repository root after Script 72."
    )

df = pd.read_csv(INPUT)

print("=" * 68)
print("SCRIPT 72 DIAGNOSTIC")
print("=" * 68)
print(f"Input: {INPUT}")
print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
print()

# ---------------------------------------------------------------------
# 1. Outcome audit
# ---------------------------------------------------------------------
print("1) OUTCOME AUDIT")
print("-" * 68)

candidate_outcomes = [
    "DOMAIN_SOMATIC_SCORE_X3",
    "DOMAIN_COGAFF_SUM_X3",
    "DOMAIN_PHQ9_TOTAL_X3",
    "SOMATIC_SCORE",
    "COGAFF_SUM",
    "PHQ9_TOTAL",
]

for c in candidate_outcomes:
    if c not in df.columns:
        print(f"{c:28s} MISSING")
        continue

    x = pd.to_numeric(df[c], errors="coerce")
    vals = np.sort(x.dropna().unique())
    is_binary = len(vals) <= 2 and set(vals).issubset({0, 1})

    print(
        f"{c:28s} "
        f"n={x.notna().sum():5d} "
        f"unique={len(vals):3d} "
        f"range=[{x.min():.3g}, {x.max():.3g}] "
        f"{'*** BINARY FLAG ***' if is_binary else ''}"
    )

print()

domain_cols = [
    c for c in [
        "DOMAIN_SOMATIC_SCORE_X3",
        "DOMAIN_COGAFF_SUM_X3",
        "DOMAIN_PHQ9_TOTAL_X3",
    ] if c in df.columns
]

for i in range(len(domain_cols)):
    for j in range(i + 1, len(domain_cols)):
        a, b = domain_cols[i], domain_cols[j]
        both = df[[a, b]].dropna()
        same = len(both) > 0 and np.array_equal(
            both[a].to_numpy(), both[b].to_numpy()
        )
        if same:
            print(f"WARNING: {a} and {b} are EXACT DUPLICATES.")

if all(c in df.columns for c in ["SOMATIC_SCORE", "COGAFF_SUM", "PHQ9_TOTAL"]):
    q = df[["SOMATIC_SCORE", "COGAFF_SUM", "PHQ9_TOTAL"]].dropna()
    if len(q):
        ok = np.isclose(
            q["SOMATIC_SCORE"] + q["COGAFF_SUM"],
            q["PHQ9_TOTAL"],
            atol=1e-10,
        )
        print(
            f"CHECK: SOMATIC_SCORE + COGAFF_SUM == PHQ9_TOTAL "
            f"for {ok.mean()*100:.2f}% of complete rows (n={len(q)})."
        )

print()
print("Recommended phenotype columns for regression:")
print("  SOMATIC     -> SOMATIC_SCORE")
print("  COGAFF      -> COGAFF_SUM")
print("  PHQ9_TOTAL  -> PHQ9_TOTAL")
print()

# ---------------------------------------------------------------------
# 2. Shared/private coordinate audit
# ---------------------------------------------------------------------
print("2) SHARED / DISCORDANCE COORDINATES")
print("-" * 68)

meaning = {
    "SP_SHARED1": "common A-G movement on coupled axis 1",
    "SP_SHARED2": "common A-G movement on coupled axis 2",
    "SP_DISCORD1": "A-vs-G mismatch on coupled axis 1",
    "SP_DISCORD2": "A-vs-G mismatch on coupled axis 2",
    "SP_A_PRIVATE3": "A-specific weak/unshared axis 3",
    "SP_A_PRIVATE4": "A-specific extra direction (A is 4D)",
    "SP_G_PRIVATE3": "G-specific weak/unshared axis 3",
}

for c, desc in meaning.items():
    if c not in df.columns:
        print(f"{c:18s} MISSING")
        continue
    x = pd.to_numeric(df[c], errors="coerce")
    print(
        f"{c:18s} mean={x.mean(): .4f} sd={x.std(): .4f} "
        f"n={x.notna().sum():5d} | {desc}"
    )

print()
print("NOTE: signs of latent/canonical axes are arbitrary; interpret patterns,")
print("      not 'positive = biologically good/bad' without decoding anchors.")
print()

# ---------------------------------------------------------------------
# 3. Covariate audit
# ---------------------------------------------------------------------
print("3) COVARIATE AUDIT")
print("-" * 68)

covariates = {
    "RIDAGEYR": "age in years",
    "RIAGENDR": "sex/gender category used by NHANES",
    "RACE": "race/ethnicity category",
    "INDFMPIR": "family income-to-poverty ratio",
    "EDUC3": "3-level education category",
    "SMOKING3": "3-level smoking category",
    "BMXBMI": "body-mass index",
    "EGFR_2021": "estimated kidney filtration (eGFR)",
    "CYCLE": "NHANES cycle fixed effect when multiple cycles are pooled",
}

categorical = {"RIAGENDR", "RACE", "EDUC3", "SMOKING3", "CYCLE"}

for c, desc in covariates.items():
    if c not in df.columns:
        print(f"{c:12s} MISSING | {desc}")
        continue

    x = df[c]
    if c in categorical:
        counts = x.value_counts(dropna=False).sort_index()
        vals = ", ".join(f"{k}:{v}" for k, v in counts.items())
        print(f"{c:12s} {desc}")
        print(f"             levels/counts -> {vals}")
    else:
        xn = pd.to_numeric(x, errors="coerce")
        print(
            f"{c:12s} {desc} | "
            f"n={xn.notna().sum():5d}, mean={xn.mean():.3f}, "
            f"sd={xn.std():.3f}, range=[{xn.min():.3f}, {xn.max():.3f}]"
        )

print()

# ---------------------------------------------------------------------
# 4. Period-specific phenotype availability
# ---------------------------------------------------------------------
print("4) PHENOTYPE AVAILABILITY BY PERIOD")
print("-" * 68)

if "PERIOD" in df.columns:
    for period, dp in df.groupby("PERIOD", dropna=False):
        print(f"{period}:")
        for c in ["SOMATIC_SCORE", "COGAFF_SUM", "PHQ9_TOTAL"]:
            if c in dp.columns:
                print(f"  {c:16s} n={dp[c].notna().sum():5d}")
else:
    print("PERIOD column missing.")

print()
print("=" * 68)

bad_domain = False
for c in domain_cols:
    vals = set(pd.to_numeric(df[c], errors="coerce").dropna().unique())
    if vals.issubset({0, 1}):
        bad_domain = True

if bad_domain:
    print("DIAGNOSTIC VERDICT:")
    print("The DOMAIN_*_X3 columns are binary domain/inclusion flags, NOT phenotype scores.")
    print("Therefore Script 72's physiology decomposition remains valid,")
    print("but its phenotype regression must be rerun using the actual score columns.")
else:
    print("DIAGNOSTIC VERDICT: no obvious binary-outcome misuse detected.")

print("=" * 68)
