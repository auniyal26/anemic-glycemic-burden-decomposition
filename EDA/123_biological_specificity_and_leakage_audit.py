
#!/usr/bin/env python3
"""
123 | Biological specificity and leakage audit

Primary objectives:
1. Independent laboratory anchor validation.
2. Conditional coordinate-group associations.
3. Survey-design-aware Wald inference.
4. Holm multiplicity correction.
5. Dimension-matched random rotation and PCA controls.
6. Frozen cross-cycle anchor prediction.
7. Original HG7 transformation reproduction.
8. Source provenance and participant separation.
9. Fold-specific representation-fitting sensitivity.

This is exploratory construct validation, not causal
identification or independent clinical validation.
"""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import traceback
import urllib.request

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import f as f_dist
from scipy.linalg import subspace_angles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm


# ============================================================
# Configuration
# ============================================================

EDA = Path(__file__).resolve().parent
ROOT = EDA.parent
R = EDA / "Results"

OUT = R / "123_biological_specificity"

SEED = 123

RAW = [
    "LBXHGB",
    "LBXRBCSI",
    "LBXMCVSI",
    "LBXRDW",
    "LBXGH",
    "LBXGLU",
    "LOG_IN",
]

Z = [
    "SP_SHARED1",
    "SP_SHARED2",
    "SP_DISCORD1",
    "SP_DISCORD2",
    "SP_A_PRIVATE3",
    "SP_A_PRIVATE4",
    "SP_G_PRIVATE3",
]

GROUPS = {
    "Shared": [0, 1],
    "Discordant": [2, 3],
    "H_private": [4, 5],
    "G_private": [6],
}

COLORS = [
    "#146F69",
    "#AC583C",
    "#47677E",
    "#8A9D91",
]

CYCLES = {
    "0506": ("2005", "D", "NHANES"),
    "0708": ("2007", "E", "NHANES"),
    "2123": ("2021", "L", "NHANES_Transfer"),
}

ANCHORS = {
    "Ferritin": {
        "file": "FERTIN",
        "variable": "LBXFER",
        "women_only": True,
    },
    "sTfR": {
        "file": "TFR",
        "variable": "LBXTFR",
        "women_only": True,
    },
    "CRP": {
        "file": "CRP",
        "variable": "LBXCRP",
        "women_only": False,
    },
    "OGTT": {
        "file": "OGTT",
        "variable": "LBXGLT",
        "women_only": False,
    },
}

BASE_COV = [
    "RIDAGEYR",
    "BMXBMI",
    "INDFMPIR",
]

EXTRA_NUM = ["EGFR_2021"]

EXTRA_CAT = [
    "RACE",
    "EDUC3",
    "SMOKING3",
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.dpi": 600,
})


# ============================================================
# Utilities
# ============================================================

def dump(path, obj):

    path.write_text(
        json.dumps(obj, indent=2, default=str),
        encoding="utf-8",
    )


def sha(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1 << 20), b""
        ):

            h.update(block)

    return h.hexdigest()


def load_module(filename, name):

    spec = importlib.util.spec_from_file_location(
        name, EDA / filename
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


def cycle_string(s):

    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(4)
    )


def wm(x, w):

    return np.average(x, axis=0, weights=w)


def mse(y, pred, w):

    return float(wm((y - pred) ** 2, w))


def variance(y, w):

    return float(wm((y - wm(y, w)) ** 2, w))


def fit_wls(x, y, w):

    a = x * np.sqrt(w[:, None])

    b = y * np.sqrt(w)

    beta, _, rank, _ = np.linalg.lstsq(
        a, b, rcond=1e-12
    )

    if rank != x.shape[1]:

        raise ValueError(
            f"Rank deficiency: {rank}/{x.shape[1]}"
        )

    return beta


def holm(p):

    # Missing prespecified tests remain in the family.
    p = np.asarray(p, float)

    valid = np.isfinite(p)

    work = np.where(valid, p, 1.0)

    order = np.argsort(work)

    adjusted = np.empty(len(p))

    adjusted[order] = np.minimum(
        1.0,
        np.maximum.accumulate(
            work[order]
            * np.arange(len(p), 0, -1)
        ),
    )

    adjusted[~valid] = np.nan

    return adjusted


def similarity(a, b, w):

    a = (
        a - wm(a, w)
    ) * np.sqrt(w[:, None])

    b = (
        b - wm(b, w)
    ) * np.sqrt(w[:, None])

    if (
        np.linalg.matrix_rank(a) != a.shape[1]
        or np.linalg.matrix_rank(b) != b.shape[1]
    ):

        raise ValueError("Rank-deficient subspace")

    return float(np.mean(
        np.cos(subspace_angles(a, b)) ** 2
    ))


# ============================================================
# Official NHANES source files
# ============================================================

def read_xpt(cycle, kind, columns, fetch):

    year, suffix, folder = CYCLES[cycle]

    name = f"{kind}_{suffix}"

    directory = ROOT / "Data" / folder / cycle

    candidates = [
        directory / f"{name}.XPT",
        directory / f"{name}.xpt",
    ]

    path = next(
        (p for p in candidates if p.exists()),
        None,
    )

    if path is None:

        if not fetch:

            raise FileNotFoundError(candidates[0])

        path = candidates[0]

        path.parent.mkdir(
            parents=True, exist_ok=True
        )

        url = (
            "https://wwwn.cdc.gov/Nchs/Data/"
            f"Nhanes/Public/{year}/DataFiles/"
            f"{name}.XPT"
        )

        tmp = path.with_suffix(".download")

        print(f"Fetching {name}...", flush=True)

        try:

            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "NHANES-research/1.0"
                },
            )

            with urllib.request.urlopen(
                request, timeout=90
            ) as response:

                tmp.write_bytes(response.read())

            check = pd.read_sas(
                tmp, format="xport"
            )

            if not set(columns).issubset(check.columns):

                raise ValueError(
                    f"Invalid CDC file: {name}"
                )

            tmp.replace(path)

        finally:

            tmp.unlink(missing_ok=True)

    q = pd.read_sas(path, format="xport")

    missing = set(columns) - set(q.columns)

    if missing:

        raise KeyError(
            f"{name}: missing {sorted(missing)}"
        )

    q = q[columns].copy()

    for c in columns:

        q[c] = pd.to_numeric(
            q[c], errors="coerce"
        )

    if q.SEQN.isna().any() or q.SEQN.duplicated().any():

        raise ValueError(
            f"Invalid participant IDs: {name}"
        )

    return q


# ============================================================
# Frozen HG7 transformation
# ============================================================

def frozen_scores(x, frozen):

    u = (
        (x[:, :4] - frozen["A_mean"])
        / frozen["A_sd"]
    ) @ frozen["WA"]

    v = (
        (x[:, 4:] - frozen["G_mean"])
        / frozen["G_sd"]
    ) @ frozen["WG"]

    rho = frozen["rho"][:2]

    shared = (
        u[:, :2] + v[:, :2]
    ) / np.sqrt(2 * (1 + rho))

    discordant = (
        u[:, :2] - v[:, :2]
    ) / np.sqrt(2 * (1 - rho))

    return np.column_stack([
        shared,
        discordant,
        u[:, 2:],
        v[:, 2],
    ])


def eligible(d):

    return (
        d.ELIGIBLE_ADULT_NONPREG.eq(1)
        & d.SURVEY_WT.gt(1e-12)
        & d[RAW + Z].notna().all(axis=1)
    )


def load_scaffold(path):

    d = pd.read_csv(
        path, low_memory=False
    )

    required = [
        "SEQN",
        "CYCLE",
        "PERIOD",
        "SURVEY_WT",
        "ELIGIBLE_ADULT_NONPREG",
        *RAW,
        *Z,
        *BASE_COV,
        "RIAGENDR",
    ]

    missing = set(required) - set(d.columns)

    if missing:

        raise KeyError(
            f"{path.name}: {sorted(missing)}"
        )

    d.CYCLE = cycle_string(d.CYCLE)

    for c in [
        "SEQN", "SURVEY_WT",
        "ELIGIBLE_ADULT_NONPREG",
        *RAW, *Z, *BASE_COV,
        "RIAGENDR", *EXTRA_NUM, *EXTRA_CAT,
    ]:

        if c in d.columns:

            d[c] = pd.to_numeric(
                d[c], errors="coerce"
            )

    d = d.replace(
        [np.inf, -np.inf], np.nan
    )

    if d.SEQN.isna().any() or d.SEQN.duplicated().any():

        raise ValueError(
            f"Duplicate or invalid IDs: {path.name}"
        )

    return d


# ============================================================
# Leakage and provenance audit
# ============================================================

def audit_provenance(d72, d73, frozen, s104):

    rows = []

    def record(test, status, evidence):

        rows.append({
            "test": test,
            "status": status,
            "evidence": str(evidence),
        })

    for name, d in [
        ("Script72", d72),
        ("Script73", d73),
    ]:

        q = d.loc[eligible(d)]

        predicted = frozen_scores(
            q[RAW].to_numpy(float),
            frozen,
        )

        error = np.max(np.abs(
            predicted - q[Z].to_numpy(float)
        ))

        record(
            f"{name}_frozen_score_reproduction",
            "PASS" if error < 1e-7 else "FAIL",
            error,
        )

        if error >= 1e-7:

            raise AssertionError(
                f"Frozen score mismatch: {name}"
            )

    # Reproduce original outcome-independent map.
    discovery = d72.loc[
        d72.PERIOD.eq("2005-2008")
        & eligible(d72)
    ].copy()

    x = discovery[RAW].to_numpy(float)
    z = discovery[Z].to_numpy(float)

    w = discovery.SURVEY_WT.to_numpy(float)

    W, mu, rho = s104.fit_hg_cca(x, w)

    refit = (x - mu) @ W

    rho_error = np.max(
        np.abs(rho - frozen["rho"])
    )

    record(
        "discovery_sample_size",
        "PASS" if len(discovery) == 4088 else "FAIL",
        len(discovery),
    )

    record(
        "discovery_canonical_correlation_reproduction",
        "PASS" if rho_error < 1e-7 else "FAIL",
        rho_error,
    )

    for group, ix in GROUPS.items():

        sim = similarity(
            z[:, ix], refit[:, ix], w
        )

        record(
            f"discovery_subspace_{group}",
            "PASS" if sim > 1 - 1e-8 else "FAIL",
            sim,
        )

    # Ensure no cross-period participant overlap.
    early = set(
        d73.loc[
            d73.PERIOD.eq("2005-2008"), "SEQN"
        ]
    )

    for period in ["2009-2018", "2021-2023"]:

        later = set(
            d73.loc[
                d73.PERIOD.eq(period), "SEQN"
            ]
        )

        overlap = len(early & later)

        record(
            f"participant_overlap_{period}",
            "PASS" if overlap == 0 else "FAIL",
            overlap,
        )

    # Compare archived Script118 source hashes.
    protocol = (
        R / "118_biomedical_utility"
        / "118_protocol.json"
    )

    if protocol.exists():

        old = json.loads(
            protocol.read_text(encoding="utf-8")
        )

        for name, expected in old[
            "source_hashes"
        ].items():

            path = EDA / name

            if not path.exists():

                record(
                    f"hash_{name}",
                    "UNAVAILABLE",
                    "Source missing",
                )

                continue

            actual = sha(path)

            record(
                f"hash_{name}",
                "PASS" if actual == expected
                else "REVIEW",
                actual,
            )

    record(
        "direct_outcome_input",
        "PASS",
        "Independent refit uses only seven raw "
        "markers and survey weights.",
    )

    record(
        "prior_outcome_informed_method_development",
        "LIMITATION",
        "Earlier exploratory feature and method "
        "selection cannot be made outcome-blind "
        "retrospectively.",
    )

    record(
        "untouched_external_population",
        "LIMITATION",
        "All NHANES periods were previously examined.",
    )

    record(
        "unmeasured_confounding",
        "UNTESTABLE",
        "Cross-sectional observational design.",
    )

    record(
        "complete_case_selection",
        "LIMITATION",
        "Inherited eligible laboratory scaffold.",
    )

    record(
        "assay_harmonisation",
        "LIMITATION",
        "Historical and modern assay methods "
        "are not formally harmonised.",
    )

    result = pd.DataFrame(rows)

    result.to_csv(
        OUT / "123_leakage_audit.csv",
        index=False,
    )

    if result.status.eq("FAIL").any():

        raise RuntimeError(
            "Provenance audit failed. "
            "Inspect 123_leakage_audit.csv."
        )

    return result, discovery


# ============================================================
# Anchor samples
# ============================================================

def build_samples(d, fetch):

    samples = {}
    frames = {}
    coverage = []

    for cycle in CYCLES:

        demo = read_xpt(
            cycle, "DEMO",
            ["SEQN", "SDMVSTRA", "SDMVPSU"],
            fetch,
        )

        frames[cycle] = (
            demo[["SDMVSTRA", "SDMVPSU"]]
            .dropna()
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if frames[cycle].groupby(
            "SDMVSTRA"
        ).SDMVPSU.nunique().min() < 2:

            raise RuntimeError(
                f"Incomplete design frame: {cycle}"
            )

        glu = read_xpt(
            cycle, "GLU",
            ["SEQN", "WTSAF2YR"],
            fetch,
        )

        base = d.loc[
            d.CYCLE.eq(cycle)
            & eligible(d)
        ].copy()

        base = base.drop(
            columns=[
                "SDMVSTRA",
                "SDMVPSU",
                "WTSAF2YR",
            ],
            errors="ignore",
        )

        base = base.merge(
            demo, on="SEQN",
            how="left",
            validate="one_to_one",
        )

        base = base.merge(
            glu, on="SEQN",
            how="left",
            validate="one_to_one",
        )

        if base[
            ["SDMVSTRA", "SDMVPSU"]
        ].isna().any().any():

            raise ValueError(
                f"Unmatched survey design: {cycle}"
            )

        for anchor, config in ANCHORS.items():

            meta = {
                "cycle": cycle,
                "anchor": anchor,
            }

            if anchor == "OGTT" and cycle == "2123":

                coverage.append({
                    **meta,
                    "status": "not_available_by_design",
                })

                continue

            try:

                kind = config["file"]
                variable = config["variable"]

                if anchor == "CRP" and cycle == "2123":

                    kind = "HSCRP"
                    variable = "LBXHSCRP"

                columns = ["SEQN", variable]

                weight = "WTSAF2YR"

                if anchor == "OGTT":

                    columns.append("WTSOG2YR")
                    weight = "WTSOG2YR"

                assay = read_xpt(
                    cycle, kind, columns, fetch
                )

                assay = assay.rename(
                    columns={variable: "ANCHOR_VALUE"}
                )

                q = base.merge(
                    assay,
                    on="SEQN",
                    how="left",
                    validate="one_to_one",
                )

                if config["women_only"]:

                    q = q.loc[
                        q.RIAGENDR.eq(2)
                        & q.RIDAGEYR.between(20, 49)
                    ].copy()

                n_eligible = len(q)

                q = q.loc[
                    q.ANCHOR_VALUE.gt(0)
                    & q[weight].gt(0)
                    & q[
                        BASE_COV + ["RIAGENDR"]
                    ].notna().all(axis=1)
                ].copy()

                if anchor == "CRP" and cycle != "2123":

                    # Convert historical mg/dL to mg/L.
                    q["ANCHOR_VALUE"] *= 10

                q["Y"] = np.log(
                    q.ANCHOR_VALUE
                )

                q["W"] = q[weight]

                q = q.reset_index(drop=True)

                if len(q) < 100:

                    raise ValueError(
                        f"Insufficient sample: n={len(q)}"
                    )

                samples[anchor, cycle] = q

                w = q.W.to_numpy(float)

                coverage.append({
                    **meta,
                    "status": "available",
                    "eligible_scaffold_n": n_eligible,
                    "n": len(q),
                    "effective_n":
                        w.sum() ** 2 / np.sum(w ** 2),
                    "weight": weight,
                    "represented_psus": q[
                        ["SDMVSTRA", "SDMVPSU"]
                    ].drop_duplicates().shape[0],
                    "outcome": "natural_log",
                })

            except Exception:

                coverage.append({
                    **meta,
                    "status": "failed",
                    "reason": traceback.format_exc(),
                })

    coverage = pd.DataFrame(coverage)

    coverage.to_csv(
        OUT / "123_anchor_coverage.csv",
        index=False,
    )

    return samples, frames, coverage


# ============================================================
# Survey Taylor-linearized covariance
# ============================================================

def survey_wls(x, y, w, q, frame):

    w = w / w.mean()

    beta = fit_wls(x, y, w)

    residual = y - x @ beta

    bread = np.linalg.inv(
        (x * w[:, None]).T @ x
    )

    scores = x * (
        w * residual
    )[:, None]

    names = [
        f"score_{i}"
        for i in range(x.shape[1])
    ]

    tmp = pd.DataFrame(
        scores, columns=names
    )

    tmp["SDMVSTRA"] = q.SDMVSTRA.to_numpy()
    tmp["SDMVPSU"] = q.SDMVPSU.to_numpy()

    totals = (
        tmp.groupby(
            ["SDMVSTRA", "SDMVPSU"]
        )[names].sum()
        .reset_index()
    )

    # Preserve PSUs absent from the analytic domain.
    totals = frame.merge(
        totals,
        on=["SDMVSTRA", "SDMVPSU"],
        how="left",
        validate="one_to_one",
    )

    totals[names] = totals[names].fillna(0)

    meat = np.zeros(
        (x.shape[1], x.shape[1])
    )

    design_df = 0

    for _, sub in totals.groupby("SDMVSTRA"):

        a = sub[names].to_numpy(float)

        m = len(a)

        if m < 2:

            raise ValueError("Singleton design stratum")

        centered = a - a.mean(axis=0)

        meat += (
            m / (m - 1)
        ) * centered.T @ centered

        design_df += m - 1

    covariance = bread @ meat @ bread

    covariance = (
        covariance + covariance.T
    ) / 2

    return beta, covariance, design_df


def wald(beta, covariance, indices, design_df):

    v = covariance[np.ix_(indices, indices)]

    b = beta[indices]

    k = len(indices)

    if np.linalg.matrix_rank(v) != k:

        return np.nan, np.nan, k

    if design_df <= 0:

        return np.nan, np.nan, k

    statistic = float(
        b @ np.linalg.solve(v, b) / k
    )

    p = f_dist.sf(
        statistic, k, design_df
    )

    return statistic, p, k


# ============================================================
# Conditional construct validation
# ============================================================

def covariates(q, anchor, expanded=False):

    cols = BASE_COV.copy()

    if not ANCHORS[anchor]["women_only"]:

        cols.append("RIAGENDR")

    if expanded:

        cols += EXTRA_NUM

    c = q[cols].to_numpy(float)

    if expanded:

        dummy = pd.get_dummies(
            q[EXTRA_CAT].astype(str),
            drop_first=True,
            dtype=float,
        ).to_numpy(float)

        c = np.column_stack([c, dummy])

    return c


def conditional_test(
    q, frame, anchor, cycle, model,
    expanded=False,
):

    w = q.W.to_numpy(float)

    y = q.Y.to_numpy(float)

    # Standardized outcome enables effect-scale comparison.
    y = (
        y - wm(y, w)
    ) / np.sqrt(variance(y, w))

    c = covariates(
        q, anchor, expanded
    )

    c = StandardScaler().fit_transform(
        c, sample_weight=w
    )

    z = q[Z].to_numpy(float)

    x = np.column_stack([
        np.ones(len(q)), c, z
    ])

    beta, covariance, df = survey_wls(
        x, y, w, q, frame
    )

    full_mse = mse(
        y, x @ beta, w
    )

    offset = x.shape[1] - 7

    tests = []
    coefs = []

    for group, positions in GROUPS.items():

        ix = [
            offset + i for i in positions
        ]

        F, p, df_num = wald(
            beta, covariance, ix, df
        )

        reduced = np.delete(
            x, ix, axis=1
        )

        pred = reduced @ fit_wls(
            reduced, y, w
        )

        reduced_mse = mse(y, pred, w)

        tests.append({
            "anchor": anchor,
            "cycle": cycle,
            "model": model,
            "group": group,
            "n": len(q),
            "partial_R2":
                (reduced_mse - full_mse)
                / reduced_mse,
            "incremental_R2":
                reduced_mse - full_mse,
            "F": F,
            "p": p,
            "df_num": df_num,
            "design_df": df,
            "status":
                "available" if np.isfinite(p)
                else "design_limited",
        })

        for j in positions:

            k = offset + j

            coefs.append({
                "anchor": anchor,
                "cycle": cycle,
                "model": model,
                "group": group,
                "coordinate": Z[j],
                "beta": beta[k],
                "design_SE": np.sqrt(
                    max(covariance[k, k], 0)
                ),
            })

    return tests, coefs


def run_conditional(samples, frames):

    tests = []
    coefs = []

    for (anchor, cycle), q in tqdm(
        samples.items(),
        desc="Conditional survey inference",
    ):

        configurations = [
            ("Primary", q, False),
        ]

        # Same-sample comparison for expanded adjustment.
        extra = EXTRA_NUM + EXTRA_CAT

        if set(extra).issubset(q.columns):

            complete = q.dropna(
                subset=extra
            ).reset_index(drop=True)

            if len(complete) >= 100:

                configurations += [
                    ("Sensitivity_base", complete, False),
                    ("Sensitivity_expanded", complete, True),
                ]

        for model, sample, expanded in configurations:

            try:

                t, c = conditional_test(
                    sample,
                    frames[cycle],
                    anchor,
                    cycle,
                    model,
                    expanded,
                )

                tests.extend(t)
                coefs.extend(c)

            except Exception:

                for group in GROUPS:

                    tests.append({
                        "anchor": anchor,
                        "cycle": cycle,
                        "model": model,
                        "group": group,
                        "status": "failed",
                        "p": np.nan,
                        "reason": traceback.format_exc(),
                    })

    # Explicitly preserve all 44 planned primary tests.
    present = {
        (t["anchor"], t["cycle"], t["group"])
        for t in tests if t["model"] == "Primary"
    }

    for anchor in ANCHORS:

        for cycle in CYCLES:

            if anchor == "OGTT" and cycle == "2123":
                continue

            for group in GROUPS:

                if (anchor, cycle, group) not in present:

                    tests.append({
                        "anchor": anchor,
                        "cycle": cycle,
                        "model": "Primary",
                        "group": group,
                        "p": np.nan,
                        "status": "unavailable",
                    })

    tests = pd.DataFrame(tests)

    tests["p_holm"] = np.nan

    for model, sub in tests.groupby("model"):

        tests.loc[sub.index, "p_holm"] = holm(
            sub.p.to_numpy(float)
        )

    coefficients = pd.DataFrame(coefs)

    tests.to_csv(
        OUT / "123_conditional_wald.csv",
        index=False,
    )

    coefficients.to_csv(
        OUT / "123_anchor_coefficients.csv",
        index=False,
    )

    return tests, coefficients


# ============================================================
# Outcome-blind comparator representations
# ============================================================

def build_comparators(discovery, rotations):

    x = discovery[RAW].to_numpy(float)

    w = discovery.SURVEY_WT.to_numpy(float)

    scaler = StandardScaler().fit(
        x, sample_weight=w
    )

    xx = scaler.transform(x)

    covariance = (
        (xx * w[:, None]).T @ xx / w.sum()
    )

    values, vectors = np.linalg.eigh(covariance)

    if min(values) <= 1e-10:

        raise ValueError("Singular reference covariance")

    whitener = (
        vectors * values ** (-0.5)
    ) @ vectors.T

    pca = vectors[:, ::-1]

    rng = np.random.default_rng(SEED)

    maps = []

    for _ in range(rotations):

        Q, r = np.linalg.qr(
            rng.normal(size=(7, 7))
        )

        # Haar-distributed orthogonal rotations.
        Q *= np.where(
            np.diag(r) >= 0, 1, -1
        )

        maps.append(Q)

    return scaler, whitener, pca, maps


# ============================================================
# PSU bootstrap for paired frozen prediction contrasts
# ============================================================

def bootstrap_contrasts(q, frame, losses, reps, rng):

    key_cols = ["SDMVSTRA", "SDMVPSU"]

    frame = frame.reset_index(drop=True).copy()

    frame["cluster"] = np.arange(len(frame))

    analytic = q[key_cols].merge(
        frame,
        on=key_cols,
        how="left",
        validate="many_to_one",
    )

    if analytic.cluster.isna().any():

        raise ValueError("Unmatched bootstrap PSU")

    ix = analytic.cluster.to_numpy(int)

    w = q.W.to_numpy(float)

    K = len(frame)

    factors = np.zeros((reps, K))

    for _, sub in frame.groupby("SDMVSTRA"):

        ids = sub.cluster.to_numpy(int)
        m = len(ids)

        draws = rng.integers(
            0, m, size=(reps, m - 1)
        )

        for j, k in enumerate(ids):

            factors[:, k] = (
                (draws == j).sum(axis=1)
                * m / (m - 1)
            )

    totals = np.bincount(
        ix, weights=w, minlength=K
    )

    numerator = np.zeros(
        (K, losses.shape[1])
    )

    for j in range(losses.shape[1]):

        numerator[:, j] = np.bincount(
            ix,
            weights=w * losses[:, j],
            minlength=K,
        )

    denominator = factors @ totals

    valid = denominator > 0

    if not valid.all():

        raise ValueError(
            "Empty design bootstrap replicate"
        )

    draws = (
        factors @ numerator
        / denominator[:, None]
    )

    point = wm(losses, w)

    low, high = np.quantile(
        draws, [0.025, 0.975], axis=0
    )

    return point, low, high


# ============================================================
# Frozen anchor prediction
# ============================================================

def run_transport(
    samples, frames, comparators, reps
):

    scaler, whitener, pca, rotations = comparators

    rng = np.random.default_rng(SEED + 1)

    performance = []
    paired = []
    rotation_results = []

    for anchor in tqdm(
        ANCHORS,
        desc="Frozen anchor models",
    ):

        if (anchor, "0506") not in samples:

            continue

        train = samples[anchor, "0506"]

        w0 = train.W.to_numpy(float)
        y0 = train.Y.to_numpy(float)

        cov_scaler = StandardScaler().fit(
            covariates(train, anchor),
            sample_weight=w0,
        )

        c0 = np.column_stack([
            np.ones(len(train)),
            cov_scaler.transform(
                covariates(train, anchor)
            ),
        ])

        z0 = train[Z].to_numpy(float)

        raw0 = scaler.transform(
            train[RAW].to_numpy(float)
        )

        u0 = raw0 @ whitener

        def design(c, z):

            return np.column_stack([c, z])

        for cycle in ["0708", "2123"]:

            if (anchor, cycle) not in samples:

                continue

            q = samples[anchor, cycle]

            w = q.W.to_numpy(float)
            y = q.Y.to_numpy(float)

            c = np.column_stack([
                np.ones(len(q)),
                cov_scaler.transform(
                    covariates(q, anchor)
                ),
            ])

            z = q[Z].to_numpy(float)

            raw = scaler.transform(
                q[RAW].to_numpy(float)
            )

            u = raw @ whitener

            def predict(a0, a):

                return a @ fit_wls(
                    a0, y0, w0
                )

            predictions = {
                "Covariates": predict(c0, c),
                "HG7": predict(
                    design(c0, z0),
                    design(c, z),
                ),
                "Raw7": predict(
                    design(c0, raw0),
                    design(c, raw),
                ),
            }

            np.testing.assert_allclose(
                predictions["HG7"],
                predictions["Raw7"],
                atol=1e-7,
                rtol=1e-6,
            )

            for group, ix in GROUPS.items():

                other = [
                    i for i in range(7)
                    if i not in ix
                ]

                predictions[group] = predict(
                    design(c0, z0[:, ix]),
                    design(c, z[:, ix]),
                )

                predictions["Without_" + group] = predict(
                    design(c0, z0[:, other]),
                    design(c, z[:, other]),
                )

            for k in [1, 2]:

                predictions[f"PCA{k}"] = predict(
                    design(c0, (raw0 @ pca)[:, :k]),
                    design(c, (raw @ pca)[:, :k]),
                )

            errors = {
                name: (y - pred) ** 2
                for name, pred in predictions.items()
            }

            meta = {
                "anchor": anchor,
                "cycle": cycle,
                "n": len(q),
                "representation_reference":
                    "2005-2008",
                "anchor_training_cycle": "0506",
            }

            for name, error in errors.items():

                value = wm(error, w)

                performance.append({
                    **meta,
                    "method": name,
                    "MSE": value,
                    "R2": 1 - value / variance(y, w),
                })

            differences = []
            labels = []

            for group, ix in GROUPS.items():

                k = len(ix)

                for comparison, diff in [
                    (
                        "group_vs_covariates",
                        errors["Covariates"]
                        - errors[group],
                    ),
                    (
                        "group_vs_PCA",
                        errors[f"PCA{k}"]
                        - errors[group],
                    ),
                    (
                        "conditional_group_contribution",
                        errors["Without_" + group]
                        - errors["HG7"],
                    ),
                ]:

                    labels.append(
                        (group, comparison)
                    )

                    differences.append(diff)

            point, low, high = bootstrap_contrasts(
                q,
                frames[cycle],
                np.column_stack(differences),
                reps,
                rng,
            )

            for i, (group, comparison) in enumerate(labels):

                paired.append({
                    **meta,
                    "group": group,
                    "comparison": comparison,
                    "gain": point[i],
                    "low95": low[i],
                    "high95": high[i],
                    "replicates": reps,
                })

            # Compare every designated group with a
            # random equal-dimensional subspace.
            #
            # These are descriptive reference distributions,
            # not permutation hypothesis-test p-values.

            for group, ix in GROUPS.items():

                k = len(ix)

                random_marginal = []
                random_conditional = []

                for Q in rotations:

                    a0 = u0 @ Q
                    a = u @ Q

                    pred = predict(
                        design(c0, a0[:, :k]),
                        design(c, a[:, :k]),
                    )

                    reduced = predict(
                        design(c0, a0[:, k:]),
                        design(c, a[:, k:]),
                    )

                    random_marginal.append(
                        wm(errors["Covariates"], w)
                        - mse(y, pred, w)
                    )

                    random_conditional.append(
                        mse(y, reduced, w)
                        - wm(errors["HG7"], w)
                    )

                actual_marginal = wm(
                    errors["Covariates"]
                    - errors[group],
                    w,
                )

                actual_conditional = wm(
                    errors["Without_" + group]
                    - errors["HG7"],
                    w,
                )

                for label, actual, null in [
                    (
                        "marginal",
                        actual_marginal,
                        random_marginal,
                    ),
                    (
                        "conditional",
                        actual_conditional,
                        random_conditional,
                    ),
                ]:

                    null = np.asarray(null)

                    rotation_results.append({
                        **meta,
                        "group": group,
                        "dimensions": k,
                        "comparison": label,
                        "HG_gain": actual,
                        "rotation_median":
                            np.median(null),
                        "rotation_q025":
                            np.quantile(null, 0.025),
                        "rotation_q975":
                            np.quantile(null, 0.975),
                        "HG_percentile":
                            100 * np.mean(null <= actual),
                        "rotations": len(null),
                    })

    performance = pd.DataFrame(performance)
    paired = pd.DataFrame(paired)
    rotations = pd.DataFrame(rotation_results)

    performance.to_csv(
        OUT / "123_frozen_performance.csv",
        index=False,
    )

    paired.to_csv(
        OUT / "123_frozen_paired_contrasts.csv",
        index=False,
    )

    rotations.to_csv(
        OUT / "123_random_rotation_controls.csv",
        index=False,
    )

    return performance, paired, rotations


# ============================================================
# Strict fold-specific HG representation sensitivity
# ============================================================

def run_cv_audit(discovery, s104):

    s103 = load_module(
        "103_jive_ajive_benchmark.py",
        "s103_for_123",
    )

    d, cov = s103.load_matched()

    tr = np.flatnonzero(
        d.SPLIT.eq("train_2005-2008")
    )

    assert len(tr) == 3815

    x = d[RAW].to_numpy(float)
    z = d[Z].to_numpy(float)

    w = d.SURVEY_WT.to_numpy(float)

    ids = d.SEQN.to_numpy()

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=103,
    )

    rows = []

    for fold, (a, b) in enumerate(
        tqdm(
            folds.split(
                tr, d.CYCLE.to_numpy()[tr]
            ),
            total=5,
            desc="Fold-specific HG audit",
        )
    ):

        it, iv = tr[a], tr[b]

        # Exclude validation participants from the
        # original physiological discovery sample.
        ref = discovery.loc[
            ~discovery.SEQN.isin(ids[iv])
        ]

        assert not set(
            ref.SEQN
        ).intersection(ids[iv])

        W, mu, rho = s104.fit_hg_cca(
            ref[RAW].to_numpy(float),
            ref.SURVEY_WT.to_numpy(float),
        )

        z_fold = (x - mu) @ W

        methods = {
            "Raw7": x,
            "DiscoveryWide_HG7": z,
            "FoldRefitted_HG7": z_fold,
        }

        for method, scores in methods.items():

            features = np.column_stack([
                cov, scores
            ])

            sc = StandardScaler().fit(
                features[it],
                sample_weight=w[it],
            )

            a_train = sc.transform(features[it])
            a_val = sc.transform(features[iv])

            for outcome, ycol in s103.PHENOTYPES.items():

                y = d[ycol].to_numpy(float)

                for alpha in [1, 10, 100, 1000]:

                    model = Ridge(alpha=alpha).fit(
                        a_train,
                        y[it],
                        sample_weight=
                            w[it] / w[it].mean(),
                    )

                    pred = model.predict(a_val)

                    rows.append({
                        "fold": fold + 1,
                        "outcome": outcome,
                        "method": method,
                        "alpha": alpha,
                        "validation_MSE":
                            mse(y[iv], pred, w[iv]),
                        "reference_fit_n": len(ref),
                        "fold_rho1": rho[0],
                        "fold_rho2": rho[1],
                    })

    result = pd.DataFrame(rows)

    result.to_csv(
        OUT / "123_fold_specific_cv.csv",
        index=False,
    )

    summary = (
        result.groupby([
            "outcome", "method", "alpha"
        ]).validation_MSE.mean()
        .reset_index()
    )

    summary = (
        summary.sort_values("validation_MSE")
        .groupby(["outcome", "method"])
        .head(1)
        .reset_index(drop=True)
    )

    summary.to_csv(
        OUT / "123_cv_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# Supplementary figures
# ============================================================

def make_figures(tests, rotations):

    primary = tests.loc[
        tests.model.eq("Primary")
    ].copy()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.1, 3.2),
        layout="constrained",
    )

    for ax, cycle in zip(axes, CYCLES):

        sub = primary.loc[
            primary.cycle.eq(cycle)
        ]

        matrix = sub.pivot(
            index="anchor",
            columns="group",
            values="partial_R2",
        ).reindex(
            index=list(ANCHORS),
            columns=list(GROUPS),
        )

        adjusted = sub.pivot(
            index="anchor",
            columns="group",
            values="p_holm",
        ).reindex(
            index=list(ANCHORS),
            columns=list(GROUPS),
        )

        vmax = max(
            float(primary.partial_R2.max()),
            0.001,
        )

        im = ax.imshow(
            matrix.to_numpy(float),
            cmap="Greens",
            vmin=0,
            vmax=vmax,
            aspect="auto",
        )

        for i in range(len(ANCHORS)):

            for j in range(len(GROUPS)):

                value = matrix.iloc[i, j]
                p = adjusted.iloc[i, j]

                if np.isfinite(value):

                    suffix = (
                        "*" if np.isfinite(p)
                        and p < 0.05 else ""
                    )

                    label = f"{value:.3f}{suffix}"

                else:

                    label = "—"

                ax.text(
                    j, i, label,
                    ha="center",
                    va="center",
                    fontsize=7,
                )

        ax.set_xticks(
            range(4),
            ["S", "D", "H", "G"],
        )

        ax.set_yticks(
            range(4),
            list(ANCHORS),
        )

        ax.set_title(cycle)

    fig.colorbar(
        im, ax=axes,
        label="Conditional partial R²",
        shrink=0.8,
    )

    fig.suptitle(
        "Independent laboratory associations\n"
        "* Holm-adjusted p < 0.05"
    )

    fig.savefig(
        OUT / "123_biological_specificity.png",
        bbox_inches="tight",
        dpi=600,
    )

    plt.close(fig)

    if rotations.empty:
        return

    q = rotations.loc[
        rotations.comparison.eq("marginal")
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.1, 3.6),
        layout="constrained",
    )

    for ax, cycle in zip(
        axes, ["0708", "2123"]
    ):

        sub = q.loc[q.cycle.eq(cycle)]

        for i, anchor in enumerate(ANCHORS):

            for j, group in enumerate(GROUPS):

                s = sub.loc[
                    sub.anchor.eq(anchor)
                    & sub.group.eq(group)
                ]

                if s.empty:
                    continue

                ax.scatter(
                    s.HG_percentile,
                    i + (j - 1.5) * 0.13,
                    color=COLORS[j],
                    s=27,
                    label=group
                        if i == 0 else None,
                )

        ax.axvline(
            50,
            color="#92999E",
            linestyle="--",
            linewidth=0.8,
        )

        ax.set_xlim(-3, 103)

        ax.set_yticks(
            range(4),
            list(ANCHORS),
        )

        ax.invert_yaxis()

        ax.set_title(cycle)

        ax.set_xlabel(
            "Percentile among matched random subspaces"
        )

    axes[0].legend(
        frameon=False,
        fontsize=6,
    )

    fig.savefig(
        OUT / "123_rotation_specificity.png",
        bbox_inches="tight",
        dpi=600,
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fetch-missing",
        action="store_true",
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=999,
    )

    parser.add_argument(
        "--rotations",
        type=int,
        default=199,
    )

    args = parser.parse_args()

    if args.replicates < 199 or args.rotations < 99:

        raise ValueError(
            "Use at least 199 PSU replicates "
            "and 99 random rotations."
        )

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    completion = OUT / "123_completion.json"

    dump(completion, {"completed": False})

    p72 = R / "72_shared_private_model_input.csv"

    p73 = R / "73_frozen_phenotype_map_input.csv"

    transform = (
        R / "72_shared_private_frozen_transform.npz"
    )

    protocol = {
        "seed": SEED,
        "replicates": args.replicates,
        "rotations": args.rotations,
        "planned_primary_tests": 44,
        "anchors": ANCHORS,
        "groups": GROUPS,
        "reference": "original frozen 2005-2008 HG7",
        "anchor_fit_cycle": "0506",
        "anchor_evaluation_cycles": ["0708", "2123"],
        "multiplicity": "Holm across 44 primary tests",
        "inference":
            "Stratified PSU Taylor-linearized WLS "
            "covariance with design-df F approximation.",
        "primary_covariates":
            "Age, BMI, income-to-poverty ratio, "
            "sex where applicable.",
        "sensitivity_covariates":
            "Additional eGFR, race, education, smoking.",
        "source_hashes": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [p72, p73, transform]
        },
        "interpretation": [
            "No assay enters the HG7 fit.",
            "Anchor models do not select the HG basis.",
            "0708 contributed to original HG7 fitting.",
            "0708 is therefore an anchor-model holdout, "
            "not a representation-discovery holdout.",
            "2123 is a previously inspected temporal cohort.",
            "Anchor-specific samples need not coincide.",
            "Random-subspace percentiles are descriptive.",
            "No causal mechanism identification.",
        ],
    }

    dump(
        OUT / "123_protocol.json",
        protocol,
    )

    print("\nLoading frozen physiological scaffold...")

    d72 = load_scaffold(p72)
    d73 = load_scaffold(p73)

    frozen = dict(np.load(transform))

    assert int(
        frozen["shared_rank"].ravel()[0]
    ) == 2

    s104 = load_module(
        "104_representation_validation_gate.py",
        "s104_for_123",
    )

    provenance, discovery = audit_provenance(
        d72, d73, frozen, s104
    )

    print("\nProvenance audit:")

    print(
        provenance.status.value_counts().to_string()
    )

    samples, frames, coverage = build_samples(
        d73, args.fetch_missing
    )

    if not samples:

        raise RuntimeError(
            "No eligible assay samples loaded."
        )

    print("\nAnchor coverage:")

    print(
        coverage[
            ["cycle", "anchor", "status"]
        ].to_string(index=False)
    )

    tests, coefs = run_conditional(
        samples, frames
    )

    comparators = build_comparators(
        discovery,
        args.rotations,
    )

    performance, paired, rotations = run_transport(
        samples,
        frames,
        comparators,
        args.replicates,
    )

    cv = run_cv_audit(
        discovery, s104
    )

    primary = tests.loc[
        tests.model.eq("Primary")
    ]

    make_figures(tests, rotations)

    status = {
        "completed": True,
        "available_anchor_cycle_samples": len(samples),
        "expected_anchor_cycle_samples": 11,
        "planned_primary_tests": len(primary),
        "available_primary_tests":
            int(primary.p.notna().sum()),
        "holm_significant_primary_tests":
            int(primary.p_holm.lt(0.05).sum()),
        "provenance_failures":
            int(provenance.status.eq("FAIL").sum()),
        "provenance_reviews":
            int(provenance.status.eq("REVIEW").sum()),
        "complete_anchor_coverage": len(samples) == 11,
        "frozen_paired_contrasts": len(paired),
        "rotation_comparisons": len(rotations),
        "fold_cv_models": len(cv),
        "causal_mechanisms_proven": False,
        "independent_clinical_validation": False,
    }

    dump(completion, status)

    supported = primary.loc[
        primary.p_holm.lt(0.05)
    ]

    sections = [
        "# Script123 — Biological specificity and leakage audit",
        "",
        "## Completion",
        "",
        "```text",
        json.dumps(status, indent=2),
        "```",
        "",
        "## Provenance",
        "",
        "```text",
        provenance.to_string(index=False),
        "```",
        "",
        "## Anchor coverage",
        "",
        "```text",
        coverage.to_string(index=False),
        "```",
        "",
        "## All primary conditional tests",
        "",
        "```text",
        primary.to_string(index=False),
        "```",
        "",
        "## Holm-supported associations",
        "",
        "```text",
        supported.to_string(index=False),
        "```",
        "",
        "## Frozen paired contrasts",
        "",
        "```text",
        paired.to_string(index=False),
        "```",
        "",
        "## Random rotation controls",
        "",
        "```text",
        rotations.to_string(index=False),
        "```",
        "",
        "## Fold-specific CV sensitivity",
        "",
        "```text",
        cv.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        "Conditional associations support construct validity "
        "only when interpreted against comparator evidence.",
        "",
        "Neither significant associations nor superiority over "
        "random subspaces establishes biological causality.",
        "",
        "CV refits the HG representation within each fold, "
        "while retaining the existing covariate encoding.",
        "",
        "Holdout bootstrap intervals condition on fitted models "
        "and are exploratory and unadjusted for multiplicity.",
        "",
        "The available scaffold may retain selection bias. "
        "Assay harmonisation and unmeasured confounding "
        "remain unresolved.",
    ]

    (OUT / "123_READOUT.md").write_text(
        "\n".join(sections),
        encoding="utf-8",
    )

    print("\n" + "=" * 64)

    print("BIOLOGICAL SPECIFICITY AUDIT COMPLETE")

    print("=" * 64)

    print(json.dumps(status, indent=2))

    print("\nHolm-supported associations:")

    print(supported.to_string(index=False))

    print("\nFold-specific CV comparison:")

    print(cv.to_string(index=False))

    print("\nResults:", OUT)


if __name__ == "__main__":

    main()
