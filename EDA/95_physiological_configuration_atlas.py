#!/usr/bin/env python3
"""Script 95 | Exploratory physiological configuration atlas, REAL NHANES.

Usage from the repository root:
    python EDA/95_physiological_configuration_atlas.py

Requires EDA/Results/73_frozen_phenotype_map_input.csv and original DPQ XPTs.
Uses ONLY 2005-06 for model fitting. 2007-08 and future periods are untouched tests.
No CLI arguments, synthetic fallback, frozen-model changes, or Git changes.

Crucial scope:
* Physiology-defined K-means groups are EXPLORATORY clusters, not diagnoses.
* True test-group labels require test physiology and are used for EVALUATION ONLY.
* Predictors for Y->group NEVER include test physiology, codes, or assignments.
* Tests whether PHQ information helps distinguish Z-defined groups above the
  physiological prior and age/sex/BMI baseline; does NOT recover individuals' Z.
* NHANES cycles are repeated cross-sections, NOT person-level trajectories.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

EDA = Path(__file__).resolve().parent
ROOT = EDA.parent
RESULTS = EDA / "Results"
INPUT = RESULTS / "73_frozen_phenotype_map_input.csv"
ZCOLS = [
    "SP_SHARED1", "SP_SHARED2", "SP_DISCORD1", "SP_DISCORD2",
    "SP_A_PRIVATE3", "SP_A_PRIVATE4", "SP_G_PRIVATE3",
]
PHQ = [f"DPQ0{i}0" for i in range(1, 10)]
COVARS = ["RIDAGEYR", "RIAGENDR", "BMXBMI"]
EXTRA_RAW = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW", "LBXGH", "LBXGLU", "LOG_IN"]
CYCLE_SUFFIX = {"0506": "D", "0708": "E", "0910": "F", "1112": "G", "1314": "H", "1516": "I", "1718": "J", "2123": "L"}
PERIODS = ["2007-2008", "2009-2018", "2021-2023"]
K = 4  # prespecified exploratory resolution; not optimized using the outcomes
SEED = 95
RNG = np.random.default_rng(SEED)


def weighted_mean_sd(x, w):
    m = np.average(x, axis=0, weights=w)
    sd = np.sqrt(np.average((x - m) ** 2, axis=0, weights=w))
    return m, np.maximum(sd, 1e-8)


def weighted_mean(x, w):
    return float(np.average(np.asarray(x, float), weights=w))


def cycle_data_path(cycle):
    parent = "NHANES" if cycle in ("0506", "0708") else "NHANES_Transfer"
    folder = ROOT / "Data" / parent / cycle
    stem = f"DPQ_{CYCLE_SUFFIX[cycle]}"
    options = [folder / (stem + ext) for ext in (".XPT", ".xpt")]
    for opt in options:
        if opt.is_file():
            return opt
    matches = list(folder.glob(stem + ".*")) if folder.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Missing authoritative PHQ items: {stem}.XPT in {folder}. "
        "The comparison needs all 9 symptom items; no fabricated fallback is allowed."
    )


def load_real_data():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing REAL NHANES Script-73 input: {INPUT}")
    header = pd.read_csv(INPUT, nrows=0).columns.tolist()
    required = ["SEQN", "CYCLE", "PERIOD", "SURVEY_WT", "ELIGIBLE_ADULT_NONPREG", "PHQ9_TOTAL"] + ZCOLS
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"Required Script-73 columns absent: {missing}")
    absent_covars = sorted(set(COVARS) - set(header))
    if absent_covars:
        raise ValueError(f"Expected covariates absent from Script-73: {absent_covars}")
    cols = required + COVARS + [s for s in EXTRA_RAW if s in header]
    d = pd.read_csv(INPUT, usecols=list(dict.fromkeys(cols)), low_memory=False)
    d["CYCLE"] = d["CYCLE"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(4)
    d["SEQN"] = pd.to_numeric(d["SEQN"], errors="coerce")
    for col in ["SURVEY_WT", "PHQ9_TOTAL"] + ZCOLS + COVARS + [s for s in EXTRA_RAW if s in d]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.loc[
        d["ELIGIBLE_ADULT_NONPREG"].eq(1)
        & d[ZCOLS + ["SURVEY_WT", "PHQ9_TOTAL"]].notna().all(axis=1)
        & d.SURVEY_WT.gt(0)
        & d.CYCLE.isin(CYCLE_SUFFIX)
    ].copy()
    if d.SEQN.duplicated().any():
        raise RuntimeError("Duplicate NHANES SEQN in eligible frozen-physiology data")

    # Rebuild each item from original NHANES DPQ files, as in Script 81.
    frames = []
    for cycle in CYCLE_SUFFIX:
        q = pd.read_sas(cycle_data_path(cycle), format="xport")
        if not set(["SEQN"] + PHQ).issubset(q.columns):
            raise ValueError(f"Cycle {cycle}: PHQ items absent from original DPQ XPT")
        q = q[["SEQN"] + PHQ].copy()
        q["SEQN"] = pd.to_numeric(q.SEQN, errors="coerce")
        for col in PHQ:
            x = pd.to_numeric(q[col], errors="coerce")
            x.loc[x.abs() < 1e-10] = 0
            q[col] = x.where(x.isin([0, 1, 2, 3]))
        frames.append(q)
    phq = pd.concat(frames, ignore_index=True)
    if phq.SEQN.duplicated().any():
        raise RuntimeError("Duplicate SEQN in raw DPQ bank")
    d = d.merge(phq, on="SEQN", how="left", validate="one_to_one")
    # Fair comparison: every model is evaluated on the SAME complete PHQ9/Z cohort.
    d = d.loc[d[PHQ].notna().all(axis=1)].copy()
    d["PHQ9_ITEMS_TOTAL"] = d[PHQ].sum(axis=1)
    # Strict audit that original score is identical to the nine items.
    mismatch = np.abs(d.PHQ9_ITEMS_TOTAL.to_numpy() - d.PHQ9_TOTAL.to_numpy()) > 1e-6
    if mismatch.any():
        raise ValueError(f"PHQ-9 total disagrees with raw items for {int(mismatch.sum())} rows; stop and reconcile")
    d["PERIOD_EXPERIMENT"] = d.CYCLE.map({"0506": "train_2005-2006", "0708": "2007-2008",
                                  "0910": "2009-2018", "1112": "2009-2018", "1314": "2009-2018",
                                  "1516": "2009-2018", "1718": "2009-2018", "2123": "2021-2023"})
    slices = {p: d.loc[d.PERIOD_EXPERIMENT.eq(p)].copy() for p in ["train_2005-2006"] + PERIODS}
    sizes = {p: len(s) for p, s in slices.items()}
    if any(n < 100 for n in sizes.values()):
        raise ValueError(f"Not enough complete real observations per split: {sizes}")
    return slices, [s for s in EXTRA_RAW if s in d]


def prep_features(d, name, median=None):
    if name == "PHQ_total":
        cols = ["PHQ9_ITEMS_TOTAL"]
    elif name == "PHQ_items":
        cols = PHQ
    elif name == "covariates_only":
        cols = COVARS
    elif name == "PHQ_total_plus_covariates":
        cols = ["PHQ9_ITEMS_TOTAL"] + COVARS
    elif name == "PHQ_items_plus_covariates":
        cols = PHQ + COVARS
    else:
        raise ValueError(name)
    a = d[cols].to_numpy(dtype=float)
    if median is None:
        median = np.nanmedian(a, axis=0)
        if not np.isfinite(median).all():
            raise ValueError(f"All values missing for some {name} columns in training data")
    a = np.where(np.isfinite(a), a, median[None, :])
    return a, median


def model_fit_predict(train, target, w, tests, name):
    xt, med = prep_features(train, name)
    scaler = StandardScaler()
    xs = scaler.fit_transform(xt, sample_weight=w)
    model = LogisticRegression(C=0.5, max_iter=1500, solver="lbfgs", random_state=SEED)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        model.fit(xs, target, sample_weight=w / w.mean())
    if len(model.classes_) != K:
        raise RuntimeError("Not all training clusters represented in classifier")
    return {period: model.predict_proba(scaler.transform(prep_features(d, name, med)[0]))
            for period, d in tests.items()}


def metrics(y, p, w, prior):
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    ix = np.arange(len(y))
    pred = p.argmax(axis=1)
    top2 = np.argsort(-p, axis=1)[:, :2]
    rec = []
    for cl in range(K):
        keep = y == cl
        if np.any(keep):
            rec.append(float(np.average(pred[keep] == cl, weights=w[keep])))
    score = {
        "weighted_accuracy": weighted_mean(pred == y, w),
        "weighted_balanced_accuracy": float(np.mean(rec)),
        "weighted_top2_hit": weighted_mean(np.any(top2 == y[:, None], axis=1), w),
        "weighted_logloss_nats": weighted_mean(-np.log(p[ix, y]), w),
        "weighted_brier": weighted_mean(np.sum((p - np.eye(K)[y]) ** 2, axis=1), w),
    }
    lp = np.broadcast_to(prior[None, :], p.shape)
    score["information_gain_vs_prior_nats"] = float(
        weighted_mean(-np.log(np.clip(lp[ix, y], 1e-12, 1)), w) - score["weighted_logloss_nats"]
    )
    return score


def entropy(p):
    pp = np.asarray(p, float)
    pp = pp[pp > 0]
    return float(-np.sum(pp * np.log(pp)))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    S, raw = load_real_data()
    train = S["train_2005-2006"]
    wt = train.SURVEY_WT.to_numpy(float)
    zt = train[ZCOLS].to_numpy(float)
    mu, sd = weighted_mean_sd(zt, wt)
    xs = (zt - mu) / sd
    km = KMeans(n_clusters=K, n_init=20, random_state=SEED)
    km.fit(xs, sample_weight=wt / wt.mean())
    labels = km.labels_.astype(int)
    # Deterministic cluster relabeling by increasing first latent coordinate.
    # Makes output tables stable and readable; labels have no clinical meaning.
    order = np.argsort(km.cluster_centers_[:, 0])
    old_to_new = {int(o): int(i) for i, o in enumerate(order)}
    labels = np.array([old_to_new[int(x)] for x in labels], int)
    centers = km.cluster_centers_[order]
    clw = np.bincount(labels, weights=wt, minlength=K)
    prior = clw / clw.sum()
    if (prior < 0.03).any():
        print("WARNING: Some discovery clusters contain <3% weighted observations; interpret cautiously.")
    # Bootstrap stability: compare cluster partitions of the SAME discovery rows.
    ari = []
    for b in range(12):
        indices = RNG.choice(len(xs), size=len(xs), replace=True)
        boot = KMeans(n_clusters=K, n_init=5, random_state=SEED + 100 + b)
        boot.fit(xs[indices], sample_weight=(wt[indices] / wt[indices].mean()))
        ari.append(adjusted_rand_score(labels, boot.predict(xs)))
    # A discovery-only separation score: KMeans ALWAYS imposes clusters.
    sample = RNG.choice(len(xs), size=min(1200, len(xs)), replace=False)
    silhouette = float(silhouette_score(xs[sample], labels[sample]))
    print(f"SCRIPT 95 | REAL NHANES Script-73 | 7 frozen Z | exploratory K={K}")
    print("Data: "+", ".join(f"{p} n={len(d)}" for p,d in S.items()))
    print(f"Discovery silhouette={silhouette:.4f}; bootstrap ARI={np.mean(ari):.4f} (SD {np.std(ari):.4f}); min prior={prior.min():.3f}")

    true_labels = {}
    distance_train = np.min(np.linalg.norm(xs[:, None, :] - centers[None, :, :], axis=2), axis=1)
    cutoff = float(np.quantile(distance_train, 0.95))
    status_rows = []
    profiles = []
    for period, d in S.items():
        z = d[ZCOLS].to_numpy(float)
        stdz = (z - mu) / sd
        group = np.argmin(np.linalg.norm(stdz[:, None, :] - centers[None, :, :], axis=2), axis=1)
        if period == "train_2005-2006" and not np.array_equal(group, labels):
            raise RuntimeError("Unexpected training label mismatch")
        true_labels[period] = group
        w = d.SURVEY_WT.to_numpy(float)
        pointdist = np.min(np.linalg.norm(stdz[:, None, :] - centers[None, :, :], axis=2), axis=1)
        allprop = np.bincount(group, weights=w, minlength=K) / np.sum(w)
        status_rows.append(dict(period=period, cluster_id="ALL", n=len(d), weighted_fraction=1.0,
                                out_of_discovery_95pct_radius=weighted_mean(pointdist > cutoff, w),
                                prevalence_total_variation_from_train=0.5 * float(np.abs(allprop - prior).sum()),
                                mean_standardized_center_distance=float("nan")))
        for cl in range(K):
            m = group == cl
            prop = weighted_mean(m, w)
            cm = np.average(stdz[m], axis=0, weights=w[m]) if m.any() else np.full(len(ZCOLS), np.nan)
            drift = float(np.linalg.norm(cm - centers[cl])) if m.any() else float("nan")
            status_rows.append(dict(period=period, cluster_id=int(cl), n=int(m.sum()),
                                    weighted_fraction=prop, out_of_discovery_95pct_radius=weighted_mean(pointdist[m] > cutoff, w[m]) if m.any() else float("nan"),
                                    prevalence_total_variation_from_train=float("nan"),
                                    mean_standardized_center_distance=drift))
            row = dict(period=period, cluster_id=cl, n=int(m.sum()), weighted_fraction=prop)
            for j, col in enumerate(ZCOLS):
                row["mean_Z_"+col] = float(np.average(z[m,j], weights=w[m])) if m.any() else np.nan
                row["mean_discovery_standardized_"+col] = float(cm[j])
            for col in raw:
                ok = m & d[col].notna().to_numpy()
                row["mean_raw_"+col] = float(np.average(d.loc[ok,col], weights=w[ok])) if ok.any() else np.nan
                row["n_raw_"+col] = int(ok.sum())
            profiles.append(row)

    # Score stratification uses exact PHQ-9 totals on every period.
    score_rows = []
    for period, d in S.items():
        group = true_labels[period]
        score = d.PHQ9_ITEMS_TOTAL.to_numpy(int)
        w = d.SURVEY_WT.to_numpy(float)
        for s in range(28):
            keep = score == s
            if not keep.any():
                continue
            weights = np.bincount(group[keep], weights=w[keep], minlength=K)
            fractions = weights / weights.sum()
            row = dict(period=period, phq_total=s, n=int(keep.sum()),
                       entropy_of_physiology_groups_nats=entropy(fractions),
                       effective_number_of_groups=float(np.exp(entropy(fractions))))
            for k in range(K):
                row[f"cluster_{k}_weighted_fraction"] = float(fractions[k])
                row[f"cluster_{k}_count"] = int(np.sum(group[keep] == k))
            score_rows.append(row)

    # Group prediction uses phenotype/covariates ONLY. True test group is used
    # only for scoring; never passed to the fitted classifier.
    comparisons = {}
    names = ["PHQ_total", "PHQ_items", "covariates_only", "PHQ_total_plus_covariates", "PHQ_items_plus_covariates"]
    for name in names:
        comparisons[name] = model_fit_predict(train, labels, wt, {p:S[p] for p in PERIODS}, name)
    rows = []
    for period in PERIODS:
        y = true_labels[period]
        w = S[period].SURVEY_WT.to_numpy(float)
        candidates = {"prior_only": np.tile(prior, (len(y), 1))}
        candidates.update({name: p[period] for name,p in comparisons.items()})
        for name, p in candidates.items():
            rows.append(dict(period=period, model=name, n=len(y), **metrics(y,p,w,prior)))
    metric_df = pd.DataFrame(rows)
    profile_df = pd.DataFrame(profiles)
    status_df = pd.DataFrame(status_rows)
    score_df = pd.DataFrame(score_rows)
    metric_df.to_csv(RESULTS / "95_phenotype_identification_comparison.csv", index=False)
    profile_df.to_csv(RESULTS / "95_physio_cluster_profiles.csv", index=False)
    status_df.to_csv(RESULTS / "95_cluster_transfer_audit.csv", index=False)
    score_df.to_csv(RESULTS / "95_phq_score_heterogeneity.csv", index=False)
    pd.DataFrame([dict(metric="train_silhouette", value=silhouette),
                  dict(metric="bootstrap_mean_ARI", value=float(np.mean(ari))),
                  dict(metric="bootstrap_SD_ARI", value=float(np.std(ari))),
                  dict(metric="K_fixed_exploratory", value=K),
                  dict(metric="discovery_distance_95pct_threshold", value=cutoff)]).to_csv(
                  RESULTS / "95_discovery_cluster_quality.csv", index=False)

    # Two presentation-ready illustrations. Neither makes clinical claims.
    fig, ax = plt.subplots(figsize=(9, 5.2))
    discovery = profile_df.loc[profile_df.period.eq("train_2005-2006")].sort_values("cluster_id")
    hm = discovery[["mean_discovery_standardized_"+col for col in ZCOLS]].to_numpy(float)
    im = ax.imshow(hm, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_yticks(np.arange(K), labels=[f"Group {i}" for i in range(K)])
    ax.set_xticks(np.arange(len(ZCOLS)), labels=[c.replace("SP_", "") for c in ZCOLS], rotation=30, ha="right")
    ax.set_title("Exploratory physiology-defined groups (NHANES 2005–06)")
    fig.colorbar(im, ax=ax, label="Mean Z in discovery SD units")
    fig.tight_layout()
    fig.savefig(RESULTS / "95_physio_cluster_profiles.png", dpi=175)
    plt.close(fig)

    disc = S["train_2005-2006"]
    mask = disc.PHQ9_ITEMS_TOTAL.eq(10).to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for k in range(K):
        ix = mask & (labels == k)
        if ix.any():
            ax.scatter(disc.loc[ix,"SP_SHARED1"], disc.loc[ix,"SP_A_PRIVATE3"], label=f"Group {k} (n={ix.sum()})", alpha=.8)
    ax.set_title(f"Same observed PHQ-9 = 10, different physiological coordinates (n={mask.sum()})")
    ax.set_xlabel("Shared1")
    ax.set_ylabel("A-private3")
    if mask.any():
        ax.legend()
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(RESULTS / "95_PHQ10_physio_heterogeneity.png", dpi=175)
    plt.close(fig)

    manifest = {
        "script": "95_physiological_configuration_atlas.py",
        "source": "real Script-73 frozen Z + original DPQ XPT PHQ items",
        "training_cycle": "2005-2006 only",
        "evaluation_periods": PERIODS,
        "cluster_method": "weighted KMeans, K=4 fixed; no phenotype used to form groups",
        "warning": "KMeans imposes partitions; clusters not independently validated biological phenotypes",
        "prediction": "regularized multinomial logistic; same complete-PHQ/Z cohort; no test physiology in features",
        "comparison": "prior-only, phenotype-only, covariate-only, phenotype plus covariates",
        "weights": "NHANES SURVEY_WT weighting for clustering, fitting, and scoring (not full complex-survey variance estimation)",
        "causal": False,
        "observed_longitudinal": False,
        "seed": SEED,
        "K": K,
        "z_columns": ZCOLS,
        "covariates": COVARS,
        "raw_biomarker_columns_available": raw,
    }
    (RESULTS / "95_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# Experiment 95 — Exploratory physiological configuration atlas",
        "", "**Dataset:** real NHANES Script-73; 2005–06 train, later cycles held out.",
        f"**Physiology:** seven frozen Z coordinates. **Clustering:** K={K}, weighted KMeans on train only.",
        f"**Discovery cluster silhouette:** {silhouette:.4f}; mean bootstrap ARI: {np.mean(ari):.4f}.",
        "", "## Held-out group identification (weighted log loss; lower better)",
        "", "| Period | Model | Log loss | Δ information vs prior (nats) | Top-2 group hit |", "|---|---|---:|---:|---:|"
    ]
    for _, r in metric_df.iterrows():
        lines.append(f"| {r.period} | {r.model} | {r.weighted_logloss_nats:.4f} | {r.information_gain_vs_prior_nats:+.4f} | {r.weighted_top2_hit:.4f} |")
    lines.extend(["", "## Interpretation guardrails",
                  "- All groups are exploratory clusters, not discovered diseases or etiologies.",
                  "- A positive held-out information gain over the prior supports distinguishability; negative means worse than the prior.",
                  "- PHQ-vs-covariate comparisons test whether symptom information adds beyond available demographics.",
                  "- These metrics classify Z-derived groups, not individual 7D states; no proof of recovering patients' true physiology.",
                  "- No causal claim, no person-level longitudinal evidence, no external-dataset validation.",
                  "- Frozen mapping unchanged, no Git/Notion changes. Full complex survey-design inference not implemented."])
    (RESULTS / "95_readout.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\nHELD-OUT PHYSIOLOGY-GROUP IDENTIFICATION (lower log loss better; info gain >0 beats prior)")
    print(metric_df.loc[:,["period","model","weighted_logloss_nats","information_gain_vs_prior_nats","weighted_top2_hit"]].round(4).to_string(index=False))
    print("\nCROSS-CYCLE TRANSFER")
    print(status_df.loc[status_df.cluster_id.eq("ALL"),
                        ["period","out_of_discovery_95pct_radius","prevalence_total_variation_from_train"]].round(4).to_string(index=False))
    print("\nSaved EDA/Results/95_* (5 CSVs, 2 PNG figures, 1 manifest, 1 MD readout).")
    print("CAUTION: clusters are exploratory; no patient-specific reconstruction or clinical biotype established.")


if __name__ == "__main__":
    main()
