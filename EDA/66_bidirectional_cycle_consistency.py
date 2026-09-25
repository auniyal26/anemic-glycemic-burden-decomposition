#!/usr/bin/env python
# =============================================================================
# 67_bidirectional_cycle_consistency.py
#
# Exact cycle-consistency experiment:
#
#   X (raw physiology)
#      -- f --> Z (existing frozen PCA space)
#      -- h --> S (ICA source space)
#      -- g --> X_hat
#
# Test 1:  X -> Z -> S -> X_hat
#          epsilon_X = distance(X, X_hat)
#
# Test 2:  S -> X_hat -> Z_hat -> S_hat
#          epsilon_S = distance(S, S_hat)
#
# f is learned from paired raw X and the EXISTING frozen PCA coordinates Z.
# h is FastICA fitted on Z.
# g is learned from ICA source coordinates S back to raw X.
#
# Both f and g are trained on 70% of 2005-08 and evaluated on the untouched 30%.
# This avoids judging the cycle only on the data used to learn the functions.
#
# No PCA refit. Locked Results untouched.
# =============================================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "EDA" / "Results"
MODELS = ROOT / "EDA" / "Models"
RES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

INPUT = RES / "63_deep_ddx_input.csv"
if not INPUT.exists():
    raise FileNotFoundError("Need EDA/Results/63_deep_ddx_input.csv")

d = pd.read_csv(INPUT)
d["PERIOD"] = d["PERIOD"].astype(str)

DOMAINS = {
    "A": {
        "raw": ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"],
        "pcs": ["A_OI_PC1_FZ", "A_OI_PC2_FZ", "A_OI_PC3_FZ"],
    },
    "G": {
        "raw": ["LBXGH", "LBXGLU", "LOG_IN"],
        "pcs": ["G3_OI_PC1_FZ", "G3_OI_PC2_FZ", "G3_OI_PC3_FZ"],
    },
}

def weighted_standardize(X, w):
    mu = np.sum(w[:, None] * X, axis=0) / np.sum(w)
    sd = np.sqrt(np.sum(w[:, None] * (X - mu) ** 2, axis=0) / np.sum(w))
    return (X - mu) / sd, mu, sd

def apply_standardize(X, mu, sd):
    return (X - mu) / sd

def wls(A, B, w):
    # B ~= intercept + A @ coef
    D = np.column_stack([np.ones(len(A)), A])
    sw = np.sqrt(w)
    return np.linalg.lstsq(D * sw[:, None], B * sw[:, None], rcond=None)[0]

def predict(A, coef):
    return np.column_stack([np.ones(len(A)), A]) @ coef

def normalized_epsilon(target, estimate, w):
    """
    sqrt(weighted squared error / weighted target variance).
    epsilon=0 is perfect.
    R2 = 1 - epsilon^2.
    """
    mu = np.sum(w[:, None] * target, axis=0) / np.sum(w)
    num = np.sum(w[:, None] * (target - estimate) ** 2)
    den = np.sum(w[:, None] * (target - mu) ** 2)
    return float(np.sqrt(num / den))

def weighted_rmse(target, estimate, w):
    return float(np.sqrt(np.sum(w[:, None] * (target-estimate)**2) /
                         (np.sum(w) * target.shape[1])))

def component_eps(target, estimate, w, names):
    rows = []
    for j, name in enumerate(names):
        y = target[:, j]
        yh = estimate[:, j]
        mu = np.sum(w*y)/np.sum(w)
        sse = np.sum(w*(y-yh)**2)
        sst = np.sum(w*(y-mu)**2)
        eps = float(np.sqrt(sse/sst))
        rmse = float(np.sqrt(sse/np.sum(w)))
        rows.append((name, eps, 1-eps**2, rmse))
    return rows

summary = []
details = []
maps_meta = {}

for domain, spec in DOMAINS.items():
    cols = spec["raw"] + spec["pcs"] + ["SURVEY_WT"]
    q = d[d["PERIOD"].eq("2005-2008")].copy()
    for c in cols:
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q = q.dropna(subset=cols)
    q = q[q["SURVEY_WT"] > 0].reset_index(drop=True)

    idx_train, idx_test = train_test_split(
        np.arange(len(q)), test_size=0.30, random_state=42
    )

    X = q[spec["raw"]].to_numpy(float)
    Z = q[spec["pcs"]].to_numpy(float)
    W = q["SURVEY_WT"].to_numpy(float)

    Xtr, Xte = X[idx_train], X[idx_test]
    Ztr, Zte = Z[idx_train], Z[idx_test]
    wtr, wte = W[idx_train], W[idx_test]

    # Standardize X using TRAIN only.
    Xtr_s, xmu, xsd = weighted_standardize(Xtr, wtr)
    Xte_s = apply_standardize(Xte, xmu, xsd)

    # f: X -> EXISTING frozen PCA coordinates Z.
    # This is not a PCA refit: it learns the deterministic mapping to the
    # already-computed frozen PC scores.
    f_X_to_Z = wls(Xtr_s, Ztr, wtr)
    Ztr_hat = predict(Xtr_s, f_X_to_Z)
    Zte_hat = predict(Xte_s, f_X_to_Z)

    # h: Z -> ICA sources S.
    ica = FastICA(
        n_components=3,
        whiten="unit-variance",
        algorithm="parallel",
        fun="logcosh",
        max_iter=3000,
        tol=1e-6,
        random_state=42,
    )
    Str = ica.fit_transform(Ztr)
    Ste = ica.transform(Zte)

    # For a fair forward cycle on unseen X, use f(X) then the SAME frozen ICA.
    Ste_from_X = ica.transform(Zte_hat)

    # g: S -> X, learned on TRAIN only.
    g_S_to_X = wls(Str, Xtr_s, wtr)

    # ---------------------------------------------------------------------
    # TEST 1: X -> f -> Z -> h -> S -> g -> X_hat
    # ---------------------------------------------------------------------
    Xte_cycle = predict(Ste_from_X, g_S_to_X)

    # ---------------------------------------------------------------------
    # TEST 2: S -> g -> X_hat -> f -> Z_hat -> h -> S_hat
    # Start from the actual frozen-ICA S for each held-out subject.
    # ---------------------------------------------------------------------
    Xte_from_S = predict(Ste, g_S_to_X)
    Zte_cycle = predict(Xte_from_S, f_X_to_Z)
    Ste_cycle = ica.transform(Zte_cycle)

    # Supporting one-way errors.
    eps_f = normalized_epsilon(Zte, Zte_hat, wte)
    eps_g = normalized_epsilon(Xte_s, Xte_from_S, wte)
    eps_X_cycle = normalized_epsilon(Xte_s, Xte_cycle, wte)
    eps_S_cycle = normalized_epsilon(Ste, Ste_cycle, wte)

    for label, eps, target, est in [
        ("f__X_to_Z", eps_f, Zte, Zte_hat),
        ("g__S_to_X", eps_g, Xte_s, Xte_from_S),
        ("cycle__X_to_Z_to_S_to_X", eps_X_cycle, Xte_s, Xte_cycle),
        ("cycle__S_to_X_to_Z_to_S", eps_S_cycle, Ste, Ste_cycle),
    ]:
        summary.append({
            "domain": domain,
            "evaluation": "30pct_held_out_discovery",
            "test": label,
            "epsilon": eps,
            "R2_equivalent": 1 - eps**2,
            "weighted_RMSE_standardized": weighted_rmse(target, est, wte),
            "n_test": len(idx_test),
        })

    # Per-variable errors for the X-cycle in original units.
    Xte_cycle_raw = Xte_cycle * xsd + xmu
    Xte_from_S_raw = Xte_from_S * xsd + xmu

    for j, var in enumerate(spec["raw"]):
        for label, est in [
            ("g__S_to_X", Xte_from_S_raw[:, j]),
            ("cycle__X_to_Z_to_S_to_X", Xte_cycle_raw[:, j]),
        ]:
            y = Xte[:, j]
            mu = np.sum(wte*y)/np.sum(wte)
            sse = np.sum(wte*(y-est)**2)
            sst = np.sum(wte*(y-mu)**2)
            details.append({
                "domain": domain,
                "space": "X_raw",
                "variable": var,
                "test": label,
                "epsilon": float(np.sqrt(sse/sst)),
                "R2_equivalent": float(1-sse/sst),
                "RMSE_raw_units": float(np.sqrt(sse/np.sum(wte))),
                "n_test": len(idx_test),
            })

    # Per-source errors for S-cycle.
    for name, eps, r2, rmse in component_eps(
        Ste, Ste_cycle, wte,
        [f"{domain}_ICA1", f"{domain}_ICA2", f"{domain}_ICA3"]
    ):
        details.append({
            "domain": domain,
            "space": "S_ICA",
            "variable": name,
            "test": "cycle__S_to_X_to_Z_to_S",
            "epsilon": eps,
            "R2_equivalent": r2,
            "RMSE_raw_units": rmse,
            "n_test": len(idx_test),
        })

    # Save maps for reproducibility.
    np.savez(
        MODELS / f"67_{domain}_bidirectional_cycle_maps.npz",
        f_X_to_Z=f_X_to_Z,
        g_S_to_X=g_S_to_X,
        X_train_mean=xmu,
        X_train_sd=xsd,
        ICA_components=ica.components_,
        ICA_mixing=ica.mixing_,
        ICA_mean=ica.mean_,
        raw_names=np.array(spec["raw"], dtype=object),
        pc_names=np.array(spec["pcs"], dtype=object),
    )

    maps_meta[domain] = {
        "n_total_complete_2005_08": int(len(q)),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "X_dimensions": len(spec["raw"]),
        "Z_dimensions": len(spec["pcs"]),
        "S_dimensions": 3,
        "ICA_iterations": int(ica.n_iter_),
    }

summary_df = pd.DataFrame(summary)
details_df = pd.DataFrame(details)

summary_df.to_csv(RES / "67_bidirectional_cycle_summary.csv", index=False)
details_df.to_csv(RES / "67_bidirectional_cycle_details.csv", index=False)

manifest = {
    "script": "67_bidirectional_cycle_consistency.py",
    "experiment": {
        "f": "X raw physiology -> existing frozen PCA coordinates Z",
        "h": "Z -> ICA source coordinates S",
        "g": "S -> raw physiology X",
        "cycle_X": "X -> f -> Z -> h -> S -> g -> X_hat",
        "cycle_S": "S -> g -> X_hat -> f -> Z_hat -> h -> S_hat",
    },
    "epsilon": "sqrt(weighted squared reconstruction error / weighted target variance)",
    "evaluation": "all maps trained on 70% of 2005-08; epsilons reported on untouched 30%",
    "pca_refit": False,
    "locked_outputs_modified": False,
    "metadata": maps_meta,
}
(RES / "67_bidirectional_cycle_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

print("PASS  Script 67 bidirectional cycle-consistency experiment complete.")
print("PASS  f: X -> existing frozen PCA space.")
print("PASS  h: frozen PCA space -> ICA sources.")
print("PASS  g: ICA sources -> X.")
print("PASS  Both X->...->X and S->...->S cycles evaluated on untouched 30% holdout.")
print("PASS  No PCA refit; locked Results untouched.")
print()
print(summary_df.to_string(index=False))
