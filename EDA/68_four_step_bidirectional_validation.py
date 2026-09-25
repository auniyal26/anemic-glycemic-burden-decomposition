#!/usr/bin/env python
# =============================================================================
# 68_four_step_bidirectional_validation.py
#
# Exact notebook experiment:
#
#   X0 --f--> Y1 --g--> X1 --f--> Y2 --g--> X2
#
# Main checks:
#   epsilon_f = d(Y1, Y2)
#   epsilon_g = d(X1, X2)
#
# Extra checks:
#   d(X0, X1)
#   d(X0, X2)
#   d(Y_reference, Y1)
#
# Y_reference is built from the EXISTING frozen PCA coordinates followed by ICA.
# No PCA refit.
#
# f and g are learned on 70% of 2005-08 and tested on untouched 30%.
# Locked Results untouched.
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

def wls(A, B, w):
    D = np.column_stack([np.ones(len(A)), A])
    sw = np.sqrt(w)
    return np.linalg.lstsq(D * sw[:, None], B * sw[:, None], rcond=None)[0]

def predict(A, coef):
    return np.column_stack([np.ones(len(A)), A]) @ coef

def epsilon(target, estimate, w):
    target = np.asarray(target, float)
    estimate = np.asarray(estimate, float)
    mu = np.sum(w[:, None] * target, axis=0) / np.sum(w)
    sse = np.sum(w[:, None] * (target - estimate) ** 2)
    sst = np.sum(w[:, None] * (target - mu) ** 2)
    return float(np.sqrt(sse / sst))

def wrmse(target, estimate, w):
    target = np.asarray(target, float)
    estimate = np.asarray(estimate, float)
    return float(np.sqrt(
        np.sum(w[:, None] * (target - estimate) ** 2)
        / (np.sum(w) * target.shape[1])
    ))

summary = []
details = []
manifest_domains = {}

for domain, spec in DOMAINS.items():
    q = d[d["PERIOD"].eq("2005-2008")].copy()
    needed = spec["raw"] + spec["pcs"] + ["SURVEY_WT"]

    for c in needed:
        q[c] = pd.to_numeric(q[c], errors="coerce")

    q = q.dropna(subset=needed)
    q = q[q["SURVEY_WT"] > 0].reset_index(drop=True)

    X = q[spec["raw"]].to_numpy(float)
    Z = q[spec["pcs"]].to_numpy(float)
    W = q["SURVEY_WT"].to_numpy(float)

    train_idx, test_idx = train_test_split(
        np.arange(len(q)),
        test_size=0.30,
        random_state=42
    )

    Xtr, Xte = X[train_idx], X[test_idx]
    Ztr, Zte = Z[train_idx], Z[test_idx]
    wtr, wte = W[train_idx], W[test_idx]

    # Standardize X using TRAIN only.
    Xtr_s, xmu, xsd = weighted_standardize(Xtr, wtr)
    Xte_s = (Xte - xmu) / xsd

    # Reference Y: existing frozen PCA coordinates -> ICA.
    ica = FastICA(
        n_components=3,
        whiten="unit-variance",
        algorithm="parallel",
        fun="logcosh",
        max_iter=3000,
        tol=1e-6,
        random_state=42,
    )
    Ytr_ref = ica.fit_transform(Ztr)
    Yte_ref = ica.transform(Zte)

    # Learn the two functions directly.
    # f : X -> Y
    # g : Y -> X
    f = wls(Xtr_s, Ytr_ref, wtr)
    g = wls(Ytr_ref, Xtr_s, wtr)

    # Exact four-step loop:
    # X0 --f--> Y1 --g--> X1 --f--> Y2 --g--> X2
    X0 = Xte_s
    Y1 = predict(X0, f)
    X1 = predict(Y1, g)
    Y2 = predict(X1, f)
    X2 = predict(Y2, g)

    # Main epsilons.
    eps_f = epsilon(Y1, Y2, wte)
    eps_g = epsilon(X1, X2, wte)

    # Supporting diagnostics.
    eps_forward_reference = epsilon(Yte_ref, Y1, wte)
    eps_first_backward = epsilon(X0, X1, wte)
    eps_full_loop = epsilon(X0, X2, wte)

    tests = [
        ("MAIN_epsilon_f__Y1_vs_Y2", Y1, Y2, eps_f),
        ("MAIN_epsilon_g__X1_vs_X2", X1, X2, eps_g),
        ("support__referenceY_vs_fX", Yte_ref, Y1, eps_forward_reference),
        ("support__originalX_vs_first_reconstructionX1", X0, X1, eps_first_backward),
        ("support__originalX_vs_finalX2", X0, X2, eps_full_loop),
    ]

    for name, target, estimate, eps in tests:
        summary.append({
            "domain": domain,
            "evaluation": "30pct_held_out_2005_08",
            "test": name,
            "epsilon": eps,
            "R2_equivalent": 1.0 - eps**2,
            "weighted_RMSE_standardized": wrmse(target, estimate, wte),
            "n_test": len(test_idx),
        })

    # Per-variable X1 vs X2 in original units.
    X1_raw = X1 * xsd + xmu
    X2_raw = X2 * xsd + xmu

    for j, var in enumerate(spec["raw"]):
        y = X1_raw[:, j]
        yh = X2_raw[:, j]
        mu = np.sum(wte * y) / np.sum(wte)
        sse = np.sum(wte * (y - yh) ** 2)
        sst = np.sum(wte * (y - mu) ** 2)

        details.append({
            "domain": domain,
            "comparison": "X1_vs_X2__validates_g",
            "variable": var,
            "epsilon": float(np.sqrt(sse / sst)),
            "R2_equivalent": float(1.0 - sse / sst),
            "RMSE_raw_units": float(np.sqrt(sse / np.sum(wte))),
            "n_test": len(test_idx),
        })

    # Per-source Y1 vs Y2.
    for j in range(3):
        y = Y1[:, j]
        yh = Y2[:, j]
        mu = np.sum(wte * y) / np.sum(wte)
        sse = np.sum(wte * (y - yh) ** 2)
        sst = np.sum(wte * (y - mu) ** 2)

        details.append({
            "domain": domain,
            "comparison": "Y1_vs_Y2__validates_f",
            "variable": f"{domain}_ICA{j+1}",
            "epsilon": float(np.sqrt(sse / sst)),
            "R2_equivalent": float(1.0 - sse / sst),
            "RMSE_raw_units": float(np.sqrt(sse / np.sum(wte))),
            "n_test": len(test_idx),
        })

    np.savez(
        MODELS / f"68_{domain}_f_g_maps.npz",
        f_X_to_Y=f,
        g_Y_to_X=g,
        X_train_mean=xmu,
        X_train_sd=xsd,
        ICA_components=ica.components_,
        ICA_mixing=ica.mixing_,
        ICA_mean=ica.mean_,
        raw_names=np.array(spec["raw"], dtype=object),
        pc_names=np.array(spec["pcs"], dtype=object),
    )

    manifest_domains[domain] = {
        "n_complete": int(len(q)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "X_dimension": int(X.shape[1]),
        "Y_dimension": 3,
        "ICA_iterations": int(ica.n_iter_),
    }

summary_df = pd.DataFrame(summary)
details_df = pd.DataFrame(details)

summary_df.to_csv(
    RES / "68_four_step_bidirectional_validation_summary.csv",
    index=False
)
details_df.to_csv(
    RES / "68_four_step_bidirectional_validation_details.csv",
    index=False
)

manifest = {
    "script": "68_four_step_bidirectional_validation.py",
    "chain": "X0 --f--> Y1 --g--> X1 --f--> Y2 --g--> X2",
    "main_epsilon_f": "distance(Y1, Y2)",
    "main_epsilon_g": "distance(X1, X2)",
    "supporting_checks": [
        "distance(reference ICA Y, f(X0))",
        "distance(X0, X1)",
        "distance(X0, X2)"
    ],
    "epsilon_definition":
        "sqrt(weighted squared error / weighted target variance)",
    "evaluation":
        "f and g learned on 70% of 2005-08; all reported errors evaluated on untouched 30%",
    "Y_definition":
        "ICA coordinates obtained from existing frozen PCA scores; ICA fitted only on training rows",
    "pca_refit": False,
    "locked_outputs_modified": False,
    "domains": manifest_domains,
}

(RES / "68_four_step_bidirectional_validation_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print("PASS  Script 68 complete.")
print("PASS  Exact loop tested: X0 ->f Y1 ->g X1 ->f Y2 ->g X2.")
print("PASS  epsilon_f = d(Y1,Y2).")
print("PASS  epsilon_g = d(X1,X2).")
print("PASS  f and g both learned on 70%; evaluated on untouched 30%.")
print("PASS  No PCA refit; locked Results untouched.")
print()
print(summary_df.to_string(index=False))
