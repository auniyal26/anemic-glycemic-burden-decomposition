
#!/usr/bin/env python3
"""
118 | Biomedical utility of physiological representations.

Primary question
----------------
Does the designated HG shared pair provide better access
to ferritin-associated information than alternative
two-dimensional representations?

Discovery:
    NHANES 2005-2006

Frozen evaluation:
    NHANES 2007-2008
    NHANES 2021-2023

Primary outcome:
    Log serum ferritin.

Secondary outcome:
    Discrimination of ferritin <15 ug/L.

Population:
    Women aged 20-49, using Script106 eligibility.

Comparators:
    Covariates
    Conventional scalar burdens
    Continuous Hb and HbA1c
    Hb and RDW
    PCA2
    Haematology PCA2
    Glycaemia PCA2
    Native JIVE joint2
    Native AJIVE joint2
    Two-domain multi-decoder VAE2
    HG shared2
    HG discordant2
    HG private2

Full-dimensional controls:
    H4
    G3
    RawX7
    HG7
    PCA7

The VAE is inspired by SYNERGY-VAE, not an exact
reproduction of its five-modality architecture.

All results are exploratory.
No clinical or causal superiority is assumed.
"""

from __future__ import annotations

import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[key] = "1"

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import traceback

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

import torch
from torch import nn

sys.dont_write_bytecode = True

EDA = Path(__file__).resolve().parent
ROOT = EDA.parent

SEED = 118

CYCLES = ("0506", "0708", "2123")
HOLDOUTS = ("0708", "2123")

BASE_COV = [
    "RIDAGEYR",
    "BMXBMI",
    "INDFMPIR",
]

RIDGE_ALPHA = 10.0

VAE_SEEDS = (118, 119, 120, 121, 122)

PRIMARY_METHOD = "HG_Shared2"

MATCHED_COMPARATORS = (
    "Threshold2",
    "Hb_HbA1c",
    "Hb_RDW",
    "PCA2",
    "H_PCA2",
    "G_PCA2",
    "JIVE_Joint2",
    "AJIVE_Joint2",
    "VAE2_mean_loss",
)

REQUIRED = (
    "72_shared_private_decomposition_and_phenotype.py",
    "103_jive_ajive_benchmark.py",
    "104_representation_validation_gate.py",
    "106_survey_ferritin_anchor.py",
    "112_finish_benchmarks.py",
    "114_structure_recovery.py",
)


# ============================================================
# Utilities
# ============================================================

def load_module(name, filename):

    path = EDA / filename

    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


def dump_json(path, data):

    path.write_text(
        json.dumps(
            data,
            indent=2,
            default=lambda v: (
                v.item()
                if isinstance(v, np.generic)
                else str(v)
            ),
        ),
        encoding="utf-8",
    )


def source_hashes():

    files = [EDA / name for name in REQUIRED]
    files.append(Path(__file__))

    return {
        p.name: hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in files
        if p.exists()
    }


def weighted_mean(x, w):

    return np.average(
        x,
        axis=0,
        weights=w,
    )


def weighted_mse(y, pred, w):

    return float(
        weighted_mean(
            (y - pred) ** 2,
            w,
        )
    )


def weighted_r2(y, pred, w):

    denominator = weighted_mean(
        (y - weighted_mean(y, w)) ** 2,
        w,
    )

    if denominator <= 0:
        return np.nan

    return 1.0 - weighted_mse(
        y, pred, w
    ) / denominator


def weighted_auc(y, pred, w):

    if np.unique(y[w > 0]).size < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            pred,
            sample_weight=w,
        )
    )


# ============================================================
# Common downstream whitening
# ============================================================

class WeightedWhitener:

    """
    Discovery-only weighted whitening.

    Removes arbitrary differences in coordinate scaling
    and invertible rotations before ridge regression.

    Full-rank linear representations should consequently
    produce equivalent ridge predictions.
    """

    def fit(self, x, w):

        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)

        self.mean = weighted_mean(x, w)

        xc = x - self.mean

        covariance = (
            (xc * w[:, None]).T @ xc
            / w.sum()
        )

        values, vectors = np.linalg.eigh(
            covariance
        )

        if values.max() <= 0:
            raise ValueError(
                "Zero score variance"
            )

        if values.min() <= (
            values.max() * 1e-10
        ):
            raise ValueError(
                "Rank-deficient representation"
            )

        self.eigenvalues = values

        self.matrix = (
            vectors * values ** (-0.5)
        ) @ vectors.T

        return self

    def transform(self, x):

        return (
            np.asarray(x) - self.mean
        ) @ self.matrix


# ============================================================
# Two-domain multi-decoder VAE
# ============================================================

class MultiDecoderVAE(nn.Module):

    """
    Shared encoder and separate reconstruction decoders
    for the haematological and glycaemic blocks.

    The latent mean is used for downstream analysis.
    """

    def __init__(self, latent_dim=2):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
        )

        self.mu = nn.Linear(
            16,
            latent_dim,
        )

        self.logvar = nn.Linear(
            16,
            latent_dim,
        )

        self.decoder_h = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, 4),
        )

        self.decoder_g = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

    def encode(self, x):

        h = self.encoder(x)

        mu = self.mu(h)

        logvar = self.logvar(h).clamp(
            -10,
            8,
        )

        return mu, logvar

    def decode(self, z):

        return (
            self.decoder_h(z),
            self.decoder_g(z),
        )

    def forward(self, x):

        mu, logvar = self.encode(x)

        std = torch.exp(
            0.5 * logvar
        )

        z = mu + std * torch.randn_like(std)

        h, g = self.decode(z)

        return h, g, mu, logvar


def vae_loss(x, h, g, mu, logvar, beta):

    # Equal weight for both laboratory domains.
    loss_h = (
        (x[:, :4] - h) ** 2
    ).mean()

    loss_g = (
        (x[:, 4:] - g) ** 2
    ).mean()

    reconstruction = (
        loss_h + loss_g
    ) / 2.0

    kl = -0.5 * torch.mean(
        torch.sum(
            1 + logvar
            - mu.square()
            - logvar.exp(),
            dim=1,
        )
    )

    return reconstruction + beta * kl


def fit_vae(x, seed, epochs, beta):

    """
    All architecture training and early stopping occur
    within the physiological discovery cohort.

    Ferritin is never accessed by the encoder.
    """

    torch.manual_seed(seed)
    torch.set_num_threads(1)

    rng = np.random.default_rng(seed)

    order = rng.permutation(len(x))

    n_val = max(
        30,
        int(0.20 * len(x)),
    )

    x_train = torch.tensor(
        x[order[n_val:]],
        dtype=torch.float32,
    )

    x_val = torch.tensor(
        x[order[:n_val]],
        dtype=torch.float32,
    )

    model = MultiDecoderVAE(
        latent_dim=2
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    best_loss = np.inf
    best_state = None
    best_epoch = 0
    patience = 25
    stale = 0

    history = []

    for epoch in range(epochs):

        model.train()

        indices = rng.permutation(
            len(x_train)
        )

        for start in range(
            0,
            len(indices),
            64,
        ):

            batch = x_train[
                indices[start:start + 64]
            ]

            optimizer.zero_grad()

            h, g, mu, logvar = model(batch)

            loss = vae_loss(
                batch,
                h,
                g,
                mu,
                logvar,
                beta,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Nonfinite VAE loss"
                )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

        model.eval()

        with torch.no_grad():

            mu, logvar = model.encode(x_val)

            h, g = model.decode(mu)

            validation_loss = float(
                vae_loss(
                    x_val,
                    h,
                    g,
                    mu,
                    logvar,
                    beta,
                )
            )

        history.append({
            "seed": seed,
            "epoch": epoch + 1,
            "validation_objective":
                validation_loss,
        })

        if validation_loss < best_loss - 1e-5:

            best_loss = validation_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch + 1
            stale = 0

        else:

            stale += 1

        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            "VAE training failed"
        )

    model.load_state_dict(best_state)
    model.eval()

    return model, history, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_objective":
            best_loss,
        "beta": beta,
        "train_n": len(x_train),
        "validation_n": len(x_val),
    }


def vae_transform(model, x):

    with torch.no_grad():

        mu, _ = model.encode(
            torch.tensor(
                x,
                dtype=torch.float32,
            )
        )

    return mu.numpy().astype(float)


def vae_reconstruct(model, x):

    with torch.no_grad():

        mu, _ = model.encode(
            torch.tensor(
                x,
                dtype=torch.float32,
            )
        )

        h, g = model.decode(mu)

        return torch.cat(
            [h, g],
            dim=1,
        ).numpy().astype(float)


# ============================================================
# Load existing research data
# ============================================================

def load_inputs(args, out):

    s106 = load_module(
        "s106_for_118",
        "106_survey_ferritin_anchor.py",
    )

    (
        cohorts,
        frames,
        s103,
        s104,
        weight_audit,
        _,
    ) = s106.read_inputs(
        args.fetch_missing
    )

    cols = s103.X_COLS

    raw = {
        cy: cohorts[cy][cols].to_numpy(float)
        for cy in CYCLES
    }

    for cy in CYCLES:

        if not np.isfinite(raw[cy]).all():
            raise ValueError(
                f"Invalid marker data: {cy}"
            )

    # Common discovery-only input standardization.
    scaler = StandardScaler().fit(
        raw["0506"]
    )

    x = {
        cy: scaler.transform(raw[cy])
        for cy in CYCLES
    }

    anchor = {}
    masks = {}
    coverage = []

    for cy in CYCLES:

        df = cohorts[cy]

        eligible = (
            df.RIAGENDR.eq(2)
            & df.RIDAGEYR.between(20, 49)
            & df.LBXFER.gt(0)
            & df[BASE_COV].notna().all(axis=1)
        )

        masks[cy] = eligible.to_numpy()

        q = df.loc[
            eligible
        ].reset_index(drop=True)

        if len(q) < 45:
            raise ValueError(
                f"Insufficient ferritin sample: {cy}"
            )

        y = np.log(
            q.LBXFER.to_numpy(float)
        )

        w = q.WTSAF2YR.to_numpy(float)

        cov = q[BASE_COV].to_numpy(float)

        if not (
            np.isfinite(y).all()
            and np.isfinite(w).all()
            and np.isfinite(cov).all()
            and (w > 0).all()
        ):
            raise ValueError(
                f"Invalid anchor sample: {cy}"
            )

        anchor[cy] = {
            "frame": q,
            "y": y,
            "w": w,
            "cov": cov,
            "low_ferritin":
                q.LBXFER.to_numpy(float) < 15,
        }

        coverage.append({
            "cohort": cy,
            "representation_sample_n":
                len(df),
            "ferritin_sample_n":
                len(q),
            "effective_weighted_n":
                w.sum() ** 2 / (w ** 2).sum(),
            "weighted_low_ferritin_prevalence":
                weighted_mean(
                    anchor[cy]["low_ferritin"],
                    w,
                ),
            "represented_psus":
                q[
                    ["SDMVSTRA", "SDMVPSU"]
                ].drop_duplicates().shape[0],
        })

    pd.DataFrame(coverage).to_csv(
        out / "118_sample_coverage.csv",
        index=False,
    )

    pd.DataFrame(weight_audit).to_csv(
        out / "118_weight_audit.csv",
        index=False,
    )

    return {
        "s106": s106,
        "s103": s103,
        "s104": s104,
        "cohorts": cohorts,
        "frames": frames,
        "raw": raw,
        "x": x,
        "anchor": anchor,
        "masks": masks,
    }


# ============================================================
# Fit all representation candidates
# ============================================================

def build_representations(data, args, out):

    x = data["x"]
    raw = data["raw"]

    x0 = x["0506"]

    reps = {}
    failures = []
    diagnostics = []

    def add(name, transform):

        values = {
            cy: np.asarray(
                transform(cy),
                dtype=float,
            )
            for cy in CYCLES
        }

        dimension = values["0506"].shape[1]

        if not all(
            np.isfinite(v).all()
            for v in values.values()
        ):
            raise ValueError(
                f"Nonfinite scores: {name}"
            )

        reps[name] = values

        diagnostics.append({
            "method": name,
            "dimension": dimension,
            "status": "fitted",
        })

    # No laboratory predictors.
    add(
        "Covariates",
        lambda cy: np.empty(
            (len(x[cy]), 0)
        ),
    )

    # Full-dimensional information controls.
    add("RawX7", lambda cy: x[cy])
    add("H4", lambda cy: x[cy][:, :4])
    add("G3", lambda cy: x[cy][:, 4:])

    # Conventional two-variable representations.
    add(
        "Threshold2",
        lambda cy: np.column_stack([
            np.maximum(
                0,
                np.where(
                    data["cohorts"][cy]
                    .RIAGENDR.to_numpy() == 1,
                    13.0,
                    12.0,
                ) - raw[cy][:, 0],
            ),
            np.maximum(
                0,
                raw[cy][:, 4] - 5.7,
            ),
        ]),
    )

    add(
        "Hb_HbA1c",
        lambda cy: x[cy][:, [0, 4]],
    )

    add(
        "Hb_RDW",
        lambda cy: x[cy][:, [0, 3]],
    )

    # Global PCA.
    pca = PCA(
        n_components=7,
        svd_solver="full",
    ).fit(x0)

    add(
        "PCA7",
        lambda cy: pca.transform(x[cy]),
    )

    add(
        "PCA2",
        lambda cy:
            pca.transform(x[cy])[:, :2],
    )

    # Domain-specific PCA controls.
    pca_h = PCA(
        n_components=2,
        svd_solver="full",
    ).fit(x0[:, :4])

    pca_g = PCA(
        n_components=2,
        svd_solver="full",
    ).fit(x0[:, 4:])

    add(
        "H_PCA2",
        lambda cy:
            pca_h.transform(x[cy][:, :4]),
    )

    add(
        "G_PCA2",
        lambda cy:
            pca_g.transform(x[cy][:, 4:]),
    )

    # HG7 primary: common unweighted representation fitting.
    # No ferritin values are supplied to fit_hg_cca.
    matrix, mean, rho = (
        data["s104"].fit_hg_cca(
            x0,
            np.ones(len(x0)),
        )
    )

    hg = {
        cy: (x[cy] - mean) @ matrix
        for cy in CYCLES
    }

    add("HG7", lambda cy: hg[cy])

    add(
        "HG_Shared2",
        lambda cy: hg[cy][:, :2],
    )

    add(
        "HG_Discordant2",
        lambda cy: hg[cy][:, 2:4],
    )

    add(
        "HG_HPrivate2",
        lambda cy: hg[cy][:, 4:6],
    )

    add(
        "HG_GPrivate1",
        lambda cy: hg[cy][:, 6:7],
    )

    # Weighted HG sensitivity reproduces the
    # representation-fitting approach of Script106.
    w0 = (
        data["cohorts"]["0506"]
        .WTSAF2YR.to_numpy(float)
    )

    wm, wmu, wrho = (
        data["s104"].fit_hg_cca(
            x0,
            w0,
        )
    )

    weighted_hg = {
        cy: (x[cy] - wmu) @ wm
        for cy in CYCLES
    }

    add(
        "HG_Weighted_Shared2",
        lambda cy: weighted_hg[cy][:, :2],
    )

    dump_json(
        out / "118_hg_reference.json",
        {
            "unweighted_rhos": rho.tolist(),
            "weighted_rhos": wrho.tolist(),
            "fixed_shared_rank": 2,
        },
    )

    # --------------------------------------------------------
    # Official native JIVE and AJIVE
    # --------------------------------------------------------

    if not args.skip_native:

        s114 = load_module(
            "s114_for_118",
            "114_structure_recovery.py",
        )

        s112 = load_module(
            "s112_for_118",
            "112_finish_benchmarks.py",
        )

        helpers = s114.cca_functions(
            EDA /
            "72_shared_private_decomposition_and_phenotype.py"
        )

        native_methods = (
            ("JIVE7", "JIVE_Joint2"),
            ("AJIVE7_adaptive", "AJIVE_Joint2"),
        )

        for native_name, output_name in native_methods:

            print(
                f"Fitting {native_name}",
                flush=True,
            )

            try:

                fitted = s114.fit_method(
                    native_name,
                    x0,
                    helpers,
                    data["s103"],
                    s112,
                    args.rscript,
                    args.timeout,
                )

                transformed = {
                    cy: fitted(x[cy])
                    for cy in CYCLES
                }

                actual_dim = (
                    transformed["0506"]["all"]
                    .shape[1]
                )

                available_joint = (
                    transformed["0506"]["shared"]
                    .shape[1]
                )

                # The primary comparison is 2D, not 7D.
                # A reduced total AJIVE rank is acceptable
                # if two valid native joint scores exist.
                if available_joint < 2:
                    raise RuntimeError(
                        "Fewer than two native joint scores"
                    )

                add(
                    output_name,
                    lambda cy:
                        transformed[cy]["shared"][:, :2],
                )

                diagnostics.append({
                    "method": native_name,
                    "dimension": actual_dim,
                    "status":
                        f"native_joint_dim={available_joint}",
                })

            except Exception:

                failures.append({
                    "method": output_name,
                    "stage": "representation",
                    "error": traceback.format_exc(),
                })

                print(
                    f"UNAVAILABLE: {output_name}",
                    flush=True,
                )

    # --------------------------------------------------------
    # Matched-input SYNERGY-VAE-inspired comparator
    # --------------------------------------------------------

    history = []
    vae_diagnostics = []
    vae_reconstruction = []

    for seed in tqdm(
        VAE_SEEDS,
        desc="Multi-decoder VAE",
    ):

        try:

            model, records, info = fit_vae(
                x0,
                seed,
                args.epochs,
                args.beta,
            )

            name = f"VAE2_seed{seed}"

            add(
                name,
                lambda cy:
                    vae_transform(model, x[cy]),
            )

            history.extend(records)
            vae_diagnostics.append(info)

            for cy in CYCLES:

                reconstructed = vae_reconstruct(
                    model,
                    x[cy],
                )

                error = np.sum(
                    (x[cy] - reconstructed) ** 2
                )

                reference = np.sum(
                    x[cy] ** 2
                )

                vae_reconstruction.append({
                    "cohort": cy,
                    "method": name,
                    "input_reconstruction_R2":
                        1 - error / reference,
                    "latent_dimension": 2,
                    "reconstruction_weighting":
                        "unweighted",
                })

        except Exception:

            failures.append({
                "method": f"VAE2_seed{seed}",
                "stage": "representation",
                "error": traceback.format_exc(),
            })

    pd.DataFrame(history).to_csv(
        out / "118_vae_training.csv",
        index=False,
    )

    pd.DataFrame(vae_diagnostics).to_csv(
        out / "118_vae_diagnostics.csv",
        index=False,
    )

    pd.DataFrame(vae_reconstruction).to_csv(
        out / "118_vae_reconstruction.csv",
        index=False,
    )

    pd.DataFrame(diagnostics).to_csv(
        out / "118_representation_status.csv",
        index=False,
    )

    return reps, failures


# ============================================================
# Identical downstream ferritin model
# ============================================================

def fit_ferritin_models(data, representations, failures):

    anchor = data["anchor"]
    masks = data["masks"]

    w0 = anchor["0506"]["w"]
    y0 = anchor["0506"]["y"]

    cov_scaler = StandardScaler().fit(
        anchor["0506"]["cov"],
        sample_weight=w0,
    )

    cov = {
        cy: cov_scaler.transform(
            anchor[cy]["cov"]
        )
        for cy in CYCLES
    }

    predictions = {}
    scores_diagnostics = []

    for method, values in representations.items():

        try:

            scores = {
                cy: values[cy][masks[cy]]
                for cy in CYCLES
            }

            dimension = scores["0506"].shape[1]

            if dimension > 0:

                whitener = (
                    WeightedWhitener().fit(
                        scores["0506"],
                        w0,
                    )
                )

                normalized = {
                    cy: whitener.transform(
                        scores[cy]
                    )
                    for cy in CYCLES
                }

                design = {
                    cy: np.column_stack([
                        cov[cy],
                        normalized[cy],
                    ])
                    for cy in CYCLES
                }

                condition = float(
                    whitener.eigenvalues.max()
                    / whitener.eigenvalues.min()
                )

            else:

                design = cov
                condition = np.nan

            model = Ridge(
                alpha=RIDGE_ALPHA
            )

            model.fit(
                design["0506"],
                y0,
                sample_weight=w0 / w0.mean(),
            )

            predictions[method] = {
                cy: model.predict(design[cy])
                for cy in CYCLES
            }

            scores_diagnostics.append({
                "method": method,
                "dimension": dimension,
                "training_covariance_condition":
                    condition,
                "ridge_alpha": RIDGE_ALPHA,
            })

        except Exception:

            failures.append({
                "method": method,
                "stage": "ferritin_model",
                "error": traceback.format_exc(),
            })

    for required in (
        "Covariates",
        "RawX7",
        "HG7",
        "HG_Shared2",
    ):

        if required not in predictions:
            raise RuntimeError(
                f"Required method failed: {required}"
            )

    return predictions, scores_diagnostics


# ============================================================
# Paired survey-aware evaluation
# ============================================================

def evaluate(data, predictions, args, out):

    performance = []
    comparisons = []
    seed_sensitivity = []
    invariance = []

    anchor = data["anchor"]

    vae_names = [
        f"VAE2_seed{s}"
        for s in VAE_SEEDS
        if f"VAE2_seed{s}" in predictions
    ]

    vae_complete = (
        len(vae_names) == len(VAE_SEEDS)
    )

    for index, cy in enumerate(CYCLES):

        q = anchor[cy]

        y = q["y"]
        w = q["w"]

        errors = {
            method: (
                y - values[cy]
            ) ** 2
            for method, values in predictions.items()
        }

        auc_values = {
            method: weighted_auc(
                q["low_ferritin"],
                -values[cy],
                w,
            )
            for method, values in predictions.items()
        }

        # Expected single-seed algorithm performance.
        # Average LOSSES, not predictions.
        # No best-seed selection or ensemble advantage.
        if vae_complete:

            errors["VAE2_mean_loss"] = np.mean(
                [
                    errors[name]
                    for name in vae_names
                ],
                axis=0,
            )

            auc_values["VAE2_mean_loss"] = float(
                np.mean([
                    auc_values[name]
                    for name in vae_names
                ])
            )

        variance = weighted_mean(
            (y - weighted_mean(y, w)) ** 2,
            w,
        )

        base_mse = weighted_mean(
            errors["Covariates"],
            w,
        )

        for method, error in errors.items():

            mse = float(
                weighted_mean(error, w)
            )

            performance.append({
                "cohort": cy,
                "method": method,
                "n": len(y),
                "weighted_mse": mse,
                "weighted_R2":
                    1 - mse / variance,
                "delta_mse_vs_covariates":
                    base_mse - mse,
                "low_ferritin_weighted_AUC":
                    auc_values[method],
                "is_training_period":
                    cy == "0506",
            })

        # Numerical equivalence of whitened full-rank
        # linear feature representations.
        for method in ("HG7", "PCA7"):

            if method in predictions:

                diff = float(
                    np.max(
                        np.abs(
                            predictions[method][cy]
                            - predictions["RawX7"][cy]
                        )
                    )
                )

                invariance.append({
                    "cohort": cy,
                    "method": method,
                    "max_abs_prediction_difference":
                        diff,
                    "equivalence_pass":
                        diff < 1e-6,
                })

        for method in vae_names:

            seed_sensitivity.append({
                "cohort": cy,
                "method": method,
                "weighted_mse":
                    weighted_mean(
                        errors[method], w
                    ),
                "HG_gain_vs_seed":
                    weighted_mean(
                        errors[method]
                        - errors[PRIMARY_METHOD],
                        w,
                    ),
            })

        if cy not in HOLDOUTS:
            continue

        # Use the full original DEMO design frame.
        # Draw once and reuse across EVERY method.
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [SEED, index, 999]
            )
        )

        bootstrap_weights = []

        for _ in tqdm(
            range(args.replicates),
            desc=f"PSU bootstrap {cy}",
        ):

            bw = (
                data["s106"].rescaled_psu_weights(
                    q["frame"],
                    data["frames"][cy],
                    rng,
                )
            )

            bootstrap_weights.append(
                bw / bw.sum()
            )

        bw = np.asarray(
            bootstrap_weights
        )

        # Direct pairwise comparisons, rather than
        # comparing separate confidence intervals.
        pairs = [
            (PRIMARY_METHOD, reference)
            for reference in errors
            if reference != PRIMARY_METHOD
        ]

        pairs.extend([
            (method, "Covariates")
            for method in errors
            if method != "Covariates"
            and method != PRIMARY_METHOD
        ])

        for candidate, reference in pairs:

            delta = (
                errors[reference]
                - errors[candidate]
            )

            point = float(
                weighted_mean(delta, w)
            )

            draws = bw @ delta

            low, high = np.percentile(
                draws,
                [2.5, 97.5],
            )

            if low > 0:
                direction = "favours_candidate"

            elif high < 0:
                direction = "favours_reference"

            else:
                direction = "inconclusive"

            comparisons.append({
                "cohort": cy,
                "candidate": candidate,
                "reference": reference,
                "delta_mse_positive_candidate":
                    point,
                "psu_low95": low,
                "psu_high95": high,
                "relative_mse_reduction":
                    point / weighted_mean(
                        errors[reference], w
                    ),
                "resamples": args.replicates,
                "direction": direction,
            })

    pd.DataFrame(performance).to_csv(
        out / "118_ferritin_performance.csv",
        index=False,
    )

    pd.DataFrame(comparisons).to_csv(
        out / "118_paired_comparisons.csv",
        index=False,
    )

    pd.DataFrame(seed_sensitivity).to_csv(
        out / "118_vae_seed_sensitivity.csv",
        index=False,
    )

    pd.DataFrame(invariance).to_csv(
        out / "118_full_rank_invariance.csv",
        index=False,
    )

    return (
        pd.DataFrame(performance),
        pd.DataFrame(comparisons),
        vae_complete,
        invariance,
    )


# ============================================================
# Claim-specific evidence gate
# ============================================================

def evidence_gate(pairs, vae_complete):

    rows = []

    for reference in MATCHED_COMPARATORS:

        subset = pairs.loc[
            pairs.candidate.eq(PRIMARY_METHOD)
            & pairs.reference.eq(reference)
        ]

        if set(subset.cohort) != set(HOLDOUTS):

            verdict = "unavailable"

        elif subset.psu_low95.gt(0).all():

            verdict = (
                "HG_advantage_both_periods"
            )

        elif subset.psu_high95.lt(0).all():

            verdict = (
                "comparator_advantage_both_periods"
            )

        elif subset.psu_low95.gt(0).any():

            verdict = "period_specific_HG_advantage"

        elif subset.psu_high95.lt(0).any():

            verdict = "period_specific_HG_disadvantage"

        else:

            verdict = "inconclusive"

        row = {
            "candidate": PRIMARY_METHOD,
            "reference": reference,
            "verdict": verdict,
        }

        for cy in HOLDOUTS:

            selected = subset.loc[
                subset.cohort.eq(cy)
            ]

            if len(selected) == 1:

                r = selected.iloc[0]

                row[f"delta_{cy}"] = (
                    r.delta_mse_positive_candidate
                )

                row[f"low95_{cy}"] = r.psu_low95
                row[f"high95_{cy}"] = r.psu_high95

            else:

                row[f"delta_{cy}"] = np.nan
                row[f"low95_{cy}"] = np.nan
                row[f"high95_{cy}"] = np.nan

        rows.append(row)

    result = pd.DataFrame(rows)

    return result


# ============================================================
# Publication figure
# ============================================================

def make_figure(performance, pairs, out):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    order = [
        "Threshold2",
        "Hb_HbA1c",
        "Hb_RDW",
        "PCA2",
        "H_PCA2",
        "G_PCA2",
        "JIVE_Joint2",
        "AJIVE_Joint2",
        "VAE2_mean_loss",
        "HG_Discordant2",
        "HG_HPrivate2",
        "RawX7",
    ]

    labels = {
        "Threshold2": "Threshold burdens",
        "Hb_HbA1c": "Hb + HbA1c",
        "Hb_RDW": "Hb + RDW",
        "PCA2": "PCA",
        "H_PCA2": "Haematology PCA",
        "G_PCA2": "Glycaemia PCA",
        "JIVE_Joint2": "JIVE joint",
        "AJIVE_Joint2": "AJIVE joint",
        "VAE2_mean_loss": "Multi-decoder VAE",
        "HG_Discordant2": "HG discordant",
        "HG_HPrivate2": "HG haematology-private",
        "RawX7": "Raw markers (7D)",
    }

    df = pairs.loc[
        pairs.candidate.eq(PRIMARY_METHOD)
    ].copy()

    order = [
        name for name in order
        if name in set(df.reference)
    ]

    if not order:
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9, 5.5),
        sharey=True,
    )

    for ax, cy, title in zip(
        axes,
        HOLDOUTS,
        ["a   2007–2008", "b   2021–2023"],
    ):

        panel = (
            df.loc[df.cohort.eq(cy)]
            .set_index("reference")
            .reindex(order)
        )

        for i, row in enumerate(
            panel.itertuples()
        ):

            value = (
                row.delta_mse_positive_candidate
            )

            lo = row.psu_low95
            hi = row.psu_high95

            if not np.isfinite(
                [value, lo, hi]
            ).all():
                continue

            color = (
                "#176D69"
                if lo > 0
                else "#A64B39"
                if hi < 0
                else "#7A8289"
            )

            ax.hlines(
                i, lo, hi,
                color=color,
                linewidth=1.4,
            )

            ax.scatter(
                value,
                i,
                color=color,
                s=24,
                zorder=3,
            )

        ax.axvline(
            0,
            color="#444444",
            linestyle="--",
            linewidth=0.8,
        )

        ax.set_yticks(
            range(len(order)),
            [labels[name] for name in order],
        )

        ax.set_title(
            title,
            loc="left",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlabel(
            "Paired reduction in log-ferritin MSE"
        )

    axes[0].invert_yaxis()

    fig.suptitle(
        "Ferritin anchoring: HG shared pair versus alternatives",
        fontsize=11,
    )

    fig.text(
        0.55,
        0.02,
        "Positive differences favour HG shared2. "
        "Approximate holdout PSU intervals.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[0, 0.06, 1, 0.94]
    )

    fig.savefig(
        out / "118_ferritin_comparison.png",
        dpi=400,
        bbox_inches="tight",
    )

    fig.savefig(
        out / "118_ferritin_comparison.svg",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Self-test
# ============================================================

def self_test():

    rng = np.random.default_rng(SEED)

    x = rng.normal(
        size=(500, 7)
    )

    y = x @ rng.normal(size=7)

    w = rng.uniform(
        0.5, 2.0, len(x)
    )

    a = rng.normal(
        size=(7, 7)
    )

    while np.linalg.cond(a) > 100:

        a = rng.normal(
            size=(7, 7)
        )

    z = x @ a

    wx = WeightedWhitener().fit(
        x, w
    ).transform(x)

    wz = WeightedWhitener().fit(
        z, w
    ).transform(z)

    p1 = Ridge(
        alpha=10
    ).fit(
        wx, y,
        sample_weight=w / w.mean(),
    ).predict(wx)

    p2 = Ridge(
        alpha=10
    ).fit(
        wz, y,
        sample_weight=w / w.mean(),
    ).predict(wz)

    assert np.max(
        np.abs(p1 - p2)
    ) < 1e-7

    model, _, _ = fit_vae(
        x,
        SEED,
        epochs=3,
        beta=0.05,
    )

    scores = vae_transform(model, x)

    reconstructed = vae_reconstruct(
        model, x
    )

    assert scores.shape == (500, 2)
    assert reconstructed.shape == (500, 7)
    assert np.isfinite(scores).all()

    print("PASS: weighted whitening invariance")
    print("PASS: multi-decoder VAE")
    print("PASS: reconstruction dimensions")


# ============================================================
# Entry point
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=999,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--run-name",
        default="118_biomedical_utility",
    )

    parser.add_argument(
        "--rscript",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
    )

    parser.add_argument(
        "--fetch-missing",
        action="store_true",
    )

    parser.add_argument(
        "--skip-native",
        action="store_true",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.replicates < 39:
        parser.error(
            "Use at least 39 PSU replicates"
        )

    if args.epochs < 10:
        parser.error("Insufficient epochs")

    if args.beta < 0:
        parser.error("Invalid beta")

    if not args.run_name or not all(
        c.isalnum() or c in "_-"
        for c in args.run_name
    ):
        parser.error("Invalid run name")

    if args.rscript is None:

        args.rscript = (
            Path(sys.prefix)
            / "Lib/R/bin/x64/Rscript.exe"
        )

        if not args.rscript.is_file():

            candidate = shutil.which("Rscript")

            if candidate:
                args.rscript = Path(candidate)

    if (
        not args.skip_native
        and not args.rscript.is_file()
    ):
        raise FileNotFoundError(
            "Rscript not found"
        )

    out = EDA / "Results" / args.run_name

    if out.exists():
        raise FileExistsError(
            f"{out} already exists. "
            "Use another --run-name."
        )

    out.mkdir(parents=True)

    dump_json(
        out / "118_protocol.json",
        {
            "seed": SEED,
            "settings": vars(args),
            "source_hashes": source_hashes(),
            "primary_outcome": "log_ferritin",
            "secondary_outcome":
                "ferritin_below_15_ug_L_AUC",
            "primary_candidate": PRIMARY_METHOD,
            "matched_comparators":
                MATCHED_COMPARATORS,
            "reference_period": "0506",
            "evaluation_periods": HOLDOUTS,
            "physiological_fit_weights":
                "unweighted primary comparison",
            "downstream_weights":
                "official fasting WTSAF2YR",
            "downstream_model":
                "weighted ridge with fixed alpha",
            "ridge_alpha": RIDGE_ALPHA,
            "downstream_whitening": True,
            "vae_seeds": VAE_SEEDS,
            "vae_architecture":
                "two-domain multi-decoder adaptation",
            "vae_reference":
                "10.1371/journal.pdig.0001719",
            "vae_aggregation":
                "mean held-out loss across fixed seeds",
            "uncertainty":
                "paired rescaled holdout PSU bootstrap",
            "ferritin_used_for_encoder": False,
            "later_periods_previously_inspected": True,
            "independent_population_validation": False,
            "multiplicity_adjusted": False,
            "assay_harmonized": False,
            "torch_version": torch.__version__,
        },
    )

    print(
        "Loading Script106 ferritin cohorts",
        flush=True,
    )

    data = load_inputs(args, out)

    print(
        "Fitting discovery representations",
        flush=True,
    )

    representations, failures = (
        build_representations(
            data,
            args,
            out,
        )
    )

    print(
        "Fitting frozen ferritin models",
        flush=True,
    )

    predictions, diagnostics = (
        fit_ferritin_models(
            data,
            representations,
            failures,
        )
    )

    pd.DataFrame(diagnostics).to_csv(
        out / "118_downstream_diagnostics.csv",
        index=False,
    )

    pd.DataFrame(
        failures,
        columns=["method", "stage", "error"],
    ).to_csv(
        out / "118_failures.csv",
        index=False,
    )

    performance, paired, vae_complete, invariance = (
        evaluate(
            data,
            predictions,
            args,
            out,
        )
    )

    gate = evidence_gate(
        paired,
        vae_complete,
    )

    gate.to_csv(
        out / "118_evidence_gate.csv",
        index=False,
    )

    make_figure(
        performance,
        paired,
        out,
    )

    invariant_pass = (
        len(invariance) == 6
        and all(
            row["equivalence_pass"]
            for row in invariance
        )
    )

    lines = [
        "# Script 118 — Biomedical utility benchmark",
        "",
        "## Primary scientific question",
        "",
        "Does the HG shared pair retain more accessible "
        "ferritin-associated information than alternative "
        "representations under a matched 2D budget?",
        "",
        "## Direct comparisons",
        "",
        "```text",
        gate.round(5).to_string(index=False),
        "```",
        "",
        "## Held-out performance",
        "",
        "```text",
        performance.loc[
            ~performance.is_training_period
        ].round(5).to_string(index=False),
        "```",
        "",
        "## Interpretation constraints",
        "",
        "- Ferritin is excluded from representation training.",
        "- Ferritin is an assay anchor, not an independent cohort.",
        "- All representations use identical marker inputs, "
        "except explicitly restricted baseline panels.",
        "- All ferritin models share identical participants, "
        "covariates and weighted ridge settings.",
        "- Whitening controls for arbitrary linear score "
        "scaling and rotation.",
        "- VAE results concern a two-domain adaptation, "
        "not the published SYNERGY-VAE model.",
        "- VAE seed averaging uses losses, not predictions.",
        "- Confidence intervals condition on frozen fits.",
        "- No multiplicity adjustment or confirmatory inference.",
        "- Low ferritin AUC is descriptive, not diagnostic validation.",
        "- Ferritin assays and complete-case selection may differ "
        "across periods.",
        "- Equivalent or adverse results must be retained.",
        "",
        f"Full-rank equivalence passed: {invariant_pass}",
        f"All VAE seeds available: {vae_complete}",
        f"Recorded exceptions: {len(failures)}",
    ]

    (out / "118_READOUT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    dump_json(
        out / "118_completion.json",
        {
            "execution_completed": True,
            "full_rank_equivalence_pass":
                invariant_pass,
            "all_vae_seeds_available":
                vae_complete,
            "fitted_methods":
                list(predictions),
            "recorded_exceptions":
                len(failures),
            "all_primary_comparators_available":
                bool(
                    gate.verdict.ne(
                        "unavailable"
                    ).all()
                ),
        },
    )

    print()
    print(gate.round(5).to_string(index=False))
    print()
    print("=" * 72)
    print("SCRIPT 118 COMPLETE")
    print(f"Results: {out}")
    print(f"Full-rank invariance: {invariant_pass}")
    print(f"All VAE seeds available: {vae_complete}")
    print(f"Recorded exceptions: {len(failures)}")
    print("=" * 72)

    if not invariant_pass:
        raise RuntimeError(
            "Full-rank invariance check failed. "
            "Do not interpret comparative findings "
            "before investigating."
        )


if __name__ == "__main__":
    main()
