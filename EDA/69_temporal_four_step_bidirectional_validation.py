#!/usr/bin/env python
# =============================================================================
# 69_temporal_four_step_bidirectional_validation.py
#
# Robust temporal version.
#
# IMPORTANT:
# We DO NOT assume that one scaffold contains all raw biomarkers.
# The Script-63 table is used only as the MASTER participant/period/frozen-PC
# table. Raw biomarkers are restored by searching the repository for any CSV
# or NHANES XPT file containing the required variables and merging by SEQN.
#
# This matches how the earlier EDA had to restore raw physiology.
#
# Exact frozen experiment:
#   X0 --f--> Y1 --g--> X1 --f--> Y2 --g--> X2
#
# Learn ONLY on 2005-08:
#   f : X -> Y
#   g : Y -> X
#   ICA map
#   X standardisation
#
# Test without relearning on 2009-18 and 2021-23.
# =============================================================================

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "EDA" / "Results"
MODELS = ROOT / "EDA" / "Models"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# MASTER table: period + survey weight + frozen PCs.
# -------------------------------------------------------------------------
MASTER_CANDIDATES = [
    ROOT / "EDA" / "Results" / "63_deep_ddx_input.csv",
    ROOT / "Results" / "63_deep_ddx_input.csv",
]
MASTER_FILE = next((p for p in MASTER_CANDIDATES if p.exists()), None)
if MASTER_FILE is None:
    raise FileNotFoundError("Could not find 63_deep_ddx_input.csv")

master = pd.read_csv(MASTER_FILE)

for c in ["SEQN", "PERIOD", "SURVEY_WT"]:
    if c not in master.columns:
        raise ValueError(f"Master file missing {c}")

master["SEQN"] = pd.to_numeric(master["SEQN"], errors="coerce")
master["PERIOD"] = master["PERIOD"].astype(str)

def detect(options):
    for cols in options:
        if all(c in master.columns for c in cols):
            return cols
    return None

A_PCS = detect([
    ["A_OI_PC1_FZ","A_OI_PC2_FZ","A_OI_PC3_FZ"],
    ["A_FROZEN_PC1_FZ","A_FROZEN_PC2_FZ","A_FROZEN_PC3_FZ"],
])
G_PCS = detect([
    ["G3_OI_PC1_FZ","G3_OI_PC2_FZ","G3_OI_PC3_FZ"],
    ["G3_FROZEN_PC1_FZ","G3_FROZEN_PC2_FZ","G3_FROZEN_PC3_FZ"],
])
if A_PCS is None or G_PCS is None:
    raise ValueError("Could not detect frozen A/G PCA columns in master.")

A_RAW = ["LBXHGB","LBXRBCSI","LBXMCVSI","LBXRDW"]
G_BASE = ["LBXGH","LBXGLU"]
INS_NAMES = ["LOG_IN","LBXIN","LBDINSI"]  # log insulin, insulin, SI insulin
WANTED = set(A_RAW + G_BASE + INS_NAMES)

# -------------------------------------------------------------------------
# Build raw-biomarker bank by SEQN from repository sources.
# NHANES SEQN is unique across cycles, so we do NOT need period in raw files.
# Master supplies PERIOD after merge.
# -------------------------------------------------------------------------
source_rows = []
pieces = []

def add_piece(df, source):
    if "SEQN" not in df.columns:
        return

    present = [c for c in df.columns if c in WANTED]
    if not present:
        return

    q = df[["SEQN"] + present].copy()
    q["SEQN"] = pd.to_numeric(q["SEQN"], errors="coerce")
    q = q.dropna(subset=["SEQN"])
    for c in present:
        q[c] = pd.to_numeric(q[c], errors="coerce")

    # Within a source, collapse duplicate SEQN by first nonmissing value.
    def first_valid(s):
        z = s.dropna()
        return z.iloc[0] if len(z) else np.nan

    q = q.groupby("SEQN", as_index=False).agg(
        {c:first_valid for c in present}
    )

    pieces.append((source, q))
    source_rows.append({
        "source": str(source),
        "rows": len(q),
        "variables": "|".join(present),
    })

# Search CSVs first.
skip_names = {
    "69_temporal_raw_bank.csv",
    "69_temporal_four_step_summary.csv",
    "69_temporal_four_step_details.csv",
    "69_temporal_four_step_availability.csv",
    "69_raw_source_inventory.csv",
}

for p in ROOT.rglob("*.csv"):
    if p.name in skip_names:
        continue
    # Avoid generated environment/cache folders.
    low = str(p).lower()
    if any(x in low for x in ["\\.git\\", "\\site-packages\\", "\\.venv\\"]):
        continue
    try:
        # Header first; only fully read potentially useful files.
        cols = pd.read_csv(p, nrows=0).columns.tolist()
        if "SEQN" in cols and any(c in WANTED for c in cols):
            add_piece(pd.read_csv(p), p)
    except Exception:
        pass

# Search likely NHANES XPT laboratory files too.
xpt_patterns = ("CBC", "GHB", "GLU", "INS")
xpts = list(ROOT.rglob("*.XPT")) + list(ROOT.rglob("*.xpt"))
seen_xpt = set()

for p in xpts:
    if p.resolve() in seen_xpt:
        continue
    seen_xpt.add(p.resolve())
    if not any(tag in p.name.upper() for tag in xpt_patterns):
        continue
    try:
        df = pd.read_sas(p, format="xport", encoding="latin1")
        add_piece(df, p)
    except Exception:
        pass

inventory = pd.DataFrame(source_rows)
inventory.to_csv(OUT/"69_raw_source_inventory.csv", index=False)

if not pieces:
    raise RuntimeError(
        "No raw biomarker sources with SEQN were found. "
        "See EDA/Results/69_raw_source_inventory.csv"
    )

# Combine source pieces column-by-column using first available nonmissing value.
all_seqn = sorted(set().union(*[
    set(q["SEQN"].dropna().tolist()) for _, q in pieces
]))
bank = pd.DataFrame({"SEQN": all_seqn}).set_index("SEQN")

# Prefer more complete sources for each variable.
for var in WANTED:
    candidates = []
    for source, q in pieces:
        if var in q.columns:
            s = q.set_index("SEQN")[var]
            candidates.append((int(s.notna().sum()), str(source), s))
    candidates.sort(key=lambda x: x[0], reverse=True)

    if candidates:
        out = pd.Series(np.nan, index=bank.index, dtype=float)
        for _, _, s in candidates:
            out = out.combine_first(s.reindex(bank.index))
        bank[var] = out

bank = bank.reset_index()

# Create LOG_IN robustly.
if "LOG_IN" not in bank.columns:
    bank["LOG_IN"] = np.nan

# Prefer direct LOG_IN if found. Otherwise derive from insulin.
if "LBXIN" in bank.columns:
    ins = pd.to_numeric(bank["LBXIN"], errors="coerce")
    derived = pd.Series(np.where(ins > 0, np.log(ins), np.nan), index=bank.index)
    bank["LOG_IN"] = bank["LOG_IN"].combine_first(derived)

# LBDINSI is pmol/L; 1 uU/mL ~= 6.0 pmol/L. Only use when LBXIN absent.
if "LBDINSI" in bank.columns:
    si = pd.to_numeric(bank["LBDINSI"], errors="coerce")
    insulin_uuml = si / 6.0
    derived_si = pd.Series(
        np.where(insulin_uuml > 0, np.log(insulin_uuml), np.nan),
        index=bank.index
    )
    bank["LOG_IN"] = bank["LOG_IN"].combine_first(derived_si)

bank.to_csv(OUT/"69_temporal_raw_bank.csv", index=False)

# Merge onto master.
needed_master = ["SEQN","PERIOD","SURVEY_WT"] + A_PCS + G_PCS
m = master[needed_master].copy()

# Master may contain repeated SEQN only if malformed; NHANES should not.
if m["SEQN"].duplicated().any():
    raise RuntimeError("Master has duplicate SEQN values; refusing ambiguous merge.")

merged = m.merge(bank, on="SEQN", how="left", validate="one_to_one")

PERIODS = ["2005-2008","2009-2018","2021-2023"]
DOMAINS = {
    "A":{"raw":A_RAW,"pcs":A_PCS},
    "G":{"raw":G_BASE+["LOG_IN"],"pcs":G_PCS},
}

def clean_period(df, period, raw_cols, pc_cols):
    q = df[df["PERIOD"].eq(period)].copy()
    cols = raw_cols + pc_cols + ["SURVEY_WT"]
    for c in cols:
        if c not in q.columns:
            q[c] = np.nan
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q = q.dropna(subset=cols)
    q = q[q["SURVEY_WT"] > 0].copy()
    return q

# Write availability BEFORE any modelling so a failure is diagnostic.
availability = []
for domain,spec in DOMAINS.items():
    for period in PERIODS:
        q = clean_period(merged, period, spec["raw"], spec["pcs"])
        availability.append({
            "domain":domain,
            "period":period,
            "n_complete_raw_plus_frozenPC":len(q),
            **{
                f"n_nonmissing_{v}":
                int(pd.to_numeric(
                    merged.loc[merged["PERIOD"].eq(period), v],
                    errors="coerce"
                ).notna().sum())
                if v in merged.columns else 0
                for v in spec["raw"]
            }
        })

availability_df = pd.DataFrame(availability)
availability_df.to_csv(
    OUT/"69_temporal_four_step_availability.csv", index=False
)

# Fail transparently with exact missing block, not an assumption.
future_bad = availability_df[
    availability_df["period"].isin(["2009-2018","2021-2023"]) &
    (availability_df["n_complete_raw_plus_frozenPC"] < 20)
]
if len(future_bad):
    print("\nRAW SOURCE INVENTORY")
    print(inventory.to_string(index=False))
    print("\nAVAILABILITY")
    print(availability_df.to_string(index=False))
    raise RuntimeError(
        "Future raw physiology still cannot be reconstructed from local sources. "
        "The tables above show exactly which variables are missing and which "
        "files were found. No temporal result has been claimed."
    )

def weighted_standardize(X,w):
    mu=np.sum(w[:,None]*X,axis=0)/np.sum(w)
    sd=np.sqrt(np.sum(w[:,None]*(X-mu)**2,axis=0)/np.sum(w))
    return (X-mu)/sd,mu,sd

def wls(A,B,w):
    D=np.column_stack([np.ones(len(A)),A])
    sw=np.sqrt(w)
    return np.linalg.lstsq(D*sw[:,None],B*sw[:,None],rcond=None)[0]

def pred(A,C):
    return np.column_stack([np.ones(len(A)),A])@C

def epsilon(target,estimate,w):
    target=np.asarray(target,float)
    estimate=np.asarray(estimate,float)
    mu=np.sum(w[:,None]*target,axis=0)/np.sum(w)
    sse=np.sum(w[:,None]*(target-estimate)**2)
    sst=np.sum(w[:,None]*(target-mu)**2)
    return float(np.sqrt(sse/sst))

def wrmse(target,estimate,w):
    return float(np.sqrt(
        np.sum(w[:,None]*(target-estimate)**2) /
        (np.sum(w)*target.shape[1])
    ))

summary=[]
details=[]
model_meta={}

for domain,spec in DOMAINS.items():
    rcols,pcols=spec["raw"],spec["pcs"]

    # Train ONLY on discovery.
    disc=clean_period(merged,"2005-2008",rcols,pcols)
    Xd=disc[rcols].to_numpy(float)
    Zd=disc[pcols].to_numpy(float)
    wd=disc["SURVEY_WT"].to_numpy(float)

    Xd_s,xmu,xsd=weighted_standardize(Xd,wd)

    ica=FastICA(
        n_components=3,
        whiten="unit-variance",
        algorithm="parallel",
        fun="logcosh",
        max_iter=3000,
        tol=1e-6,
        random_state=42
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Yd=ica.fit_transform(Zd)

    f=wls(Xd_s,Yd,wd)
    g=wls(Yd,Xd_s,wd)

    model_meta[domain]={
        "discovery_n":len(disc),
        "X_dimension":len(rcols),
        "Y_dimension":3,
        "ICA_iterations":int(ica.n_iter_)
    }

    for period in PERIODS:
        q=clean_period(merged,period,rcols,pcols)

        X0_raw=q[rcols].to_numpy(float)
        Z=q[pcols].to_numpy(float)
        w=q["SURVEY_WT"].to_numpy(float)

        # Freeze discovery scaling and ICA.
        X0=(X0_raw-xmu)/xsd
        Yref=ica.transform(Z)

        # Exact four-step loop.
        Y1=pred(X0,f)
        X1=pred(Y1,g)
        Y2=pred(X1,f)
        X2=pred(Y2,g)

        for name,target,estimate in [
            ("MAIN_cycle_epsilon_f__Y1_vs_Y2",Y1,Y2),
            ("MAIN_cycle_epsilon_g__X1_vs_X2",X1,X2),
            ("TRANSFER_forward__Yref_vs_Y1",Yref,Y1),
            ("TRANSFER_backward__X0_vs_X1",X0,X1),
            ("TRANSFER_full__X0_vs_X2",X0,X2),
        ]:
            ep=epsilon(target,estimate,w)
            summary.append({
                "domain":domain,
                "period":period,
                "test":name,
                "epsilon":ep,
                "R2_equivalent":1-ep**2,
                "weighted_RMSE_standardized":wrmse(target,estimate,w),
                "n":len(q)
            })

        X1_raw=X1*xsd+xmu
        X2_raw=X2*xsd+xmu

        for j,var in enumerate(rcols):
            for label,target_raw,estimate_raw in [
                ("X0_vs_X1__backward_transfer",X0_raw[:,j],X1_raw[:,j]),
                ("X0_vs_X2__full_transfer",X0_raw[:,j],X2_raw[:,j]),
                ("X1_vs_X2__g_cycle",X1_raw[:,j],X2_raw[:,j]),
            ]:
                mu=np.sum(w*target_raw)/np.sum(w)
                sse=np.sum(w*(target_raw-estimate_raw)**2)
                sst=np.sum(w*(target_raw-mu)**2)
                details.append({
                    "domain":domain,
                    "period":period,
                    "comparison":label,
                    "variable":var,
                    "epsilon":float(np.sqrt(sse/sst)),
                    "R2_equivalent":float(1-sse/sst),
                    "RMSE_raw_units":float(np.sqrt(sse/np.sum(w))),
                    "n":len(q)
                })

summary_df=pd.DataFrame(summary)
details_df=pd.DataFrame(details)

summary_df.to_csv(OUT/"69_temporal_four_step_summary.csv",index=False)
details_df.to_csv(OUT/"69_temporal_four_step_details.csv",index=False)

manifest={
    "script":"69_temporal_four_step_bidirectional_validation.py",
    "master":str(MASTER_FILE),
    "raw_restoration":"repository-wide SEQN merge from CSV and likely NHANES XPT lab files",
    "training":"2005-08 only",
    "testing":["2009-2018","2021-2023"],
    "chain":"X0 --f--> Y1 --g--> X1 --f--> Y2 --g--> X2",
    "future_relearning":False,
    "pca_refit":False,
    "ica_refit_future":False,
    "locked_outputs_modified":False,
    "models":model_meta
}
(OUT/"69_temporal_four_step_manifest.json").write_text(
    json.dumps(manifest,indent=2),encoding="utf-8"
)

print("PASS  Script 69 complete.")
print("PASS  Raw physiology restored by SEQN from actual repository sources.")
print("PASS  f, g, scaling and ICA learned ONLY on 2005-08.")
print("PASS  Same frozen routes tested on 2009-18 and 2021-23.")
print()
print("AVAILABILITY")
print(availability_df.to_string(index=False))
print()
print("RESULTS")
print(summary_df.to_string(index=False))
