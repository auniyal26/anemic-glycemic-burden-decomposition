#!/usr/bin/env python3
"""Experiment 90: audit reversible quantized coding on EXISTING NHANES-derived Z rows.

Requires EDA/Results/73_frozen_phenotype_map_input.csv from Script 73.
NO synthetic fallback, NO estimation from Y, and NO edits to frozen data.

Methods: signed integer map, fixed-width decimal interleaving, nested Cantor
pairing. Exact inverse is defined only relative to the quantized integers.
"""
from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

ZCOLS = ["SP_SHARED1", "SP_SHARED2", "SP_DISCORD1", "SP_DISCORD2",
         "SP_A_PRIVATE3", "SP_A_PRIVATE4", "SP_G_PRIVATE3"]
NAME = ["Shared1", "Shared2", "Discord1", "Discord2", "A_private3", "A_private4", "G_private3"]
ROOT = Path(__file__).resolve().parent


def quant(x: float, decimals: int) -> int:
    if not math.isfinite(x):
        raise ValueError("Nonfinite latent component")
    return int((Decimal(str(float(x))) * (10 ** decimals)).to_integral_value(rounding=ROUND_HALF_EVEN))


def to_nat(x: int) -> int:
    return 2*x if x >= 0 else -2*x-1


def from_nat(u: int) -> int:
    return (u//2) if u % 2 == 0 else -(u+1)//2


def pair(a: int, b: int) -> int:
    t = a+b
    return t*(t+1)//2 + b


def unpair(c: int) -> tuple[int, int]:
    t = (math.isqrt(8*c+1)-1)//2
    b = c-t*(t+1)//2
    return t-b, b


def pair7(vals: tuple[int, ...]) -> int:
    out = vals[0]
    for v in vals[1:]:
        out = pair(out, v)
    return out


def unpair7(c: int, d: int = 7) -> tuple[int, ...]:
    vals = []
    for _ in range(d-1):
        c, b = unpair(c)
        vals.append(b)
    vals.append(c)
    return tuple(reversed(vals))


def interleave(vals: tuple[int, ...], width: int) -> str:
    digits = [str(v).zfill(width) for v in vals]
    if any(len(s) != width for s in digits):
        raise ValueError("Integer digit width exceeded")
    return "".join(s[i] for i in range(width) for s in digits)


def deinterleave(code: str, dims: int, width: int) -> tuple[int, ...]:
    if len(code) != dims*width or not code.isdigit():
        raise ValueError("Invalid interleaved code / metadata")
    return tuple(int(code[j::dims]) for j in range(dims))


def read_real(path: Path, period: str | None, limit: int | None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing real Script 73 output: {path}\nRun EDA/73_frozen_phenotype_map_time_travel.py first.")
    headers = pd.read_csv(path, nrows=0).columns.tolist()
    required = [*ZCOLS, "PERIOD", "SEQN"]
    missing = sorted(set(required)-set(headers))
    if missing:
        raise ValueError(f"Script 73 file lacks: {missing}")
    optional = [x for x in ("CYCLE", "PHQ9_TOTAL") if x in headers]
    df = pd.read_csv(path, usecols=required+optional, low_memory=False)
    if period and period.lower() != "all":
        df = df[df["PERIOD"].astype(str).eq(period)]
    for col in ZCOLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df[ZCOLS].notna().all(axis=1)].copy()
    if limit is not None:
        df = df.head(limit)
    if df.empty:
        raise ValueError(f"No complete seven-coordinate real Z rows for period={period}")
    return df


def run(df: pd.DataFrame, decimals: list[int], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    X = df[ZCOLS].to_numpy(float)
    ids = df["SEQN"].astype(str).tolist()
    periods = df["PERIOD"].astype(str).tolist()
    n, dims = X.shape
    metrics, axis_metrics, examples = [], [], []

    for dp in decimals:
        states = [tuple(to_nat(quant(v, dp)) for v in x) for x in X]
        # Fixed digit width is part of metadata; we retain leading zeros as strings.
        width = max(len(str(v)) for state in states for v in state)
        recon = np.empty_like(X)
        codes = []
        bits = []
        for i, state in enumerate(states):
            code = interleave(state, width)
            if deinterleave(code, dims, width) != state:
                raise AssertionError(f"Interleave round-trip failed at row {i}")
            cantor = pair7(state)
            if unpair7(cantor, dims) != state:
                raise AssertionError(f"Cantor round-trip failed at row {i}")
            recon[i] = [from_nat(v)/(10**dp) for v in state]
            codes.append(code)
            bits.append(cantor.bit_length())
        err = np.abs(recon-X)
        max_err = float(err.max())
        assert max_err <= (0.5*10**(-dp)+1e-12)
        distinct = len(set(codes))
        collisions = sum(v-1 for v in Counter(codes).values() if v>1)
        metrics.append({"decimals":dp, "n_real_rows":n, "distinct_codes":distinct,
                        "duplicated_code_rows":collisions, "max_abs_Z_error":max_err,
                        "mean_abs_Z_error":float(err.mean()),
                        "roundtrip_failures_relative_to_quantized_Z":0,
                        "digits_per_component":width, "interleave_total_digits":7*width,
                        "mean_nested_cantor_bits":float(np.mean(bits)),
                        "max_nested_cantor_bits":max(bits)})
        for j, name in enumerate(NAME):
            axis_metrics.append({"decimals":dp, "component":name,
                                 "z_observed_sd":float(np.std(X[:,j])),
                                 "mean_abs_round_error":float(err[:,j].mean()),
                                 "max_abs_round_error":float(err[:,j].max()),
                                 "relative_mae_to_observed_sd":float(err[:,j].mean()/max(np.std(X[:,j]),1e-12))})
        for i in range(min(n, 15)):
            row = {"decimals":dp, "SEQN":ids[i], "PERIOD":periods[i],
                   "code_as_string_NOT_float":codes[i], "cantor_bits":bits[i]}
            row.update({"original_"+NAME[j]:float(X[i,j]) for j in range(dims)})
            row.update({"decoded_"+NAME[j]:float(recon[i,j]) for j in range(dims)})
            examples.append(row)
        if "PHQ9_TOTAL" in df:
            y = pd.to_numeric(df["PHQ9_TOTAL"],errors="coerce").to_numpy()
            m = np.isfinite(y) & (np.abs(y-10)<1e-8)
            metrics[-1].update({"rows_with_observed_PHQ9_10":int(m.sum()),
                                "distinct_Z_codes_at_observed_PHQ9_10":len({codes[i] for i in np.flatnonzero(m)})})

    pd.DataFrame(metrics).to_csv(out/"90_real_encoding_summary.csv",index=False)
    pd.DataFrame(axis_metrics).to_csv(out/"90_component_precision_audit.csv",index=False)
    pd.DataFrame(examples).to_csv(out/"90_patient_code_examples.csv",index=False)
    meta = {"provenance":"REAL frozen Script 73 NHANES-derived Z (no simulated fallback)",
            "n_complete_Z_rows":n, "period_counts":dict(Counter(periods)),
            "decimals":decimals,
            "interpretation":"Encoding exactly recovers only the rounded values. A code is NOT a uniquely identifiable physiological state given PHQ-9.",
            "files":["90_real_encoding_summary.csv","90_component_precision_audit.csv","90_patient_code_examples.csv"]}
    (out/"90_run_manifest.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"\nResults saved: {out.resolve()}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"Results"/"73_frozen_phenotype_map_input.csv")
    p.add_argument("--period",default="2005-2008",help='2005-2008, 2009-2018, 2021-2023, or all')
    p.add_argument("--decimals",default="2,3,4",help="Comma-separated number of decimals")
    p.add_argument("--max-rows",type=int,default=None,help="Optional first N complete real observations")
    p.add_argument("--out",type=Path,default=ROOT/"Results")
    a=p.parse_args()
    ds=[int(x) for x in a.decimals.split(',')]
    if not ds or not all(0<=d<=6 for d in ds):
        p.error("Supported precisions: 0 to 6 decimal places")
    df=read_real(a.data,a.period,a.max_rows)
    run(df,ds,a.out)


if __name__=="__main__":
    main()
