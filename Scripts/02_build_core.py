from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
DATA = ROOT / "Data" / "NHANES"
OUT = ROOT / "Data" / "Processed"
OUT.mkdir(parents=True, exist_ok=True)

cycles = {
    "0506": {
        "cbc": "CBC_D.XPT",
        "demo": "DEMO_D.xpt",
        "dpq": "DPQ_D.XPT",
        "ghb": "GHB_D.XPT",
        "glu": "GLU_D.XPT",
        "ret": "OPXRET_D.XPT",
    },
    "0708": {
        "cbc": "CBC_E.XPT",
        "demo": "DEMO_E.XPT",
        "dpq": "DPQ_E.XPT",
        "ghb": "GHB_E.XPT",
        "glu": "GLU_E.XPT",
        "ret": "OPXRET_E.XPT",
    },
}

frames = []

for cycle, f in cycles.items():
    p = DATA / cycle

    cbc = pd.read_sas(p / f["cbc"], format="xport")
    demo = pd.read_sas(p / f["demo"], format="xport")
    dpq = pd.read_sas(p / f["dpq"], format="xport")
    ghb = pd.read_sas(p / f["ghb"], format="xport")

    df = (
        demo.merge(cbc, on="SEQN", how="inner", validate="one_to_one")
        .merge(ghb, on="SEQN", how="inner", validate="one_to_one")
        .merge(dpq, on="SEQN", how="inner", validate="one_to_one")
    )

    df["CYCLE"] = cycle
    frames.append(df)

reference = pd.concat(frames, ignore_index=True)
reference.to_parquet(OUT / "nhanes_core_reference.parquet", compression="zstd")

work = reference.copy()
work = work[work["RIDAGEYR"] >= 18].copy()

items = [f"DPQ0{i}0" for i in range(1, 10)]

for c in items:
    work.loc[work[c].abs() < 1e-10, c] = 0
    work.loc[~work[c].isin([0, 1, 2, 3]), c] = pd.NA

work["PHQ9_TOTAL"] = work[items].sum(axis=1, min_count=9)
work["DEPRESSION_10"] = (
    work["PHQ9_TOTAL"].ge(10).where(work["PHQ9_TOTAL"].notna()).astype("Int64")
)

keep = [
    "SEQN", "CYCLE",
    "RIDAGEYR", "RIAGENDR", "RIDRETH1",
    "WTMEC2YR", "SDMVPSU", "SDMVSTRA",
    "LBXHGB", "LBXRBCSI", "LBXHCT",
    "LBXMCVSI", "LBXMCHSI", "LBXMC", "LBXRDW",
    "LBXGH", "PHQ9_TOTAL", "DEPRESSION_10"
] + items

work = work[keep]
work.to_parquet(OUT / "nhanes_core_working.parquet", compression="zstd")

print("Raw matched:", len(reference))
print("Adults:", len(work))
print("Complete Hb:", work["LBXHGB"].notna().sum())
print("Complete HbA1c:", work["LBXGH"].notna().sum())
print("Complete PHQ-9:", work["PHQ9_TOTAL"].notna().sum())

complete = work.dropna(subset=["LBXHGB", "LBXGH", "PHQ9_TOTAL"])
print("Complete A/G/P:", len(complete))
print()
print(complete[["LBXHGB", "LBXGH", "PHQ9_TOTAL"]].describe())
print()
print("PHQ-9 >= 10:", complete["DEPRESSION_10"].sum())