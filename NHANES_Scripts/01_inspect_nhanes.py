from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression\Data\NHANES")

cycles = {
    "0506": ["CBC_D.XPT", "DEMO_D.xpt", "DPQ_D.XPT", "GHB_D.XPT", "GLU_D.XPT", "OPXRET_D.XPT"],
    "0708": ["CBC_E.XPT", "DEMO_E.XPT", "DPQ_E.XPT", "GHB_E.XPT", "GLU_E.XPT", "OPXRET_E.XPT"]
}

for cycle, files in cycles.items():
    print(f"\n{'='*70}\n{cycle}\n{'='*70}")

    dfs = {}

    for file in files:
        path = ROOT / cycle / file
        df = pd.read_sas(path, format="xport")
        dfs[file] = df

        print(f"\n{file}")
        print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
        print(list(df.columns))

    print("\nSEQN overlap")
    common = None

    for file, df in dfs.items():
        ids = set(df["SEQN"].dropna())
        print(f"{file}: {len(ids):,}")

        common = ids if common is None else common & ids
        print(f"Current intersection: {len(common):,}")

    print(f"\nALL SIX FILES MATCHED: {len(common):,}")