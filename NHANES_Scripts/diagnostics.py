import pandas as pd

p = r"Data\NHANES\0506\DPQ_D.XPT"
df = pd.read_sas(p, format="xport")

items = ["DPQ010","DPQ020","DPQ030","DPQ040","DPQ050",
         "DPQ060","DPQ070","DPQ080","DPQ090"]

print(df[items].dtypes)

for c in items:
    print("\n", c)
    print(df[c].value_counts(dropna=False).sort_index())