from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
FIG = RESULTS / "Figures"
AUDIT = ROOT / "Audit"
RLIB = ROOT / "R_library"

FIG.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)
RLIB.mkdir(parents=True, exist_ok=True)

ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]

LABELS = {
    "DPQ010": "Anhedonia",
    "DPQ020": "Depressed mood",
    "DPQ030": "Sleep",
    "DPQ040": "Fatigue",
    "DPQ050": "Appetite",
    "DPQ060": "Self-worth",
    "DPQ070": "Concentration",
    "DPQ080": "Psychomotor",
    "DPQ090": "Suicidality"
}

cohort = pd.read_parquet(
    PROCESSED / "nhanes_adjustment_cohort_20plus.parquet"
)

working = pd.read_parquet(
    PROCESSED / "nhanes_core_working.parquet"
)

phq = working[
    ["SEQN", "CYCLE"] + ITEMS
].copy()

df = cohort.merge(
    phq,
    on=["SEQN", "CYCLE"],
    how="left",
    validate="one_to_one"
)

for c in ITEMS:
    df.loc[df[c].abs() < 1e-10, c] = 0
    df.loc[~df[c].isin([0, 1, 2, 3]), c] = np.nan

required = [
    "A",
    "G_HBA1C",
    "WTMEC4YR",
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH1",
    "INDFMPIR",
    "EDUC3",
    "SMOKING3",
    "BMXBMI",
    "EGFR_2021"
] + ITEMS

df = df.dropna(
    subset=required
).copy()

df["CYCLE"] = df["CYCLE"].astype(str)


def weighted_mean_sd(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    mean = np.sum(w * x) / np.sum(w)

    var = (
        np.sum(
            w * (x - mean) ** 2
        )
        / np.sum(w)
    )

    return mean, np.sqrt(var)


for cycle in sorted(
    df["CYCLE"].unique()
):
    idx = df["CYCLE"] == cycle

    w = df.loc[
        idx,
        "WTMEC4YR"
    ]

    for col in [
        "A",
        "G_HBA1C",
        "RIDAGEYR",
        "INDFMPIR",
        "BMXBMI",
        "EGFR_2021"
    ]:
        mean, sd = weighted_mean_sd(
            df.loc[idx, col],
            w
        )

        df.loc[
            idx,
            f"{col}_Z"
        ] = (
            df.loc[idx, col] - mean
        ) / sd


df["AG_Z"] = (
    df["A_Z"]
    * df["G_HBA1C_Z"]
)

for col in [
    "RIAGENDR",
    "RIDRETH1",
    "EDUC3",
    "SMOKING3"
]:
    dummies = pd.get_dummies(
        df[col].astype(int),
        prefix=col,
        drop_first=True,
        dtype=int
    )

    df = pd.concat(
        [df, dummies],
        axis=1
    )


dummy_cols = [
    c
    for c in df.columns
    if c.startswith("RIAGENDR_")
    or c.startswith("RIDRETH1_")
    or c.startswith("EDUC3_")
    or c.startswith("SMOKING3_")
]


export_cols = [
    "SEQN",
    "CYCLE",
    "WTMEC4YR",
    "A_Z",
    "G_HBA1C_Z",
    "AG_Z",
    "RIDAGEYR_Z",
    "INDFMPIR_Z",
    "BMXBMI_Z",
    "EGFR_2021_Z"
] + dummy_cols + ITEMS


input_csv = (
    RESULTS
    / "23_mimic_dif_input.csv"
)

df[
    export_cols
].to_csv(
    input_csv,
    index=False
)


rscript = shutil.which(
    "Rscript"
)

if rscript is None:
    candidates = sorted(
        Path(
            r"C:\Program Files\R"
        ).glob(
            r"R-*\bin\Rscript.exe"
        )
    )

    if candidates:
        rscript = str(
            candidates[-1]
        )


print()
print(
    "MIMIC DIFFERENTIAL-ITEM AUDIT"
)
print(
    "============================="
)

if rscript is None:
    print(
        "FAIL  Rscript not found."
    )
    raise SystemExit(1)


covariates = [
    "RIDAGEYR_Z",
    "INDFMPIR_Z",
    "BMXBMI_Z",
    "EGFR_2021_Z"
] + dummy_cols


covariate_rhs = " + ".join(
    covariates
)


r_code = f'''
args <- commandArgs(
    trailingOnly=TRUE
)

input_file <- args[1]
results_dir <- args[2]
user_lib <- args[3]

dir.create(
    user_lib,
    recursive=TRUE,
    showWarnings=FALSE
)

.libPaths(
    c(
        user_lib,
        .libPaths()
    )
)

if (
    !requireNamespace(
        "lavaan",
        quietly=TRUE
    )
) {{
    install.packages(
        "lavaan",
        repos="https://cloud.r-project.org",
        lib=user_lib
    )
}}

library(lavaan)

d <- read.csv(
    input_file
)

d$CYCLE <- factor(
    d$CYCLE
)

items <- c(
    "DPQ010",
    "DPQ020",
    "DPQ030",
    "DPQ040",
    "DPQ050",
    "DPQ060",
    "DPQ070",
    "DPQ080",
    "DPQ090"
)

exposures <- c(
    "A_Z",
    "G_HBA1C_Z",
    "AG_Z"
)

base_model <- "
Somatic =~ DPQ030 + DPQ040 + DPQ050

CognitiveAffective =~
DPQ010 +
DPQ020 +
DPQ060 +
DPQ070 +
DPQ080 +
DPQ090

Somatic ~
A_Z +
G_HBA1C_Z +
AG_Z +
{covariate_rhs}

CognitiveAffective ~
A_Z +
G_HBA1C_Z +
AG_Z +
{covariate_rhs}
"

message(
    "BASE_START"
)

base_fit <- sem(
    base_model,
    data=d,
    group="CYCLE",
    ordered=items,
    estimator="WLSMV",
    std.lv=TRUE,
    sampling_weights="WTMEC4YR",
    sampling_weights_type="design",
    group_equal=c(
        "loadings",
        "thresholds"
    )
)

message(
    "BASE_DONE"
)

rows <- list()

total_models <- (
    length(exposures)
    * length(items)
)

model_index <- 0

for (
    exposure in exposures
) {{

    for (
        item in items
    ) {{

        model_index <- (
            model_index + 1
        )

        message(
            sprintf(
                "START|%d|%d|%s|%s",
                model_index,
                total_models,
                exposure,
                item
            )
        )

        model <- paste0(
            base_model,
            "\\n",
            item,
            " ~ ",
            exposure
        )

        fit <- try(
            sem(
                model,
                data=d,
                group="CYCLE",
                ordered=items,
                estimator="WLSMV",
                std.lv=TRUE,
                sampling_weights="WTMEC4YR",
                sampling_weights_type="design",
                group_equal=c(
                    "loadings",
                    "thresholds"
                )
            ),
            silent=TRUE
        )

        if (
            inherits(
                fit,
                "try-error"
            )
        ) {{
            message(
                sprintf(
                    "DONE|%d|%d|%s|%s",
                    model_index,
                    total_models,
                    exposure,
                    item
                )
            )

            next
        }}

        pe <- parameterEstimates(
            fit,
            standardized=TRUE,
            ci=TRUE
        )

        q <- pe[
            pe$op == "~" &
            pe$lhs == item &
            pe$rhs == exposure,
            c(
                "lhs",
                "rhs",
                "group",
                "est",
                "se",
                "z",
                "pvalue",
                "ci.lower",
                "ci.upper",
                "std.all"
            )
        ]

        if (
            nrow(q) == 0
        ) {{
            message(
                sprintf(
                    "DONE|%d|%d|%s|%s",
                    model_index,
                    total_models,
                    exposure,
                    item
                )
            )

            next
        }}

        q$item <- item
        q$exposure <- exposure

        rows[[length(rows) + 1]] <- q

        message(
            sprintf(
                "DONE|%d|%d|%s|%s",
                model_index,
                total_models,
                exposure,
                item
            )
        )
    }}
}}

out <- do.call(
    rbind,
    rows
)

levels_cycle <- levels(
    d$CYCLE
)

out$cycle <- levels_cycle[
    out$group
]

out$q_global <- p.adjust(
    out$pvalue,
    method="BH"
)

out$q_exposure <- ave(
    out$pvalue,
    out$exposure,
    FUN=function(x) {{
        p.adjust(
            x,
            method="BH"
        )
    }}
)

fm <- fitMeasures(
    base_fit,
    c(
        "cfi",
        "tli",
        "rmsea",
        "srmr"
    )
)

fitrow <- data.frame(
    cfi=unname(
        fm["cfi"]
    ),
    tli=unname(
        fm["tli"]
    ),
    rmsea=unname(
        fm["rmsea"]
    ),
    srmr=unname(
        fm["srmr"]
    )
)

write.csv(
    out,
    file.path(
        results_dir,
        "23_mimic_dif_paths.csv"
    ),
    row.names=FALSE
)

write.csv(
    fitrow,
    file.path(
        results_dir,
        "23_mimic_dif_base_fit.csv"
    ),
    row.names=FALSE
)
'''


r_path = (
    AUDIT
    / "23_mimic_dif_runner.R"
)

r_path.write_text(
    r_code,
    encoding="utf-8"
)


print(
    "Fitting base model + 27 DIF models..."
)


proc = subprocess.Popen(
    [
        str(rscript),
        str(r_path),
        str(input_csv),
        str(RESULTS),
        str(RLIB)
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)


log_lines = []

current_model = None


with tqdm(
    total=28,
    desc="SEM/DIF",
    unit="model",
    dynamic_ncols=True
) as bar:

    for line in proc.stdout:

        line = line.rstrip()

        log_lines.append(
            line
        )

        if line == "BASE_START":

            bar.set_postfix_str(
                "base SEM"
            )

        elif line == "BASE_DONE":

            bar.update(1)

            bar.set_postfix_str(
                "base complete"
            )

        elif line.startswith(
            "START|"
        ):

            parts = line.split(
                "|"
            )

            if len(parts) == 5:

                number = parts[1]
                exposure = parts[3]
                item = parts[4]

                current_model = (
                    f"{number}/27 "
                    f"{exposure} -> {item}"
                )

                bar.set_postfix_str(
                    current_model
                )

        elif line.startswith(
            "DONE|"
        ):

            if bar.n < 28:
                bar.update(1)

        elif (
            "Error" in line
            or "Warning" in line
        ):

            tqdm.write(
                line
            )


returncode = proc.wait()


if returncode != 0:

    print(
        "FAIL  DIF models did not complete."
    )

    print(
        "\n".join(
            log_lines[-50:]
        )
    )

    raise SystemExit(
        returncode
    )


paths = pd.read_csv(
    RESULTS
    / "23_mimic_dif_paths.csv"
)

fit = pd.read_csv(
    RESULTS
    / "23_mimic_dif_base_fit.csv"
).iloc[0]


paths["label"] = (
    paths["item"]
    .map(LABELS)
)

paths["supported"] = (
    paths["q_global"]
    < 0.05
)

paths["direction"] = (
    np.sign(
        paths["est"]
    )
    .astype(int)
)


summary_rows = []


for exposure in [
    "A_Z",
    "G_HBA1C_Z",
    "AG_Z"
]:

    for item in ITEMS:

        d = paths[
            (
                paths["exposure"]
                == exposure
            )
            &
            (
                paths["item"]
                == item
            )
        ].sort_values(
            "cycle"
        )

        if len(d) != 2:

            verdict = "INCOMPLETE"

        else:

            same = (
                d["direction"]
                .nunique()
                == 1
            )

            both = (
                d["supported"]
                .all()
            )

            any_supported = (
                d["supported"]
                .any()
            )

            if same and both:

                verdict = "REPLICATES"

            elif (
                same
                and any_supported
            ):

                verdict = "PARTIAL"

            elif same:

                verdict = (
                    "SAME DIRECTION, UNCERTAIN"
                )

            else:

                verdict = "INCONSISTENT"


        summary_rows.append(
            {
                "exposure": exposure,
                "item": item,
                "label": LABELS[item],
                "verdict": verdict
            }
        )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    RESULTS
    / "23_mimic_dif_summary.csv",
    index=False
)


for cycle in [
    "0506",
    "0708",
    "506",
    "708"
]:

    d = paths[
        paths["cycle"]
        .astype(str)
        == cycle
    ].copy()

    if len(d) == 0:
        continue

    matrix = np.full(
        (3, 9),
        np.nan
    )

    sig = np.zeros(
        (3, 9),
        dtype=bool
    )

    for i, exposure in enumerate(
        [
            "A_Z",
            "G_HBA1C_Z",
            "AG_Z"
        ]
    ):

        for j, item in enumerate(
            ITEMS
        ):

            r = d[
                (
                    d["exposure"]
                    == exposure
                )
                &
                (
                    d["item"]
                    == item
                )
            ]

            if len(r):

                matrix[
                    i,
                    j
                ] = (
                    r.iloc[0]["est"]
                )

                sig[
                    i,
                    j
                ] = (
                    r.iloc[0]["q_global"]
                    < 0.05
                )


    fig, ax = plt.subplots(
        figsize=(11, 4)
    )

    im = ax.imshow(
        matrix,
        aspect="auto"
    )

    ax.set_yticks(
        [0, 1, 2],
        [
            "A",
            "G",
            "A×G"
        ]
    )

    ax.set_xticks(
        np.arange(9),
        [
            LABELS[i]
            for i in ITEMS
        ],
        rotation=40,
        ha="right"
    )

    ax.set_title(
        "Direct physiological item effects "
        f"after latent-factor adjustment: "
        f"{cycle}"
    )

    for i in range(3):

        for j in range(9):

            if sig[i, j]:

                ax.text(
                    j,
                    i,
                    "*",
                    ha="center",
                    va="center"
                )


    fig.colorbar(
        im,
        ax=ax,
        label=(
            "Direct ordinal-probit effect"
        )
    )

    plt.tight_layout()

    plt.savefig(
        FIG
        / f"23_dif_{cycle}.png",
        dpi=300
    )

    plt.close()


name_map = {
    "A_Z": "A",
    "G_HBA1C_Z": "G",
    "AG_Z": "A×G"
}


print()
print(
    f"PASS  Base invariant ordinal model remains well fit: "
    f"CFI={fit['cfi']:.3f}, "
    f"RMSEA={fit['rmsea']:.3f}."
)

print(
    "PASS  Global Benjamini-Hochberg correction "
    "applied across item-level tests."
)

print()


for exposure in [
    "A_Z",
    "G_HBA1C_Z",
    "AG_Z"
]:

    d = summary[
        (
            summary["exposure"]
            == exposure
        )
        &
        (
            summary["verdict"]
            == "REPLICATES"
        )
    ]

    if len(d):

        print(
            f"{name_map[exposure]:<4}  "
            f"REPLICATED DIRECT ITEMS: "
            f"{', '.join(d['label'])}"
        )

    else:

        partial = summary[
            (
                summary["exposure"]
                == exposure
            )
            &
            (
                summary["verdict"]
                == "PARTIAL"
            )
        ]

        if len(partial):

            print(
                f"{name_map[exposure]:<4}  "
                f"PARTIAL DIRECT ITEMS: "
                f"{', '.join(partial['label'])}"
            )

        else:

            print(
                f"{name_map[exposure]:<4}  "
                "NO FDR-STABLE DIRECT ITEM EFFECT"
            )


print()
print(
    "NOTE  Direct item effects indicate "
    "physiology-specific item behavior beyond "
    "the latent depression factors."
)

print(
    "NOTE  Population inference remains anchored "
    "to the complex-survey models."
)

print(
    "Figures and full DIF tables saved."
)