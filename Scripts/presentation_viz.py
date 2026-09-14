# -*- coding: utf-8 -*-
"""
37_build_presentation_visuals.py

Builds the main-deck presentation visuals for:
Computational Decomposition of Physiological Contributors to Depressive Phenotypes

Design principles:
- non-destructive;
- formal presentation language;
- reuse existing vetted GitHub assets where available;
- generate only the conceptual figures that do not already exist;
- output a clean, deck-ready visual set.

Expected repository layout:
repo_root/
    Scripts/
        37_build_presentation_visuals.py
    Results/
        Presentation_Final_Candidates/
            Figures/
            Tables/

Outputs:
repo_root/
    Results/
        Presentation_Deck_Visuals/
            Main/
            Appendix/
            slide_visual_manifest.csv
"""

from pathlib import Path
import shutil
import textwrap
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageOps, ImageDraw, ImageFont


# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC_FIG = ROOT / "Results" / "Presentation_Final_Candidates" / "Figures"
SRC_TBL = ROOT / "Results" / "Presentation_Final_Candidates" / "Tables"

OUT_ROOT = ROOT / "Results" / "Presentation_Deck_Visuals"
OUT_MAIN = OUT_ROOT / "Main"
OUT_APPENDIX = OUT_ROOT / "Appendix"

OUT_MAIN.mkdir(parents=True, exist_ok=True)
OUT_APPENDIX.mkdir(parents=True, exist_ok=True)

MANIFEST_ROWS = []


# -----------------------------------------------------------------------------
# GLOBAL FORMATTING
# -----------------------------------------------------------------------------

plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11

CANVAS_W = 1800
CANVAS_H = 1000
PIL_FONT = ImageFont.load_default()


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def register(slide_id, title, outfile, source_type, notes=""):
    MANIFEST_ROWS.append({
        "slide_id": slide_id,
        "title": title,
        "outfile": str(outfile),
        "source_type": source_type,
        "notes": notes
    })


def wrap(s, width=28):
    return "\n".join(textwrap.wrap(s, width=width))


def new_figure(figsize=(12, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save_matplotlib(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_box(ax, x, y, w, h, text, fontsize=11, lw=1.6):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=lw, edgecolor="black", facecolor="white"
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, wrap(text, 24),
            ha="center", va="center", fontsize=fontsize)
    return box


def add_arrow(ax, x1, y1, x2, y2, text=None, fontsize=10):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle="-|>", mutation_scale=14,
                          linewidth=1.4, color="black")
    ax.add_patch(arr)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, wrap(text, 20),
                ha="center", va="center", fontsize=fontsize)


def placeholder_image(text, width=CANVAS_W, height=CANVAS_H):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    msg = wrap(text, 40)
    bbox = draw.multiline_textbbox((0, 0), msg, font=PIL_FONT, spacing=6)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.multiline_text(((width - tw) / 2, (height - th) / 2),
                        msg, fill="black", font=PIL_FONT, spacing=6, align="center")
    return img


def load_or_placeholder(path, width=CANVAS_W, height=CANVAS_H):
    if path.exists():
        return Image.open(path).convert("RGB")
    return placeholder_image(f"Missing source asset:\n{path.name}", width, height)


def copy_or_placeholder(src_name, dst_name, slide_id, title, notes=""):
    src = SRC_FIG / src_name
    dst = OUT_MAIN / dst_name
    if src.exists():
        shutil.copy2(src, dst)
        register(slide_id, title, dst, "existing_direct_copy", notes or src_name)
    else:
        img = placeholder_image(f"Missing source asset:\n{src_name}")
        img.save(dst)
        register(slide_id, title, dst, "placeholder_missing_source", notes or src_name)
    return dst


def make_panel(image_names, panel_titles, out_path, figure_title="", ncols=2):
    images = [load_or_placeholder(SRC_FIG / name) for name in image_names]
    n = len(images)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    pad = 30
    title_space = 90 if figure_title else 30
    label_space = 40
    cell_w = (CANVAS_W - pad * (ncols + 1)) // ncols
    cell_h = (CANVAS_H - title_space - pad * (nrows + 1)) // nrows

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    draw = ImageDraw.Draw(canvas)

    if figure_title:
        draw.text((pad, 20), figure_title, fill="black", font=PIL_FONT)

    for i, img in enumerate(images):
        row = i // ncols
        col = i % ncols
        x0 = pad + col * (cell_w + pad)
        y0 = title_space + pad + row * (cell_h + pad)

        title = panel_titles[i] if i < len(panel_titles) else f"Panel {i+1}"
        draw.text((x0, y0 - label_space + 10), title, fill="black", font=PIL_FONT)

        inner_h = cell_h - 10
        thumb = ImageOps.contain(img, (cell_w, inner_h))
        paste_x = x0 + (cell_w - thumb.size[0]) // 2
        paste_y = y0 + (inner_h - thumb.size[1]) // 2
        canvas.paste(thumb, (paste_x, paste_y))

        draw.rectangle([x0, y0, x0 + cell_w, y0 + inner_h], outline="black", width=1)

    canvas.save(out_path)


def write_manifest():
    df = pd.DataFrame(MANIFEST_ROWS)
    df.sort_values(["slide_id", "outfile"], inplace=True)
    manifest_path = OUT_ROOT / "slide_visual_manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"SAVED manifest: {manifest_path}")


# -----------------------------------------------------------------------------
# CONCEPTUAL FIGURES
# -----------------------------------------------------------------------------

def slide02_motivation():
    fig, ax = new_figure()
    ax.text(0.02, 0.95,
            "Clinical and Translational Motivation",
            ha="left", va="top", fontsize=16, fontweight="bold")

    add_box(ax, 0.06, 0.58, 0.22, 0.18,
            "Anemia burden\n\nHigh prevalence and broad systemic relevance")
    add_box(ax, 0.39, 0.58, 0.22, 0.18,
            "Dysglycemia burden\n\nPrediabetes and diabetes affect multiple organs and systems")
    add_box(ax, 0.72, 0.58, 0.22, 0.18,
            "Depressive symptom burden\n\nCommon, heterogeneous, and clinically consequential")

    add_box(ax, 0.28, 0.22, 0.44, 0.18,
            "Translational problem:\ncoexisting burdens create ambiguity in observed phenotype,\nmeasurement, and interpretation")
    add_arrow(ax, 0.17, 0.58, 0.45, 0.40)
    add_arrow(ax, 0.50, 0.58, 0.50, 0.40)
    add_arrow(ax, 0.83, 0.58, 0.55, 0.40)

    out = OUT_MAIN / "S02_clinical_translational_motivation.png"
    save_matplotlib(fig, out)
    register("S02", "Clinical and Translational Motivation", out, "generated_conceptual")


def slide03_domain_coupling():
    fig, ax = new_figure()
    ax.text(0.02, 0.95,
            "The Domains Are Coupled, but Not Equivalent",
            ha="left", va="top", fontsize=16, fontweight="bold")

    add_box(ax, 0.10, 0.62, 0.22, 0.16, "Hematological state")
    add_box(ax, 0.68, 0.62, 0.22, 0.16, "Glycemic state")
    add_box(ax, 0.39, 0.18, 0.22, 0.16, "Depressive phenotype")

    add_arrow(ax, 0.32, 0.70, 0.68, 0.70, "Measurement and statistical coupling")
    add_arrow(ax, 0.22, 0.62, 0.46, 0.34, "Association")
    add_arrow(ax, 0.78, 0.62, 0.54, 0.34, "Association")

    ax.text(0.50, 0.50,
            "Key distinction:\nassociation does not imply equivalence,\nand measurement coupling does not imply shared biology.",
            ha="center", va="center", fontsize=11)

    out = OUT_MAIN / "S03_domains_coupled_not_equivalent.png"
    save_matplotlib(fig, out)
    register("S03", "The Domains Are Coupled, but Not Equivalent", out, "generated_conceptual")


def slide04_phenotype_ambiguity():
    fig, ax = new_figure()
    ax.text(0.02, 0.95,
            "The Observed Phenotype Does Not Identify Its Physiological Contributors",
            ha="left", va="top", fontsize=16, fontweight="bold")

    add_box(ax, 0.07, 0.60, 0.28, 0.18,
            "Participant A\n\nPredominantly hematological physiological disturbance")
    add_box(ax, 0.07, 0.28, 0.28, 0.18,
            "Participant B\n\nPredominantly glycemic physiological disturbance")

    add_box(ax, 0.63, 0.44, 0.28, 0.18,
            "Observed phenotype\n\nSimilar PHQ-9 profile or similar somatic symptom score")

    add_arrow(ax, 0.35, 0.69, 0.63, 0.54)
    add_arrow(ax, 0.35, 0.37, 0.63, 0.54)

    ax.text(0.50, 0.18,
            r"$P_1 \approx P_2 \;\; \mathrm{does\ not\ imply} \;\; x_{\mathrm{phys},1} = x_{\mathrm{phys},2}$",
            ha="center", va="center", fontsize=13)

    out = OUT_MAIN / "S04_observed_phenotype_ambiguity.png"
    save_matplotlib(fig, out)
    register("S04", "Observed Phenotype Ambiguity", out, "generated_conceptual")


def slide06_why_open():
    fig, ax = new_figure()
    ax.text(0.02, 0.95,
            "Why Has This Exact Problem Remained Open?",
            ha="left", va="top", fontsize=16, fontweight="bold")

    add_box(ax, 0.04, 0.62, 0.20, 0.18,
            "Clinical epidemiology\n\nEstablishes associations and risk-factor models")
    add_box(ax, 0.28, 0.62, 0.20, 0.18,
            "Psychometrics\n\nStudies phenotype structure and symptom dimensions")
    add_box(ax, 0.52, 0.62, 0.20, 0.18,
            "Representation and decomposition methods\n\nFocus on signals or latent structure")
    add_box(ax, 0.76, 0.62, 0.20, 0.18,
            "Validation and transportability\n\nStudy generalization across settings and time")

    add_box(ax, 0.27, 0.20, 0.46, 0.20,
            "Open synthesis problem:\nreference-preserving physiological representation,\ncomponent separation, phenotype mapping,\nreconstruction accounting, and transfer testing")

    add_arrow(ax, 0.14, 0.62, 0.42, 0.40)
    add_arrow(ax, 0.38, 0.62, 0.48, 0.40)
    add_arrow(ax, 0.62, 0.62, 0.56, 0.40)
    add_arrow(ax, 0.86, 0.62, 0.62, 0.40)

    out = OUT_MAIN / "S06_why_problem_remained_open.png"
    save_matplotlib(fig, out)
    register("S06", "Why Has This Exact Problem Remained Open?", out, "generated_conceptual")


def slide07_rqs():
    fig, ax = new_figure()
    ax.text(0.02, 0.95,
            "Thesis Objective and Research Questions",
            ha="left", va="top", fontsize=16, fontweight="bold")

    add_box(ax, 0.18, 0.73, 0.64, 0.14,
            "Objective:\nDevelop and validate a reference-preserving computational framework\nfor separating physiological contributors to heterogeneous clinical phenotypes.",
            fontsize=12)

    add_box(ax, 0.06, 0.40, 0.20, 0.18,
            "RQ1\nCan hematological and glycemic information be represented as distinguishable components?")
    add_box(ax, 0.29, 0.40, 0.20, 0.18,
            "RQ2\nDo the components associate differently with depressive symptom dimensions?")
    add_box(ax, 0.52, 0.40, 0.20, 0.18,
            "RQ3\nHow much information is lost during decomposition and reconstruction?")
    add_box(ax, 0.75, 0.40, 0.20, 0.18,
            "RQ4\nWhich components transfer, recalibrate, or fail across time and population?")

    ax.text(0.50, 0.17,
            "Interaction is treated as secondary.\nThe primary thesis problem is separability, preservation, and transfer.",
            ha="center", va="center", fontsize=11)

    out = OUT_MAIN / "S07_thesis_objective_and_rqs.png"
    save_matplotlib(fig, out)
    register("S07", "Thesis Objective and Research Questions", out, "generated_conceptual")


def slide09_reference_preserving():
    fig, ax = new_figure()
    ax.text(0.02, 0.95,
            "Reference-Preserving Decomposition",
            ha="left", va="top", fontsize=16, fontweight="bold")

    add_box(ax, 0.04, 0.58, 0.16, 0.14, "Measured physiological inputs\n\nCBC, Hb, HbA1c, glucose, covariates")
    add_box(ax, 0.25, 0.58, 0.16, 0.14, r"Preserved source state" + "\n\n" + r"$x_{\mathrm{phys}}$")
    add_box(ax, 0.46, 0.58, 0.16, 0.14, r"Working representation" + "\n\n" + r"$y = T(x_{\mathrm{phys}})$")
    add_box(ax, 0.71, 0.68, 0.18, 0.12, r"$y_A$")
    add_box(ax, 0.71, 0.52, 0.18, 0.12, r"$y_G$")
    add_box(ax, 0.71, 0.36, 0.18, 0.12, r"$y_{shared}$ or $y_{AG}$")

    add_box(ax, 0.46, 0.18, 0.20, 0.12, r"Reconstruction" + "\n" + r"$\hat{y}=R(y_A, y_G, y_{shared}, \ldots)$")
    add_box(ax, 0.73, 0.18, 0.18, 0.12, r"Preservation loss" + "\n" + r"$L_y=d(y,\hat{y})$")
    add_box(ax, 0.04, 0.18, 0.28, 0.12,
            r"Separate phenotype branch" + "\n" + r"$P \sim f_A(y_A) + f_G(y_G) + f_{shared}(y_{shared}) + f_X(X)$")

    add_arrow(ax, 0.20, 0.65, 0.25, 0.65)
    add_arrow(ax, 0.41, 0.65, 0.46, 0.65)
    add_arrow(ax, 0.62, 0.65, 0.71, 0.74)
    add_arrow(ax, 0.62, 0.65, 0.71, 0.58)
    add_arrow(ax, 0.62, 0.65, 0.71, 0.42)
    add_arrow(ax, 0.80, 0.68, 0.58, 0.30)
    add_arrow(ax, 0.80, 0.52, 0.58, 0.30)
    add_arrow(ax, 0.80, 0.36, 0.58, 0.30)
    add_arrow(ax, 0.66, 0.24, 0.73, 0.24)
    add_arrow(ax, 0.46, 0.58, 0.18, 0.30)

    ax.text(0.50, 0.05,
            "The thesis-level reconstruction test remains proposed work.\nThe current preliminary study establishes feasibility rather than completion.",
            ha="center", va="bottom", fontsize=10)

    out = OUT_MAIN / "S09_reference_preserving_decomposition.png"
    save_matplotlib(fig, out)
    register("S09", "Reference-Preserving Decomposition", out, "generated_conceptual")


def slide10_validation_strategy():
    fig, ax = new_figure()
    ax.text(0.02, 0.95,
            "Data and Validation Strategy",
            ha="left", va="top", fontsize=16, fontweight="bold")

    xs = [0.05, 0.24, 0.43, 0.62, 0.81]
    labels = [
        "NHANES 2005-2008\nDiscovery",
        "NHANES 2009-2018\nTemporal replication",
        "NHANES 2021-2023\nModern holdout",
        "LASI-DAD\nIndian population and instrument bridge",
        "Prospective Indian validation\nif feasible"
    ]

    for x, lab in zip(xs, labels):
        add_box(ax, x, 0.48, 0.14, 0.18, lab, fontsize=10)

    for i in range(len(xs) - 1):
        add_arrow(ax, xs[i] + 0.14, 0.57, xs[i + 1], 0.57)

    ax.text(0.50, 0.22,
            "Optional multimodal extensions, including retinal information,\nremain downstream extensions rather than thesis prerequisites.",
            ha="center", va="center", fontsize=11)

    out = OUT_MAIN / "S10_data_and_validation_strategy.png"
    save_matplotlib(fig, out)
    register("S10", "Data and Validation Strategy", out, "generated_conceptual")


def slide17_methodological_program():
    fig, ax = new_figure()
    ax.text(0.02, 0.95,
            "Methodological Program",
            ha="left", va="top", fontsize=16, fontweight="bold")

    steps = [
        "Survey-weighted regressions\nand staged adjustment",
        "Sensitivity analyses\nand nonlinear checks",
        "Ordinal CFA, SEM,\nand MIMIC models",
        "Multivariate physiological\nrepresentations",
        "Reference-preserving decomposition\nand reconstruction",
        "Conditional source separation\nonly if justified",
        "Machine learning only if it\nadds value beyond simpler models"
    ]

    y = 0.64
    x = 0.04
    w = 0.12
    h = 0.16
    gap = 0.015

    for i, step in enumerate(steps):
        add_box(ax, x + i * (w + gap), y, w, h, step, fontsize=9)
        if i < len(steps) - 1:
            add_arrow(ax, x + i * (w + gap) + w, y + h / 2, x + (i + 1) * (w + gap), y + h / 2)

    ax.text(0.50, 0.25,
            "Principle:\nprogress from interpretable and auditable models to more complex methods only when necessary.",
            ha="center", va="center", fontsize=11)

    out = OUT_MAIN / "S17_methodological_program.png"
    save_matplotlib(fig, out)
    register("S17", "Methodological Program", out, "generated_conceptual")


def slide18_failure_criteria():
    fig, ax = new_figure()
    ax.text(0.02, 0.95,
            "Validation and Explicit Failure Criteria",
            ha="left", va="top", fontsize=16, fontweight="bold")

    add_box(ax, 0.38, 0.72, 0.24, 0.12, "Evaluation of the proposed framework")
    add_box(ax, 0.06, 0.46, 0.24, 0.12, "Failure to reproduce\ncore discovery structure")
    add_box(ax, 0.38, 0.46, 0.24, 0.12, "Reconstruction loss exceeds\npredefined tolerance")
    add_box(ax, 0.70, 0.46, 0.24, 0.12, "Latent or source structure\nis unstable")
    add_box(ax, 0.22, 0.22, 0.24, 0.12, "Phenotype relations are not\ninvariant or interpretable")
    add_box(ax, 0.54, 0.22, 0.24, 0.12, "Temporal or cross-population\ntransfer fails")

    for bx in [(0.18, 0.58), (0.50, 0.58), (0.82, 0.58), (0.34, 0.34), (0.66, 0.34)]:
        add_arrow(ax, 0.50, 0.72, bx[0], bx[1])

    add_box(ax, 0.24, 0.03, 0.52, 0.10,
            "Interpretation:\nfailure defines boundary conditions of the method; it is not hidden and is itself informative.",
            fontsize=10)

    out = OUT_MAIN / "S18_validation_and_failure_criteria.png"
    save_matplotlib(fig, out)
    register("S18", "Validation and Explicit Failure Criteria", out, "generated_conceptual")


def slide19_timeline():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title("Work Packages, Timeline, and Deliverables", loc="left", fontsize=16, fontweight="bold")

    phases = [
        ("WP1 Benchmark formulation", 0, 2),
        ("WP2 Decomposition and reconstruction", 1, 3),
        ("WP3 Phenotype decomposition", 3, 2),
        ("WP4 Transfer and robustness", 4, 3),
        ("WP5 Indian validation pathway", 6, 4),
        ("WP6 Optional multimodal extension", 7, 3),
        ("Integration, writing, and thesis consolidation", 9, 3),
    ]

    y_positions = np.arange(len(phases))[::-1]
    for y, (label, start, dur) in zip(y_positions, phases):
        ax.broken_barh([(start, dur)], (y - 0.35, 0.7),
                       facecolors="lightgray", edgecolors="black")
        ax.text(-0.2, y, label, ha="right", va="center", fontsize=10)

    ax.set_ylim(-1, len(phases))
    ax.set_xlim(0, 12)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.5, 12.5, 1.0))
    ax.set_xticklabels([
        "Y1-Q1", "Y1-Q2", "Y1-Q3", "Y1-Q4",
        "Y2-Q1", "Y2-Q2", "Y2-Q3", "Y2-Q4",
        "Y3-Q1", "Y3-Q2", "Y3-Q3", "Y3-Q4"
    ], rotation=0)
    ax.grid(axis="x", linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    out = OUT_MAIN / "S19_work_packages_timeline.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    register("S19", "Work Packages, Timeline, and Deliverables", out, "generated_conceptual")


def slide20_contribution_boundary():
    fig, ax = new_figure()
    ax.text(0.02, 0.95,
            "Expected Contribution and Thesis Boundary",
            ha="left", va="top", fontsize=16, fontweight="bold")

    ax.text(0.25, 0.83, "Expected contribution", ha="center", va="center", fontsize=13, fontweight="bold")
    add_box(ax, 0.07, 0.60, 0.36, 0.12,
            "A reproducible benchmark for physiological representation\nin a depression-related testbed")
    add_box(ax, 0.07, 0.42, 0.36, 0.12,
            "A reference-preserving decomposition framework with\nexplicit information-loss accounting")
    add_box(ax, 0.07, 0.24, 0.36, 0.12,
            "A transfer-oriented evaluation strategy across time,\npopulation, and instrumentation")

    ax.text(0.75, 0.83, "Explicit boundary", ha="center", va="center", fontsize=13, fontweight="bold")
    add_box(ax, 0.57, 0.60, 0.36, 0.10, "Not a claim of causality")
    add_box(ax, 0.57, 0.46, 0.36, 0.10, "Not a claim of immediate clinical utility")
    add_box(ax, 0.57, 0.32, 0.36, 0.10, "Not proof of biological independence")
    add_box(ax, 0.57, 0.18, 0.36, 0.10, "Not dependent on hospital access or multimodal private data")

    ax.plot([0.50, 0.50], [0.12, 0.88], color="black", linewidth=1.2)

    out = OUT_MAIN / "S20_expected_contribution_and_boundary.png"
    save_matplotlib(fig, out)
    register("S20", "Expected Contribution and Thesis Boundary", out, "generated_conceptual")


# -----------------------------------------------------------------------------
# EXISTING-ASSET DECK VISUALS
# -----------------------------------------------------------------------------

def slide08_existing_architecture():
    copy_or_placeholder(
        "36_01_current_vs_proposed_architecture.png",
        "S08_current_vs_proposed_architecture.png",
        "S08",
        "From the Preliminary Study to the Proposed PhD Framework",
        "Direct existing asset"
    )


def slide11_existing_cohort_flow():
    copy_or_placeholder(
        "36_02_discovery_cohort_flow.png",
        "S11_discovery_cohort_flow.png",
        "S11",
        "Preliminary Discovery Cohort",
        "Direct existing asset"
    )


def slide12_operational_representation():
    out = OUT_MAIN / "S12_first_operational_physiological_representation.png"
    make_panel(
        image_names=[
            "34_08_hb_to_A_transform.png",
            "34_09_hba1c_to_G_transform.png",
        ],
        panel_titles=[
            "Hemoglobin deficit representation: A",
            "HbA1c excess representation: G",
        ],
        out_path=out,
        figure_title="First Operational Physiological Representation"
    )
    register("S12", "First Operational Physiological Representation", out, "combined_existing_assets")


def slide13_feasibility():
    out = OUT_MAIN / "S13_feasibility_nonidentical_information.png"
    make_panel(
        image_names=[
            "36_04_discovery_primary_effects.png",
            "36_05_separability_incremental_information.png",
        ],
        panel_titles=[
            "Primary adjusted discovery result",
            "Incremental information and separability",
        ],
        out_path=out,
        figure_title="Feasibility: The Two Domains Contain Non-Identical Information"
    )
    register("S13", "Feasibility: The Two Domains Contain Non-Identical Information", out, "combined_existing_assets")


def slide14_phenotype_structure():
    out = OUT_MAIN / "S14_phenotype_structure.png"
    make_panel(
        image_names=[
            "36_06_cfa_fit_summary.png",
            "36_08_mimic_somatic_paths.png",
        ],
        panel_titles=[
            "Ordinal CFA fit summary",
            "MIMIC paths to the latent somatic phenotype",
        ],
        out_path=out,
        figure_title="The Phenotype Also Contains Reproducible Structure"
    )
    register("S14", "The Phenotype Also Contains Reproducible Structure", out, "combined_existing_assets")


def slide15_scalar_not_enough():
    copy_or_placeholder(
        "36_09_reduced_hematology_pca.png",
        "S15_scalar_representation_is_not_enough.png",
        "S15",
        "Why the Scalar Hemoglobin Representation Is Not Enough",
        "Direct existing asset"
    )


def slide16_transfer():
    out = OUT_MAIN / "S16_transfer_is_a_scientific_test.png"
    make_panel(
        image_names=[
            "36_10_frozen_temporal_transfer_summary.png",
            "36_11_2123_A_diagnostic_ladder.png",
        ],
        panel_titles=[
            "Frozen temporal transfer summary",
            "2021-2023 A diagnostic ladder",
        ],
        out_path=out,
        figure_title="Transfer Is a Scientific Test, Not an Assumption"
    )
    register("S16", "Transfer Is a Scientific Test, Not an Assumption", out, "combined_existing_assets")


# -----------------------------------------------------------------------------
# OPTIONAL APPENDIX
# -----------------------------------------------------------------------------

def appendix_reference_preservation():
    src = SRC_FIG / "34_30_reference_preserving_branch_architecture.png"
    dst = OUT_APPENDIX / "A7_reference_preservation_audit_architecture.png"
    if src.exists():
        shutil.copy2(src, dst)
        register("A07", "Reference-Preservation Audit Architecture", dst, "existing_direct_copy")
    else:
        placeholder_image("Missing source asset:\n34_30_reference_preserving_branch_architecture.png").save(dst)
        register("A07", "Reference-Preservation Audit Architecture", dst, "placeholder_missing_source")


def appendix_modern_G_ladder():
    src = SRC_FIG / "36_12_2123_G_diagnostic_ladder.png"
    dst = OUT_APPENDIX / "A6_modern_G_diagnostic_ladder.png"
    if src.exists():
        shutil.copy2(src, dst)
        register("A06", "2021-2023 G Diagnostic Ladder", dst, "existing_direct_copy")
    else:
        placeholder_image("Missing source asset:\n36_12_2123_G_diagnostic_ladder.png").save(dst)
        register("A06", "2021-2023 G Diagnostic Ladder", dst, "placeholder_missing_source")


# -----------------------------------------------------------------------------
# BUILD
# -----------------------------------------------------------------------------

def main():
    print("37 — BUILD PRESENTATION VISUALS")
    print("================================")
    print("Non-destructive deck-visual build.\n")

    # Generated conceptual figures
    slide02_motivation()
    slide03_domain_coupling()
    slide04_phenotype_ambiguity()
    slide06_why_open()
    slide07_rqs()
    slide09_reference_preserving()
    slide10_validation_strategy()
    slide17_methodological_program()
    slide18_failure_criteria()
    slide19_timeline()
    slide20_contribution_boundary()

    # Existing or combined assets
    slide08_existing_architecture()
    slide11_existing_cohort_flow()
    slide12_operational_representation()
    slide13_feasibility()
    slide14_phenotype_structure()
    slide15_scalar_not_enough()
    slide16_transfer()

    # Optional appendix
    appendix_reference_preservation()
    appendix_modern_G_ladder()

    write_manifest()

    print(f"SAVED main visuals to: {OUT_MAIN}")
    print(f"SAVED appendix visuals to: {OUT_APPENDIX}")
    print("\nDone.")


if __name__ == "__main__":
    main()