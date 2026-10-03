#!/usr/bin/env python
# =============================================================================
# 89_local_math_ui.py
#
# Local Tkinter + Matplotlib UI for the physiology-decomposition project.
# No HTML/browser required.
#
# Views:
#   1) Physiology: H and G clouds separately
#   2) X -> Z: animated coordinate transformation
#   3) Z -> Y: latent components feeding phenotype outputs
#   4) Derivatives: local fitted sensitivities
#   5) Reverse: one phenotype band -> a region in latent physiology
#
# Output: none required; this launches a local interactive window.
# =============================================================================

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# -----------------------------------------------------------------------------
# Paths / data
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "EDA" / "Results"

INPUT_CANDIDATES = [
    RESULTS / "73_frozen_phenotype_map_input.csv",
    RESULTS / "72_shared_private_model_input.csv",
    ROOT / "73_frozen_phenotype_map_input.csv",
    ROOT / "72_shared_private_model_input.csv",
]
INPUT = next((p for p in INPUT_CANDIDATES if p.exists()), None)
if INPUT is None:
    raise FileNotFoundError("Could not find Script-72/73 model input CSV.")

RAW_ALIAS = {
    "Hb": ["LBXHGB", "Hb", "HGB"],
    "RBC": ["LBXRBCSI", "RBC"],
    "MCV": ["LBXMCVSI", "MCV"],
    "RDW": ["LBXRDW", "RDW"],
    "HbA1c": ["LBXGH", "HbA1c", "GHB", "LBDGH"],
    "Glucose": ["LBXGLU", "Glucose", "GLU"],
    "logInsulin": ["LOG_IN", "logInsulin", "LogInsulin", "LOG_INSULIN", "log_insulin"],
}

Y_ALIAS = {
    "PHQ": ["PHQ9_TOTAL"],
    "SOMATIC": ["SOMATIC_SCORE", "SOMATIC"],
    "COGAFF": ["COGAFF_SUM", "COGAFF"],
}

Z_COLS = [
    "SP_SHARED1", "SP_SHARED2", "SP_DISCORD1", "SP_DISCORD2",
    "SP_A_PRIVATE3", "SP_A_PRIVATE4", "SP_G_PRIVATE3"
]

Z_LABELS = [
    "Shared 1", "Shared 2", "Mismatch 1", "Mismatch 2",
    "H-private 3", "H-private 4", "G-private 3"
]

# Actual coefficients from the current phenotype-map audit.
BETA = {
    "Somatic": np.array([0.121246, -0.071556, -0.062547, -0.133865, 0.194142, 0.064871, 0.067662]),
    "Cognitive-affective": np.array([0.117490, -0.026556, -0.053620, -0.080085, 0.184665, 0.093569, 0.064631]),
    "PHQ-9 total": np.array([0.244403, -0.098316, -0.118846, -0.219315, 0.379132, 0.156600, 0.129840]),
}

RAW_DERIV_PHQ = {
    "Hb": 0.519628,
    "RBC": -0.768724,
    "MCV": -0.259267,
    "RDW": 0.392235,
    "HbA1c": 0.153647,
    "Glucose": 0.057087,
    "logInsulin": 0.198612,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def resolve_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def standardize(X):
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def pca_fit(X, ndim=3):
    Z, _, _ = standardize(X)
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    scores = Z @ vt.T[:, :ndim]
    var = (s**2) / np.sum(s**2)
    return scores, vt[:ndim, :].T, var[:ndim]


def top_axis_name(loadings, names, axis_i, prefix):
    col = loadings[:, axis_i]
    idx = np.argsort(np.abs(col))[::-1][:2]
    return f"{prefix} ({names[idx[0]]}/{names[idx[1]]})"


def draw_ellipsoid(ax, pts, color, alpha=0.12, radius=1.55):
    pts = np.asarray(pts, float)
    center = pts.mean(axis=0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-6, None)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]

    u = np.linspace(0, 2*np.pi, 34)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)

    A = vecs @ np.diag(radius * np.sqrt(vals))
    ell = (A @ sphere).T + center
    X = ell[:,0].reshape(len(u), len(v))
    Y = ell[:,1].reshape(len(u), len(v))
    Z = ell[:,2].reshape(len(u), len(v))

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=False)


def dark_3d(ax):
    ax.set_facecolor("#070B12")
    try:
        ax.xaxis.set_pane_color((0.04,0.07,0.11,1))
        ax.yaxis.set_pane_color((0.04,0.07,0.11,1))
        ax.zaxis.set_pane_color((0.04,0.07,0.11,1))
    except Exception:
        pass
    ax.tick_params(colors="#8194A8", labelsize=8)
    ax.xaxis.label.set_color("#EAF2FF")
    ax.yaxis.label.set_color("#EAF2FF")
    ax.zaxis.label.set_color("#EAF2FF")
    ax.grid(False)


def procrustes_align(A, B):
    """Return B rotated/scaled to A for a smooth visual morph."""
    A0 = A - A.mean(axis=0)
    B0 = B - B.mean(axis=0)
    m = B0.T @ A0
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    Br = B0 @ r
    scale = np.sqrt(np.mean(np.sum(A0*A0, axis=1))) / max(
        np.sqrt(np.mean(np.sum(Br*Br, axis=1))), 1e-12
    )
    return A0, Br * scale


# -----------------------------------------------------------------------------
# Load / prepare
# -----------------------------------------------------------------------------

d = pd.read_csv(INPUT)

raw_cols = {k: resolve_col(d, v) for k, v in RAW_ALIAS.items()}
y_cols = {k: resolve_col(d, v) for k, v in Y_ALIAS.items()}

missing_raw = [k for k,v in raw_cols.items() if v is None]
missing_y = [k for k,v in y_cols.items() if v is None]
missing_z = [z for z in Z_COLS if z not in d.columns]

if missing_raw:
    raise ValueError(f"Missing raw columns: {missing_raw}")
if missing_y:
    raise ValueError(f"Missing phenotype columns: {missing_y}")
if missing_z:
    raise ValueError(f"Missing latent columns: {missing_z}")

cols = list(raw_cols.values()) + list(y_cols.values()) + Z_COLS
for c in cols:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d[cols].dropna().copy()

rng = np.random.default_rng(12)
if len(d) > 1800:
    d = d.iloc[rng.choice(len(d), 1800, replace=False)].reset_index(drop=True)

H_NAMES = np.array(["Hb","RBC","MCV","RDW"])
G_NAMES = np.array(["HbA1c","Glucose","logInsulin"])
X_NAMES = np.array(["Hb","RBC","MCV","RDW","HbA1c","Glucose","logInsulin"])

H = d[[raw_cols[x] for x in H_NAMES]].to_numpy(float)
G = d[[raw_cols[x] for x in G_NAMES]].to_numpy(float)
Xraw = d[[raw_cols[x] for x in X_NAMES]].to_numpy(float)

Hscore, Hload, Hvar = pca_fit(H, 3)
Gscore, Gload, Gvar = pca_fit(G, 3)
Xscore, Xload, Xvar = pca_fit(Xraw, 3)

Zstd, _, _ = standardize(d[Z_COLS].to_numpy(float))
# Display 3 interpretable coordinates from the 7-D Z.
Z3 = Zstd[:, [0, 2, 4]]

Xvis, Zvis = procrustes_align(Xscore, Z3)

phq = d[y_cols["PHQ"]].to_numpy(float)
mask_mod = (phq >= 10) & (phq <= 14)
if mask_mod.sum() < 30:
    mask_mod = (phq >= 8) & (phq <= 14)
Z_mod = Z3[mask_mod]


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

BG = "#050912"
PANEL = "#091321"
TEXT = "#EDF4FF"
MUTED = "#93A6BA"
CYAN = "#31D7FF"
AMBER = "#FFBF00"
GREEN = "#00E69A"
BLUE = "#49C6FF"
PINK = "#FF6685"
WHITE = "#FFFFFF"


class StoryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Physiological Decomposition — Math Story")
        self.geometry("1420x900")
        self.configure(bg=BG)

        self.current_animation = None
        self.morph_t = tk.DoubleVar(value=0.0)

        self._build_header()
        self._build_controls()
        self._build_canvas()
        self.show_physiology()

    def _build_header(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=18, pady=(12,5))

        tk.Label(
            top,
            text="From measured physiology to latent structure to phenotype",
            bg=BG, fg=TEXT,
            font=("Arial", 18, "bold")
        ).pack(anchor="w")

        tk.Label(
            top,
            text="Local conference explainer — clouds, equations, and only the main modelling steps",
            bg=BG, fg=MUTED,
            font=("Arial", 10)
        ).pack(anchor="w", pady=(2,0))

    def _build_controls(self):
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=18, pady=(2,8))

        self.buttons = {}
        specs = [
            ("Physiology", self.show_physiology),
            ("X → Z", self.show_transform),
            ("Z → Y", self.show_zy),
            ("Derivatives", self.show_derivatives),
            ("Reverse", self.show_reverse),
        ]
        for label, cmd in specs:
            b = tk.Button(
                bar, text=label, command=cmd,
                bg="#0A1624", fg=TEXT, activebackground="#12314A",
                activeforeground=WHITE, bd=0,
                padx=14, pady=7,
                font=("Arial", 10, "bold"),
                cursor="hand2"
            )
            b.pack(side="left", padx=(0,7))
            self.buttons[label] = b

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)

        self.play_btn = tk.Button(
            bar, text="▶ Play transform",
            command=self.play_transform,
            bg="#12314A", fg=TEXT, bd=0,
            padx=12, pady=7,
            font=("Arial", 10),
            state="disabled"
        )
        self.play_btn.pack(side="left", padx=(0,7))

        self.slider = tk.Scale(
            bar,
            from_=0, to=100, orient="horizontal",
            variable=self.morph_t,
            command=self._slider_changed,
            showvalue=True,
            length=280,
            bg=BG, fg=TEXT,
            troughcolor="#1C2C3F",
            highlightthickness=0,
            state="disabled"
        )
        self.slider.pack(side="left")

    def _build_canvas(self):
        self.fig = plt.Figure(figsize=(12.5, 7.1), facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=(0,12))

    def _activate(self, name, morph=False):
        for k,b in self.buttons.items():
            b.configure(bg="#12314A" if k == name else "#0A1624")
        self.play_btn.configure(state="normal" if morph else "disabled")
        self.slider.configure(state="normal" if morph else "disabled")
        if self.current_animation is not None:
            try:
                self.current_animation.event_source.stop()
            except Exception:
                pass
            self.current_animation = None

    def _title(self, text, sub=None):
        self.fig.suptitle(text, color=TEXT, fontsize=18, fontweight="bold", x=.03, ha="left", y=.98)
        if sub:
            self.fig.text(.03, .935, sub, color=MUTED, fontsize=10, ha="left")

    # -------------------------------------------------------------------------
    # View 1: Physiology
    # -------------------------------------------------------------------------

    def show_physiology(self):
        self._activate("Physiology", morph=False)
        self.fig.clear()

        ax1 = self.fig.add_subplot(121, projection="3d")
        ax2 = self.fig.add_subplot(122, projection="3d")
        dark_3d(ax1); dark_3d(ax2)

        self._title(
            "1. Start with the measured physiology",
            "H and G are shown separately before they are combined into the full 7-D state X = [H, G]."
        )

        # H cloud + points
        draw_ellipsoid(ax1, Hscore, CYAN, alpha=.14)
        ids = rng.choice(len(Hscore), min(150, len(Hscore)), replace=False)
        ax1.scatter(Hscore[ids,0], Hscore[ids,1], Hscore[ids,2],
                    s=10, c=CYAN, alpha=.28, edgecolors="none")

        h_axes = [
            top_axis_name(Hload, H_NAMES, 0, "axis 1"),
            top_axis_name(Hload, H_NAMES, 1, "axis 2"),
            top_axis_name(Hload, H_NAMES, 2, "axis 3"),
        ]
        ax1.set_xlabel(h_axes[0], labelpad=7)
        ax1.set_ylabel(h_axes[1], labelpad=7)
        ax1.set_zlabel(h_axes[2], labelpad=7)
        ax1.set_title(
            f"H: haematological state\nHb · RBC · MCV · RDW\n3-PC view = {100*Hvar.sum():.1f}% variance",
            color=TEXT, fontsize=12, pad=12
        )

        # G cloud + points
        draw_ellipsoid(ax2, Gscore, AMBER, alpha=.14)
        ids = rng.choice(len(Gscore), min(150, len(Gscore)), replace=False)
        ax2.scatter(Gscore[ids,0], Gscore[ids,1], Gscore[ids,2],
                    s=10, c=AMBER, alpha=.28, edgecolors="none")

        g_axes = [
            top_axis_name(Gload, G_NAMES, 0, "axis 1"),
            top_axis_name(Gload, G_NAMES, 1, "axis 2"),
            top_axis_name(Gload, G_NAMES, 2, "axis 3"),
        ]
        ax2.set_xlabel(g_axes[0], labelpad=7)
        ax2.set_ylabel(g_axes[1], labelpad=7)
        ax2.set_zlabel(g_axes[2], labelpad=7)
        ax2.set_title(
            f"G: glycaemic state\nHbA1c · glucose · log insulin\n3-PC view = {100*Gvar.sum():.1f}% variance",
            color=TEXT, fontsize=12, pad=12
        )

        self.fig.text(
            .5, .035,
            "cloud = population structure     •     soft dots = example subjects     •     axis names follow dominant loadings",
            color=MUTED, fontsize=10, ha="center"
        )
        self.canvas.draw_idle()

    # -------------------------------------------------------------------------
    # View 2: X -> Z
    # -------------------------------------------------------------------------

    def show_transform(self):
        self._activate("X → Z", morph=True)
        self.morph_t.set(0)
        self._draw_transform(0.0)

    def _slider_changed(self, value):
        if self.buttons["X → Z"].cget("bg") == "#12314A":
            self._draw_transform(float(value)/100.0)

    def _draw_transform(self, t):
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection="3d")
        dark_3d(ax)

        P = (1-t)*Xvis + t*Zvis

        # smooth color transition
        c0 = np.array([180,140,255], float)
        c1 = np.array([0,230,154], float)
        cc = ((1-t)*c0 + t*c1).astype(int)
        color = "#{:02x}{:02x}{:02x}".format(*cc)

        draw_ellipsoid(ax, P, color, alpha=.13)
        ids = rng.choice(len(P), min(180, len(P)), replace=False)
        ax.scatter(P[ids,0], P[ids,1], P[ids,2],
                   s=10, c=color, alpha=.30, edgecolors="none")

        # Representative trajectories
        track = ids[:8]
        for i in track:
            start = Xvis[i]
            end = Zvis[i]
            cur = (1-t)*start + t*end
            ax.plot([start[0], cur[0]], [start[1], cur[1]], [start[2], cur[2]],
                    color=WHITE, alpha=.20, lw=1.0)

        if t < .45:
            axes = [
                top_axis_name(Xload, X_NAMES, 0, "raw physiology axis 1"),
                top_axis_name(Xload, X_NAMES, 1, "raw physiology axis 2"),
                top_axis_name(Xload, X_NAMES, 2, "raw physiology axis 3"),
            ]
            state = "X: raw physiology view"
        elif t < .75:
            axes = ["transition coordinate 1", "transition coordinate 2", "transition coordinate 3"]
            state = "changing coordinates"
        else:
            axes = [
                "shared H–G physiology",
                "H–G mismatch",
                "H-specific physiology",
            ]
            state = "Z: learned physiology view"

        ax.set_xlabel(axes[0], labelpad=8)
        ax.set_ylabel(axes[1], labelpad=8)
        ax.set_zlabel(axes[2], labelpad=8)

        self._title(
            "2. X → Z is a coordinate change",
            f"Same subjects, same underlying physiology — the representation becomes easier to interpret.   progress: {100*t:.0f}%"
        )
        ax.set_title(state, color=TEXT, fontsize=12, pad=12)

        self.fig.text(
            .5, .055,
            r"$X=[Hb,RBC,MCV,RDW,HbA1c,Glucose,\log Insulin]$   →   "
            r"$Z=[shared,\ mismatch,\ H\!-\!private,\ G\!-\!private]$",
            color=TEXT, fontsize=12, ha="center"
        )
        self.fig.text(
            .5, .025,
            "The animation is a visual interpolation between the real endpoint coordinate systems.",
            color=MUTED, fontsize=9, ha="center"
        )
        self.canvas.draw_idle()

    def play_transform(self):
        if self.buttons["X → Z"].cget("bg") != "#12314A":
            return

        frames = np.linspace(0, 1, 31)

        def update(i):
            t = frames[i]
            self.morph_t.set(t*100)
            self._draw_transform(t)

        self.current_animation = FuncAnimation(
            self.fig, update, frames=len(frames),
            interval=90, repeat=False
        )
        self.canvas.draw_idle()

    # -------------------------------------------------------------------------
    # View 3: Z -> Y
    # -------------------------------------------------------------------------

    def show_zy(self):
        self._activate("Z → Y", morph=False)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        self._title(
            "3. Z → Y: latent physiology feeds three phenotype outputs",
            "Thickness shows the magnitude of the fitted component coefficient; blue is positive and pink is negative."
        )

        ys = np.linspace(.84, .18, 7)
        outputs = {"Somatic": .75, "Cognitive-affective": .50, "PHQ-9 total": .25}

        for i, (lab, yy) in enumerate(zip(Z_LABELS, ys)):
            ax.scatter(.17, yy, s=280, c=GREEN, alpha=.88, edgecolors=WHITE, linewidths=.8)
            ax.text(.21, yy, lab, va="center", ha="left", color=TEXT, fontsize=10)

        for out, yy in outputs.items():
            ax.scatter(.84, yy, s=650, c=AMBER, alpha=.88, edgecolors=WHITE, linewidths=1.0)
            ax.text(.79, yy, out, va="center", ha="right", color=TEXT, fontsize=11, fontweight="bold")

        for out, vals in BETA.items():
            yy2 = outputs[out]
            maxabs = max(abs(vals))
            for i,v in enumerate(vals):
                lw = .8 + 4.8*abs(v)/maxabs
                color = BLUE if v >= 0 else PINK
                ax.plot([.31,.70], [ys[i], yy2], color=color, lw=lw, alpha=.48)

        ax.text(.5,.94, r"$Y=h(Z)$", color=WHITE, fontsize=22, ha="center")
        ax.text(.5,.06,
                "left = learned physiological components     →     right = symptom phenotypes",
                color=MUTED, fontsize=10, ha="center")
        self.canvas.draw_idle()

    # -------------------------------------------------------------------------
    # View 4: derivatives
    # -------------------------------------------------------------------------

    def show_derivatives(self):
        self._activate("Derivatives", morph=False)
        self.fig.clear()

        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        for ax in (ax1, ax2):
            ax.set_facecolor(BG)
            ax.tick_params(colors=MUTED)
            for s in ax.spines.values():
                s.set_color("#33465A")
            ax.grid(axis="x", color="#1D2C3C", alpha=.55)

        self._title(
            "4. Derivatives: local sensitivity of the fitted phenotype map",
            r"First in latent coordinates, then chained back to the measured variables:   dY/dX = (dY/dZ)(dZ/dX)"
        )

        vals_z = BETA["PHQ-9 total"]
        cols_z = [BLUE if v >= 0 else PINK for v in vals_z]
        ax1.barh(Z_LABELS, vals_z, color=cols_z, alpha=.82)
        ax1.axvline(0, color=WHITE, lw=.8)
        ax1.set_title(r"$\partial Y/\partial z_i$", color=TEXT, fontsize=13)
        ax1.set_xlabel("local sensitivity", color=TEXT)
        ax1.invert_yaxis()

        names = list(RAW_DERIV_PHQ)
        vals_x = np.array([RAW_DERIV_PHQ[n] for n in names])
        cols_x = [BLUE if v >= 0 else PINK for v in vals_x]
        ax2.barh(names, vals_x, color=cols_x, alpha=.82)
        ax2.axvline(0, color=WHITE, lw=.8)
        ax2.set_title(r"$\partial Y/\partial X$", color=TEXT, fontsize=13)
        ax2.set_xlabel("standardized local sensitivity", color=TEXT)
        ax2.invert_yaxis()

        self.fig.text(
            .5, .045,
            "Read it as: if this coordinate changes slightly while the others are held fixed, how does the fitted PHQ output change?",
            color=MUTED, fontsize=10, ha="center"
        )
        self.canvas.draw_idle()

    # -------------------------------------------------------------------------
    # View 5: reverse
    # -------------------------------------------------------------------------

    def show_reverse(self):
        self._activate("Reverse", morph=False)
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection="3d")
        dark_3d(ax)

        self._title(
            "5. Reverse inference: one symptom band corresponds to a region in physiology",
            "Example shown: participants with PHQ-9 10–14 projected into three latent physiological coordinates."
        )

        draw_ellipsoid(ax, Z_mod, PINK, alpha=.15, radius=1.70)
        ids = rng.choice(len(Z_mod), min(140, len(Z_mod)), replace=False)
        ax.scatter(
            Z_mod[ids,0], Z_mod[ids,1], Z_mod[ids,2],
            s=10, c=WHITE, alpha=.30, edgecolors="none"
        )

        centroid = Z_mod.mean(axis=0)
        ax.scatter([centroid[0]],[centroid[1]],[centroid[2]],
                   s=80,c=AMBER,edgecolors=WHITE,linewidths=.8)
        ax.text(centroid[0],centroid[1],centroid[2],"  conditional center",color=TEXT,fontsize=9)

        ax.set_xlabel("shared H–G physiology", labelpad=8)
        ax.set_ylabel("H–G mismatch", labelpad=8)
        ax.set_zlabel("H-specific physiology", labelpad=8)
        ax.set_title(r"$p(Z\mid Y)$", color=TEXT, fontsize=15, pad=12)

        self.fig.text(
            .5, .045,
            "Same symptom burden ≠ one unique physiological state. The reverse object is a plausible region.",
            color=MUTED, fontsize=10, ha="center"
        )
        self.canvas.draw_idle()


if __name__ == "__main__":
    app = StoryApp()
    app.mainloop()
