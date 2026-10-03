#!/usr/bin/env python
# =============================================================================
# 89_conference_math_story.py
#
# Single-file conference explainer for:
#   H block -> G block -> X -> PCA(X) -> Z -> Y
#   derivatives -> time drift -> probabilistic reverse -> longitudinal refinement
#
# Output:
#   EDA/Results/89_conference_math_story/89_conference_math_story.html
#
# Design:
#   - one HTML
#   - discrete chapter buttons
#   - animation only where the intermediate states have a meaning
#   - soft clouds + sparse points
#   - layman labels + equations
#   - no "script summary" ending
# =============================================================================

from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "EDA" / "Results"
OUT = RESULTS / "89_conference_math_story"
OUT.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    RESULTS / "73_frozen_phenotype_map_input.csv",
    RESULTS / "72_shared_private_model_input.csv",
    ROOT / "73_frozen_phenotype_map_input.csv",
    ROOT / "72_shared_private_model_input.csv",
]
INPUT = next((p for p in INPUT_CANDIDATES if p.exists()), None)
if INPUT is None:
    raise FileNotFoundError("Could not find Script-72/73 model input CSV.")

# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------

C = {
    "bg": "#050912",
    "panel": "#091321",
    "grid": "#243348",
    "text": "#EDF4FF",
    "muted": "#96A8BC",
    "h": "#31D7FF",
    "g": "#FFBF00",
    "z": "#00E69A",
    "pos": "#49C6FF",
    "neg": "#FF6685",
    "white": "#FFFFFF",
    "grey": "#AAB6C2",
    "purple": "#B48CFF",
}

PERIODS = ["2005-2008", "2009-2018", "2021-2023"]
PERIOD_COLOR = {
    "2005-2008": "#31D7FF",
    "2009-2018": "#FFBF00",
    "2021-2023": "#00E69A",
}

# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

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

def resolve_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None

def derive_period(df):
    if "PERIOD" in df.columns:
        return df["PERIOD"].astype(str).str.replace(".0", "", regex=False)
    if "CYCLE" in df.columns:
        cyc = df["CYCLE"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
        mp = {
            "0506": "2005-2008", "0708": "2005-2008",
            "0910": "2009-2018", "1112": "2009-2018", "1314": "2009-2018",
            "1516": "2009-2018", "1718": "2009-2018",
            "2123": "2021-2023",
        }
        return cyc.map(mp)
    raise ValueError("Need PERIOD or CYCLE column.")

def standardize(X):
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd

def pca_fit(X, ndim=3):
    Z, mu, sd = standardize(X)
    U, S, VT = np.linalg.svd(Z, full_matrices=False)
    scores = Z @ VT.T[:, :ndim]
    var = (S**2) / np.sum(S**2)
    return scores, VT[:ndim, :].T, var[:ndim], mu, sd

def top_loading_label(loadings_col, names, prefix):
    idx = np.argsort(np.abs(loadings_col))[::-1][:2]
    return f"{prefix} ({names[idx[0]]} / {names[idx[1]]})"

def covariance_surface(points, radius=1.55, nu=34, nv=22):
    pts = np.asarray(points, float)
    center = pts.mean(axis=0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-7, None)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]

    u = np.linspace(0, 2*np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)

    A = vecs @ np.diag(radius * np.sqrt(vals))
    ell = (A @ sphere).T + center
    X = ell[:, 0].reshape(nu, nv)
    Y = ell[:, 1].reshape(nu, nv)
    Z = ell[:, 2].reshape(nu, nv)
    return X, Y, Z

def cloud_figure(points, color, title, subtitle, axis_titles,
                 labels=None, label_points=None, note=None, n_dots=130):
    points = np.asarray(points, float)
    rng = np.random.default_rng(42)
    if len(points) > n_dots:
        idx = rng.choice(len(points), size=n_dots, replace=False)
        pts = points[idx]
    else:
        pts = points

    Xs, Ys, Zs = covariance_surface(points, radius=1.50)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=Xs, y=Ys, z=Zs,
        surfacecolor=np.zeros_like(Xs),
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=0.16,
        hoverinfo="skip",
        name="population cloud",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode="markers",
        marker=dict(size=3.4, color=color, opacity=0.42),
        name="sample subjects",
        hovertemplate="example subject<extra></extra>"
    ))

    if labels is not None and label_points is not None:
        for lab, endpoint in zip(labels, label_points):
            endpoint = np.asarray(endpoint, float)
            fig.add_trace(go.Scatter3d(
                x=[0, endpoint[0]], y=[0, endpoint[1]], z=[0, endpoint[2]],
                mode="lines+text",
                line=dict(color=C["white"], width=3),
                text=["", lab],
                textposition="top center",
                textfont=dict(size=11, color=C["white"]),
                hoverinfo="skip",
                showlegend=False,
            ))

    fig.update_layout(
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        font=dict(color=C["text"], family="Arial, Helvetica, sans-serif"),
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:13px;color:{C['muted']}'>{subtitle}</span>",
            x=0.01, xanchor="left"
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"])),
        scene=dict(
            bgcolor=C["bg"],
            xaxis=dict(
                title=dict(text=axis_titles[0], font=dict(color=C["text"])),
                backgroundcolor=C["panel"], gridcolor=C["grid"],
                showticklabels=False, zeroline=False
            ),
            yaxis=dict(
                title=dict(text=axis_titles[1], font=dict(color=C["text"])),
                backgroundcolor=C["panel"], gridcolor=C["grid"],
                showticklabels=False, zeroline=False
            ),
            zaxis=dict(
                title=dict(text=axis_titles[2], font=dict(color=C["text"])),
                backgroundcolor=C["panel"], gridcolor=C["grid"],
                showticklabels=False, zeroline=False
            ),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.0)),
        )
    )
    if note:
        fig.add_annotation(
            x=0.01, y=0.02, xref="paper", yref="paper",
            text=note, showarrow=False, align="left",
            font=dict(size=12, color=C["muted"])
        )
    return fig

def to_fragment(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False)

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------

d = pd.read_csv(INPUT)
d["PERIOD_VIS"] = derive_period(d)

raw_cols = {k: resolve_col(d, v) for k, v in RAW_ALIAS.items()}
y_cols = {k: resolve_col(d, v) for k, v in Y_ALIAS.items()}

missing_raw = [k for k,v in raw_cols.items() if v is None]
missing_y = [k for k,v in y_cols.items() if v is None]
missing_z = [c for c in Z_COLS if c not in d.columns]
if missing_raw:
    raise ValueError(f"Missing raw physiology columns: {missing_raw}")
if missing_y:
    raise ValueError(f"Missing phenotype columns: {missing_y}")
if missing_z:
    raise ValueError(f"Missing latent columns: {missing_z}")

numeric_cols = list(raw_cols.values()) + list(y_cols.values()) + Z_COLS
for c in numeric_cols:
    d[c] = pd.to_numeric(d[c], errors="coerce")

d = d[d["PERIOD_VIS"].isin(PERIODS)].copy()
d = d[["PERIOD_VIS"] + numeric_cols].dropna().copy()

# deterministic sample for interactive visuals
rng = np.random.default_rng(7)
if len(d) > 1800:
    idx = rng.choice(len(d), 1800, replace=False)
    d = d.iloc[idx].reset_index(drop=True)

H_NAMES = np.array(["Hb", "RBC", "MCV", "RDW"])
G_NAMES = np.array(["HbA1c", "Glucose", "logInsulin"])
X_NAMES = np.array(["Hb", "RBC", "MCV", "RDW", "HbA1c", "Glucose", "logInsulin"])

H = d[[raw_cols[x] for x in H_NAMES]].to_numpy(float)
G = d[[raw_cols[x] for x in G_NAMES]].to_numpy(float)
Xraw = d[[raw_cols[x] for x in X_NAMES]].to_numpy(float)

Hscore, Hload, Hvar, *_ = pca_fit(H, 3)
Gscore, Gload, Gvar, *_ = pca_fit(G, 3)
Xscore, Xload, Xvar, *_ = pca_fit(Xraw, 3)

Zfull, _, _ = standardize(d[Z_COLS].to_numpy(float))
# 3 displayed Z coordinates: Shared1, Discord1, H-private3
Z3 = Zfull[:, [0, 2, 4]]

# Align Z3 to Xscore for a smoother visual coordinate-change morph.
Xc = Xscore - Xscore.mean(axis=0)
Zc = Z3 - Z3.mean(axis=0)
M = Zc.T @ Xc
U, _, VT = np.linalg.svd(M)
R = U @ VT
Zalign = Zc @ R
# match overall radius
scale = np.sqrt(np.mean(np.sum(Xc**2, axis=1))) / max(np.sqrt(np.mean(np.sum(Zalign**2, axis=1))), 1e-9)
Zalign *= scale

# -----------------------------------------------------------------------------
# 1. H block
# -----------------------------------------------------------------------------

h_axis = [
    top_loading_label(Hload[:,0], H_NAMES, "Blood axis 1"),
    top_loading_label(Hload[:,1], H_NAMES, "Blood axis 2"),
    top_loading_label(Hload[:,2], H_NAMES, "Blood axis 3"),
]
h_vecs = Hload * 2.3
fig_h = cloud_figure(
    Hscore, C["h"],
    "H block — haematological state",
    f"H = [Hb, RBC, MCV, RDW] · first 3 PCs show {100*Hvar.sum():.1f}% of H-block variance",
    h_axis,
    labels=H_NAMES.tolist(),
    label_points=h_vecs,
    note="Cloud = population structure · soft dots = example subjects · white rays = raw-variable loadings"
)

# -----------------------------------------------------------------------------
# 2. G block
# -----------------------------------------------------------------------------

g_axis = [
    top_loading_label(Gload[:,0], G_NAMES, "Glycaemic axis 1"),
    top_loading_label(Gload[:,1], G_NAMES, "Glycaemic axis 2"),
    top_loading_label(Gload[:,2], G_NAMES, "Glycaemic axis 3"),
]
g_vecs = Gload * 2.3
fig_g = cloud_figure(
    Gscore, C["g"],
    "G block — glycaemic state",
    f"G = [HbA1c, Glucose, logInsulin] · first 3 PCs show {100*Gvar.sum():.1f}% of G-block variance",
    g_axis,
    labels=G_NAMES.tolist(),
    label_points=g_vecs,
    note="Axis names are generated from the dominant loadings so the geometry stays tied to the actual data."
)

# -----------------------------------------------------------------------------
# 3. X = [H,G] concept
# -----------------------------------------------------------------------------

fig_x = go.Figure()

# Two conceptual clouds: use Hscore and Gscore shifted in one scene
Hshift = Hscore.copy()
Gshift = Gscore.copy()
Hshift[:,0] -= 2.4
Gshift[:,0] += 2.4

for pts, color, name in [(Hshift, C["h"], "H: haematological block"), (Gshift, C["g"], "G: glycaemic block")]:
    Xs,Ys,Zs = covariance_surface(pts, radius=1.35)
    fig_x.add_trace(go.Surface(
        x=Xs,y=Ys,z=Zs,
        surfacecolor=np.zeros_like(Xs),
        colorscale=[[0,color],[1,color]],
        opacity=0.15, showscale=False, hoverinfo="skip", name=name, showlegend=False
    ))
    ids = rng.choice(len(pts), min(80,len(pts)), replace=False)
    fig_x.add_trace(go.Scatter3d(
        x=pts[ids,0],y=pts[ids,1],z=pts[ids,2],
        mode="markers", marker=dict(size=3.2,color=color,opacity=.38),
        name=name
    ))

fig_x.add_annotation(
    x=.5,y=.91,xref="paper",yref="paper",
    text="<b>X = [H, G]</b>",showarrow=False,
    font=dict(size=24,color=C["white"])
)
fig_x.add_annotation(
    x=.5,y=.06,xref="paper",yref="paper",
    text="X is not yet a 3-D object — it is the full 7-variable physiological state. The two clouds simply show its two measured domains.",
    showarrow=False, font=dict(size=12,color=C["muted"])
)
fig_x.update_layout(
    paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
    font=dict(color=C["text"],family="Arial, Helvetica, sans-serif"),
    title=dict(
        text=f"<b>Combine the two physiological domains</b><br><span style='font-size:13px;color:{C['muted']}'>H and G together form the raw 7-D state X</span>",
        x=.01,xanchor="left"
    ),
    margin=dict(l=0,r=0,t=80,b=0),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    scene=dict(
        bgcolor=C["bg"],
        xaxis=dict(showticklabels=False,title="",backgroundcolor=C["panel"],gridcolor=C["grid"]),
        yaxis=dict(showticklabels=False,title="",backgroundcolor=C["panel"],gridcolor=C["grid"]),
        zaxis=dict(showticklabels=False,title="",backgroundcolor=C["panel"],gridcolor=C["grid"]),
        aspectmode="cube"
    )
)

# -----------------------------------------------------------------------------
# 4. PCA(X)
# -----------------------------------------------------------------------------

x_axis = [
    top_loading_label(Xload[:,0], X_NAMES, "Physiology axis 1"),
    top_loading_label(Xload[:,1], X_NAMES, "Physiology axis 2"),
    top_loading_label(Xload[:,2], X_NAMES, "Physiology axis 3"),
]
x_vecs = Xload * 2.5
fig_pca = cloud_figure(
    Xscore, C["purple"],
    "PCA(X) — a 3-D viewing window into the 7-D physiology",
    f"First 3 PCs show {100*Xvar.sum():.1f}% of full-X variance. This is for visualization, not the final representation.",
    x_axis,
    labels=X_NAMES.tolist(),
    label_points=x_vecs,
    note="Important: the real X ↔ Z map remains 7-D and full-rank; this PCA view is only for human visualization."
)

# -----------------------------------------------------------------------------
# 5. Animated X -> Z coordinate change
# -----------------------------------------------------------------------------

n_anim = min(260, len(d))
anim_idx = rng.choice(len(d), n_anim, replace=False)
Xa = Xc[anim_idx]
Za = Zalign[anim_idx]

fig_morph = go.Figure()
# initial soft cloud
Xs,Ys,Zs = covariance_surface(Xa, radius=1.50)
fig_morph.add_trace(go.Surface(
    x=Xs,y=Ys,z=Zs,
    surfacecolor=np.zeros_like(Xs),
    colorscale=[[0,C["purple"]],[1,C["purple"]]],
    opacity=.14, showscale=False, hoverinfo="skip",
    name="population cloud", showlegend=False
))
fig_morph.add_trace(go.Scatter3d(
    x=Xa[:,0],y=Xa[:,1],z=Xa[:,2],
    mode="markers",
    marker=dict(size=3.3,color=C["white"],opacity=.40),
    name="same subjects"
))

frames = []
ts = np.linspace(0,1,25)
for k,t in enumerate(ts):
    P = (1-t)*Xa + t*Za
    # color fades purple -> green
    r1,g1,b1 = (180,140,255)
    r2,g2,b2 = (0,230,154)
    rr = int((1-t)*r1+t*r2); gg=int((1-t)*g1+t*g2); bb=int((1-t)*b1+t*b2)
    col = f"rgb({rr},{gg},{bb})"
    SX,SY,SZ = covariance_surface(P, radius=1.50)
    frames.append(go.Frame(
        name=f"{k}",
        data=[
            go.Surface(
                x=SX,y=SY,z=SZ,
                surfacecolor=np.zeros_like(SX),
                colorscale=[[0,col],[1,col]],
                opacity=.14, showscale=False, hoverinfo="skip"
            ),
            go.Scatter3d(
                x=P[:,0],y=P[:,1],z=P[:,2],
                mode="markers",
                marker=dict(size=3.3,color=C["white"],opacity=.40)
            )
        ],
        layout=go.Layout(
            title=dict(
                text=(
                    f"<b>Coordinate change: X → Z</b><br>"
                    f"<span style='font-size:13px;color:{C['muted']}'>"
                    f"visual interpolation {100*t:.0f}% toward the learned latent coordinates"
                    f"</span>"
                )
            )
        )
    ))

fig_morph.frames = frames
fig_morph.update_layout(
    paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
    font=dict(color=C["text"],family="Arial, Helvetica, sans-serif"),
    title=dict(
        text=f"<b>Coordinate change: X → Z</b><br><span style='font-size:13px;color:{C['muted']}'>Same subjects, reorganized into a more interpretable coordinate system</span>",
        x=.01,xanchor="left"
    ),
    margin=dict(l=0,r=0,t=90,b=0),
    scene=dict(
        bgcolor=C["bg"],
        xaxis=dict(title="visual coordinate 1",showticklabels=False,backgroundcolor=C["panel"],gridcolor=C["grid"]),
        yaxis=dict(title="visual coordinate 2",showticklabels=False,backgroundcolor=C["panel"],gridcolor=C["grid"]),
        zaxis=dict(title="visual coordinate 3",showticklabels=False,backgroundcolor=C["panel"],gridcolor=C["grid"]),
        aspectmode="cube"
    ),
    updatemenus=[
        dict(
            type="buttons", direction="left", x=.01, y=1.05,
            buttons=[
                dict(label="▶ Transform X → Z", method="animate",
                     args=[None, {"frame":{"duration":90,"redraw":True},"fromcurrent":True,"transition":{"duration":0}}]),
                dict(label="⏮ Reset", method="animate",
                     args=[["0"], {"frame":{"duration":0,"redraw":True},"mode":"immediate"}])
            ]
        )
    ],
    sliders=[dict(
        x=.18,y=.02,len=.64,
        currentvalue=dict(prefix="coordinate interpolation: "),
        steps=[
            dict(method="animate", label=f"{int(100*t)}%",
                 args=[[f"{k}"],{"frame":{"duration":0,"redraw":True},"mode":"immediate"}])
            for k,t in enumerate(ts)
        ]
    )]
)
fig_morph.add_annotation(
    x=.5,y=.94,xref="paper",yref="paper",
    text="<b>X</b>: measured variables &nbsp;&nbsp;→&nbsp;&nbsp; <b>Z</b>: shared / mismatch / H-private / G-private coordinates",
    showarrow=False,font=dict(size=13,color=C["text"])
)
fig_morph.add_annotation(
    x=.01,y=.02,xref="paper",yref="paper",
    text="The in-between frames are a visual interpolation between the real endpoints, not optimization steps.",
    showarrow=False,font=dict(size=11,color=C["muted"])
)

# -----------------------------------------------------------------------------
# 6. Learned Z with semantic axes
# -----------------------------------------------------------------------------

fig_z = cloud_figure(
    Z3, C["z"],
    "Z — learned physiological coordinates",
    "Outcome-independent representation: phenotype was not used to construct these coordinates.",
    [
        "Shared physiology\n(coordinated H–G variation)",
        "Mismatch physiology\n(H and G move differently)",
        "H-specific signal\n(variation unique to blood block)"
    ],
    note=(
        "Shown axes are 3 of the 7 full-rank latent coordinates. "
        "The full Z also contains Shared 2, Mismatch 2, H-private 4 and G-private 3."
    )
)

# -----------------------------------------------------------------------------
# 7. Z -> Y network (no plane)
# -----------------------------------------------------------------------------

beta = {
    "Somatic": np.array([0.121246,-0.071556,-0.062547,-0.133865,0.194142,0.064871,0.067662]),
    "Cognitive-affective": np.array([0.117490,-0.026556,-0.053620,-0.080085,0.184665,0.093569,0.064631]),
    "PHQ-9 total": np.array([0.244403,-0.098316,-0.118846,-0.219315,0.379132,0.156600,0.129840]),
}
comp_y = np.linspace(.88,.18,7)
out_y = {"Somatic":.77,"Cognitive-affective":.50,"PHQ-9 total":.23}

fig_y = go.Figure()
# nodes
fig_y.add_trace(go.Scatter(
    x=np.repeat(.16,7), y=comp_y,
    mode="markers+text",
    marker=dict(size=18,color=C["z"],line=dict(color=C["white"],width=1)),
    text=Z_LABELS, textposition="middle right",
    hovertemplate="%{text}<extra></extra>",
    showlegend=False
))
for out, yy in out_y.items():
    fig_y.add_trace(go.Scatter(
        x=[.84],y=[yy],mode="markers+text",
        marker=dict(size=28,color=C["g"],line=dict(color=C["white"],width=1.3)),
        text=[out],textposition="middle left",
        hoverinfo="skip",showlegend=False
    ))
# edges
for oi,(out, vals) in enumerate(beta.items()):
    yy2 = out_y[out]
    maxabs = max(abs(vals))
    for i,v in enumerate(vals):
        width = 1.0 + 7.0*abs(v)/maxabs
        color = C["pos"] if v >= 0 else C["neg"]
        fig_y.add_trace(go.Scatter(
            x=[.30,.72],y=[comp_y[i],yy2],
            mode="lines",
            line=dict(width=width,color=color),
            opacity=.60,
            hovertemplate=f"{Z_LABELS[i]} → {out}<br>β={v:+.3f}<extra></extra>",
            showlegend=False
        ))
fig_y.add_annotation(
    x=.50,y=.96,
    text="<b>Y = h(Z)</b>",
    showarrow=False,font=dict(size=25,color=C["white"])
)
fig_y.add_annotation(
    x=.50,y=.05,
    text="Line thickness = |fitted association| · blue = positive · pink = negative",
    showarrow=False,font=dict(size=12,color=C["muted"])
)
fig_y.update_xaxes(visible=False,range=[0,1])
fig_y.update_yaxes(visible=False,range=[0,1])
fig_y.update_layout(
    paper_bgcolor=C["bg"],plot_bgcolor=C["bg"],
    font=dict(color=C["text"],family="Arial, Helvetica, sans-serif"),
    title=dict(
        text=f"<b>Latent physiology → depressive phenotype</b><br><span style='font-size:13px;color:{C['muted']}'>The model maps the same physiological coordinates to three related symptom outputs</span>",
        x=.01,xanchor="left"
    ),
    margin=dict(l=10,r=10,t=80,b=10)
)

# -----------------------------------------------------------------------------
# 8. Derivatives — local fitted sensitivity, plus causal boundary
# -----------------------------------------------------------------------------

raw_deriv_phq = {
    "Hb": 0.519628,
    "RBC": -0.768724,
    "MCV": -0.259267,
    "RDW": 0.392235,
    "HbA1c": 0.153647,
    "Glucose": 0.057087,
    "logInsulin": 0.198612,
}
names = list(raw_deriv_phq)
vals = np.array([raw_deriv_phq[n] for n in names])
colors = [C["pos"] if v >= 0 else C["neg"] for v in vals]

fig_d = go.Figure(go.Bar(
    x=vals,
    y=names,
    orientation="h",
    marker=dict(color=colors),
    hovertemplate="%{y}<br>standardized derivative=%{x:+.3f}<extra></extra>"
))
fig_d.add_vline(x=0,line_color=C["white"],line_width=1)
fig_d.update_layout(
    paper_bgcolor=C["bg"],plot_bgcolor=C["bg"],
    font=dict(color=C["text"],family="Arial, Helvetica, sans-serif"),
    title=dict(
        text=f"<b>Derivative audit — how much does predicted PHQ change locally?</b><br><span style='font-size:13px;color:{C['muted']}'>"
             "dY/dX = (dY/dZ)(dZ/dX) · standardized raw-variable sensitivities</span>",
        x=.01,xanchor="left"
    ),
    margin=dict(l=100,r=20,t=95,b=70),
    xaxis=dict(title="Local fitted sensitivity (standardized scale)",gridcolor=C["grid"],zeroline=False),
    yaxis=dict(title="",gridcolor=C["grid"],autorange="reversed"),
)
fig_d.add_annotation(
    x=.99,y=.04,xref="paper",yref="paper",
    text="<b>What this supports</b><br>functional dependence / local sensitivity",
    showarrow=False,align="right",
    bgcolor="rgba(49,215,255,.10)",bordercolor=C["h"],borderwidth=1,
    font=dict(size=12,color=C["text"])
)
fig_d.add_annotation(
    x=.99,y=.23,xref="paper",yref="paper",
    text="<b>What it does NOT establish</b><br>causal effect",
    showarrow=False,align="right",
    bgcolor="rgba(255,102,133,.10)",bordercolor=C["neg"],borderwidth=1,
    font=dict(size=12,color=C["text"])
)
fig_d.add_annotation(
    x=.56,y=-.18,xref="paper",yref="paper",
    text="Causal target would be ∂E[Y | do(Xᵢ=x)]/∂x and needs identification assumptions / longitudinal or quasi-experimental structure.",
    showarrow=False,align="center",
    font=dict(size=11,color=C["muted"])
)

# -----------------------------------------------------------------------------
# 9. Time drift
# -----------------------------------------------------------------------------

time_beta = {
    "2005-2008": np.array([0.3418,-0.1312,-0.1551,-0.3388,0.4530,0.2935,0.1009]),
    "2009-2018": np.array([0.2567,-0.1504,-0.1104,-0.2993,0.2861,0.0856,0.0047]),
    "2021-2023": np.array([0.1398,0.1870,-0.2698,-0.4386,0.0215,0.1952,-0.1804]),
}

fig_t = go.Figure()
for p in PERIODS:
    vals = time_beta[p]
    fig_t.add_trace(go.Bar(
        x=vals,y=Z_LABELS,orientation="h",
        name=p,visible=(p=="2005-2008"),
        marker=dict(color=[C["pos"] if v>=0 else C["neg"] for v in vals]),
        hovertemplate=f"{p}<br>%{{y}}: %{{x:+.3f}}<extra></extra>"
    ))
fig_t.add_vline(x=0,line_color=C["white"],line_width=1)
fig_t.update_layout(
    paper_bgcolor=C["bg"],plot_bgcolor=C["bg"],
    font=dict(color=C["text"],family="Arial, Helvetica, sans-serif"),
    title=dict(
        text=f"<b>Time changes the Z → Y bridge</b><br><span style='font-size:13px;color:{C['muted']}'>PHQ-total component coefficients by NHANES period</span>",
        x=.01,xanchor="left"
    ),
    margin=dict(l=120,r=20,t=100,b=60),
    xaxis=dict(title="Component → PHQ coefficient",gridcolor=C["grid"]),
    yaxis=dict(autorange="reversed",gridcolor=C["grid"]),
    updatemenus=[dict(
        type="buttons",direction="left",x=.01,y=1.08,
        buttons=[
            dict(label=p,method="update",
                 args=[{"visible":[q==p for q in PERIODS]},
                       {"title":{"text":f"<b>Time changes the Z → Y bridge</b><br><span style='font-size:13px;color:{C['muted']}'>{p}: PHQ-total component coefficients</span>"}}])
            for p in PERIODS
        ]
    )]
)
fig_t.add_annotation(
    x=.99,y=.02,xref="paper",yref="paper",
    text="Z stays the coordinate system; the phenotype mapping h(Z,t) changes.",
    showarrow=False,font=dict(size=12,color=C["muted"]),align="right"
)

# -----------------------------------------------------------------------------
# 10. Reverse inference: same phenotype band -> region in Z
# -----------------------------------------------------------------------------

phq = d[y_cols["PHQ"]].to_numpy(float)
moderate = (phq >= 10) & (phq <= 14)
Pmod = Z3[moderate]
if len(Pmod) < 25:
    # fallback 8-14
    moderate = (phq >= 8) & (phq <= 14)
    Pmod = Z3[moderate]

fig_r = go.Figure()
SX,SY,SZ = covariance_surface(Pmod, radius=1.75)
fig_r.add_trace(go.Surface(
    x=SX,y=SY,z=SZ,
    surfacecolor=np.zeros_like(SX),
    colorscale=[[0,C["neg"]],[1,C["neg"]]],
    opacity=.18,showscale=False,hoverinfo="skip",name="plausible region"
))
rid = rng.choice(len(Pmod), min(100,len(Pmod)), replace=False)
fig_r.add_trace(go.Scatter3d(
    x=Pmod[rid,0],y=Pmod[rid,1],z=Pmod[rid,2],
    mode="markers",
    marker=dict(size=3.4,color=C["white"],opacity=.38),
    name="subjects with PHQ 10–14"
))
cent = Pmod.mean(axis=0)
fig_r.add_trace(go.Scatter3d(
    x=[cent[0]],y=[cent[1]],z=[cent[2]],
    mode="markers+text",
    marker=dict(size=9,color=C["g"],line=dict(color=C["white"],width=1)),
    text=["same symptom band"],textposition="top center",
    name="conditional center"
))
fig_r.update_layout(
    paper_bgcolor=C["bg"],plot_bgcolor=C["bg"],
    font=dict(color=C["text"],family="Arial, Helvetica, sans-serif"),
    title=dict(
        text=f"<b>Reverse inference is a region, not a point</b><br><span style='font-size:13px;color:{C['muted']}'>p(Z | Y): people with PHQ 10–14 occupy many plausible physiological states</span>",
        x=.01,xanchor="left"
    ),
    margin=dict(l=0,r=0,t=90,b=0),
    scene=dict(
        bgcolor=C["bg"],
        xaxis=dict(title="Shared physiology",showticklabels=False,backgroundcolor=C["panel"],gridcolor=C["grid"]),
        yaxis=dict(title="Mismatch physiology",showticklabels=False,backgroundcolor=C["panel"],gridcolor=C["grid"]),
        zaxis=dict(title="H-specific physiology",showticklabels=False,backgroundcolor=C["panel"],gridcolor=C["grid"]),
        aspectmode="cube"
    )
)
fig_r.add_annotation(
    x=.01,y=.02,xref="paper",yref="paper",
    text="This is why deterministic Y → Z reconstruction failed: similar symptom burden does not identify one unique physiology.",
    showarrow=False,font=dict(size=12,color=C["muted"])
)

# -----------------------------------------------------------------------------
# 11. Longitudinal refinement — meaningful slider
# -----------------------------------------------------------------------------

rho_vals = np.array([0.00,0.25,0.50,0.75,0.90])
rel_vol = np.array([0.995,0.928,0.423,0.052,0.004])  # Script 84 simulation
base_center = Pmod.mean(axis=0)
centered = Pmod - base_center

# choose representative sample
Lidx = rng.choice(len(Pmod), min(120,len(Pmod)), replace=False)
sample_base = centered[Lidx]

fig_l = go.Figure()
scale0 = rel_vol[0] ** (1/3)
P0 = base_center + centered * scale0
SX,SY,SZ = covariance_surface(P0, radius=1.70)
fig_l.add_trace(go.Surface(
    x=SX,y=SY,z=SZ,
    surfacecolor=np.zeros_like(SX),
    colorscale=[[0,C["h"]],[1,C["h"]]],
    opacity=.18,showscale=False,hoverinfo="skip"
))
S0 = base_center + sample_base*scale0
fig_l.add_trace(go.Scatter3d(
    x=S0[:,0],y=S0[:,1],z=S0[:,2],
    mode="markers",
    marker=dict(size=3.2,color=C["white"],opacity=.35),
    showlegend=False
))

frames_l = []
for i,(rho,rv) in enumerate(zip(rho_vals,rel_vol)):
    s = rv ** (1/3)
    P = base_center + centered*s
    SX,SY,SZ = covariance_surface(P, radius=1.70)
    Sp = base_center + sample_base*s
    frames_l.append(go.Frame(
        name=str(i),
        data=[
            go.Surface(
                x=SX,y=SY,z=SZ,
                surfacecolor=np.zeros_like(SX),
                colorscale=[[0,C["h"]],[1,C["h"]]],
                opacity=.18,showscale=False,hoverinfo="skip"
            ),
            go.Scatter3d(
                x=Sp[:,0],y=Sp[:,1],z=Sp[:,2],
                mode="markers",
                marker=dict(size=3.2,color=C["white"],opacity=.35)
            )
        ],
        layout=go.Layout(
            title=dict(
                text=(
                    f"<b>Longitudinal reference narrows p(Zₜ | ·)</b><br>"
                    f"<span style='font-size:13px;color:{C['muted']}'>"
                    f"simulation: physiological continuity ρ={rho:.2f} · relative uncertainty volume={rv:.3f}"
                    f"</span>"
                )
            )
        )
    ))
fig_l.frames = frames_l
fig_l.update_layout(
    paper_bgcolor=C["bg"],plot_bgcolor=C["bg"],
    font=dict(color=C["text"],family="Arial, Helvetica, sans-serif"),
    title=dict(
        text=f"<b>Longitudinal reference narrows p(Zₜ | ·)</b><br><span style='font-size:13px;color:{C['muted']}'>Simulation: add previous physiology Zₜ₋₁ to the reverse bridge</span>",
        x=.01,xanchor="left"
    ),
    margin=dict(l=0,r=0,t=90,b=0),
    scene=dict(
        bgcolor=C["bg"],
        xaxis=dict(title="Shared physiology",showticklabels=False,backgroundcolor=C["panel"],gridcolor=C["grid"]),
        yaxis=dict(title="Mismatch physiology",showticklabels=False,backgroundcolor=C["panel"],gridcolor=C["grid"]),
        zaxis=dict(title="H-specific physiology",showticklabels=False,backgroundcolor=C["panel"],gridcolor=C["grid"]),
        aspectmode="cube"
    ),
    updatemenus=[dict(
        type="buttons",direction="left",x=.01,y=1.05,
        buttons=[
            dict(label="▶ increase continuity",method="animate",
                 args=[None,{"frame":{"duration":550,"redraw":True},"fromcurrent":True,"transition":{"duration":0}}]),
            dict(label="⏮ reset",method="animate",
                 args=[["0"],{"frame":{"duration":0,"redraw":True},"mode":"immediate"}])
        ]
    )],
    sliders=[dict(
        x=.18,y=.02,len=.64,currentvalue=dict(prefix="ρ = "),
        steps=[
            dict(method="animate",label=f"{rho:.2f}",
                 args=[[str(i)],{"frame":{"duration":0,"redraw":True},"mode":"immediate"}])
            for i,rho in enumerate(rho_vals)
        ]
    )]
)
fig_l.add_annotation(
    x=.01,y=.02,xref="paper",yref="paper",
    text="SIMULATION only. The slider is meaningful here: each position changes the assumed continuity and contracts the plausible physiology region accordingly.",
    showarrow=False,font=dict(size=11,color=C["muted"])
)

# -----------------------------------------------------------------------------
# Common legend
# -----------------------------------------------------------------------------

legend_html = f"""
<div class="legend-card">
  <b>How to read the visuals</b>
  <span><i class="swatch cloud"></i> translucent volume = population / plausible region</span>
  <span><i class="dot"></i> soft dots = example subjects</span>
  <span><i class="line pos"></i> blue = positive fitted direction</span>
  <span><i class="line neg"></i> pink = negative fitted direction</span>
  <span><b>X → Z</b> = coordinate change / representation</span>
  <span><b>Z → Y</b> = phenotype association map</span>
  <span><b>p(Z|Y)</b> = probabilistic reverse inference</span>
</div>
"""

# -----------------------------------------------------------------------------
# Build single HTML
# -----------------------------------------------------------------------------

sections = [
    ("h", "H block", fig_h, "Start with the blood measurements."),
    ("g", "G block", fig_g, "Then isolate the glycaemic measurements."),
    ("x", "X = [H,G]", fig_x, "Combine both domains into one 7-D physiological state."),
    ("pca", "PCA(X)", fig_pca, "Open a 3-D viewing window; this is not the final model."),
    ("morph", "X → Z", fig_morph, "Watch the same subjects reorganize into latent coordinates."),
    ("z", "Meaning of Z", fig_z, "Interpret the learned physiology axes."),
    ("zy", "Z → Y", fig_y, "See which latent components connect to each symptom phenotype."),
    ("deriv", "Derivatives", fig_d, "Trace local fitted sensitivity back to raw physiology."),
    ("time", "Time drift", fig_t, "The phenotype bridge changes across periods."),
    ("reverse", "Reverse", fig_r, "Symptoms map back to a region, not one exact physiology."),
    ("long", "Longitudinal", fig_l, "A previous physiological state can narrow that region in simulation."),
]

plotly_js = get_plotlyjs()

nav_buttons = "\n".join(
    f'<button class="navbtn" onclick="showSection(\'{sid}\', this)">{label}</button>'
    for sid,label,_,_ in sections
)

section_html = []
for i,(sid,label,fig,caption) in enumerate(sections):
    display = "block" if i==0 else "none"
    section_html.append(f"""
    <section id="sec-{sid}" class="story-section" style="display:{display}">
      <div class="chapter-kicker">CHAPTER {i+1:02d}</div>
      <div class="chapter-caption">{caption}</div>
      {to_fragment(fig)}
    </section>
    """)

html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Physiological Decomposition — Conference Math Story</title>
<script>{plotly_js}</script>
<style>
html, body {{
  margin:0; padding:0; background:{C["bg"]}; color:{C["text"]};
  font-family:Arial,Helvetica,sans-serif;
}}
.header {{
  padding:18px 26px 8px 26px;
  border-bottom:1px solid #1b2a3a;
  position:sticky; top:0; z-index:20;
  background:rgba(5,9,18,.96);
}}
.header h1 {{
  margin:0; font-size:22px; letter-spacing:.2px;
}}
.header p {{
  margin:4px 0 10px 0; color:{C["muted"]}; font-size:13px;
}}
.nav {{
  display:flex; flex-wrap:wrap; gap:7px;
}}
.navbtn {{
  border:1px solid #294258; background:#0a1624; color:{C["text"]};
  border-radius:7px; padding:7px 11px; cursor:pointer; font-size:12px;
}}
.navbtn:hover, .navbtn.active {{
  background:#12314a; border-color:{C["h"]};
}}
.wrap {{
  max-width:1320px; margin:0 auto; padding:14px 18px 36px 18px;
}}
.legend-card {{
  display:flex; flex-wrap:wrap; gap:14px 20px;
  border:1px solid #24384c; background:#08121e; border-radius:10px;
  padding:10px 14px; color:{C["muted"]}; font-size:12px;
  margin-bottom:12px;
}}
.legend-card b {{ color:{C["text"]}; }}
.swatch.cloud {{
  display:inline-block;width:18px;height:10px;border-radius:8px;
  background:rgba(49,215,255,.20);border:1px solid {C["h"]};margin-right:5px;
}}
.dot {{
  display:inline-block;width:7px;height:7px;border-radius:50%;
  background:{C["white"]};opacity:.55;margin-right:5px;
}}
.line {{
  display:inline-block;width:18px;height:3px;margin-right:5px;vertical-align:middle;
}}
.line.pos {{ background:{C["pos"]}; }}
.line.neg {{ background:{C["neg"]}; }}
.chapter-kicker {{
  color:{C["h"]};font-size:11px;font-weight:bold;letter-spacing:1.6px;
  margin:4px 0;
}}
.chapter-caption {{
  color:{C["muted"]};font-size:13px;margin-bottom:4px;
}}
.story-section {{
  min-height:760px;
}}
</style>
</head>
<body>
  <div class="header">
    <h1>From measured physiology to latent structure to depressive phenotype</h1>
    <p>Interactive mathematical story · H/G decomposition · derivatives · temporal drift · probabilistic reverse inference</p>
    <div class="nav">{nav_buttons}</div>
  </div>
  <div class="wrap">
    {legend_html}
    {''.join(section_html)}
  </div>
<script>
function showSection(id, btn) {{
  document.querySelectorAll('.story-section').forEach(s => s.style.display='none');
  document.getElementById('sec-' + id).style.display='block';
  document.querySelectorAll('.navbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
  window.scrollTo({{top:0,behavior:'smooth'}});
}}
document.querySelector('.navbtn').classList.add('active');
</script>
</body>
</html>
"""

outfile = OUT / "89_conference_math_story.html"
outfile.write_text(html, encoding="utf-8")

print("="*100)
print("SCRIPT 89 — CONFERENCE MATH STORY")
print("="*100)
print(f"Input: {INPUT}")
print(f"H 3-PC variance shown: {100*Hvar.sum():.2f}%")
print(f"G 3-PC variance shown: {100*Gvar.sum():.2f}%")
print(f"X 3-PC variance shown: {100*Xvar.sum():.2f}%")
print(f"Output: {outfile}")
print("="*100)
