"""Manuscript figures from saved NHANES results. Run from the project's Python environment."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "Results"
OUT = RESULTS / "115_paper_figures" / "revised"
SURVEY = RESULTS / "106_survey_ferritin_anchor"
INK, TEAL, RUST = "#20272B", "#086762", "#A74E29"
DENSITY = LinearSegmentedColormap.from_list("weighted_mass", ["#FFFFFF","#E6EFED","#ABCBC5","#559B91","#18645F","#103B3B"])
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8.5, "axes.labelsize":8.5,
                     "axes.titlesize":10, "axes.titleweight":"bold", "axes.titlelocation":"left",
                     "axes.edgecolor":"#859396", "axes.linewidth":.6, "axes.spines.top":False,
                     "axes.spines.right":False, "text.color":INK, "axes.labelcolor":INK,
                     "xtick.color":INK,"ytick.color":INK,"xtick.labelsize":8,"ytick.labelsize":8,
                     "pdf.fonttype":42, "savefig.facecolor":"white"})
EDGES = np.linspace(-3.5,3.5,57)


def unique_index(frame, keys):
    indexed = frame.set_index(keys)
    if not indexed.index.is_unique:
        raise ValueError(f"Duplicate figure input keys: {keys}")
    return indexed

def export(fig, name, caption):
    fig.savefig(OUT/name, metadata={"Title":name[:-4], "Subject":caption})
    plt.close(fig)
    print(f"Saved {name}", flush=True)


def inputs():
    pairs = pd.read_csv(RESULTS/"72_shared_private_cca_summary.csv").pipe(unique_index, "pair")
    cols = ["PERIOD","SURVEY_WT"] + [f"{prefix}{j}" for j in (1,2)
            for prefix in ("CCA_A","CCA_G","SP_SHARED","SP_DISCORD")]
    data = pd.read_csv(RESULTS/"72_shared_private_model_input.csv", usecols=cols)
    d = data.loc[data.PERIOD.eq("2005-2008")].replace([np.inf,-np.inf],np.nan).dropna()
    d = d.loc[d.SURVEY_WT.gt(0)]
    if d.empty:
        raise ValueError("No complete discovery observations for figure 1")
    for j in (1,2):
        u,v = d[f"CCA_A{j}"], d[f"CCA_G{j}"]
        rho = pairs.loc[j,"discovery_canonical_corr"]
        if not (np.allclose((u+v)/np.sqrt(2*(1+rho)),d[f"SP_SHARED{j}"]) and
                np.allclose((u-v)/np.sqrt(2*(1-rho)),d[f"SP_DISCORD{j}"])):
            raise ValueError(f"Saved coordinate convention differs for pair {j}")
    return d, pairs


def mass(x,y,w):
    return np.histogram2d(x,y,bins=(EDGES,EDGES),weights=w)[0].T * (100/w.sum())


def field(ax,h,norm):
    mesh = ax.pcolormesh(EDGES,EDGES,h,cmap=DENSITY,norm=norm,shading="flat",rasterized=True)
    ax.set(aspect="equal",xticks=[-3,0,3],yticks=[-3,0,3],xlim=(-3.5,3.5),ylim=(-3.5,3.5))
    ax.axhline(0,color=INK,lw=.45,alpha=.45)
    ax.axvline(0,color=INK,lw=.45,alpha=.45)
    return mesh


def geometry(d,pairs):
    w = d.SURVEY_WT.to_numpy()
    fields = []
    for j in (1,2):
        fields.append(mass(d[f"CCA_A{j}"],d[f"CCA_G{j}"],w))
        fields.append(mass(d[f"SP_SHARED{j}"],d[f"SP_DISCORD{j}"],w))
    norm = PowerNorm(.55,vmin=0,vmax=max(float(h.max()) for h in fields))
    fig = plt.figure(figsize=(7.15,6.35))
    grid = fig.add_gridspec(2,2,left=.10,right=.81,top=.89,bottom=.15,wspace=.26,hspace=.51)
    for i,h in enumerate(fields):
        j = i//2+1
        ax = fig.add_subplot(grid[i//2,i%2])
        mesh = field(ax,h,norm)
        if i%2==0:
            ax.set(xlabel="Haematology score U",ylabel="Glycaemic score V")
            for sign,color in [(1,TEAL),(-1,RUST)]:
                ax.plot([-3.3,3.3],[-3.3*sign,3.3*sign],ls="--",lw=.8,color=color)
            subtitle = f"Pair {j}   ·   ρ = {pairs.loc[j,'discovery_canonical_corr']:.2f}"
        else:
            ax.set(xlabel="Shared coordinate S",ylabel="Discordant coordinate D")
            subtitle = f"Pair {j}   ·   frozen Z coordinates"
        ax.set_title(f"{'ABCD'[i]}    {subtitle}",pad=7)
    cax = fig.add_axes([.865,.40,.018,.25])
    fig.colorbar(mesh,cax=cax)
    cax.set_title("Weight\nper bin (%)",fontsize=8,loc="left")
    fig.text(.08,.965,"From paired domain scores to shared and discordant coordinates",fontsize=12,weight="bold")
    fig.text(.08,.065,"Dashed diagonals: agreement (teal) and divergence (rust). Each bin is 0.125 × 0.125 score units.",fontsize=8)
    fig.text(.08,.038,f"NHANES 2005–2008   ·   n = {len(d):,}   ·   Percentages use total discovery weight, including tails outside the panels.",fontsize=7.5)
    export(fig,"01_coordinate_geometry.pdf","Survey-weighted observed distributions of both retained canonical pairs, before and after frozen sum/difference reparameterisation.")


def period_key(ax, labels):
    handles = [Line2D([],[],color=c,marker=m,lw=1.2,ms=4,label=lab)
               for c,m,lab in zip((TEAL,RUST),("o","s"),labels)]
    ax.legend(handles=handles,loc="center",frameon=False,ncol=2,columnspacing=2.5)
    ax.set_axis_off()


def dot_intervals(ax, subset, rows, labels, key, periods, period_col, value, lo, hi):
    tab = subset.pipe(unique_index, [key,period_col])
    for i,row in enumerate(rows):
        if i%2==0:
            ax.axhspan(i-.48,i+.48,color="#F1F4F3",zorder=0)
        for p,dy,c,m in zip(periods,(-.14,.14),(TEAL,RUST),("o","s")):
            a,b,v = tab.loc[(row,p),[lo,hi,value]].to_numpy(dtype=float)
            if not np.isfinite([a,b,v]).all() or b<a:
                raise ValueError(f"Invalid interval for {row}, {p}")
            ax.hlines(i+dy,a,b,color=c,lw=1.5)
            ax.plot(v,i+dy,m,color=c,ms=4)
    ax.set_yticks(range(len(rows)),labels)
    ax.set_ylim(len(rows)-.45,-.55)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y",length=0)
    ax.xaxis.set_major_locator(MaxNLocator(5))


def transport():
    redis = pd.read_csv(SURVEY/"106_design_rediscovery.csv",dtype={"reference":str,"target":str})
    phq = pd.read_csv(SURVEY/"106_phq_psu_uncertainty.csv",dtype={"cohort":str})
    fig = plt.figure(figsize=(7.15,5.75))
    grid = fig.add_gridspec(3,1,left=.26,right=.96,top=.85,bottom=.14,height_ratios=[4,3,.48],hspace=.8)
    a,b,legend = [fig.add_subplot(grid[i]) for i in range(3)]
    periods = ["0708","2123"]
    low, high = "psu_rescaled_low_2p5","psu_rescaled_high_97p5"
    dot_intervals(a,redis.loc[redis.reference.eq("0506")],
                  ["Shared2","Discordant2","H_Isolated2","G_Isolated1"],
                  ["Shared","Discordant","Haematology-specific","Glycaemia-specific"],
                  "group",periods,"target","observed_similarity",low,high)
    a.set_xlim(0,1.015)
    a.set(title="A    Rediscovery of physiological subspaces",xlabel="Similarity to the 2005–2006 reference")
    dot_intervals(b,phq,["Covariates","RawX7","H_G_CCA7"],["Covariates","Covariates + raw markers","Covariates + Z"],
                  "method",periods,"cohort","point_frozen_phq_r2",low,high)
    b.axvline(0,color=INK,ls=":",lw=1)
    b.set(title="B    Transfer of the frozen PHQ-9 model",xlabel="Weighted out-of-period R²")
    period_key(legend,["2007–2008","2021–2023"])
    fig.text(.055,.95,"Physiological rediscovery and depression-score transport",fontsize=12,weight="bold")
    fig.text(.055,.051,"Approximate 95% intervals from 199 survey PSU resamples. PHQ-9 intervals condition on the discovery fit.",fontsize=7.5)
    fig.text(.055,.025,"R² below zero indicates greater squared error than predicting the target-period mean.",fontsize=7.5)
    export(fig,"02_structure_and_transport.pdf","Independent rediscovery and frozen total PHQ-9 prediction share a 2005–2006 discovery reference. Panels have distinct estimands.")


def benchmarks():
    df = pd.read_csv(RESULTS/"112_closure/112_paired_PSU_comparisons.csv")
    methods = [("rank5","PCA5",["FrozenZ5","JIVE5","AJIVE5_rankchecked"],["Z","JIVE","AJIVE"]),
               ("rank7","RawX7",["FrozenZ7","JIVE7","AJIVE7_rankchecked","PCA7"],["Z","JIVE","AJIVE","PCA"])]
    fig = plt.figure(figsize=(7.15,5.55))
    grid = fig.add_gridspec(3,3,left=.13,right=.975,top=.83,bottom=.14,height_ratios=[3,4,.52],wspace=.35,hspace=.83)
    for col,(outcome,label) in enumerate([("PHQ_total","Total PHQ-9"),("Somatic","Somatic"),("CogAff","Cognitive-affective")]):
        for r,(family,ref,rows,labels) in enumerate(methods):
            ax = fig.add_subplot(grid[r,col])
            sub = df.loc[df.analysis.eq("prediction") & df.family.eq(family) & df.reference.eq(ref) & df.outcome.eq(outcome)]
            dot_intervals(ax,sub,rows,labels,"method",["test_2009-2018","validate_2021-2023"],"period",
                          "paired_gain_positive_better","approx_PSU_low","approx_PSU_high")
            ax.axvline(0,color=INK,ls=":",lw=1)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            if col: ax.tick_params(axis="y",labelleft=False)
            ax.set_title(label if r==0 else "",fontsize=9)
    period_key(fig.add_subplot(grid[2,:]),["2009–2018","2021–2023"])
    fig.text(.055,.95,"Prediction differences at matched dimensional budgets",fontsize=12,weight="bold")
    fig.text(.055,.895,"A    Five coordinates compared with PCA5",weight="bold")
    pos = grid[1,:].get_position(fig)
    fig.text(.055,pos.y1+.017,"B    Seven coordinates compared with seven raw markers",weight="bold")
    fig.text(.055,.059,"Horizontal axis: reference MSE minus method MSE. Positive values favour the tested method. Panel scales vary.",fontsize=7.5)
    fig.text(.055,.032,"Approximate 95% paired intervals from 999 survey PSU resamples, conditional on fitted models and unadjusted for multiplicity.",fontsize=7)
    export(fig,"03_matched_benchmarks.pdf","Complete matched-rank depression endpoint comparisons for both later periods.")


def animation(d,pairs):
    u,v,w = (d[k].to_numpy() for k in ("CCA_A1","CCA_G1","SURVEY_WT"))
    rho = float(pairs.loc[1,"discovery_canonical_corr"])
    angles = np.r_[np.zeros(7),np.linspace(0,np.pi/4,37),np.full(14,np.pi/4)]
    def project(theta):
        c,s = np.cos(theta),np.sin(theta)
        return ((c*u+s*v)/np.sqrt(1+2*c*s*rho),
                (s*u-c*v)/np.sqrt(1-2*c*s*rho))
    hist = [mass(*project(theta),w) for theta in angles]
    norm = PowerNorm(.55,vmin=0,vmax=max(h.max() for h in hist))
    fig, (a,b) = plt.subplots(1,2,figsize=(7.15,3.9))
    fig.subplots_adjust(left=.09,right=.93,bottom=.24,top=.82,wspace=.36)
    field(a,mass(u,v,w),norm)
    mesh = field(b,hist[0],norm)
    a.set(title="A    Projection directions",xlabel="Haematology score U",ylabel="Glycaemic score V")
    b.set(title="B    Projected distribution",xlabel="First projection",ylabel="Second projection")
    vectors = [a.annotate("",xy=(0,0),xytext=(0,0),arrowprops=dict(arrowstyle="-|>",lw=2,color=c)) for c in (TEAL,RUST)]
    note = fig.text(.08,.12,"",weight="bold",fontsize=9)
    fig.text(.08,.055,f"Discovery pair 1   ·   n = {len(d):,}   ·   Survey-weighted bin mass on a fixed colour scale",fontsize=8)
    def update(i):
        theta = angles[i]
        c,s = np.cos(theta),np.sin(theta)
        vectors[0].xy = (2.8*c,2.8*s)
        vectors[1].xy = (2.8*s,-2.8*c)
        mesh.set_array(hist[i].ravel())
        note.set_text("Start: U and −V" if i<7 else ("Final: standardised shared (U + V) and discordant (U − V) scores" if i>=43 else "Rotating and standardising both projections of the same observations"))
        return mesh,note,*vectors
    movie = FuncAnimation(fig,update,frames=len(angles),interval=110)
    movie.save(OUT/"S1_observed_coordinate_transform.gif",writer=PillowWriter(fps=9),dpi=125)
    plt.close(fig)
    print("Saved supplementary animation",flush=True)


def ferritin_anchor():
    frame = pd.read_csv(SURVEY/"106_ferritin_psu_uncertainty.csv", dtype={"cohort":str})
    coverage = pd.read_csv(SURVEY/"106_ferritin_coverage.csv", dtype={"cohort":str}).set_index("cohort")
    groups = ["Shared2","Discordant2","H_Isolated2","G_Isolated1","H4","G3","PCA2","RawX7"]
    names = ["Shared pair","Discordant pair","H-specific pair","G-specific score",
             "Four H markers","Three G markers","First two PCs","Seven raw markers"]
    periods = ["0708","2123"]
    fig, axes = plt.subplots(1,2,figsize=(7.15,4.45),sharex=True,sharey=True)
    fig.subplots_adjust(left=.20,right=.96,bottom=.25,top=.77,wspace=.12)
    for ax, period, year, colour in zip(axes,periods,["2007–2008","2021–2023"],[TEAL,RUST]):
        tab = frame.loc[frame.cohort.eq(period)].pipe(unique_index, "method").loc[groups]
        y = np.array([0,1,2,3,4.6,5.6,6.6,7.6])
        ax.axhspan(-.5,3.5,color="#EDF3F1")
        ax.axvline(0,ls=":",lw=1,color=INK)
        ax.hlines(y,tab.psu_rescaled_low_2p5,tab.psu_rescaled_high_97p5,color=colour,lw=1.5)
        ax.scatter(tab.delta_mse_vs_covariates,y,s=20,color=colour,zorder=3)
        ax.set_yticks(y,names)
        ax.set_ylim(8.15,-.55)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y",length=0)
        n = int(coverage.loc[period,"eligible_women_20_49"])
        ax.set_title(f"{year}   ·   n = {n}",fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    fig.text(.045,.93,"Ferritin as an independent assay of iron status",fontsize=13,weight="bold")
    fig.text(.045,.865,"Women aged 20–49 with complete fasting measurements and ferritin",fontsize=9)
    fig.text(.5,.15,"Reduction in log-ferritin MSE relative to covariates alone",ha="center",fontsize=9)
    fig.text(.045,.065,"Shading identifies Z components. Positive values indicate lower prediction error.",fontsize=8)
    fig.text(.045,.026,"Frozen 2005–2006 fits. Bars span the 2.5th to 97.5th percentiles of 199 holdout PSU resamples.",fontsize=7.5)
    export(fig,"04_ferritin_anchor.pdf","Ferritin prediction among eligible women. Panels share an effect scale. Intervals condition on the frozen fit.")


def domain_contributions():
    frame = pd.read_csv(RESULTS/"112_closure/113_triad_evidence.csv")
    contrasts = [("H4","Covariates"),("G3","Covariates"),("RawX7","Covariates"),
                 ("RawX7","H4"),("RawX7","G3")]
    refs = ["H added to covariates","G added to covariates","H + G added to covariates",
            "G added to H","H added to G"]
    table = frame.loc[frame.family.eq("triad") & frame.analysis.eq("prediction")].pipe(unique_index, ["outcome","method","reference","period"])
    fig, axes = plt.subplots(1,3,figsize=(7.15,4.05), sharey=True)
    fig.subplots_adjust(left=.26,right=.97,top=.74,bottom=.29,wspace=.23)
    y = np.array([0,1,2,3.5,4.5])
    for ax, outcome, title in zip(axes,["PHQ_total","Somatic","CogAff"],["Total PHQ-9","Somatic","Cognitive-affective"]):
        ax.axhspan(3,5,color="#EDF3F1")
        ax.axvline(0,color=INK,ls=":",lw=.9)
        for period, colour, marker, offset in zip(["test_2009-2018","validate_2021-2023"],
                                                  [TEAL,RUST],["o","s"],[-.14,.14]):
            rows = table.loc[[(outcome,m,r,period) for m,r in contrasts]]
            ax.hlines(y+offset,rows.approx_PSU_low,rows.approx_PSU_high,color=colour,lw=1.4)
            ax.scatter(rows.paired_gain_positive_better,y+offset,marker=marker,s=15,color=colour,zorder=3)
        ax.set_title(title,fontsize=9)
        ax.set_yticks(y,refs)
        ax.set_ylim(5.1,-.55)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y",length=0)
        ax.xaxis.set_major_locator(MaxNLocator(3))
    fig.text(.045,.935,"What does each marker domain add to depression-score prediction?",fontsize=12,weight="bold")
    fig.text(.045,.855,"H = haematology     G = glycaemia     Shaded rows compare combined and single-domain models.",fontsize=8)
    legend = fig.add_axes([.31,.125,.58,.055])
    period_key(legend,["2009–2018","2021–2023"])
    fig.text(.045,.073,"Reduction in mean squared error relative to each row’s reference. Positive favours adding the indicated domain.",fontsize=7.6)
    fig.text(.045,.033,"Outcome-specific scales. Approximate 95% paired intervals from 999 PSU resamples, unadjusted for multiplicity.",fontsize=7.3)
    export(fig,"05_domain_contributions.pdf","Frozen models on identical held-out rows. Adding a domain denotes a model comparison, not a causal intervention.")


def weighted_quantiles(values, weights, probabilities):
    order = np.argsort(values)
    x = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    midpoints = (np.cumsum(w) - .5*w) / w.sum()
    return np.interp(probabilities, midpoints, x)


def discordance_pilot(discovery):
    """Observed paired-score geometry, simulated selectivity and compression cost."""
    raw = pd.read_csv(RESULTS/"114_structure_recovery_v2/114_recovery.csv")
    sim = raw.loc[raw.scenario.eq("discordant")].pipe(
        unique_index, ["method","scope","target","replicate"])
    def pair(method, scope, target):
        rows = sim.loc[[(method,scope,target,r) for r in (0,1)]]
        vals = rows.heldout_R2.to_numpy(float)
        if not np.isfinite(vals).all():
            raise ValueError(f"Missing pilot recovery for {method}/{scope}/{target}")
        return vals

    fig = plt.figure(figsize=(7.35,7.4))
    fig.text(.065,.964,"Discordance in the physiological representation",size=16,weight="bold")
    fig.text(.065,.926,"Observed domain profiles and recovery of known signals in a two-replicate simulation",size=9)
    fig.add_artist(Line2D([.065,.955],[.903,.903],transform=fig.transFigure,color=INK,lw=.8))

    # Same narrow range of shared scores, with the difference coordinate varied.
    q = weighted_quantiles(discovery.SP_SHARED1, discovery.SURVEY_WT, [.45,.55])
    band = discovery.loc[discovery.SP_SHARED1.between(*q)].copy()
    cut = weighted_quantiles(band.SP_DISCORD1, band.SURVEY_WT, [1/3,2/3])
    bands = np.searchsorted(cut,band.SP_DISCORD1)
    fig.text(.065,.863,"A   Matched shared scores",size=10,weight="bold")
    fig.text(.065,.836,"NHANES 2005–2008  /  first canonical pair",size=8,color="#47565A")
    ax = fig.add_axes([.13,.58,.294,.213])
    profiles = []
    for k in range(3):
        group = band.loc[bands==k]
        profiles.append(np.average(group[["CCA_A1","CCA_G1"]],axis=0,weights=group.SURVEY_WT))
    bound = max(.5,np.max(np.abs(profiles))*1.25)
    ax.axhline(0,color="#ABB6B7",lw=.7)
    for profile,colour,marker,name in zip(profiles,[TEAL,"#6B7478",RUST],["o","D","s"],["Low D1","Mid D1","High D1"]):
        ax.plot([0,1],profile,color=colour,marker=marker,lw=1.8,ms=4)
        ax.annotate(name,(1,profile[1]),xytext=(6,0),textcoords="offset points",
                    va="center",color=colour,fontsize=7.4)
    ax.set(xlim=(-.12,1.58),ylim=(-bound,bound),xticks=[0,1],xticklabels=["Haematology","Glycaemia"],
           ylabel="Mean canonical score")
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x",length=0)
    fig.text(.065,.505,f"Shared score: weighted 45th–55th percentiles (n = {len(band):,}).\nLines connect domain means by discordance tertile.",va="top",size=7.2,linespacing=1.4)

    fig.text(.51,.863,"B   Component selectivity",size=10,weight="bold")
    fig.text(.51,.836,"Simulated discordance scenario  /  full Z7",size=8,color="#47565A")
    mat = fig.add_axes([.637,.58,.315,.213])
    groups = ["shared","discordant","H_private","G_private"]
    labels = ["Shared","Discord.","H-spec.","G-spec."]
    cells = np.array([[pair("HG7",scope,target) for target in groups] for scope in groups])
    # Both replicates printed in every cell. Colour summarises their mean.
    norm = matplotlib.colors.Normalize(0,1,clip=True)
    mat.imshow(cells.mean(2),cmap=DENSITY,norm=norm,aspect="auto",interpolation="nearest")
    for row in range(4):
        for col in range(4):
            vals = cells[row,col]
            colour = "white" if vals.mean()>.63 else INK
            mat.text(col,row,"\n".join(f"{v:.2f}" for v in vals),ha="center",va="center",
                     color=colour,fontsize=7.2,linespacing=1.03)
    mat.set_xticks(range(4),labels,fontsize=7)
    mat.set_yticks(range(4),labels,fontsize=7)
    mat.tick_params(length=0)
    mat.set_xlabel("Simulated target",fontsize=8,labelpad=2)
    mat.set_ylabel("Z component group",fontsize=8)
    for spine in mat.spines.values():
        spine.set_visible(False)
    fig.text(.51,.505,"Cell values: held-out R² for replicates 1 and 2.\nDarker cells indicate higher mean recovery.",va="top",size=7.2,linespacing=1.4)

    fig.add_artist(Line2D([.065,.955],[.446,.446],transform=fig.transFigure,color="#B6C1C1",lw=.65))
    fig.text(.065,.406,"C   Recovery at five and seven dimensions",size=10,weight="bold")
    fig.text(.065,.379,"All retained coordinates  /  matched fixed budgets  /  held-out simulation R²",size=8,color="#47565A")
    # Equal axes and identical replicate offsets across methods and targets.
    methods = ["HG","PCA","JIVE"]
    for panel,(target,title) in enumerate([("discordant","Discordance"),("H_private","H-specific variation")]):
        lower = fig.add_axes([.13+.49*panel,.166,.32,.145])
        lower.set_title(title,size=9,pad=5)
        lower.axvline(0,color="#A9B3B5",lw=.7)
        for row,method in enumerate(methods):
            five = pair(method+"5","all",target)
            seven = pair(method+"7","all",target)
            for r,dy in enumerate([-.10,.10]):
                lower.plot([five[r],seven[r]],[row+dy]*2,color="#A5B1B2",lw=1)
                lower.plot(five[r],row+dy,"o",mfc="white",mec=RUST,ms=4)
                lower.plot(seven[r],row+dy,"s",color=TEAL,ms=3.8)
        lower.set(ylim=(2.45,-.45),xlim=(-.075,1.05),xticks=[0,.5,1],
                  yticks=range(3),yticklabels=["Z","PCA","JIVE"])
        lower.spines["left"].set_visible(False)
        lower.tick_params(axis="y",length=0)
    fig.legend(handles=[Line2D([],[],ls="",marker="o",mfc="white",mec=RUST,label="5 scores"),
                        Line2D([],[],ls="",marker="s",color=TEAL,label="7 scores")],
               loc="center",bbox_to_anchor=(.5,.095),ncol=2,frameon=False,fontsize=8)
    fig.text(.065,.038,"Each segment pairs budgets within one replicate. Adaptive AJIVE retained variable ranks and is outside this fixed-budget panel.",size=7.2)
    export(fig,"S2_discordance_recovery.pdf","A: observed survey-weighted subgroup means at comparable S1. B: both replicate values in each group-by-target cell. C: fixed-budget recovery for discordant and H-specific targets. Exploratory two-replicate pilot.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--new-only",action="store_true",help="Render figures 4 and 5 only")
    modes.add_argument("--s2-only",action="store_true",help="Render supplementary figure S2 only")
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.new_only:
        ferritin_anchor()
        domain_contributions()
    else:
        discovery,canonical = inputs()
        if not args.s2_only:
            geometry(discovery,canonical)
            transport()
            benchmarks()
            animation(discovery,canonical)
            ferritin_anchor()
            domain_contributions()
        discordance_pilot(discovery)