#!/usr/bin/env python3
"""
Script 126 v2 — cleaner publication-grade figures for Script 125.

Outputs:
- 126_main_adversarial_validation_v2.(png|pdf)
- 126_supplementary_adversarial_controls_v2.(png|pdf)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({
    'font.family':'DejaVu Sans',
    'font.size':10,
    'axes.titlesize':12,
    'axes.labelsize':10,
    'xtick.labelsize':9,
    'ytick.labelsize':9,
    'figure.facecolor':'white',
    'axes.facecolor':'white',
    'axes.spines.top':False,
    'axes.spines.right':False,
})

SCENARIO_ORDER = ['asymmetric_mixing','nonlinear_coupling','correlated_factors','heteroskedastic','contamination']
SCENARIO_LABELS = {
    'asymmetric_mixing':'Asym.\nmixing',
    'nonlinear_coupling':'Nonlinear\ncoupling',
    'correlated_factors':'Correlated\nfactors',
    'heteroskedastic':'Heteroskedastic\nnoise',
    'contamination':'Outliers /\ncontamination',
}
GROUP_ORDER = ['shared','discordant','H_private','G_private']
GROUP_LABELS = {'shared':'Shared','discordant':'Discordant','H_private':'H-private','G_private':'G-private'}
METHOD_LABELS = {
    'FiGuRO_fixed':'FiGuRO fixed',
    'FiGuRO_adaptive':'FiGuRO adaptive',
    'DCCA':'D-CCA',
    'AJIVE7_adaptive':'AJIVE',
    'PCA7':'PCA',
    'White7':'Whitened Raw7',
    'Raw7':'Raw7',
    'HG7':'HG7'
}
CORE_METHODS = ['FiGuRO_fixed','FiGuRO_adaptive','DCCA','AJIVE7_adaptive','PCA7']
COUNT_METHODS = ['FiGuRO_fixed','FiGuRO_adaptive','DCCA','AJIVE7_adaptive','PCA7','White7','Raw7']
CYCLE_LABELS = {708:'2007–2008',2123:'2021–2023'}
COLORS = {
    'FiGuRO_fixed':'#4C78A8',
    'FiGuRO_adaptive':'#72B7B2',
    'DCCA':'#9D7660',
    'AJIVE7_adaptive':'#ECA82C',
    'PCA7':'#8A60BF',
    'White7':'#59A14F',
    'Raw7':'#E15759',
    'HG7':'#1F3A5F',
}

def find_results_dir(cli=None):
    if cli:
        p=Path(cli).expanduser().resolve();
        if not p.exists(): raise FileNotFoundError(p)
        return p
    here=Path(__file__).resolve()
    candidates=[here.parent/'Results'/'125_adversarial_validation', Path.cwd()/'EDA'/'Results'/'125_adversarial_validation', Path.cwd()]
    for c in candidates:
        if (c/'125_sim_paired.csv').is_file(): return c.resolve()
    raise FileNotFoundError('Could not locate Script 125 outputs. Pass --results-dir.')

def infer_output_dir(results_dir, cli=None):
    if cli: out=Path(cli).expanduser().resolve()
    else:
        out=results_dir.parent/'126_adversarial_figures_v2' if results_dir.name=='125_adversarial_validation' else results_dir/'126_adversarial_figures_v2'
    out.mkdir(parents=True, exist_ok=True)
    return out

def rcsv(results_dir,name):
    p=results_dir/name
    if not p.is_file(): raise FileNotFoundError(p)
    return pd.read_csv(p)

def sig(p):
    if pd.isna(p): return ''
    return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''

def label_panel(ax, lab, x=-0.14, y=1.08):
    ax.text(x,y,lab, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')

def save(fig, outbase):
    fig.savefig(f'{outbase}.png', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{outbase}.pdf', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def make_main(results_dir, outdir):
    sim=rcsv(results_dir,'125_sim_paired.csv')
    sim = sim.copy()
    sim = sim[sim['method'].isin(COUNT_METHODS)]
    # counts
    counts=[]
    for m,d in sim.groupby('method'):
        favourable=((d['delta_R2']>0)&(d['p_signed_rank_holm']<0.05)).sum()
        unfavourable=((d['delta_R2']<0)&(d['p_signed_rank_holm']<0.05)).sum()
        inconclusive=len(d)-favourable-unfavourable
        counts.append({'method':m,'favourable':favourable,'inconclusive':inconclusive,'unfavourable':unfavourable})
    counts=pd.DataFrame(counts)
    counts['method']=pd.Categorical(counts['method'],COUNT_METHODS,ordered=True)
    counts=counts.sort_values('method')

    # forest shared
    forest=sim[(sim['group']=='shared') & (sim['method'].isin(CORE_METHODS))].copy()
    forest['scenario']=pd.Categorical(forest['scenario'], SCENARIO_ORDER, ordered=True)
    forest=forest.sort_values(['scenario','method'])

    fig=plt.figure(figsize=(14.5,11.5), facecolor='white')
    outer=gridspec.GridSpec(2,2, figure=fig, width_ratios=[1.35,1.0], height_ratios=[1.45,0.95], hspace=0.34, wspace=0.26)
    top=gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=outer[0,:], wspace=0.15, hspace=0.22)

    # group heatmaps
    heat_df=sim[sim['method'].isin(CORE_METHODS) & sim['kind'].isin(['native','native_truncated'])].copy()
    # color range consistent
    vmax=max(0.05, np.nanmax(np.abs(heat_df['delta_R2'].to_numpy())))
    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap='coolwarm'

    hm_axes=[]
    for idx, group in enumerate(GROUP_ORDER):
        ax=fig.add_subplot(top[idx//2, idx%2])
        hm_axes.append(ax)
        sub=heat_df[heat_df['group']==group].copy()
        row_methods=[m for m in CORE_METHODS if m in sub['method'].unique()]
        mat=sub.pivot(index='method', columns='scenario', values='delta_R2').reindex(index=row_methods, columns=SCENARIO_ORDER)
        pmat=sub.pivot(index='method', columns='scenario', values='p_signed_rank_holm').reindex(index=row_methods, columns=SCENARIO_ORDER)
        arr=mat.to_numpy(dtype=float)
        ax.set_title(GROUP_LABELS[group], pad=8, fontweight='bold')
        if len(row_methods)==0:
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5,0.52,'No native comparator\ndiscordance group', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='#555555')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color('#333333')
            continue
        im=ax.imshow(arr, aspect='auto', cmap=cmap, norm=norm)
        ax.set_xticks(np.arange(len(SCENARIO_ORDER)))
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
        ax.set_yticks(np.arange(len(row_methods)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in row_methods])
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isnan(arr[i,j]):
                    continue
                text=f'{arr[i,j]:+.2f}{sig(pmat.iloc[i,j])}'
                color='white' if abs(arr[i,j])>0.23*vmax else 'black'
                ax.text(j,i,text, ha='center', va='center', fontsize=8, color=color)
        ax.set_xlim(-0.5, len(SCENARIO_ORDER)-0.5)
        ax.set_ylim(len(row_methods)-0.5,-0.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('#333333')
    cbar=fig.colorbar(im, ax=hm_axes, location='right', shrink=0.94, pad=0.015)
    cbar.set_label('HG7 improvement in held-out $R^2$')
    label_panel(hm_axes[0],'A',x=-0.18,y=1.16)

    # forest plot
    axB=fig.add_subplot(outer[1,0])
    y_base=np.arange(len(SCENARIO_ORDER))[::-1]
    offsets=np.linspace(-0.24,0.24,len(CORE_METHODS))
    for off,m in zip(offsets, CORE_METHODS):
        d=forest[forest['method']==m].set_index('scenario').reindex(SCENARIO_ORDER).reset_index()
        x=d['delta_R2'].to_numpy(); lo=d['lo'].to_numpy(); hi=d['hi'].to_numpy()
        y=y_base+off
        axB.errorbar(x, y, xerr=[x-lo, hi-x], fmt='o', ms=5.5, lw=1.6, capsize=2.2, color=COLORS[m], label=METHOD_LABELS[m])
    axB.axvline(0, color='black', lw=1, ls='--')
    axB.set_yticks(y_base)
    axB.set_yticklabels([SCENARIO_LABELS[s].replace('\n',' ') for s in SCENARIO_ORDER])
    axB.set_xlabel('HG7 improvement in shared-target held-out $R^2$')
    axB.set_title('Shared-target recovery effect sizes', pad=6)
    axB.legend(frameon=False, ncol=2, fontsize=8.5, loc='lower right')
    axB.grid(axis='x', alpha=0.18)
    label_panel(axB,'B')

    # counts
    axC=fig.add_subplot(outer[1,1])
    y=np.arange(len(counts))
    left=np.zeros(len(counts))
    palette={'favourable':'#2A7F62','inconclusive':'#D9D9D9','unfavourable':'#B4466E'}
    display={'favourable':'Holm-significant favourable','inconclusive':'Inconclusive','unfavourable':'Holm-significant unfavourable'}
    for key in ['favourable','inconclusive','unfavourable']:
        vals=counts[key].to_numpy()
        axC.barh(y, vals, left=left, color=palette[key], height=0.72, edgecolor='white', label=display[key])
        for yi,l,v in zip(y,left,vals):
            if v>0:
                axC.text(l+v/2, yi, f'{int(v)}', ha='center', va='center', fontsize=8)
        left += vals
    axC.set_yticks(y)
    axC.set_yticklabels([METHOD_LABELS[m] for m in counts['method']])
    axC.invert_yaxis()
    axC.set_xlabel('Number of paired comparisons')
    axC.set_title('Overall adversarial comparison counts', pad=6)
    axC.legend(frameon=False, fontsize=8, loc='lower right')
    axC.grid(axis='x', alpha=0.16)
    label_panel(axC,'C')

    fig.suptitle('Main Figure — Adversarial comparative validation of HG7', y=0.985, fontsize=17, fontweight='bold')
    fig.text(0.055, 0.948, 'Panel A shows designated coordinate groups only (native or deterministically truncated native groups).', fontsize=10)
    fig.text(0.02, 0.012, 'Positive values favour HG7. Asterisks denote Holm-adjusted significance. Shared-group forest plots show paired bootstrap intervals.', fontsize=9.2)
    save(fig, outdir/'126_main_adversarial_validation_v2')

def make_supp(results_dir, outdir):
    assay=rcsv(results_dir,'125_assay_paired.csv')
    random=rcsv(results_dir,'125_random_percentiles.csv')
    dims=rcsv(results_dir,'125_sim_dimensions.csv')
    num=rcsv(results_dir,'125_numerical.csv')
    fail=rcsv(results_dir,'125_failures.csv')

    # selected assay contrasts
    selected=[('Ferritin',708,'shared'),('Ferritin',2123,'shared'),('sTfR',708,'discordant'),('sTfR',2123,'discordant'),('OGTT',708,'G_private')]
    sset=set(selected)
    assay=assay[assay.apply(lambda r: (r['anchor'], int(r['cycle']), r['group']) in sset, axis=1)].copy()
    assay=assay[assay['comparator'].isin(['FiGuRO_fixed','FiGuRO_adaptive','DCCA','AJIVE7_adaptive','PCA7','White7','Raw7'])]
    order_labels=[f'{a} {CYCLE_LABELS[c]} — {GROUP_LABELS[g]}' for a,c,g in selected]
    assay['row_label']=assay.apply(lambda r: f"{r['anchor']} {CYCLE_LABELS[int(r['cycle'])]} — {GROUP_LABELS[r['group']]}", axis=1)

    # selected random percentile summary
    keep_random=[]
    for a,c,g in [('Ferritin',708,'shared'),('Ferritin',2123,'shared'),('sTfR',708,'discordant'),('sTfR',2123,'discordant'),('OGTT',708,'G_private'),('CRP',2123,'shared')]:
        sub=random[(random['anchor']==a)&(random['cycle']==c)&(random['group']==g)]
        if not sub.empty: keep_random.append((a,c,g))
    random=random[random.apply(lambda r: (r['anchor'], int(r['cycle']), r['group']) in set(keep_random), axis=1)].copy()
    random['label']=random.apply(lambda r: f"{r['anchor']} {CYCLE_LABELS[int(r['cycle'])]} — {GROUP_LABELS[r['group']]}", axis=1)
    rand_labels=[f'{a} {CYCLE_LABELS[c]} — {GROUP_LABELS[g]}' for a,c,g in keep_random]
    random['dim_label']=random['dim'].astype(int).astype(str)+'D'

    # dimension eligibility proportions
    methods=['HG7','PCA7','White7','Raw7','DCCA','AJIVE7_adaptive','FiGuRO_fixed','FiGuRO_adaptive']
    dims=dims[dims['method'].isin(methods)].copy()
    prop=(dims.groupby(['method','total_dim']).size().rename('n').reset_index())
    total_n=prop.groupby('method')['n'].transform('sum')
    prop['prop']=prop['n']/total_n
    total_dims=sorted(prop['total_dim'].unique())

    num=num.copy(); num['method']=pd.Categorical(num['method'], ['Raw7','White7','PCA7','HG7'], ordered=True); num=num.sort_values('method')

    fig=plt.figure(figsize=(15,12), facecolor='white')
    outer=gridspec.GridSpec(3,2, figure=fig, height_ratios=[1.15,0.85,0.9], width_ratios=[1.25,1.0], hspace=0.35, wspace=0.28)

    # A forest full width top
    axA=fig.add_subplot(outer[0,:])
    comps=['FiGuRO_fixed','FiGuRO_adaptive','DCCA','AJIVE7_adaptive','PCA7','White7','Raw7']
    y_base=np.arange(len(order_labels))[::-1]
    offsets=np.linspace(-0.30,0.30,len(comps))
    for off,m in zip(offsets, comps):
        d=assay[assay['comparator']==m].set_index('row_label').reindex(order_labels).reset_index()
        x=d['delta_MSE'].to_numpy(); lo=d['lo'].to_numpy(); hi=d['hi'].to_numpy(); y=y_base+off
        mask=~np.isnan(x)
        axA.errorbar(x[mask], y[mask], xerr=[x[mask]-lo[mask], hi[mask]-x[mask]], fmt='o', ms=4.7, lw=1.35, capsize=2, color=COLORS[m], label=METHOD_LABELS[m])
    axA.axvline(0,color='black',lw=1,ls='--')
    axA.set_yticks(y_base)
    axA.set_yticklabels(order_labels)
    axA.set_xlabel('Comparator MSE minus HG7 MSE')
    axA.set_title('Selected biological prediction contrasts', pad=7)
    axA.legend(frameon=False, ncol=4, fontsize=8.3, loc='lower right')
    axA.grid(axis='x', alpha=0.18)
    label_panel(axA,'A')

    # B percentile dot plot better than crowded heatmap
    axB=fig.add_subplot(outer[1,0])
    y=np.arange(len(rand_labels))[::-1]
    for dlabel, marker in [('1D','o'),('2D','s')]:
        sub=random[random['dim_label']==dlabel].set_index('label').reindex(rand_labels).reset_index()
        x=sub['HG_percentile_better'].to_numpy()
        mask=~np.isnan(x)
        axB.scatter(x[mask], y[mask], s=56 if dlabel=='2D' else 42, marker=marker, color='#1F3A5F' if dlabel=='2D' else '#72B7B2', label=dlabel, zorder=3)
        for xv,yv in zip(x[mask], y[mask]):
            axB.plot([0,xv],[yv,yv], color='#D5D5D5', lw=1, zorder=1)
    axB.axvline(50, color='black', lw=1, ls='--')
    axB.set_yticks(y)
    axB.set_yticklabels(rand_labels)
    axB.set_xlim(0,100)
    axB.set_xlabel('HG7 percentile versus matched-dimensional random subspaces')
    axB.set_title('Random-subspace control percentile ranking', pad=6)
    axB.legend(frameon=False, title='Group size', fontsize=8.5)
    axB.grid(axis='x', alpha=0.15)
    label_panel(axB,'B')

    # C dimension eligibility
    axC=fig.add_subplot(outer[1,1])
    yy=np.arange(len(methods)); left=np.zeros(len(methods))
    palette=['#E8EEF5','#C7D7EA','#9ABBDC','#6B9FD0','#3E82C4','#2B5D9A','#183E69','#0B2344']
    dim_color={d:palette[i] for i,d in enumerate(total_dims)}
    for d in total_dims:
        vals=[]
        for m in methods:
            row=prop[(prop['method']==m)&(prop['total_dim']==d)]
            vals.append(0.0 if row.empty else float(row['prop'].iloc[0]))
        vals=np.asarray(vals)
        axC.barh(yy, vals, left=left, height=0.72, color=dim_color[d], edgecolor='white', label=f'{d} dimensions')
        left+=vals
    axC.set_yticks(yy)
    axC.set_yticklabels([METHOD_LABELS[m] for m in methods])
    axC.invert_yaxis(); axC.set_xlim(0,1)
    axC.set_xlabel('Proportion of simulation fits')
    axC.set_title('Dimensional eligibility across simulation fits', pad=6)
    axC.legend(frameon=False, fontsize=8, ncol=2, loc='lower right')
    axC.grid(axis='x', alpha=0.14)
    label_panel(axC,'C')

    # D numerical controls full width bottom
    axD=fig.add_subplot(outer[2,:])
    x=np.arange(len(num)); width=0.22
    axD.bar(x-width/2, num['max_VIF'], width=width, color='#4C78A8', label='Maximum VIF')
    axD.bar(x+width/2, num['condition_number'], width=width, color='#72B7B2', label='Condition number')
    axD.set_xticks(x)
    axD.set_xticklabels([METHOD_LABELS[m] for m in num['method']])
    axD.set_ylabel('Value')
    axD.set_title('Numerical controls', pad=6)
    axD.legend(frameon=False, ncol=2, loc='upper right')
    axD.grid(axis='y', alpha=0.14)
    for xi,v,c in zip(x, num['max_VIF'], num['condition_number']):
        axD.text(xi-width/2, v+0.2, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        axD.text(xi+width/2, c+0.12, f'{c:.2f}', ha='center', va='bottom', fontsize=8)
    if not fail.empty:
        msg='; '.join(f"{METHOD_LABELS.get(r.method,r.method)} not evaluated: {r.reason}" for r in fail.itertuples(index=False))
        axD.text(0.0,-0.27,msg, transform=axD.transAxes, fontsize=8.6, ha='left', va='top')
    label_panel(axD,'D')

    fig.suptitle('Supplementary Figure — Comparative specificity and control analyses', y=0.985, fontsize=17, fontweight='bold')
    fig.text(0.02, 0.012, 'Positive values in Panel A favour HG7. Panel B shows HG7 percentile against 199 random matched-dimensional subspaces. Panel C summarises retained total dimensionality. Panel D confirms that orthogonalised controls explain conditioning improvements but not HG7 structure.', fontsize=9.2)
    save(fig, outdir/'126_supplementary_adversarial_controls_v2')

def write_manifest(outdir, results_dir):
    manifest={
        'script':Path(__file__).name,
        'input_dir':str(results_dir),
        'outputs':['126_main_adversarial_validation_v2.png','126_main_adversarial_validation_v2.pdf','126_supplementary_adversarial_controls_v2.png','126_supplementary_adversarial_controls_v2.pdf'],
        'notes':['v2 layout uses four separate group heatmaps in the main figure and replaces the crowded random-percentile heatmap with a cleaner percentile dot plot in the supplementary figure.']
    }
    (outdir/'126_manifest_v2.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results-dir', default=None)
    ap.add_argument('--output-dir', default=None)
    args=ap.parse_args()
    results_dir=find_results_dir(args.results_dir)
    outdir=infer_output_dir(results_dir,args.output_dir)
    make_main(results_dir,outdir)
    make_supp(results_dir,outdir)
    write_manifest(outdir,results_dir)
    print(f'Saved revised figures to: {outdir}')

if __name__=='__main__':
    main()
