#!/usr/bin/env python3
"""Summarize complete Script112 results and export aggregate transformation maps.
No research source edits and no participant-level records exported.
"""
import os,sys,json,argparse,hashlib,importlib.util,importlib.metadata
from pathlib import Path
sys.dont_write_bytecode=True
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def imp(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def mdtable(d):
    headers=list(d.columns)
    def cell(v):return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(cell(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=Path(r'D:\Hematological and Glycemic Decomposition of Depression'))
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args();root=args.repo.resolve();out=(args.out or root/'EDA/Results/112_closure').resolve()
    manifest=json.loads((out/'112_run_manifest.json').read_text())
    perf=pd.read_csv(out/'112_phenotype_performance.csv');rec=pd.read_csv(out/'112_reconstruction.csv')
    paired=pd.read_csv(out/'112_paired_PSU_comparisons.csv');invariance=pd.read_csv(out/'112_7D_OLS_invariance.csv')
    diagnostics=json.loads((out/'112_representation_diagnostics.json').read_text())
    assert len(perf)==3*3*17
    assert len(paired)==128 and len(invariance)==15
    assert invariance.max_prediction_difference_from_raw_OLS.max()<1e-7
    assert paired.resamples.eq(999).all()
    b=imp('export_source103',root/'EDA/103_jive_ajive_benchmark.py');d,cov=b.load_matched();tr=d.SPLIT.eq('train_2005-2008').to_numpy()
    scaler=StandardScaler().fit(d.loc[tr,b.X_COLS].to_numpy(float));x=scaler.transform(d[b.X_COLS].to_numpy(float));xt=x[tr]
    z=d[b.Z_COLS].to_numpy(float);z5=manifest['Z5_indices'];maps=dict(np.load(out/'112_frozen_score_maps.npz'))
    for name,score in [('RawX7',x),('PCA7',b.WeightedPCA(7).fit(xt).transform(x)),('PCA5',b.WeightedPCA(5).fit(xt).transform(x)),('FrozenZ7',z),('FrozenZ5',z[:,z5])]:
        cf,di=b.learn_projection(xt,score[tr],min_fit_r2=.99999999)
        assert np.max(abs(b.affine_predict(x,cf)-score))<1e-7
        maps[name]=cf
    np.savez(out/'113_all_representation_maps.npz',**maps)
    support=[]
    for name,cf in maps.items():
        for j in range(cf.shape[1]):
            h=np.linalg.norm(cf[1:5,j]);g=np.linalg.norm(cf[5:8,j]);norm=np.hypot(h,g)
            source='H_only' if g/max(norm,1e-30)<1e-8 else ('G_only' if h/max(norm,1e-30)<1e-8 else 'mixed')
            if name=='FrozenZ7':label=['shared1','shared2','discordant1','discordant2','H_private1','H_private2','G_private1'][j]
            elif name.startswith(('AJIVE','JIVE')):label='joint' if j<2 else ('H_individual' if j<(4 if cf.shape[1]==5 else 5) else 'G_individual')
            else:label='prespecified_or_unordered'
            support.append(dict(method=name,coordinate=j+1,label=label,H_loading_norm=h,G_loading_norm=g,raw_domain_support=source,qualification='algebraic loading support; not biological independence or method superiority'))
    pd.DataFrame(support).to_csv(out/'113_coordinate_domain_support.csv',index=False)
    covnames=sys.modules['s98_for_103'].covariates('full')
    parameters=json.loads((out/'112_frozen_outcome_parameters.json').read_text())
    for key,entry in parameters.items():
        n=len(entry['coef'])-len(covnames)
        entry['feature_names']=covnames+[f'physiological_coordinate_{j+1}' for j in range(n)]
    (out/'112_frozen_outcome_parameters.json').write_text(json.dumps(parameters,indent=2))
    versions={p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn')}
    jive_path=Path(r'C:\Users\auniy\anaconda3\envs\physiodecomp\Lib\site-packages\jive\AJIVE.py')
    versions['AJIVE_source_sha256']=hashlib.sha256(jive_path.read_bytes()).hexdigest()
    versions['R_JIVE']='2.4; original official implementation; see native execution log'
    (out/'113_transform_metadata.json').write_text(json.dumps(dict(raw_feature_order=b.X_COLS,raw_training_mean=scaler.mean_.tolist(),raw_training_scale=scaler.scale_.tolist(),covariate_order=covnames,convention='score = [1, discovery-standardized raw X] @ affine_map',versions=versions),indent=2))
    r5=paired[(paired.family=='rank5')&(paired.analysis=='prediction')]
    r7=paired[(paired.family=='rank7')&(paired.analysis=='prediction')]
    triad=paired[(paired.family=='triad')&(paired.method.isin(['H4','G3','RawX7','FrozenZ7','Threshold2']))]
    score=pd.read_csv(out/'112_advantage_scorecard.csv')
    concise=['outcome','period','method','reference','paired_gain_positive_better','approx_PSU_low','approx_PSU_high','direction']
    report=['# Script work closure — 5 November 2026','',
      '## Decision','',
      'The scripts now support specific evidence-based comparisons. They do not require the H/G representation to win. Read each claim against its named endpoint, reference method, dimensional budget and era. All later eras have been inspected previously; the results remain exploratory.','',
      '## Corrected AJIVE comparator','',
      'The old `reconsider_joint_components=False` path retains all initial common-space vectors in downstream block projections while exposing only the requested common vectors in the reported scores. On these discovery data, the resulting relative projection errors are about 0.719 (H) and 0.635 (G), and one requested H-individual singular value is about 2e-13. This numerically null score is not a valid stable physiological axis.','',
      'Production Script112 uses the original package with its default joint-component check enabled and exact dense SVD as the numerical backend. No installed package or original research script was edited. The retained-projector identity, nonzero extracted singular values, score-adapter recovery and repeated-fit subspaces are explicitly checked. Old AJIVE performance and rediscovery comparisons using the faulty setting need replacement; do not interpret their weakness as a limitation of AJIVE in general. Existing RawX/PCA/H/G and primary survey comparisons do not use that setting.','',
      '## Dimensional budgets','',
      '5D compares the same fixed five-score budget. 7D compares explicitly specified full-budget maps. The 7D AJIVE/JIVE settings are not automatically selected denoised signal ranks and should be labelled full-budget sensitivity analyses. It is inappropriate to claim superiority for a seven-variable invertible map merely because it retains more than a five-score compression.','',
      'Future JIVE/AJIVE scores use discovery-trained affine adapters, not each package\'s native prediction routine. These corrected fits have exact discovery-score recovery. Reconstruction uses a discovery-trained free linear decoder over scores, rather than each package\'s own joint-plus-individual reconstructed matrix. The original frozen H/G map retains its historical discovery fit; identical matched participants apply to downstream fitting/evaluation, not to a new identical-row unsupervised refit of every representation.','',
      f'The maximum difference between weighted OLS predictions from the five full-rank coordinate systems is {invariance.max_prediction_difference_from_raw_OLS.max():.3g}. This verifies equal linear information. Ridge differences can reflect coordinate scaling and penalty geometry rather than newly discovered information.','',
      '### Reconstruction','',mdtable(rec[rec.period.ne('train_2005-2008')]),'',
      '### Rank-5 phenotype comparisons','',
      'Positive gain is lower MSE than PCA5. All outcomes and both later eras are shown.', '',mdtable(r5[concise]),'',
      '### Rank-7 phenotype comparisons','',
      'Positive gain is lower MSE than RawX7. These are regularized-pipeline comparisons with equivalent input information.', '',mdtable(r7[concise]),'',
      '## A-only, G-only and A+G versus depressive phenotypes','',
      'H4 is the four-marker haematological block; G3 is the three-marker glycaemic block. Every model includes the same covariates and uses identical participants. PHQ total, somatic and cognitive-affective outcomes are symptom scores, not diagnoses or cognitive-test performance. Positive gains do not identify individual biological causes.', '',mdtable(triad[concise]),'',
      '## What counts as an advantage','',
      '- The previously reproduced survey result supports modest information beyond threshold scalars. Continuous raw markers can retain that information too; it is not proof of a uniquely superior decomposition.',
      '- Full-rank fidelity and explicitly named shared/discordant/domain-specific coordinates are construction properties. `113_coordinate_domain_support.csv` tests their algebraic loading support, not biological validity. Raw markers themselves also have domain-specific support.',
      '- A predictive advantage must survive the specific paired comparisons above. A mixture of wins, losses and inconclusive intervals does not establish general superiority.',
      '- Previous independent-refit/ferritin analyses provide exploratory structural and biological-anchor evidence. They were not unified into this matched prediction benchmark, and no new external cohort is introduced here.','',
      '## Uncertainty and reproducibility','',
      '999 paired Rao-Wu-style rescaled PSU replicates use the complete original DEMO design frame within cycle and stratum. Cluster-aggregated bootstrap calculations were checked against direct participant-weight calculations. Intervals condition on fitted models; they omit training/model-selection uncertainty, do not correct multiplicity, and do not undo repeated inspection of later outcomes. They are approximate sensitivity intervals, not universal design-based coverage guarantees.','',
      'The frozen protocol, model-selection table, source hashes, aggregate affine transformation maps and downstream model parameters are saved. No participant-level final records are exported. Check source hashes before reusing cached native JIVE fits.','',
      '## Paper implications','',
      'Present a transparent physiological decomposition and validation study with a complete A–D, G–D and combined evidence map. Show where threshold summaries discard information, where the full-coordinate maps tie, which compressed versions lose information, and how outcome transfer changes. Treat any domain-interpretation benefit as a specific design objective until independent anchors validate it. Independent statistical review and a genuinely independent replication remain separate requirements for stronger claims.','',
      '## Aggregate scorecard','',mdtable(score),'']
    (out/'113_READOUT.md').write_text('\n'.join(report),encoding='utf-8')
    print(score.to_string(index=False))
    print('Complete: validated result counts; exported aggregate maps and outcome parameters; wrote 113_READOUT.md.')
if __name__=='__main__':main()
