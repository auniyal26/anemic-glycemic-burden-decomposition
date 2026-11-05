#!/usr/bin/env python3
"""Evidence audit for a JBI methodological manuscript.

Reads unchanged public-repository result CSVs. Makes no new subject-level or
survey-design estimates. Keeps distinct source populations/estimands separate.
From the repository root, run: python EDA/110_evidence_gate.py
The script automatically discovers original source results under EDA/Results.
You may instead pass --repo PATH or --source-snapshot PATH explicitly.
No participant-level data or refitting is needed for this evidence audit.
"""
import argparse
import csv
from pathlib import Path

EXPECTED_N={'train_2005-2008':3815,'test_2009-2018':9893,'validate_2021-2023':2247}
PERIODS=['test_2009-2018','validate_2021-2023']
BASE='EDA/Results'
INPUTS={
 'reconstruction':'103_jive_ajive_benchmark/103_reconstruction.csv',
 'paired':'103_jive_ajive_benchmark/103_paired_bootstrap.csv',
 'null':'105_specificity_and_transfer_audit/105_conditional_null.csv',
 'design':'106_survey_ferritin_anchor/106_design_rediscovery.csv',
 'utility':'109_clinical_utility_benchmark/109_application_criteria.csv',
}

def read_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def num(r,key):
    return float(r[key])

def write_csv(path,rows):
    if not rows: raise ValueError('No rows: '+str(path))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def resolve_source_dir(repo=None, source_snapshot=None):
    """Locate the original CSV outputs, without requiring a bundled snapshot."""
    script = Path(__file__).resolve()
    candidates = []
    if repo is not None:
        repo = repo.expanduser().resolve()
        candidates.extend([repo / BASE, repo])
    if source_snapshot is not None:
        candidates.append(source_snapshot.expanduser().resolve())
    # Script placed in the original repository's EDA/ directory.
    candidates.extend([
        script.parent / 'Results',
        script.parent.parent / BASE,
        Path.cwd() / BASE,
        script.parent.parent / 'evidence' / 'source_snapshot',
    ])
    seen = set()
    attempts = []
    for directory in candidates:
        directory = directory.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        missing = [rel for rel in INPUTS.values() if not (directory / rel).is_file()]
        if not missing:
            return directory
        attempts.append((directory, missing))
    details = '\n'.join(
        f"  {directory}\n" + '\n'.join(f"    MISSING: {rel}" for rel in missing)
        for directory, missing in attempts
    )
    raise FileNotFoundError(
        'Could not locate all five source CSV outputs.\n'
        'Run Scripts 103, 105, 106 and 109 if their result files are absent.\n'
        'Or supply --repo with the path to the original repository.\n'
        f'Checked:\n{details}'
    )


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--repo', type=Path, help='Original research repository root (optional; autodetected).')
    a.add_argument('--source-snapshot', type=Path, help='Alternate base folder holding results (optional).')
    a.add_argument('--out', type=Path, help='Folder for generated evidence-audit outputs.')
    args=a.parse_args()
    base = resolve_source_dir(repo=args.repo, source_snapshot=args.source_snapshot)
    out = args.out or (base / '110_evidence_gate' if base.name == 'Results' else base.parent / 'generated')
    args.out = out
    print(f'Source CSVs: {base}')
    print(f'Audit output: {args.out}')
    data={k:read_csv(base/v) for k,v in INPUTS.items()}
    args.out.mkdir(parents=True,exist_ok=True)

    reconstruction=data['reconstruction']
    assert len(reconstruction)>=21, 'Missing 103 benchmark records'
    expected_5={'FrozenZ5_trainX','PCA5','JIVE5_rjive','AJIVE5_pyjive'}
    recon=[]
    for period in PERIODS:
        vals={r['method']:r for r in reconstruction if r['period']==period and int(r['dimension'])==5}
        assert expected_5<=set(vals), (period,'missing matched-rank methods')
        for method in sorted(expected_5):
            r=vals[method]
            assert int(r['n'])==EXPECTED_N[period], f'Unequal cohort in {period}, {method}'
            recon.append({'period':period,'method':method,'rank':5,'n':r['n'],
                'reconstruction_R2':r['reconstruction_R2'],
                'delta_R2_vs_PCA5':f"{num(r,'reconstruction_R2')-num(vals['PCA5'],'reconstruction_R2'):.8f}",
                'design_aware_CI':'NOT AVAILABLE IN SCRIPT 103'})
    write_csv(args.out/'110_common_sample_rank5_reconstruction.csv',recon)

    phen=[]
    for r in data['paired']:
        if r['period'] not in PERIODS or r['reference']!='PCA5':continue
        low=num(r,'descriptive_CI_low');hi=num(r,'descriptive_CI_high')
        d=num(r,'delta_MSE_positive_better')
        if low>0: conclusion='FAVOURS_METHOD_DESCRIPTIVE_ONLY'
        elif hi<0: conclusion='FAVOURS_PCA5_DESCRIPTIVE_ONLY'
        else: conclusion='INCONCLUSIVE_DESCRIPTIVE_ONLY'
        phen.append({'outcome':r['outcome'],'period':r['period'],'method':r['method'],
                     'comparator':'PCA5','delta_MSE_positive_favours_method':r['delta_MSE_positive_better'],
                     'case_bootstrap_low':r['descriptive_CI_low'],'case_bootstrap_high':r['descriptive_CI_high'],
                     'interpretation':conclusion,'survey_design_corrected':False})
    assert len(phen)==18,'Not all 3 outcomes x 3 methods x 2 later periods in 103'
    write_csv(args.out/'110_common_sample_rank5_phenotype.csv',phen)

    spec=[]
    for r in data['null']:
        margin=num(r,'specificity_margin_vs_null95')
        p=num(r,'exploratory_permutation_tail_p')
        spec.append({'cohort':r['cohort'],'group':r['group'],
             'observed_similarity':r['observed_similarity'],
             'null_95th_percentile':r['conditioned_null_high95'],
             'excess_similarity':r['specificity_margin_vs_null95'],
             'permutation_p_exploratory':r['exploratory_permutation_tail_p'],
             'descriptive_flag':'ABOVE_CONDITIONAL_NULL' if margin>0 and p<=0.05 else 'NOT_CLEARLY_ABOVE_NULL',
             'distinct_analysis':'105: sampled independent refits; NOT 103 matched-rank comparison'})
    write_csv(args.out/'110_conditional_null_specificity.csv',spec)

    survey=[]
    for r in data['design']:
        survey.append({'reference':r['reference'],'target':r['target'],'group':r['group'],
         'similarity':r['observed_similarity'],'approx_PSU_bootstrap_low':r['psu_rescaled_low_2p5'],
         'approx_PSU_bootstrap_high':r['psu_rescaled_high_97p5'],
         'interpretation':'Approximate survey-aware structural uncertainty; not 103 common sample'})
    write_csv(args.out/'110_survey_structural_sensitivity.csv',survey)

    utility=[]
    for r in data['utility']:
        d_auc=num(r,'delta_auc_z_minus_raw');d_brier=num(r,'delta_brier_raw_minus_z')
        utility.append({'period':r['period'],'Z_minus_raw_AUROC':r['delta_auc_z_minus_raw'],
             'raw_minus_Z_Brier':r['delta_brier_raw_minus_z'],
             'Z_gate_minus_best_simple':r['gate_advantage'],
             'both_discrimination_and_calibration_better':d_auc>0 and d_brier>0,
             'cohort_qualification':'109: same Script98 cohort, but distinct endpoint/model protocol'})
    write_csv(args.out/'110_clinical_utility.csv',utility)

    z_rows=[r for r in recon if r['method']=='FrozenZ5_trainX']
    z_phq=[r for r in phen if r['method']=='FrozenZ5_trainX' and r['outcome']=='PHQ_total']
    claims=[
      {'criterion':'Information preservation','requirement':'Dimension-matched R2 >= PCA5 in both later periods','observed':'FAIL' if all(float(r['delta_R2_vs_PCA5'])<0 for r in z_rows) else 'MIXED','status':'NOT SUPPORTED','scope':'103 common sample'},
      {'criterion':'PHQ prediction advantage','requirement':'Paired delta MSE CI lower > 0 in BOTH later periods, survey-design-aware','observed':'FAIL' if all(r['interpretation']=='FAVOURS_PCA5_DESCRIPTIVE_ONLY' for r in z_phq) else 'MIXED','status':'NOT SUPPORTED','scope':'103 common sample; intervals are case bootstrap only'},
      {'criterion':'Statistical subspace specificity','requirement':'Observed rediscovery exceeds prespecified conditioned null; multiplicity and survey design accounted for','observed':'PARTIAL','status':'EXPLORATORY SUPPORT ONLY','scope':'105 and 106 distinct refits; not pooled with 103'},
      {'criterion':'External physiological anchoring','requirement':'Independent, harmonized assay and population with comparable full panel','observed':'NOT MET','status':'NOT ESTABLISHED','scope':'106 ferritin is within NHANES'},
      {'criterion':'Clinical utility beyond raw biomarkers','requirement':'Positive paired AUROC and Brier in both periods with design-aware uncertainty','observed':'FAIL' if not all(r['both_discrimination_and_calibration_better'] for r in utility) else 'PASS','status':'NOT SUPPORTED','scope':'109 observational screening retrospective'},
      {'criterion':'Confirmatory transport','requirement':'Untouched external dataset, stable measurement and independent hypothesis test','observed':'NOT MET','status':'NOT ESTABLISHED','scope':'All available NHANES periods were previously inspected'}]
    write_csv(args.out/'110_claim_decision_ledger.csv',claims)
    first=next(r for r in z_rows if r['period']==PERIODS[0]);last=next(r for r in z_rows if r['period']==PERIODS[1]);
    body=f'''# Script 110: Evidence audit (NOT a new full benchmark)

*Source:* repository Scripts 103, 105, 106 and 109. No subject-level rows were accessed or fitted here. Scripts 105 and 106 have distinct analytic samples; their estimates are **not pooled** with Script 103.

## Same-sample 5D comparisons, Script 103

- 2009–2018: FrozenZ5 R²={float(first['reconstruction_R2']):.4f}, difference vs PCA5={float(first['delta_R2_vs_PCA5']):+.4f}.
- 2021–2023: FrozenZ5 R²={float(last['reconstruction_R2']):.4f}, difference vs PCA5={float(last['delta_R2_vs_PCA5']):+.4f}.
- All six FrozenZ5 vs PCA5 outcome-period MSE differences favour PCA5 in descriptive participant bootstraps.
- JIVE5 and PCA5 are essentially tied at this rank; native JIVE/AJIVE out-of-period scores use **training-only affine adapters**, not native out-of-sample transforms.

## Specificity and uncertainty, supplemental tests

- Script 105: shared/discordant subspaces exceed age/sex-conditioned permutation nulls in tested eras; 2021–2023 glycaemic-isolated component is close to its conditional null and does not pass nominal p<0.05.
- Script 106: rescaled-PSU bootstrap gives approximate survey-aware structural uncertainty. These are **different sample definitions** from the Script 103 matched-rank benchmark.
- Script 109: no consistent improvement over raw biomarkers in screening discrimination and Brier score.

**Result:** Statistical reproducibility is plausible for specific subspaces; reconstruction or predictive superiority is not established. Full unified **survey-design-based, rank-matched, same-sample comparison remains unrun**.
'''
    (args.out/'110_readout.md').write_text(body,encoding='utf-8')
    print(body)

if __name__=='__main__': main()
