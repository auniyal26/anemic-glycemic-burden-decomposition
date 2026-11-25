# Script 118 — Biomedical utility benchmark

## Primary scientific question

Does the HG shared pair retain more accessible ferritin-associated information than alternative representations under a matched 2D budget?

## Direct comparisons

```text
 candidate      reference                         verdict  delta_0708  low95_0708  high95_0708  delta_2123  low95_2123  high95_2123
HG_Shared2     Threshold2                    inconclusive    -0.03032    -0.08952      0.01985     0.00051    -0.05229      0.04523
HG_Shared2       Hb_HbA1c                    inconclusive    -0.08494    -0.18467      0.00206     0.00160    -0.04746      0.04967
HG_Shared2         Hb_RDW    period_specific_HG_advantage    -0.14264    -0.25129     -0.04743     0.18852     0.06746      0.32494
HG_Shared2           PCA2 period_specific_HG_disadvantage    -0.10403    -0.20112     -0.02093     0.02957    -0.03324      0.09371
HG_Shared2         H_PCA2    period_specific_HG_advantage    -0.14241    -0.25808     -0.04243     0.11161     0.02670      0.20191
HG_Shared2         G_PCA2       HG_advantage_both_periods     0.15192     0.09856      0.22098     0.09348     0.00939      0.18633
HG_Shared2    JIVE_Joint2       HG_advantage_both_periods     0.06072     0.02844      0.09786     0.04407     0.00349      0.08957
HG_Shared2   AJIVE_Joint2 period_specific_HG_disadvantage    -0.05627    -0.10803     -0.01325     0.00328    -0.03161      0.03582
HG_Shared2 VAE2_mean_loss period_specific_HG_disadvantage    -0.09529    -0.17925     -0.02303     0.06152    -0.00795      0.13572
```

## Held-out performance

```text
cohort              method   n  weighted_mse  weighted_R2  delta_mse_vs_covariates  low_ferritin_weighted_AUC  is_training_period
  0708          Covariates 500       0.90831     -0.02733                  0.00000                    0.44155               False
  0708               RawX7 500       0.62291      0.29546                  0.28540                    0.85665               False
  0708                  H4 500       0.62656      0.29133                  0.28174                    0.85761               False
  0708                  G3 500       0.91236     -0.03192                 -0.00405                    0.44625               False
  0708          Threshold2 500       0.73570      0.16789                  0.17261                    0.62069               False
  0708            Hb_HbA1c 500       0.68108      0.22967                  0.22723                    0.81184               False
  0708              Hb_RDW 500       0.62338      0.29494                  0.28493                    0.87086               False
  0708                PCA7 500       0.62291      0.29546                  0.28540                    0.85665               False
  0708                PCA2 500       0.66199      0.25126                  0.24632                    0.82093               False
  0708              H_PCA2 500       0.62361      0.29468                  0.28470                    0.85926               False
  0708              G_PCA2 500       0.91794     -0.03823                 -0.00964                    0.40757               False
  0708                 HG7 500       0.62291      0.29546                  0.28540                    0.85665               False
  0708          HG_Shared2 500       0.76602      0.13360                  0.14229                    0.80900               False
  0708      HG_Discordant2 500       0.76314      0.13686                  0.14517                    0.75326               False
  0708        HG_HPrivate2 500       0.78095      0.11671                  0.12736                    0.66704               False
  0708        HG_GPrivate1 500       0.90202     -0.02022                  0.00629                    0.49312               False
  0708 HG_Weighted_Shared2 500       0.74910      0.15274                  0.15921                    0.81545               False
  0708         JIVE_Joint2 500       0.82674      0.06492                  0.08157                    0.66590               False
  0708        AJIVE_Joint2 500       0.70975      0.19725                  0.19856                    0.82008               False
  0708        VAE2_seed118 500       0.65502      0.25915                  0.25329                    0.83668               False
  0708        VAE2_seed119 500       0.69291      0.21630                  0.21540                    0.82550               False
  0708        VAE2_seed120 500       0.68508      0.22514                  0.22323                    0.76864               False
  0708        VAE2_seed121 500       0.66988      0.24234                  0.23843                    0.83515               False
  0708        VAE2_seed122 500       0.65075      0.26397                  0.25755                    0.84010               False
  0708      VAE2_mean_loss 500       0.67073      0.24138                  0.23758                    0.82121               False
  2123          Covariates 460       0.74490      0.01463                  0.00000                    0.47879               False
  2123               RawX7 460       0.83463     -0.10406                 -0.08972                    0.86176               False
  2123                  H4 460       0.83340     -0.10244                 -0.08850                    0.85826               False
  2123                  G3 460       0.75023      0.00759                 -0.00533                    0.53935               False
  2123          Threshold2 460       0.66346      0.12237                  0.08145                    0.71731               False
  2123            Hb_HbA1c 460       0.66454      0.12094                  0.08036                    0.82506               False
  2123              Hb_RDW 460       0.85147     -0.12633                 -0.10656                    0.85597               False
  2123                PCA7 460       0.83463     -0.10406                 -0.08972                    0.86176               False
  2123                PCA2 460       0.69251      0.08393                  0.05239                    0.84583               False
  2123              H_PCA2 460       0.77455     -0.02459                 -0.02965                    0.86146               False
  2123              G_PCA2 460       0.75642     -0.00060                 -0.01152                    0.51307               False
  2123                 HG7 460       0.83463     -0.10406                 -0.08972                    0.86176               False
  2123          HG_Shared2 460       0.66294      0.12305                  0.08196                    0.77363               False
  2123      HG_Discordant2 460       0.69030      0.08686                  0.05461                    0.78561               False
  2123        HG_HPrivate2 460       0.73294      0.03046                  0.01196                    0.74180               False
  2123        HG_GPrivate1 460       0.75011      0.00775                 -0.00520                    0.50282               False
  2123 HG_Weighted_Shared2 460       0.65836      0.12912                  0.08655                    0.78104               False
  2123         JIVE_Joint2 460       0.70701      0.06475                  0.03789                    0.69043               False
  2123        AJIVE_Joint2 460       0.66623      0.11871                  0.07868                    0.81761               False
  2123        VAE2_seed118 460       0.75744     -0.00195                 -0.01254                    0.84847               False
  2123        VAE2_seed119 460       0.70152      0.07201                  0.04338                    0.83042               False
  2123        VAE2_seed120 460       0.72625      0.03931                  0.01866                    0.81651               False
  2123        VAE2_seed121 460       0.68592      0.09266                  0.05898                    0.86700               False
  2123        VAE2_seed122 460       0.75119      0.00631                 -0.00629                    0.84720               False
  2123      VAE2_mean_loss 460       0.72446      0.04167                  0.02044                    0.84192               False
```

## Interpretation constraints

- Ferritin is excluded from representation training.
- Ferritin is an assay anchor, not an independent cohort.
- All representations use identical marker inputs, except explicitly restricted baseline panels.
- All ferritin models share identical participants, covariates and weighted ridge settings.
- Whitening controls for arbitrary linear score scaling and rotation.
- VAE results concern a two-domain adaptation, not the published SYNERGY-VAE model.
- VAE seed averaging uses losses, not predictions.
- Confidence intervals condition on frozen fits.
- No multiplicity adjustment or confirmatory inference.
- Low ferritin AUC is descriptive, not diagnostic validation.
- Ferritin assays and complete-case selection may differ across periods.
- Equivalent or adverse results must be retained.

Full-rank equivalence passed: True
All VAE seeds available: True
Recorded exceptions: 0