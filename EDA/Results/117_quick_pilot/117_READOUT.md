# Script 117 — Comparative structural validation

## Interpretation

- Full-rank reconstruction is an information control.
- The comparative refits use common unweighted fitting.
- Empirical nulls preserve survey cycle, sex and broad age bands.
- Empirical bootstrap intervals condition on the reference fit.
- Empirical intervals are descriptive, not survey-design CIs.
- Native score groups are dimension matched, not semantically identical.
- PCA leading components are variance controls.
- Block PCA components are domain-restricted controls.
- No artificial discordance groups are assigned to comparators.
- Rank-deficient AJIVE fits are excluded from strict comparisons.
- Synthetic recovery depends on the stipulated generator.
- Jacobian restrictions do not identify biological sources.
- All intervals are exploratory and unadjusted for multiplicity.

## Actual fitted dimensions

```text
              phase         condition          method  recorded_fits  eligible_7D_fits
   conditional_null              0708 AJIVE7_adaptive              5                 5
   conditional_null              0708       BlockPCA7              5                 5
   conditional_null              0708             HG7              5                 5
   conditional_null              0708           JIVE7              5                 5
   conditional_null              0708            PCA7              5                 5
   conditional_null              0918 AJIVE7_adaptive              5                 5
   conditional_null              0918       BlockPCA7              5                 5
   conditional_null              0918             HG7              5                 5
   conditional_null              0918           JIVE7              5                 5
   conditional_null              0918            PCA7              5                 5
   conditional_null              2123 AJIVE7_adaptive              5                 5
   conditional_null              2123       BlockPCA7              5                 5
   conditional_null              2123             HG7              5                 5
   conditional_null              2123           JIVE7              5                 5
   conditional_null              2123            PCA7              5                 5
empirical_reference              0506 AJIVE7_adaptive              1                 1
empirical_reference              0506       BlockPCA7              1                 1
empirical_reference              0506             HG7              1                 1
empirical_reference              0506           JIVE7              1                 1
empirical_reference              0506            PCA7              1                 1
empirical_reference              0506           RawX7              1                 1
           observed              0708 AJIVE7_adaptive              1                 1
           observed              0708       BlockPCA7              1                 1
           observed              0708             HG7              1                 1
           observed              0708           JIVE7              1                 1
           observed              0708            PCA7              1                 1
           observed              0918 AJIVE7_adaptive              1                 1
           observed              0918       BlockPCA7              1                 1
           observed              0918             HG7              1                 1
           observed              0918           JIVE7              1                 1
           observed              0918            PCA7              1                 1
           observed              2123 AJIVE7_adaptive              1                 1
           observed              2123       BlockPCA7              1                 1
           observed              2123             HG7              1                 1
           observed              2123           JIVE7              1                 1
           observed              2123            PCA7              1                 1
          synthetic        high_noise AJIVE7_adaptive              5                 0
          synthetic        high_noise       BlockPCA7              5                 5
          synthetic        high_noise             HG7              5                 5
          synthetic        high_noise           JIVE7              5                 5
          synthetic        high_noise            PCA7              5                 5
          synthetic        high_noise           RawX7              5                 5
          synthetic measurement_shift AJIVE7_adaptive              5                 0
          synthetic measurement_shift       BlockPCA7              5                 5
          synthetic measurement_shift             HG7              5                 5
          synthetic measurement_shift           JIVE7              5                 5
          synthetic measurement_shift            PCA7              5                 5
          synthetic measurement_shift           RawX7              5                 5
          synthetic        no_sharing AJIVE7_adaptive              5                 0
          synthetic        no_sharing       BlockPCA7              5                 5
          synthetic        no_sharing             HG7              5                 5
          synthetic        no_sharing           JIVE7              5                 5
          synthetic        no_sharing            PCA7              5                 5
          synthetic        no_sharing           RawX7              5                 5
          synthetic  population_shift AJIVE7_adaptive              5                 0
          synthetic  population_shift       BlockPCA7              5                 5
          synthetic  population_shift             HG7              5                 5
          synthetic  population_shift           JIVE7              5                 5
          synthetic  population_shift            PCA7              5                 5
          synthetic  population_shift           RawX7              5                 5
          synthetic  private_dominant AJIVE7_adaptive              5                 4
          synthetic  private_dominant       BlockPCA7              5                 5
          synthetic  private_dominant             HG7              5                 5
          synthetic  private_dominant           JIVE7              5                 5
          synthetic  private_dominant            PCA7              5                 5
          synthetic  private_dominant           RawX7              5                 5
          synthetic         reference AJIVE7_adaptive              5                 0
          synthetic         reference       BlockPCA7              5                 5
          synthetic         reference             HG7              5                 5
          synthetic         reference           JIVE7              5                 5
          synthetic         reference            PCA7              5                 5
          synthetic         reference           RawX7              5                 5
          synthetic   strong_coupling AJIVE7_adaptive              5                 0
          synthetic   strong_coupling       BlockPCA7              5                 5
          synthetic   strong_coupling             HG7              5                 5
          synthetic   strong_coupling           JIVE7              5                 5
          synthetic   strong_coupling            PCA7              5                 5
          synthetic   strong_coupling           RawX7              5                 5
          synthetic     weak_coupling AJIVE7_adaptive              5                 0
          synthetic     weak_coupling       BlockPCA7              5                 5
          synthetic     weak_coupling             HG7              5                 5
          synthetic     weak_coupling           JIVE7              5                 5
          synthetic     weak_coupling            PCA7              5                 5
          synthetic     weak_coupling           RawX7              5                 5
   target_bootstrap              0708 AJIVE7_adaptive              5                 4
   target_bootstrap              0708       BlockPCA7              5                 5
   target_bootstrap              0708             HG7              5                 5
   target_bootstrap              0708           JIVE7              5                 5
   target_bootstrap              0708            PCA7              5                 5
   target_bootstrap              0918 AJIVE7_adaptive              5                 5
   target_bootstrap              0918       BlockPCA7              5                 5
   target_bootstrap              0918             HG7              5                 5
   target_bootstrap              0918           JIVE7              5                 5
   target_bootstrap              0918            PCA7              5                 5
   target_bootstrap              2123 AJIVE7_adaptive              5                 5
   target_bootstrap              2123       BlockPCA7              5                 5
   target_bootstrap              2123             HG7              5                 5
   target_bootstrap              2123           JIVE7              5                 5
   target_bootstrap              2123            PCA7              5                 5
```

## Empirical structural specificity

```text
 cohort          method      group  group_dim  observed_similarity  conditional_null_q95  specificity_margin  descriptive_bootstrap_low  descriptive_bootstrap_high  valid_null_reps  requested_null_reps  valid_bootstrap_reps  requested_bootstrap_reps
    708             HG7     shared          2              0.73241               0.43986             0.29254                    0.42579                     0.74395                5                    5                     5                         5
    708             HG7 discordant          2              0.68365               0.41783             0.26581                    0.35709                     0.65561                5                    5                     5                         5
    708             HG7  H_private          2              0.77976               0.61128             0.16848                    0.47646                     0.94049                5                    5                     5                         5
    708             HG7  G_private          1              0.66066               0.86654            -0.20587                    0.00325                     0.74035                5                    5                     5                         5
    708            PCA7     shared          2              0.96503               0.96617            -0.00114                    0.70062                     0.90393                5                    5                     5                         5
    708       BlockPCA7  H_private          2              0.99659               0.99659             0.00000                    0.99213                     0.99834                5                    5                     5                         5
    708       BlockPCA7  G_private          1              0.99628               0.99628            -0.00000                    0.97048                     0.99664                5                    5                     5                         5
    708           JIVE7     shared          2              0.45949               0.42344             0.03605                    0.15974                     0.77093                5                    5                     5                         5
    708           JIVE7  H_private          2              0.97331               0.86725             0.10606                    0.33692                     0.96890                5                    5                     5                         5
    708           JIVE7  G_private          1              0.03280               0.55593            -0.52313                    0.00306                     0.82327                5                    5                     5                         5
    708 AJIVE7_adaptive     shared          2              0.98673               0.77116             0.21557                    0.78134                     0.98834                5                    5                     4                         5
    708 AJIVE7_adaptive  H_private          2              0.97242               0.62070             0.35172                    0.61132                     0.97357                5                    5                     4                         5
    708 AJIVE7_adaptive  G_private          1              0.95959               0.87182             0.08776                    0.73314                     0.93816                5                    5                     4                         5
    918             HG7     shared          2              0.70242               0.59636             0.10606                    0.50746                     0.69894                5                    5                     5                         5
    918             HG7 discordant          2              0.66233               0.44815             0.21419                    0.40723                     0.66344                5                    5                     5                         5
    918             HG7  H_private          2              0.49997               0.70428            -0.20430                    0.49330                     0.56202                5                    5                     5                         5
    918             HG7  G_private          1              0.98209               0.85498             0.12711                    0.71985                     0.98348                5                    5                     5                         5
    918            PCA7     shared          2              0.82129               0.96444            -0.14315                    0.54069                     0.94515                5                    5                     5                         5
    918       BlockPCA7  H_private          2              0.99898               0.99898             0.00000                    0.99462                     0.99872                5                    5                     5                         5
    918       BlockPCA7  G_private          1              0.99811               0.99811            -0.00000                    0.99471                     0.99990                5                    5                     5                         5
    918           JIVE7     shared          2              0.49841               0.42318             0.07523                    0.31106                     0.49622                5                    5                     5                         5
    918           JIVE7  H_private          2              0.98851               0.66500             0.32352                    0.81695                     0.99244                5                    5                     5                         5
    918           JIVE7  G_private          1              0.00034               0.28037            -0.28003                    0.00087                     0.61394                5                    5                     5                         5
    918 AJIVE7_adaptive     shared          2              0.88280               0.85545             0.02735                    0.71763                     0.93386                5                    5                     5                         5
    918 AJIVE7_adaptive  H_private          2              0.81555               0.70163             0.11392                    0.65265                     0.89886                5                    5                     5                         5
    918 AJIVE7_adaptive  G_private          1              0.72952               0.92520            -0.19568                    0.39608                     0.81792                5                    5                     5                         5
   2123             HG7     shared          2              0.94544               0.59591             0.34953                    0.53629                     0.93780                5                    5                     5                         5
   2123             HG7 discordant          2              0.93018               0.49883             0.43135                    0.48553                     0.92227                5                    5                     5                         5
   2123             HG7  H_private          2              0.81619               0.48328             0.33291                    0.39124                     0.83479                5                    5                     5                         5
   2123             HG7  G_private          1              0.90262               0.54334             0.35928                    0.07195                     0.89368                5                    5                     5                         5
   2123            PCA7     shared          2              0.92586               0.98663            -0.06077                    0.79683                     0.95181                5                    5                     5                         5
   2123       BlockPCA7  H_private          2              0.99803               0.99803            -0.00000                    0.99665                     0.99946                5                    5                     5                         5
   2123       BlockPCA7  G_private          1              0.99820               0.99820            -0.00000                    0.99258                     0.99861                5                    5                     5                         5
   2123           JIVE7     shared          2              0.46360               0.25107             0.21253                    0.32067                     0.45408                5                    5                     5                         5
   2123           JIVE7  H_private          2              0.84026               0.52240             0.31785                    0.51217                     0.83645                5                    5                     5                         5
   2123           JIVE7  G_private          1              0.00171               0.04589            -0.04418                    0.00012                     0.07488                5                    5                     5                         5
   2123 AJIVE7_adaptive     shared          2              0.98585               0.87309             0.11275                    0.82024                     0.98369                5                    5                     5                         5
   2123 AJIVE7_adaptive  H_private          2              0.96250               0.72676             0.23574                    0.69060                     0.95088                5                    5                     5                         5
   2123 AJIVE7_adaptive  G_private          1              0.92448               0.95434            -0.02985                    0.68224                     0.92023                5                    5                     5                         5
```

## Matched designated-group recovery

```text
         scenario     group  group_dim    target      comparator  paired_reps  requested_reps  mean_delta_R2  paired_bootstrap_low95  paired_bootstrap_high95                 decision
       high_noise    shared          2    shared            PCA7            5               5        0.02979                 0.02003                  0.03565              favours_HG7
measurement_shift    shared          2    shared            PCA7            5               5        0.00573                -0.00545                  0.01753             inconclusive
 population_shift    shared          2    shared            PCA7            5               5        0.00859                 0.00261                  0.01374              favours_HG7
 private_dominant    shared          2    shared            PCA7            5               5        0.92147                 0.88030                  0.95229              favours_HG7
        reference    shared          2    shared            PCA7            5               5        0.01367                 0.00148                  0.03146              favours_HG7
  strong_coupling    shared          2    shared            PCA7            5               5        0.00804                 0.00390                  0.01349              favours_HG7
    weak_coupling    shared          2    shared            PCA7            5               5        0.13504                 0.04478                  0.25051              favours_HG7
       high_noise G_private          1 G_private       BlockPCA7            5               5        0.61363                 0.51642                  0.69712              favours_HG7
       high_noise H_private          2 H_private       BlockPCA7            5               5        0.27658                 0.19579                  0.36124              favours_HG7
measurement_shift G_private          1 G_private       BlockPCA7            5               5        0.50649                 0.22107                  0.78122              favours_HG7
measurement_shift H_private          2 H_private       BlockPCA7            5               5        0.41455                 0.32278                  0.48087              favours_HG7
       no_sharing G_private          1 G_private       BlockPCA7            5               5        0.10586                -0.25424                  0.46597             inconclusive
       no_sharing H_private          2 H_private       BlockPCA7            5               5       -0.17341                -0.39818                 -0.03162       favours_comparator
 population_shift G_private          1 G_private       BlockPCA7            5               5        0.85808                 0.67899                  0.97394              favours_HG7
 population_shift H_private          2 H_private       BlockPCA7            5               5        0.43097                 0.30883                  0.55094              favours_HG7
 private_dominant G_private          1 G_private       BlockPCA7            5               5        0.03488                 0.00167                  0.06539              favours_HG7
 private_dominant H_private          2 H_private       BlockPCA7            5               5        0.09139                -0.00614                  0.26480             inconclusive
        reference G_private          1 G_private       BlockPCA7            5               5        0.56457                 0.29331                  0.81117              favours_HG7
        reference H_private          2 H_private       BlockPCA7            5               5        0.51038                 0.47017                  0.55957              favours_HG7
  strong_coupling G_private          1 G_private       BlockPCA7            5               5        0.15270                 0.06863                  0.24458              favours_HG7
  strong_coupling H_private          2 H_private       BlockPCA7            5               5        0.50792                 0.31553                  0.71918              favours_HG7
    weak_coupling G_private          1 G_private       BlockPCA7            5               5        0.38923                 0.04427                  0.73418              favours_HG7
    weak_coupling H_private          2 H_private       BlockPCA7            5               5        0.27050                -0.00132                  0.52466             inconclusive
       high_noise G_private          1 G_private           JIVE7            5               5        0.06132                 0.00605                  0.15512              favours_HG7
       high_noise H_private          2 H_private           JIVE7            5               5        0.07437                 0.00658                  0.19621              favours_HG7
       high_noise    shared          2    shared           JIVE7            5               5        0.25985                 0.25348                  0.26741              favours_HG7
measurement_shift G_private          1 G_private           JIVE7            5               5        0.04808                 0.01731                  0.08408              favours_HG7
measurement_shift H_private          2 H_private           JIVE7            5               5        0.02005                -0.01939                  0.08305             inconclusive
measurement_shift    shared          2    shared           JIVE7            5               5        0.25443                 0.23635                  0.27249              favours_HG7
       no_sharing G_private          1 G_private           JIVE7            5               5        0.26347                 0.06457                  0.46408              favours_HG7
       no_sharing H_private          2 H_private           JIVE7            5               5       -0.14635                -0.26656                 -0.02614       favours_comparator
 population_shift G_private          1 G_private           JIVE7            5               5        0.07465                 0.00690                  0.14241              favours_HG7
 population_shift H_private          2 H_private           JIVE7            5               5        0.01309                 0.00267                  0.02452              favours_HG7
 population_shift    shared          2    shared           JIVE7            5               5        0.21305                 0.20170                  0.22476              favours_HG7
 private_dominant G_private          1 G_private           JIVE7            5               5        0.21002                 0.01083                  0.59204              favours_HG7
 private_dominant H_private          2 H_private           JIVE7            5               5        0.02098                -0.01014                  0.07711             inconclusive
 private_dominant    shared          2    shared           JIVE7            5               5        0.33115                 0.23907                  0.48042              favours_HG7
        reference G_private          1 G_private           JIVE7            5               5        0.03752                 0.01842                  0.05711              favours_HG7
        reference H_private          2 H_private           JIVE7            5               5        0.00745                -0.00116                  0.01923             inconclusive
        reference    shared          2    shared           JIVE7            5               5        0.26451                 0.24834                  0.28068              favours_HG7
  strong_coupling G_private          1 G_private           JIVE7            5               5        0.00042                -0.00360                  0.00383             inconclusive
  strong_coupling H_private          2 H_private           JIVE7            5               5       -0.00067                -0.00137                  0.00003             inconclusive
  strong_coupling    shared          2    shared           JIVE7            5               5        0.09002                 0.08737                  0.09166              favours_HG7
    weak_coupling G_private          1 G_private           JIVE7            5               5        0.38001                 0.09329                  0.67856              favours_HG7
    weak_coupling H_private          2 H_private           JIVE7            5               5        0.20402                 0.06920                  0.32573              favours_HG7
    weak_coupling    shared          2    shared           JIVE7            5               5        0.40993                 0.38580                  0.42897              favours_HG7
 private_dominant G_private          1 G_private AJIVE7_adaptive            4               5        0.50203                     NaN                      NaN insufficient_uncertainty
 private_dominant H_private          2 H_private AJIVE7_adaptive            4               5        0.23837                     NaN                      NaN insufficient_uncertainty
 private_dominant    shared          2    shared AJIVE7_adaptive            4               5        0.46516                     NaN                      NaN insufficient_uncertainty
```

## Rank-selection sensitivity

```text
         scenario  mean_selected_rank  fraction_detecting_coupling  repetitions
       high_noise                 2.0                          1.0            5
measurement_shift                 2.0                          1.0            5
       no_sharing                 0.2                          0.2            5
 population_shift                 2.0                          1.0            5
 private_dominant                 2.0                          1.0            5
        reference                 2.0                          1.0            5
  strong_coupling                 2.0                          1.0            5
    weak_coupling                 0.6                          0.6            5
```

## Recorded fitting exceptions

0

No universal method ranking is inferred.
