# Script 97: Can Y recover a seven-biomarker encoded physiological state?

## What this experiment establishes
- Mathematical exactness of the X code is tested independently of statistical reverse inference.
- Identical phenotypes with different X codes prove a unique Y-to-X inverse does not exist for those observed cases.
- Learned frequency and token-motif models rank TRAIN physiology codes using Y only; exact decode then returns a real X candidate.

## Cohorts
```json
{
  "train_2005-2006": 1805,
  "2007-2008": 2340,
  "2009-2018": 11002,
  "2021-2023": 2590
}
```

## Mean held-out weighted R2
```
output                          continuous_prediction_NOT_code  decoded_top1  topk_mean_NOT_single_code
period    model                                                                                        
2007-2008 cryptanalysis_2grams                             NaN     -0.861002                  -0.126716
          cryptanalysis_3grams                             NaN     -0.873641                  -0.124876
          items                                            NaN     -0.856398                  -0.136279
          pair                                             NaN     -0.910445                  -0.166482
          prior_train_mean                           -0.033392           NaN                        NaN
          prior_weighted                                   NaN     -0.687193                  -0.146898
          ridge_items                                -0.029965           NaN                        NaN
          ridge_pair                                 -0.023654           NaN                        NaN
          ridge_total                                -0.022695           NaN                        NaN
          total                                            NaN     -0.841968                  -0.170816
2009-2018 cryptanalysis_2grams                             NaN     -0.894630                  -0.162794
          cryptanalysis_3grams                             NaN     -0.887540                  -0.163000
          items                                            NaN     -0.889026                  -0.185905
          pair                                             NaN     -0.947465                  -0.207495
          prior_train_mean                           -0.069890           NaN                        NaN
          prior_weighted                                   NaN     -0.761223                  -0.201472
          ridge_items                                -0.068246           NaN                        NaN
          ridge_pair                                 -0.061527           NaN                        NaN
          ridge_total                                -0.062205           NaN                        NaN
          total                                            NaN     -0.829530                  -0.215895
2021-2023 cryptanalysis_2grams                             NaN     -0.945286                  -0.232410
          cryptanalysis_3grams                             NaN     -0.949156                  -0.238430
          items                                            NaN     -0.952013                  -0.256181
          pair                                             NaN     -0.999640                  -0.277052
          prior_train_mean                           -0.146922           NaN                        NaN
          prior_weighted                                   NaN     -0.759583                  -0.293060
          ridge_items                                -0.135073           NaN                        NaN
          ridge_pair                                 -0.132332           NaN                        NaN
          ridge_total                                -0.133737           NaN                        NaN
          total                                            NaN     -0.876214                  -0.293720
```

## Interpretation
Candidate top-1 is an exactly decoded TRAIN participant code, not proof of matching the held-out patient.
The top-k mean is a regression-style average, NOT the decoding of one encoded state.
Any gains need comparison with ridge/prior and replication; no causal inference follows.
