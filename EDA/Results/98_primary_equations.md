# Script 98 — Primary numerical regression equations

All coefficients fitted using NHANES 2005–08 exclusively; model choice from training CV.
The label 'test 2009–18' refers to the five available 2-year NHANES cycles, not 2019 subjects.

## PHQ_total

### Overall training-CV winner

### PHQ_total: full / Z_pairwise / ElasticNet

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 2.4495309 -0.008371611 \cdot \mathrm{RIDAGEYR} +0.0289734 \cdot \mathrm{BMXBMI} -0.26657573 \cdot \mathrm{INDFMPIR} +0.80620004 \cdot \mathrm{RIAGENDR=2} +0.26448168 \cdot \mathrm{RACE=2} +0.037298937 \cdot \mathrm{RACE=3} +0.060789555 \cdot \mathrm{RACE=4} -0.14869253 \cdot \mathrm{EDUC3=3} +0.21724699 \cdot \mathrm{SMOKING3=1} +1.2044302 \cdot \mathrm{SMOKING3=2} +0.27772636 \cdot \mathrm{SP\_SHARED1} -0.054974267 \cdot \mathrm{SP\_SHARED2} -0.040873001 \cdot \mathrm{SP\_DISCORD1} -0.141135 \cdot \mathrm{SP\_DISCORD2} +0.41350903 \cdot \mathrm{SP\_A\_PRIVATE3} +0.15539376 \cdot \mathrm{SP\_A\_PRIVATE4} +0.033384861 \cdot \mathrm{SP\_G\_PRIVATE3} +0.0025995464 \cdot \mathrm{SP\_SHARED1*SP\_DISCORD1} +0.084405688 \cdot \mathrm{SP\_SHARED1*SP\_DISCORD2} +0.010359901 \cdot \mathrm{SP\_SHARED1*SP\_G\_PRIVATE3} -0.083268093 \cdot \mathrm{SP\_SHARED2*SP\_DISCORD2} -0.071397645 \cdot \mathrm{SP\_SHARED2*SP\_A\_PRIVATE4} +0.017097434 \cdot \mathrm{SP\_SHARED2*SP\_G\_PRIVATE3} +0.16794077 \cdot \mathrm{SP\_DISCORD1*SP\_DISCORD2} -0.00027491441 \cdot \mathrm{SP\_DISCORD1*SP\_A\_PRIVATE3} -0.0016435234 \cdot \mathrm{SP\_DISCORD2*SP\_A\_PRIVATE3} -0.0083293121 \cdot \mathrm{SP\_DISCORD2*SP\_A\_PRIVATE4} +0.012450281 \cdot \mathrm{SP\_A\_PRIVATE3*SP\_A\_PRIVATE4} -0.024616307 \cdot \mathrm{SP\_A\_PRIVATE4*SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.


### Best Z_main equation (CV selection)

### PHQ_total: full / Z_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 2.504025 -0.015382163 \cdot \mathrm{RIDAGEYR} +0.032017443 \cdot \mathrm{BMXBMI} -0.27671109 \cdot \mathrm{INDFMPIR} -0.0011341547 \cdot \mathrm{EGFR\_2021} +0.79457296 \cdot \mathrm{RIAGENDR=2} +0.84434119 \cdot \mathrm{RACE=2} +0.63105983 \cdot \mathrm{RACE=3} +0.61192192 \cdot \mathrm{RACE=4} +0.52072732 \cdot \mathrm{RACE=5} -0.30798863 \cdot \mathrm{EDUC3=2} -0.45177461 \cdot \mathrm{EDUC3=3} +0.34882562 \cdot \mathrm{SMOKING3=1} +1.2657803 \cdot \mathrm{SMOKING3=2} +0.25589871 \cdot \mathrm{SP\_SHARED1} -0.074164909 \cdot \mathrm{SP\_SHARED2} -0.12221292 \cdot \mathrm{SP\_DISCORD1} -0.20240215 \cdot \mathrm{SP\_DISCORD2} +0.39090449 \cdot \mathrm{SP\_A\_PRIVATE3} +0.20496311 \cdot \mathrm{SP\_A\_PRIVATE4} +0.12360738 \cdot \mathrm{SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.


### Best X_main equation (CV selection)

### PHQ_total: full / X_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 4.9015858 -0.016654379 \cdot \mathrm{RIDAGEYR} +0.031978952 \cdot \mathrm{BMXBMI} -0.28151427 \cdot \mathrm{INDFMPIR} -0.0014317468 \cdot \mathrm{EGFR\_2021} +0.80944168 \cdot \mathrm{RIAGENDR=2} +0.93418435 \cdot \mathrm{RACE=2} +0.72759286 \cdot \mathrm{RACE=3} +0.68883754 \cdot \mathrm{RACE=4} +0.61988341 \cdot \mathrm{RACE=5} -0.34581886 \cdot \mathrm{EDUC3=2} -0.48477394 \cdot \mathrm{EDUC3=3} +0.37298558 \cdot \mathrm{SMOKING3=1} +1.2949389 \cdot \mathrm{SMOKING3=2} +0.42743553 \cdot \mathrm{LBXHGB} -1.8641549 \cdot \mathrm{LBXRBCSI} -0.067970481 \cdot \mathrm{LBXMCVSI} +0.3576256 \cdot \mathrm{LBXRDW} +0.184038 \cdot \mathrm{LBXGH} +0.0016615891 \cdot \mathrm{LBXGLU} +0.2970213 \cdot \mathrm{LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.


## Somatic

### Overall training-CV winner

### Somatic: full / Z_pairwise / ElasticNet

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 1.280576 -0.0033499051 \cdot \mathrm{RIDAGEYR} +0.014895982 \cdot \mathrm{BMXBMI} -0.10756342 \cdot \mathrm{INDFMPIR} +0.40639736 \cdot \mathrm{RIAGENDR=2} +0.060849765 \cdot \mathrm{RACE=3} -0.033168458 \cdot \mathrm{EDUC3=3} +0.14705566 \cdot \mathrm{SMOKING3=1} +0.54866207 \cdot \mathrm{SMOKING3=2} +0.13952468 \cdot \mathrm{SP\_SHARED1} -0.040097312 \cdot \mathrm{SP\_SHARED2} -0.023068142 \cdot \mathrm{SP\_DISCORD1} -0.085851372 \cdot \mathrm{SP\_DISCORD2} +0.20248644 \cdot \mathrm{SP\_A\_PRIVATE3} +0.067235076 \cdot \mathrm{SP\_A\_PRIVATE4} +0.011491562 \cdot \mathrm{SP\_G\_PRIVATE3} +0.042301702 \cdot \mathrm{SP\_SHARED1*SP\_DISCORD2} -0.028553737 \cdot \mathrm{SP\_SHARED2*SP\_DISCORD2} -0.019070813 \cdot \mathrm{SP\_SHARED2*SP\_A\_PRIVATE4} +0.012632029 \cdot \mathrm{SP\_SHARED2*SP\_G\_PRIVATE3} +0.072474498 \cdot \mathrm{SP\_DISCORD1*SP\_DISCORD2} -0.0035128098 \cdot \mathrm{SP\_DISCORD1*SP\_A\_PRIVATE3} -0.017155555 \cdot \mathrm{SP\_DISCORD2*SP\_A\_PRIVATE4}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.


### Best Z_main equation (CV selection)

### Somatic: full / Z_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 1.666809 -0.0099200941 \cdot \mathrm{RIDAGEYR} +0.016453705 \cdot \mathrm{BMXBMI} -0.11649383 \cdot \mathrm{INDFMPIR} -0.0032580164 \cdot \mathrm{EGFR\_2021} +0.40430455 \cdot \mathrm{RIAGENDR=2} +0.33230788 \cdot \mathrm{RACE=2} +0.36777641 \cdot \mathrm{RACE=3} +0.26127966 \cdot \mathrm{RACE=4} +0.28290097 \cdot \mathrm{RACE=5} -0.10464394 \cdot \mathrm{EDUC3=2} -0.16359158 \cdot \mathrm{EDUC3=3} +0.24445535 \cdot \mathrm{SMOKING3=1} +0.6152668 \cdot \mathrm{SMOKING3=2} +0.13648247 \cdot \mathrm{SP\_SHARED1} -0.053038248 \cdot \mathrm{SP\_SHARED2} -0.068456166 \cdot \mathrm{SP\_DISCORD1} -0.12671945 \cdot \mathrm{SP\_DISCORD2} +0.20458261 \cdot \mathrm{SP\_A\_PRIVATE3} +0.098475627 \cdot \mathrm{SP\_A\_PRIVATE4} +0.061458629 \cdot \mathrm{SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.


### Best X_main equation (CV selection)

### Somatic: full / X_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 1.0582303 -0.0098804471 \cdot \mathrm{RIDAGEYR} +0.01641882 \cdot \mathrm{BMXBMI} -0.1169486 \cdot \mathrm{INDFMPIR} -0.0030970747 \cdot \mathrm{EGFR\_2021} +0.38932657 \cdot \mathrm{RIAGENDR=2} +0.32303725 \cdot \mathrm{RACE=2} +0.36906915 \cdot \mathrm{RACE=3} +0.23075329 \cdot \mathrm{RACE=4} +0.27480902 \cdot \mathrm{RACE=5} -0.10620801 \cdot \mathrm{EDUC3=2} -0.16186726 \cdot \mathrm{EDUC3=3} +0.24144755 \cdot \mathrm{SMOKING3=1} +0.62109927 \cdot \mathrm{SMOKING3=2} +0.074967166 \cdot \mathrm{LBXHGB} -0.56693628 \cdot \mathrm{LBXRBCSI} -0.010686661 \cdot \mathrm{LBXMCVSI} +0.17603106 \cdot \mathrm{LBXRDW} +0.080827962 \cdot \mathrm{LBXGH} +0.0011840098 \cdot \mathrm{LBXGLU} +0.16629604 \cdot \mathrm{LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.


## CogAff

### Overall training-CV winner

### CogAff: full / Z_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 0.83721595 -0.0054620692 \cdot \mathrm{RIDAGEYR} +0.015563738 \cdot \mathrm{BMXBMI} -0.16021725 \cdot \mathrm{INDFMPIR} +0.0021238618 \cdot \mathrm{EGFR\_2021} +0.3902684 \cdot \mathrm{RIAGENDR=2} +0.51203331 \cdot \mathrm{RACE=2} +0.26328342 \cdot \mathrm{RACE=3} +0.35064226 \cdot \mathrm{RACE=4} +0.23782635 \cdot \mathrm{RACE=5} -0.20334469 \cdot \mathrm{EDUC3=2} -0.28818302 \cdot \mathrm{EDUC3=3} +0.10437027 \cdot \mathrm{SMOKING3=1} +0.65051347 \cdot \mathrm{SMOKING3=2} +0.11941624 \cdot \mathrm{SP\_SHARED1} -0.021126662 \cdot \mathrm{SP\_SHARED2} -0.053756749 \cdot \mathrm{SP\_DISCORD1} -0.075682704 \cdot \mathrm{SP\_DISCORD2} +0.18632188 \cdot \mathrm{SP\_A\_PRIVATE3} +0.10648748 \cdot \mathrm{SP\_A\_PRIVATE4} +0.06214875 \cdot \mathrm{SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.


### Best Z_main equation (CV selection)

### CogAff: full / Z_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 0.83721595 -0.0054620692 \cdot \mathrm{RIDAGEYR} +0.015563738 \cdot \mathrm{BMXBMI} -0.16021725 \cdot \mathrm{INDFMPIR} +0.0021238618 \cdot \mathrm{EGFR\_2021} +0.3902684 \cdot \mathrm{RIAGENDR=2} +0.51203331 \cdot \mathrm{RACE=2} +0.26328342 \cdot \mathrm{RACE=3} +0.35064226 \cdot \mathrm{RACE=4} +0.23782635 \cdot \mathrm{RACE=5} -0.20334469 \cdot \mathrm{EDUC3=2} -0.28818302 \cdot \mathrm{EDUC3=3} +0.10437027 \cdot \mathrm{SMOKING3=1} +0.65051347 \cdot \mathrm{SMOKING3=2} +0.11941624 \cdot \mathrm{SP\_SHARED1} -0.021126662 \cdot \mathrm{SP\_SHARED2} -0.053756749 \cdot \mathrm{SP\_DISCORD1} -0.075682704 \cdot \mathrm{SP\_DISCORD2} +0.18632188 \cdot \mathrm{SP\_A\_PRIVATE3} +0.10648748 \cdot \mathrm{SP\_A\_PRIVATE4} +0.06214875 \cdot \mathrm{SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.


### Best X_main equation (CV selection)

### CogAff: full / X_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 2.7331631 -0.0058361836 \cdot \mathrm{RIDAGEYR} +0.015586767 \cdot \mathrm{BMXBMI} -0.16303127 \cdot \mathrm{INDFMPIR} +0.0021635935 \cdot \mathrm{EGFR\_2021} +0.39827179 \cdot \mathrm{RIAGENDR=2} +0.55803082 \cdot \mathrm{RACE=2} +0.31235437 \cdot \mathrm{RACE=3} +0.39197752 \cdot \mathrm{RACE=4} +0.2872301 \cdot \mathrm{RACE=5} -0.22587441 \cdot \mathrm{EDUC3=2} -0.30742619 \cdot \mathrm{EDUC3=3} +0.11391758 \cdot \mathrm{SMOKING3=1} +0.66432006 \cdot \mathrm{SMOKING3=2} +0.25191617 \cdot \mathrm{LBXHGB} -1.0045315 \cdot \mathrm{LBXRBCSI} -0.041878387 \cdot \mathrm{LBXMCVSI} +0.165793 \cdot \mathrm{LBXRDW} +0.090375515 \cdot \mathrm{LBXGH} +0.00076045685 \cdot \mathrm{LBXGLU} +0.12743874 \cdot \mathrm{LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.


## Caution

These are predictive/associative equations, not PDEs or causal models.
Centering, units, contrast baselines and raw interaction definitions are documented in the full equations file.