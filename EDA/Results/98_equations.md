# Script 98 — Fitted phenotype equations

Frozen NHANES cycle protocol: train 2005–08 (including training-only 5-fold CV); test 2009–18; validate 2021–23.
No tuning using test or validation outcomes. Discovery transformation Z is unchanged.
All designs use the same fully observed X/Z/Y/covariate cohort for fair score comparison.

## CV-selected overall design for each outcome

```csv
outcome,selected_scope,selected_design,method,alpha,l1_ratio,cv_MSE,test_R2,validation_R2
PHQ_total,full,Z_pairwise,ElasticNet,0.05,0.8,12.335021184101018,0.07537586651341832,0.0007724494628471579
Somatic,full,Z_pairwise,ElasticNet,0.05,0.5,3.4182576886562246,0.07359977878691071,0.03844011499515254
CogAff,full,Z_main,Ridge,100.0,,4.451276293623979,0.05921799857740451,-0.03517776866182731

```

## Every fitted equation

### PHQ_total: lean / covariates / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 0.93831898 -0.0062114895 \cdot \mathrm{RIDAGEYR} +0.055510715 \cdot \mathrm{BMXBMI} +0.99126468 \cdot \mathrm{RIAGENDR=2}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### PHQ_total: lean / Z_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 2.465747 -0.020450462 \cdot \mathrm{RIDAGEYR} +0.028921827 \cdot \mathrm{BMXBMI} +0.82640703 \cdot \mathrm{RIAGENDR=2} +0.27010713 \cdot \mathrm{SP\_SHARED1} -0.11176492 \cdot \mathrm{SP\_SHARED2} -0.17107305 \cdot \mathrm{SP\_DISCORD1} -0.18587041 \cdot \mathrm{SP\_DISCORD2} +0.53364153 \cdot \mathrm{SP\_A\_PRIVATE3} +0.08007319 \cdot \mathrm{SP\_A\_PRIVATE4} +0.17965189 \cdot \mathrm{SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### PHQ_total: lean / X_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = -1.245486 -0.021571936 \cdot \mathrm{RIDAGEYR} +0.028779439 \cdot \mathrm{BMXBMI} +0.8460221 \cdot \mathrm{RIAGENDR=2} +0.50121422 \cdot \mathrm{LBXHGB} -1.8956058 \cdot \mathrm{LBXRBCSI} -0.040663885 \cdot \mathrm{LBXMCVSI} +0.5328769 \cdot \mathrm{LBXRDW} +0.30216775 \cdot \mathrm{LBXGH} +0.00081080807 \cdot \mathrm{LBXGLU} +0.28260748 \cdot \mathrm{LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### PHQ_total: lean / X_AxG / ElasticNet

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = -3.6983685 -0.020830676 \cdot \mathrm{RIDAGEYR} +0.030944583 \cdot \mathrm{BMXBMI} +0.82530432 \cdot \mathrm{RIAGENDR=2} +0.29201587 \cdot \mathrm{LBXHGB} -1.2843196 \cdot \mathrm{LBXRBCSI} -0.0043766746 \cdot \mathrm{LBXMCVSI} +0.4923678 \cdot \mathrm{LBXRDW} +0.2569403 \cdot \mathrm{LBXGH} +0.0014365099 \cdot \mathrm{LBXGLU} +0.25791657 \cdot \mathrm{LOG\_IN} +0.083219338 \cdot \mathrm{LBXHGB*LBXGLU} +0.12803126 \cdot \mathrm{LBXHGB*LOG\_IN} -0.17076912 \cdot \mathrm{LBXRBCSI*LOG\_IN} -0.083316959 \cdot \mathrm{LBXMCVSI*LBXGH} +0.07558273 \cdot \mathrm{LBXMCVSI*LBXGLU} +0.035621843 \cdot \mathrm{LBXMCVSI*LOG\_IN} +0.12865011 \cdot \mathrm{LBXRDW*LBXGH} -0.078005596 \cdot \mathrm{LBXRDW*LBXGLU} +0.012411433 \cdot \mathrm{LBXRDW*LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### PHQ_total: lean / Z_pairwise / ElasticNet

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 2.4333515 -0.017238661 \cdot \mathrm{RIDAGEYR} +0.02412911 \cdot \mathrm{BMXBMI} +0.85848961 \cdot \mathrm{RIAGENDR=2} +0.31146887 \cdot \mathrm{SP\_SHARED1} -0.10142442 \cdot \mathrm{SP\_SHARED2} -0.10220225 \cdot \mathrm{SP\_DISCORD1} -0.12123765 \cdot \mathrm{SP\_DISCORD2} +0.58462252 \cdot \mathrm{SP\_A\_PRIVATE3} +0.027922692 \cdot \mathrm{SP\_A\_PRIVATE4} +0.10002921 \cdot \mathrm{SP\_G\_PRIVATE3} +0.061324921 \cdot \mathrm{SP\_SHARED1*SP\_DISCORD2} -0.091926697 \cdot \mathrm{SP\_SHARED2*SP\_DISCORD2} -0.068108442 \cdot \mathrm{SP\_SHARED2*SP\_A\_PRIVATE4} +0.027119396 \cdot \mathrm{SP\_SHARED2*SP\_G\_PRIVATE3} +0.17645272 \cdot \mathrm{SP\_DISCORD1*SP\_DISCORD2} -0.026289008 \cdot \mathrm{SP\_DISCORD1*SP\_A\_PRIVATE3} -0.030178356 \cdot \mathrm{SP\_DISCORD2*SP\_A\_PRIVATE4} +0.029155511 \cdot \mathrm{SP\_A\_PRIVATE3*SP\_A\_PRIVATE4} -0.035794002 \cdot \mathrm{SP\_A\_PRIVATE4*SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### PHQ_total: full / covariates / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 1.2463505 -0.0045615813 \cdot \mathrm{RIDAGEYR} +0.056746516 \cdot \mathrm{BMXBMI} -0.31136213 \cdot \mathrm{INDFMPIR} -1.3572133e-05 \cdot \mathrm{EGFR\_2021} +1.0547576 \cdot \mathrm{RIAGENDR=2} +0.82872127 \cdot \mathrm{RACE=2} +0.56526985 \cdot \mathrm{RACE=3} +0.68933542 \cdot \mathrm{RACE=4} +0.46620299 \cdot \mathrm{RACE=5} -0.38323231 \cdot \mathrm{EDUC3=2} -0.50985327 \cdot \mathrm{EDUC3=3} +0.33751472 \cdot \mathrm{SMOKING3=1} +1.2584134 \cdot \mathrm{SMOKING3=2}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### PHQ_total: full / Z_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 2.504025 -0.015382163 \cdot \mathrm{RIDAGEYR} +0.032017443 \cdot \mathrm{BMXBMI} -0.27671109 \cdot \mathrm{INDFMPIR} -0.0011341547 \cdot \mathrm{EGFR\_2021} +0.79457296 \cdot \mathrm{RIAGENDR=2} +0.84434119 \cdot \mathrm{RACE=2} +0.63105983 \cdot \mathrm{RACE=3} +0.61192192 \cdot \mathrm{RACE=4} +0.52072732 \cdot \mathrm{RACE=5} -0.30798863 \cdot \mathrm{EDUC3=2} -0.45177461 \cdot \mathrm{EDUC3=3} +0.34882562 \cdot \mathrm{SMOKING3=1} +1.2657803 \cdot \mathrm{SMOKING3=2} +0.25589871 \cdot \mathrm{SP\_SHARED1} -0.074164909 \cdot \mathrm{SP\_SHARED2} -0.12221292 \cdot \mathrm{SP\_DISCORD1} -0.20240215 \cdot \mathrm{SP\_DISCORD2} +0.39090449 \cdot \mathrm{SP\_A\_PRIVATE3} +0.20496311 \cdot \mathrm{SP\_A\_PRIVATE4} +0.12360738 \cdot \mathrm{SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### PHQ_total: full / X_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 4.9015858 -0.016654379 \cdot \mathrm{RIDAGEYR} +0.031978952 \cdot \mathrm{BMXBMI} -0.28151427 \cdot \mathrm{INDFMPIR} -0.0014317468 \cdot \mathrm{EGFR\_2021} +0.80944168 \cdot \mathrm{RIAGENDR=2} +0.93418435 \cdot \mathrm{RACE=2} +0.72759286 \cdot \mathrm{RACE=3} +0.68883754 \cdot \mathrm{RACE=4} +0.61988341 \cdot \mathrm{RACE=5} -0.34581886 \cdot \mathrm{EDUC3=2} -0.48477394 \cdot \mathrm{EDUC3=3} +0.37298558 \cdot \mathrm{SMOKING3=1} +1.2949389 \cdot \mathrm{SMOKING3=2} +0.42743553 \cdot \mathrm{LBXHGB} -1.8641549 \cdot \mathrm{LBXRBCSI} -0.067970481 \cdot \mathrm{LBXMCVSI} +0.3576256 \cdot \mathrm{LBXRDW} +0.184038 \cdot \mathrm{LBXGH} +0.0016615891 \cdot \mathrm{LBXGLU} +0.2970213 \cdot \mathrm{LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### PHQ_total: full / X_AxG / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 1.9588168 -0.015227663 \cdot \mathrm{RIDAGEYR} +0.03474547 \cdot \mathrm{BMXBMI} -0.27770032 \cdot \mathrm{INDFMPIR} -0.00076823955 \cdot \mathrm{EGFR\_2021} +0.78335829 \cdot \mathrm{RIAGENDR=2} +0.82905102 \cdot \mathrm{RACE=2} +0.6417111 \cdot \mathrm{RACE=3} +0.55163261 \cdot \mathrm{RACE=4} +0.50367239 \cdot \mathrm{RACE=5} -0.33765224 \cdot \mathrm{EDUC3=2} -0.46549935 \cdot \mathrm{EDUC3=3} +0.34342067 \cdot \mathrm{SMOKING3=1} +1.2785848 \cdot \mathrm{SMOKING3=2} +0.18250117 \cdot \mathrm{LBXHGB} -1.1419966 \cdot \mathrm{LBXRBCSI} -0.026485599 \cdot \mathrm{LBXMCVSI} +0.31580477 \cdot \mathrm{LBXRDW} +0.14129684 \cdot \mathrm{LBXGH} +0.0020535978 \cdot \mathrm{LBXGLU} +0.27094441 \cdot \mathrm{LOG\_IN} +0.0071094995 \cdot \mathrm{LBXHGB*LBXGH} +0.10998758 \cdot \mathrm{LBXHGB*LBXGLU} +0.20795656 \cdot \mathrm{LBXHGB*LOG\_IN} -0.046968039 \cdot \mathrm{LBXRBCSI*LBXGH} +0.027350842 \cdot \mathrm{LBXRBCSI*LBXGLU} -0.20122899 \cdot \mathrm{LBXRBCSI*LOG\_IN} -0.13111072 \cdot \mathrm{LBXMCVSI*LBXGH} +0.091829381 \cdot \mathrm{LBXMCVSI*LBXGLU} +0.026721038 \cdot \mathrm{LBXMCVSI*LOG\_IN} +0.10236499 \cdot \mathrm{LBXRDW*LBXGH} -0.076885955 \cdot \mathrm{LBXRDW*LBXGLU} +0.050622273 \cdot \mathrm{LBXRDW*LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### PHQ_total: full / Z_pairwise / ElasticNet

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 2.4495309 -0.008371611 \cdot \mathrm{RIDAGEYR} +0.0289734 \cdot \mathrm{BMXBMI} -0.26657573 \cdot \mathrm{INDFMPIR} +0.80620004 \cdot \mathrm{RIAGENDR=2} +0.26448168 \cdot \mathrm{RACE=2} +0.037298937 \cdot \mathrm{RACE=3} +0.060789555 \cdot \mathrm{RACE=4} -0.14869253 \cdot \mathrm{EDUC3=3} +0.21724699 \cdot \mathrm{SMOKING3=1} +1.2044302 \cdot \mathrm{SMOKING3=2} +0.27772636 \cdot \mathrm{SP\_SHARED1} -0.054974267 \cdot \mathrm{SP\_SHARED2} -0.040873001 \cdot \mathrm{SP\_DISCORD1} -0.141135 \cdot \mathrm{SP\_DISCORD2} +0.41350903 \cdot \mathrm{SP\_A\_PRIVATE3} +0.15539376 \cdot \mathrm{SP\_A\_PRIVATE4} +0.033384861 \cdot \mathrm{SP\_G\_PRIVATE3} +0.0025995464 \cdot \mathrm{SP\_SHARED1*SP\_DISCORD1} +0.084405688 \cdot \mathrm{SP\_SHARED1*SP\_DISCORD2} +0.010359901 \cdot \mathrm{SP\_SHARED1*SP\_G\_PRIVATE3} -0.083268093 \cdot \mathrm{SP\_SHARED2*SP\_DISCORD2} -0.071397645 \cdot \mathrm{SP\_SHARED2*SP\_A\_PRIVATE4} +0.017097434 \cdot \mathrm{SP\_SHARED2*SP\_G\_PRIVATE3} +0.16794077 \cdot \mathrm{SP\_DISCORD1*SP\_DISCORD2} -0.00027491441 \cdot \mathrm{SP\_DISCORD1*SP\_A\_PRIVATE3} -0.0016435234 \cdot \mathrm{SP\_DISCORD2*SP\_A\_PRIVATE3} -0.0083293121 \cdot \mathrm{SP\_DISCORD2*SP\_A\_PRIVATE4} +0.012450281 \cdot \mathrm{SP\_A\_PRIVATE3*SP\_A\_PRIVATE4} -0.024616307 \cdot \mathrm{SP\_A\_PRIVATE4*SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### Somatic: lean / covariates / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 0.59509164 -0.0017242734 \cdot \mathrm{RIDAGEYR} +0.028789064 \cdot \mathrm{BMXBMI} +0.52013728 \cdot \mathrm{RIAGENDR=2}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### Somatic: lean / Z_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 1.3770411 -0.0090935799 \cdot \mathrm{RIDAGEYR} +0.015558428 \cdot \mathrm{BMXBMI} +0.42177783 \cdot \mathrm{RIAGENDR=2} +0.12710675 \cdot \mathrm{SP\_SHARED1} -0.06685883 \cdot \mathrm{SP\_SHARED2} -0.092492964 \cdot \mathrm{SP\_DISCORD1} -0.11303292 \cdot \mathrm{SP\_DISCORD2} +0.26861977 \cdot \mathrm{SP\_A\_PRIVATE3} +0.035236522 \cdot \mathrm{SP\_A\_PRIVATE4} +0.081045341 \cdot \mathrm{SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### Somatic: lean / X_main / ElasticNet

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = -1.7696909 -0.0093158496 \cdot \mathrm{RIDAGEYR} +0.015431164 \cdot \mathrm{BMXBMI} +0.41490788 \cdot \mathrm{RIAGENDR=2} +0.13908663 \cdot \mathrm{LBXHGB} -0.65644712 \cdot \mathrm{LBXRBCSI} -0.00079314929 \cdot \mathrm{LBXMCVSI} +0.25410727 \cdot \mathrm{LBXRDW} +0.12817351 \cdot \mathrm{LBXGH} +0.00063850953 \cdot \mathrm{LBXGLU} +0.15342133 \cdot \mathrm{LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### Somatic: lean / X_AxG / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = -1.9870859 -0.0092047869 \cdot \mathrm{RIDAGEYR} +0.017115158 \cdot \mathrm{BMXBMI} +0.42053404 \cdot \mathrm{RIAGENDR=2} +0.11292082 \cdot \mathrm{LBXHGB} -0.58009386 \cdot \mathrm{LBXRBCSI} +0.0037569632 \cdot \mathrm{LBXMCVSI} +0.24231242 \cdot \mathrm{LBXRDW} +0.11540252 \cdot \mathrm{LBXGH} +0.00093859925 \cdot \mathrm{LBXGLU} +0.14047895 \cdot \mathrm{LOG\_IN} +0.10281809 \cdot \mathrm{LBXHGB*LBXGH} +0.052509576 \cdot \mathrm{LBXHGB*LBXGLU} +0.031156113 \cdot \mathrm{LBXHGB*LOG\_IN} -0.043740152 \cdot \mathrm{LBXRBCSI*LBXGH} -0.054570399 \cdot \mathrm{LBXRBCSI*LBXGLU} -0.083985285 \cdot \mathrm{LBXRBCSI*LOG\_IN} -0.087216102 \cdot \mathrm{LBXMCVSI*LBXGH} +0.022195723 \cdot \mathrm{LBXMCVSI*LBXGLU} +0.040795827 \cdot \mathrm{LBXMCVSI*LOG\_IN} +0.096860397 \cdot \mathrm{LBXRDW*LBXGH} -0.035989849 \cdot \mathrm{LBXRDW*LBXGLU} -0.023347103 \cdot \mathrm{LBXRDW*LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### Somatic: lean / Z_pairwise / ElasticNet

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 1.3102367 -0.0063291629 \cdot \mathrm{RIDAGEYR} +0.013233999 \cdot \mathrm{BMXBMI} +0.42601512 \cdot \mathrm{RIAGENDR=2} +0.13964703 \cdot \mathrm{SP\_SHARED1} -0.054273231 \cdot \mathrm{SP\_SHARED2} -0.047141313 \cdot \mathrm{SP\_DISCORD1} -0.071225292 \cdot \mathrm{SP\_DISCORD2} +0.27011388 \cdot \mathrm{SP\_A\_PRIVATE3} +0.014455328 \cdot \mathrm{SP\_A\_PRIVATE4} +0.029312592 \cdot \mathrm{SP\_G\_PRIVATE3} +0.032650109 \cdot \mathrm{SP\_SHARED1*SP\_DISCORD2} -0.033593363 \cdot \mathrm{SP\_SHARED2*SP\_DISCORD2} -0.019300976 \cdot \mathrm{SP\_SHARED2*SP\_A\_PRIVATE4} +0.018269655 \cdot \mathrm{SP\_SHARED2*SP\_G\_PRIVATE3} +0.07556151 \cdot \mathrm{SP\_DISCORD1*SP\_DISCORD2} -0.013909889 \cdot \mathrm{SP\_DISCORD1*SP\_A\_PRIVATE3} -0.027503701 \cdot \mathrm{SP\_DISCORD2*SP\_A\_PRIVATE4}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### Somatic: full / covariates / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 0.99076121 -0.0040045352 \cdot \mathrm{RIDAGEYR} +0.02976004 \cdot \mathrm{BMXBMI} -0.1351021 \cdot \mathrm{INDFMPIR} -0.0028019668 \cdot \mathrm{EGFR\_2021} +0.55571549 \cdot \mathrm{RIAGENDR=2} +0.32693012 \cdot \mathrm{RACE=2} +0.33288368 \cdot \mathrm{RACE=3} +0.31560386 \cdot \mathrm{RACE=4} +0.25971665 \cdot \mathrm{RACE=5} -0.14510446 \cdot \mathrm{EDUC3=2} -0.19552642 \cdot \mathrm{EDUC3=3} +0.24139164 \cdot \mathrm{SMOKING3=1} +0.61251383 \cdot \mathrm{SMOKING3=2}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### Somatic: full / Z_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 1.666809 -0.0099200941 \cdot \mathrm{RIDAGEYR} +0.016453705 \cdot \mathrm{BMXBMI} -0.11649383 \cdot \mathrm{INDFMPIR} -0.0032580164 \cdot \mathrm{EGFR\_2021} +0.40430455 \cdot \mathrm{RIAGENDR=2} +0.33230788 \cdot \mathrm{RACE=2} +0.36777641 \cdot \mathrm{RACE=3} +0.26127966 \cdot \mathrm{RACE=4} +0.28290097 \cdot \mathrm{RACE=5} -0.10464394 \cdot \mathrm{EDUC3=2} -0.16359158 \cdot \mathrm{EDUC3=3} +0.24445535 \cdot \mathrm{SMOKING3=1} +0.6152668 \cdot \mathrm{SMOKING3=2} +0.13648247 \cdot \mathrm{SP\_SHARED1} -0.053038248 \cdot \mathrm{SP\_SHARED2} -0.068456166 \cdot \mathrm{SP\_DISCORD1} -0.12671945 \cdot \mathrm{SP\_DISCORD2} +0.20458261 \cdot \mathrm{SP\_A\_PRIVATE3} +0.098475627 \cdot \mathrm{SP\_A\_PRIVATE4} +0.061458629 \cdot \mathrm{SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### Somatic: full / X_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 1.0582303 -0.0098804471 \cdot \mathrm{RIDAGEYR} +0.01641882 \cdot \mathrm{BMXBMI} -0.1169486 \cdot \mathrm{INDFMPIR} -0.0030970747 \cdot \mathrm{EGFR\_2021} +0.38932657 \cdot \mathrm{RIAGENDR=2} +0.32303725 \cdot \mathrm{RACE=2} +0.36906915 \cdot \mathrm{RACE=3} +0.23075329 \cdot \mathrm{RACE=4} +0.27480902 \cdot \mathrm{RACE=5} -0.10620801 \cdot \mathrm{EDUC3=2} -0.16186726 \cdot \mathrm{EDUC3=3} +0.24144755 \cdot \mathrm{SMOKING3=1} +0.62109927 \cdot \mathrm{SMOKING3=2} +0.074967166 \cdot \mathrm{LBXHGB} -0.56693628 \cdot \mathrm{LBXRBCSI} -0.010686661 \cdot \mathrm{LBXMCVSI} +0.17603106 \cdot \mathrm{LBXRDW} +0.080827962 \cdot \mathrm{LBXGH} +0.0011840098 \cdot \mathrm{LBXGLU} +0.16629604 \cdot \mathrm{LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### Somatic: full / X_AxG / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 0.94352626 -0.0098798053 \cdot \mathrm{RIDAGEYR} +0.018121199 \cdot \mathrm{BMXBMI} -0.11690113 \cdot \mathrm{INDFMPIR} -0.0030164741 \cdot \mathrm{EGFR\_2021} +0.3996373 \cdot \mathrm{RIAGENDR=2} +0.33029263 \cdot \mathrm{RACE=2} +0.37872638 \cdot \mathrm{RACE=3} +0.2348698 \cdot \mathrm{RACE=4} +0.28239775 \cdot \mathrm{RACE=5} -0.11579736 \cdot \mathrm{EDUC3=2} -0.16703597 \cdot \mathrm{EDUC3=3} +0.23681639 \cdot \mathrm{SMOKING3=1} +0.61091036 \cdot \mathrm{SMOKING3=2} +0.072587444 \cdot \mathrm{LBXHGB} -0.55793893 \cdot \mathrm{LBXRBCSI} -0.0089255431 \cdot \mathrm{LBXMCVSI} +0.17234684 \cdot \mathrm{LBXRDW} +0.077944023 \cdot \mathrm{LBXGH} +0.0010264566 \cdot \mathrm{LBXGLU} +0.15547103 \cdot \mathrm{LOG\_IN} +0.078277197 \cdot \mathrm{LBXHGB*LBXGH} +0.066673811 \cdot \mathrm{LBXHGB*LBXGLU} +0.049735686 \cdot \mathrm{LBXHGB*LOG\_IN} -0.032504104 \cdot \mathrm{LBXRBCSI*LBXGH} -0.049831495 \cdot \mathrm{LBXRBCSI*LBXGLU} -0.083334769 \cdot \mathrm{LBXRBCSI*LOG\_IN} -0.088210197 \cdot \mathrm{LBXMCVSI*LBXGH} +0.024333189 \cdot \mathrm{LBXMCVSI*LBXGLU} +0.037923521 \cdot \mathrm{LBXMCVSI*LOG\_IN} +0.07372829 \cdot \mathrm{LBXRDW*LBXGH} -0.02594739 \cdot \mathrm{LBXRDW*LBXGLU} -0.015429083 \cdot \mathrm{LBXRDW*LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### Somatic: full / Z_pairwise / ElasticNet

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 1.280576 -0.0033499051 \cdot \mathrm{RIDAGEYR} +0.014895982 \cdot \mathrm{BMXBMI} -0.10756342 \cdot \mathrm{INDFMPIR} +0.40639736 \cdot \mathrm{RIAGENDR=2} +0.060849765 \cdot \mathrm{RACE=3} -0.033168458 \cdot \mathrm{EDUC3=3} +0.14705566 \cdot \mathrm{SMOKING3=1} +0.54866207 \cdot \mathrm{SMOKING3=2} +0.13952468 \cdot \mathrm{SP\_SHARED1} -0.040097312 \cdot \mathrm{SP\_SHARED2} -0.023068142 \cdot \mathrm{SP\_DISCORD1} -0.085851372 \cdot \mathrm{SP\_DISCORD2} +0.20248644 \cdot \mathrm{SP\_A\_PRIVATE3} +0.067235076 \cdot \mathrm{SP\_A\_PRIVATE4} +0.011491562 \cdot \mathrm{SP\_G\_PRIVATE3} +0.042301702 \cdot \mathrm{SP\_SHARED1*SP\_DISCORD2} -0.028553737 \cdot \mathrm{SP\_SHARED2*SP\_DISCORD2} -0.019070813 \cdot \mathrm{SP\_SHARED2*SP\_A\_PRIVATE4} +0.012632029 \cdot \mathrm{SP\_SHARED2*SP\_G\_PRIVATE3} +0.072474498 \cdot \mathrm{SP\_DISCORD1*SP\_DISCORD2} -0.0035128098 \cdot \mathrm{SP\_DISCORD1*SP\_A\_PRIVATE3} -0.017155555 \cdot \mathrm{SP\_DISCORD2*SP\_A\_PRIVATE4}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### CogAff: lean / covariates / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 0.34322735 -0.0044872161 \cdot \mathrm{RIDAGEYR} +0.026721651 \cdot \mathrm{BMXBMI} +0.4711274 \cdot \mathrm{RIAGENDR=2}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### CogAff: lean / Z_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 1.0887059 -0.011356882 \cdot \mathrm{RIDAGEYR} +0.0133634 \cdot \mathrm{BMXBMI} +0.4046292 \cdot \mathrm{RIAGENDR=2} +0.14300038 \cdot \mathrm{SP\_SHARED1} -0.04490609 \cdot \mathrm{SP\_SHARED2} -0.078580087 \cdot \mathrm{SP\_DISCORD1} -0.072837494 \cdot \mathrm{SP\_DISCORD2} +0.26502176 \cdot \mathrm{SP\_A\_PRIVATE3} +0.044836668 \cdot \mathrm{SP\_A\_PRIVATE4} +0.098606547 \cdot \mathrm{SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### CogAff: lean / X_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = -0.25488665 -0.011945206 \cdot \mathrm{RIDAGEYR} +0.013244367 \cdot \mathrm{BMXBMI} +0.41464109 \cdot \mathrm{RIAGENDR=2} +0.28446362 \cdot \mathrm{LBXHGB} -1.0155214 \cdot \mathrm{LBXRBCSI} -0.028069299 \cdot \mathrm{LBXMCVSI} +0.26511679 \cdot \mathrm{LBXRDW} +0.16230113 \cdot \mathrm{LBXGH} +0.00039093652 \cdot \mathrm{LBXGLU} +0.1277065 \cdot \mathrm{LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### CogAff: lean / X_AxG / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = -1.7954022 -0.011362927 \cdot \mathrm{RIDAGEYR} +0.014176384 \cdot \mathrm{BMXBMI} +0.40077091 \cdot \mathrm{RIAGENDR=2} +0.14626276 \cdot \mathrm{LBXHGB} -0.60557533 \cdot \mathrm{LBXRBCSI} -0.0050112507 \cdot \mathrm{LBXMCVSI} +0.23588557 \cdot \mathrm{LBXRDW} +0.12733416 \cdot \mathrm{LBXGH} +0.0010049251 \cdot \mathrm{LBXGLU} +0.11484692 \cdot \mathrm{LOG\_IN} -0.043541692 \cdot \mathrm{LBXHGB*LBXGH} +0.023511876 \cdot \mathrm{LBXHGB*LBXGLU} +0.13934305 \cdot \mathrm{LBXHGB*LOG\_IN} -0.020834875 \cdot \mathrm{LBXRBCSI*LBXGH} +0.068971779 \cdot \mathrm{LBXRBCSI*LBXGLU} -0.12326679 \cdot \mathrm{LBXRBCSI*LOG\_IN} -0.050068137 \cdot \mathrm{LBXMCVSI*LBXGH} +0.073045048 \cdot \mathrm{LBXMCVSI*LBXGLU} -0.014921328 \cdot \mathrm{LBXMCVSI*LOG\_IN} +0.051419371 \cdot \mathrm{LBXRDW*LBXGH} -0.062468472 \cdot \mathrm{LBXRDW*LBXGLU} +0.054272617 \cdot \mathrm{LBXRDW*LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### CogAff: lean / Z_pairwise / ElasticNet

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 1.0246472 -0.0084346999 \cdot \mathrm{RIDAGEYR} +0.010735408 \cdot \mathrm{BMXBMI} +0.40626815 \cdot \mathrm{RIAGENDR=2} +0.13882466 \cdot \mathrm{SP\_SHARED1} -0.025577104 \cdot \mathrm{SP\_SHARED2} -0.043586321 \cdot \mathrm{SP\_DISCORD1} -0.037897786 \cdot \mathrm{SP\_DISCORD2} +0.26595154 \cdot \mathrm{SP\_A\_PRIVATE3} +0.011566585 \cdot \mathrm{SP\_A\_PRIVATE4} +0.047210525 \cdot \mathrm{SP\_G\_PRIVATE3} +0.0010602179 \cdot \mathrm{SP\_SHARED1*SP\_DISCORD1} +0.011982862 \cdot \mathrm{SP\_SHARED1*SP\_DISCORD2} -0.046264404 \cdot \mathrm{SP\_SHARED2*SP\_DISCORD2} -0.042807323 \cdot \mathrm{SP\_SHARED2*SP\_A\_PRIVATE4} +0.0076056074 \cdot \mathrm{SP\_SHARED2*SP\_G\_PRIVATE3} +0.069118588 \cdot \mathrm{SP\_DISCORD1*SP\_DISCORD2} -0.0030842343 \cdot \mathrm{SP\_DISCORD1*SP\_A\_PRIVATE3} -0.00082058922 \cdot \mathrm{SP\_DISCORD2*SP\_A\_PRIVATE3} +0.020656594 \cdot \mathrm{SP\_A\_PRIVATE3*SP\_A\_PRIVATE4} -0.023627145 \cdot \mathrm{SP\_A\_PRIVATE3*SP\_G\_PRIVATE3} -0.027504305 \cdot \mathrm{SP\_A\_PRIVATE4*SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### CogAff: full / covariates / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 0.25558932 -0.00055704613 \cdot \mathrm{RIDAGEYR} +0.026986476 \cdot \mathrm{BMXBMI} -0.17626003 \cdot \mathrm{INDFMPIR} +0.0027883947 \cdot \mathrm{EGFR\_2021} +0.49904208 \cdot \mathrm{RIAGENDR=2} +0.50179115 \cdot \mathrm{RACE=2} +0.23238617 \cdot \mathrm{RACE=3} +0.37373156 \cdot \mathrm{RACE=4} +0.20648635 \cdot \mathrm{RACE=5} -0.23812785 \cdot \mathrm{EDUC3=2} -0.31432685 \cdot \mathrm{EDUC3=3} +0.096123079 \cdot \mathrm{SMOKING3=1} +0.64589961 \cdot \mathrm{SMOKING3=2}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### CogAff: full / Z_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 0.83721595 -0.0054620692 \cdot \mathrm{RIDAGEYR} +0.015563738 \cdot \mathrm{BMXBMI} -0.16021725 \cdot \mathrm{INDFMPIR} +0.0021238618 \cdot \mathrm{EGFR\_2021} +0.3902684 \cdot \mathrm{RIAGENDR=2} +0.51203331 \cdot \mathrm{RACE=2} +0.26328342 \cdot \mathrm{RACE=3} +0.35064226 \cdot \mathrm{RACE=4} +0.23782635 \cdot \mathrm{RACE=5} -0.20334469 \cdot \mathrm{EDUC3=2} -0.28818302 \cdot \mathrm{EDUC3=3} +0.10437027 \cdot \mathrm{SMOKING3=1} +0.65051347 \cdot \mathrm{SMOKING3=2} +0.11941624 \cdot \mathrm{SP\_SHARED1} -0.021126662 \cdot \mathrm{SP\_SHARED2} -0.053756749 \cdot \mathrm{SP\_DISCORD1} -0.075682704 \cdot \mathrm{SP\_DISCORD2} +0.18632188 \cdot \mathrm{SP\_A\_PRIVATE3} +0.10648748 \cdot \mathrm{SP\_A\_PRIVATE4} +0.06214875 \cdot \mathrm{SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### CogAff: full / X_main / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 2.7331631 -0.0058361836 \cdot \mathrm{RIDAGEYR} +0.015586767 \cdot \mathrm{BMXBMI} -0.16303127 \cdot \mathrm{INDFMPIR} +0.0021635935 \cdot \mathrm{EGFR\_2021} +0.39827179 \cdot \mathrm{RIAGENDR=2} +0.55803082 \cdot \mathrm{RACE=2} +0.31235437 \cdot \mathrm{RACE=3} +0.39197752 \cdot \mathrm{RACE=4} +0.2872301 \cdot \mathrm{RACE=5} -0.22587441 \cdot \mathrm{EDUC3=2} -0.30742619 \cdot \mathrm{EDUC3=3} +0.11391758 \cdot \mathrm{SMOKING3=1} +0.66432006 \cdot \mathrm{SMOKING3=2} +0.25191617 \cdot \mathrm{LBXHGB} -1.0045315 \cdot \mathrm{LBXRBCSI} -0.041878387 \cdot \mathrm{LBXMCVSI} +0.165793 \cdot \mathrm{LBXRDW} +0.090375515 \cdot \mathrm{LBXGH} +0.00076045685 \cdot \mathrm{LBXGLU} +0.12743874 \cdot \mathrm{LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### CogAff: full / X_AxG / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 1.0152906 -0.0053478574 \cdot \mathrm{RIDAGEYR} +0.016624271 \cdot \mathrm{BMXBMI} -0.16079919 \cdot \mathrm{INDFMPIR} +0.0022482345 \cdot \mathrm{EGFR\_2021} +0.383721 \cdot \mathrm{RIAGENDR=2} +0.49875839 \cdot \mathrm{RACE=2} +0.26298471 \cdot \mathrm{RACE=3} +0.31676281 \cdot \mathrm{RACE=4} +0.22127464 \cdot \mathrm{RACE=5} -0.22185488 \cdot \mathrm{EDUC3=2} -0.29846337 \cdot \mathrm{EDUC3=3} +0.10660429 \cdot \mathrm{SMOKING3=1} +0.66767447 \cdot \mathrm{SMOKING3=2} +0.10991373 \cdot \mathrm{LBXHGB} -0.58405763 \cdot \mathrm{LBXRBCSI} -0.017560056 \cdot \mathrm{LBXMCVSI} +0.14345793 \cdot \mathrm{LBXRDW} +0.063352818 \cdot \mathrm{LBXGH} +0.0010271412 \cdot \mathrm{LBXGLU} +0.11547338 \cdot \mathrm{LOG\_IN} -0.071167697 \cdot \mathrm{LBXHGB*LBXGH} +0.043313769 \cdot \mathrm{LBXHGB*LBXGLU} +0.15822087 \cdot \mathrm{LBXHGB*LOG\_IN} -0.014463935 \cdot \mathrm{LBXRBCSI*LBXGH} +0.077182338 \cdot \mathrm{LBXRBCSI*LBXGLU} -0.11789422 \cdot \mathrm{LBXRBCSI*LOG\_IN} -0.042900522 \cdot \mathrm{LBXMCVSI*LBXGH} +0.067496192 \cdot \mathrm{LBXMCVSI*LBXGLU} -0.011202483 \cdot \mathrm{LBXMCVSI*LOG\_IN} +0.028636696 \cdot \mathrm{LBXRDW*LBXGH} -0.050938565 \cdot \mathrm{LBXRDW*LBXGLU} +0.066051356 \cdot \mathrm{LBXRDW*LOG\_IN}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

### CogAff: full / Z_pairwise / Ridge

Model equation on ORIGINAL feature basis (not z-scored in this display):

$$
\hat{Y} = 0.89136548 -0.003638192 \cdot \mathrm{RIDAGEYR} +0.014369781 \cdot \mathrm{BMXBMI} -0.13809786 \cdot \mathrm{INDFMPIR} +0.0021710626 \cdot \mathrm{EGFR\_2021} +0.31693856 \cdot \mathrm{RIAGENDR=2} +0.32120213 \cdot \mathrm{RACE=2} +0.071105486 \cdot \mathrm{RACE=3} +0.19233933 \cdot \mathrm{RACE=4} +0.05787515 \cdot \mathrm{RACE=5} -0.10223742 \cdot \mathrm{EDUC3=2} -0.20376496 \cdot \mathrm{EDUC3=3} +0.045269668 \cdot \mathrm{SMOKING3=1} +0.52756884 \cdot \mathrm{SMOKING3=2} +0.10333626 \cdot \mathrm{SP\_SHARED1} -0.039135706 \cdot \mathrm{SP\_SHARED2} -0.051477747 \cdot \mathrm{SP\_DISCORD1} -0.047642494 \cdot \mathrm{SP\_DISCORD2} +0.17316939 \cdot \mathrm{SP\_A\_PRIVATE3} +0.079599024 \cdot \mathrm{SP\_A\_PRIVATE4} +0.022124569 \cdot \mathrm{SP\_G\_PRIVATE3} +0.026108948 \cdot \mathrm{SP\_SHARED1*SP\_SHARED2} +0.024419908 \cdot \mathrm{SP\_SHARED1*SP\_DISCORD1} +0.037234347 \cdot \mathrm{SP\_SHARED1*SP\_DISCORD2} +0.027241483 \cdot \mathrm{SP\_SHARED1*SP\_A\_PRIVATE3} -0.0032219181 \cdot \mathrm{SP\_SHARED1*SP\_A\_PRIVATE4} +0.0097632265 \cdot \mathrm{SP\_SHARED1*SP\_G\_PRIVATE3} +0.0035066011 \cdot \mathrm{SP\_SHARED2*SP\_DISCORD1} -0.041128975 \cdot \mathrm{SP\_SHARED2*SP\_DISCORD2} +0.0093771009 \cdot \mathrm{SP\_SHARED2*SP\_A\_PRIVATE3} -0.057786066 \cdot \mathrm{SP\_SHARED2*SP\_A\_PRIVATE4} +0.011528222 \cdot \mathrm{SP\_SHARED2*SP\_G\_PRIVATE3} +0.05764237 \cdot \mathrm{SP\_DISCORD1*SP\_DISCORD2} -0.017107604 \cdot \mathrm{SP\_DISCORD1*SP\_A\_PRIVATE3} -0.014902275 \cdot \mathrm{SP\_DISCORD1*SP\_A\_PRIVATE4} -0.0067287375 \cdot \mathrm{SP\_DISCORD1*SP\_G\_PRIVATE3} -0.031056036 \cdot \mathrm{SP\_DISCORD2*SP\_A\_PRIVATE3} -0.0040861704 \cdot \mathrm{SP\_DISCORD2*SP\_A\_PRIVATE4} -0.0056769138 \cdot \mathrm{SP\_DISCORD2*SP\_G\_PRIVATE3} +0.021446937 \cdot \mathrm{SP\_A\_PRIVATE3*SP\_A\_PRIVATE4} -0.036914177 \cdot \mathrm{SP\_A\_PRIVATE3*SP\_G\_PRIVATE3} -0.045691021 \cdot \mathrm{SP\_A\_PRIVATE4*SP\_G\_PRIVATE3}
$$

Categorical terms `VAR=level` are 0/1 indicators; omitted category is the discovery reference.
Interaction feature `u*v` is `((u−train_mean_u)/train_sd_u)×((v−train_mean_v)/train_sd_v)`.
The intercept is the value when all included original features equal 0; it is not a biological baseline.

## Interpretation

- A model coefficient is an adjusted association/conditional model sensitivity, not a causal effect.
- Model contribution magnitudes depend on centering, feature basis and penalty; do not label them biological percentages.
- A combined Z representation and raw X include the same underlying information, not independent modalities.
- Later NHANES cycles are distinct participants, not repeated measurements of the same individuals.
- Complex NHANES survey-design standard errors and bootstrap confidence intervals are not supplied here.
