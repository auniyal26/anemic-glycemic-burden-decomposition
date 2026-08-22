
args <- commandArgs(trailingOnly=TRUE)
input <- args[1]
output <- args[2]
userlib <- args[3]
.libPaths(c(userlib,.libPaths()))
library(survey)
options(survey.lonely.psu="adjust")

d <- read.csv(input, stringsAsFactors=FALSE)
d$GROUP <- ifelse(d$A<=0 & d$G_HBA1C<=0, "Neither",
           ifelse(d$A<=0 & d$G_HBA1C>0, "G only",
           ifelse(d$A>0 & d$G_HBA1C<=0, "A only", "Both")))
d$GROUP <- factor(d$GROUP, levels=c("Neither","G only","A only","Both"))
d$PHQ_GE10 <- as.numeric(d$PHQ9_TOTAL >= 10)

des <- svydesign(
  ids=~PSU,
  strata=~STRATUM,
  weights=~WTMEC4YR,
  nest=TRUE,
  data=d
)

out <- list()
for (g in levels(d$GROUP)) {
  sg <- subset(des, GROUP==g)
  raw_n <- sum(d$GROUP==g, na.rm=TRUE)

  m1 <- svymean(~PHQ9_TOTAL, sg, na.rm=TRUE)
  c1 <- confint(m1)

  m2 <- svymean(~SOMATIC_SCORE, sg, na.rm=TRUE)
  c2 <- confint(m2)

  m3 <- svymean(~PHQ_GE10, sg, na.rm=TRUE)
  c3 <- confint(m3)

  out[[length(out)+1]] <- data.frame(
    group=g,
    unweighted_n=raw_n,
    weighted_mean_PHQ9=coef(m1)[1],
    phq_ci_low=c1[1,1],
    phq_ci_high=c1[1,2],
    weighted_mean_somatic=coef(m2)[1],
    somatic_ci_low=c2[1,1],
    somatic_ci_high=c2[1,2],
    weighted_PHQ9_ge10_pct=100*coef(m3)[1],
    phq10_ci_low_pct=100*c3[1,1],
    phq10_ci_high_pct=100*c3[1,2]
  )
}
write.csv(do.call(rbind,out), output, row.names=FALSE)
