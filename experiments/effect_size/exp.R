library(PhenotypeSimulator)

# simulate simple bi-allelic genotypes and estimate kinship
genotypes <- readStandardGenotypes(N = 600, filename = "/home/shussain/Simulated_data/13072021/0.5/sim_10/prelim/prelim_plink_chr22", format = "plink")
genotypes_sd <- standardiseGenotypes(genotypes$genotypes)
kinship <- getKinship(N=600, X=genotypes_sd, verbose = FALSE)

genotypes <- simulateGenotypes(600, 1000)
genotypes_sd <- standardiseGenotypes(genotypes$genotypes)
kinship <- getKinship(N=600, X=genotypes_sd, verbose = FALSE)
# read genotypes from external file
# use one of the sample genotype file provided in the 
# extdata/genotypes/subfolders (e.g.extdata/genotypes/hapgen )
# simulate 30 genetic variant effects (from non-standardised SNP genotypes)
causalSNPs <- getCausalSNPs(N=600, genotypes = genotypes$genotypes, 
                            NrCausalSNPs = 5, verbose = FALSE)
genFixed <- geneticFixedEffects(N=600, P = 1, X_causal = causalSNPs, distBeta = "unif", mBeta = 1, sdBeta = 0.000005)  
# simulate infinitesimal genetic effects
genBg <- geneticBgEffects(N=600, kinship = kinship, P = 1)
# simulate 4 different confounder effects:
# * 1 binomial covariate effect shared across all traits
# * 2 categorical (3 categories) independent covariate effects
# * 1 categorical (4 categories) independent  covariate effect
# * 2 normally distributed independent and shared covariate effects
noiseFixed <- noiseFixedEffects(N=600, P = 1, NrFixedEffects = 4, 
                                NrConfounders = c(1, 2, 1, 2),
                                pIndependentConfounders = c(0, 1, 1, 0.5),  
                                distConfounders = c("bin", "cat_norm", 
                                                    "cat_unif", "norm"),
                                probConfounders = 0.2, 
                                catConfounders = c(3, 4))
# simulate correlated effects with max correlation of 0.8
correlatedBg <- correlatedBgEffects(N=600, P = 1, pcorr = 0.8)
# simulate observational noise effects
noiseBg <- noiseBgEffects(N=600, P = 1)
# total SNP effect on phenotype: 0.01
genVar <- 0.999
noiseVar <- 1 - genVar
totalSNPeffect <- 0.9
h2s <- totalSNPeffect/genVar
phi <- 0.6 
rho <- 0.1
delta <- 0.3
shared <- 0.01
independent <- 1 - shared
# rescale phenotype components
genFixed_shared_scaled <- rescaleVariance(genFixed$shared, shared * h2s *genVar)
genFixed_independent_scaled <- rescaleVariance(genFixed$independent, 
                                               independent * h2s *genVar)
genBg_shared_scaled <- rescaleVariance(genBg$shared, shared * (1-h2s) *genVar)
genBg_independent_scaled <- rescaleVariance(genBg$independent, 
                                            independent * (1-h2s) * genVar)
noiseBg_shared_scaled <- rescaleVariance(noiseBg$shared, shared * phi* noiseVar)
noiseBg_independent_scaled <- rescaleVariance(noiseBg$independent, 
                                              independent * phi* noiseVar)
correlatedBg_scaled <- rescaleVariance(correlatedBg$correlatedBg, 
                                       shared * rho * noiseVar)
noiseFixed_shared_scaled <- rescaleVariance(noiseFixed$shared, shared * delta * 
                                              noiseVar)
noiseFixed_independent_scaled <- rescaleVariance(noiseFixed$independent, 
                                                 independent * delta * noiseVar)
# Total variance proportions have to add up yo 1
total <- shared * h2s *genVar +  independent * h2s * genVar +
  shared * (1-h2s) * genVar +   independent * (1-h2s) * genVar +
  shared * phi* noiseVar +  independent * phi* noiseVar +
  rho * noiseVar +
  shared * delta * noiseVar +  independent * delta * noiseVar
total == 1
# combine components into final phenotype
Y <- scale(genBg_shared_scaled$component + noiseBg_shared_scaled$component +
             genBg_independent_scaled$component + noiseBg_independent_scaled$component +
             genFixed_shared_scaled$component + noiseFixed_shared_scaled$component +
             genFixed_independent_scaled$component + noiseFixed_independent_scaled$component +
             correlatedBg_scaled$component)

Y_bin <- Y > 0

pvals <- c()
for(i in 1:1000){
  pval <- coef(summary(glm(Y_bin ~ genotypes$genotypes[, i], family = "binomial")))[2, 4]
  pvals[i] <- -log10(pval)
}

ggplot() +
  geom_point(aes(1:1000, pvals))

