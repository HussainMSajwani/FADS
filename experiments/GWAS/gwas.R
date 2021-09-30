library(bigsnpr)

for(h2s in c("0.3", "0.5", "0.7")){
  for(j in 1:25){
    path2files <- str_interp("/home/shussain/Simulated_data/29072021/${h2s}/sim_${j}/sim_${j}")
    rds <- snp_readBed(bedfile = str_c(path2files, "/prelim/prelim_plink_chr22.bed"))
    bed <- snp_attach(rds)
    
    pheno <- read_csv(str_c(path2files, "/PS/output/Ysim_output.csv"))
    
    Y_bin <- 1*(pheno$Trait_1 > 0)
    
    betas <- c()
    betas_se <- c()
    ps <- c()
    
    for(i in 1:dim(bed$genotypes)[2]){
      lr <- glm(Y_bin ~ bed$genotypes[, i], family = "binomial")
      
      if(dim(coef(summary(lr)))[1] > 1){
        beta <- coef(summary(lr))[2,1]
        beta_se <- coef(summary(lr))[2,2]
        p <- coef(summary(lr))[2,3]
      } else {
        beta <- NA
        beta_se <- NA
        p <- NA
      }
      
      betas[i] <- beta
      betas_se[i] <- beta_se
      ps[i] <- p
    })
  }
}

