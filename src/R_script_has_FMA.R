
require(Rcpp)

# source("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP/src/R_script_load_OMP_Linux.R")
# Rcpp::sourceCpp("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP/src/cpu_check.cpp")

## source("R_script_load_OMP_Linux.R")
Rcpp::sourceCpp("cpu_check.cpp")

features <- checkCPUFeatures(); 
has_fma <- features$has_fma
writeLines(as.character(as.integer(has_fma)))
