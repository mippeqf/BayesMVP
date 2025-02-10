
require(Rcpp)

## source("R_script_load_OMP_Linux.R")
Rcpp::sourceCpp("cpu_check.cpp")

features <- checkCPUFeatures(); 
has_avx512 <- features$has_avx512
writeLines(as.character(as.integer(has_avx512)))
 