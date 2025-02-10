
require(Rcpp)

## source("R_script_load_OMP_Linux.R")
Rcpp::sourceCpp("cpu_check.cpp")

features <- checkCPUFeatures()
has_avx <- features$has_avx
writeLines(as.character(as.integer(has_avx)))
