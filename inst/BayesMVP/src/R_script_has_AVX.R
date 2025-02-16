
require(Rcpp)

Rcpp::sourceCpp("cpu_check.cpp")

features <- checkCPUFeatures()
has_avx  <- features$has_avx
writeLines(as.character(as.integer(has_avx)))
