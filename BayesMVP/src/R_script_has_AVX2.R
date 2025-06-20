
require(Rcpp)

Rcpp::sourceCpp("cpu_check.cpp")

features <- checkCPUFeatures()
has_avx2 <- features$has_avx2
writeLines(as.character(as.integer(has_avx2)))
