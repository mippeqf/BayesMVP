require(Rcpp)

Rcpp::sourceCpp("cpu_check.cpp")


features <- checkCPUFeatures(); 
has_avx2 = features$has_avx2
cat(has_avx2)
