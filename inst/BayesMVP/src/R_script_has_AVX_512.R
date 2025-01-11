

Rcpp::sourceCpp("cpu_check.cpp")


features <- checkCPUFeatures(); 
has_avx512 = features$has_avx512
cat(has_avx512)
