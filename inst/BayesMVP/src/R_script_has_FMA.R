

Rcpp::sourceCpp("cpu_check.cpp")


features <- checkCPUFeatures(); 
has_fma = features$has_fma
cat(has_fma)
