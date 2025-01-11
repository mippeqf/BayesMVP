require(Rcpp)

Rcpp::sourceCpp("C:\\Users\\enzoc\\Documents\\BayesMVP\\inst\\BayesMVP\\src\\cpu_check.cpp")


features <- checkCPUFeatures(); 
features


has_avx = features$has_avx ; has_avx
has_avx2 = features$has_avx2 ; has_avx2
has_avx512 = features$has_avx512 ; has_avx512