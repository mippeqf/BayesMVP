
#pragma once


// [[Rcpp::depends(StanHeaders)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppEigen)]]

 
 
 
#include <Eigen/Dense>
 

 
 
 
 
 
using namespace Eigen;
 




#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE






inline bool is_NaN_or_Inf_Eigen(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> mat) {
  
  if (!((mat.array() == mat.array()).all())) {
    return true;
  }   
  
  if ((mat.array().isInf()).any()) {
    return true;
  }   
  
  return false; // if no NaN or Inf values  
  
}  




 





























