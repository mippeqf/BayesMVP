#pragma once 


#ifndef FN_WRAPPERS_OVERALL
#define FN_WRAPPERS_OVERALL

 

  
#include <stan/math/prim/fun/sqrt.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/prob/std_normal_log_qf.hpp>
#include <stan/math/prim/fun/Phi.hpp>
#include <stan/math/prim/fun/inv_Phi.hpp>
#include <stan/math/prim/fun/Phi_approx.hpp>
#include <stan/math/prim/fun/tanh.hpp>
#include <stan/math/prim/fun/log_inv_logit.hpp>
 
#include <Eigen/Dense>
#include <Eigen/Core>
 
#include <immintrin.h>

 

 
using namespace Eigen;


 

 
 
 
  


// 
// void log_sum_exp_pair(const Eigen::Matrix<double, -1, 1>  &log_a,
//                       const Eigen::Matrix<double, -1, 1>  &log_b,
//                       const std::string &vect_type_exp,
//                       const std::string &vect_type_log,
//                       Eigen::Matrix<double, -1, 1> &log_sum_abs_result) {       // output parameter
//   
//   // for each element i, find max(log_a[i], log_b[i])
//   Eigen::Matrix<double, -1, 1> max_logs = log_a.array().max(log_b.array());
//   // for each element i, compute sign_a[i]*exp_a[i] + sign_b[i]*exp_b[i]
//   Eigen::Matrix<double, -1, 1> combined = (fn_EIGEN_double(log_a - max_logs, "exp", vect_type_exp).array()  + 
//     fn_EIGEN_double(log_b - max_logs, "exp", vect_type_exp).array()).matrix(); 
//   // fill both output vectors
//   log_sum_abs_result = max_logs + fn_EIGEN_double(combined.array().abs().matrix(), "log", vect_type_log);
//   
// }
//  




inline void log_sum_exp_general(     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_vals,  
                                     const std::string &vect_type_exp,
                                     const std::string &vect_type_log,
                                     Eigen::Ref<Eigen::Matrix<double, -1, 1>> log_sum_result,
                                     Eigen::Ref<Eigen::Matrix<double, -1, 1>> container_max_logs) {
  
  // find max for each row across all columns
  container_max_logs = log_vals.rowwise().maxCoeff();
  // sum across columns
 //Eigen::Matrix<double, -1, 1> sum_exp =  (fn_EIGEN_double( (log_vals.colwise() - container_max_logs) , "exp", vect_type_exp).array()).matrix().rowwise().sum();
  // compute results
  log_sum_result = container_max_logs + fn_EIGEN_double( (fn_EIGEN_double( (log_vals.colwise() - container_max_logs) , "exp", vect_type_exp).array()).matrix().rowwise().sum().array().abs(), "log", vect_type_log);
   
}



  
 


struct LogSumVecSingedResult { 
  
     double log_sum;
     double sign;
 
};
 
 
 
 
 
 inline LogSumVecSingedResult log_sum_vec_signed_v1(   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> log_abs_vec,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> signs,
                                                       const std::string &vect_type) {
   
             // const double huge_neg = -700.0;
             double max_log_abs = stan::math::max(log_abs_vec);  // find max 
           
             const Eigen::Matrix<double, -1, 1> &shifted_logs = (log_abs_vec.array() - max_log_abs);   ///// Shift logs and clip
             // shifted_logs = (shifted_logs.array() < huge_neg).select(huge_neg, shifted_logs);   ///// additionally clip (can comment out for no clipping)
             
             // Compute sum with signs carefully
             const Eigen::Matrix<double, -1, 1> &exp_terms = fn_EIGEN_double((log_abs_vec.array() - max_log_abs), "exp", vect_type);
             double sum_exp = (signs.array() * exp_terms.array()).sum();
             
             // // Handle near-zero sums (optional)
             // if (stan::math::abs(sum_exp) < stan::math::exp(huge_neg)) {
             //   return {huge_neg, 0.0};
             // }
             
             double log_abs_sum = max_log_abs + stan::math::log(stan::math::abs(sum_exp));   
             
             // // Clip final result if too large (optional)
             // if (log_abs_sum > 10.0) {  // exp(10) ≈ 22026, reasonable bound
             //   log_abs_sum = 10.0;
             // }
             
             return {log_abs_sum, sum_exp > 0 ? 1.0 : -1.0};
   
 }

 
 
 
 
 
 
 
//// with optional additional underflow protection (can be commented out easily)
inline void log_abs_sum_exp_general_v2(  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_vals,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> signs,
                                         const std::string &vect_type_exp,
                                         const std::string &vect_type_log,
                                         Eigen::Ref<Eigen::Matrix<double, -1, 1>> log_sum_abs_result,
                                         Eigen::Ref<Eigen::Matrix<double, -1, 1>> sign_result,
                                         Eigen::Ref<Eigen::Matrix<double, -1, 1>> container_max_logs,
                                         Eigen::Ref<Eigen::Matrix<double, -1, 1>> container_sum_exp_signed) {

  const double min_exp_neg = -700.0 ;
  const double max_exp_arg =  700.0;
  const double tiny = stan::math::exp(min_exp_neg);

  container_max_logs = log_abs_vals.rowwise().maxCoeff();    // Find max log_abs value for each row 


  const Eigen::Matrix<double, -1, -1>  &shifted_logs = (log_abs_vals.colwise() - container_max_logs);   //// Compute shifted logs with underflow protection
  
  //  Eigen::Matrix<double, -1, -1>  shifted_logs = (log_abs_vals.colwise() - container_max_logs);   //// Compute shifted logs with underflow protection
  //  shifted_logs = (shifted_logs.array() < -max_exp_arg).select( -max_exp_arg, shifted_logs );  ////// Clip very negative values to avoid unnecessary exp computations

  //// Compute exp terms and sum over columns with signs 
  container_sum_exp_signed = (fn_EIGEN_double((log_abs_vals.colwise() - container_max_logs).matrix(), "exp", vect_type_exp).array() *  signs.array()).matrix().rowwise().sum();

  //// Compute sign_result and log_sum_abs_result
  sign_result = container_sum_exp_signed.array().sign();
  log_sum_abs_result.array() = container_max_logs.array() + fn_EIGEN_double( container_sum_exp_signed.array().abs(), "log", vect_type_log).array();
 
   // sign_result(i) = std::copysign(1.0, sum_exp);
   // log_sum_abs_result(i) = container_max_logs(i) +  stan::math::log(stan::math::abs(sum_exp));
 
 // for (Eigen::Index i = 0; i < container_sum_exp_signed.rows(); ++i) {
 // 
 //       double sum_exp = container_sum_exp_signed(i);
 // 
 //       if (stan::math::abs(sum_exp) < tiny) {   //  if exp's cancel out or are too small
 // 
 //             sign_result(i) = 0.0;
 //             log_sum_abs_result(i) = min_exp_neg;
 // 
 //       } else {  // Normal case
 // 
 //             sign_result(i) = std::copysign(1.0, sum_exp);
 //             log_sum_abs_result(i) = container_max_logs(i) +  stan::math::log(stan::math::abs(sum_exp));
 //       }
 // 
 // }
  

} 


 
 
 
 
 
 
 

 
inline  void log_abs_matrix_vector_mult_v1(  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_matrix,
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_matrix,
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> log_abs_vector,
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> sign_vector,
                                             const std::string &vect_type_exp,
                                             const std::string &vect_type_log,
                                             Eigen::Ref<Eigen::Matrix<double, -1, 1>> log_abs_result,
                                             Eigen::Ref<Eigen::Matrix<double, -1, 1>> sign_result) {

   int n_rows = log_abs_matrix.rows();
   int n_cols = log_abs_matrix.cols();

             // Initialize temp storage for max finding pass
             Eigen::Matrix<double, -1, 1> max_logs = Eigen::Matrix<double, -1, 1>::Constant(n_rows, -700.0);

             // First pass: find max_log for each row
             for (int j = 0; j < n_cols; j++) {
               double log_vec_j = log_abs_vector(j);
               for (int i = 0; i < n_rows; i++) {
                 max_logs(i) = std::max(max_logs(i), log_abs_matrix(i,j) + log_vec_j);
               }
             }

             // Second pass: compute sums using exp-trick
             Eigen::Matrix<double, -1, 1> sums = Eigen::Matrix<double, -1, 1>::Zero(n_rows);
             for (int j = 0; j < n_cols; j++) {
               double log_vec_j = log_abs_vector(j);
               double sign_vec_j = sign_vector(j);

               for (int i = 0; i < n_rows; i++) {
                 double term = std::exp(log_abs_matrix(i,j) + log_vec_j - max_logs(i)) *
                   sign_matrix(i,j) * sign_vec_j;
                 sums(i) += term;
               }
             }

             // Final pass: compute results
             for (int i = 0; i < n_rows; i++) {
               sign_result(i) = (sums(i) >= 0) ? 1.0 : -1.0;
               log_abs_result(i) = std::log(std::abs(sums(i))) + max_logs(i);
             }

 }


 
 
 
 
 
inline void log_abs_matrix_vector_mult_v2(   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_matrix,
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_matrix,
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> log_abs_vector,
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> sign_vector,
                                             const std::string &vect_type_exp,
                                             const std::string &vect_type_log,
                                             Eigen::Ref<Eigen::Matrix<double, -1, 1>> log_abs_result,
                                             Eigen::Ref<Eigen::Matrix<double, -1, 1>> sign_result) {
   
   const int n_rows = log_abs_matrix.rows();
   const int n_cols = log_abs_matrix.cols();
   
   // Add broadcasted log_abs_vector to each row of log_abs_matrix
   Eigen::Matrix<double, -1, -1> combined_logs = (log_abs_matrix.rowwise() + log_abs_vector.transpose());
   
   // Find max along rows
   Eigen::Matrix<double, -1, 1> max_logs = combined_logs.rowwise().maxCoeff();
   
   // Compute exp(log_abs - max) * sign for all elements
   /// Eigen::Matrix<double, -1, -1> exp_terms = (combined_logs.colwise() - max_logs).array().exp();
   //// Eigen::Matrix<double, -1, -1> signed_terms = ( (combined_logs.colwise() - max_logs).array().exp().array() * sign_matrix.array() * sign_vector.transpose().array() ) ;
   
   // Sum rows
   Eigen::Matrix<double, -1, 1> sums = ( (combined_logs.colwise() - max_logs).array().exp().array() * sign_matrix.array() * sign_vector.transpose().array() ).rowwise().sum();
   
   // Compute final results
   sign_result = sums.array().sign().max(0.0); // Handles zero case
   log_abs_result = sums.array().abs().log() + max_logs.array();
   
}
 
 
 
 
 
 
 
 
 
 
  
  
 ----
 
 inline Eigen::Matrix<double, -1, 1  >   log_sum_exp_2d_Eigen_double( const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  x )  {
   
   int N = x.rows();
   Eigen::Matrix<double, -1, -1> rowwise_maxes_2d_array(N, 2);
   rowwise_maxes_2d_array.col(0) = x.array().rowwise().maxCoeff().matrix();
   rowwise_maxes_2d_array.col(1) = rowwise_maxes_2d_array.col(0);
   
   /// Eigen::Matrix<double, -1, 1>  rowwise_maxes_1d_vec = rowwise_maxes_2d_array.col(0);
   Eigen::Matrix<double, -1, 1>  sum_exp_vec =  (  (x.array()  -  rowwise_maxes_2d_array.array()).matrix() ).array().exp().matrix().rowwise().sum() ;
   
   return     ( rowwise_maxes_2d_array.col(0).array()    +    sum_exp_vec.array().log() ).matrix() ;
   
   
 }




 

 


 
 
 
 
  
 
 
inline Eigen::Matrix<double, -1, 1  > fn_log_sum_exp_2d_double(     Eigen::Ref<Eigen::Matrix<double, -1, -1>>  x,    // Eigen::Matrix<double, -1, 2> &x,
                                                                     const std::string &vect_type = "Stan",
                                                                     const bool &skip_checks = false) {
   
   {
     if (vect_type == "Eigen") {
       return  log_sum_exp_2d_Eigen_double(x);
     } else if (vect_type == "Stan") {
       return  log_sum_exp_2d_Stan_double(x);
     } else if (vect_type == "AVX2") {
#if defined(__AVX2__) && ( !(defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)) ) // use AVX2
       if (skip_checks == false)   return  fast_log_sum_exp_2d_AVX2_double(x);
       else                        return  fast_log_sum_exp_2d_AVX2_double(x);
#endif
     } else if (vect_type == "AVX512") {
#if defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)
       if (skip_checks == false)   return  fast_log_sum_exp_2d_AVX512_double(x);
       else                        return  fast_log_sum_exp_2d_AVX512_double(x);
#endif
     } else {
              return  log_sum_exp_2d_Stan_double(x);
     }
     
   }
   
   return  log_sum_exp_2d_Stan_double(x);
   
}
 
 
  






#endif

  
  
  
  
  