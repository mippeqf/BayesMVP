#pragma once 
 
#ifndef FN_WRAPPERS_SIMD_AVX2_HPP
#define FN_WRAPPERS_SIMD_AVX2_HPP
 
 
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
 
 
#if defined(__AVX512__) && ( !(defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX2DQ__)) ) // use AVX2
 
 
#include <immintrin.h>

 

 
using namespace Eigen;


 

 

typedef double (*FuncDouble)(double);
typedef double (*FuncDouble_wo_checks)(double);
 
typedef __m256d (*FuncAVX2)(const __m256d);
typedef __m256d (*FuncAVX2_wo_checks)(const __m256d);
 

 
 
 
 
 template <typename T, typename FuncAVX2, typename FuncDouble>
 inline void fn_AVX2_row_or_col_vector(   Eigen::Ref<T>  x, 
                                            FuncAVX2 fn_AVX2, 
                                            FuncDouble fn_double) {
   
   
   const int N = x.size();
   const int vect_size = 4;
   const double vect_siz_dbl = 4.0;
   const int N_divisible_by_vect_size = std::floor( static_cast<double>(N) / vect_siz_dbl) * vect_size;
   
   T x_temp = x; // make a copy 
   
   if (N >= vect_size) {
     
     for (int i = 0; i + vect_size <= N_divisible_by_vect_size; i += vect_size) {
       const __m256d AVX_array = _mm256_loadu_pd(&x(i));
       const __m256d AVX_array_out = fn_AVX2(AVX_array);
       _mm256_storeu_pd(&x_temp(i), AVX_array_out);
     }
     
     const int start_index = N - vect_size;
     const int end_index = N;
     for (int i = start_index; i < end_index; ++i) {
       x_temp(i) = fn_double(x(i));
     }
     // }
     
   }  else {   // If N < 4, handle everything with scalar operations
     
     for (int i = 0; i < N; ++i) {
       x_temp(i) = fn_double(x(i));
     }
     
   }
   
   x = x_temp;
   
   
 }
 
 
 
 
 template<typename T, typename FuncAVX2, typename FuncDouble>
 inline void fn_AVX2_matrix(  Eigen::Ref<T> x, 
                                FuncAVX2 fn_AVX2,
                                FuncDouble fn_double) {
   
   const int n_rows = x.rows();
   const int n_cols = x.cols();
   const int vect_size = 4;
   const double vect_siz_dbl = 4.0;
   const int rows_divisible_by_vect_size = std::floor( static_cast<double>(n_rows) / vect_siz_dbl) * vect_size;
    
   T x_temp = x; // make a copy 
   
   for (int j = 0; j < n_cols; ++j) { /// loop through cols first as col-major storage
     
     //// Make sure we have at least 8 rows before trying AVX
     if (n_rows >= vect_size) {
       
       for (int i = 0; i < rows_divisible_by_vect_size; i += vect_size) {
         const __m256d AVX_array = _mm256_loadu_pd(&x(i, j));
         const __m256d AVX_array_out = fn_AVX2(AVX_array);
         _mm256_storeu_pd(&x_temp(i, j), AVX_array_out);
       }
       
       //// Handle remaining rows with double fns
       const int start_index = n_rows - vect_size;
       const int end_index = n_rows;
       for (int i = start_index; i < end_index; ++i) { 
         x_temp(i, j) = fn_double(x(i, j));
       }

       
     } else {    //// If n_rows < 4, handle entire row with double operations
       for (int i = 0; i < n_rows; ++i) {
         x_temp(i, j) = fn_double(x(i, j));
       } 
     }
     
   }
   
   x = x_temp; 
   
   
 }
  
 
 
 
 template <typename T, typename FuncAVX2, typename FuncDouble>
 inline void fn_AVX2_dbl_Eigen(Eigen::Ref<T> x, 
                                 FuncAVX2 fn_AVX2, 
                                 FuncDouble fn_double) {
   
   constexpr int n_rows = T::RowsAtCompileTime;
   constexpr int n_cols = T::ColsAtCompileTime;
    
   if constexpr (n_rows == 1 && n_cols == -1) {   // Row vector case
     
         fn_AVX2_row_or_col_vector(x, fn_AVX2, fn_double);
     
   } else if constexpr (n_rows == -1 && n_cols == 1) {  // Column vector case
     
         fn_AVX2_row_or_col_vector(x, fn_AVX2, fn_double);
     
   } else {   // General matrix case
      
         fn_AVX2_matrix(x, fn_AVX2, fn_double);
     
   }
   
 }
 
 
 

 
 
 
  
template<typename FuncAVX2, typename FuncDouble, typename FuncAVX2_wo_checks, typename FuncDouble_wo_checks, typename T>
inline void    fn_process_double_AVX2_sub_function(     Eigen::Ref<T> x,  
                                                        FuncAVX2 fn_fast_AVX2_function,
                                                        FuncDouble fn_fast_double_function,
                                                        FuncAVX2_wo_checks fn_fast_AVX2_function_wo_checks,
                                                        FuncDouble_wo_checks fn_fast_double_function_wo_checks, 
                                                        const bool skip_checks) {
  
  if (skip_checks == false) {
    
    fn_AVX2_dbl_Eigen(x, fn_fast_AVX2_function, fn_fast_double_function);
    
  }   else  {
    
    fn_AVX2_dbl_Eigen(x, fn_fast_AVX2_function_wo_checks, fn_fast_double_function_wo_checks);
    
  }
  
}

 


 

 
 
 
template <typename T>
inline  void       fn_return_Ref_double_AVX2( Eigen::Ref<T> x,
                                              const std::string &fn,
                                              const bool &skip_checks) {
 
 
    if        (fn == "exp") {                               fn_process_double_AVX2_sub_function(x, fast_exp_1_AVX2,  mvp_std_exp,   fast_exp_1_wo_checks_AVX2, mvp_std_exp, skip_checks) ;
    } else if (fn == "log") {                               fn_process_double_AVX2_sub_function(x, fast_log_1_AVX2, mvp_std_log, fast_log_1_wo_checks_AVX2, mvp_std_log, skip_checks) ;
    } else if (fn == "log1p") {                             fn_process_double_AVX2_sub_function(x, fast_log1p_1_AVX2, mvp_std_log1p, fast_log1p_1_wo_checks_AVX2, mvp_std_log1p, skip_checks) ;
    } else if (fn == "log1m") {                             fn_process_double_AVX2_sub_function(x, fast_log1m_1_AVX2, mvp_std_log1m, fast_log1m_1_wo_checks_AVX2, mvp_std_log1m, skip_checks) ;
    } else if (fn == "logit") {                             fn_process_double_AVX2_sub_function(x, fast_logit_AVX2, mvp_std_logit, fast_logit_wo_checks_AVX2, mvp_std_logit, skip_checks) ;
    } else if (fn == "tanh") {                              fn_process_double_AVX2_sub_function(x, fast_tanh_AVX2, mvp_std_tanh, fast_tanh_wo_checks_AVX2, mvp_std_tanh, skip_checks) ;
    } else if (fn == "Phi_approx") {                        fn_process_double_AVX2_sub_function(x, fast_Phi_approx_AVX2, mvp_std_Phi_approx, fast_Phi_approx_wo_checks_AVX2, mvp_std_Phi_approx, skip_checks) ;
    } else if (fn == "log_Phi_approx") {                    fn_process_double_AVX2_sub_function(x, fast_log_Phi_approx_AVX2, fast_log_Phi_approx, fast_log_Phi_approx_wo_checks_AVX2, fast_log_Phi_approx_wo_checks, skip_checks) ;
    } else if (fn == "inv_Phi_approx") {                    fn_process_double_AVX2_sub_function(x, fast_inv_Phi_approx_AVX2, fast_inv_Phi_approx, fast_inv_Phi_approx_wo_checks_AVX2, fast_inv_Phi_approx_wo_checks, skip_checks) ;
    } else if (fn == "inv_Phi_approx_from_logit_prob") {    fn_process_double_AVX2_sub_function(x, fast_inv_Phi_approx_from_logit_prob_AVX2, fast_inv_Phi_approx_from_logit_prob, fast_inv_Phi_approx_from_logit_prob_wo_checks_AVX2, fast_inv_Phi_approx_from_logit_prob_wo_checks, skip_checks) ;
    } else if (fn == "Phi") {                               fn_process_double_AVX2_sub_function(x, fast_Phi_AVX2, mvp_std_Phi, fast_Phi_wo_checks_AVX2, mvp_std_Phi, skip_checks) ;
    } else if (fn == "inv_Phi") {                           fn_process_double_AVX2_sub_function(x, fast_inv_Phi_wo_checks_AVX2, mvp_std_inv_Phi, fast_inv_Phi_wo_checks_AVX2, mvp_std_inv_Phi, skip_checks) ;
    } else if (fn == "inv_logit") {                         fn_process_double_AVX2_sub_function(x, fast_inv_logit_AVX2, mvp_std_inv_logit, fast_inv_logit_wo_checks_AVX2, mvp_std_inv_logit, skip_checks) ;
    } else if (fn == "log_inv_logit") {                     fn_process_double_AVX2_sub_function(x, fast_log_inv_logit_AVX2, fast_log_inv_logit, fast_log_inv_logit_wo_checks_AVX2, fast_log_inv_logit_wo_checks, skip_checks) ;
    }
 

}
 


 
 

 
 


 
 
 
 
 
  
  


#endif



#endif

  
  