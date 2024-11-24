#pragma once 

#ifndef FN_WRAPPERS_SIMD_AVX512_HPP
#define FN_WRAPPERS_SIMD_AVX512_HPP

 

  
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
 
 
#if defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__) // use AVX-512
 
#include <immintrin.h>

 

 
using namespace Eigen;


 

 

typedef double (*FuncDouble)(double);
typedef double (*FuncDouble_wo_checks)(double);


typedef __m512d (*FuncAVX512)(const __m512d);
typedef __m512d (*FuncAVX512_wo_checks)(const __m512d);
 

 
 
 
 
template <typename T, typename FuncAVX512, typename FuncDouble>
inline  void fn_AVX512_row_or_col_vector(   Eigen::Ref<T>  x, 
                                           FuncAVX512 fn_AVX512, 
                                           FuncDouble fn_double) {
  
  
  const int N = x.size();
  const int vect_size = 8;
  const double vect_siz_dbl = 8.0;
  const int N_divisible_by_vect_size = std::floor( static_cast<double>(N) / vect_siz_dbl) * vect_size;
 
 // T x_temp = x; // make a copy 
 
 const Eigen::Matrix<double, -1, 1> x_tail = x.tail(vect_size); // copy of last 8 elements 
  
  if (N >= vect_size) {
    
          for (int i = 0; i + 8 <= N_divisible_by_vect_size; i += vect_size) {
            const __m512d AVX_array = _mm512_load_pd(&x(i));
            const __m512d AVX_array_out = fn_AVX512(AVX_array);
            _mm512_store_pd(&x(i), AVX_array_out);
          }
          
           if (N_divisible_by_vect_size != N) {    // Handle remainder 
             const int start_index = N - vect_size;
             const int end_index = N;
             int counter = 0;
               for (int i = start_index; i < end_index; ++i) {
                        x(i) = fn_double(x_tail(counter));
                        counter += 1;
               }
            }
           
  }  else {   // If N < 8, handle everything with scalar operations
    
        for (int i = 0; i < N; ++i) {
          x(i) = fn_double(x(i));
        }
        
  }
  
  /// x = x_temp;

  
}

 
 
 
 template<typename T, typename FuncAVX512, typename FuncDouble>
 inline  void fn_AVX512_matrix(  Eigen::Ref<T> x,
                                FuncAVX512 fn_AVX512,
                                FuncDouble fn_double) {
   
   const int n_rows = x.rows();
   const int n_cols = x.cols();
   
   if (n_rows > n_cols) { // if data in "long" format
       for (int j = 0; j < n_cols; ++j) {  
          Eigen::Matrix<double, -1, 1> x_col = x.col(j);
          Eigen::Ref<Eigen::Matrix<double, -1, 1>> x_col_Ref(x_col);
          fn_AVX512_row_or_col_vector<typename T::ColXpr>(x_col_Ref, fn_AVX512, fn_double);
          x.col(j) = x_col_Ref;
       }
   } else { 
     for (int j = 0; j < n_rows; ++j) {
          Eigen::Matrix<double, 1, -1> row = x.row(j);
          using RowType = decltype(row);
          Eigen::Ref<Eigen::Matrix<double, 1, -1>> row_Ref(row);
          fn_AVX512_row_or_col_vector<RowType>(row_Ref, fn_AVX512, fn_double);
          x.row(j) = row_Ref;
     }
   }
   
   
 }
 
 
 
 
 
 
// template<typename T, typename FuncAVX512, typename FuncDouble>
// inline void fn_AVX512_matrix(  Eigen::Ref<T> x,
//                                FuncAVX512 fn_AVX512,
//                                FuncDouble fn_double) {
//   
//    const int n_rows = x.rows();
//    const int n_cols = x.cols();
//    const int vect_size = 8;
//    const double vect_siz_dbl = 8.0;
//    const int rows_divisible_by_vect_size = std::floor( static_cast<double>(n_rows) / vect_siz_dbl) * vect_size;
//  
//   // T x_temp = x; // make a copy 
//    
//    for (int j = 0; j < n_cols; ++j) { /// loop through cols first as col-major storage
//      
//      const Eigen::Matrix<double, -1, 1> x_tail = x.col(j).tail(vect_size); // copy of last 8 elements 
// 
//         // Make sure we have at least 8 rows before trying AVX
//         if (n_rows >= vect_size) {
//           
//               for (int i = 0; i < rows_divisible_by_vect_size; i += vect_size) {
//                 const __m512d AVX_array = _mm512_loadu_pd(&x(i, j));
//                 const __m512d AVX_array_out = fn_AVX512(AVX_array);
//                 _mm512_storeu_pd(&x(i, j), AVX_array_out);
//               }
//               
//               // Handle remaining rows with double fns
//               const int start_index = n_rows - vect_size;
//               const int end_index = n_rows;
//               int counter = 0;
//                 for (int i = start_index; i < end_index; ++i) {
//                         x(i, j) = fn_double(x_tail(counter));
//                          counter += 1;
//                 }
//  
//               
//         } else {    // If n_rows < 8, handle entire row with double operations
//           for (int i = 0; i < n_rows; ++i) {
//             x(i, j) = fn_double(x(i, j));
//           } 
//         }
// 
//   }
//    
//  //   x = x_temp;
//    
// 
// }

 


template <typename T, typename FuncAVX512, typename FuncDouble>
inline  void fn_AVX512_dbl_Eigen(Eigen::Ref<T> x, 
                                FuncAVX512 fn_AVX512, 
                                FuncDouble fn_double) {
  
    constexpr int n_rows = T::RowsAtCompileTime;
    constexpr int n_cols = T::ColsAtCompileTime;
    
    if constexpr (n_rows == 1 && n_cols == -1) {   // Row vector case
    
      fn_AVX512_row_or_col_vector(x, fn_AVX512, fn_double);
      
    } else if constexpr (n_rows == -1 && n_cols == 1) {  // Column vector case
     
      fn_AVX512_row_or_col_vector(x, fn_AVX512, fn_double);
      
    } else {   // General matrix case
    
      fn_AVX512_matrix(x, fn_AVX512, fn_double);
      
    }
  
}



 
 
  
 

template<typename FuncAVX512, typename FuncDouble, typename FuncAVX512_wo_checks, typename FuncDouble_wo_checks, typename T>
inline  void                   fn_process_double_AVX512_sub_function(     Eigen::Ref<T> x, // since this is helper function we call x by reference "&" not "&&"
                                                                         FuncAVX512 fn_fast_AVX512_function,
                                                                         FuncDouble fn_fast_double_function,
                                                                         FuncAVX512_wo_checks fn_fast_AVX512_function_wo_checks,
                                                                         FuncDouble_wo_checks fn_fast_double_function_wo_checks, 
                                                                         const bool skip_checks) {
  
  if (skip_checks == false) {
    
    fn_AVX512_dbl_Eigen(x, fn_fast_AVX512_function, fn_fast_double_function);
    
  }   else  {
    
    fn_AVX512_dbl_Eigen(x, fn_fast_AVX512_function_wo_checks, fn_fast_double_function_wo_checks);
 
  }
  
}


 





 
  
template <typename T>
inline    void        fn_return_Ref_double_AVX512(  Eigen::Ref<T> x,
                                                   const std::string &fn,
                                                   const bool &skip_checks) {
  
  if        (fn == "exp") {                              fn_process_double_AVX512_sub_function(x, fast_exp_1_AVX512, fast_exp_1, fast_exp_1_wo_checks_AVX512, fast_exp_1, skip_checks) ;
  } else if (fn == "log") {                              fn_process_double_AVX512_sub_function(x, fast_log_1_AVX512, fast_log_1, fast_log_1_wo_checks_AVX512, fast_log_1, skip_checks) ;
  } else if (fn == "log1p") {                            fn_process_double_AVX512_sub_function(x, fast_log1p_1_AVX512, fast_log1p_1, fast_log1p_1_wo_checks_AVX512, fast_log1p_1, skip_checks) ;
  } else if (fn == "log1m") {                            fn_process_double_AVX512_sub_function(x, fast_log1m_1_AVX512, fast_log1m_1, fast_log1m_1_wo_checks_AVX512, fast_log1m_1, skip_checks) ;
  } else if (fn == "logit") {                            fn_process_double_AVX512_sub_function(x, fast_logit_AVX512, mvp_std_logit, fast_logit_wo_checks_AVX512, mvp_std_logit, skip_checks) ;
  } else if (fn == "tanh") {                             fn_process_double_AVX512_sub_function(x, fast_tanh_AVX512, fast_tanh, fast_tanh_wo_checks_AVX512, fast_tanh, skip_checks) ;
  } else if (fn == "Phi_approx") {                       fn_process_double_AVX512_sub_function(x, fast_Phi_approx_AVX512, fast_Phi_approx, fast_Phi_approx_wo_checks_AVX512, fast_Phi_approx, skip_checks) ;
  } else if (fn == "log_Phi_approx") {                   fn_process_double_AVX512_sub_function(x, fast_log_Phi_approx_AVX512, fast_log_Phi_approx, fast_log_Phi_approx_wo_checks_AVX512, fast_log_Phi_approx_wo_checks, skip_checks) ;
  } else if (fn == "inv_Phi_approx") {                   fn_process_double_AVX512_sub_function(x, fast_inv_Phi_approx_AVX512, fast_inv_Phi_approx, fast_inv_Phi_approx_wo_checks_AVX512, fast_inv_Phi_approx_wo_checks, skip_checks) ;
  } else if (fn == "inv_Phi_approx_from_logit_prob") {   fn_process_double_AVX512_sub_function(x, fast_inv_Phi_approx_from_logit_prob_AVX512, fast_inv_Phi_approx_from_logit_prob, fast_inv_Phi_approx_from_logit_prob_wo_checks_AVX512, fast_inv_Phi_approx_from_logit_prob_wo_checks, skip_checks) ;
  } else if (fn == "Phi") {                              fn_process_double_AVX512_sub_function(x, fast_Phi_AVX512, fast_Phi, fast_Phi_wo_checks_AVX512, fast_Phi, skip_checks) ;
  } else if (fn == "inv_Phi") {                          fn_process_double_AVX512_sub_function(x, fast_inv_Phi_wo_checks_AVX512, fast_inv_Phi_wo_checks, fast_inv_Phi_wo_checks_AVX512, fast_inv_Phi_wo_checks, skip_checks) ;
  } else if (fn == "inv_logit") {                        fn_process_double_AVX512_sub_function(x, fast_inv_logit_AVX512, fast_inv_logit, fast_inv_logit_wo_checks_AVX512, fast_inv_logit, skip_checks) ;
  } else if (fn == "log_inv_logit") {                    fn_process_double_AVX512_sub_function(x, fast_log_inv_logit_AVX512, fast_log_inv_logit, fast_log_inv_logit_wo_checks_AVX512, fast_log_inv_logit, skip_checks) ;
  }
  
  
}


 
 
 
 

  




#endif

  
#endif 
  
  
  
  
  
  