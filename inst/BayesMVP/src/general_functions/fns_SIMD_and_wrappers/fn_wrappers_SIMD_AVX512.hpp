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
 ALWAYS_INLINE  void fn_AVX512_row_or_col_vector(   Eigen::Ref<T>  x,
                                                    const FuncAVX512 &fn_AVX512,
                                                    const FuncDouble &fn_double) {


   const int N = x.size();
   const int vect_size = 8;
   const double vect_siz_dbl = 8.0;
   const int N_divisible_by_vect_size = std::floor( static_cast<double>(N) / vect_siz_dbl) * vect_size;

   Eigen::Matrix<double, -1, 1> x_tail(vect_size); // last vect_size elements
   {
       int counter = 0;
       for (int i = N - vect_size; i < N; ++i) {
         x_tail(counter) = fn_double(x(i));
         counter += 1;
       }
   }

   if (N >= vect_size) {

           for (int i = 0; i + vect_size <= N_divisible_by_vect_size; i += vect_size) {

             const __m512d AVX_array = _mm512_load_pd(&x(i));
             const __m512d AVX_array_out = fn_AVX512(AVX_array);
             _mm512_store_pd(&x(i), AVX_array_out);

           }
           
          /// x.tail(vect_size) = x_tail

           if (N_divisible_by_vect_size != N) {    // Handle remainder
             int counter = 0;
               for (int i = N - vect_size; i < N; ++i) {
                      x(i) =  (x_tail(counter));
                      counter += 1;
               }
            }


   }  else {   // If N < vect_size, handle everything with scalar operations

           for (int i = 0; i < N; ++i) {
             x(i) = fn_double(x(i));
           }

   }


 }
 // 
 
 
 
 
// template <typename T, typename FuncAVX512, typename FuncDouble>
// ALWAYS_INLINE  void fn_AVX512_row_or_col_vector(   Eigen::Ref<T>  x, 
//                                                    const FuncAVX512 &fn_AVX512, 
//                                                    const FuncDouble &fn_double) {
//   
//   
//   const int N = x.size();
//   const int vect_size = 8;
//   const double vect_siz_dbl = 8.0;
//   const int N_divisible_by_vect_size = std::floor( static_cast<double>(N) / vect_siz_dbl) * vect_size;
//  
//   typename T::PlainObject x_temp = x; // make a copy
//   
//   if (N >= vect_size) {
//     
//           for (int i = 0; i + vect_size <= N_divisible_by_vect_size; i += vect_size) {
//             
//                 const __m512d AVX_array = _mm512_load_pd(&x(i)); 
//                 const __m512d AVX_array_out = fn_AVX512(AVX_array);
//                 _mm512_store_pd(&x_temp(i), AVX_array_out);
//            
//           }
//            
//            if (N_divisible_by_vect_size != N) {    // Handle remainder
//              for (int i = N - vect_size; i < N; ++i) {
//                x_temp(i) = fn_double(x(i));
//              }
//            }
//            
//   }  else {   // If N < vect_size, handle everything with scalar operations
//     
//         for (int i = 0; i < N; ++i) {
//           x_temp(i) = fn_double(x(i));
//         }
//         
//   }
//   
//     x = x_temp;
// 
//   
// }

 
 
 
template<typename T, typename FuncAVX512, typename FuncDouble>
ALWAYS_INLINE  void fn_AVX512_matrix(  Eigen::Ref<T> x,
                                       const FuncAVX512 &fn_AVX512,
                                       const FuncDouble &fn_double) {
   
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
          Eigen::Matrix<double, 1, -1> x_row = x.row(j);
          using RowType = decltype(x_row);
          Eigen::Ref<Eigen::Matrix<double, 1, -1>> x_row_Ref(x_row);
          fn_AVX512_row_or_col_vector<RowType>(x_row_Ref, fn_AVX512, fn_double);
          x.row(j) = x_row_Ref;
     }
   }
   
   
}
 
 
 
 
 


template <typename T, typename FuncAVX512, typename FuncDouble>
ALWAYS_INLINE  void fn_AVX512_dbl_Eigen(Eigen::Ref<T> x, 
                                        const FuncAVX512 &fn_AVX512, 
                                        const FuncDouble &fn_double) {
  
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
ALWAYS_INLINE  void                   fn_process_double_AVX512_sub_function(     Eigen::Ref<T> x, // since this is helper function we call x by reference "&" not "&&"
                                                                                 const FuncAVX512 &fn_fast_AVX512_function,
                                                                                 const FuncDouble &fn_fast_double_function,
                                                                                 const FuncAVX512_wo_checks &fn_fast_AVX512_function_wo_checks,
                                                                                 const FuncDouble_wo_checks &fn_fast_double_function_wo_checks, 
                                                                                 bool skip_checks) {
  
  if (skip_checks == false) {
    
     fn_AVX512_dbl_Eigen(x, fn_fast_AVX512_function, fn_fast_double_function);
    
  } else {
    
     fn_AVX512_dbl_Eigen(x, fn_fast_AVX512_function_wo_checks, fn_fast_double_function_wo_checks);
 
  }
  
}


 





 
  
template <typename T>
ALWAYS_INLINE    void        fn_return_Ref_double_AVX512(  Eigen::Ref<T> x,
                                                   const std::string &fn,
                                                   const bool skip_checks) {
  
  if        (fn == "exp") {                              fn_process_double_AVX512_sub_function(x, fast_exp_1_AVX512, mvp_std_exp, fast_exp_1_wo_checks_AVX512, mvp_std_exp, skip_checks) ;
  } else if (fn == "log") {                              fn_process_double_AVX512_sub_function(x, fast_log_1_AVX512, mvp_std_log, fast_log_1_wo_checks_AVX512, mvp_std_log, skip_checks) ;
  } else if (fn == "log1p") {                            fn_process_double_AVX512_sub_function(x, fast_log1p_1_AVX512, mvp_std_log1p, fast_log1p_1_wo_checks_AVX512, mvp_std_log1p, skip_checks) ;
  } else if (fn == "log1m") {                            fn_process_double_AVX512_sub_function(x, fast_log1m_1_AVX512, mvp_std_log1m, fast_log1m_1_wo_checks_AVX512, mvp_std_log1m, skip_checks) ;
  } else if (fn == "logit") {                            fn_process_double_AVX512_sub_function(x, fast_logit_AVX512, mvp_std_logit, fast_logit_wo_checks_AVX512, mvp_std_logit, skip_checks) ;
  } else if (fn == "tanh") {                             fn_process_double_AVX512_sub_function(x, fast_tanh_AVX512, mvp_std_tanh, fast_tanh_wo_checks_AVX512, mvp_std_tanh, skip_checks) ;
  } else if (fn == "Phi_approx") {                       fn_process_double_AVX512_sub_function(x, fast_Phi_approx_AVX512, mvp_std_Phi_approx, fast_Phi_approx_wo_checks_AVX512, mvp_std_Phi_approx, skip_checks) ;
  } else if (fn == "log_Phi_approx") {                   fn_process_double_AVX512_sub_function(x, fast_log_Phi_approx_AVX512, fast_log_Phi_approx, fast_log_Phi_approx_wo_checks_AVX512, fast_log_Phi_approx_wo_checks, skip_checks) ;
  } else if (fn == "inv_Phi_approx") {                   fn_process_double_AVX512_sub_function(x, fast_inv_Phi_approx_AVX512, fast_inv_Phi_approx_wo_checks, fast_inv_Phi_approx_wo_checks_AVX512, fast_inv_Phi_approx_wo_checks, skip_checks) ;
  } else if (fn == "inv_Phi_approx_from_logit_prob") {   fn_process_double_AVX512_sub_function(x, fast_inv_Phi_approx_from_logit_prob_AVX512, fast_inv_Phi_approx_from_logit_prob, fast_inv_Phi_approx_from_logit_prob_wo_checks_AVX512, fast_inv_Phi_approx_from_logit_prob_wo_checks, skip_checks) ;
  } else if (fn == "Phi") {                              fn_process_double_AVX512_sub_function(x, fast_Phi_AVX512, mvp_std_Phi, fast_Phi_wo_checks_AVX512, mvp_std_Phi, skip_checks) ;
  } else if (fn == "inv_Phi") {                          fn_process_double_AVX512_sub_function(x, fast_inv_Phi_wo_checks_AVX512, mvp_std_inv_Phi, fast_inv_Phi_wo_checks_AVX512, mvp_std_inv_Phi, skip_checks) ;
  } else if (fn == "inv_logit") {                        fn_process_double_AVX512_sub_function(x, fast_inv_logit_AVX512, mvp_std_inv_logit, fast_inv_logit_wo_checks_AVX512, mvp_std_inv_logit, skip_checks) ;
  } else if (fn == "log_inv_logit") {                    fn_process_double_AVX512_sub_function(x, fast_log_inv_logit_AVX512, mvp_std_log_inv_logit, fast_log_inv_logit_wo_checks_AVX512, mvp_std_log_inv_logit, skip_checks) ;
  }
  
  
}


 
 
 
 

  




#endif

  
#endif 
  
  
  
  
  
  