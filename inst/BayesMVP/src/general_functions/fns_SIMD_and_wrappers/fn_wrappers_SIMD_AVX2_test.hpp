#pragma once 
 
#ifndef FN_WRAPPERS_SIMD_AVX2_TEST_HPP
#define FN_WRAPPERS_SIMD_AVX2_TEST_HPP

 
 
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
 
 
#if defined(__AVX2__) && ( !(defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)) ) // use AVX2 if AVX-512 not available 
 
 
#include <immintrin.h>

 

 
using namespace Eigen;





template <typename T>
inline  void TEST_fn_AVX2_row_or_col_vector(    Eigen::Ref<T>  x_Ref,
                                                FuncAVX fn_AVX,
                                                FuncDouble fn_double) {
 
       REprintf("Entering TEST_fn_AVX2_row_or_col_vector\n");
       R_FlushConsole();
       
       const int N = x_Ref.size();
       const int vect_size = 4;
       const double vect_siz_dbl = static_cast<double>(vect_size);
       const double N_dbl = static_cast<double>(N);
       const int N_divisible_by_vect_size = std::floor(N_dbl / vect_siz_dbl) * vect_size;
       
       REprintf("Creating x_tail\n"); R_FlushConsole();
       Eigen::Matrix<double, -1, 1> x_tail = Eigen::Matrix<double, -1, 1>::Zero(vect_size); // last vect_size elements
       {
         int counter = 0;
         for (int i = N - vect_size; i < N; ++i) {
           x_tail(counter) = x_Ref(i);
           counter += 1;
         } 
       }
       REprintf("Created x_tail\n"); R_FlushConsole();
       
       if (N >= vect_size) {
               
               ALIGN32 double buffer[4];  // using an aligned buffer for AVX operations
               
               for (int i = 0; i + vect_size <= N_divisible_by_vect_size; i += vect_size) {
                 
                       // Copy data to aligned buffer
                       REprintf("Copying results to buffer\n"); R_FlushConsole();
                       for(int j = 0; j < vect_size; j++) {
                         buffer[j] = x_Ref(i + j);
                       }
                       REprintf("Copied results to buffer\n"); R_FlushConsole();
                       
                       REprintf("Before _mm256_load_pd(buffer) \n"); R_FlushConsole();
                       ALIGN32  __m256d AVX_array = _mm256_load_pd(buffer);
                       REprintf("After _mm256_load_pd(buffer) \n"); R_FlushConsole();
                       
                       REprintf("Before fn_AVX(AVX_array) \n"); R_FlushConsole();
                       ALIGN32  __m256d AVX_array_out = fn_AVX(AVX_array);
                       REprintf("After fn_AVX(AVX_array) \n"); R_FlushConsole();
                       
                       REprintf("Before _mm256_store_pd(buffer, AVX_array_out) \n"); R_FlushConsole();
                       _mm256_store_pd(buffer, AVX_array_out);
                       REprintf("After  _mm256_store_pd(buffer, AVX_array_out) \n"); R_FlushConsole();
                       
                       // Copy back to Eigen
                       REprintf("Copying results back to x_Ref\n"); R_FlushConsole();
                       for(int j = 0; j < vect_size; j++) {
                         x_Ref(i + j) = buffer[j];
                       }
                       REprintf("Copied results back to x_Ref\n"); R_FlushConsole();
                 
               }
               
               if (N_divisible_by_vect_size != N) {    // Handle remainder
                 int counter = 0;
                 for (int i = N - vect_size; i < N; ++i) {
                   x_Ref(i) =  fn_double(x_tail(counter));
                   counter += 1;
                 }
               }
         
       }  else {   // If N < vect_size, handle everything with scalar operations
         
               for (int i = 0; i < N; ++i) {
                 x_Ref(i) = fn_double(x_Ref(i));
               }
               
       }
       
       REprintf("Exiting TEST_fn_AVX2_row_or_col_vector\n");
       R_FlushConsole();
 
} 

 
 
 
 
 
 
 

 

template <typename T>
inline  void TEST_fn_AVX2_matrix(   Eigen::Ref<T> x_Ref,
                                    FuncAVX fn_AVX, 
                                    FuncDouble fn_double) {
 
     REprintf("Entering fn_AVX2_matrix\n");
     R_FlushConsole();
     
     const int n_rows = x_Ref.rows();
     const int n_cols = x_Ref.cols();
     
     if (n_rows > n_cols) { // if data in "long" format
       for (int j = 0; j < n_cols; ++j) {   
         Eigen::Matrix<double, -1, 1> x_col = x_Ref.col(j);
         using ColType = decltype(x_col);
         Eigen::Ref<Eigen::Matrix<double, -1, 1>> x_col_Ref(x_col); 
         fn_AVX2_row_or_col_vector<ColType>(x_col_Ref, fn_AVX, fn_double);
         x_Ref.col(j) = x_col_Ref;
       }
     } else { 
       for (int j = 0; j < n_rows; ++j) {
         Eigen::Matrix<double, 1, -1> x_row = x_Ref.row(j);
         using RowType = decltype(x_row);
         Eigen::Ref<Eigen::Matrix<double, 1, -1>> x_row_Ref(x_row); 
         fn_AVX2_row_or_col_vector<RowType>(x_row_Ref, fn_AVX, fn_double);
         x_Ref.row(j) = x_row_Ref;
       }
     }
     
     REprintf("Exiting fn_AVX2_matrix\n");
     R_FlushConsole();
     
} 


 
 
 



template <typename T>
inline  void TEST_fn_AVX2_dbl_Eigen(    Eigen::Ref<T> x_Ref, 
                                       FuncAVX fn_AVX, 
                                       FuncDouble fn_double) {
 
       REprintf("Entering TEST_fn_AVX2_dbl_Eigen\n");
       R_FlushConsole();
       
       constexpr int n_rows = T::RowsAtCompileTime;
       constexpr int n_cols = T::ColsAtCompileTime;
       
       if constexpr (n_rows == 1 && n_cols == -1) {   // Row vector case
         
             REprintf("TEST_fn_AVX2_dbl_Eigen - Row vector case\n");
             R_FlushConsole();
             TEST_fn_AVX2_row_or_col_vector(x_Ref, fn_AVX, fn_double);
         
       } else if constexpr (n_rows == -1 && n_cols == 1) {  // Column vector case
         
             REprintf("TEST_fn_AVX2_dbl_Eigen - Column vector case\n");
             R_FlushConsole();
             TEST_fn_AVX2_row_or_col_vector(x_Ref, fn_AVX, fn_double);
         
       } else {   // General matrix case
         
             REprintf("TEST_fn_AVX2_dbl_Eigen - Matrix case\n");
             R_FlushConsole();
             TEST_fn_AVX2_matrix(x_Ref, fn_AVX, fn_double);
         
       }
       
       REprintf("Exiting TEST_fn_AVX2_dbl_Eigen\n");
       R_FlushConsole();
 
}






 
template <typename T>
inline void   TEST_fn_process_double_AVX2_sub_function(   Eigen::Ref<T> x_Ref,  
                                                     FuncAVX    fn_fast_AVX2_function,
                                                     FuncDouble fn_fast_double_function,
                                                     FuncAVX    fn_fast_AVX2_function_wo_checks,
                                                     FuncDouble fn_fast_double_function_wo_checks, 
                                                     const bool skip_checks) {
  
     if (skip_checks == false) {
       
           TEST_fn_AVX2_dbl_Eigen(x_Ref, fn_fast_AVX2_function, fn_fast_double_function);
       
     } else {
       
           TEST_fn_AVX2_dbl_Eigen(x_Ref, fn_fast_AVX2_function_wo_checks, fn_fast_double_function_wo_checks);
       
     }
 
}
 
 
 
 
 

 
 
 
template <typename T>
inline  void       TEST_fn_process_Ref_double_AVX2(     Eigen::Ref<T> x_Ref,
                                                        const std::string &fn,
                                                        const bool &skip_checks) {
  
    std::cout << "Entering TEST_fn_process_Ref_double_AVX2" << std::endl;
  
    if (fn == "test_simple") {    
          std::cout << "Calling test_simple function" << std::endl;
          try { 
              TEST_fn_AVX2_dbl_Eigen(x_Ref, test_simple_AVX2, test_simple_double);
          } catch (const std::exception& e) { 
              std::cout << "Exception caught: " << e.what() << std::endl;
              throw;
          } catch (...) {
              std::cout << "Unknown exception caught" << std::endl;
              throw;
          }
    } 
    
    std::cout << "Exiting TEST_fn_process_Ref_double_AVX2" << std::endl;

}
 


 
 

 
 


 
 
 
 
 
  
  


#endif



#endif

  
  