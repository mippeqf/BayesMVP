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
 
 
#if defined(__AVX2__) && ( !(defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)) ) // use AVX2 if AVX-512 not available 
 
 
#include <immintrin.h>

 

 
using namespace Eigen;



// typedef VECTORCALL __m256d (*FuncAVX)(const __m256d);
// typedef VECTORCALL __m256d (*FuncAVX_wo_checks)(const __m256d);
 
// typedef __m256d (__attribute__((sysv_abi)) *FuncAVX)(const __m256d);
// typedef __m256d (__attribute__((sysv_abi)) *FuncAVX_wo_checks)(const __m256d);


class DoubleWrapper {
  
      using FuncType = double(double);
      FuncType* fn;
      
    public:
      DoubleWrapper(FuncType* f) : fn(f) {}
      double operator()(double x) const { return fn(x); }
  
}; 

class DoubleWrapper_wo_checks {
  
      using FuncType = double(double);
      FuncType* fn;
      
    public:
      DoubleWrapper_wo_checks(FuncType* f) : fn(f) {} 
      double operator()(double x) const { return fn(x); }
  
};


class AVXWrapper {
  
    using FuncType = __m256d(__m256d);
    alignas(32) FuncType* fn;  // Align the function pointer  
      
    public:
      AVXWrapper(FuncType* f) : fn(f) {}
      
      __m256d operator()(__m256d x) const {
            return fn(x);
        }
  
};

class AVXWrapper_wo_checks {
  
    using FuncType = __m256d(__m256d);
    alignas(32) FuncType* fn;  // Align the function pointer  
      
    public:
      AVXWrapper_wo_checks(FuncType* f) : fn(f) {}
      
      __m256d operator()(__m256d x) const {
            return fn(x);
      }
  
};






// using FuncAVX = std::function<__m256d(__m256d)>;
// using FuncAVX_wo_checks = std::function<__m256d(__m256d)>;



// template <typename T, typename FuncAVX, typename FuncDouble>
template<typename T, typename WrapperType, typename DoubleWrapper>
inline  void fn_AVX2_row_or_col_vector(    Eigen::Ref<T>  x_ref,
                                           WrapperType fn_AVX,
                                           DoubleWrapper fn_double) {
  
       std::cout << "Entering fn_AVX2_row_or_col_vector" << std::endl;
  
       // auto avx_wrapper = [&fn_AVX](__m256d x) -> __m256d {
       //   return fn_AVX(x);
       // };
     
       // Eigen::Matrix<double, -1, 1> x = x_ref;
       // Eigen::Matrix<double, -1, 1> x_res = x_ref;
       //// Eigen::Ref<T> x_res_Ref(x_res); 
  
       const int N = x_ref.size();
       std::cout << "Vector size: " << N << std::endl;
       
       
       const int vect_size = 4;
       const double vect_siz_dbl = 4.0;
       const int N_divisible_by_vect_size = std::floor(static_cast<double>(N) / vect_siz_dbl) * vect_size;
        
       Eigen::Matrix<double, -1, 1> x_tail = Eigen::Matrix<double, -1, 1>::Zero(vect_size); // last vect_size elements
       {
           int counter = 0;
           for (int i = N - vect_size; i < N; ++i) {
             x_tail(counter) = x_ref(i);
             counter += 1;
           } 
       }
       
       if (N >= vect_size) {
         
               std::cout << "Processing with AVX2" << std::endl;
             
               try {
             
                     for (int i = 0; i + vect_size <= N_divisible_by_vect_size; i += vect_size) {
                       
                             std::cout << "Processing index " << i << std::endl;
                       
                          // try {
                             std::cout << "Loading data" << std::endl;
                             const __m256d AVX_array = _mm256_loadu_pd(&x_ref(i));
                             std::cout << "Calling AVX function" << std::endl; //// gets up to here fine, but fails after (so when calling "fn_AVX"). 
                             const __m256d AVX_array_out = fn_AVX(AVX_array);
                             // // Instead of using function pointer, do the operation directly (DEBUG ONLY)
                             // const __m256d two = _mm256_set1_pd(2.0);
                             // const __m256d AVX_array_out = _mm256_mul_pd(AVX_array, two);
                             std::cout << "Storing result" << std::endl;
                             _mm256_storeu_pd(&x_ref(i), AVX_array_out);
                           // } catch (...) {
                           //   std::cout << "Exception in AVX2 processing at index " << i << std::endl;
                           //   throw;
                           // }
                           
                           std::cout << "Completed iteration " << i/4 << std::endl;
                           std::cout << "Completed index " << i << std::endl;
                           
                     }
                     
                     if (N_divisible_by_vect_size != N) {    // Handle remainder
                       int counter = 0;
                       for (int i = N - vect_size; i < N; ++i) {
                         x_ref(i) =  fn_double(x_tail(counter));
                         counter += 1;
                       }
                     }
                     
                     std::cout << "AVX2 processing completed" << std::endl;
                 
             } catch (const std::exception& e) {
                   std::cout << "Exception caught: " << e.what() << std::endl;
                   throw;
             } catch (...) {
                   std::cout << "Unknown exception in AVX2 processing" << std::endl;
                   throw;
             }
         
       }  else {   // If N < vect_size, handle everything with scalar operations
         
             for (int i = 0; i < N; ++i) {
               x_ref(i) = fn_double(x_ref(i));
             }
         
       }
       
       // x = x_res;
       std::cout << "Exiting fn_AVX2_row_or_col_vector" << std::endl;
   
}
 
 
 
 
 
 
 
// template<typename T, typename FuncAVX, typename FuncDouble>
template<typename T, typename WrapperType, typename DoubleWrapper>
inline  void fn_AVX2_matrix(   Eigen::Ref<T> x,
                               WrapperType fn_AVX, 
                               DoubleWrapper fn_double) {
  
     // T x_res = x; // TEMP
     // Eigen::Ref<T> x_res_Ref(x_res); 
     
     const int n_rows = x.rows();
     const int n_cols = x.cols();
     
     if (n_rows > n_cols) { // if data in "long" format
       for (int j = 0; j < n_cols; ++j) {   
         Eigen::Matrix<double, -1, 1> x_col = x.col(j);
         using ColType = decltype(x_col);
         Eigen::Ref<Eigen::Matrix<double, -1, 1>> x_col_Ref(x_col); 
         fn_AVX2_row_or_col_vector<ColType>(x_col_Ref, fn_AVX, fn_double);
         x.col(j) = x_col_Ref;
       }
     } else { 
       for (int j = 0; j < n_rows; ++j) {
         Eigen::Matrix<double, 1, -1> x_row = x.row(j);
         using RowType = decltype(x_row);
         Eigen::Ref<Eigen::Matrix<double, 1, -1>> x_row_Ref(x_row); 
         fn_AVX2_row_or_col_vector<RowType>(x_row_Ref, fn_AVX, fn_double);
         x.row(j) = x_row_Ref;
       }
     }
     
     //  x = x_res_Ref; // re-assign - TEMP
   
} 
 
 
 
 
 
 
 
// template<typename T, typename FuncAVX, typename FuncDouble>
template<typename T, typename WrapperType, typename DoubleWrapper>
inline  void fn_AVX2_dbl_Eigen(   Eigen::Ref<T> x, 
                                         WrapperType fn_AVX, 
                                         DoubleWrapper fn_double) {
    
     // T x_res = x; // TEMP
     // Eigen::Ref<T> x_res_Ref(x_res); 
     
     constexpr int n_rows = T::RowsAtCompileTime;
     constexpr int n_cols = T::ColsAtCompileTime;
     
     if constexpr (n_rows == 1 && n_cols == -1) {   // Row vector case
        
          fn_AVX2_row_or_col_vector(x, fn_AVX, fn_double);
       
     } else if constexpr (n_rows == -1 && n_cols == 1) {  // Column vector case
       
          fn_AVX2_row_or_col_vector(x, fn_AVX, fn_double);
       
     } else {   // General matrix case
       
          fn_AVX2_matrix(x, fn_AVX, fn_double);
       
     }
     
     // x = x_res_Ref; // re-assign - TEMP
   
}
  
 
 

 
 
 
  
// template<typename FuncAVX, typename FuncDouble, typename FuncAVX_wo_checks, typename FuncDouble_wo_checks, typename T>
template<typename T, typename WrapperType_1, typename WrapperType_2, typename DoubleWrapper_1, typename DoubleWrapper_2>
inline void    fn_process_double_AVX2_sub_function(   Eigen::Ref<T> x,  
                                                      WrapperType_1   fn_fast_AVX2_function,
                                                      DoubleWrapper_1 fn_fast_double_function,
                                                      WrapperType_2   fn_fast_AVX2_function_wo_checks,
                                                      DoubleWrapper_2 fn_fast_double_function_wo_checks, 
                                                      const bool skip_checks) {
  
      // T x_res = x; // TEMP
      // Eigen::Ref<T> x_res_Ref(x_res); 
      
      if (skip_checks == false) {
        
           fn_AVX2_dbl_Eigen(x, fn_fast_AVX2_function, fn_fast_double_function);
        
      } else {
        
           fn_AVX2_dbl_Eigen(x, fn_fast_AVX2_function_wo_checks, fn_fast_double_function_wo_checks);
        
      }
      
      // x = x_res_Ref; // re-assign - TEMP
  
}

 


 

 
 
 
template <typename T>
inline  void       fn_process_Ref_double_AVX2(    Eigen::Ref<T> x,
                                                  const std::string &fn,
                                                  const bool &skip_checks) {
  
    std::cout << "Entering fn_process_Ref_double_AVX2" << std::endl;
  
    // T x_res = x; // TEMP
    // Eigen::Ref<T> x_res_Ref(x_res); 
  
    if        (fn == "test_simple") {    
          std::cout << "Calling test_simple function" << std::endl;
          try { 
            AVXWrapper AVX_wrapped(test_simple_AVX2);  
            AVXWrapper_wo_checks AVX_wrapped_wo_checks(test_simple_AVX2);
            DoubleWrapper double_wrapped(test_simple_scalar);
            DoubleWrapper_wo_checks double_wrapped_wo_checks(test_simple_scalar);
            
            fn_process_double_AVX2_sub_function(x, 
                                                AVX_wrapped,  double_wrapped,  
                                                AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
            // fn_process_double_AVX2_sub_function(x, test_simple_AVX2,  test_simple_scalar,   test_simple_AVX2, test_simple_scalar, skip_checks) ;
          } catch (const std::exception& e) { 
              std::cout << "Exception caught: " << e.what() << std::endl;
              throw;
          } catch (...) {
              std::cout << "Unknown exception caught" << std::endl;
              throw;
          }
    } else if (fn == "exp") {       
          AVXWrapper AVX_wrapped(fast_exp_1_AVX2);  
          AVXWrapper_wo_checks AVX_wrapped_wo_checks(fast_exp_1_wo_checks_AVX2);
          DoubleWrapper double_wrapped(mvp_std_exp);
          DoubleWrapper_wo_checks double_wrapped_wo_checks(mvp_std_exp);
          fn_process_double_AVX2_sub_function(x, 
                                              AVX_wrapped, double_wrapped, 
                                              AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
    } else if (fn == "log") {   
          AVXWrapper AVX_wrapped(fast_log_1_AVX2);  
          AVXWrapper_wo_checks AVX_wrapped_wo_checks(fast_log_1_wo_checks_AVX2);
          DoubleWrapper double_wrapped(mvp_std_log);
          DoubleWrapper_wo_checks double_wrapped_wo_checks(mvp_std_log);
          fn_process_double_AVX2_sub_function(x, 
                                              AVX_wrapped, double_wrapped, 
                                              AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
    } else if (fn == "log1p") {    
          AVXWrapper AVX_wrapped(fast_log1p_1_AVX2);  
          AVXWrapper_wo_checks AVX_wrapped_wo_checks(fast_log1p_1_wo_checks_AVX2);
          DoubleWrapper double_wrapped(mvp_std_log1p);
          DoubleWrapper_wo_checks double_wrapped_wo_checks(mvp_std_log1p);
          fn_process_double_AVX2_sub_function(x, 
                                              AVX_wrapped, double_wrapped, 
                                              AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
    } else if (fn == "log1m") {     
          AVXWrapper AVX_wrapped(fast_log1m_1_AVX2);  
          AVXWrapper_wo_checks AVX_wrapped_wo_checks(fast_log1m_1_wo_checks_AVX2);
          DoubleWrapper double_wrapped(mvp_std_log1m);
          DoubleWrapper_wo_checks double_wrapped_wo_checks(mvp_std_log1m);
          fn_process_double_AVX2_sub_function(x, 
                                              AVX_wrapped, double_wrapped, 
                                              AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
    } else if (fn == "logit") {        
          AVXWrapper AVX_wrapped(fast_logit_AVX2);  
          AVXWrapper_wo_checks AVX_wrapped_wo_checks(fast_logit_wo_checks_AVX2);
          DoubleWrapper double_wrapped(mvp_std_logit);
          DoubleWrapper_wo_checks double_wrapped_wo_checks(mvp_std_logit);
          fn_process_double_AVX2_sub_function(x, 
                                              AVX_wrapped, double_wrapped, 
                                              AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
    } else if (fn == "tanh") {  
          AVXWrapper AVX_wrapped(fast_tanh_AVX2);  
          AVXWrapper_wo_checks AVX_wrapped_wo_checks(fast_tanh_wo_checks_AVX2);
          DoubleWrapper double_wrapped(mvp_std_tanh);
          DoubleWrapper_wo_checks double_wrapped_wo_checks(mvp_std_tanh);
          fn_process_double_AVX2_sub_function(x, 
                                              AVX_wrapped, double_wrapped, 
                                              AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
    } else if (fn == "Phi_approx") {    
          AVXWrapper AVX_wrapped(fast_Phi_approx_AVX2);  
          AVXWrapper_wo_checks AVX_wrapped_wo_checks(fast_Phi_approx_wo_checks_AVX2);
          DoubleWrapper double_wrapped(mvp_std_Phi_approx);
          DoubleWrapper_wo_checks double_wrapped_wo_checks(mvp_std_Phi_approx);
          fn_process_double_AVX2_sub_function(x, 
                                              AVX_wrapped, double_wrapped, 
                                              AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
    } else if (fn == "log_Phi_approx") {      
          AVXWrapper AVX_wrapped(fast_log_Phi_approx_AVX2);  
          AVXWrapper_wo_checks AVX_wrapped_wo_checks(fast_log_Phi_approx_wo_checks_AVX2);
          DoubleWrapper double_wrapped(fast_log_Phi_approx);
          DoubleWrapper_wo_checks double_wrapped_wo_checks(fast_log_Phi_approx_wo_checks);
          fn_process_double_AVX2_sub_function(x, 
                                              AVX_wrapped, double_wrapped, 
                                              AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
    } else if (fn == "inv_Phi_approx") {      
          AVXWrapper AVX_wrapped(fast_inv_Phi_approx_AVX2);  
          AVXWrapper_wo_checks AVX_wrapped_wo_checks(fast_inv_Phi_approx_wo_checks_AVX2);
          DoubleWrapper double_wrapped(fast_inv_Phi_approx);
          DoubleWrapper_wo_checks double_wrapped_wo_checks(fast_inv_Phi_approx_wo_checks);
          fn_process_double_AVX2_sub_function(x, 
                                              AVX_wrapped, double_wrapped, 
                                              AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
    } else if (fn == "inv_Phi_approx_from_logit_prob") { 
          AVXWrapper AVX_wrapped(fast_inv_Phi_approx_from_logit_prob_AVX2);  
          AVXWrapper_wo_checks AVX_wrapped_wo_checks(fast_inv_Phi_approx_from_logit_prob_wo_checks_AVX2);
          DoubleWrapper double_wrapped(fast_inv_Phi_approx_from_logit_prob);
          DoubleWrapper_wo_checks double_wrapped_wo_checks(fast_inv_Phi_approx_from_logit_prob_wo_checks);
          fn_process_double_AVX2_sub_function(x, 
                                              AVX_wrapped, double_wrapped, 
                                              AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
    } else if (fn == "Phi") {             
          AVXWrapper AVX_wrapped(fast_Phi_AVX2);  
          AVXWrapper_wo_checks AVX_wrapped_wo_checks(fast_Phi_wo_checks_AVX2);
          DoubleWrapper double_wrapped(mvp_std_Phi);
          DoubleWrapper_wo_checks double_wrapped_wo_checks(mvp_std_Phi);
          fn_process_double_AVX2_sub_function(x, 
                                              AVX_wrapped, double_wrapped, 
                                              AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
    } else if (fn == "inv_Phi") {            
          AVXWrapper AVX_wrapped(fast_inv_Phi_wo_checks_AVX2);  
          AVXWrapper_wo_checks AVX_wrapped_wo_checks(fast_inv_Phi_wo_checks_AVX2);
          DoubleWrapper double_wrapped(mvp_std_inv_Phi);
          DoubleWrapper_wo_checks double_wrapped_wo_checks(mvp_std_inv_Phi);
          fn_process_double_AVX2_sub_function(x, 
                                              AVX_wrapped, double_wrapped, 
                                              AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
    } else if (fn == "inv_logit") {           
          AVXWrapper AVX_wrapped(fast_inv_logit_AVX2);  
          AVXWrapper_wo_checks AVX_wrapped_wo_checks(fast_inv_logit_wo_checks_AVX2);
          DoubleWrapper double_wrapped(mvp_std_inv_logit);
          DoubleWrapper_wo_checks double_wrapped_wo_checks(mvp_std_inv_logit);
          fn_process_double_AVX2_sub_function(x, 
                                              AVX_wrapped, double_wrapped, 
                                              AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
    } else if (fn == "log_inv_logit") {     
          AVXWrapper AVX_wrapped(fast_log_inv_logit_AVX2);  
          AVXWrapper_wo_checks AVX_wrapped_wo_checks(fast_log_inv_logit_wo_checks_AVX2);
          DoubleWrapper double_wrapped(fast_log_inv_logit);
          DoubleWrapper_wo_checks double_wrapped_wo_checks(fast_log_inv_logit_wo_checks);
          fn_process_double_AVX2_sub_function(x, 
                                              AVX_wrapped, double_wrapped, 
                                              AVX_wrapped_wo_checks, double_wrapped_wo_checks, skip_checks) ;
    }
 
    // x = x_res_Ref; // re-assign - TEMP
    
    std::cout << "Exiting fn_process_Ref_double_AVX2" << std::endl;

}
 


 
 

 
 


 
 
 
 
 
  
  


#endif



#endif

  
  