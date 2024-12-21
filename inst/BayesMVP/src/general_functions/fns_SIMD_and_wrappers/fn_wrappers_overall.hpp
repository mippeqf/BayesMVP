#pragma once 


#ifndef FN_WRAPPERS_OVERALL_HPP
#define FN_WRAPPERS_OVERALL_HPP

 
 
  
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


 

 
 
 
 
 


 
///// ----------------------------------------------------------------------------- Colvec function callers / wrappers

 


template <typename T>
ALWAYS_INLINE  void               fn_EIGEN_Ref_double(         Eigen::Ref<T> x,
                                                        const std::string &fn,
                                                        const std::string &vect_type,
                                                        const bool &skip_checks) {
   
   // T  x_copy = x; /// make a copy (for debug)
  
 /////  stan::math::check_finite(fn.c_str(), "x", x);  // using c_str() to convert std::string to const char*

  if (fn == "inv_Phi_from_log_prob") {

      x = stan::math::std_normal_log_qf(x); 
      //return x;

  } else {

    if (vect_type == "Stan") {

         fn_void_Ref_double_Stan(Eigen::Ref<T>(x), fn, skip_checks);
        // return x;
 
    } else if (vect_type == "AVX2" ) { // use AVX-512 or AVX2 or loop (i.e., rely on automatic vectorisation)

#if defined(__AVX2__) && ( !(defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)) ) // use AVX2
           fn_return_Ref_double_AVX2(Eigen::Ref<T>(x), fn, skip_checks);
          // return x;
#endif

    } else if (vect_type == "AVX512" ) { // use AVX-512 or AVX2 or loop (i.e., rely on automatic vectorisation)

#if defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)
          fn_return_Ref_double_AVX512(Eigen::Ref<T>(x), fn, skip_checks);
         // return x;
#endif

    } else {
      
          fn_return_Loop(Eigen::Ref<T>(x), fn, skip_checks);
 
        // throw std::invalid_argument( os.str() ); /// note: std::invalid_argument doesnt seem to work w/ Stan math lib

       //return x;

    }

  }


  //  return x;

}


 

 
 
 ////// ------- "master" function  w/ return  ------------------------------------------------------------- 
 
 
 //// R-value
 template <typename T>
 ALWAYS_INLINE  auto          fn_EIGEN_double(              T  &&x_R_val,
                                                     const std::string &fn,
                                                     const std::string &vect_type = "Stan",
                                                     const bool &skip_checks = false) {
   
   using matrix_type = Eigen::Matrix<double, -1, -1>;
   matrix_type x_matrix = x_R_val;   
   fn_EIGEN_Ref_double(Eigen::Ref<matrix_type>(x_matrix), fn, vect_type, skip_checks);
   return x_matrix;
   
 }
 
 //// Eigen Ref (this will also accept L_value [&T] as well as other types)
 template <typename T>
 ALWAYS_INLINE  auto          fn_EIGEN_double(  Eigen::Ref<T> x_L_val,
                                         const std::string &fn,
                                         const std::string &vect_type = "Stan",
                                         const bool &skip_checks = false) {
   
   fn_EIGEN_Ref_double(x_L_val, fn, vect_type, skip_checks);
   return x_L_val;
   
 }
 
 //// const Eigen Ref (this will also accept L_value [&T] as well as other types)
 template <typename T>
 ALWAYS_INLINE  auto          fn_EIGEN_double(  const Eigen::Ref<const T> x_L_val,
                                         const std::string &fn,
                                         const std::string &vect_type = "Stan",
                                         const bool &skip_checks = false) {
   
   T x_copy = x_L_val;
   fn_EIGEN_Ref_double(Eigen::Ref<T>(x_copy), fn, vect_type, skip_checks);
   return x_copy;
   
 }
 
 
 //// blocks
 template <typename T, int n_rows = Eigen::Dynamic, int n_cols = Eigen::Dynamic>
 ALWAYS_INLINE auto  fn_EIGEN_double(                       Eigen::Ref<Eigen::Block<T, n_rows, n_cols>> x_Ref,
                                                     const std::string &fn,
                                                     const std::string &vect_type = "Stan",
                                                     const bool &skip_checks = false) {
   
   T x_matrix = x_Ref;
   fn_EIGEN_Ref_double(Eigen::Ref<T>(x_matrix), fn, vect_type, skip_checks);
   return x_matrix;
   
 }
 
 
 
 //// arrays
 template <int n_rows = Eigen::Dynamic, int n_cols = Eigen::Dynamic>
 ALWAYS_INLINE auto  fn_EIGEN_double(               const Eigen::Array<double, n_rows, n_cols> &x,
                                             const std::string &fn, 
                                             const std::string &vect_type = "Stan",
                                             const bool &skip_checks = false) {
   
   using T = Eigen::Matrix<double, n_rows, n_cols>;
   T x_matrix = x.matrix();
   fn_EIGEN_Ref_double(Eigen::Ref<T>(x_matrix), fn, vect_type, skip_checks);
   return x_matrix;
   
   
 }
 
 
 // New overload for general expressions
 template <typename Derived>
 ALWAYS_INLINE auto fn_EIGEN_double(const Eigen::EigenBase<Derived> &x,
                             const std::string &fn,
                             const std::string &vect_type = "Stan",
                             const bool &skip_checks = false) {
   
   using T = Eigen::Matrix<typename Derived::Scalar, 
                           Derived::RowsAtCompileTime, 
                           Derived::ColsAtCompileTime>;
   T x_copy = x;
   fn_EIGEN_Ref_double(Eigen::Ref<T>(x_copy), fn, vect_type, skip_checks);
   return x_copy;
 }
 
 
 
 // Additional Matrix expression overload
 template <typename Derived>
 ALWAYS_INLINE auto fn_EIGEN_double(const Eigen::MatrixBase<Derived> &x,
                             const std::string &fn,
                             const std::string &vect_type = "Stan",
                             const bool &skip_checks = false) {
   
   using T = Eigen::Matrix<typename Derived::Scalar,
                           Derived::RowsAtCompileTime,
                           Derived::ColsAtCompileTime>;
   T x_copy = x;
   fn_EIGEN_Ref_double(Eigen::Ref<T>(x_copy), fn, vect_type, skip_checks);
   return x_copy;
 }
 
 
 
 // Additional overload for array expressions
 template <typename Derived>
 ALWAYS_INLINE auto fn_EIGEN_double(const Eigen::ArrayBase<Derived> &x,
                             const std::string &fn,
                             const std::string &vect_type = "Stan", 
                             const bool &skip_checks = false) {
   
   using T = Eigen::Matrix<typename Derived::Scalar,
                           Derived::RowsAtCompileTime,
                           Derived::ColsAtCompileTime>;
   T x_copy = x.matrix();
   fn_EIGEN_Ref_double(Eigen::Ref<T>(x_copy), fn, vect_type, skip_checks);
   return x_copy;
 }
 
 
 
 
 
  
 
 
  
  
 
 
 





#endif

  
  
  
  
  