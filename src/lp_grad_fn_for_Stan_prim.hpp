#ifndef LP_GRAD_FN_FOR_STAN_PRIM_HPP
#define LP_GRAD_FN_FOR_STAN_PRIM_HPP

 
 
#include <stan/math/prim.hpp>
 

#include <Eigen/Dense>
#include <Eigen/Core>

#include <vector>

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/functor/apply_scalar_unary.hpp>

 
#include <typeinfo>
#include <sstream>
#include <stdexcept>
#include <stan/math/prim/err/invalid_argument.hpp>
 
#include <stan/math/prim/meta/is_matrix.hpp>
#include <stan/math/prim/meta/is_matrix_cl.hpp>
#include <stan/math/prim/meta/is_vector.hpp>
 
 
 
#include <stan/math/prim/fun/Eigen.hpp>
 
 
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
 
 
 
 
#include "eigen_config.hpp"


#include "general_functions/var_fns.hpp"
#include "general_functions/double_fns.hpp" 
 
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_Stan.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_Loop.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fast_and_approx_AVX512_AVX2_fns.hpp" // will only compile if AVX2 or AVX-512 is available 
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_SIMD_AVX2.hpp" // will only compile if AVX2 is available 
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_SIMD_AVX512.hpp" // will only compile if AVX-512 is available 
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_overall.hpp" 
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_log_sum_exp_dbl.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_log_sum_exp_SIMD.hpp"
 
#include "general_functions/structures.hpp"
#include "general_functions/array_creators_Eigen_fns.hpp"
#include "general_functions/misc_helper_fns_1.hpp"
 
////// Now load in (mostly) MVP-specific (and MVP-LC) model functions
#include "MVP_functions/MVP_manual_grad_calc_fns.hpp" 
#include "MVP_functions/MVP_log_scale_grad_calc_fns.hpp"
#include "MVP_functions/MVP_manual_trans_and_J_fns.hpp"
#include "MVP_functions/MVP_lp_grad_AD_fns.hpp"
#include "MVP_functions/MVP_lp_grad_MD_AD_fns.hpp"
#include "MVP_functions/MVP_lp_grad_log_scale_MD_AD_fns.hpp"
////// #include "MVP_functions/MVP_manual_Hessian_calc_fns.hpp"
#include "MVP_functions/MVP_lp_grad_multi_attempts.hpp"
 
////// Latent trait stuff
#include "LC_LT_functions/LC_LT_manual_grad_calc_fns.hpp"
#include "LC_LT_functions/LC_LT_log_scale_grad_calc_fns.hpp"
#include "LC_LT_functions/LC_LT_lp_grad_AD_fns.hpp"
#include "LC_LT_functions/LC_LT_lp_grad_MD_AD_fns.hpp"
#include "LC_LT_functions/LC_LT_lp_grad_log_scale_MD_AD_fns.hpp"
 
////// general lp_grad fn  
#if __has_include("bridgestan.h")
#define HAS_BRIDGESTAN_H 1
    #include "bridgestan.h" 
    #include "version.hpp"
    #include "model_rng.hpp" 
#else 
    #define HAS_BRIDGESTAN_H 0 
#endif


#if HAS_BRIDGESTAN_H
    #include "general_functions/Stan_model_helper_fns.hpp" 
#endif

#include "general_functions/lp_grad_model_selector.hpp"  


 
 
////////////////-------- first determine SIMD (i.e. vectorisation) type to use as global static variable - will add more in future -------------------------------------------
static const std::string vect_type = [] {
      #ifdef __AVX512F__
       return "AVX512"; 
      #elif defined(__AVX2__) && (!(defined(__AVX512F__)))
       return "AVX2";
      #else 
       return "Stan";
      #endif
}();
 

 
 
 
   
 
// //////////////  ---------  LC-MVP manual-gradient lp_grad function  --------------------------------------------------------------------------------------------------------------------


 
namespace stan {
namespace math {



Eigen::Matrix<double, -1, 1>     Stan_wrapper_lp_fn(                      const int Model_type_int,
                                                                          const int multi_attempts_int,
                                                                          const Eigen::Matrix<double, -1, 1> &theta_main_vec,
                                                                          const Eigen::Matrix<double, -1, 1> &theta_us_vec,
                                                                          const Eigen::Matrix<int, -1, -1>  &y,
                                                                          const int n_chunks,
                                                                          const double overflow_threshold,
                                                                          const double underflow_threshold,
                                                                          const double prev_prior_a,
                                                                          const double prev_prior_b,
                                                                          const Eigen::Matrix<double, -1, 1>  &lkj_cholesky_eta,
                                                                          const Eigen::Matrix<double, -1, -1> &n_covariates_per_outcome_vec,
                                                                          const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_mean,
                                                                          const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_sd,
                                                                          const std::vector<Eigen::Matrix<double, -1, -1>> &prior_for_corr_a,
                                                                          const std::vector<Eigen::Matrix<double, -1, -1>> &prior_for_corr_b,
                                                                          const std::vector<Eigen::Matrix<double, -1, -1>> &lb_corr,
                                                                          const std::vector<Eigen::Matrix<double, -1, -1>> &ub_corr,
                                                                          const std::vector<Eigen::Matrix<double, -1, -1>> &known_values,
                                                                          const std::vector<Eigen::Matrix<double, -1, -1>> &known_values_indicator,
                                                                          const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>  &X,
                                                                          std::ostream* pstream__ = nullptr
) {

   const int N = y.rows();
   const int n_nuisance = theta_us_vec.rows();
   const int n_params_main =  theta_main_vec.rows();
   const int n_params = n_params_main + n_nuisance;
   
   const int n_tests = y.cols();
   
   std::string Model_type;  
   if (Model_type_int == 1)  Model_type = "MVP";
   if (Model_type_int == 2)  Model_type = "LC_MVP";
   if (Model_type_int == 3)  Model_type = "latent_trait";
   
   bool multi_attempts = true;
   if (multi_attempts_int == 0) multi_attempts = false;

   const std::string grad_option = "all";
   const bool force_autodiff = false;
   const bool force_PartialLog = false;

   int n_class = 2;
   if (Model_type == "MVP") n_class = 1;
   
   Stan_model_struct Stan_model_as_cpp_struct; /// dummy struct

   /////  --------  create Model_fn_args_struct object --------------------------------------------
   Model_fn_args_struct Model_args_as_cpp_struct(N, n_nuisance, n_params_main, 
                                                 15, 4, 4, 13,
                                                 1, 1, 1, 1, 
                                                 1, 1, 7, 1, 
                                                 1, 1, 1, 1);
   
   Model_args_as_cpp_struct.n_nuisance = n_nuisance;
   Model_args_as_cpp_struct.n_params_main = n_params_main;

   Model_args_as_cpp_struct.Model_args_bools(0) =  false;  // exclude_priors
   Model_args_as_cpp_struct.Model_args_bools(1) =  false;  // CI
   Model_args_as_cpp_struct.Model_args_bools(2) =  false;  // corr_force_positive
   Model_args_as_cpp_struct.Model_args_bools(3) =  false;  // corr_prior_beta
   Model_args_as_cpp_struct.Model_args_bools(4) =  false;  // corr_prior_norm
   Model_args_as_cpp_struct.Model_args_bools(5) =  true;   // handle_numerical_issues
   Model_args_as_cpp_struct.Model_args_bools(6) =  false;  // skip_checks_exp
   Model_args_as_cpp_struct.Model_args_bools(7) =  false;  // skip_checks_log
   Model_args_as_cpp_struct.Model_args_bools(8) =  false;  // skip_checks_lse
   Model_args_as_cpp_struct.Model_args_bools(9) =  false;  // skip_checks_tanh
   Model_args_as_cpp_struct.Model_args_bools(10) = false;  // skip_checks_Phi
   Model_args_as_cpp_struct.Model_args_bools(11) = false;  // skip_checks_log_Phi
   Model_args_as_cpp_struct.Model_args_bools(12) = false;  // skip_checks_inv_Phi
   Model_args_as_cpp_struct.Model_args_bools(13) = false;  // skip_checks_inv_Phi_approx_from_logit_prob
   Model_args_as_cpp_struct.Model_args_bools(14) = false;  // debug

   Model_args_as_cpp_struct.Model_args_ints(0) = 1;  // n_cores
   Model_args_as_cpp_struct.Model_args_ints(1) = n_class;  // n_class
   Model_args_as_cpp_struct.Model_args_ints(2) = 5;  // ub_threshold_phi_approx
   Model_args_as_cpp_struct.Model_args_ints(3) = n_chunks;  // n_chunks

   Model_args_as_cpp_struct.Model_args_doubles(0) = prev_prior_a;
   Model_args_as_cpp_struct.Model_args_doubles(1) = prev_prior_b;
   Model_args_as_cpp_struct.Model_args_doubles(2) = overflow_threshold;
   Model_args_as_cpp_struct.Model_args_doubles(3) = underflow_threshold;

   const std::string Phi_type = "Phi";
   const std::string inv_Phi_type = "inv_Phi";
   const std::string nuisance_transformation = "Phi";
   
  /////  const std::string vect_type = "Stan";

   Model_args_as_cpp_struct.Model_args_strings(0) = vect_type; // vect_type
   Model_args_as_cpp_struct.Model_args_strings(1) = Phi_type; // Phi_type
   Model_args_as_cpp_struct.Model_args_strings(2) = inv_Phi_type; // inv_Phi_type
   Model_args_as_cpp_struct.Model_args_strings(3) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(4) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(5) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(6) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(7) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(8) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(9) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(10) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(12) = nuisance_transformation;

   Model_args_as_cpp_struct.Model_args_col_vecs_double[0] = lkj_cholesky_eta;

   Eigen::Matrix<int, -1, -1> n_covariates_per_outcome_vec_int = n_covariates_per_outcome_vec.cast<int>();
   Model_args_as_cpp_struct.Model_args_mats_int[0] = n_covariates_per_outcome_vec_int;

   Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0] = prior_coeffs_mean;
   Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1] = prior_coeffs_sd;
   Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2] = prior_for_corr_a;
   Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3] = prior_for_corr_b;
   Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4] = lb_corr;
   Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5] = ub_corr;
   Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6] = known_values;
   
   std::vector<Eigen::Matrix<int, -1, -1>> known_values_indicator_int = vec_of_mats<int>(n_tests, n_tests, n_class);
   for (int c = 0; c < n_class; c++) {
     known_values_indicator_int[c] = known_values_indicator[0].cast<int>();
   } 
   Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0] = known_values_indicator_int;

   Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0] = X;

   /////  --------  call lp_grad function  --------------------------------
   Eigen::Matrix<double, -1, 1> lp_grad_outs = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);

   fn_lp_grad_InPlace(   lp_grad_outs,  Model_type, 
                         force_autodiff, force_PartialLog, multi_attempts, 
                         theta_main_vec, theta_us_vec, y, grad_option,
                         Model_args_as_cpp_struct, Stan_model_as_cpp_struct);
   
   Eigen::Matrix<double, -1, 1> log_lik = lp_grad_outs.tail(N);
   double log_posterior = lp_grad_outs(0);
   
   Eigen::Matrix<double, -1, 1> outs(1 + N);
   outs(0) = log_posterior;
   outs.tail(N) = log_lik;
   
   return outs;

   
}








}  // namespace math
}  // namespace stan
// 


// 
// 
// 





 
#endif
    
     

 
