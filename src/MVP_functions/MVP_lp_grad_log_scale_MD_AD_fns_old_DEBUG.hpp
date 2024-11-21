
#pragma once


 
// [[Rcpp::depends(StanHeaders)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppEigen)]]

 
 

#include <stan/math/rev.hpp>


#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/prim/err/check_pos_definite.hpp>
#include <stan/math/prim/err/check_square.hpp>
#include <stan/math/prim/err/check_symmetric.hpp>


#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/sqrt.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/transpose.hpp>
#include <stan/math/prim/fun/dot_product.hpp>
#include <stan/math/prim/fun/norm2.hpp>
#include <stan/math/prim/fun/diagonal.hpp>
#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/eigenvalues_sym.hpp>
#include <stan/math/prim/fun/diag_post_multiply.hpp>




#include <stan/math/prim/prob/multi_normal_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/lkj_corr_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/weibull_lpdf.hpp>
#include <stan/math/prim/prob/gamma_lpdf.hpp>
#include <stan/math/prim/prob/beta_lpdf.hpp>



 
#include <Eigen/Dense>
 



#include <unsupported/Eigen/SpecialFunctions>


 
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
 
 

// [[Rcpp::plugins(cpp17)]]



 
using namespace Eigen;

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE




using std_vec_of_EigenVecs = std::vector<Eigen::Matrix<double, -1, 1>>;
using std_vec_of_EigenVecs_int = std::vector<Eigen::Matrix<int, -1, 1>>;

using std_vec_of_EigenMats = std::vector<Eigen::Matrix<double, -1, -1>>;
using std_vec_of_EigenMats_int = std::vector<Eigen::Matrix<int, -1, -1>>;

using two_layer_std_vec_of_EigenVecs =  std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>;
using two_layer_std_vec_of_EigenVecs_int = std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>;

using two_layer_std_vec_of_EigenMats = std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>;
using two_layer_std_vec_of_EigenMats_int = std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>;


using three_layer_std_vec_of_EigenVecs =  std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>>;
using three_layer_std_vec_of_EigenVecs_int =  std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>>;

using three_layer_std_vec_of_EigenMats = std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>>; 
using three_layer_std_vec_of_EigenMats_int = std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>>;










 

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///// This model ccan be either the "standard" MVP model or the latent class MVP model (w/ 2 classes) for analysis of test accuracy data. 
void                             fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process(    Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat ,
                                                                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                                        const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                                        const std::string grad_option,
                                                                                                        const Model_fn_args_struct &Model_args_as_cpp_struct
) {

  out_mat.setZero();
  
 //// stan::math::nested_rev_autodiff nested;
  
  //// important params   
  const int N = y_ref.rows();
  const int n_tests = y_ref.cols();
  const int n_us = theta_us_vec_ref.rows()  ; 
  const int n_params_main =  theta_main_vec_ref.rows()  ; 
  const int n_params = n_params_main + n_us;
  
  //////////////  access elements from struct and read 
  const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>  &X =  Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0]; 
  
  const bool exclude_priors = Model_args_as_cpp_struct.Model_args_bools(0);
  const bool CI =             Model_args_as_cpp_struct.Model_args_bools(1);
  const bool corr_force_positive = Model_args_as_cpp_struct.Model_args_bools(2);
  const bool corr_prior_beta = Model_args_as_cpp_struct.Model_args_bools(3);
  const bool corr_prior_norm = Model_args_as_cpp_struct.Model_args_bools(4);
  const bool handle_numerical_issues = Model_args_as_cpp_struct.Model_args_bools(5);
  const bool skip_checks_exp =   Model_args_as_cpp_struct.Model_args_bools(6);
  const bool skip_checks_log =   Model_args_as_cpp_struct.Model_args_bools(7);
  const bool skip_checks_lse =   Model_args_as_cpp_struct.Model_args_bools(8);
  const bool skip_checks_tanh =  Model_args_as_cpp_struct.Model_args_bools(9);
  const bool skip_checks_Phi =  Model_args_as_cpp_struct.Model_args_bools(10);
  const bool skip_checks_log_Phi = Model_args_as_cpp_struct.Model_args_bools(11);
  const bool skip_checks_inv_Phi = Model_args_as_cpp_struct.Model_args_bools(12);
  const bool skip_checks_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_bools(13);
  const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
  
  const int n_cores = Model_args_as_cpp_struct.Model_args_ints(0);
  const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
  const int ub_threshold_phi_approx = Model_args_as_cpp_struct.Model_args_ints(2);
  const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
  
  const double prev_prior_a = Model_args_as_cpp_struct.Model_args_doubles(0);
  const double prev_prior_b = Model_args_as_cpp_struct.Model_args_doubles(1);
  const double overflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(2);
  const double underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(3);
  
  std::string vect_type = Model_args_as_cpp_struct.Model_args_strings(0); // NOT const 
  const std::string &Phi_type = Model_args_as_cpp_struct.Model_args_strings(1);
  const std::string &inv_Phi_type = Model_args_as_cpp_struct.Model_args_strings(2);
  std::string vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);  // NOT const 
  std::string vect_type_log = Model_args_as_cpp_struct.Model_args_strings(4);  // NOT const 
  std::string vect_type_lse = Model_args_as_cpp_struct.Model_args_strings(5);  // NOT const 
  std::string vect_type_tanh = Model_args_as_cpp_struct.Model_args_strings(6);  // NOT const 
  std::string vect_type_Phi = Model_args_as_cpp_struct.Model_args_strings(7);  // NOT const 
  std::string vect_type_log_Phi = Model_args_as_cpp_struct.Model_args_strings(8); // NOT const 
  std::string vect_type_inv_Phi = Model_args_as_cpp_struct.Model_args_strings(9);  // NOT const 
  std::string vect_type_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_strings(10);  // NOT const 
  // const std::string grad_option =  Model_args_as_cpp_struct.Model_args_strings(11);
  const std::string nuisance_transformation =   Model_args_as_cpp_struct.Model_args_strings(12);
  
  const Eigen::Matrix<double, -1, 1>  &lkj_cholesky_eta =   Model_args_as_cpp_struct.Model_args_col_vecs_double[0];
  
  // const Eigen::Matrix<double, -1, -1> &LT_b_priors_shape  = Model_args_as_cpp_struct.Model_args_mats_double[0]; 
  // const Eigen::Matrix<double, -1, -1> &LT_b_priors_scale  = Model_args_as_cpp_struct.Model_args_mats_double[1]; 
  // const Eigen::Matrix<double, -1, -1> &LT_known_bs_indicator = Model_args_as_cpp_struct.Model_args_mats_double[2]; 
  // const Eigen::Matrix<double, -1, -1> &LT_known_bs_values = Model_args_as_cpp_struct.Model_args_mats_double[3]; 
  
  const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0]; 
  
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &prior_coeffs_mean  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &prior_coeffs_sd   =  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &prior_for_corr_a   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &prior_for_corr_b   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &lb_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &ub_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &known_values    = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6]; 
  
  const std::vector<Eigen::Matrix<int, -1, -1 >> &known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];
  
  
  //////////////
  const int n_corrs =  n_class * n_tests * (n_tests - 1) * 0.5;
  
  int n_covariates_total_nd, n_covariates_total_d, n_covariates_total;
  int n_covariates_max_nd, n_covariates_max_d, n_covariates_max;
  
  if (n_class > 1)  {
      n_covariates_total_nd = n_covariates_per_outcome_vec.row(0).sum();
      n_covariates_total_d = n_covariates_per_outcome_vec.row(1).sum();
      n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
      
      n_covariates_max_nd = n_covariates_per_outcome_vec.row(0).maxCoeff();
      n_covariates_max_d = n_covariates_per_outcome_vec.row(1).maxCoeff();
      n_covariates_max = std::max(n_covariates_max_nd, n_covariates_max_d);
  } else { 
      n_covariates_total = n_covariates_per_outcome_vec.sum();
      n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();
  }
  
  const double sqrt_2_pi_recip = 1.0 / sqrt(2.0 * M_PI);
  const double sqrt_2_recip = 1.0 / stan::math::sqrt(2.0);
  const double minus_sqrt_2_recip = -sqrt_2_recip;
  const double a = 0.07056;
  const double b = 1.5976;
  const double a_times_3 = 3.0 * 0.07056;
  const double s = 1.0 / 1.702;
  
  //// ---- determine chunk size -------------------------- 
  const int desired_n_chunks = n_chunks;
  
  int vec_size;
  if (vect_type == "AVX512") { 
    vec_size = 8;
  } else  if (vect_type == "AVX2") { 
    vec_size = 4;
  } else  if (vect_type == "AVX") { 
    vec_size = 2;
  } else { 
    vec_size = 1;
  }
  
  const double N_double = static_cast<double>(N);
  const double vec_size_double =   static_cast<double>(vec_size);
  const double desired_n_chunks_double = static_cast<double>(desired_n_chunks);
  
  int normal_chunk_size = vec_size_double * std::floor(N_double / (vec_size_double * desired_n_chunks_double));    // Make sure main chunks are divisible by 8
  int n_full_chunks = std::floor(N_double / static_cast<double>(normal_chunk_size));    ///  How many complete chunks we can have
  int last_chunk_size = N_double - (static_cast<double>(n_full_chunks) * static_cast<double>(normal_chunk_size));  //// remainder
  
  int n_total_chunks;
  if (last_chunk_size == 0) { 
    n_total_chunks = n_full_chunks;
  } else { 
    n_total_chunks = n_full_chunks + 1;
  }  
 
  int chunk_size = normal_chunk_size; // set initial chunk_size (this may be modified later so non-const)
  int chunk_size_orig = normal_chunk_size;     // store original chunk size for indexing
  
  if (desired_n_chunks == 1) { 
        chunk_size = N;
        chunk_size_orig = N;
        normal_chunk_size = N;
        last_chunk_size = N;
        n_total_chunks = 1;
        n_full_chunks = 1;
  } 
  
  //////////////
  // corrs
  Eigen::Matrix<double, -1, 1  >  Omega_raw_vec_double = theta_main_vec_ref.head(n_corrs); // .cast<double>();

  // coeffs
  std::vector<Eigen::Matrix<double, -1, -1 > > beta_double_array = vec_of_mats_double(n_covariates_max, n_tests,  n_class);

  {
    int i = n_corrs;
    for (int c = 0; c < n_class; ++c) {
      for (int t = 0; t < n_tests; ++t) {
        for (int k = 0; k < n_covariates_per_outcome_vec(c, t); ++k) {
          beta_double_array[c](k, t) = theta_main_vec_ref(i);
          i += 1;
        }
      }
    }
  }

  // prev
  double u_prev_diseased = theta_main_vec_ref(n_params_main - 1);

  Eigen::Matrix<double, -1, 1 >  target_AD_grad(n_corrs);
  stan::math::var target_AD = 0.0;
  double grad_prev_AD = 0.0;

  int dim_choose_2 = n_tests * (n_tests - 1) * 0.5 ;
  std::vector<Eigen::Matrix<double, -1, -1 > > deriv_L_wrt_unc_full = vec_of_mats_double(dim_choose_2 + n_tests, dim_choose_2, n_class);
  std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_double = vec_of_mats_double(n_tests, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_recip_double = L_Omega_double;
  std::vector<Eigen::Matrix<double, -1, -1 > > log_abs_L_Omega_double = L_Omega_double;
  std::vector<Eigen::Matrix<double, -1, -1 > > sign_L_Omega_double = L_Omega_double;

  double log_jac_p_double = 0.0;

  {     ////////////////////////// local AD block
    
   stan::math::start_nested();  ////////////////////////
 
    
    Eigen::Matrix<stan::math::var, -1, 1  >  Omega_raw_vec_var =  stan::math::to_var(Omega_raw_vec_double) ;
    std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > Omega_unconstrained_var = fn_convert_std_vec_of_corrs_to_3d_array_var(Eigen_vec_to_std_vec_var(Omega_raw_vec_var),  n_tests, n_class);
    std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > L_Omega_var = vec_of_mats_var(n_tests, n_tests, n_class);
    std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_var   = vec_of_mats_var(n_tests, n_tests, n_class);

    for (int c = 0; c < n_class; ++c) {
      
            Eigen::Matrix<stan::math::var, -1, -1 >  ub = stan::math::to_var(ub_corr[c]);
            Eigen::Matrix<stan::math::var, -1, -1 >  lb = stan::math::to_var(lb_corr[c]);
            Eigen::Matrix<stan::math::var, -1, -1  >  Chol_Schur_outs =  Pinkney_LDL_bounds_opt(n_tests, lb, ub, Omega_unconstrained_var[c], known_values_indicator[c], known_values[c]) ;
            L_Omega_var[c]   =  Chol_Schur_outs.block(1, 0, n_tests, n_tests);
            Omega_var[c] =   L_Omega_var[c] * L_Omega_var[c].transpose() ;
            
            target_AD +=   Chol_Schur_outs(0, 0); // now can set prior directly on Omega
            
    }

    for (int c = 0; c < n_class; ++c) {
      
              if ( (corr_prior_beta == false)   &&  (corr_prior_norm == false) ) {
                target_AD +=  stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c], lkj_cholesky_eta(c)) ;
              } else if ( (corr_prior_beta == true)   &&  (corr_prior_norm == false) ) {
                for (int i = 1; i < n_tests; i++) {
                  for (int j = 0; j < i; j++) {
                    target_AD +=  stan::math::beta_lpdf(  (Omega_var[c](i, j) + 1)/2, prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
                  }
                }
                //  Jacobian for  Omega -> L_Omega transformation for prior log-densities (since both LKJ and truncated normal prior densities are in terms of Omega, not L_Omega)
                Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
                for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i+1) ) * stan::math::log(L_Omega_var[c](i, i));
                target_AD  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
              } else if  ( (corr_prior_beta == false)   &&  (corr_prior_norm == true) ) {
                for (int i = 1; i < n_tests; i++) {
                  for (int j = 0; j < i; j++) {
                    target_AD +=  stan::math::normal_lpdf(  Omega_var[c](i, j), prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
                  }
                }
                Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
                for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i+1) ) * stan::math::log(L_Omega_var[c](i, i));
                target_AD  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
              }
              
    }

    ///////////////////////
    target_AD.grad();   // differentiating this (i.e. NOT wrt this!! - this is the subject)
    target_AD_grad =  Omega_raw_vec_var.adj();    // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
    stan::math::set_zero_all_adjoints();
    //////////////////////////////////////////////////////////// end of AD part

    /////////////  prev stuff  ---- vars
    if (n_class > 1) {  //// if latent class
      
                std::vector<stan::math::var> 	 u_prev_var_vec_var(n_class, 0.0);
                std::vector<stan::math::var> 	 prev_var_vec_var(n_class, 0.0);
                std::vector<stan::math::var> 	 tanh_u_prev_var(n_class, 0.0);
                Eigen::Matrix<stan::math::var, -1, -1>	 prev_var = Eigen::Matrix<stan::math::var, -1, -1>::Zero(1, 2);
                stan::math::var tanh_pu_deriv_var = 0.0;
                stan::math::var deriv_p_wrt_pu_var = 0.0;
                stan::math::var tanh_pu_second_deriv_var = 0.0;
                stan::math::var log_jac_p_deriv_wrt_pu_var = 0.0;
                stan::math::var log_jac_p_var = 0.0;
                stan::math::var target_AD_prev = 0.0;
          
                u_prev_var_vec_var[1] =  stan::math::to_var(u_prev_diseased);
                tanh_u_prev_var[1] = ( stan::math::exp(2.0*u_prev_var_vec_var[1] ) - 1.0) / ( stan::math::exp(2*u_prev_var_vec_var[1] ) + 1.0) ;
                u_prev_var_vec_var[0] =   0.5 *  stan::math::log( (1.0 + ( (1.0 - 0.5 * ( tanh_u_prev_var[1] + 1.0))*2.0 - 1.0) ) / (1.0 - ( (1.0 - 0.5 * ( tanh_u_prev_var[1] + 1.0))*2.0 - 1.0) ) )  ;
                tanh_u_prev_var[0] = (stan::math::exp(2.0*u_prev_var_vec_var[0] ) - 1.0) / ( stan::math::exp(2*u_prev_var_vec_var[0] ) + 1.0) ;
                
                prev_var_vec_var[1] =  0.5 * ( tanh_u_prev_var[1] + 1.0);
                prev_var_vec_var[0] =  0.5 * ( tanh_u_prev_var[0] + 1.0);
                prev_var(0,1) =  prev_var_vec_var[1];
                prev_var(0,0) =  prev_var_vec_var[0];
                
                tanh_pu_deriv_var = ( 1.0 - (tanh_u_prev_var[1] * tanh_u_prev_var[1])  );
                deriv_p_wrt_pu_var = 0.5 *  tanh_pu_deriv_var;
                tanh_pu_second_deriv_var  = -2.0 * tanh_u_prev_var[1]  * tanh_pu_deriv_var;
                log_jac_p_deriv_wrt_pu_var  = ( 1.0 / deriv_p_wrt_pu_var) * 0.5 * tanh_pu_second_deriv_var; // for gradient of u's
                log_jac_p_var =    stan::math::log( deriv_p_wrt_pu_var );
                log_jac_p_double =  log_jac_p_var.val() ; // = 0.0;
          
    
            
                target_AD_prev = beta_lpdf(prev_var(0, 1), prev_prior_a, prev_prior_b)  ;// + log_jac_p_var ; // weakly informative prior - helps avoid boundaries with slight negative skew (for lower N)
                //  target_AD_prev += log_jac_p_var ;
                target_AD  +=  target_AD_prev;
                ///////////////////////
                target_AD_prev.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
                grad_prev_AD  =  u_prev_var_vec_var[1].adj() - u_prev_var_vec_var[0].adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
                stan::math::set_zero_all_adjoints();
    
    } 
    //////////////////////////////////////////////////////////// end of AD part

    for (int c = 0; c < n_class; ++c) {
      int cnt_1 = 0;
      for (int k = 0; k < n_tests; k++) {
        for (int l = 0; l < k + 1; l++) {
          (  L_Omega_var[c](k, l)).grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
          int cnt_2 = 0;
          for (int i = 1; i < n_tests; i++) {
            for (int j = 0; j < i; j++) {
              deriv_L_wrt_unc_full[c](cnt_1, cnt_2)  =   Omega_unconstrained_var[c](i, j).adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
              cnt_2 += 1;
            }
          }
          stan::math::set_zero_all_adjoints();
          cnt_1 += 1;
        }
      }
    }
    ///////////////// get cholesky factor's (lower-triangular) of corr matrices
    // convert to 3d var array
    for (int c = 0; c < n_class; ++c) {
      for (int t1 = 0; t1 < n_tests; ++t1) {
        for (int t2 = 0; t2 < n_tests; ++t2) {
          L_Omega_double[c](t1, t2) =   L_Omega_var[c](t1, t2).val()  ;
          log_abs_L_Omega_double[c](t1, t2) =   stan::math::log(stan::math::fabs( L_Omega_double[c](t1, t2) ))  ;
          sign_L_Omega_double[c](t1, t2) = stan::math::sign( L_Omega_double[c](t1, t2) );
          L_Omega_recip_double[c](t1, t2) =   1.0 / L_Omega_double[c](t1, t2) ;
        }
      }
      log_abs_L_Omega_double[c] =    log_abs_L_Omega_double[c].array().min(700.0).max(-700.0);
    }

   //   stan::math::recover_memory();
   
  stan::math::recover_memory_nested();  //////////////////////////////////////////
  
  }   //////////////////////////  end of local AD block

  /////////////  prev stuff
  std::vector<double> 	 u_prev_var_vec(n_class, 0.0);
  std::vector<double> 	 prev_var_vec(n_class, 0.0);
  std::vector<double> 	 tanh_u_prev(n_class, 0.0);
  Eigen::Matrix<double, -1, -1>	 prev = Eigen::Matrix<double, -1, -1>::Zero(1, n_class);
  double tanh_pu_deriv = 0.0;
  double deriv_p_wrt_pu_double = 0.0;
  double tanh_pu_second_deriv = 0.0;
  double log_jac_p_deriv_wrt_pu = 0.0;
  double log_jac_p = 0.0;

  if (n_class > 1) {  //// if latent class
    
    u_prev_var_vec[1] =  (double) u_prev_diseased ;
    tanh_u_prev[1] = stan::math::tanh(u_prev_var_vec[1]); //  ( exp(2.0*u_prev_var_vec[1] ) - 1.0) / ( exp(2.0*u_prev_var_vec[1] ) + 1.0) ;
    u_prev_var_vec[0] =   0.5 *  stan::math::log( (1.0 + ( (1.0 - 0.5 * ( tanh_u_prev[1] + 1.0))*2.0 - 1.0) ) / (1.0 - ( (1.0 - 0.5 * ( tanh_u_prev[1] + 1.0))*2.0 - 1.0) ) )  ;
    tanh_u_prev[0] = stan::math::tanh(u_prev_var_vec[0]); //  (exp(2.0*u_prev_var_vec[0] ) - 1.0) / ( exp(2.0*u_prev_var_vec[0] ) + 1.0) ;
    
    prev_var_vec[1] =  0.5 * ( tanh_u_prev[1] + 1.0);
    prev_var_vec[0] =  0.5 * ( tanh_u_prev[0] + 1.0);
    prev(0,1) =  prev_var_vec[1];
    prev(0,0) =  prev_var_vec[0];
    
    tanh_pu_deriv = ( 1.0 - (tanh_u_prev[1] * tanh_u_prev[1])  );
    deriv_p_wrt_pu_double = 0.5 *  tanh_pu_deriv;
    tanh_pu_second_deriv  = -2.0 * tanh_u_prev[1]  * tanh_pu_deriv;
    log_jac_p_deriv_wrt_pu  = ( 1.0 / deriv_p_wrt_pu_double) * 0.5 * tanh_pu_second_deriv; // for gradient of u's
    log_jac_p =    stan::math::log( deriv_p_wrt_pu_double );
    
  }

  ///////////////////////////////////////////////////////////////////////// prior densities
  double prior_densities = 0.0;

  if (exclude_priors == false) {
    ///////////////////// priors for coeffs
    double prior_densities_coeffs = 0.0;
    for (int c = 0; c < n_class; c++) {
      for (int t = 0; t < n_tests; t++) {
        for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++) {
          prior_densities_coeffs  += stan::math::normal_lpdf(beta_double_array[c](k, t), prior_coeffs_mean[c](k, t), prior_coeffs_sd[c](k, t));
        }
      }
    }
    double prior_densities_corrs = target_AD.val();
    prior_densities = prior_densities_coeffs  +      prior_densities_corrs ;     // total prior densities and Jacobian adjustments
  }

  
  
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////// likelihood
  double log_prob_out = 0.0;
  
  Eigen::Matrix<double, -1, -1 >  log_prev = stan::math::log(prev);
  
  //// define unconstrained nuisance parameter vec 
  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec = theta_us_vec_ref; 

  
  ///////////////////////////////////////////////
  Eigen::Matrix<double , -1, 1>   beta_grad_vec   =  Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);  //
  std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array =  vec_of_mats<double>(n_covariates_max, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array =  vec_of_mats<double>(n_tests, n_tests, n_class);
  Eigen::Matrix<double, -1, 1 > L_Omega_grad_vec(n_corrs + (2 * n_tests));
  Eigen::Matrix<double, -1, 1 > U_Omega_grad_vec(n_corrs);
  Eigen::Matrix<double, -1, 1>  prev_unconstrained_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(2); //
  Eigen::Matrix<double, -1, 1>  prev_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(2); //
  Eigen::Matrix<double,  1, 1>  prev_unconstrained_grad_vec_out =   Eigen::Matrix<double, 1, 1>::Zero(2 - 1); //
  // ///////////////////////////////////////////////
  Eigen::Matrix<double , -1, 1>   log_abs_beta_grad_vec   =  Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);  //
  std::vector<Eigen::Matrix<double, -1, -1>> log_abs_beta_grad_array =  vec_of_mats<double>(n_covariates_max, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> log_abs_U_Omega_grad_array =  vec_of_mats<double>(n_tests, n_tests, n_class);
  Eigen::Matrix<double, -1, 1 > log_abs_L_Omega_grad_vec(n_corrs + (2 * n_tests));
  Eigen::Matrix<double, -1, 1 > log_abs_U_Omega_grad_vec(n_corrs);
  Eigen::Matrix<double, -1, 1>  log_abs_prev_unconstrained_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(2); //
  Eigen::Matrix<double, -1, 1>  log_abs_prev_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(2); //
  Eigen::Matrix<double,  1, 1>  log_abs_prev_unconstrained_grad_vec_out =   Eigen::Matrix<double, 1, 1>::Zero(2 - 1); //
  // ///////////////////////////////////////////////
  Eigen::Matrix<double , -1, 1>   sign_beta_grad_vec   =  Eigen::Matrix<double, -1, 1>::Ones(n_covariates_total);  //
  std::vector<Eigen::Matrix<double, -1, -1>> sign_beta_grad_array =  vec_of_mats<double>(n_covariates_max, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> sign_U_Omega_grad_array =  vec_of_mats<double>(n_tests, n_tests, n_class);
  Eigen::Matrix<double, -1, 1 > sign_L_Omega_grad_vec(n_corrs + (2 * n_tests));
  Eigen::Matrix<double, -1, 1 > sign_U_Omega_grad_vec(n_corrs);
  Eigen::Matrix<double, -1, 1>  sign_prev_unconstrained_grad_vec =   Eigen::Matrix<double, -1, 1>::Ones(2); //
  Eigen::Matrix<double, -1, 1>  sign_prev_grad_vec =   Eigen::Matrix<double, -1, 1>::Ones(2); //
  Eigen::Matrix<double,  1, 1>  sign_prev_unconstrained_grad_vec_out =   Eigen::Matrix<double, 1, 1>::Ones(2 - 1); //
  // ///////////////////////////////////////////////

  
  
  
  {
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<Eigen::Matrix<double, -1, -1>>   Z_std_norm =  vec_of_mats<double>(chunk_size, n_tests, n_class);
    std::vector<Eigen::Matrix<double, -1, -1>>   log_Z_std_norm =  vec_of_mats<double>(chunk_size, n_tests, n_class);
    std::vector<Eigen::Matrix<double, -1, -1>>   Bound_Z =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>   Bound_U_Phi_Bound_Z =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>   Phi_Z =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>   phi_Bound_Z =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>   y1_log_prob =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>   prob =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>   prob_recip =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>   log_abs_Bound_Z =  Z_std_norm;
    std::vector<Eigen::Matrix<double, -1, -1>>   sign_Bound_Z =  Z_std_norm;
    ///////////////////////////////////////////////
    std::vector<Eigen::Matrix<double, -1, -1>>     phi_Z_recip =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>     log_phi_Z_recip =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>     log_Bound_U_Phi_Bound_Z =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>     log_phi_Bound_Z =  Z_std_norm ;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> y_chunk =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1> u_array =  y_chunk ;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1>      inc_array  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    Eigen::Matrix<double, -1, 1>      log_abs_inc_array  =  Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
    Eigen::Matrix<double, -1, 1>      sign_inc_array  =  Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
    Eigen::Matrix<double, -1, 1>      prob_n  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    Eigen::Matrix<double, -1, 1>      prob_n_recip  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1>     u_grad_array_CM_chunk   =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     log_abs_u_grad_array_CM_chunk   =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> y_sign_chunk =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1> y_m_y_sign_x_u = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1> log_sum_result = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    Eigen::Matrix<double, -1, 1> log_sum_abs_result = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    Eigen::Matrix<double, -1, 1> sign_result = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    Eigen::Matrix<double, -1, 1> container_max_logs = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    Eigen::Matrix<double, -1, 1> container_sum_exp_signed = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1>   u_unc_vec_chunk =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests); 
    Eigen::Matrix<double, -1, 1>   u_vec_chunk =        Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests); 
    Eigen::Matrix<double, -1, 1>   du_wrt_duu_chunk =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests); 
    Eigen::Matrix<double, -1, 1>   d_J_wrt_duu_chunk =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests); 
    ///////////////////////////////////////////////
    double log_jac_u = 0.0;
    ///////////////////////////////////////////////
    std::vector<int> i_log_grad(0); // initialise at size 0
    std::vector<int> i_grad(chunk_size);
    bool any_log_grad = false;
    bool any_non_log_grad = true;
    ///////////////////////////////////////////////
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1>     A_common_grad_term_1   =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     A_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip   =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests) ;
    Eigen::Matrix<double, -1, -1>     A_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip   =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests) ;
    Eigen::Matrix<double, -1, -1>     A_prop_rowwise_prod_temp   =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests) ;
    Eigen::Matrix<double, -1, -1>     A_prop_recip_rowwise_prod_temp   =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests) ;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1>      A_prod_container_or_inc_array  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    Eigen::Matrix<double, -1, 1>      A_derivs_chain_container_vec  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    Eigen::Matrix<double, -1, 1>      A_prop_rowwise_prod_temp_all  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1>     A_grad_prob =    Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     A_z_grad_term =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1>     A_prob =    Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     A_prob_recip =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     A_phi_Z_recip =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     A_phi_Bound_Z =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     A_Phi_Z =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     A_Bound_Z =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     A_Z_std_norm =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, 1>      A_prob_n_recip =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1>     A_y_chunk =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     A_u_array =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     A_y_sign_chunk =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     A_y_m_y_sign_x_u =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1>     A_u_grad_array_CM_chunk_block  =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     A_L_Omega_recip_double_array  =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ///////////////////////////////////////////////
    ///////////////////////////////////////////////
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1>     B_log_common_grad_term_1   =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_log_L_Omega_diag_recip_array   = Eigen::Matrix<double, -1, -1>::Zero(1, n_tests) ;
    Eigen::Matrix<double, -1, -1>     B_log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip   =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests) ;
    Eigen::Matrix<double, -1, -1>     B_log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip   =   Eigen::Matrix<double, -1, -1>::Zero(1, n_tests) ;
    Eigen::Matrix<double, -1, -1>     B_sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip   =  Eigen::Matrix<double, -1, -1>::Ones(1, n_tests) ;
    Eigen::Matrix<double, -1, -1>     B_sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip   =   Eigen::Matrix<double, -1, -1>::Ones(1, n_tests) ;
    Eigen::Matrix<double, -1, -1>     B_log_prob_rowwise_prod_temp   =   Eigen::Matrix<double, -1, -1>::Zero(1, n_tests) ;
    Eigen::Matrix<double, -1, -1>     B_log_prob_recip_rowwise_prod_temp   =   Eigen::Matrix<double, -1, -1>::Zero(1, n_tests) ;
    Eigen::Matrix<double, -1, -1>     B_log_abs_grad_prob =    Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_log_abs_z_grad_term =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_sign_grad_prob =    Eigen::Matrix<double, -1, -1>::Ones(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_sign_z_grad_term =  Eigen::Matrix<double, -1, -1>::Ones(1, n_tests);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1>     B_log_abs_prod_container_or_inc_array_comp  =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_sign_prod_container_or_inc_array_comp  =  Eigen::Matrix<double, -1, -1>::Ones(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_log_abs_derivs_chain_container_vec_comp =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_sign_derivs_chain_container_vec_comp =  Eigen::Matrix<double, -1, -1>::Ones(1, n_tests);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1>      B_log_abs_prod_container_or_inc_array  =  Eigen::Matrix<double, -1, 1>::Zero(1);
    Eigen::Matrix<double, -1, 1>      B_sign_prod_container_or_inc_array  =  Eigen::Matrix<double, -1, 1>::Ones(1);
    Eigen::Matrix<double, -1, 1>      B_log_abs_derivs_chain_container_vec  =  Eigen::Matrix<double, -1, 1>::Zero(1);
    Eigen::Matrix<double, -1, 1>      B_sign_derivs_chain_container_vec  =  Eigen::Matrix<double, -1, 1>::Ones(1);
    Eigen::Matrix<double, -1, 1>      B_log_prob_rowwise_prod_temp_all  =  Eigen::Matrix<double, -1, 1>::Zero(1);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1>     B_y1_log_prob =    Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_prob =    Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_Bound_Z =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_Z_std_norm =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1>     B_log_phi_Bound_Z =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_log_phi_Z_recip =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_log_Z_std_norm =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_sign_Z_std_norm =  Eigen::Matrix<double, -1, -1>::Ones(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_log_abs_Bound_Z =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1>     B_sign_Bound_Z =  Eigen::Matrix<double, -1, -1>::Ones(1, n_tests);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1>  B_prob_n_recip =  Eigen::Matrix<double, -1, 1>::Zero(1);
    Eigen::Matrix<double, -1, 1>  B_log_prob_n_recip =  Eigen::Matrix<double, -1, 1>::Zero(1);
    Eigen::Matrix<double, -1, 1>  B_sign_beta_grad_array_col_for_each_n =  Eigen::Matrix<double, -1, 1>::Ones(1);
    Eigen::Matrix<double, -1, 1>  B_log_abs_beta_grad_array_col_for_each_n =  Eigen::Matrix<double, -1, 1>::Zero(1);
    Eigen::Matrix<double, -1, 1>  B_sign_Omega_grad_array_col_for_each_n =  Eigen::Matrix<double, -1, 1>::Ones(1);
    Eigen::Matrix<double, -1, 1>  B_log_abs_Omega_grad_array_col_for_each_n =  Eigen::Matrix<double, -1, 1>::Zero(1);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1>  B_log_abs_a  =  Eigen::Matrix<double, -1, 1>::Zero(1);
    Eigen::Matrix<double, -1, 1>  B_sign_a =  Eigen::Matrix<double, -1, 1>::Ones(1);
    Eigen::Matrix<double, -1, 1>  B_log_abs_b =  Eigen::Matrix<double, -1, 1>::Zero(1);
    Eigen::Matrix<double, -1, 1>  B_sign_b =  Eigen::Matrix<double, -1, 1>::Ones(1);
    Eigen::Matrix<double, -1, 1>  B_log_sum_result =  Eigen::Matrix<double, -1, 1>::Zero(1);
    Eigen::Matrix<double, -1, 1>  B_sign_sum_result =  Eigen::Matrix<double, -1, 1>::Ones(1);
    Eigen::Matrix<double, -1, -1> B_log_terms =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1> B_sign_terms =  Eigen::Matrix<double, -1, -1>::Ones(1, n_tests);
    Eigen::Matrix<double, -1, 1>  B_final_log_sum =  Eigen::Matrix<double, -1, 1>::Zero(1);
    Eigen::Matrix<double, -1, 1>  B_final_sign =  Eigen::Matrix<double, -1, 1>::Ones(1);
    Eigen::Matrix<double, -1, 1>  B_container_max_logs =  Eigen::Matrix<double, -1, 1>::Ones(1);
    Eigen::Matrix<double, -1, 1>  B_container_sum_exp_signed =  Eigen::Matrix<double, -1, 1>::Ones(1);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1> B_log_abs_prev_grad_array_col_for_each_n =  Eigen::Matrix<double, -1, 1>::Zero(1);
    Eigen::Matrix<double, -1, 1> B_sign_prev_grad_array_col_for_each_n =  Eigen::Matrix<double, -1, 1>::Zero(1);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> B_y_chunk =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1> B_u_array =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1> B_y_sign =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    Eigen::Matrix<double, -1, -1> B_y_m_y_sign_x_u =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1>   B_log_prob_recip  =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
    ///////////////////////////////////////////////
    std::vector<std::vector<Eigen::VectorXi>> OK_indices(n_class);
    std::vector<std::vector<Eigen::VectorXi>> overflow_indices(n_class);
    std::vector<std::vector<Eigen::VectorXi>> underflow_indices(n_class);
    ///////////////////////////////////////////////
    
    
    { // start of big local block

      Eigen::Matrix<double, -1, -1>    lp_array  =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_class); ///////

    for (int nc = 0; nc < n_total_chunks; nc++) { // Note: if remainder, then n_total_chunks =  n_full_chunks + 1 and then nc goes from 0 -> n_total_chunks - 1 = n_full_chunks

        int chunk_counter = nc;
        
        if ((chunk_counter == n_full_chunks) && (n_chunks > 1) && (last_chunk_size > 0)) { // Last chunk (remainder - don't use AVX / SIMD for this)
          
                  chunk_size = last_chunk_size;  //// update chunk_size 
                  
                  if (debug == true) {
                    
                        std::cout << "processing last [remainder?] chunk " << "\n";
              
                        std::cout << "n_total_chunks =  " << n_total_chunks << "\n";
                        std::cout << "chunk_size (new) =  " << chunk_size << "\n";
                        std::cout << "last_chunk_size =  " << last_chunk_size << "\n";
                        std::cout << "chunk_counter =  " << chunk_counter << "\n";
                        std::cout << "n_full_chunks =  " << n_full_chunks << "\n";
                        std::cout << "chunk_size_orig =  " << chunk_size_orig << "\n";
                        
                  }
                  
                  
                  if (debug == true) {
                    std::cout << "vec_size =  " << vec_size << "\n";
                  }
        
                          /// use either Loop (i.e. double fn's) or Stan's vectorisation for the remainder (i.e. last) chunk, regardless of input
                          vect_type = "Stan";
                          vect_type_exp = "Stan";
                          vect_type_log = "Stan";
                          vect_type_lse = "Stan";
                          vect_type_tanh = "Stan";
                          vect_type_Phi = "Stan";
                          vect_type_log_Phi = "Stan";
                          vect_type_inv_Phi = "Stan";
                          vect_type_inv_Phi_approx_from_logit_prob = "Stan";
                          
                          i_grad.resize(last_chunk_size);
                          i_log_grad.resize(0);
                          
                          // vectors
                          inc_array.resize(last_chunk_size);
                          log_abs_inc_array.resize(last_chunk_size);
                          sign_inc_array.resize(last_chunk_size);
                          prob_n.resize(last_chunk_size);
                          prob_n_recip.resize(last_chunk_size);
                          log_sum_result.resize(last_chunk_size);
                          log_sum_abs_result.resize(last_chunk_size);
                          container_max_logs.resize(last_chunk_size);
                          container_sum_exp_signed.resize(last_chunk_size);
                          sign_result.resize(last_chunk_size);
                          
                          A_prob_n_recip.resize(last_chunk_size);
                          A_prod_container_or_inc_array.resize(last_chunk_size);
                          A_derivs_chain_container_vec.resize(last_chunk_size);
                          A_prop_rowwise_prod_temp_all.resize(last_chunk_size);
                          
                          // matrices
                          y_chunk.resize(last_chunk_size, n_tests); y_chunk.setZero();
                          u_array.resize(last_chunk_size, n_tests); u_array.setZero();
                          u_grad_array_CM_chunk.resize(last_chunk_size, n_tests); u_grad_array_CM_chunk.setZero();
                          log_abs_u_grad_array_CM_chunk.resize(last_chunk_size, n_tests); log_abs_u_grad_array_CM_chunk.setZero();
                          y_sign_chunk.resize(last_chunk_size, n_tests); y_sign_chunk.setZero();
                          y_m_y_sign_x_u.resize(last_chunk_size, n_tests); y_m_y_sign_x_u.setZero();
                          lp_array.resize(last_chunk_size, n_class); lp_array.setZero();
                          
                          A_common_grad_term_1.resize(last_chunk_size, n_tests); A_common_grad_term_1.setZero();
                          A_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(last_chunk_size, n_tests); A_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.setZero();
                          A_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(last_chunk_size, n_tests); A_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.setZero();
                          A_prop_rowwise_prod_temp.resize(last_chunk_size, n_tests); A_prop_rowwise_prod_temp.setZero();
                          A_prop_recip_rowwise_prod_temp.resize(last_chunk_size, n_tests); A_prop_recip_rowwise_prod_temp.setZero();
                          A_grad_prob.resize(last_chunk_size, n_tests); A_grad_prob.setZero();
                          A_z_grad_term.resize(last_chunk_size, n_tests); A_z_grad_term.setZero();
                          
                          A_prob.resize(last_chunk_size, n_tests); A_prob.setZero();
                          A_prob_recip.resize(last_chunk_size, n_tests); A_prob_recip.setZero();
                          A_phi_Z_recip.resize(last_chunk_size, n_tests); A_phi_Z_recip.setZero();
                          A_phi_Bound_Z.resize(last_chunk_size, n_tests); A_phi_Bound_Z.setZero();
                          A_Phi_Z.resize(last_chunk_size, n_tests); A_Phi_Z.setZero();
                          A_Bound_Z.resize(last_chunk_size, n_tests); A_Bound_Z.setZero();
                          A_Z_std_norm.resize(last_chunk_size, n_tests); A_Z_std_norm.setZero();
                        
                          A_y_chunk.resize(last_chunk_size, n_tests); A_y_chunk.setZero();
                          A_u_array.resize(last_chunk_size, n_tests); A_u_array.setZero();
                          A_y_sign_chunk.resize(last_chunk_size, n_tests); A_y_sign_chunk.setZero();
                          A_y_m_y_sign_x_u.resize(last_chunk_size, n_tests); A_y_m_y_sign_x_u.setZero();
                          A_u_grad_array_CM_chunk_block.resize(last_chunk_size, n_tests); A_u_grad_array_CM_chunk_block.setZero();
                          A_L_Omega_recip_double_array.resize(last_chunk_size, n_tests); A_L_Omega_recip_double_array.setZero();
                          
                          u_unc_vec_chunk.resize(last_chunk_size * n_tests);
                          u_vec_chunk.resize(last_chunk_size * n_tests);
                          du_wrt_duu_chunk.resize(last_chunk_size * n_tests);
                          d_J_wrt_duu_chunk.resize(last_chunk_size * n_tests);
           
                          // natrix arrays
                          for (int c = 0; c < n_class; c++) {
                            
                                Z_std_norm[c].resize(last_chunk_size, n_tests);
                                log_Z_std_norm[c].resize(last_chunk_size, n_tests);
                                Bound_Z[c].resize(last_chunk_size, n_tests);
                                Bound_U_Phi_Bound_Z[c].resize(last_chunk_size, n_tests);
                                Phi_Z[c].resize(last_chunk_size, n_tests);
                                phi_Bound_Z[c].resize(last_chunk_size, n_tests);
                                y1_log_prob[c].resize(last_chunk_size, n_tests);
                                prob[c].resize(last_chunk_size, n_tests);
                                prob_recip[c].resize(last_chunk_size, n_tests);
                                phi_Z_recip[c].resize(last_chunk_size, n_tests);
                                log_phi_Z_recip[c].resize(last_chunk_size, n_tests);
                                log_Bound_U_Phi_Bound_Z[c].resize(last_chunk_size, n_tests);
                                log_phi_Bound_Z[c].resize(last_chunk_size, n_tests);
                                log_abs_Bound_Z[c].resize(last_chunk_size, n_tests);
                                sign_Bound_Z[c].resize(last_chunk_size, n_tests);
                            
                          }
                          
        }

        u_grad_array_CM_chunk.setZero() ;// = 0.0; // reset between chunks as re-using same container
        log_abs_u_grad_array_CM_chunk.setConstant(-700);  // reset between chunks as re-using same container

        y_chunk = y_ref.middleRows( chunk_size_orig * chunk_counter , chunk_size).array().cast<double>() ;
        
        ////// Nuisance parameter transformation step
        u_unc_vec_chunk = u_unc_vec.segment( chunk_size_orig * n_tests * chunk_counter , chunk_size * n_tests);
        
        u_vec_chunk =  fn_MVP_compute_nuisance(    u_unc_vec_chunk,
                                                   nuisance_transformation,
                                                   vect_type_Phi, 
                                                   vect_type_log,
                                                   vect_type_tanh);
        
        log_jac_u +=    fn_MVP_compute_nuisance_log_jac_u(   u_vec_chunk,
                                                             u_unc_vec_chunk,
                                                             nuisance_transformation,
                                                             vect_type_Phi,
                                                             vect_type_log,
                                                             vect_type_tanh,
                                                             skip_checks_log);
        
        if (debug == true) {
           std::cout << "u_vec_chunk.array().maxCoeff()" << u_vec_chunk.array().maxCoeff()  << "\n";
        }
        
        u_array  =  u_vec_chunk.reshaped(chunk_size, n_tests); /// .array(); 
        
        if (debug == true) {
           std::cout << "u_array.array().maxCoeff()" << u_array.array().maxCoeff()  << "\n";
        }
        
        
        {
          // START of c loop
          for (int c = 0; c < n_class; c++) {

            inc_array.setZero(); //   = 0.0; // needs to be reset to 0
            sign_inc_array.setOnes();
            log_abs_inc_array.setConstant(-700.0);
            
            // start of t loop
            for (int t = 0; t < n_tests; t++) {

              if (n_covariates_max > 1) {
                
                      Eigen::Matrix<double, -1, 1>    Xbeta_given_class_c_col_t = X[c][t].block(chunk_size_orig * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(c, t)).cast<double>()  * beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t));
                      Bound_Z[c].col(t).array() =     L_Omega_recip_double[c](t, t) * (  - ( Xbeta_given_class_c_col_t.array()    +      inc_array.array()   )  ) ;
                      sign_Bound_Z[c].col(t) =   Bound_Z[c].col(t).array().sign();
                      log_abs_Bound_Z[c].col(t) =   fn_EIGEN_double(Bound_Z[c].col(t).array().abs(), "log", vect_type_log);
                      bound_log(log_abs_Bound_Z[c].col(t));  /// add bounds checking
                      
                      // {
                      //   
                      //   Eigen::Matrix<double, -1, 1>  log_abs_a =  fn_EIGEN_double(stan::math::abs(- Xbeta_given_class_c_col_t.array() ), "log", vect_type_log);
                      //   Eigen::Matrix<double, -1, 1>  sign_a  = stan::math::sign( -Xbeta_given_class_c_col_t.array() );
                      //   Eigen::Matrix<double, -1, 1>  log_abs_b =    log_abs_inc_array.array(); ///   fn_EIGEN_double( (- inc_array.array()).abs(), "log", vect_type_log);
                      //   Eigen::Matrix<double, -1, 1>  sign_b =  - sign_inc_array.array(); ///   stan::math::sign((- inc_array.array()).matrix());
                      //   
                      //   Eigen::Matrix<double, -1, -1>  log_abs_a_and_b(chunk_size, 2); log_abs_a_and_b.col(0) = log_abs_a;  log_abs_a_and_b.col(1) = log_abs_b;
                      //   Eigen::Matrix<double, -1, -1>  signs_a_and_b(chunk_size, 2); signs_a_and_b.col(0) = sign_a;  signs_a_and_b.col(1) = sign_b;
                      //   
                      //   log_abs_a_and_b.array() = log_abs_a_and_b.array().min(700.0).max(-700.0);     /// add bounds checking
                      //   
                      //   log_abs_sum_exp_general_v2(log_abs_a_and_b, 
                      //                              signs_a_and_b, 
                      //                              vect_type_exp, vect_type_log, 
                      //                              log_sum_abs_result, sign_result, 
                      //                              container_max_logs, container_sum_exp_signed);
                      //   
                      //   log_abs_Bound_Z[c].col(t).array() = log_sum_abs_result.array()  + stan::math::log(stan::math::abs(L_Omega_recip_double[c](t, t)));
                      //   log_abs_Bound_Z[c].col(t).array() =  log_abs_Bound_Z[c].col(t).array().min(700.0).max(-700.0);     /// add bounds checking
                      //   
                      //   sign_Bound_Z[c].col(t) = sign_result *  stan::math::sign(L_Omega_recip_double[c](t, t));
                      //   
                      //   Bound_Z[c].col(t).array() = fn_EIGEN_double(   log_abs_Bound_Z[c].col(t) , "exp", vect_type_exp).array() *    sign_Bound_Z[c].col(t).array(); 
                      //   
                      // }
                
              } else {  // intercept-only
                
                      Bound_Z[c].col(t).array() =     L_Omega_recip_double[c](t, t) * (  - ( beta_double_array[c](0, t) +      inc_array.array()   )  ) ;
                      
                      sign_Bound_Z[c].col(t) =   Bound_Z[c].col(t).array().sign();
                      log_abs_Bound_Z[c].col(t) =   fn_EIGEN_double(Bound_Z[c].col(t).array().abs(), "log", vect_type_log);
                      bound_log(log_abs_Bound_Z[c].col(t));  /// add bounds checking
                      
                      // {
                      // 
                      //       Eigen::Matrix<double, -1, 1>  beta_val_rep_vec(chunk_size);  beta_val_rep_vec.setConstant(beta_double_array[c](0, t));
                      //       Eigen::Matrix<double, -1, 1>  log_abs_a   = fn_EIGEN_double(stan::math::abs( (- beta_val_rep_vec.array()).matrix()), "log", vect_type_log);
                      //       Eigen::Matrix<double, -1, 1>  sign_a = stan::math::sign( (- beta_val_rep_vec.array()).matrix());
                      // 
                      //       Eigen::Matrix<double, -1, 1>  log_abs_b =  log_abs_inc_array.array(); ///  fn_EIGEN_double( (- inc_array.array()).abs(), "log", vect_type_log);
                      //       Eigen::Matrix<double, -1, 1>  sign_b = - sign_inc_array.array(); ///  stan::math::sign((- inc_array.array()).matrix());
                      // 
                      //       Eigen::Matrix<double, -1, -1>  log_abs_a_and_b(chunk_size, 2); log_abs_a_and_b.col(0) = log_abs_a;  log_abs_a_and_b.col(1) = log_abs_b;
                      //       Eigen::Matrix<double, -1, -1>  signs_a_and_b(chunk_size, 2); signs_a_and_b.col(0) = sign_a;  signs_a_and_b.col(1) = sign_b;
                      // 
                      //       log_abs_a_and_b.array() = log_abs_a_and_b.array().min(700.0).max(-700.0);     /// add bounds checking
                      // 
                      //       log_abs_sum_exp_general_v2(log_abs_a_and_b,
                      //                                  signs_a_and_b,
                      //                                  vect_type_exp, vect_type_log,
                      //                                  log_sum_abs_result, sign_result,
                      //                                  container_max_logs, container_sum_exp_signed);
                      // 
                      //      log_abs_Bound_Z[c].col(t).array() = log_sum_abs_result.array()  + stan::math::log(stan::math::abs(L_Omega_recip_double[c](t, t)));
                      //      log_abs_Bound_Z[c].col(t).array() =  log_abs_Bound_Z[c].col(t).array().min(700.0).max(-700.0);     /// add bounds checking
                      // 
                      // 
                      //      sign_Bound_Z[c].col(t) = sign_result *  stan::math::sign(L_Omega_recip_double[c](t, t));
                      // 
                      //      Bound_Z[c].col(t).array() = fn_EIGEN_double(   log_abs_Bound_Z[c].col(t) , "exp", vect_type_exp).array() *    sign_Bound_Z[c].col(t).array();
                      // 
                      // }
                      
                           // log_abs_Bound_Z[c].col(t).array() =          stan::math::log(Bound_Z[c].col(t).array().abs());
                           // log_abs_Bound_Z[c].col(t).array() =  log_abs_Bound_Z[c].col(t).array().min(700.0).max(-700.0);     /// add bounds checking
                           // sign_Bound_Z[c].col(t).array() = Bound_Z[c].col(t).array().sign();
                      
              }

              // const Eigen::VectorXi overflow_mask =   ( (Bound_Z[c].col(t).array() > overflow_threshold)  && (y_chunk.col(t).cast<double>().array() == 1.0) ).cast<int>().matrix() ;
              // const Eigen::VectorXi underflow_mask =  ( (Bound_Z[c].col(t).array() < underflow_threshold) && (y_chunk.col(t).cast<double>().array() == 0.0) ).cast<int>().matrix() ;
              // const Eigen::VectorXi ok_mask = ( (overflow_mask.array() == 0) && (underflow_mask.array() == 0) ).cast<int>().matrix() ;
              
              int num_overflows = 0;// overflow_mask.sum();
              int num_underflows = 0;// underflow_mask.sum();
              int num_OK = 0;// ok_mask.sum();
              
              for (int n = 0; n < chunk_size; ++n) {
                    if  (        (Bound_Z[c](n, t)  >  overflow_threshold)  &&  (y_chunk(n, t) == 1.0) ) {          num_overflows += 1;
                    } else if  ( (Bound_Z[c](n, t)  <  underflow_threshold)   &&  (y_chunk(n, t) == 0.0) )  {       num_underflows += 1;
                    } else {    num_OK += 1;
                    }
              }
              
              if (debug == true) {
                    std::cout << "\nCounts:\n";
                    std::cout << "num_overflows:  " << num_overflows << "\n";
                    std::cout << "num_underflows: " << num_underflows << "\n";
                    std::cout << "num_OK:         " << num_OK << "\n";
                    std::cout << "Total:          " << (num_overflows + num_underflows + num_OK) << "\n";
                    std::cout << "chunk_size:     " << chunk_size << "\n";
              }
              
              // // Verify counts add up to chunk_size
              // assert(num_overflows + num_underflows + num_OK == chunk_size);
              
              std::vector<int> OK_index(num_OK);
              std::vector<int> over_index(num_overflows);
              std::vector<int> under_index(num_underflows);
              
              int counter_ok  = 0;
              int counter_over  = 0;
              int counter_under  = 0;
              
             for (int n = 0; n < chunk_size; ++n) {
               
                     if  (    (Bound_Z[c](n, t)  >  overflow_threshold)  &&  (y_chunk(n, t) == 1.0) ) {
                         //  if (debug == true) std::cout << "Overflow -> position " << counter_over << "\n";
                          // assert(counter_over <= num_overflows); // bounds checking
                           over_index[counter_over] = n;
                           counter_over += 1;
                     } else if  ( (Bound_Z[c](n, t)  <  underflow_threshold)   &&  (y_chunk(n, t) == 0.0) )  {
                         //  if (debug == true) std::cout << "Underflow -> position " << counter_under << "\n";
                          // assert(counter_under <= num_underflows); // bounds checking
                           under_index[counter_under] = n;
                           counter_under += 1;
                     } else { 
                         //  if (debug == true) std::cout << "OK -> position " << counter_ok << "\n";
                          // assert(counter_ok <= num_OK); // bounds checking
                           OK_index[counter_ok] = n;
                           counter_ok += 1;
                     }
               
             }
             

             if (debug == true) {
                 // Print resulting indices
                 // std::cout << "\nResulting indices:\n";
                 // printVectorContents("OK_index", OK_index);
                 // printVectorContents("over_index", over_index);
                 // printVectorContents("under_index", under_index);
                 
                 // Verify counters match expected sizes
                 std::cout << "\nCounter verification:\n";
                 std::cout << "counter_ok = " << counter_ok << " (expected " << num_OK << ")\n";
                 std::cout << "counter_over = " << counter_over << " (expected " << num_overflows << ")\n";
                 std::cout << "counter_under = " << counter_under << " (expected " << num_underflows << ")\n";
             }
   
             // /////// now store inside the nested vectors of indicies
             // // Initialize
             // for(int c = 0; c < n_class; c++) {
             //   OK_indices[c].resize(n_tests);        
             //   overflow_indices[c].resize(n_tests);
             //   underflow_indices[c].resize(n_tests);
             // }
             // 
             // // When setting for a specific (c,t):
             // OK_indices[c][t] = OK_index;  // Each one can be different size
             // overflow_indices[c][t] = over_index;
             // underflow_indices[c][t] = under_index;
             
                   ////// first compute on NON-log-scale for all observations using ".col(t)". 
                   Bound_U_Phi_Bound_Z[c].col(t).array() =   fn_EIGEN_double( Bound_Z[c].col(t).array(), Phi_type, vect_type_Phi);  /////
                   Phi_Z[c].col(t).array() = y_chunk.col(t).array() * Bound_U_Phi_Bound_Z[c].col(t).array() +   (y_chunk.col(t).array() -  Bound_U_Phi_Bound_Z[c].col(t).array()) *   ((y_chunk.col(t).array()  + (y_chunk.col(t).array()  - 1.0)) * u_array.col(t).array());
                   Z_std_norm[c].col(t).array() =   fn_EIGEN_double( Phi_Z[c].col(t).array(),   inv_Phi_type, vect_type_inv_Phi);      ////
                   prob[c].col(t).array() =    y_chunk.col(t).array()  * (1.0 - Bound_U_Phi_Bound_Z[c].col(t).array() ) + ( y_chunk.col(t).array()  -  1.0)  *  Bound_U_Phi_Bound_Z[c].col(t).array() * ( y_chunk.col(t).array()  +  (  y_chunk.col(t).array()  - 1.0)  )  ;
                   y1_log_prob[c].col(t).array()  =      fn_EIGEN_double( prob[c].col(t),  "log", vect_type_log);
                   
                   ///////// grad stuff
                   if ( (Phi_type == "Phi_approx") || (Phi_type == "Phi_approx_2") ) { // vect_type
                     phi_Bound_Z[c].col(t).array()  =         ( a_times_3 * Bound_Z[c].col(t).array().square() + b  ).array()  *  Bound_U_Phi_Bound_Z[c].col(t).array() * (1.0 -  Bound_U_Phi_Bound_Z[c].col(t).array() )   ;
                     phi_Z_recip[c].col(t).array()  =    1.0 / ((  ( a_times_3 * Z_std_norm[c].col(t).array().square() + b  ).array()  ).array()*  Phi_Z[c].col(t).array() * (1.0 -  Phi_Z[c].col(t).array())  ).array() ;
                   }  else if (Phi_type == "Phi")   {
                     phi_Bound_Z[c].col(t).array()  =         sqrt_2_pi_recip * fn_EIGEN_double( ( - 0.5 * Bound_Z[c].col(t).array().square() ).matrix(),  "exp", vect_type_exp);
                     phi_Z_recip[c].col(t).array()  =    1.0 /  (   sqrt_2_pi_recip * fn_EIGEN_double( ( - 0.5 * Z_std_norm[c].col(t).array().square() ).matrix(),  "exp", vect_type_exp) ).array();
                   }
                   
                   log_phi_Bound_Z[c].col(t).array() =  fn_EIGEN_double(  phi_Bound_Z[c].col(t),  "log", vect_type_log);    ///////
                   bound_log(log_phi_Bound_Z[c].col(t)); /// add bounds checking
                   
                   log_phi_Z_recip[c].col(t).array() =  fn_EIGEN_double(  phi_Z_recip[c].col(t),  "log", vect_type_log);    ///////
                   bound_log(log_phi_Z_recip[c].col(t)); /// add bounds checking
                   
                   log_Z_std_norm[c].col(t).array()    = fn_EIGEN_double(Z_std_norm[c].col(t).array().abs().matrix(), "log", vect_type_log); 
                   bound_log(log_Z_std_norm[c].col(t)); /// add bounds checking
                   
                   
             
             
              
            if   (num_OK == chunk_size)  { // carry on as normal as no * problematic * overflows/underflows

                              any_log_grad = false;
                              any_non_log_grad = true;
                              i_grad = OK_index; // _oversized;

                              i_log_grad.resize(0);
                              const int total_log_indices = 0;
                   
            }  else if (num_OK < chunk_size)  {
                 
                              any_log_grad = true;

                              if (num_OK < 1)  {

                                  any_non_log_grad = false;

                              } else {

                                 // assert(num_OK <= OK_index.size() && "Invalid OK_index size");
                                  // i_grad.resize(num_OK);
                                  // i_grad = OK_index; // .head(num_OK);
                                  // any_non_log_grad = true;
                              }
                              // Validate counts
                              num_overflows = std::max(0, num_overflows);
                              num_underflows = std::max(0, num_underflows);

                              // Handle log grad indices
                              const int total_log_indices = num_overflows + num_underflows;
                            //  std::cout << " total_log_indices " << total_log_indices << std::endl;

                              if (total_log_indices > 0) {

                                        i_log_grad.resize(total_log_indices);

                                        if (num_overflows > 0) {
                                         // assert(num_overflows <= over_index.size() &&   "Invalid overflows index size");
                                        //  i_log_grad.head(num_overflows) =   over_index;//.head(num_overflows);
                                        }

                                        if (num_underflows > 0) {
                                        //  assert(num_underflows <= under_index.size() &&    "Invalid underflows index size");
                                       //   i_log_grad.segment(num_overflows, num_underflows) =   under_index;//.head(num_underflows);
                                        }

                              } else {    // Both counts are 0, ensure i_log_grad is empty but valid
                                   i_log_grad.resize(0);
                              }


                              // if (num_overflows < 1)  {
                              //   num_overflows = 0;
                              //   indicator_overflows_empty = 1;
                              // }
                              // if (num_underflows < 1)  {
                              //   num_underflows = 0;
                              //   indicator_underflows_empty = 1;
                              // }
                   
                   
                     // if (debug == true)  { 
                     //         std::set<int> all_indices;
                     //          std::cout << "OK indices: ";
                     //          for(int i = 0; i < OK_index.size(); i++) {
                     //            std::cout << OK_index(i) << " ";
                     //            all_indices.insert(OK_index(i));
                     //          }
                     //          std::cout << "\nUnderflow indices: ";
                     //          for(int i = 0; i < under_index.size(); i++) {
                     //            std::cout << under_index(i) << " ";
                     //            all_indices.insert(under_index(i));
                     //          }
                     //          std::cout << "\nOverflow indices: ";
                     //          for(int i = 0; i < over_index.size(); i++) {
                     //            std::cout << over_index(i) << " ";
                     //            all_indices.insert(over_index(i));
                     //          }
                     //          std::cout << "\nTotal unique indices: " << all_indices.size() << " (should be " << chunk_size << ")\n";
                     // }
                     
 
                // if (num_OK > 0) {
                // 
                //        std::vector<int> index = OK_index; 
                //        const int index_size = index.size();
                // 
                //               {  ///// NON-log-scale
                //                 
                //                 for (int n = 0; n < index_size; ++n) {
                //                   
                //                       Bound_U_Phi_Bound_Z[c](index[n], t) = stan::math::Phi(Bound_Z[c](index[n], t)); ///   fn_EIGEN_double( Bound_Z[c](index[n], t), Phi_type, vect_type_Phi);  /////
                //                       Phi_Z[c](index[n], t) = y_chunk(index[n], t) * Bound_U_Phi_Bound_Z[c](index[n], t) +   (y_chunk(index[n], t) -  Bound_U_Phi_Bound_Z[c](index[n], t)) *   ((y_chunk(index[n], t)  + (y_chunk(index[n], t)  - 1.0)) * u_array(index[n], t));
                //                       Z_std_norm[c](index[n], t) =   stan::math::inv_Phi(Phi_Z[c](index[n], t)); //  fn_EIGEN_double( Phi_Z[c](index[n], t),   inv_Phi_type, vect_type_inv_Phi);      ////
                //                       prob[c](index[n], t) =    y_chunk(index[n], t)  * (1.0 - Bound_U_Phi_Bound_Z[c](index[n], t) ) + ( y_chunk(index[n], t)  -  1.0)  *  Bound_U_Phi_Bound_Z[c](index[n], t) * ( y_chunk(index[n], t)  +  (  y_chunk(index[n], t)  - 1.0)  )  ;
                //                       y1_log_prob[c](index[n], t)  =   stan::math::log(prob[c](index[n], t)); //     fn_EIGEN_double( prob[c](index[n], t),  "log", vect_type_log);
                //                       
                //                       ///////// grad stuff
                //                       if ( (Phi_type == "Phi_approx") || (Phi_type == "Phi_approx_2") ) { // vect_type
                //                         phi_Bound_Z[c](index[n], t)  =         ( a_times_3 * Bound_Z[c](index[n], t)*Bound_Z[c](index[n], t) + b  )  *  Bound_U_Phi_Bound_Z[c](index[n], t) * (1.0 -  Bound_U_Phi_Bound_Z[c](index[n], t) )   ;
                //                         phi_Z_recip[c](index[n], t)  =    1.0 / ((  ( a_times_3 * Z_std_norm[c](index[n], t)*Z_std_norm[c](index[n], t)+ b  )  ) *  Phi_Z[c](index[n], t) * (1.0 -  Phi_Z[c](index[n], t))  ) ;
                //                       }  else if (Phi_type == "Phi")   {
                //                         phi_Bound_Z[c](index[n], t)  =         sqrt_2_pi_recip * stan::math::exp( ( - 0.5 * Bound_Z[c](index[n], t) * Bound_Z[c](index[n], t) )) ;//   fn_EIGEN_double( ( - 0.5 * Bound_Z[c](index[n], t).square() ).matrix(),  "exp", vect_type_exp);
                //                         phi_Z_recip[c](index[n], t)  =    1.0 /  (   sqrt_2_pi_recip * stan::math::exp(( - 0.5 * Z_std_norm[c](index[n], t) * Z_std_norm[c](index[n], t) )) ) ; //  fn_EIGEN_double( ( - 0.5 * Z_std_norm[c](index[n], t).square() ).matrix(),  "exp", vect_type_exp) );
                //                       }
                //                       
                //                       log_phi_Bound_Z[c](index[n], t) =  stan::math::log(stan::math::abs(phi_Bound_Z[c](index[n], t))); //  fn_EIGEN_double(  phi_Bound_Z[c](index[n], t).abs(),  "log", vect_type_log);    ///////
                //                       bound_log(log_phi_Bound_Z[c](index[n], t)); /// add bounds checking
                //                       
                //                       log_phi_Z_recip[c](index[n], t) =   stan::math::log(stan::math::abs(phi_Z_recip[c](index[n], t)));   //  fn_EIGEN_double(  phi_Z_recip[c](index[n], t).abs(),  "log", vect_type_log);    ///////
                //                       bound_log(log_phi_Z_recip[c](index[n], t)); /// add bounds checking
                //                       
                //                       log_Z_std_norm[c](index[n], t)    =   stan::math::log(stan::math::abs(Z_std_norm[c](index[n], t)));   //  fn_EIGEN_double(Z_std_norm[c](index[n], t).abs().matrix(), "log", vect_type_log);
                //                       bound_log(log_Z_std_norm[c](index[n], t));  /// add bounds checking
                //                   
                //                 }
                //                 
                //               }
                // 
                // 
                // }

                
                if (num_underflows > 0) { //// underflow (w/ y == 0)

                                std::vector<int> index = under_index;
                                const int index_size = index.size();

                                  {

                                     // { //// LOG-scale
                                     // 
                                     //        Eigen::Matrix<double, -1, 1> log_Bound_U_Phi_Bound_Z =   fn_EIGEN_double(   Bound_Z[c](index, t), "log_Phi_approx",  vect_type_log_Phi) ;
                                     //        log_Bound_U_Phi_Bound_Z = log_Bound_U_Phi_Bound_Z.array().min(700.0).max(-700.0); //// add bounds checking
                                     //        Bound_U_Phi_Bound_Z[c](index, t)    =    fn_EIGEN_double( log_Bound_U_Phi_Bound_Z, "exp", vect_type_exp); /// same as Stan (should work well??)
                                     // 
                                     //        Eigen::Matrix<double, -1, 1>  u_log =  fn_EIGEN_double(u_array(index, t), "log",  vect_type_log).array();
                                     //        Eigen::Matrix<double, -1, 1>  log_Phi_Z =   u_log.array() +  log_Bound_U_Phi_Bound_Z.array() ; /// log(u * Phi_Bound_Z); /// same as Stan (should work well)
                                     //        log_Phi_Z = log_Phi_Z.array().min(700.0).max(-700.0); //// add bounds checking
                                     //        Phi_Z[c](index, t).array() =   fn_EIGEN_double(log_Phi_Z, "exp",  vect_type_exp).array();  //// computed but not actually used
                                     // 
                                     //        //// Eigen::Matrix<double, -1, 1> log_1m_Phi_Z =   stan::math::log1m_exp( u_log + log_Bound_U_Phi_Bound_Z );
                                     //        Eigen::Matrix<double, -1, 1> log_1m_Phi_Z = fn_EIGEN_double(  (  Bound_U_Phi_Bound_Z[c](index, t).array()  * u_array(index, t).array() ).matrix() , "log1m",  vect_type_log);   /// same as Stan (should work well)
                                     //        log_1m_Phi_Z = log_1m_Phi_Z.array().min(700.0).max(-700.0); //// add bounds checking
                                     //        Eigen::Matrix<double, -1, 1> logit_Phi_Z =   log_Phi_Z - log_1m_Phi_Z; /// same as Stan (should work well)
                                     //        Z_std_norm[c](index, t).array()    =  inv_Phi_approx_from_logit_prob(logit_Phi_Z);  //  fn_EIGEN_double(  logit_Phi_Z, "inv_Phi_approx_from_logit_prob",  vect_type_inv_Phi_approx_from_logit_prob); /// same as Stan (should work well)
                                     // 
                                     //        log_Z_std_norm[c](index, t)    = fn_EIGEN_double(Z_std_norm[c](index, t).array().abs().matrix(), "log", vect_type_log);
                                     //        log_Z_std_norm[c](index, t) = log_Z_std_norm[c](index, t).array().min(700.0).max(-700.0); //// add bounds checking
                                     // 
                                     //        y1_log_prob[c](index, t)  =    log_Bound_U_Phi_Bound_Z ; /// same as Stan (should work well)
                                     //        y1_log_prob[c](index, t) = y1_log_prob[c](index, t).array().min(700.0).max(-700.0); //// add bounds checking
                                     // 
                                     //        prob[c](index, t) =        Bound_U_Phi_Bound_Z[c](index, t) ; //// computed but not actually used
                                     // 
                                     //        //// Eigen::Matrix<double, -1, 1>  log_Bound_U_Phi_Bound_Z_1m =  stan::math::log1m_exp(log_Bound_U_Phi_Bound_Z); //// use log1m_exp for stability!
                                     //        Eigen::Matrix<double, -1, 1>  log_Bound_U_Phi_Bound_Z_1m = fn_EIGEN_double(  Bound_U_Phi_Bound_Z[c](index, t), "log1m",  vect_type_log);
                                     //        log_Bound_U_Phi_Bound_Z_1m = log_Bound_U_Phi_Bound_Z_1m.array().min(700.0).max(-700.0); //// add bounds checking
                                     // 
                                     //       log_phi_Bound_Z[c](index, t).array()  =         stan::math::log( a_times_3 * Bound_Z[c](index, t).array().square() + b  ).array()  +   log_Bound_U_Phi_Bound_Z.array()  +   log_Bound_U_Phi_Bound_Z_1m.array();
                                     //       log_phi_Bound_Z[c](index, t) = log_phi_Bound_Z[c](index, t).array().min(700.0).max(-700.0); //// add bounds checking
                                     // 
                                     //       log_phi_Z_recip[c](index, t).array()  =    - (   stan::math::log(  ( a_times_3 * Z_std_norm[c](index, t).array().square() + b  ).array()  ).array()  +   log_Phi_Z.array()  +  log_1m_Phi_Z.array()  ).array() ;
                                     //       log_phi_Z_recip[c](index, t) = log_phi_Z_recip[c](index, t).array().min(700.0).max(-700.0); //// add bounds checking
                                     // 
                                     //       phi_Bound_Z[c](index, t).array() =  fn_EIGEN_double(  log_phi_Bound_Z[c](index, t),  "exp", vect_type_exp);    ///////
                                     //       phi_Z_recip[c](index, t).array() =  fn_EIGEN_double(  log_phi_Z_recip[c](index, t),  "exp", vect_type_exp);    ///////
                                     // 
                                     // }


                                     {  ///// NON-log-scale (STOPS WORKING IF THIS IS UNCOMMENTED)

                                       for (int n = 0; n < index_size; ++n) {

                                             try {

                                                 Bound_U_Phi_Bound_Z[c](index[n], t) = stan::math::Phi(Bound_Z[c](index[n], t)); ///   fn_EIGEN_double( Bound_Z[c](index[n], t), Phi_type, vect_type_Phi);  /////
                                                 Phi_Z[c](index[n], t) = y_chunk(index[n], t) * Bound_U_Phi_Bound_Z[c](index[n], t) +   (y_chunk(index[n], t) -  Bound_U_Phi_Bound_Z[c](index[n], t)) *   ((y_chunk(index[n], t)  + (y_chunk(index[n], t)  - 1.0)) * u_array(index[n], t));
                                                 Z_std_norm[c](index[n], t) =   stan::math::inv_Phi(Phi_Z[c](index[n], t)); //  fn_EIGEN_double( Phi_Z[c](index[n], t),   inv_Phi_type, vect_type_inv_Phi);      ////
                                                 prob[c](index[n], t) =    y_chunk(index[n], t)  * (1.0 - Bound_U_Phi_Bound_Z[c](index[n], t) ) + ( y_chunk(index[n], t)  -  1.0)  *  Bound_U_Phi_Bound_Z[c](index[n], t) * ( y_chunk(index[n], t)  +  (  y_chunk(index[n], t)  - 1.0)  )  ;
                                                 y1_log_prob[c](index[n], t)  =   stan::math::log(prob[c](index[n], t)); //     fn_EIGEN_double( prob[c](index[n], t),  "log", vect_type_log);

                                                 ///////// grad stuff
                                                 if ( (Phi_type == "Phi_approx") || (Phi_type == "Phi_approx_2") ) { // vect_type
                                                   phi_Bound_Z[c](index[n], t)  =         ( a_times_3 * Bound_Z[c](index[n], t)*Bound_Z[c](index[n], t) + b  )  *  Bound_U_Phi_Bound_Z[c](index[n], t) * (1.0 -  Bound_U_Phi_Bound_Z[c](index[n], t) )   ;
                                                   phi_Z_recip[c](index[n], t)  =    1.0 / ((  ( a_times_3 * Z_std_norm[c](index[n], t)*Z_std_norm[c](index[n], t)+ b  )  ) *  Phi_Z[c](index[n], t) * (1.0 -  Phi_Z[c](index[n], t))  ) ;
                                                 }  else if (Phi_type == "Phi")   {
                                                   phi_Bound_Z[c](index[n], t)  =         sqrt_2_pi_recip * stan::math::exp( ( - 0.5 * Bound_Z[c](index[n], t) * Bound_Z[c](index[n], t) )) ;//   fn_EIGEN_double( ( - 0.5 * Bound_Z[c](index[n], t).square() ).matrix(),  "exp", vect_type_exp);
                                                   phi_Z_recip[c](index[n], t)  =    1.0 /  (   sqrt_2_pi_recip * stan::math::exp(( - 0.5 * Z_std_norm[c](index[n], t) * Z_std_norm[c](index[n], t) )) ) ; //  fn_EIGEN_double( ( - 0.5 * Z_std_norm[c](index[n], t).square() ).matrix(),  "exp", vect_type_exp) );
                                                 }

                                                 log_phi_Bound_Z[c](index[n], t) =  stan::math::log(stan::math::abs(phi_Bound_Z[c](index[n], t))); //  fn_EIGEN_double(  phi_Bound_Z[c](index[n], t).abs(),  "log", vect_type_log);    ///////
                                                 bound_log(log_phi_Bound_Z[c](index[n], t)); /// add bounds checking

                                                 log_phi_Z_recip[c](index[n], t) =   stan::math::log(stan::math::abs(phi_Z_recip[c](index[n], t)));   //  fn_EIGEN_double(  phi_Z_recip[c](index[n], t).abs(),  "log", vect_type_log);    ///////
                                                 bound_log(log_phi_Z_recip[c](index[n], t)); /// add bounds checking

                                                 log_Z_std_norm[c](index[n], t)    =   stan::math::log(stan::math::abs(Z_std_norm[c](index[n], t)));   //  fn_EIGEN_double(Z_std_norm[c](index[n], t).abs().matrix(), "log", vect_type_log);
                                                 bound_log(log_Z_std_norm[c](index[n], t)); /// add bounds checking

                                             } catch (const std::exception &e) {

                                                   std::cout << "Exception caught during indexing operation:\n";
                                                   std::cout << "Exception: " << e.what() << "\n";
                                                   std::cout << "index size: " << index.size() << "\n";

                                                   std::cout << "First few indices: ";
                                                   for(int i = 0; i < std::min(5, (int)index.size()); i++) {
                                                     std::cout << index[i] << " ";
                                                   }

                                                   std::cout << "\nLast: ";
                                                   for(int i = std::max(0, (int)index.size() - 5); i < index.size(); i++) {
                                                     std::cout << index[i] << " ";
                                                   }

                                                   const int problem_index = index[n];

                                                   std::cout << "For problem index:\n";
                                                   std::cout << "y_chunk = " << y_chunk(problem_index, t) << "\n";
                                                   std::cout << "Bound_U_Phi = " << Bound_U_Phi_Bound_Z[c](problem_index, t) << "\n";
                                                   std::cout << "u_array = " << u_array(problem_index, t) << "\n";
                                                   std::cout << "Result = " << Phi_Z[c](problem_index, t) << "\n";


                                                   std::cout << "u_array values for high indices:\n";
                                                   for(int i = 990; i < 1000; i++) {
                                                     std::cout << "u_array[" << i << ", " << t << "] = " << u_array(i, t) << "\n";
                                                   }

                                                   std::cout << "\n";

                                             } catch (const std::runtime_error &e) {

                                                   std::cout << "Eigen::RuntimeError caught: " << e.what() << "\n";

                                             } catch (const std::out_of_range &e) {

                                                   std::cout << "out_of_range error caught: " << e.what() << "\n";

                                             }  catch (...) {

                                                   std::cout << "Unknown exception type caught\n";

                                             }

                                       } /// end of n loop

                                     }
                                  
                                  }

                         }
                        
                      
                        
                        if (num_overflows > 0) { //// overflow

                                std::vector<int>   index = over_index;
                                const int index_size = index.size();

                                  {


                                     // { ///// LOG-scale
                                     // 
                                     //      Eigen::Matrix<double, -1, 1> log_Bound_U_Phi_Bound_Z_1m =  ( fn_EIGEN_double( - Bound_Z[c](index, t), "log_Phi_approx",  vect_type_log_Phi) ); /// TEMP
                                     //      log_Bound_U_Phi_Bound_Z_1m = log_Bound_U_Phi_Bound_Z_1m.array().min(700.0).max(-700.0); //// add bounds checking
                                     //      Eigen::Matrix<double, -1, 1> Bound_U_Phi_Bound_Z_1m =     fn_EIGEN_double( log_Bound_U_Phi_Bound_Z_1m, "exp",  vect_type_exp);
                                     // 
                                     //      Eigen::Matrix<double, -1, 1> log_Bound_U_Phi_Bound_Z     =  fn_EIGEN_double(Bound_U_Phi_Bound_Z_1m, "log1m",  vect_type_log);
                                     //      log_Bound_U_Phi_Bound_Z = log_Bound_U_Phi_Bound_Z.array().min(700.0).max(-700.0); //// add bounds checking
                                     //      // Eigen::Matrix<double, -1, 1>  log_Bound_U_Phi_Bound_Z =  stan::math::log1m_exp(log_Bound_U_Phi_Bound_Z_1m); //// use log1m_exp for stability!
                                     //      // log_Bound_U_Phi_Bound_Z = log_Bound_U_Phi_Bound_Z.array().min(700.0).max(-700.0); //// add bounds checking
                                     // 
                                     //      Bound_U_Phi_Bound_Z[c](index, t).array() =   1.0 - Bound_U_Phi_Bound_Z_1m.array(); //// this is computed but not actually used?
                                     // 
                                     //      Eigen::Matrix<double, -1, -1>  tmp_array_2d_to_lse(num_overflows, 2);
                                     //      tmp_array_2d_to_lse.col(0)   =  (log_Bound_U_Phi_Bound_Z_1m + fn_EIGEN_double(u_array(index, t), "log",  vect_type_log)).matrix() ;
                                     //      tmp_array_2d_to_lse.col(1)  =    log_Bound_U_Phi_Bound_Z;
                                     //      tmp_array_2d_to_lse = tmp_array_2d_to_lse.array().min(700.0).max(-700.0); //// add bounds checking
                                     // 
                                     //      Eigen::Matrix<double, -1, 1> log_Phi_Z  =      fn_log_sum_exp_2d_double(tmp_array_2d_to_lse, vect_type_lse, skip_checks_lse);
                                     //      log_Phi_Z = log_Phi_Z.array().min(700.0).max(-700.0); //// add bounds checking
                                     // 
                                     //      Phi_Z[c](index, t)  =   fn_EIGEN_double(log_Phi_Z, "exp",  vect_type_exp);
                                     //      Eigen::Matrix<double, -1, 1> log_1m_Phi_Z  =  fn_EIGEN_double(u_array(index, t), "log1m",  vect_type_log)  + log_Bound_U_Phi_Bound_Z_1m;
                                     //      log_1m_Phi_Z = log_1m_Phi_Z.array().min(700.0).max(-700.0); //// add bounds checking
                                     // 
                                     //      Eigen::Matrix<double, -1, 1> logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
                                     //      Z_std_norm[c](index, t).array()    =  inv_Phi_approx_from_logit_prob(logit_Phi_Z);  // fn_EIGEN_double(logit_Phi_Z, "inv_Phi_approx_from_logit_prob", vect_type_inv_Phi_approx_from_logit_prob);
                                     // 
                                     //      log_Z_std_norm[c](index, t)    = fn_EIGEN_double(Z_std_norm[c](index, t).array().abs().matrix(), "log", vect_type_log);
                                     //      log_Z_std_norm[c](index, t) = log_Z_std_norm[c](index, t).array().min(700.0).max(-700.0); //// add bounds checking
                                     // 
                                     //      y1_log_prob[c](index, t)  =    log_Bound_U_Phi_Bound_Z_1m;
                                     //      y1_log_prob[c](index, t) = y1_log_prob[c](index, t).array().min(700.0).max(-700.0); //// add bounds checking
                                     // 
                                     //      prob[c](index, t)   =     Bound_U_Phi_Bound_Z_1m;  //// this is computed but not actually used?
                                     // 
                                     //      log_phi_Bound_Z[c](index, t).array()  =         stan::math::log( a_times_3 * Bound_Z[c](index, t).array().square() + b  ).array()  +   log_Bound_U_Phi_Bound_Z.array()  +   log_Bound_U_Phi_Bound_Z_1m.array();
                                     //      log_phi_Bound_Z[c](index, t) = log_phi_Bound_Z[c](index, t).array().min(700.0).max(-700.0); //// add bounds checking
                                     // 
                                     //      log_phi_Z_recip[c](index, t).array()  =    - (   stan::math::log(  ( a_times_3 * Z_std_norm[c](index, t).array().square() + b  ).array()  ).array()  +   log_Phi_Z.array()  +  log_1m_Phi_Z.array()  ).array() ;
                                     //      log_phi_Z_recip[c](index, t) = log_phi_Z_recip[c](index, t).array().min(700.0).max(-700.0); //// add bounds checking
                                     // 
                                     //      phi_Bound_Z[c](index, t).array() =  fn_EIGEN_double(  phi_Bound_Z[c](index, t),  "exp", vect_type_exp);    ///////
                                     //      phi_Z_recip[c](index, t).array() =  fn_EIGEN_double(  log_phi_Z_recip[c](index, t),  "exp", vect_type_exp);    ///////
                                     // 
                                     // }


                                     {  ///// NON-log-scale (STOPS WORKING IF THIS IS UNCOMMENTED)

                                       for (int n = 0; n < index_size; ++n) {

                                         try {

                                               Bound_U_Phi_Bound_Z[c](index[n], t) = stan::math::Phi(Bound_Z[c](index[n], t)); ///   fn_EIGEN_double( Bound_Z[c](index[n], t), Phi_type, vect_type_Phi);  /////
                                               Phi_Z[c](index[n], t) = y_chunk(index[n], t) * Bound_U_Phi_Bound_Z[c](index[n], t) +   (y_chunk(index[n], t) -  Bound_U_Phi_Bound_Z[c](index[n], t)) *   ((y_chunk(index[n], t)  + (y_chunk(index[n], t)  - 1.0)) * u_array(index[n], t));
                                               Z_std_norm[c](index[n], t) =   stan::math::inv_Phi(Phi_Z[c](index[n], t)); //  fn_EIGEN_double( Phi_Z[c](index[n], t),   inv_Phi_type, vect_type_inv_Phi);      ////
                                               prob[c](index[n], t) =    y_chunk(index[n], t)  * (1.0 - Bound_U_Phi_Bound_Z[c](index[n], t) ) + ( y_chunk(index[n], t)  -  1.0)  *  Bound_U_Phi_Bound_Z[c](index[n], t) * ( y_chunk(index[n], t)  +  (  y_chunk(index[n], t)  - 1.0)  )  ;
                                               y1_log_prob[c](index[n], t)  =   stan::math::log(prob[c](index[n], t)); //     fn_EIGEN_double( prob[c](index[n], t),  "log", vect_type_log);

                                               ///////// grad stuff
                                               if ( (Phi_type == "Phi_approx") || (Phi_type == "Phi_approx_2") ) { // vect_type
                                                 phi_Bound_Z[c](index[n], t)  =         ( a_times_3 * Bound_Z[c](index[n], t)*Bound_Z[c](index[n], t) + b  )  *  Bound_U_Phi_Bound_Z[c](index[n], t) * (1.0 -  Bound_U_Phi_Bound_Z[c](index[n], t) )   ;
                                                 phi_Z_recip[c](index[n], t)  =    1.0 / ((  ( a_times_3 * Z_std_norm[c](index[n], t)*Z_std_norm[c](index[n], t)+ b  )  ) *  Phi_Z[c](index[n], t) * (1.0 -  Phi_Z[c](index[n], t))  ) ;
                                               }  else if (Phi_type == "Phi")   {
                                                 phi_Bound_Z[c](index[n], t)  =         sqrt_2_pi_recip * stan::math::exp( ( - 0.5 * Bound_Z[c](index[n], t) * Bound_Z[c](index[n], t) )) ;//   fn_EIGEN_double( ( - 0.5 * Bound_Z[c](index[n], t).square() ).matrix(),  "exp", vect_type_exp);
                                                 phi_Z_recip[c](index[n], t)  =    1.0 /  (   sqrt_2_pi_recip * stan::math::exp(( - 0.5 * Z_std_norm[c](index[n], t) * Z_std_norm[c](index[n], t) )) ) ; //  fn_EIGEN_double( ( - 0.5 * Z_std_norm[c](index[n], t).square() ).matrix(),  "exp", vect_type_exp) );
                                               }

                                               log_phi_Bound_Z[c](index[n], t) =  stan::math::log(stan::math::abs(phi_Bound_Z[c](index[n], t))); //  fn_EIGEN_double(  phi_Bound_Z[c](index[n], t).abs(),  "log", vect_type_log);    ///////
                                               bound_log(log_phi_Bound_Z[c](index[n], t)); /// add bounds checking

                                               log_phi_Z_recip[c](index[n], t) =   stan::math::log(stan::math::abs(phi_Z_recip[c](index[n], t)));   //  fn_EIGEN_double(  phi_Z_recip[c](index[n], t).abs(),  "log", vect_type_log);    ///////
                                               bound_log(log_phi_Z_recip[c](index[n], t)); /// add bounds checking

                                               log_Z_std_norm[c](index[n], t)    =   stan::math::log(stan::math::abs(Z_std_norm[c](index[n], t)));   //  fn_EIGEN_double(Z_std_norm[c](index[n], t).abs().matrix(), "log", vect_type_log);
                                               bound_log(log_Z_std_norm[c](index[n], t)); /// add bounds checking

                                         } catch (const std::exception &e) {

                                               std::cout << "Exception caught during indexing operation:\n";
                                               std::cout << "Exception: " << e.what() << "\n";
                                               std::cout << "index size: " << index.size() << "\n";

                                               std::cout << "First few indices: ";
                                               for(int i = 0; i < std::min(5, (int)index.size()); i++) {
                                                 std::cout << index[i] << " ";
                                               }

                                               std::cout << "\nLast: ";
                                               for(int i = std::max(0, (int)index.size() - 5); i < index.size(); i++) {
                                                 std::cout << index[i] << " ";
                                               }

                                               const int problem_index = index[n];

                                               std::cout << "For problem index:\n";
                                               std::cout << "y_chunk = " << y_chunk(problem_index, t) << "\n";
                                               std::cout << "Bound_U_Phi = " << Bound_U_Phi_Bound_Z[c](problem_index, t) << "\n";
                                               std::cout << "u_array = " << u_array(problem_index, t) << "\n";
                                               std::cout << "Result = " << Phi_Z[c](problem_index, t) << "\n";

                                               std::cout << "u_array values for high indices:\n";
                                               for(int i = 990; i < 1000; i++) {
                                                 std::cout << "u_array[" << i << ", " << t << "] = " << u_array(i, t) << "\n";
                                               }

                                               std::cout << "\n";

                                               ///  std::cout << " Indexing failed for overflow" << ")\n";





                                         } catch (const std::runtime_error &e) {

                                               std::cout << "Eigen::RuntimeError caught: " << e.what() << "\n";

                                         } catch (const std::out_of_range &e) {

                                               std::cout << "out_of_range error caught: " << e.what() << "\n";

                                         }  catch (...) {

                                               std::cout << "Unknown exception type caught\n";

                                         }

                                       }

                                     }


                                  }
                                  
                           

                        }
                    
                        // log_phi_Bound_Z[c].array() = log_phi_Bound_Z[c].array().min(700.0).max(-700.0);     /// add bounds checking
                        // log_phi_Z_recip[c].array() = log_phi_Z_recip[c].array().min(700.0).max(-700.0);     /// add bounds checking
                        // log_Z_std_norm[c].array() = log_Z_std_norm[c].array().min(700.0).max(-700.0);     /// add bounds checking

              }  ///// end of "if overflow or underflow" block
            
                  if (t < n_tests - 1)    {
                    
                          inc_array.array()  =   ( Z_std_norm[c].leftCols(t + 1)  *   ( L_Omega_double[c].row(t+1).head(t+1).transpose()  ) ) ;
                          
                          // Eigen::Matrix<double, -1, -1> log_abs_mat =   log_Z_std_norm[c].leftCols(t + 1).array();
                          // Eigen::Matrix<double, -1, -1> sign_mat =      Z_std_norm[c].leftCols(t + 1).array().sign();
                          // Eigen::Matrix<double, -1, 1>  log_abs_vec = L_Omega_double[c].row(t+1).head(t+1).transpose().array().abs().log();
                          // Eigen::Matrix<double, -1, 1>  sign_vec =    L_Omega_double[c].row(t+1).head(t+1).transpose().array().sign();
                          // 
                          // log_abs_matrix_vector_mult_v1(log_abs_mat, 
                          //                               sign_mat,
                          //                               log_abs_vec,
                          //                               sign_vec,
                          //                               vect_type_exp,
                          //                               vect_type_log,
                          //                               log_abs_inc_array,
                          //                               sign_inc_array);
                          // 
                          // log_abs_inc_array.array() = log_abs_inc_array.array().min(700.0).max(-700.0);     /// add bounds checking
                    
                  }
              
            }
            /// / end of t loop
            prob_recip[c].array() = 1.0 / prob[c].array();
            lp_array.col(c).array() =     y1_log_prob[c].rowwise().sum().array() + log_prev(0, c) ;
          }
          // end of c loop
        }

        Eigen::Matrix<double, -1, -1> signs_Ones = Eigen::Matrix<double, -1, -1>::Ones(chunk_size, 2);
        
        log_abs_sum_exp_general_v2(lp_array,
                                   signs_Ones,
                                   vect_type_exp,
                                   vect_type_log,
                                   log_sum_result,
                                   sign_result,
                                   container_max_logs,
                                   container_sum_exp_signed);
        
        log_sum_result.array() = log_sum_result.array().min(700.0).max(-700.0);     /// add bounds checking
        
        out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size) = log_sum_result ; // fn_log_sum_exp_2d_double(lp_array,  vect_type_lse, skip_checks_lse).array() ;  //  fast_log_sum_exp_2d_AVX512(lp_array);
        prob_n  =  fn_EIGEN_double(out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size).matrix(), "exp",  vect_type_exp,  skip_checks_exp);
        prob_n_recip.array()  = 1.0 / prob_n.array();
 
        ///////////////////////////////////////////////
        const Eigen::VectorXi ALL_index = Eigen::VectorXi::LinSpaced(chunk_size, 0, chunk_size-1);
        const Eigen::VectorXi A_index = ALL_index; /// i_grad;
        const int A_index_size = chunk_size;// A_index.rows();
        ///////////////////////////////////////////////
         // if  (any_non_log_grad == true) {
              ///////////////////////////////////////////////
              
              ///////////////////////////////////////////////
              ///////////////////////////////////////////////
              A_common_grad_term_1.resize(A_index_size, n_tests);  A_common_grad_term_1.setZero();
              A_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(A_index_size, n_tests);  A_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.setZero();
              A_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(A_index_size, n_tests);  A_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.setZero();
              A_prop_rowwise_prod_temp.resize(A_index_size, n_tests);  A_prop_rowwise_prod_temp.setZero();
              A_prop_recip_rowwise_prod_temp.resize(A_index_size, n_tests); A_prop_recip_rowwise_prod_temp.setZero();
              ///////////////////////////////////////////////
              A_prod_container_or_inc_array.resize(A_index_size);  A_prod_container_or_inc_array.setZero();
              A_derivs_chain_container_vec.resize(A_index_size); A_derivs_chain_container_vec.setZero();
              A_prop_rowwise_prod_temp_all.resize(A_index_size); A_prop_rowwise_prod_temp_all.setZero();
              A_prob_n_recip.resize(A_index_size); A_prob_n_recip.setZero();
              ///////////////////////////////////////////////
              A_grad_prob.resize(A_index_size, n_tests); A_grad_prob.setZero();
              A_z_grad_term.resize(A_index_size, n_tests); A_z_grad_term.setZero();
              A_prob.resize(A_index_size, n_tests);  A_prob.setZero();
              A_prob_recip.resize(A_index_size, n_tests); A_prob_recip.setZero();
              A_phi_Z_recip.resize(A_index_size, n_tests); A_phi_Z_recip.setZero();
              A_phi_Bound_Z.resize(A_index_size, n_tests); A_phi_Bound_Z.setZero();
              A_Phi_Z.resize(A_index_size, n_tests); A_Phi_Z.setZero();
              A_Bound_Z.resize(A_index_size, n_tests); A_Bound_Z.setZero();
              A_Z_std_norm.resize(A_index_size, n_tests); A_Z_std_norm.setZero();
              ///////////////////////////////////////////////
              A_y_chunk.resize(A_index_size, n_tests);  
              A_u_array.resize(A_index_size, n_tests); 
              ///////////////////////////////////////////////
              A_u_grad_array_CM_chunk_block.resize(A_index_size, n_tests);
              A_y_m_y_sign_x_u.resize(A_index_size, n_tests);
              A_y_sign_chunk.resize(A_index_size, n_tests);
              ///////////////////////////////////////////////
              A_L_Omega_recip_double_array.resize(A_index_size, n_tests);
              ///////////////////////////////////////////////
         // }
        // ///////////////////////////////////////////////
        // const Eigen::VectorXi B_index = i_log_grad;
        // const int B_index_size = B_index.size();
        // ///////////////////////////////////////////////
        //     if (  (any_log_grad == true) && (B_index_size > 0) ) {
        //       B_y_chunk.resize(B_index_size, n_tests);   
        //       B_u_array.resize(B_index_size, n_tests);  
        //       ///////////////////////////////////////////////
        //       B_log_common_grad_term_1.resize(B_index_size, n_tests); B_log_common_grad_term_1.setConstant(-700.0);
        //       B_log_L_Omega_diag_recip_array.resize(B_index_size, n_tests); B_log_L_Omega_diag_recip_array.setConstant(-700.0);
        //       B_log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(B_index_size, n_tests); B_log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.setConstant(-700.0);
        //       B_log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(B_index_size, n_tests); B_log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.setConstant(-700.0);
        //       B_sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(B_index_size, n_tests);  B_sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.setOnes();
        //       B_sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(B_index_size, n_tests); B_sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.setOnes();
        //       B_log_prob_rowwise_prod_temp.resize(B_index_size, n_tests); B_log_prob_rowwise_prod_temp.setConstant(-700.0);
        //       B_log_prob_recip_rowwise_prod_temp.resize(B_index_size, n_tests); B_log_prob_recip_rowwise_prod_temp.setConstant(-700.0);
        //       /////////////////////////////////////
        //       B_log_abs_prod_container_or_inc_array.resize(B_index_size); B_log_abs_prod_container_or_inc_array.setConstant(-700.0);
        //       B_sign_prod_container_or_inc_array.resize(B_index_size); B_sign_prod_container_or_inc_array.setOnes();
        //       B_log_abs_derivs_chain_container_vec.resize(B_index_size); B_log_abs_derivs_chain_container_vec.setConstant(-700.0);
        //       B_sign_derivs_chain_container_vec.resize(B_index_size); B_sign_derivs_chain_container_vec.setOnes();
        //       ///////////////////////////////////////////////
        //       B_log_abs_prod_container_or_inc_array_comp.resize(B_index_size, n_tests); B_log_abs_prod_container_or_inc_array_comp.setConstant(-700.0);
        //       B_sign_prod_container_or_inc_array_comp.resize(B_index_size, n_tests); B_sign_prod_container_or_inc_array_comp.setOnes();
        //       B_log_abs_derivs_chain_container_vec_comp.resize(B_index_size, n_tests); B_log_abs_derivs_chain_container_vec_comp.setConstant(-700.0);
        //       B_sign_derivs_chain_container_vec_comp.resize(B_index_size, n_tests); B_sign_derivs_chain_container_vec_comp.setOnes();
        //       ///////////////////////////////////////////////
        //       B_log_prob_rowwise_prod_temp_all.resize(B_index_size); B_log_prob_rowwise_prod_temp_all.setConstant(-700.0);
        //       ///////////////////////////////////////////////
        //       B_log_abs_grad_prob.resize(B_index_size, n_tests); B_log_abs_grad_prob.setConstant(-700.0);
        //       B_log_abs_z_grad_term.resize(B_index_size, n_tests); B_log_abs_z_grad_term.setConstant(-700.0);
        //       B_sign_grad_prob.resize(B_index_size, n_tests); B_sign_grad_prob.setOnes();
        //       B_sign_z_grad_term.resize(B_index_size, n_tests); B_sign_z_grad_term.setOnes();
        //       ///////////////////////////////////////////
        //       B_y1_log_prob.resize(B_index_size, n_tests); B_y1_log_prob.setConstant(-700.0);  /////prob so sign always +'ve
        //       B_prob.resize(B_index_size, n_tests); B_prob.setConstant(-700.0); ///// prob so sign always +'ve
        //       B_Bound_Z.resize(B_index_size, n_tests); B_Bound_Z.setConstant(-700.0);
        //       B_Z_std_norm.resize(B_index_size, n_tests); B_Z_std_norm.setConstant(-700.0);
        //       ///////////////////////////////////////////////
        //       B_log_phi_Bound_Z.resize(B_index_size, n_tests); B_log_phi_Bound_Z.setConstant(-700.0);  ////density so sign always +'ve
        //       B_log_phi_Z_recip.resize(B_index_size, n_tests); B_log_phi_Z_recip.setConstant(-700.0); //// density so sign always +'ve
        //       B_log_Z_std_norm.resize(B_index_size, n_tests); B_log_Z_std_norm.setConstant(-700.0);
        //       B_sign_Z_std_norm.resize(B_index_size, n_tests); B_sign_Z_std_norm.setOnes();
        //       B_log_abs_Bound_Z.resize(B_index_size, n_tests); B_log_abs_Bound_Z.setConstant(-700.0);
        //       B_sign_Bound_Z.resize(B_index_size, n_tests); B_sign_Bound_Z.setOnes();
        //       ///////////////////////////////////////////////
        //       B_prob_n_recip.resize(B_index_size);   B_prob_n_recip = prob_n_recip(B_index);
        //       ///////////////////////////////////////////////
        //       B_log_abs_beta_grad_array_col_for_each_n.resize(B_index_size);  B_log_abs_beta_grad_array_col_for_each_n.setConstant(-700.0);
        //       B_log_abs_Omega_grad_array_col_for_each_n.resize(B_index_size);  B_log_abs_Omega_grad_array_col_for_each_n.setConstant(-700.0);
        //       B_log_prob_n_recip.resize(B_index_size); B_log_prob_n_recip.setConstant(-700.0);
        //       B_sign_beta_grad_array_col_for_each_n.resize(B_index_size); B_sign_beta_grad_array_col_for_each_n.setOnes();
        //       B_sign_Omega_grad_array_col_for_each_n.resize(B_index_size); B_sign_Omega_grad_array_col_for_each_n.setOnes();
        //       ///////////////////////////////////////////////
        //       B_log_abs_a.resize(B_index_size); B_log_abs_a.setConstant(-700.0);
        //       B_log_abs_b.resize(B_index_size); B_log_abs_b.setConstant(-700.0);
        //       B_log_sum_result.resize(B_index_size); B_log_sum_result.setConstant(-700.0);
        //       B_log_terms.resize(B_index_size, n_tests); B_log_terms.setConstant(-700.0);
        //       B_final_log_sum.resize(B_index_size) ; B_final_log_sum.setConstant(-700.0);
        //       B_sign_a.resize(B_index_size); B_sign_a.setOnes();
        //       B_sign_b.resize(B_index_size); B_sign_b.setOnes();
        //       B_sign_sum_result.resize(B_index_size); B_sign_sum_result.setOnes();
        //       B_sign_terms.resize(B_index_size, n_tests); B_sign_terms.setOnes();
        //       B_final_sign.resize(B_index_size); B_final_sign.setOnes();
        //       B_container_max_logs.resize(B_index_size); B_container_max_logs.setConstant(-700.0);
        //       B_container_sum_exp_signed.resize(B_index_size); B_container_sum_exp_signed.setZero();
        //       ///////////////////////////////////////////////
        //       B_log_abs_prev_grad_array_col_for_each_n.resize(B_index_size); B_log_abs_prev_grad_array_col_for_each_n.setConstant(-700.0);
        //       B_sign_prev_grad_array_col_for_each_n.resize(B_index_size); B_sign_prev_grad_array_col_for_each_n.setOnes();
        //       ///////////////////////////////////////////////
        //       B_y_sign.resize(B_index_size, n_tests);
        //       B_y_m_y_sign_x_u.resize(B_index_size, n_tests);
        //       ///////////////////////////////////////////////
        //       B_log_prob_recip.resize(B_index_size, n_tests);
        //       ///////////////////////////////////////////////
        //     }
        
        
        
  const int B_index_size = 0;
  
  
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1>     A_common_grad_term_1   =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1>     A_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip   =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests) ;
  Eigen::Matrix<double, -1, -1>     A_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip   =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests) ;
  Eigen::Matrix<double, -1, -1>     A_prop_rowwise_prod_temp   =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests) ;
  Eigen::Matrix<double, -1, -1>     A_prop_recip_rowwise_prod_temp   =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests) ;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1>      A_prod_container_or_inc_array  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  Eigen::Matrix<double, -1, 1>      A_derivs_chain_container_vec  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  Eigen::Matrix<double, -1, 1>      A_prop_rowwise_prod_temp_all  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1>     A_grad_prob =    Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1>     A_z_grad_term =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1>     A_prob =    Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1>     A_prob_recip =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1>     A_phi_Z_recip =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1>     A_phi_Bound_Z =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1>     A_Phi_Z =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1>     A_Bound_Z =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1>     A_Z_std_norm =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, 1>      A_prob_n_recip =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1>     A_y_chunk =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1>     A_u_array =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1>     A_y_sign_chunk =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1>     A_y_m_y_sign_x_u =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1>     A_u_grad_array_CM_chunk_block  =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1>     A_L_Omega_recip_double_array  =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
        
        
  /////////////////  ------------------------- compute grad  ---------------------------------------------------------------------------------
  
  if (A_index_size > 0) {
    
    A_y_chunk = y_chunk(A_index, Eigen::all);
    A_u_array = u_array(A_index, Eigen::all);
    A_y_sign_chunk = ( (A_y_chunk.array()  + (A_y_chunk.array() - 1.0)) ).matrix();
    A_y_m_y_sign_x_u = ( A_y_chunk.array()  - A_y_sign_chunk.array() * A_u_array.array() ).matrix();
    A_prob_n_recip = prob_n_recip(A_index);
    
  }
  
  // if (B_index_size > 0) {
  //   
  //   B_log_prob_n_recip.array() =   fn_EIGEN_double(  B_prob_n_recip, "log",  vect_type_log);
  //   B_log_prob_n_recip.array() = B_log_prob_n_recip.array().min(700.0).max(-700.0);     /// add bounds checking
  //   B_y_chunk  = y_chunk(B_index, Eigen::all);
  //   B_u_array  = u_array(B_index, Eigen::all);
  //   B_y_sign = ( (B_y_chunk.array()  + (B_y_chunk.array() - 1.0)) ).matrix();
  //   B_y_m_y_sign_x_u = ( B_y_chunk.array()  - B_y_sign.array() * B_u_array.array() ).matrix();
  //   
  // }
  
  for (int c = 0; c < n_class; c++) {
 

      if (grad_option != "none") {   
 
          //  if  (any_non_log_grad == true) {
              
                    A_prob =  (prob[c](A_index, Eigen::all));
                    A_prob_recip =  (prob_recip[c](A_index, Eigen::all));
                    A_phi_Bound_Z =  (phi_Bound_Z[c](A_index, Eigen::all));
                    A_phi_Z_recip =  (phi_Z_recip[c](A_index, Eigen::all));
                    A_Phi_Z =  (Phi_Z[c](A_index, Eigen::all));
                    A_Bound_Z =  (Bound_Z[c](A_index, Eigen::all));
                    A_Z_std_norm =  (Z_std_norm[c](A_index, Eigen::all));
      
                    fn_MVP_grad_prep(       n_class,
                                            Phi_type,
                                            vect_type_exp,
                                            skip_checks_exp,
                                            A_prob,
                                            A_y_sign_chunk,
                                            A_y_m_y_sign_x_u,
                                            L_Omega_recip_double[c],
                                            prev(0, c),
                                            A_prob_n_recip,
                                            A_phi_Z_recip,
                                            A_phi_Bound_Z,
                                            A_prob_recip,
                                            A_prop_rowwise_prod_temp,
                                            A_prop_recip_rowwise_prod_temp,
                                            A_prop_rowwise_prod_temp_all,
                                            A_common_grad_term_1,
                                            A_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                            A_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                            A_L_Omega_recip_double_array, 
                                            Model_args_as_cpp_struct);
                    
          //   }
       
       
            // if (B_index_size > 0) {
            //   
            //                 B_y1_log_prob = (y1_log_prob[c](B_index, Eigen::all));
            //                 B_prob = (prob[c](B_index, Eigen::all));
            //                 B_log_phi_Bound_Z =  (log_phi_Bound_Z[c](B_index, Eigen::all));
            //                 B_log_phi_Z_recip =  (log_phi_Z_recip[c](B_index, Eigen::all));
            //                 B_Bound_Z =    (Bound_Z[c](B_index, Eigen::all));
            //                 B_Z_std_norm =     (Z_std_norm[c](B_index, Eigen::all));
            //                 B_sign_Z_std_norm = B_Z_std_norm.array().sign().matrix();
            //                 B_log_Z_std_norm = log_Z_std_norm[c](B_index, Eigen::all);   ///  fn_EIGEN_double(B_Z_std_norm.array().abs().matrix(), "log", vect_type_log, skip_checks_log);  ; //  log_Z_std_norm[c](B_index, Eigen::all);
            //                 B_log_abs_Bound_Z =    log_abs_Bound_Z[c](B_index, Eigen::all); ///   fn_EIGEN_double(B_Bound_Z.array().abs().matrix(), "log", vect_type_log, skip_checks_log);
            //                 B_sign_Bound_Z = sign_Bound_Z[c](B_index, Eigen::all);
            //                 B_log_prob_recip = ( - B_y1_log_prob.array() ).matrix() ; //  fn_EIGEN_double( 1.0 / B_prob.array(), "log", vect_type_log, skip_checks_log);
            //   
            //                 for (int i = 0; i < n_tests; i++) {
            //                   int t = n_tests - (i + 1) ;
            //                   B_log_prob_rowwise_prod_temp.col(t).array()    =               B_y1_log_prob.block(0, t + 0, B_index_size, i + 1).rowwise().sum().array();
            //                   B_log_prob_recip_rowwise_prod_temp.col(t).array()  =        B_log_prob_recip.block(0, t + 0, B_index_size, i + 1).rowwise().sum().array();
            //                 }
            //                 
            //                 B_log_prob_rowwise_prod_temp.array() = B_log_prob_rowwise_prod_temp.array().min(700.0).max(-700.0);     /// add bounds checking
            //                 B_log_prob_recip_rowwise_prod_temp.array() = B_log_prob_recip_rowwise_prod_temp.array().min(700.0).max(-700.0);     /// add bounds checking
            //                 
            //                   B_log_prob_rowwise_prod_temp_all  =   B_y1_log_prob.rowwise().sum();
            //                   B_log_prob_rowwise_prod_temp_all.array() = B_log_prob_rowwise_prod_temp_all.array().min(700.0).max(-700.0);     /// add bounds checking
            //   
            //                 if (n_class > 1) { ///// i.e. if latent class
            //                   
            //                       for (int i = 0; i < n_tests; i++) {
            //                         int t = n_tests - (i + 1) ;
            //                         B_log_common_grad_term_1.col(t) =     (  log_prev(0, c) + B_log_prob_n_recip.array() ) + B_y1_log_prob.rowwise().sum().array()  +    B_log_prob_recip_rowwise_prod_temp.col(t).array()   ;
            //                       }
            //                       
            //                       B_log_common_grad_term_1.array() = B_log_common_grad_term_1.array().min(700.0).max(-700.0);     /// add bounds checking
            //                       
            //                 } else { 
            //                   
            //                         B_log_common_grad_term_1.setConstant(-700);
            //                   
            //                 }
            //                 
            //              
            //   
            //                 for (int t = 0; t < n_tests; t++) {
            //   
            //                   B_log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   =   B_log_phi_Bound_Z.col(t).array() + stan::math::log(stan::math::abs(L_Omega_recip_double[c](t, t))) ; 
            //                                                                                                    
            //                   B_log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array() = fn_EIGEN_double( B_y_m_y_sign_x_u.col(t).array().abs().matrix(), "log",  vect_type_log).array()
            //                                                                                                                         + B_log_phi_Z_recip.col(t).array()  + B_log_phi_Bound_Z.col(t).array()  +
            //                                                                                                                           stan::math::log(stan::math::abs(L_Omega_recip_double[c](t, t)));
            //   
            //                   //// note that densities and probs are always positive so signs = +1
            //                   B_sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()  = B_y_sign.col(t).array().sign() *  stan::math::sign(L_Omega_recip_double[c](t, t)) ; 
            //                   B_sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array()  = B_y_m_y_sign_x_u.col(t).array().sign() *  stan::math::sign(L_Omega_recip_double[c](t, t)) ;
            //   
            //                 }
            //                 
            //                 B_log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.array() = B_log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.array().min(700.0).max(-700.0);     /// add bounds checking
            //                 B_log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.array() = B_log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.array().min(700.0).max(-700.0);     /// add bounds checking
            // 
            //        }

          }

          ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of nuisance parameters / u's (manual)
          if ( (grad_option == "us_only") || (grad_option == "all") ) {
              
             // if  (any_non_log_grad == true) {

                A_u_grad_array_CM_chunk_block =  u_grad_array_CM_chunk(A_index, Eigen::all).block(0, 0, A_index_size, n_tests);

                fn_MVP_compute_nuisance_grad_v2(n_class,
                                                A_u_grad_array_CM_chunk_block,
                                                A_phi_Z_recip,
                                                A_common_grad_term_1,
                                                L_Omega_double[c],
                                                A_prob,
                                                A_prob_recip,
                                                A_prop_rowwise_prod_temp,
                                                A_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                A_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                A_z_grad_term,
                                                A_grad_prob,
                                                A_prod_container_or_inc_array,
                                                A_derivs_chain_container_vec,
                                                Model_args_as_cpp_struct);

                u_grad_array_CM_chunk(A_index, Eigen::all).block(0, 0, A_index_size, n_tests).array() += A_u_grad_array_CM_chunk_block.array() ;


              //}

              

            if (B_index_size > 0) {

              // Eigen::Matrix<double, -1, -1>   B_log_abs_u_grad_array_CM_chunk_block = log_abs_u_grad_array_CM_chunk(B_index, Eigen::all).block(0, 0, B_index_size, n_tests);
              // Eigen::Matrix<double, -1, -1>   B_u_grad_array_CM_chunk_block =         u_grad_array_CM_chunk(B_index, Eigen::all).block(0, 0, B_index_size, n_tests);
              // fn_MVP_compute_nuisance_grad_log_scale(   n_class,
              //                                           B_log_abs_u_grad_array_CM_chunk_block,
              //                                           B_u_grad_array_CM_chunk_block,
              //                                           vect_type_exp,
              //                                           L_Omega_double[c],
              //                                           log_abs_L_Omega_double[c],
              //                                           B_log_phi_Z_recip,
              //                                           B_y1_log_prob,
              //                                           B_log_prob_recip,
              //                                           B_log_prob_rowwise_prod_temp,
              //                                           B_log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
              //                                           B_sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
              //                                           B_log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
              //                                           B_sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
              //                                           B_log_common_grad_term_1,
              //                                           B_log_abs_z_grad_term,
              //                                           B_sign_z_grad_term,
              //                                           B_log_abs_grad_prob,
              //                                           B_sign_grad_prob,
              //                                           B_log_abs_prod_container_or_inc_array,
              //                                           B_sign_prod_container_or_inc_array,
              //                                           B_log_sum_result,
              //                                           B_sign_sum_result,
              //                                           B_log_terms,
              //                                           B_sign_terms,
              //                                           B_log_abs_a,
              //                                           B_log_abs_b,
              //                                           B_sign_a,
              //                                           B_sign_b,
              //                                           B_container_max_logs,
              //                                           B_container_sum_exp_signed,
              //                                           Model_args_as_cpp_struct);
              
              // using namespace stan::math;
              // 
              // Eigen::Matrix<double, -1, -1>  B_common_grad_term_1  = exp(B_log_common_grad_term_1).array() ;
              // Eigen::Matrix<double, -1, -1>  B_prob = exp(B_y1_log_prob);
              // Eigen::Matrix<double, -1, -1>  B_prob_recip = 1.0 / B_prob.array();
              // Eigen::Matrix<double, -1, -1>  B_prop_rowwise_prod_temp = exp(B_log_prob_rowwise_prod_temp);
              // Eigen::Matrix<double, -1, -1>  B_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = exp(B_log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip).array() * B_sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.array();
              // Eigen::Matrix<double, -1, -1>  B_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = exp(B_log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip).array() * B_sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.array();
              // 
              // Eigen::Matrix<double, -1, -1>  B_phi_Z_recip = B_log_phi_Z_recip.array().exp() ; /// * B_phi_Z_recip.array().sign();
              // 
              // Eigen::Matrix<double, -1, -1>  B_u_grad_array_CM_chunk_block =         u_grad_array_CM_chunk(B_index, Eigen::all).block(0, 0, B_index_size, n_tests);
              // 
              // fn_MVP_compute_nuisance_grad_v2(n_class,
              //                                 B_u_grad_array_CM_chunk_block,
              //                                 B_phi_Z_recip,
              //                                 B_common_grad_term_1,
              //                                 L_Omega_double[c],
              //                                 B_prob,
              //                                 B_prob_recip,
              //                                 B_prop_rowwise_prod_temp,
              //                                 B_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
              //                                 B_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
              //                                 B_log_abs_z_grad_term,
              //                                 B_log_abs_grad_prob,
              //                                 B_log_abs_prod_container_or_inc_array,
              //                                 B_sign_prod_container_or_inc_array,
              //                                 Model_args_as_cpp_struct);
              // 
              // 
              //   u_grad_array_CM_chunk(B_index, Eigen::all).block(0, 0, B_index_size, n_tests).array() += B_u_grad_array_CM_chunk_block.array() ;

              }
              
              if (c == n_class - 1) {
                
                  //// update output vector once all u_grad computations are done 
                  out_mat.segment(1, n_us).segment(chunk_size_orig * n_tests * chunk_counter , chunk_size * n_tests).array()  =  u_grad_array_CM_chunk.reshaped();
                  
                  //// account for unconstrained -> constrained transformations and Jacobian adjustments
                  du_wrt_duu_chunk =  fn_MVP_nuisance_first_deriv(   u_vec_chunk,
                                                                     u_unc_vec_chunk,
                                                                     nuisance_transformation,
                                                                     vect_type_exp);
                   
                  d_J_wrt_duu_chunk =  fn_MVP_nuisance_deriv_of_log_det_J(    u_vec_chunk,
                                                                              u_unc_vec_chunk,
                                                                              nuisance_transformation,
                                                                              du_wrt_duu_chunk);
                   
                  out_mat.segment(1, n_us).segment(chunk_size_orig * n_tests * chunk_counter , chunk_size * n_tests).array() =  
                         out_mat.segment(1, n_us).segment(chunk_size_orig * n_tests * chunk_counter , chunk_size * n_tests).array() * du_wrt_duu_chunk.array() + d_J_wrt_duu_chunk.array() ;
                  
              }
            


          }
          ///////////////////////////////////////////////////////////////////////////// Grad of intercepts / coefficients (beta's)

          if ( (grad_option == "main_only") || (grad_option == "all") ) {
            {   // compute (some or all) of grads on log-scale


             // if  (any_non_log_grad == true)  {

                Eigen::Matrix<int, -1, 1> n_covariates_per_outcome_vec_temp =   n_covariates_per_outcome_vec.row(c).transpose();
                
                fn_MVP_compute_coefficients_grad_v2(   n_class,
                                                       beta_grad_array[c],
                                                       chunk_counter,
                                                       X[c],
                                                       n_covariates_max,
                                                       n_covariates_per_outcome_vec_temp,
                                                       A_common_grad_term_1,
                                                       L_Omega_double[c],
                                                       A_prob,
                                                       A_prob_recip,
                                                       A_prop_rowwise_prod_temp,
                                                       A_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                       A_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                       A_z_grad_term,
                                                       A_grad_prob,
                                                       A_prod_container_or_inc_array,
                                                       A_derivs_chain_container_vec,
                                                       Model_args_as_cpp_struct);

              //}


              // if (B_index_size > 0) {
              //   
              //   using namespace stan::math;
              // 
              //   Eigen::Matrix<double, -1, -1>  B_common_grad_term_1  = exp(B_log_common_grad_term_1).array() ;
              //   Eigen::Matrix<double, -1, -1>  B_prob = exp(B_y1_log_prob);
              //   Eigen::Matrix<double, -1, -1>  B_prob_recip = 1.0 / B_prob.array();
              //   Eigen::Matrix<double, -1, -1>  B_prop_rowwise_prod_temp = exp(B_log_prob_rowwise_prod_temp);
              //   Eigen::Matrix<double, -1, -1>  B_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = exp(B_log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip).array() * B_sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.array();
              //   Eigen::Matrix<double, -1, -1>  B_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = exp(B_log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip).array() * B_sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.array();
              // 
              //   Eigen::Matrix<int, -1, 1> n_covariates_per_outcome_vec_temp =   n_covariates_per_outcome_vec.row(c).transpose();
              // 
              //   fn_MVP_compute_coefficients_grad_v2(   n_class,
              //                                          beta_grad_array[c],
              //                                          chunk_counter,
              //                                          X[c],
              //                                          n_covariates_max,
              //                                          n_covariates_per_outcome_vec_temp,
              //                                          B_common_grad_term_1,
              //                                          L_Omega_double[c],
              //                                          B_prob,
              //                                          B_prob_recip,
              //                                          B_prop_rowwise_prod_temp,
              //                                          B_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
              //                                          B_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
              //                                          B_log_abs_z_grad_term,
              //                                          B_log_abs_grad_prob,
              //                                          B_log_abs_prod_container_or_inc_array,
              //                                          B_sign_prod_container_or_inc_array_comp,
              //                                          Model_args_as_cpp_struct);
              //   
              // 
              //   // fn_MVP_compute_coefficients_grad_log_scale( n_class,
              //   //                                             beta_grad_array[c],
              //   //                                             B_sign_beta_grad_array_col_for_each_n,
              //   //                                             B_log_abs_beta_grad_array_col_for_each_n,
              //   //                                             vect_type_exp,
              //   //                                             L_Omega_double[c],
              //   //                                             log_abs_L_Omega_double[c],
              //   //                                             B_log_phi_Z_recip,
              //   //                                             B_y1_log_prob,
              //   //                                             B_log_prob_rowwise_prod_temp,
              //   //                                             B_log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
              //   //                                             B_sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
              //   //                                             B_log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
              //   //                                             B_sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
              //   //                                             B_log_common_grad_term_1,
              //   //                                             B_log_abs_z_grad_term,
              //   //                                             B_sign_z_grad_term,
              //   //                                             B_log_abs_grad_prob,
              //   //                                             B_sign_grad_prob,
              //   //                                             B_log_abs_prod_container_or_inc_array,
              //   //                                             B_sign_prod_container_or_inc_array,
              //   //                                             B_log_abs_prod_container_or_inc_array_comp,
              //   //                                             B_sign_prod_container_or_inc_array_comp,
              //   //                                             B_log_sum_result,
              //   //                                             B_sign_sum_result,
              //   //                                             B_log_terms,
              //   //                                             B_sign_terms,
              //   //                                             B_container_max_logs,
              //   //                                             B_container_sum_exp_signed,
              //   //                                             Model_args_as_cpp_struct);
              // 
              //   }

              }

            }

          ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of L_Omega ('s)
          if ( (grad_option == "main_only") || (grad_option == "all") ) {
            
               //   if  (any_non_log_grad == true)  {

                          fn_MVP_compute_L_Omega_grad_v2(       n_class,
                                                                U_Omega_grad_array[c],
                                                                A_common_grad_term_1,
                                                                L_Omega_double[c],
                                                                A_prob,
                                                                A_prob_recip,
                                                                A_Bound_Z,
                                                                A_Z_std_norm,
                                                                A_prop_rowwise_prod_temp,
                                                                A_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                                A_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                                A_z_grad_term,
                                                                A_grad_prob,
                                                                A_prod_container_or_inc_array,
                                                                A_derivs_chain_container_vec,
                                                                Model_args_as_cpp_struct);


                 //  }
 
            //    if (B_index_size > 0) {
            //      
            //      
            //      using namespace stan::math;
            //      
            //      Eigen::Matrix<double, -1, -1>  B_common_grad_term_1  = exp(B_log_common_grad_term_1).array() ;
            //      Eigen::Matrix<double, -1, -1>  B_prob = exp(B_y1_log_prob);
            //      Eigen::Matrix<double, -1, -1>  B_prob_recip = 1.0 / B_prob.array();
            //      Eigen::Matrix<double, -1, -1>  B_prop_rowwise_prod_temp = exp(B_log_prob_rowwise_prod_temp);
            //      Eigen::Matrix<double, -1, -1>  B_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = exp(B_log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip).array() * B_sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.array();
            //      Eigen::Matrix<double, -1, -1>  B_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = exp(B_log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip).array() * B_sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.array();
            //      
            //      Eigen::Matrix<double, -1, -1> B_Bound_Z = B_log_abs_Bound_Z.array().exp() * B_sign_Bound_Z.array();
            //      Eigen::Matrix<double, -1, -1> B_Z_std_norm = B_log_Z_std_norm.array().exp() * B_sign_Z_std_norm.array();
            //      
            //      Eigen::Matrix<int, -1, 1> n_covariates_per_outcome_vec_temp =   n_covariates_per_outcome_vec.row(c).transpose();
            //      
            //      
            //      fn_MVP_compute_L_Omega_grad_v2(       n_class,
            //                                            U_Omega_grad_array[c],
            //                                            B_common_grad_term_1,
            //                                            L_Omega_double[c],
            //                                            B_prob,
            //                                            B_prob_recip,
            //                                            B_Bound_Z,
            //                                            B_Z_std_norm,
            //                                            B_prop_rowwise_prod_temp,
            //                                            B_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
            //                                            B_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
            //                                            B_log_abs_z_grad_term,
            //                                            B_log_abs_grad_prob,
            //                                            B_log_abs_prod_container_or_inc_array,
            //                                            B_sign_prod_container_or_inc_array,
            //                                            Model_args_as_cpp_struct);
            //      
            //      
            // 
            //     // fn_MVP_compute_L_Omega_grad_log_scale(      n_class,
            //     //                                             U_Omega_grad_array[c],
            //     //                                             B_sign_Omega_grad_array_col_for_each_n,
            //     //                                             B_log_abs_Omega_grad_array_col_for_each_n,
            //     //                                             B_log_abs_Bound_Z, // not in other grads
            //     //                                             B_sign_Bound_Z, // not in other grads
            //     //                                             B_log_Z_std_norm, // not in other grads
            //     //                                             B_sign_Z_std_norm, // not in other grads
            //     //                                             vect_type_exp,
            //     //                                             L_Omega_double[c],
            //     //                                             log_abs_L_Omega_double[c],
            //     //                                             B_log_phi_Z_recip,
            //     //                                             B_y1_log_prob,
            //     //                                             B_log_prob_rowwise_prod_temp,
            //     //                                             B_log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
            //     //                                             B_sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
            //     //                                             B_log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
            //     //                                             B_sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
            //     //                                             B_log_common_grad_term_1,
            //     //                                             B_log_abs_z_grad_term,
            //     //                                             B_sign_z_grad_term,
            //     //                                             B_log_abs_grad_prob,
            //     //                                             B_sign_grad_prob,
            //     //                                             B_log_abs_prod_container_or_inc_array,
            //     //                                             B_sign_prod_container_or_inc_array,
            //     //                                             B_log_abs_prod_container_or_inc_array_comp,
            //     //                                             B_sign_prod_container_or_inc_array_comp,
            //     //                                             B_log_abs_derivs_chain_container_vec_comp,
            //     //                                             B_sign_derivs_chain_container_vec_comp,
            //     //                                             B_log_sum_result,
            //     //                                             B_sign_sum_result,
            //     //                                             B_log_terms,
            //     //                                             B_sign_terms,
            //     //                                             B_log_abs_a,
            //     //                                             B_log_abs_b,
            //     //                                             B_sign_a,
            //     //                                             B_sign_b,
            //     //                                             B_container_max_logs,
            //     //                                             B_container_sum_exp_signed,
            //     //                                             Model_args_as_cpp_struct);
            // 
            // }
               
          }

          if (n_class > 1) { /// prevelance only estimated for latent class models
            
              if ( (grad_option == "main_only") || (grad_option == "all") ) {
    
                         // if (B_index_size > 0) {
                         // 
                         //      // B_log_abs_prev_grad_array_col_for_each_n   =    B_log_prob_n_recip.array() + B_y1_log_prob.rowwise().sum().array() ; 
                         //      // B_sign_prev_grad_array_col_for_each_n.setOnes(); //// just a vector of +1's since probs are always positive 
                         //      // 
                         //      // // Final scalar grad using log-sum-exp
                         //      // LogSumVecSingedResult log_sum_vec_signed_struct = log_sum_vec_signed_v1(B_log_abs_prev_grad_array_col_for_each_n,
                         //      //                                                                         B_sign_prev_grad_array_col_for_each_n, 
                         //      //                                                                         vect_type);
                         //      // prev_grad_vec(c)  +=   stan::math::exp(log_sum_vec_signed_struct.log_sum) * log_sum_vec_signed_struct.sign;
                         //      // 
                         //      // 
                         // 
                         //        using namespace stan::math;
                         //        Eigen::Matrix<double, -1, -1>  B_prob = exp(B_y1_log_prob);
                         //        Eigen::Matrix<double, -1, 1> B_prob_n_recip = exp(B_log_prob_n_recip);
                         //   
                         //        prev_grad_vec(c)  +=  ( (B_prob_n_recip.array() ) *  B_prob.rowwise().prod().array() ).matrix().sum()  ;
                         // 
                         //   }
    
                        //  if  (any_non_log_grad == true) {
    
                              prev_grad_vec(c)  +=  ( (A_prob_n_recip.array() ) *  A_prob.rowwise().prod().array() ).matrix().sum()  ;
    
                        //  }
                          
              }
              
          }

        }

      } /// end of chunk block
  
    }



    //////////////////////// gradients for latent class membership probabilitie(s) (i.e. disease prevalence)
    if (n_class > 1) {
        for (int c = 0; c < n_class; c++) {
          prev_unconstrained_grad_vec(c)  =   prev_grad_vec(c)   * deriv_p_wrt_pu_double ;
        }
        prev_unconstrained_grad_vec(0) = prev_unconstrained_grad_vec(1) - prev_unconstrained_grad_vec(0) - 2 * tanh_u_prev[1];
        prev_unconstrained_grad_vec_out(0) = prev_unconstrained_grad_vec(0);
    }

    log_prob_out += out_mat.segment(1 + n_params, N).sum();       // log_prob_out += log_lik.sum();
    if (exclude_priors == false)  log_prob_out += prior_densities;
    log_prob_out +=  log_jac_u;
    log_prob_out += log_jac_p_double;

    int i = 0; // probs_all_range.prod() cancels out
    for (int c = 0; c < n_class; c++ ) {
      for (int t = 0; t < n_tests; t++) {
        for (int k = 0; k <  n_covariates_per_outcome_vec(c, t); k++) {
          if (exclude_priors == false) {
            beta_grad_array[c](k, t) +=  - ((beta_double_array[c](k, t) - prior_coeffs_mean[c](k, t)) / prior_coeffs_sd[c](k, t) ) * (1.0/ prior_coeffs_sd[c](k, t) ) ;     // add normal prior density derivative to gradient
          }
          beta_grad_vec(i) = beta_grad_array[c](k, t);
          i += 1;
        }
      }
    }
    
    {
      int i = 0;
      for (int c = 0; c < n_class; c++ ) {
        for (int t1 = 0; t1 < n_tests  ; t1++ ) {
          for (int t2 = 0; t2 <  t1 + 1; t2++ ) {
            L_Omega_grad_vec(i) = U_Omega_grad_array[c](t1,t2);
            i += 1;
          }
        }
      }
    }
    
    Eigen::Matrix<double, -1, 1>  grad_wrt_L_Omega_nd(dim_choose_2 + n_tests);
    Eigen::Matrix<double, -1, 1>  grad_wrt_L_Omega_d(dim_choose_2 + n_tests);
    
    if (n_class > 1) {
      grad_wrt_L_Omega_nd =   L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
      grad_wrt_L_Omega_d =   L_Omega_grad_vec.segment(dim_choose_2 + n_tests, dim_choose_2 + n_tests);
      U_Omega_grad_vec.head(dim_choose_2) =  ( grad_wrt_L_Omega_nd.transpose()  *  deriv_L_wrt_unc_full[0].cast<double>() ).transpose() ;
      U_Omega_grad_vec.segment(dim_choose_2, dim_choose_2) =   ( grad_wrt_L_Omega_d.transpose()  *  deriv_L_wrt_unc_full[1].cast<double>() ).transpose()  ;
    } else { 
      grad_wrt_L_Omega_nd =   L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
      U_Omega_grad_vec.head(dim_choose_2) =  ( grad_wrt_L_Omega_nd.transpose()  *  deriv_L_wrt_unc_full[0].cast<double>() ).transpose() ;
    }
    
  }

  {  ////////////////////////////  outputs // add log grad and sign stuff';///////////////
    
    out_mat(0) =  log_prob_out;
    out_mat.segment(1 + n_us, n_corrs) = target_AD_grad ;          // .cast<float>();
    out_mat.segment(1 + n_us, n_corrs) += U_Omega_grad_vec ;        //.cast<float>()  ;
    out_mat.segment(1 + n_us + n_corrs, n_covariates_total) = beta_grad_vec ; //.cast<float>() ;
    out_mat(1 + n_us + n_corrs + n_covariates_total) = ((grad_prev_AD +  prev_unconstrained_grad_vec_out(0)));
    
  }

//  return(out_mat);

}























// 
// Internal function using Eigen::Ref as inputs for matrices
void     fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace(   Eigen::Matrix<double, -1, 1> &&out_mat_R_val,
                                                                   const Eigen::Matrix<double, -1, 1> &&theta_main_vec_R_val,
                                                                   const Eigen::Matrix<double, -1, 1> &&theta_us_vec_R_val,
                                                                   const Eigen::Matrix<int, -1, -1> &&y_R_val,
                                                                   const std::string &grad_option,
                                                                   const Model_fn_args_struct &Model_args_as_cpp_struct




) {


  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat_ref(out_mat_R_val);
  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref(theta_main_vec_R_val);  // create Eigen::Ref from R-value
  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref(theta_us_vec_R_val);  // create Eigen::Ref from R-value
  const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref(y_R_val);  // create Eigen::Ref from R-value


  fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process( out_mat_ref,
                                                                  theta_main_vec_ref,
                                                                  theta_us_vec_ref,
                                                                  y_ref,
                                                                  grad_option,
                                                                  Model_args_as_cpp_struct);


}








// Internal function using Eigen::Ref as inputs for matrices
void     fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace(   Eigen::Matrix<double, -1, 1> &out_mat_ref,
                                                                   const Eigen::Matrix<double, -1, 1> &theta_main_vec_ref,
                                                                   const Eigen::Matrix<double, -1, 1> &theta_us_vec_ref,
                                                                   const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                   const std::string &grad_option,
                                                                   const Model_fn_args_struct &Model_args_as_cpp_struct




) {


  fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process( out_mat_ref,
                                                                  theta_main_vec_ref,
                                                                  theta_us_vec_ref,
                                                                  y_ref,
                                                                  grad_option,
                                                                  Model_args_as_cpp_struct);


}





// Internal function using Eigen::Ref as inputs for matrices
template <typename MatrixType>
void     fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace(   Eigen::Ref<Eigen::Block<MatrixType, -1, 1>>  &out_mat_ref,
                                                                   const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_main_vec_ref,
                                                                   const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_us_vec_ref,
                                                                   const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                   const std::string &grad_option,
                                                                   const Model_fn_args_struct &Model_args_as_cpp_struct




) {


  fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process( out_mat_ref,
                                                                  theta_main_vec_ref,
                                                                  theta_us_vec_ref,
                                                                  y_ref,
                                                                  grad_option,
                                                                  Model_args_as_cpp_struct);


}














// Internal function using Eigen::Ref as inputs for matrices
Eigen::Matrix<double, -1, 1>    fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD(  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                 const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                 const std::string &grad_option,
                                                                                 const Model_fn_args_struct &Model_args_as_cpp_struct




) {

  int n_params_main = theta_main_vec_ref.rows();
  int n_us = theta_us_vec_ref.rows();
  int n_params = n_us + n_params_main;
  int N = y_ref.rows();

  Eigen::Matrix<double, -1, 1> out_mat = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);

  fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace( out_mat,
                                                          theta_main_vec_ref,
                                                          theta_us_vec_ref,
                                                          y_ref,
                                                          grad_option,
                                                          Model_args_as_cpp_struct);

  return out_mat;

}





// 













 



 
 
 
 
 
 