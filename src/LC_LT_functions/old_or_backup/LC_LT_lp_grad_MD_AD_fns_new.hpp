#pragma once


 
 
#include <Eigen/Dense>
 
 

 


 




 
inline  void      fn_lp_grad_LT_LC_NoLog_MD_and_AD_InPlace_process(      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat ,
                                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                         const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                         const std::string &grad_option,
                                                                         const Model_fn_args_struct &Model_args_as_cpp_struct
) { 
  
  out_mat.setZero(); //// set log_prob and grad vec to zero at the start (only do this on outer fns, not inner/likelihood fns)
  
  
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
  const int n_class = 2; //// Model_args_as_cpp_struct.Model_args_ints(1);
  const int ub_threshold_phi_approx = Model_args_as_cpp_struct.Model_args_ints(2);
  const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
  
  const double prev_prior_a = Model_args_as_cpp_struct.Model_args_doubles(0);
  const double prev_prior_b = Model_args_as_cpp_struct.Model_args_doubles(1);
  const double overflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(2);
  const double underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(3);
  
  std::string vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
  const std::string &Phi_type = Model_args_as_cpp_struct.Model_args_strings(1);
  const std::string &inv_Phi_type = Model_args_as_cpp_struct.Model_args_strings(2);
  std::string vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);
  std::string vect_type_log = Model_args_as_cpp_struct.Model_args_strings(4);
  std::string vect_type_lse = Model_args_as_cpp_struct.Model_args_strings(5);
  std::string vect_type_tanh = Model_args_as_cpp_struct.Model_args_strings(6);
  std::string vect_type_Phi = Model_args_as_cpp_struct.Model_args_strings(7);
  std::string vect_type_log_Phi = Model_args_as_cpp_struct.Model_args_strings(8);
  std::string vect_type_inv_Phi = Model_args_as_cpp_struct.Model_args_strings(9);
  std::string vect_type_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_strings(10);
  // const std::string grad_option =  Model_args_as_cpp_struct.Model_args_strings(11);
  const std::string &nuisance_transformation =   Model_args_as_cpp_struct.Model_args_strings(12);
  
  //  const Eigen::Matrix<double, -1, 1>  lkj_cholesky_eta =   Model_args_as_cpp_struct.Model_args_col_vecs_double[0]; // not needed for latent_trait
  
  const Eigen::Matrix<double, -1, -1> &LT_b_priors_shape  = Model_args_as_cpp_struct.Model_args_mats_double[0]; 
  const Eigen::Matrix<double, -1, -1> &LT_b_priors_scale  = Model_args_as_cpp_struct.Model_args_mats_double[1]; 
  const Eigen::Matrix<double, -1, -1> &LT_known_bs_indicator = Model_args_as_cpp_struct.Model_args_mats_double[2]; 
  const Eigen::Matrix<double, -1, -1> &LT_known_bs_values = Model_args_as_cpp_struct.Model_args_mats_double[3]; 
  
  const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
  
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &prior_coeffs_mean  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &prior_coeffs_sd   =  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1]; 
  // const std::vector<Eigen::Matrix<double, -1, -1 > >   prior_for_corr_a   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2];   // not needed for latent_trait
  // const std::vector<Eigen::Matrix<double, -1, -1 > >   prior_for_corr_b   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3];   // not needed for latent_trait
  // const std::vector<Eigen::Matrix<double, -1, -1 > >   lb_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4];  // not needed for latent_trait
  // const std::vector<Eigen::Matrix<double, -1, -1 > >   ub_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5];  // not needed for latent_trait
  // const std::vector<Eigen::Matrix<double, -1, -1 > >   known_values    = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6];  // not needed for latent_trait
  // 
  // const std::vector<Eigen::Matrix<int, -1, -1 >> known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0]; // not needed for latent_trait
  
  ////////////// Note: latent_trait doesn't currently support covariates
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
  const double Inf = std::numeric_limits<double>::infinity();
  
  //// ---- determine chunk size --------------------------------------------------
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
  
  ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, desired_n_chunks);
  
  int chunk_size = chunk_size_info.chunk_size;
  const int chunk_size_orig = chunk_size_info.chunk_size_orig;
  const int normal_chunk_size = chunk_size_info.normal_chunk_size;
  const int last_chunk_size = chunk_size_info.last_chunk_size;
  const int n_total_chunks = chunk_size_info.n_total_chunks;
  const int n_full_chunks = chunk_size_info.n_full_chunks;
  
  //////////////  --------------------------------------------------------------------
  //// corrs (LT_b's)
  const int n_b_LT = n_class * n_tests;
  Eigen::Matrix<double, -1, 1  >  LT_b_raw_vec_dbl  = theta_main_vec_ref.head(n_b_LT) ;
  Eigen::Matrix<double, -1, -1 >  LT_b_mat_dbl =    Eigen::Matrix<double, -1, -1 >::Zero(n_class, n_tests); // assign later (in AD block)
  ////
  std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_dbl =  vec_of_mats<double>(n_tests, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_recip_dbl =  vec_of_mats<double>(n_tests, n_tests, n_class);
  
  //// coeffs
  const int n_a_LT = n_class * n_tests;
  const int n_coeffs = n_class * n_tests; //// latent-trait currently does not support covariates!
  Eigen::Matrix<double, -1, 1>    LT_a_vec_dbl  = Eigen::Matrix<double, -1, 1>::Zero(n_a_LT); ////  theta_main_vec_ref.segment(0 + n_b_LT, n_a_LT);
  Eigen::Matrix<double, -1, -1>   LT_a_mat_dbl = Eigen::Matrix<double, -1, -1>::Zero(n_class, n_tests);
  
  {
    int i = n_b_LT;
    for (int c = 0; c < n_class; ++c) {
      for (int t = 0; t < n_tests; ++t) {
         //// for (int k = 0; k < n_covariates_per_outcome_vec(c, t); ++k) {
          LT_a_vec_dbl(i - n_b_LT) = theta_main_vec_ref(i);
          LT_a_mat_dbl(c, t) = theta_main_vec_ref(i);
          i += 1;
         //// }
      }
    }
  }
  
  //// prev
  const double u_prev_diseased = theta_main_vec_ref(n_params_main - 1);
  
  //// 
  double prior_densities_LT_b_dbl = 0.0;
  double log_det_J_LT_b_dbl = 0.0;
  Eigen::Matrix<double, -1, 1>  grad_LT_b_raw_priors_and_log_det_J = Eigen::Matrix<double, -1, 1>::Zero(n_b_LT);
  
  double prior_densities_LT_a_dbl = 0.0;
  double log_det_J_LT_a_dbl = 0.0;
  Eigen::Matrix<double, -1, 1>  grad_LT_a_priors_and_log_det_J= Eigen::Matrix<double, -1, 1>::Zero(n_a_LT);
  
  double prior_densities_prev_dbl = 0.0;
  double log_det_J_prev_dbl = 0.0;
  double grad_prev_raw_priors_and_log_det_J = 0.0;
  
  //// latent-trait only (global vars):
  std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> Jacobian_d_L_Sigma_wrt_b_3d_arrays_dbl =  vec_of_vec_of_mats<double>(n_tests, n_tests, n_tests, n_class);
  
  {  ///////////   -------------------  start of AD block  ------------------------------------------------------------------------------------------------------------------
    
        stan::math::start_nested();  ///////////////////////

        stan::math::var target_AD = 0.0; //// checked
        ////
        std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  L_Omega_var = vec_of_mats_var(n_tests, n_tests, n_class); //// checked
        std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_var   = vec_of_mats_var(n_tests, n_tests, n_class); //// checked
        ////
        Eigen::Matrix<stan::math::var, -1, -1> identity_dim_T =  stan::math::identity_matrix(n_tests); //// checked
    
        //// LT_b's / correlation params  ---------------------------------------------------------------------------------------------------------------------- 
        stan::math::var LT_known_b_raw_sum = 0.0; //// checked
        ////
        Eigen::Matrix<stan::math::var, -1, 1>  LT_b_raw_vec_var =  stan::math::to_var(LT_b_raw_vec_dbl) ;  //// checked
        ////
        Eigen::Matrix<stan::math::var, -1, -1> LT_b_raw_mat_var =      Eigen::Matrix<stan::math::var, -1, -1>::Zero(n_class, n_tests);  //// checked
        LT_b_raw_mat_var.row(0) =  LT_b_raw_vec_var.segment(0, n_tests).transpose();  //// checked
        LT_b_raw_mat_var.row(1) =  LT_b_raw_vec_var.segment(n_tests, n_tests).transpose();  //// checked
        ////
        Eigen::Matrix<stan::math::var, -1, -1> LT_b_mat_var = stan::math::exp(LT_b_raw_mat_var) ;  //// checked
        LT_b_mat_dbl = LT_b_mat_var.val();
        Eigen::Matrix<stan::math::var, -1, 1>  LT_b_nd_var  =   LT_b_mat_var.row(0).transpose() ; //// checked
        Eigen::Matrix<stan::math::var, -1, 1>  LT_b_d_var   =   LT_b_mat_var.row(1).transpose() ; //// checked
      
        ////
        Omega_var[0] = identity_dim_T +  LT_b_nd_var * LT_b_nd_var.transpose();  //// checked
        Omega_var[1] = identity_dim_T +  LT_b_d_var *  LT_b_d_var.transpose(); //// checked
        ////
        for (int c = 0; c < n_class; ++c) {
          L_Omega_var[c]   = stan::math::cholesky_decompose(Omega_var[c]) ; //// checked
        }
        ////
        //// priors / Jacobians for  LT_b's / corr (using Weibull priors)
        stan::math::var prior_densities_LT_b_var = 0.0;  //// checked
        for (int t = 0; t < n_tests; ++t) {
          prior_densities_LT_b_var += stan::math::weibull_lpdf(  LT_b_nd_var(t) ,   LT_b_priors_shape(0, t), LT_b_priors_scale(0, t)  ); //// checked
          prior_densities_LT_b_var += stan::math::weibull_lpdf(  LT_b_d_var(t)  ,   LT_b_priors_shape(1, t), LT_b_priors_scale(1, t)  ); //// checked
        }
        ////
        target_AD += prior_densities_LT_b_var; //// checked
        prior_densities_LT_b_dbl += prior_densities_LT_b_var.val(); //// checked
        ////
        stan::math::var log_det_J_LT_b_var = 0.0;
        log_det_J_LT_b_var +=  (LT_b_raw_mat_var).sum()  - LT_known_b_raw_sum ; // Jacobian b -> raw_b //// checked
        target_AD += log_det_J_LT_b_var;   //// checked
        log_det_J_LT_b_dbl += log_det_J_LT_b_var.val(); //// checked
        ////
        
        {
          ////////////////////////////////////////////////////////////
          target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject) //// checked
          grad_LT_b_raw_priors_and_log_det_J =  LT_b_raw_vec_var.adj();    // differentiating WRT this  //// checked
          out_mat.segment(1 + n_us, n_b_LT) =  grad_LT_b_raw_priors_and_log_det_J ;   //// add grad constribution to output //// checked
          stan::math::set_zero_all_adjoints(); //// checked
          ////////////////////////////////////////////////////////////  
        }
        
        
        //// coeffs (LT_a_mat_var's and LT_theta's)  ------------------------------------------------------------------------------------------------------------ 
        Eigen::Matrix<stan::math::var, -1, 1>  LT_a_vec_var =  stan::math::to_var(LT_a_vec_dbl); //// checked
        ////
        Eigen::Matrix<stan::math::var, -1, -1> LT_a_mat_var(n_class, n_tests); //// checked
        Eigen::Matrix<stan::math::var, -1, -1> LT_theta_mat_var(n_class, n_tests); //// checked
        ////
        {
          int i = 0 ;  
          for (int c = 0; c < n_class; ++c) {
            for (int t = 0; t < n_tests; ++t) {
              LT_a_mat_var(c, t) = LT_a_vec_var(i);
              i += 1;
            }
          }
        }
        ////
        //// LT_theta_mat_var as TRANSFORMED parameter (need Jacobian adj. if wish to put prior on theta!!!)
        for (int t = 0; t < n_tests; ++t) {
          LT_theta_mat_var(1, t)   =    LT_a_mat_var(1, t) /  stan::math::sqrt(1.0 + ( LT_b_d_var(t) *  LT_b_d_var(t)));
          LT_theta_mat_var(0, t)   =    LT_a_mat_var(0, t) /  stan::math::sqrt(1.0 + ( LT_b_nd_var(t) * LT_b_nd_var(t)));
        }
        ////
        //// priors / Jacobians for coeffs (LT_a's)
        stan::math::var prior_densities_LT_a_var = 0.0;
        stan::math::var log_det_J_LT_a_var = 0.0;
        ////
        for (int c = 0; c < n_class; ++c) {
          for (int t = 0; t < n_tests; ++t) {
            prior_densities_LT_a_var += stan::math::normal_lpdf(LT_theta_mat_var(c, t), prior_coeffs_mean[c](0, t), prior_coeffs_sd[c](0, t)); //// Prior on theta  (hence need Jacobian below!!)
            log_det_J_LT_a_var +=  - 0.5 * stan::math::log(1.0 + stan::math::square(stan::math::abs(LT_b_mat_var(c, t) ))); // Jacobian for LT_theta_mat_var -> LT_a_mat_var
          }
        }
        ////
        target_AD += prior_densities_LT_a_var;
        prior_densities_LT_a_dbl += prior_densities_LT_a_var.val();
        ////
        target_AD += log_det_J_LT_a_var;
        log_det_J_LT_a_dbl += log_det_J_LT_a_var.val();
        ////

        {
          //////////////////////////////////////////////////////////// 
          target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
          grad_LT_a_priors_and_log_det_J =  LT_a_vec_var.adj();    // differentiating WRT this 
          out_mat.segment(1 + n_us + n_b_LT, n_a_LT) =  grad_LT_a_priors_and_log_det_J ;   //// add grad constribution to output vec
          stan::math::set_zero_all_adjoints();
          ////////////////////////////////////////////////////////////  
        }
        
        
        //// Jacobian L_Sigma -> b's ------------------------------------------------------------------------------------------- 
        std::vector< std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > > Jacobian_d_L_Sigma_wrt_b_3d_arrays_var = vec_of_vec_of_mats<stan::math::var>(n_tests, n_tests, n_tests, n_class);

        for (int c = 0; c < n_class; ++c) {

          //  # -----------  wrt last b first
          int t = n_tests;
          stan::math::var sum_sq_1 = 0.0;
          for (int j = 1; j < t; ++j) {
            Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, j-1) = ( L_Omega_var[c](n_tests-1, j-1) / LT_b_mat_var(c, n_tests-1) ) ; 
            sum_sq_1 +=   LT_b_mat_var(c, j-1) * LT_b_mat_var(c, j-1) ;
          }
          stan::math::var big_denom_p1 =  1 + sum_sq_1;
          Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, n_tests-1) =   (1 / L_Omega_var[c](n_tests-1, n_tests-1) ) * ( LT_b_mat_var(c, n_tests-1) / big_denom_p1 ) ; 

          //  # -----------  wrt 2nd-to-last b
          t = n_tests - 1;
          sum_sq_1 = 0;
          stan::math::var  sum_sq_2 = 0.0;
          for (int j = 1; j < t + 1; ++j) {
            Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, j-1) = ( L_Omega_var[c](t-1, j-1) / LT_b_mat_var(c, t-1) ); 
            sum_sq_1 +=   LT_b_mat_var(c, j-1) * LT_b_mat_var(c, j-1) ;
            if (j < (t))   sum_sq_2 +=  LT_b_mat_var(c, j-1) * LT_b_mat_var(c, j-1) ;
          }
          big_denom_p1 =  1 + sum_sq_1;
          stan::math::var big_denom_p2 =  1 + sum_sq_2;
          stan::math::var  big_denom_part =  big_denom_p1 * big_denom_p2;
          Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, t-1) =   (1 / L_Omega_var[c](t-1, t-1)) * ( LT_b_mat_var(c, t-1) / big_denom_p2 ); 

          for (int j = t+1; j < n_tests + 1; ++j) {
            Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](j-1, t-1) =   ( 1/L_Omega_var[c](j-1, t-1) ) * (LT_b_mat_var(c, j-1) *  LT_b_mat_var(c, j-1)  ) * (   LT_b_mat_var(c, t-1)  / big_denom_part) * (1 - ( LT_b_mat_var(c, t-1) * LT_b_mat_var(c, t-1)  / big_denom_p1 ) );
          }

          Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t, t)   =  - ( 1/L_Omega_var[c](t, t) ) * (LT_b_mat_var(c, t) * LT_b_mat_var(c, t)) * ( LT_b_mat_var(c, t-1)  / (big_denom_p1*big_denom_p1));

          // # -----------  wrt rest of b's
          for (int t = 1; t < (n_tests - 2) + 1; ++t) {

            sum_sq_1  = 0;
            sum_sq_2  = 0;

            for (int j = 1; j < t + 1; ++j) {
              if (j < (t)) Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, j-1) = ( L_Omega_var[c](t-1, j-1) /  LT_b_mat_var(c, t-1) ) ; 
              sum_sq_1 +=   LT_b_mat_var(c, j-1) *   LT_b_mat_var(c, j-1) ;
              if (j < (t))   sum_sq_2 +=    LT_b_mat_var(c, j-1) *   LT_b_mat_var(c, j-1) ;
            }
            big_denom_p1 = 1 + sum_sq_1;
            big_denom_p2 = 1 + sum_sq_2;
            big_denom_part =  big_denom_p1 * big_denom_p2;

            Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, t-1) =   (1 / L_Omega_var[c](t-1, t-1) ) * (  LT_b_mat_var(c, t-1) / big_denom_p2 ) ; 

            for (int j = t + 1; j < n_tests + 1; ++j) {
              Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](j-1, t-1)  =   (1/L_Omega_var[c](j-1, t-1)) * (  LT_b_mat_var(c, j-1) *   LT_b_mat_var(c, j-1) ) * (   LT_b_mat_var(c, t-1) / big_denom_part) * (1 - ( ( LT_b_mat_var(c, t-1) *  LT_b_mat_var(c, t-1) ) / big_denom_p1 ) ) ;
            }

            for (int j = t + 1; j < n_tests ; ++j) {
              Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](j-1, j-1) =  - (1/L_Omega_var[c](j-1, j-1)) * (  LT_b_mat_var(c, j-1) *   LT_b_mat_var(c, j-1) ) * ( LT_b_mat_var(c, t-1) / (big_denom_p1*big_denom_p1)) ;
              big_denom_p1 = big_denom_p1 +   LT_b_mat_var(c, j-1) *   LT_b_mat_var(c, j-1) ;
              big_denom_p2 = big_denom_p2 + LT_b_mat_var(c, j-2) * LT_b_mat_var(c, j-2) ;
              big_denom_part =  big_denom_p1 * big_denom_p2 ;
              if (t < n_tests - 1) {
                for (int k = j + 1; k < n_tests + 1; ++k) {
                  Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](k-1, j-1) =   (-1 / L_Omega_var[c](k-1, j-1)) * (  LT_b_mat_var(c, k-1) *   LT_b_mat_var(c, k-1) ) * (  LT_b_mat_var(c, j-1) *   LT_b_mat_var(c, j-1) ) * (  LT_b_mat_var(c, t-1) / big_denom_part ) * ( ( 1 / big_denom_p2 )  +  ( 1 / big_denom_p1 ) ) ;
                }
              }
            }

            Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](n_tests-1, n_tests-1) =  - (1/L_Omega_var[c](n_tests-1, n_tests-1)) * (LT_b_mat_var(c, n_tests-1) * LT_b_mat_var(c, n_tests-1)) * ( LT_b_mat_var(c, t-1) / (big_denom_p1*big_denom_p1)) ;

          }

          for (int t1 = 0; t1 < n_tests; ++t1) {
            for (int t2 = 0; t2 < n_tests; ++t2) {
              for (int t3 = 0; t3 < n_tests; ++t3) {
                Jacobian_d_L_Sigma_wrt_b_3d_arrays_dbl[c][t1](t2, t3)    =      Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t1](t2, t3).val();
              }
            }
          }
        }
        
        for (int c = 0; c < n_class; ++c) {
          for (int t1 = 0; t1 < n_tests; ++t1) {
            for (int t2 = 0; t2 < n_tests; ++t2) {
              L_Omega_dbl[c](t1, t2) =   L_Omega_var[c](t1, t2).val();
              L_Omega_recip_dbl[c](t1, t2) =   1.0 / L_Omega_dbl[c](t1, t2) ;
            }
          }
        }
        
        /////////////  prev stuff  ---- vars
        {
          if (n_class > 1) {  //// if latent class
            
            std::vector<stan::math::var> 	 u_prev_var_vec_var(n_class, 0.0);
            std::vector<stan::math::var> 	 prev_var_vec_var(n_class, 0.0);
            std::vector<stan::math::var> 	 tanh_u_prev_var(n_class, 0.0);
            Eigen::Matrix<stan::math::var, -1, -1>	 prev_var = Eigen::Matrix<stan::math::var, -1, -1>::Zero(1, 2);
            stan::math::var tanh_pu_deriv_var = 0.0;
            stan::math::var deriv_p_wrt_pu_var = 0.0;
            stan::math::var tanh_pu_second_deriv_var = 0.0;
            stan::math::var log_jac_p_deriv_wrt_pu_var = 0.0;
            stan::math::var log_det_J_prev_var = 0.0;
            stan::math::var target_AD_prev = 0.0;
            
            u_prev_var_vec_var[1] =  stan::math::to_var(u_prev_diseased);
            tanh_u_prev_var[1] =  stan::math::tanh(u_prev_var_vec_var[1]); /// ( stan::math::exp(2.0*u_prev_var_vec_var[1] ) - 1.0) / ( stan::math::exp(2*u_prev_var_vec_var[1] ) + 1.0) ;
            u_prev_var_vec_var[0] =   0.5 *  stan::math::log( (1.0 + ( (1.0 - 0.5 * ( tanh_u_prev_var[1] + 1.0))*2.0 - 1.0) ) / (1.0 - ( (1.0 - 0.5 * ( tanh_u_prev_var[1] + 1.0))*2.0 - 1.0) ) )  ;
            tanh_u_prev_var[0] =  stan::math::tanh(u_prev_var_vec_var[0]); ///  (stan::math::exp(2.0*u_prev_var_vec_var[0] ) - 1.0) / ( stan::math::exp(2*u_prev_var_vec_var[0] ) + 1.0) ;
            
            prev_var_vec_var[1] =  0.5 * ( tanh_u_prev_var[1] + 1.0);
            prev_var_vec_var[0] =  0.5 * ( tanh_u_prev_var[0] + 1.0);
            prev_var(0,1) =  prev_var_vec_var[1];
            prev_var(0,0) =  prev_var_vec_var[0];
            
            tanh_pu_deriv_var = ( 1.0 - (tanh_u_prev_var[1] * tanh_u_prev_var[1])  );
            deriv_p_wrt_pu_var = 0.5 *  tanh_pu_deriv_var;
            tanh_pu_second_deriv_var  = -2.0 * tanh_u_prev_var[1]  * tanh_pu_deriv_var;
            log_jac_p_deriv_wrt_pu_var  = ( 1.0 / deriv_p_wrt_pu_var) * 0.5 * tanh_pu_second_deriv_var; // for gradient of u's
            
            log_det_J_prev_var =    stan::math::log( deriv_p_wrt_pu_var );
            log_det_J_prev_dbl =  log_det_J_prev_var.val() ;
            
            stan::math::var prior_densities_prev = beta_lpdf(prev_var(0, 1), prev_prior_a, prev_prior_b)  ;  // weakly informative prior - helps avoid boundaries with slight negative skew (for lower N)
            prior_densities_prev_dbl = prior_densities_prev.val();
            
            //  target_AD_prev += log_det_J_prev_var ; /// done manually later
            target_AD  +=  prior_densities_prev;
            
            ///////////////////////
            prior_densities_prev.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
            grad_prev_raw_priors_and_log_det_J  =  u_prev_var_vec_var[1].adj() - u_prev_var_vec_var[0].adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
            out_mat(1 + n_us + n_b_LT + n_a_LT) = grad_prev_raw_priors_and_log_det_J; //// add grad constribution to output vec
            stan::math::set_zero_all_adjoints();
            ///////////////////////
            
          }
        } 
        
        // target_AD += beta_lpdf(  prev_var(0, 1), prev_prior_a, prev_prior_b  ); // weakly informative prior - helps avoid boundaries with slight negative skew (for lower N)
        // target_AD += log_jac_p_var;
        // 
        // log_jac_p_dbl = log_jac_p_var.val();
        // prior_densities = target_AD.val() ; // target_AD_coeffs.val() + target_AD_corrs.val();
        
        // ///////////////////////
        // stan::math::set_zero_all_adjoints();
        // target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
        // grad_prev_AD  =  u_prev_var_vec_var[1].adj() - u_prev_var_vec_var[0].adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
        // stan::math::set_zero_all_adjoints();
        // ////////////////////////////////////////////////////////////  

        stan::math::recover_memory_nested(); //////////

  }  ///////////   -------------------  end of AD block   ------------------------------------
  
  
  /////////////  prev stuff
  std::vector<double> 	 u_prev_vec_dbl(n_class, 0.0);
  std::vector<double> 	 prev_vec_dbl(n_class, 0.0);
  std::vector<double> 	 tanh_u_prev_dbl(n_class, 0.0);
  Eigen::Matrix<double, -1, -1>	 prev_dbl = Eigen::Matrix<double, -1, -1>::Zero(1, n_class);
  double tanh_pu_deriv = 0.0;
  double deriv_p_wrt_pu_dbl = 0.0;
  double tanh_pu_second_deriv = 0.0;
  double log_jac_p_deriv_wrt_pu = 0.0;
  double log_jac_p = 0.0;
  
  if (n_class > 1) {  //// if latent class
    
    u_prev_vec_dbl[1] =  (double) u_prev_diseased ;
    tanh_u_prev_dbl[1] = stan::math::tanh(u_prev_vec_dbl[1]); 
    u_prev_vec_dbl[0] =   0.5 *  stan::math::log( (1.0 + ( (1.0 - 0.5 * ( tanh_u_prev_dbl[1] + 1.0))*2.0 - 1.0) ) / (1.0 - ( (1.0 - 0.5 * ( tanh_u_prev_dbl[1] + 1.0))*2.0 - 1.0) ) )  ;
    tanh_u_prev_dbl[0] = stan::math::tanh(u_prev_vec_dbl[0]); 
    
    prev_vec_dbl[1] =  0.5 * ( tanh_u_prev_dbl[1] + 1.0);
    prev_vec_dbl[0] =  0.5 * ( tanh_u_prev_dbl[0] + 1.0);
    prev_dbl(0,1) =  prev_vec_dbl[1];
    prev_dbl(0,0) =  prev_vec_dbl[0];
    
    tanh_pu_deriv = ( 1.0 - (tanh_u_prev_dbl[1] * tanh_u_prev_dbl[1])  );
    deriv_p_wrt_pu_dbl = 0.5 *  tanh_pu_deriv;
    tanh_pu_second_deriv  = -2.0 * tanh_u_prev_dbl[1]  * tanh_pu_deriv;
    log_jac_p_deriv_wrt_pu  = ( 1.0 / deriv_p_wrt_pu_dbl) * 0.5 * tanh_pu_second_deriv; // for gradient of u's
    log_jac_p =    stan::math::log( deriv_p_wrt_pu_dbl );
    
  }
  
  ///////////////////////////////////////////////////////////////////////// prior densities
  double prior_densities = 0.0;
   
  if (exclude_priors == false) {
    
    ///////////////////// priors for coeffs
    const double prior_densities_coeffs_dbl = prior_densities_LT_a_dbl;
    
    prior_densities += prior_densities_coeffs_dbl;
    prior_densities += prior_densities_LT_b_dbl;
    prior_densities += prior_densities_prev_dbl;
    
  }
  
 
  ////////  ------- likelihood function  ------------------------------------------------------------------------------------------------------------------------------------------------
  double log_prob_out = 0.0;
  
  //// Jacobian adjustments (none needed for coeffs as unconstrained - so only for L_Omega -> Omega and u_prev -> prev, and the one for u's is computed in the likelihood)
  const double log_det_J_main = log_det_J_prev_dbl + log_det_J_LT_b_dbl + log_det_J_LT_a_dbl;
  
  const Eigen::Matrix<double, -1, -1>  log_prev_dbl = stan::math::log(prev_dbl);
  
  //// define unconstrained nuisance parameter vec
  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec = theta_us_vec_ref;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //// coeffs:
  Eigen::Matrix<double , -1, 1>   beta_grad_vec   =  Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);  //
  std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array =     vec_of_mats<double>(n_covariates_max, n_tests, n_class);
  //// prev:
  Eigen::Matrix<double, -1, 1>  prev_unconstrained_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(n_class); //
  Eigen::Matrix<double, -1, 1>  prev_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(n_class); //
  Eigen::Matrix<double, -1, 1>  prev_unconstrained_grad_vec_out = Eigen::Matrix<double, -1, 1>::Zero(n_class - 1); //
  //// correlatins (LT_b's):
  Eigen::Matrix<double, -1, -1> grad_pi_wrt_b_raw =  Eigen::Matrix<double, -1, -1>::Zero(n_class, n_tests) ;
  Eigen::Matrix<double, -1, 1>  deriv_L_t1 =     Eigen::Matrix<double, -1, 1>::Zero( n_tests);
  Eigen::Matrix<double, -1, 1>  deriv_L_t1_output_vec =     Eigen::Matrix<double, -1, 1>::Zero( n_tests);
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  {
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::Matrix<double, -1, -1>> Z_std_norm =          vec_of_mats<double>(chunk_size, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> Bound_Z =             vec_of_mats<double>(chunk_size, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z = vec_of_mats<double>(chunk_size, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> prob =                vec_of_mats<double>(chunk_size, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> Phi_Z =               vec_of_mats<double>(chunk_size, n_tests, n_class);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> y1_log_prob =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1> phi_Z_recip = y1_log_prob;
  Eigen::Matrix<double, -1, -1> phi_Bound_Z = y1_log_prob;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk = y1_log_prob;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> common_grad_term_1 = y1_log_prob;
  Eigen::Matrix<double, -1, -1> y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = y1_log_prob;
  Eigen::Matrix<double, -1, -1> y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = y1_log_prob;
  Eigen::Matrix<double, -1, -1> prob_rowwise_prod_temp = y1_log_prob;
  Eigen::Matrix<double, -1, -1> prob_recip_rowwise_prod_temp = y1_log_prob;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> prod_container_or_inc_array =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  Eigen::Matrix<double, -1, 1> derivs_chain_container_vec =     Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  Eigen::Matrix<double, -1, 1> prob_rowwise_prod_temp_all =     Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> grad_prob =   y1_log_prob;
  Eigen::Matrix<double, -1, -1> z_grad_term = y1_log_prob;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> y_chunk = y1_log_prob;
  Eigen::Matrix<double, -1, -1> u_array = y1_log_prob;
  Eigen::Matrix<double, -1, -1> y_sign =  y1_log_prob;
  Eigen::Matrix<double, -1, -1> y_m_y_sign_x_u = y1_log_prob;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk_block = y1_log_prob;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> u_unc_vec_chunk =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
  Eigen::Matrix<double, -1, 1> u_vec_chunk =       Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
  Eigen::Matrix<double, -1, 1> du_wrt_duu_chunk =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
  Eigen::Matrix<double, -1, 1> d_J_wrt_duu_chunk = Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> lp_array = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_class);
  ///////////////////////////////////////////////
  double log_jac_u = 0.0;
  ///////////////////////////////////////////////
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> prob_n       =          Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  Eigen::Matrix<double, -1, 1> prob_n_recip       =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  Eigen::Matrix<double, -1, 1> log_sum_result     =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  Eigen::Matrix<double, -1, 1> container_max_logs =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  ///////////////////////////////////////////////
  //#ifdef _WIN32
      Eigen::Matrix<double, -1, 1>  rowwise_log_sum =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
      Eigen::Matrix<double, -1, 1>  rowwise_prod =      Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
      Eigen::Matrix<double, -1, 1>  rowwise_sum =       Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
      Eigen::Matrix<double, -1, 1>  log_lik_chunk =     Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  //#endif
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> prob_recip =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  /////////////////////////////////////////////// 
  //// for latent_trait:
  /////////////////////////////////////////////// 
  Eigen::Matrix<double, -1, -1> grad_bound_z = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1> grad_Phi_bound_z = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  std::vector<Eigen::Matrix<double, -1, -1 > >  deriv_Bound_Z_x_L = vec_of_mats(chunk_size, n_tests*2, 2) ;
  Eigen::Matrix< double , -1, 1>  temp_L_Omega_x_grad_z_sum_1 =  Eigen::Matrix< double , -1, 1>::Zero(chunk_size);
  Eigen::Matrix<double, -1, -1 >   deriv_inc  =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, 1> deriv_comp_2  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  /////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  { // start of big local block
    
    for (int nc = 0; nc < n_total_chunks; nc++) {

      int chunk_counter = nc;
      
      if ((chunk_counter == n_full_chunks) && (n_chunks > 1) && (last_chunk_size > 0)) { // Last chunk (remainder - don't use AVX / SIMD for this)
        
                          chunk_size = last_chunk_size;  //// update chunk_size
                          
                          //// use either Loop (i.e. double fn's) or Stan's vectorisation for the remainder (i.e. last) chunk, regardless of input
                          vect_type = "Stan";
                          vect_type_exp = "Stan";
                          vect_type_log = "Stan";
                          vect_type_lse = "Stan";
                          vect_type_tanh = "Stan";
                          vect_type_Phi =  "Stan";
                          vect_type_log_Phi = "Stan";
                          vect_type_inv_Phi = "Stan";
                          vect_type_inv_Phi_approx_from_logit_prob = "Stan";
                          
                          ///////////////////////////////////////////////
                          for (int c = 0; c < n_class; c++) {
                            Z_std_norm[c].resize(last_chunk_size, n_tests);
                            Bound_Z[c].resize(last_chunk_size, n_tests);
                            Bound_U_Phi_Bound_Z[c].resize(last_chunk_size, n_tests);
                            prob[c].resize(last_chunk_size, n_tests);
                            Phi_Z[c].resize(last_chunk_size, n_tests);
                          } 
                          ///////////////////////////////////////////////
                          y1_log_prob.resize(last_chunk_size, n_tests);
                          phi_Z_recip.resize(last_chunk_size, n_tests);
                          phi_Bound_Z.resize(last_chunk_size, n_tests);
                          ///////////////////////////////////////////////
                          u_grad_array_CM_chunk.resize(last_chunk_size, n_tests);
                          ///////////////////////////////////////////////
                          common_grad_term_1.resize(last_chunk_size, n_tests);
                          y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
                          y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
                          prob_rowwise_prod_temp.resize(last_chunk_size, n_tests);
                          prob_recip_rowwise_prod_temp.resize(last_chunk_size, n_tests);
                          ///////////////////////////////////////////////
                          prod_container_or_inc_array.resize(last_chunk_size);
                          derivs_chain_container_vec.resize(last_chunk_size);
                          prob_rowwise_prod_temp_all.resize(last_chunk_size);
                          ///////////////////////////////////////////////
                          grad_prob.resize(last_chunk_size, n_tests);
                          z_grad_term.resize(last_chunk_size, n_tests);
                          ///////////////////////////////////////////////
                          y_chunk.resize(last_chunk_size, n_tests);
                          u_array.resize(last_chunk_size, n_tests);
                          y_sign.resize(last_chunk_size, n_tests);
                          y_m_y_sign_x_u.resize(last_chunk_size, n_tests);
                          ///////////////////////////////////////////////
                          u_grad_array_CM_chunk_block.resize(last_chunk_size, n_tests);
                          ///////////////////////////////////////////////
                          u_unc_vec_chunk.resize(last_chunk_size * n_tests);
                          u_vec_chunk.resize(last_chunk_size * n_tests);
                          du_wrt_duu_chunk.resize(last_chunk_size * n_tests);
                          d_J_wrt_duu_chunk.resize(last_chunk_size * n_tests);
                          ///////////////////////////////////////////////
                          lp_array.resize(last_chunk_size, n_class);
                          ///////////////////////////////////////////////
                          prob_n.resize(last_chunk_size);
                          prob_n_recip.resize(last_chunk_size);
                          log_sum_result.resize(last_chunk_size);
                          container_max_logs.resize(last_chunk_size);
                          ///////////////////////////////////////////////
                          //#ifdef _WIN32
                          rowwise_log_sum.resize(last_chunk_size);
                          rowwise_prod.resize(last_chunk_size);
                          rowwise_sum.resize(last_chunk_size); 
                          log_lik_chunk.resize(last_chunk_size);
                          //#endif
                          ///////////////////////////////////////////////
                          prob_recip.resize(last_chunk_size, n_tests);
                          ///////////////////////////////////////////////
                          //// For latent_trait (not LC_MVP):
                          ///////////////////////////////////////////////
                          temp_L_Omega_x_grad_z_sum_1.resize(last_chunk_size);
                          deriv_comp_2.resize(last_chunk_size);
                          deriv_inc.resize(last_chunk_size, n_tests);
                          grad_bound_z.resize(last_chunk_size, n_tests);
                          grad_Phi_bound_z.resize(last_chunk_size, n_tests);
                          
                          for (int c = 0; c < n_class; c++) {
                            deriv_Bound_Z_x_L[c].resize(last_chunk_size, n_tests*2);
                          }
                          ///////////////////////////////////////////////
                          
        
      }
      
      ////-----------------------------------------------
      u_grad_array_CM_chunk.setZero(); //// reset between chunks as re-using same container
      
      y_chunk = y_ref.middleRows( chunk_size_orig * chunk_counter, chunk_size).cast<double>();
      
      //// Nuisance parameter transformation step
      u_unc_vec_chunk = u_unc_vec.segment( chunk_size_orig * n_tests * chunk_counter, chunk_size * n_tests);
      fn_MVP_compute_nuisance( u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);
      log_jac_u += fn_MVP_compute_nuisance_log_jac_u( u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);
      
      u_array  =  u_vec_chunk.reshaped(chunk_size, n_tests);
      y_sign.array() =  y_chunk.array() + (y_chunk.array() - 1.0) ;
      y_m_y_sign_x_u.array() =  y_chunk.array() - (y_sign.array() * u_array.array());
      ////-----------------------------------------------
      

      {
        //// START of c loop
        for (int c = 0; c < n_class; c++) {
        
                  prod_container_or_inc_array.setZero(); //// reset to 0
                  
                  //// start of t loop
                  for (int t = 0; t < n_tests; t++) {
          
                          ////-----------------------------------------------
                          Bound_Z[c].col(t).array() =     L_Omega_recip_dbl[c](t, t) * (  -1.0*( LT_a_mat_dbl(c, t) + prod_container_or_inc_array.array()   )  ) ;
                          ////-----------------------------------------------
                          
                          ////-----------------------------------------------
                          //// compute/update important log-lik quantities for GHK-MVP
                          fn_MVP_compute_lp_GHK_cols(t, 
                                                     Bound_U_Phi_Bound_Z[c], // computing this
                                                     Phi_Z[c], // computing this
                                                     Z_std_norm[c], // computing this
                                                     prob[c],        // computing this                               
                                                     y1_log_prob, // computing this
                                                     Bound_Z[c],  
                                                     y_chunk,
                                                     u_array,
                                                     Model_args_as_cpp_struct);  
                          ////-----------------------------------------------
          
                          ////-----------------------------------------------
                          if (t < n_tests - 1) {
                            auto L_Omega_row = L_Omega_dbl[c].row(t + 1);
                            prod_container_or_inc_array = Z_std_norm[c].leftCols(t + 1)  *   L_Omega_row.head(t+1).transpose();
                          }
                          ////-----------------------------------------------
          
                  }  //// end of t loop
          
                  ////-----------------------------------------------
                  if (n_class > 1) { // if latent class
                    rowwise_sum = y1_log_prob.rowwise().sum();
                    rowwise_sum.array() += log_prev_dbl(0, c);
                    lp_array.col(c) = rowwise_sum;
                  } else {
                    rowwise_sum = y1_log_prob.rowwise().sum();
                    lp_array.col(0) =     rowwise_sum;
                  }
                  ////-----------------------------------------------
          
        } //// end of c loop
      
      }
      
      ////-----------------------------------------------  
      if (n_class > 1) {
        
        log_sum_exp_general(lp_array,
                            vect_type_exp,
                            vect_type_log,
                            log_sum_result,
                            container_max_logs);
        const int index_start = 1 + n_params + chunk_size_orig * chunk_counter;
        out_mat.segment(index_start, chunk_size) = log_sum_result;
        
      } else {
        
        out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size)  = lp_array.col(0);
        
      }
      ////-----------------------------------------------  
      

      ////-----------------------------------------------
      log_lik_chunk = out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size);
      prob_n  =  fn_EIGEN_double(log_lik_chunk, "exp",  vect_type_exp);
      prob_n_recip  =  stan::math::inv(prob_n); // this CANNOT be a temporary otherwise it created a "dangling reference", since prob_n is a temporary!!
      ////-----------------------------------------------

      /////////////////  ------------------------- compute grad  ---------------------------------------------------------------------------------
      for (int c = 0; c < n_class; c++) {
        
                  ////-----------------------------------------------
                  prob_recip = stan::math::inv(prob[c]);  
                  ////-----------------------------------------------
                  
                  ////-----------------------------------------------
                  //// compute/update important log-lik quantities for GHK-MVP
                  for (int t = 0; t < n_tests; t++) {
                    
                    fn_MVP_compute_phi_Z_recip_cols(   t,
                                                       phi_Z_recip, // computing this
                                                       Phi_Z[c], Z_std_norm[c], Model_args_as_cpp_struct);
                    
                    fn_MVP_compute_phi_Bound_Z_cols(   t,
                                                       phi_Bound_Z, // computing this
                                                       Bound_U_Phi_Bound_Z[c], Bound_Z[c], Model_args_as_cpp_struct);
                  }
                  ////-----------------------------------------------
                  
                  if (grad_option != "none") {  
                    
                        fn_MVP_grad_prep(       prob[c],
                                                y_sign,
                                                y_m_y_sign_x_u,
                                                L_Omega_recip_dbl[c],
                                                prev_dbl(0, c),
                                                prob_n_recip,
                                                phi_Z_recip,
                                                phi_Bound_Z,
                                                prob_recip,
                                                prob_rowwise_prod_temp, 
                                                prob_recip_rowwise_prod_temp,
                                                prob_rowwise_prod_temp_all,
                                                common_grad_term_1,
                                                y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                Model_args_as_cpp_struct) ;
                    
                  }

          ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of nuisance parameters / u's (manual)
          if ( (grad_option == "us_only")  || (grad_option == "all") ) {
            
            u_grad_array_CM_chunk_block =  u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests);

            fn_MVP_compute_nuisance_grad_v2(      u_grad_array_CM_chunk_block,
                                                  phi_Z_recip,
                                                  common_grad_term_1,
                                                  L_Omega_dbl[c],
                                                  prob[c],
                                                  prob_recip,
                                                  prob_rowwise_prod_temp,
                                                  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                  y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                  z_grad_term,
                                                  grad_prob,
                                                  prod_container_or_inc_array,
                                                  derivs_chain_container_vec,
                                                  Model_args_as_cpp_struct);

            u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests).array() += u_grad_array_CM_chunk_block.array();
            
            const int start_index = 1 + (chunk_size_orig * n_tests * chunk_counter);
            const int length = chunk_size * n_tests;
            
            if (c == n_class - 1) {
              
              //// update output vector once all u_grad computations are done
              out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();
              
              //// account for unconstrained -> constrained transformations and Jacobian adjustments
              fn_MVP_nuisance_first_deriv(  du_wrt_duu_chunk,
                                            u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);
              
              fn_MVP_nuisance_deriv_of_log_det_J(    d_J_wrt_duu_chunk,
                                                     u_vec_chunk, u_unc_vec_chunk, du_wrt_duu_chunk, Model_args_as_cpp_struct);
              
              out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
              out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();
              
            }

          }

          /////////////////////////////////////////////////////////////////////////// Grad of intercepts / coefficients (beta's)
          if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only") ) {

            fn_MVP_compute_coefficients_grad_v3(      c,
                                                      beta_grad_array[c],
                                                      chunk_counter,
                                                      n_covariates_max,
                                                      common_grad_term_1,
                                                      L_Omega_dbl[c],
                                                      prob[c],
                                                      prob_recip,
                                                      prob_rowwise_prod_temp,
                                                      y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                      y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                      z_grad_term,
                                                      grad_prob,
                                                      prod_container_or_inc_array,
                                                      derivs_chain_container_vec,
                                                      true,  ///   compute_final_scalar_grad,
                                                      Model_args_as_cpp_struct);

          }
          
          ////////////////////////////////////////////////////////////////////////////////////////////////// Grad of LT_b's (correlation parameters)
          if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only") ) {
              
              fn_LC_LT_compute_bs_grad_v1(   grad_pi_wrt_b_raw,
                                             deriv_Bound_Z_x_L[c],
                                             c,
                                             Jacobian_d_L_Sigma_wrt_b_3d_arrays_dbl[c],
                                             common_grad_term_1,
                                             L_Omega_dbl[c], 
                                             prob[c],
                                             prob_recip,
                                             Bound_Z[c],
                                             Z_std_norm[c],
                                             phi_Bound_Z,
                                             phi_Z_recip,
                                             y_sign,
                                             y_m_y_sign_x_u,
                                             prob_rowwise_prod_temp,
                                             grad_bound_z,
                                             grad_Phi_bound_z,
                                             z_grad_term,
                                             grad_prob,
                                             prod_container_or_inc_array,
                                             derivs_chain_container_vec,
                                             true,  ///   compute_final_scalar_grad,
                                             Model_args_as_cpp_struct);
            

          }

          if (n_class > 1) { /// prevelance only estimated for latent class models
            
                if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only" ) ) {
                  
                      rowwise_prod = prob[c].rowwise().prod();
                      rowwise_prod.array() = prob_n_recip.array() * rowwise_prod.array();
                      double prev_grad = rowwise_prod.sum();
                      prev_grad_vec(c)  +=  prev_grad;
                      
                }
            
          }


      }

    }


    ////////////////////////  --------------------------------------------------------------------------
    for (int c = 0; c < n_class; c++) {
      prev_unconstrained_grad_vec(c)  =   prev_grad_vec(c) * deriv_p_wrt_pu_dbl ;
    }
    prev_unconstrained_grad_vec(0) = prev_unconstrained_grad_vec(1) - prev_unconstrained_grad_vec(0) - 2 * tanh_u_prev_dbl[1];
    prev_unconstrained_grad_vec_out(0) = prev_unconstrained_grad_vec(0);
    
    ////////////////////////  --------------------------------------------------------------------------
    log_prob_out +=  out_mat.tail(N).sum();  ////  log_lik
    log_prob_out +=  log_jac_u;
    if (exclude_priors == false)  log_prob_out += prior_densities;
    log_prob_out +=  log_det_J_main ; // log_jac_p_dbl;

    {
      int i = 0;
      for (int c = 0; c < n_class; c++ ) {
        for (int t = 0; t < n_tests; t++) {
          //// for (int k = 0; k <  n_covariates_per_outcome_vec(c, t); k++) {
            beta_grad_vec(i) = beta_grad_array[c](0, t);
            i += 1;
          //// }
        }
      }
      
     }
    
  ////  }
    
    const Eigen::Matrix<double, -1, 1>  LT_b_grad_vec_nd =  (grad_pi_wrt_b_raw.row(0).transpose().array() * LT_b_mat_dbl.row(0).transpose().array()).matrix() ; //     ( deriv_log_pi_wrt_L_Omega[0].asDiagonal().diagonal().array() * bs_nd_dbl.array()  ).matrix()  ; //  Jacobian_d_L_Sigma_wrt_b_matrix[0].transpose() * deriv_log_pi_wrt_L_Omega_vec_nd;
    const Eigen::Matrix<double, -1, 1>  LT_b_grad_vec_d =   (grad_pi_wrt_b_raw.row(1).transpose().array() * LT_b_mat_dbl.row(1).transpose().array()).matrix() ; //    ( deriv_log_pi_wrt_L_Omega[1].asDiagonal().diagonal().array() * bs_d_dbl.array()  ).matrix()  ; //   Jacobian_d_L_Sigma_wrt_b_matrix[1].transpose()  * deriv_log_pi_wrt_L_Omega_vec_d;
    
    Eigen::Matrix<double, -1, 1>   LT_b_grad_vec(n_b_LT);
    LT_b_grad_vec.head(n_tests)              = LT_b_grad_vec_nd ;
    LT_b_grad_vec.segment(n_tests, n_tests)  = LT_b_grad_vec_d;
    
   
    {   ////////////////////////////  outputs // add log grad and sign stuff';///////////////
        out_mat(0) =  log_prob_out;
        out_mat.segment(1 + n_us, n_b_LT)  += LT_b_grad_vec ;
        out_mat.segment(1 + n_us + n_b_LT, n_a_LT) += beta_grad_vec;
        out_mat(1 + n_us + n_b_LT + n_a_LT) += prev_unconstrained_grad_vec_out(0) ;  
    }
    
 
  }
    
    
    // ////////////////////////  --------------------------------------------------------------------------
    // //// add derivative of normal prior density to beta / coeffs gradient
    // {
    //   int i = n_us + n_corrs + 1; /// + 1 because first element is the log_prob !!
    //   for (int c = 0; c < n_class; c++) {
    //     for (int t = 0; t < n_tests; t++) {
    //       for (int k = 0; k <  n_covariates_per_outcome_vec(c, t); k++) {
    //         if (exclude_priors == false) {
    //           out_mat(i) += - ((beta_dbl_array[c](k, t) - prior_coeffs_mean[c](k, t)) / prior_coeffs_sd[c](k, t) ) * (1.0 / prior_coeffs_sd[c](k, t) ) ;
    //           i += 1;
    //         }
    //       }
    //     }
    //   }
    // }
    // ////////////////////////  --------------------------------------------------------------------------


    }
   
  // int LT_cnt_2 = 0;
  // for (int c = 0; c < n_class; ++c) {
  //   for (int t = 0; t < n_tests; ++t) {
  //     if (LT_known_bs_indicator(c, t) == 1) {
  //       out_mat(1 + n_us + LT_cnt_2) = 0;
  //     }
  //     LT_cnt_2 += 1;
  //   }
  // }
  // // 
  
}











  

// Internal function using Eigen::Ref as inputs for matrices
void     fn_lp_grad_LT_LC_NoLog_MD_and_AD_InPlace(     Eigen::Matrix<double, -1, 1> &&out_mat_R_val,
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
  
   
   fn_lp_grad_LT_LC_NoLog_MD_and_AD_InPlace_process( out_mat_ref,
                                                     theta_main_vec_ref,
                                                     theta_us_vec_ref,
                                                     y_ref,
                                                     grad_option, 
                                                     Model_args_as_cpp_struct);
  
  
}








// Internal function using Eigen::Ref as inputs for matrices
void     fn_lp_grad_LT_LC_NoLog_MD_and_AD_InPlace(    Eigen::Matrix<double, -1, 1> &out_mat_ref,
                                                      const Eigen::Matrix<double, -1, 1> &theta_main_vec_ref,
                                                      const Eigen::Matrix<double, -1, 1> &theta_us_vec_ref,
                                                      const Eigen::Matrix<int, -1, -1> &y_ref,
                                                      const std::string &grad_option,
                                                      const Model_fn_args_struct &Model_args_as_cpp_struct
                                                                    
                                                                    
                                                                    
                                                                    
) {
  
  
  fn_lp_grad_LT_LC_NoLog_MD_and_AD_InPlace_process(  out_mat_ref,
                                                     theta_main_vec_ref,
                                                     theta_us_vec_ref,
                                                     y_ref,
                                                     grad_option,
                                                     Model_args_as_cpp_struct); 
  
  
}





// Internal function using Eigen::Ref as inputs for matrices
template <typename MatrixType>
void     fn_lp_grad_LT_LC_NoLog_MD_and_AD_InPlace(    Eigen::Ref<Eigen::Block<MatrixType, -1, 1>>  &out_mat_ref,
                                                      const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_main_vec_ref,
                                                      const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_us_vec_ref,
                                                      const Eigen::Matrix<int, -1, -1> &y_ref,
                                                      const std::string &grad_option,
                                                      const Model_fn_args_struct &Model_args_as_cpp_struct 
                                                                    
                                                                    
                                                                    
                                                                    
) {
  
  
  fn_lp_grad_LT_LC_NoLog_MD_and_AD_InPlace_process(  out_mat_ref,
                                                                 theta_main_vec_ref,
                                                                 theta_us_vec_ref, 
                                                                 y_ref,
                                                                 grad_option,
                                                                 Model_args_as_cpp_struct);
  
  
}














// Internal function using Eigen::Ref as inputs for matrices
Eigen::Matrix<double, -1, 1>    fn_lp_grad_LT_LC_NoLog_MD_and_AD(    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
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
   
   fn_lp_grad_LT_LC_NoLog_MD_and_AD_InPlace(  out_mat,
                                                          theta_main_vec_ref,
                                                          theta_us_vec_ref,
                                                          y_ref,
                                                          grad_option,
                                                          Model_args_as_cpp_struct);
   
  return out_mat;
   
}


 












