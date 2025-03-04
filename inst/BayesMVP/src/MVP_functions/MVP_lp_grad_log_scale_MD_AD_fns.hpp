
#pragma once


 
#include <Eigen/Dense>
 



#include <unsupported/Eigen/SpecialFunctions>







 

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// This model ccan be either the "standard" MVP model or the latent class MVP model (w/ 2 classes) for analysis of test accuracy data. 
inline  void                             fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process(    Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat ,
                                                                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                                            const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                                            const std::string grad_option,
                                                                                                            const Model_fn_args_struct &Model_args_as_cpp_struct
) {
  

  out_mat.setZero();
  
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
  const double a_times_3 = 3.0 * a;
  const double s = 1.0 / 1.702;
  const double Inf = std::numeric_limits<double>::infinity();

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
  
  ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, desired_n_chunks);
  
  int chunk_size = chunk_size_info.chunk_size;
  int chunk_size_orig = chunk_size_info.chunk_size_orig;
  int normal_chunk_size = chunk_size_info.normal_chunk_size;
  int last_chunk_size = chunk_size_info.last_chunk_size;
  int n_total_chunks = chunk_size_info.n_total_chunks;
  int n_full_chunks = chunk_size_info.n_full_chunks;
  
  //////////////  -----------------------------------------------------------------------------------------------------------------------------------------------------------
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

  //// prev
  double u_prev_diseased = theta_main_vec_ref(n_params_main - 1);

  Eigen::Matrix<double, -1, 1 >  target_AD_grad(n_corrs);
  double grad_prev_AD = 0.0;

  double target_AD_double = 0.0;
  
  int dim_choose_2 = n_tests * (n_tests - 1) * 0.5 ;
  std::vector<Eigen::Matrix<double, -1, -1 > > deriv_L_wrt_unc_full = vec_of_mats_double(dim_choose_2 + n_tests, dim_choose_2, n_class);
  std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_double = vec_of_mats_double(n_tests, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_recip_double = L_Omega_double;
  std::vector<Eigen::Matrix<double, -1, -1 > > log_abs_L_Omega_double = L_Omega_double;
  std::vector<Eigen::Matrix<double, -1, -1 > > sign_L_Omega_double = L_Omega_double;

  double log_jac_p_double = 0.0;
 
  {     ////////////////////////// local AD block

    stan::math::start_nested();  ////////////////////////
    
    stan::math::var target_AD = 0.0;

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
      ////log_abs_L_Omega_double[c] =    log_abs_L_Omega_double[c].array().min(700.0).max(-700.0);
    }

    target_AD_double = target_AD.val();
    
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
    double prior_densities_corrs = target_AD_double;
    prior_densities = prior_densities_coeffs + prior_densities_corrs ;     // total prior densities and Jacobian adjustments
  }
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////// likelihood
  double log_prob_out = 0.0;
  
  Eigen::Matrix<double, -1, -1 >  log_prev = stan::math::log(prev);
  
  //// define unconstrained nuisance parameter vec 
  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec = theta_us_vec_ref; 
  
  /////////////////////////////////////////////////
  Eigen::Matrix<double , -1, 1>   beta_grad_vec   =  Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);  //
  std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array =  vec_of_mats<double>(n_covariates_max, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array =  vec_of_mats<double>(n_tests, n_tests, n_class);
  Eigen::Matrix<double, -1, 1 > L_Omega_grad_vec(n_corrs + (2 * n_tests));
  Eigen::Matrix<double, -1, 1 > U_Omega_grad_vec(n_corrs);
  Eigen::Matrix<double, -1, 1>  prev_unconstrained_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(n_class); //
  Eigen::Matrix<double, -1, 1>  prev_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(n_class); //
  Eigen::Matrix<double, -1, 1>  prev_unconstrained_grad_vec_out = Eigen::Matrix<double, -1, 1>::Zero(n_class - 1); //
  /////////////////////////////////////////////////
  
  {
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<Eigen::Matrix<double, -1, -1>>   Z_std_norm =  vec_of_mats<double>(chunk_size, n_tests, n_class);
    std::vector<Eigen::Matrix<double, -1, -1>>   Bound_Z =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>   Bound_U_Phi_Bound_Z =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>   Phi_Z =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>   phi_Bound_Z =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>   y1_log_prob =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>   prob =  Z_std_norm ;
    std::vector<Eigen::Matrix<double, -1, -1>>   phi_Z_recip =  Z_std_norm ;
    ///////////////////////////////////////////////
    std::vector<Eigen::Matrix<double, -1, -1>>     log_phi_Z_recip =  vec_of_mats<double>(chunk_size, n_tests, n_class);
    std::vector<Eigen::Matrix<double, -1, -1>>     log_phi_Bound_Z =  vec_of_mats<double>(chunk_size, n_tests, n_class);
    std::vector<Eigen::Matrix<double, -1, -1>>     log_Z_std_norm =   vec_of_mats<double>(chunk_size, n_tests, n_class);
    std::vector<Eigen::Matrix<double, -1, -1>>     log_abs_Bound_Z =  vec_of_mats<double>(chunk_size, n_tests, n_class);
    std::vector<Eigen::Matrix<double, -1, -1>>     sign_Bound_Z =     vec_of_mats<double>(chunk_size, n_tests, n_class);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> y_chunk =        Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1> u_array =        Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1> y_sign_chunk =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1> y_m_y_sign_x_u = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1>  inc_array  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk   =           Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1> log_abs_u_grad_array_CM_chunk   =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1> log_sum_result =           Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
    Eigen::Matrix<double, -1, 1> log_sum_abs_result =       Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
    Eigen::Matrix<double, -1, 1> sign_result =              Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
    Eigen::Matrix<double, -1, 1> container_max_logs =       Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
    Eigen::Matrix<double, -1, 1> container_sum_exp_signed = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1> u_unc_vec_chunk =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests); 
    Eigen::Matrix<double, -1, 1> u_vec_chunk =        Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests); 
    Eigen::Matrix<double, -1, 1> du_wrt_duu_chunk =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests); 
    Eigen::Matrix<double, -1, 1> d_J_wrt_duu_chunk =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests); 
    ///////////////////////////////////////////////
    double log_jac_u = 0.0;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1>     log_common_grad_term_1   =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0);
    Eigen::Matrix<double, -1, -1>     log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip   =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0) ;
    Eigen::Matrix<double, -1, -1>     log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip   =   Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0) ;
    Eigen::Matrix<double, -1, -1>     sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip   =  Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests) ;
    Eigen::Matrix<double, -1, -1>     sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip   =   Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests) ;
    Eigen::Matrix<double, -1, -1>     log_prob_rowwise_prod_temp   =   Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0) ;
    Eigen::Matrix<double, -1, -1>     log_prob_recip_rowwise_prod_temp   =   Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0) ;
    Eigen::Matrix<double, -1, -1>     log_abs_grad_prob =    Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0);
    Eigen::Matrix<double, -1, -1>     log_abs_z_grad_term =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0);
    Eigen::Matrix<double, -1, -1>     sign_grad_prob =    Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     sign_z_grad_term =  Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1>     log_abs_prod_container_or_inc_array_comp  =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests, -700.0);
    Eigen::Matrix<double, -1, -1>     sign_prod_container_or_inc_array_comp  =     Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1>     log_abs_derivs_chain_container_vec_comp =    Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests, -700.0);
    Eigen::Matrix<double, -1, -1>     sign_derivs_chain_container_vec_comp =       Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1>      log_abs_prod_container_or_inc_array  =  Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
    Eigen::Matrix<double, -1, 1>      sign_prod_container_or_inc_array  =     Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
    Eigen::Matrix<double, -1, 1>      log_abs_derivs_chain_container_vec  =   Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
    Eigen::Matrix<double, -1, 1>      sign_derivs_chain_container_vec  =      Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
    Eigen::Matrix<double, -1, 1>      log_prob_rowwise_prod_temp_all  =       Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1>  log_abs_prev_grad_array_col_for_each_n =  Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
    Eigen::Matrix<double, -1, 1>  sign_prev_grad_array_col_for_each_n =     Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1>  log_abs_a  =       Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
    Eigen::Matrix<double, -1, 1>  sign_a =           Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
    Eigen::Matrix<double, -1, 1>  log_abs_b =        Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
    Eigen::Matrix<double, -1, 1>  sign_b =           Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
    Eigen::Matrix<double, -1, 1>  sign_sum_result =  Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
    Eigen::Matrix<double, -1, -1> log_terms =        Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests, -700.0);
    Eigen::Matrix<double, -1, -1> sign_terms =       Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
    Eigen::Matrix<double, -1, 1>  final_log_sum =    Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
    Eigen::Matrix<double, -1, 1>  final_sign =       Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
    ///////////////////////////////////////////////
    Eigen::VectorXi overflow_mask(chunk_size);
    Eigen::VectorXi underflow_mask(chunk_size);
    Eigen::VectorXi OK_mask(chunk_size);
    /////////////////////////////////////////////  ////-----------------------------------------------
    std::vector<std::vector<std::vector<int>>> problem_index_array(n_class);
    std::vector<std::vector<int>> n_problem_array(n_class);
    for (int c = 0; c < n_class; c++) {
        problem_index_array[c].resize(n_tests); // initialise
        n_problem_array[c].resize(n_tests);     // initialise
        for (int t = 0; t < n_tests; t++) {
          n_problem_array[c][t] = 0; // initialise
        }
    }
    /////////////////////////////////////////////  ////-----------------------------------------------
    std::vector<Eigen::Matrix<double, -1, -1>>   Omega_grad_array_for_each_n =     vec_of_mats<double>(chunk_size, n_tests, n_tests);
    std::vector<Eigen::Matrix<double, -1, -1>>   sign_Omega_grad_array_for_each_n =     Omega_grad_array_for_each_n;
    std::vector<Eigen::Matrix<double, -1, -1>>   log_abs_Omega_grad_array_for_each_n =  Omega_grad_array_for_each_n;
    std::vector<Eigen::Matrix<double, -1, -1>>   beta_grad_array_for_each_n =      vec_of_mats<double>(chunk_size, n_tests, n_covariates_max);
    std::vector<Eigen::Matrix<double, -1, -1>>   sign_beta_grad_array_for_each_n =      beta_grad_array_for_each_n;
    std::vector<Eigen::Matrix<double, -1, -1>>   log_abs_beta_grad_array_for_each_n =   beta_grad_array_for_each_n;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1>  rowwise_log_sum = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    Eigen::Matrix<double, -1, 1>  rowwise_prod =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    Eigen::Matrix<double, -1, 1>  rowwise_sum =     Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    ///////////////////////////////////////////////
    
    { // start of big local block
      
      Eigen::Matrix<double, -1, -1>    lp_array  =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_class); 

   for (int nc = 0; nc < n_total_chunks; nc++) { // Note: if remainder, then n_total_chunks =  n_full_chunks + 1 and then nc goes from 0 -> n_total_chunks - 1 = n_full_chunks

        int chunk_counter = nc;
        
        if ((chunk_counter == n_full_chunks) && (n_chunks > 1) && (last_chunk_size > 0)) { // Last chunk (remainder - don't use AVX / SIMD for this)

                          chunk_size = last_chunk_size;  //// update chunk_size

                          //// use either Loop (i.e. double fn's) or Stan's vectorisation for the remainder (i.e. last) chunk, regardless of input
                          vect_type = "Stan";
                          vect_type_exp =  "Stan";
                          vect_type_log =  "Stan";
                          vect_type_lse =  "Stan";
                          vect_type_tanh = "Stan";
                          vect_type_Phi =  "Stan";
                          vect_type_log_Phi = "Stan";
                          vect_type_inv_Phi = "Stan";
                          vect_type_inv_Phi_approx_from_logit_prob = "Stan";

                          //// vectors
                          inc_array.resize(last_chunk_size);
                          log_sum_result.resize(last_chunk_size);
                          log_sum_abs_result.resize(last_chunk_size);
                          container_max_logs.resize(last_chunk_size);
                          container_sum_exp_signed.resize(last_chunk_size);
                          sign_result.resize(last_chunk_size);

                          u_unc_vec_chunk.resize(last_chunk_size * n_tests);
                          u_vec_chunk.resize(last_chunk_size * n_tests);
                          du_wrt_duu_chunk.resize(last_chunk_size * n_tests);
                          d_J_wrt_duu_chunk.resize(last_chunk_size * n_tests);

                          log_abs_a.resize(last_chunk_size);
                          sign_a.resize(last_chunk_size);
                          log_abs_b.resize(last_chunk_size);
                          sign_b.resize(last_chunk_size);
                          sign_sum_result.resize(last_chunk_size);
                          final_log_sum.resize(last_chunk_size);
                          final_sign.resize(last_chunk_size);

                          log_abs_prev_grad_array_col_for_each_n.resize(last_chunk_size);
                          sign_prev_grad_array_col_for_each_n.resize(last_chunk_size);

                          log_abs_prod_container_or_inc_array.resize(last_chunk_size);
                          sign_prod_container_or_inc_array.resize(last_chunk_size);
                          log_abs_derivs_chain_container_vec.resize(last_chunk_size);
                          sign_derivs_chain_container_vec.resize(last_chunk_size);
                          log_prob_rowwise_prod_temp_all.resize(last_chunk_size);

                          overflow_mask.resize(last_chunk_size);
                          underflow_mask.resize(last_chunk_size);
                          OK_mask.resize(last_chunk_size);

                          //// matrices
                          u_grad_array_CM_chunk.resize(last_chunk_size, n_tests); u_grad_array_CM_chunk.setZero();
                          log_abs_u_grad_array_CM_chunk.resize(last_chunk_size, n_tests); log_abs_u_grad_array_CM_chunk.setZero();
                          lp_array.resize(last_chunk_size, n_class); lp_array.setZero();
                          log_common_grad_term_1.resize(last_chunk_size, n_tests); log_common_grad_term_1.setConstant(-700.0);

                          y_chunk.resize(last_chunk_size, n_tests); y_chunk.setZero();
                          u_array.resize(last_chunk_size, n_tests); u_array.setZero();
                          y_sign_chunk.resize(last_chunk_size, n_tests); y_sign_chunk.setZero();
                          y_m_y_sign_x_u.resize(last_chunk_size, n_tests); y_m_y_sign_x_u.setZero();

                          log_terms.resize(last_chunk_size, n_tests);
                          sign_terms.resize(last_chunk_size, n_tests);

                          log_abs_prod_container_or_inc_array_comp.resize(last_chunk_size, n_tests);
                          sign_prod_container_or_inc_array_comp.resize(last_chunk_size, n_tests);
                          log_abs_derivs_chain_container_vec_comp.resize(last_chunk_size, n_tests);
                          sign_derivs_chain_container_vec_comp.resize(last_chunk_size, n_tests);

                          log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
                          log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
                          sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
                          sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
                          log_prob_recip_rowwise_prod_temp.resize(last_chunk_size, n_tests);
                          log_abs_grad_prob.resize(last_chunk_size, n_tests);
                          log_abs_z_grad_term.resize(last_chunk_size, n_tests);
                          sign_grad_prob.resize(last_chunk_size, n_tests);
                          sign_z_grad_term.resize(last_chunk_size, n_tests);

                          //// matrix arrays
                          for (int c = 0; c < n_class; c++) {
                                Z_std_norm[c].resize(last_chunk_size, n_tests);
                                log_Z_std_norm[c].resize(last_chunk_size, n_tests);
                                Bound_Z[c].resize(last_chunk_size, n_tests);
                                Bound_U_Phi_Bound_Z[c].resize(last_chunk_size, n_tests);
                                Phi_Z[c].resize(last_chunk_size, n_tests);
                                phi_Bound_Z[c].resize(last_chunk_size, n_tests);
                                y1_log_prob[c].resize(last_chunk_size, n_tests);
                                prob[c].resize(last_chunk_size, n_tests);
                                phi_Z_recip[c].resize(last_chunk_size, n_tests);
                                log_phi_Z_recip[c].resize(last_chunk_size, n_tests);
                                log_phi_Bound_Z[c].resize(last_chunk_size, n_tests);
                                log_abs_Bound_Z[c].resize(last_chunk_size, n_tests);
                                sign_Bound_Z[c].resize(last_chunk_size, n_tests);
                          }

                          for (int i = 0; i < Omega_grad_array_for_each_n.size(); i++) {
                            Omega_grad_array_for_each_n[i].resize(last_chunk_size, n_tests);
                            sign_Omega_grad_array_for_each_n[i].resize(last_chunk_size, n_tests);
                            log_abs_Omega_grad_array_for_each_n[i].resize(last_chunk_size, n_tests);
                          }
                          for (int i = 0; i < beta_grad_array_for_each_n.size(); i++) {
                            beta_grad_array_for_each_n[i].resize(last_chunk_size, n_tests);
                            sign_beta_grad_array_for_each_n[i].resize(last_chunk_size, n_tests);
                            log_abs_beta_grad_array_for_each_n[i].resize(last_chunk_size, n_tests);
                          }

                            rowwise_log_sum.resize(last_chunk_size);
                            rowwise_prod.resize(last_chunk_size);
                            rowwise_sum.resize(last_chunk_size);

        }
        
        { ////-----------------------------------------------
            u_grad_array_CM_chunk.setZero() ; //// reset between chunks as re-using same container
            log_abs_u_grad_array_CM_chunk.setConstant(-700);  // reset between chunks as re-using same container

            y_chunk = y_ref.middleRows( chunk_size_orig * chunk_counter, chunk_size).cast<double>();

            //// Nuisance parameter transformation step
            u_unc_vec_chunk = u_unc_vec.segment( chunk_size_orig * n_tests * chunk_counter, chunk_size * n_tests);

            fn_MVP_compute_nuisance( u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);
            log_jac_u +=    fn_MVP_compute_nuisance_log_jac_u(   u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);

            u_array  =  u_vec_chunk.reshaped(chunk_size, n_tests);
            y_sign_chunk.array() =      y_chunk.array() + (y_chunk.array() - 1.0) ;
            y_m_y_sign_x_u.array() =  y_chunk.array() - (y_sign_chunk.array() * u_array.array());
        } ////-----------------------------------------------
        
        
        {
          // START of c loop
          for (int c = 0; c < n_class; c++) {

                  inc_array.setZero();  //// needs to be reset to 0
                
                  for (int t = 0; t < n_tests; t++) {   //// start of t loop
      
                    if (n_covariates_max > 1) {
                      
                            const Eigen::Matrix<double, -1, 1>    Xbeta_given_class_c_col_t = X[c][t].block(chunk_size_orig * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(c, t)).cast<double>()  * beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t));
                            Bound_Z[c].col(t) =     L_Omega_recip_double[c](t, t) * (  -1.0*( Xbeta_given_class_c_col_t + inc_array  )  ) ;
                            sign_Bound_Z[c].col(t) =   stan::math::sign(Bound_Z[c].col(t));
                            log_abs_Bound_Z[c].col(t) =    fn_EIGEN_double( stan::math::abs(Bound_Z[c].col(t)), "log", vect_type_log);
                      
                    } else {  //// intercept-only
                      
                            Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * ( -1.0*( beta_double_array[c](0, t) + inc_array.array() ) ) ;

                            { ////-----------------------------------------------
                                Eigen::Matrix<double, -1, 1> Bound_Z_col_t = Bound_Z[c].col(t);
                                Eigen::Matrix<double, -1, 1> Bound_Z_col_t_sign = stan::math::sign(Bound_Z_col_t);
                                Eigen::Matrix<double, -1, 1> Bound_Z_col_t_abs =  stan::math::abs(Bound_Z_col_t);
                                ////
                                sign_Bound_Z[c].col(t) =   Bound_Z_col_t_sign;
                                log_abs_Bound_Z[c].col(t) =    fn_EIGEN_double( Bound_Z_col_t_abs, "log", vect_type_log);
                            } ////-----------------------------------------------
                            
                    }
      
                    //// create masks
                    { ////-----------------------------------------------
                        Eigen::Matrix<double, -1, 1> y_chunk_col_t_dbl = y_chunk.col(t);
                        Eigen::Matrix<double, -1, 1> Bound_Z_col_t = Bound_Z[c].col(t);
                        overflow_mask.array() =   ( (Bound_Z_col_t.array() > overflow_threshold)  && (y_chunk_col_t_dbl.array() == 1.0) ).cast<int>()  ;
                        underflow_mask.array() =  ( (Bound_Z_col_t.array() < underflow_threshold) && (y_chunk_col_t_dbl.array() == 0.0) ).cast<int>()  ;
                        OK_mask.array() = ( (overflow_mask.array() == 0) && (underflow_mask.array() == 0) ).cast<int>() ;
                    } ////-----------------------------------------------

                    //// counts
                    const int n_overflows =   overflow_mask.sum();
                    const int n_underflows =  underflow_mask.sum();
                    const int n_problem = n_overflows + n_underflows;
                    const int n_OK = OK_mask.sum();

                    std::vector<int> over_index(n_overflows, 0);
                    std::vector<int> under_index(n_underflows, 0);
                    std::vector<int> problem_index(n_problem, 0);

                    int counter_over  = 0;
                    int counter_under  = 0;
                    
                   for (int n = 0; n < chunk_size; ++n) {

                             if  (overflow_mask(n) == 1) {
                                   over_index[counter_over] = n;
                                   counter_over += 1;
                             } else if  (underflow_mask(n) == 1)   {
                                   under_index[counter_under] = n;
                                   counter_under += 1;
                             } else {
                                   // OK_index[counter_ok] = n;
                                   // counter_ok += 1;
                             }

                   }
                   
                   {
                             int counter = 0;

                             for (int i = 0; i < n_overflows; ++i) {
                                 problem_index[counter] = over_index[i];
                                 counter += 1;
                             }

                             for (int i = 0; i < n_underflows; ++i) {
                                 problem_index[counter] = under_index[i];
                                 counter += 1;
                             }
                   }
                   
                   //// fill the index arrays
                   n_problem_array[c][t] = n_problem;
                   problem_index_array[c][t].resize(n_problem);
                   problem_index_array[c][t] = problem_index;

                   //// compute/update important log-lik quantities for GHK-MVP
                   fn_MVP_compute_lp_GHK_cols(t,
                                              Bound_U_Phi_Bound_Z[c], // computing this
                                              Phi_Z[c], // computing this
                                              Z_std_norm[c], // computing this
                                              prob[c],        // computing this
                                              y1_log_prob[c], // computing this
                                              Bound_Z[c],
                                              y_chunk,
                                              u_array,
                                              Model_args_as_cpp_struct);
                   
                   //// compute/update important grad quantities for GHK-MVP
                   fn_MVP_compute_phi_Z_recip_cols(    t,
                                                       phi_Z_recip[c], // computing this
                                                       Phi_Z[c], Z_std_norm[c], Model_args_as_cpp_struct);

                   fn_MVP_compute_phi_Bound_Z_cols(     t,
                                                        phi_Bound_Z[c], // computing this
                                                        Bound_U_Phi_Bound_Z[c], Bound_Z[c], Model_args_as_cpp_struct);
                                
                   //// compute log-scale quantities (for grad)
                   { ////-----------------------------------------------
                       Eigen::Matrix<double, -1, -1> temp_mat =       Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
                       Eigen::Matrix<double, -1, 1>  temp_col_t =     Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
                       Eigen::Matrix<double, -1, 1>  temp_col_t_abs = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);

                       temp_mat = phi_Bound_Z[c];
                       temp_col_t = temp_mat.col(t);
                       temp_col_t_abs = stan::math::abs(temp_col_t);
                       log_phi_Bound_Z[c].col(t) =   fn_EIGEN_double( temp_col_t_abs,  "log", vect_type_log);

                       temp_mat = phi_Z_recip[c];
                       temp_col_t = temp_mat.col(t);
                       temp_col_t_abs = stan::math::abs(temp_col_t);
                       log_phi_Z_recip[c].col(t) =   fn_EIGEN_double( temp_col_t_abs,  "log", vect_type_log);

                       temp_mat = Z_std_norm[c];
                       temp_col_t = temp_mat.col(t);
                       temp_col_t_abs = stan::math::abs(temp_col_t);
                       log_Z_std_norm[c].col(t)  =   fn_EIGEN_double( temp_col_t_abs,  "log", vect_type_log);
                   } ////-----------------------------------------------
              
            if (n_OK == chunk_size)  {
                        //// carry on as normal as (likely) no * problematic * overflows/underflows
            }  else if (n_OK < chunk_size)  {
              
                        //----------------------------------------------- currently testing this
                        if (n_underflows > 0) { //// underflow (w/ y == 0)

                                    const std::vector<int> index = under_index;
                                    const int index_size = index.size();

                                    fn_MVP_compute_lp_GHK_cols_log_scale_underflow(t,
                                                                                   index,
                                                                                   Bound_U_Phi_Bound_Z[c], // computing this
                                                                                   Phi_Z[c],  // computing this
                                                                                   Z_std_norm[c],  // computing this
                                                                                   log_Z_std_norm[c],  // computing this
                                                                                   prob[c],  // computing this
                                                                                   y1_log_prob[c], // computing this
                                                                                   log_phi_Bound_Z[c], // computing this
                                                                                   log_phi_Z_recip[c], // computing this
                                                                                   Bound_Z[c],
                                                                                   u_array,
                                                                                   Model_args_as_cpp_struct);

                                    phi_Bound_Z[c](index, t) =  fn_EIGEN_double( log_phi_Bound_Z[c](index, t),  "exp", vect_type_exp); // not needed if comp. grad on log-scale
                                    phi_Z_recip[c](index, t) =  fn_EIGEN_double( log_phi_Z_recip[c](index, t),  "exp", vect_type_exp); // not needed if comp. grad on log-scale

                        }
                        //----------------------------------------------- currently testing this
                       
                        //-----------------------------------------------
                        if (n_overflows > 0) { //// overflow (w/ y == 1)

                                  const std::vector<int> index = over_index;
                                  const int index_size = index.size();

                                  fn_MVP_compute_lp_GHK_cols_log_scale_overflow( t,
                                                                                 n_overflows,
                                                                                 index,
                                                                                 Bound_U_Phi_Bound_Z[c],  // computing this
                                                                                 Phi_Z[c],  // computing this
                                                                                 Z_std_norm[c],  // computing this
                                                                                 log_Z_std_norm[c],  // computing this
                                                                                 prob[c],  // computing this
                                                                                 y1_log_prob[c],  // computing this
                                                                                 log_phi_Bound_Z[c],  // computing this
                                                                                 log_phi_Z_recip[c],  // computing this
                                                                                 Bound_Z[c],
                                                                                 u_array,
                                                                                 Model_args_as_cpp_struct);

                                  phi_Bound_Z[c](index, t) =  fn_EIGEN_double( log_phi_Bound_Z[c](index, t),  "exp", vect_type_exp);  // not needed if comp. grad on log-scale
                                  phi_Z_recip[c](index, t) =  fn_EIGEN_double( log_phi_Z_recip[c](index, t),  "exp", vect_type_exp);  // not needed if comp. grad on log-scale

                        }
                        //----------------------------------------------- 

              }  ///// end of "if overflow or underflow" block
            
              if (t < n_tests - 1) { 
                
                      //// -----------------------------------------------
                      Eigen::Matrix<double, 1, -1> L_Omega_row = L_Omega_double[c].row(t + 1);
                      inc_array = Z_std_norm[c].leftCols(t + 1) * L_Omega_row.head(t + 1).transpose();
                      //// -----------------------------------------------

              }
              
            }  //// end of t loop
                  
                //// -----------------------------------------------  
                if (n_class > 1) {  //// if latent class
                    rowwise_sum = y1_log_prob[c].rowwise().sum();
                    rowwise_sum.array() += log_prev(0, c);
                    lp_array.col(c) = rowwise_sum;
                } else {
                    rowwise_sum = y1_log_prob[c].rowwise().sum();
                    lp_array.col(0) =     rowwise_sum;
                }
                //// -----------------------------------------------  
            
          }   //// end of c loop
         
        }  //// end of local block
  
        if (n_class > 1) {  /// if latent class 
          
              //// -----------------------------------------------   
              log_sum_exp_general(   lp_array,
                                     vect_type_exp,
                                     vect_type_log,
                                     log_sum_result,
                                     container_max_logs);
              const int index_start = 1 + n_params + chunk_size_orig * chunk_counter;
              out_mat.segment(index_start, chunk_size) = log_sum_result;
              //// -----------------------------------------------   
          
        } else {
          
              const int index_start = 1 + n_params + chunk_size_orig * chunk_counter;
              out_mat.segment(index_start, chunk_size) = lp_array.col(0);
          
        }
        
          Eigen::Matrix<double, -1, 1> log_prob_n_recip =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
          { //// -----------------------------------------------  
            Eigen::Matrix<double, -1, 1> log_lik = out_mat.tail(N);
            Eigen::Matrix<double, -1, 1> log_lik_segment = log_lik.segment(chunk_size_orig * chunk_counter, chunk_size);
            Eigen::Matrix<double, -1, 1> prob_n  =  fn_EIGEN_double( log_lik_segment, "exp",  vect_type_exp);
            Eigen::Matrix<double, -1, 1> prob_n_recip = stan::math::inv(prob_n);
            log_prob_n_recip = fn_EIGEN_double( prob_n_recip, "log", vect_type_log);
          } //// -----------------------------------------------  
        
        const bool compute_final_scalar_grad = false;
 
  /////////////////////////////////////////////////
  ///////////////// ------------------------- compute grad  -----------------------------------------------------------------------------------------------------------------
  for (int c = 0; c < n_class; c++) {
    
          ////-----------------------------------------------  
          for (int i = 0; i < beta_grad_array_for_each_n.size(); i++) {
            beta_grad_array_for_each_n[i].setZero();
            sign_beta_grad_array_for_each_n[i].setOnes();
            log_abs_beta_grad_array_for_each_n[i].setConstant(-700.0);
          }
          for (int i = 0; i < Omega_grad_array_for_each_n.size(); i++) {
            Omega_grad_array_for_each_n[i].setZero();
            sign_Omega_grad_array_for_each_n[i].setOnes();
            log_abs_Omega_grad_array_for_each_n[i].setConstant(-700.0);
          }

          Eigen::Matrix<double, -1, -1> y1_log_prob_recip = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> sign_Z_std_norm =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> prob_recip =        Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> common_grad_term_1 =                                                         Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> prob_rowwise_prod_temp =                                                     Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip =                        Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          ////-----------------------------------------------  

          { ////-----------------------------------------------  

              Eigen::Matrix<double, -1, -1> temp_mat =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
              Eigen::Matrix<double, -1, -1> temp_mat_2 = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);

                {
                      temp_mat = y1_log_prob[c];
                      temp_mat_2 = -1.0*temp_mat;
                      y1_log_prob_recip = temp_mat_2;
                }
                {
                      temp_mat = Z_std_norm[c];
                      temp_mat_2 = stan::math::sign(temp_mat);
                      sign_Z_std_norm = temp_mat_2;
                }
                {
                      temp_mat = prob[c];
                      temp_mat_2 = stan::math::inv(temp_mat);
                      prob_recip = temp_mat_2;
                }

          } ////-----------------------------------------------  
     
      if ( (grad_option != "none") || (grad_option == "test") ) {

                    ////-----------------------------------------------  
                    Eigen::Matrix<double, -1, -1> abs_L_Omega_recip_double =     Eigen::Matrix<double, -1, -1>::Zero(n_tests, n_tests);
                    Eigen::Matrix<double, -1, -1> log_abs_L_Omega_recip_double = Eigen::Matrix<double, -1, -1>::Zero(n_tests, n_tests);
                    Eigen::Matrix<double, -1, -1> sign_L_Omega_recip_double =    Eigen::Matrix<double, -1, -1>::Ones(n_tests, n_tests);
                    ////-----------------------------------------------  

                    ////-----------------------------------------------  
                     abs_L_Omega_recip_double =  stan::math::abs(L_Omega_recip_double[c]);    //  std::cout << "After abs" << std::endl;
                     sign_L_Omega_recip_double = stan::math::sign(L_Omega_recip_double[c]);   //  std::cout << "After sign" << std::endl;
                     for (int t = 0; t < n_tests; t++) {
                        log_abs_L_Omega_recip_double(t, t) = stan::math::log(abs_L_Omega_recip_double(t, t));
                     }
                     ////-----------------------------------------------  

                    fn_MVP_grad_prep_log_scale(       log_prob_rowwise_prod_temp,
                                                      log_prob_recip_rowwise_prod_temp,
                                                      log_prob_rowwise_prod_temp_all,
                                                      log_common_grad_term_1,
                                                      log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                      log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                      sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                      sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                      y1_log_prob[c],
                                                      y1_log_prob_recip,
                                                      log_prob_n_recip,
                                                      log_prev(0, c), /// if not latent class, this is a dummy variable
                                                      log_phi_Bound_Z[c],
                                                      log_phi_Z_recip[c],
                                                      log_abs_L_Omega_recip_double,
                                                      sign_L_Omega_recip_double,
                                                      y_sign_chunk,
                                                      y_m_y_sign_x_u,
                                                      Model_args_as_cpp_struct);

                    ////-----------------------------------------------  
                    //// these should all be OK on windows  (not dangling reference)
                    common_grad_term_1 = fn_EIGEN_double(log_common_grad_term_1, "exp", vect_type_exp);
                    prob_rowwise_prod_temp = fn_EIGEN_double(log_prob_rowwise_prod_temp, "exp", vect_type_exp);
                    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip  =  fn_EIGEN_double(log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip, "exp", vect_type_exp).array() *
                                                                            sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.array();
                    y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = fn_EIGEN_double(log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip, "exp", vect_type_exp).array() *
                                                                                                 sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.array();
                    ////-----------------------------------------------  

          }
      
      //// --- up to here OK  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
      
          ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of nuisance parameters / u's (manual)
          if ( (grad_option == "us_only") || (grad_option == "all") ) {

                Eigen::Matrix<double, -1, -1>   u_grad_array_CM_chunk_block = u_grad_array_CM_chunk;

                { /// first compute gradients on the standard (non-log) scale.

                      fn_MVP_compute_nuisance_grad_v2(  u_grad_array_CM_chunk_block,
                                                        phi_Z_recip[c],
                                                        common_grad_term_1,
                                                        L_Omega_double[c],
                                                        prob[c],
                                                        prob_recip,
                                                        prob_rowwise_prod_temp,
                                                        y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                        y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                        log_abs_z_grad_term,
                                                        log_abs_grad_prob,
                                                        log_abs_prod_container_or_inc_array,
                                                        sign_prod_container_or_inc_array,
                                                        Model_args_as_cpp_struct);

                }

                { /// then compute gradients on the LOG-scale, but ONLY where we have underflow or overflow.

                    fn_MVP_compute_nuisance_grad_log_scale(   n_problem_array[c],
                                                              problem_index_array[c],
                                                              log_abs_u_grad_array_CM_chunk,
                                                              u_grad_array_CM_chunk_block,
                                                              L_Omega_double[c],
                                                              log_abs_L_Omega_double[c],
                                                              log_phi_Z_recip[c],
                                                              y1_log_prob[c],
                                                              y1_log_prob_recip,
                                                              log_prob_rowwise_prod_temp,
                                                              log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                              sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                              log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                              sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                              log_common_grad_term_1,
                                                              log_abs_z_grad_term,
                                                              sign_z_grad_term,
                                                              log_abs_grad_prob,
                                                              sign_grad_prob,
                                                              log_abs_prod_container_or_inc_array,
                                                              sign_prod_container_or_inc_array,
                                                              log_sum_result,
                                                              sign_sum_result,
                                                              log_terms,
                                                              sign_terms,
                                                              log_abs_a,
                                                              log_abs_b,
                                                              sign_a,
                                                              sign_b,
                                                              container_max_logs,
                                                              container_sum_exp_signed,
                                                              Model_args_as_cpp_struct);

                  }

                  //// update u_grad_array_CM_chunk once standard-scale and log-scale grad computations are done
                  u_grad_array_CM_chunk.array() += u_grad_array_CM_chunk_block.array();
                
                  const int start_index = 1 + (chunk_size_orig * n_tests * chunk_counter);
                  const int length = chunk_size * n_tests;

                  if (c == n_class - 1) {

                            //// update output vector once all u_grad computations are done
                            out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();

                            //// account for unconstrained -> constrained transformations and Jacobian adjustments
                            fn_MVP_nuisance_first_deriv( du_wrt_duu_chunk,
                                                         u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct); 

                            fn_MVP_nuisance_deriv_of_log_det_J(    d_J_wrt_duu_chunk,
                                                                   u_vec_chunk, u_unc_vec_chunk, du_wrt_duu_chunk, Model_args_as_cpp_struct);
                             
                            out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
                            out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();

                  }

                  sign_z_grad_term.setOnes();
                  sign_grad_prob.setOnes();
                  sign_prod_container_or_inc_array.setOnes();
                  sign_sum_result.setOnes();
                  sign_terms.setOnes();
                  sign_a.setOnes();
                  sign_b.setOnes();

                  log_abs_z_grad_term.setConstant(-700.0);
                  log_abs_grad_prob.setConstant(-700.0);
                  log_abs_prod_container_or_inc_array.setConstant(-700.0);
                  log_sum_result.setConstant(-700.0);
                  log_terms.setConstant(-700.0);
                  log_abs_a.setConstant(-700.0);
                  log_abs_b.setConstant(-700.0);
                  container_max_logs.setConstant(-700.0);

          }

          
          ///////////////////////////////////////////////////////////////////////////// Grad of intercepts / coefficients (beta's)
          if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only") ) {

            if (n_covariates_max > 1) {  //// log-scale grad for coefficients not (yet) implemented for > 1 covariate (e.g., standard-MVP)

                  fn_MVP_compute_coefficients_grad_v2(   c,
                                                         beta_grad_array[c],
                                                         beta_grad_array_for_each_n,
                                                         chunk_counter,
                                                         n_covariates_max,
                                                         common_grad_term_1,
                                                         L_Omega_double[c],
                                                         prob[c],
                                                         prob_recip,
                                                         prob_rowwise_prod_temp,
                                                         y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                         y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                         log_abs_grad_prob,
                                                         sign_grad_prob,
                                                         log_abs_prod_container_or_inc_array,
                                                         sign_prod_container_or_inc_array,
                                                         true, /// compute_final_scalar_grad
                                                         Model_args_as_cpp_struct);

            } else {

            {   // compute (some or all) of grads on log-scale

               { /// first compute gradients on the standard (non-log) scale.

                 log_abs_grad_prob.setZero();
                 sign_grad_prob.setZero();
                 log_abs_prod_container_or_inc_array.setZero();
                 sign_prod_container_or_inc_array.setZero();

                 fn_MVP_compute_coefficients_grad_v2(    c,
                                                         beta_grad_array[c],
                                                         beta_grad_array_for_each_n,
                                                         chunk_counter,
                                                         n_covariates_max,
                                                         common_grad_term_1,
                                                         L_Omega_double[c],
                                                         prob[c],
                                                         prob_recip,
                                                         prob_rowwise_prod_temp,
                                                         y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                         y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                         log_abs_grad_prob,
                                                         sign_grad_prob,
                                                         log_abs_prod_container_or_inc_array,
                                                         sign_prod_container_or_inc_array,
                                                         false, /// compute_final_scalar_grad
                                                         Model_args_as_cpp_struct);

               }

               { // then compute gradients on the LOG-scale, but ONLY where we have underflow or overflow.

                  //// extract previous signs and log_abs values
                  for (int i = 0; i <  beta_grad_array_for_each_n.size();  i++) {
                      auto temp_array = beta_grad_array_for_each_n[i];
                      auto temp_array_sign = stan::math::sign(temp_array);
                      sign_beta_grad_array_for_each_n[i] =  temp_array_sign;
                      auto temp_array_abs = stan::math::abs(temp_array);
                      log_abs_beta_grad_array_for_each_n[i] = fn_EIGEN_double( temp_array_abs, "log", vect_type_log);
                      for (int t = 0; t < n_tests; t++) {
                        sign_beta_grad_array_for_each_n[i].col(t)(problem_index_array[c][t]).setOnes();
                        log_abs_beta_grad_array_for_each_n[i].col(t)(problem_index_array[c][t]).setConstant(-700.0);
                      }
                  }

                  fn_MVP_compute_coefficients_grad_log_scale( n_problem_array[c],
                                                              problem_index_array[c],
                                                              beta_grad_array[c],
                                                              sign_beta_grad_array_for_each_n,
                                                              log_abs_beta_grad_array_for_each_n,
                                                              L_Omega_double[c],
                                                              log_abs_L_Omega_double[c],
                                                              log_phi_Z_recip[c],
                                                              y1_log_prob[c],
                                                              log_prob_rowwise_prod_temp,
                                                              log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                              sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                              log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                              sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                              log_common_grad_term_1,
                                                              log_abs_z_grad_term,
                                                              sign_z_grad_term,
                                                              log_abs_grad_prob,
                                                              sign_grad_prob,
                                                              log_abs_prod_container_or_inc_array,
                                                              sign_prod_container_or_inc_array,
                                                              log_abs_prod_container_or_inc_array_comp,
                                                              sign_prod_container_or_inc_array_comp,
                                                              log_sum_result,
                                                              sign_sum_result,
                                                              log_terms,
                                                              sign_terms,
                                                              container_max_logs,
                                                              container_sum_exp_signed,
                                                              Model_args_as_cpp_struct);

                 for (int t = 0; t < n_tests; t++) {
                   for (int k = 0; k <  beta_grad_array_for_each_n.size(); k++) {
                       LogSumVecSingedResult log_sum_vec_signed_struct = log_sum_vec_signed_v1(log_abs_beta_grad_array_for_each_n[k].col(t),  // entire array as carrying fwd the non-log-scale grad previously computed
                                                                                               sign_beta_grad_array_for_each_n[k].col(t), // entire array as carrying fwd the non-log-scale grad previously computed
                                                                                               vect_type);

                       beta_grad_array[c](k, t) +=   stan::math::exp(log_sum_vec_signed_struct.log_sum) * log_sum_vec_signed_struct.sign;
                   }
                 }

                }

              }

            }

            sign_z_grad_term.setOnes();
            sign_grad_prob.setOnes();
            sign_prod_container_or_inc_array.setOnes();
            sign_sum_result.setOnes();
            sign_terms.setOnes();
            sign_a.setOnes();
            sign_b.setOnes();
            sign_prod_container_or_inc_array_comp.setOnes();

            log_abs_z_grad_term.setConstant(-700.0);
            log_abs_grad_prob.setConstant(-700.0);
            log_abs_prod_container_or_inc_array.setConstant(-700.0);
            log_sum_result.setConstant(-700.0);
            log_terms.setConstant(-700.0);
            log_abs_a.setConstant(-700.0);
            log_abs_b.setConstant(-700.0);
            container_max_logs.setConstant(-700.0);
            log_abs_prod_container_or_inc_array_comp.setConstant(-700.0);

          }

          ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of L_Omega ('s)
          if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only") ) {

                { /// first compute gradients on the standard (non-log) scale.

                          log_abs_z_grad_term.setZero();
                          log_abs_grad_prob.setZero();
                          log_abs_prod_container_or_inc_array.setZero();
                          sign_prod_container_or_inc_array.setZero();

                          fn_MVP_compute_L_Omega_grad_v2(       U_Omega_grad_array[c],
                                                                Omega_grad_array_for_each_n,
                                                                common_grad_term_1,
                                                                L_Omega_double[c],
                                                                prob[c],
                                                                prob_recip,
                                                                Bound_Z[c],
                                                                Z_std_norm[c],
                                                                prob_rowwise_prod_temp,
                                                                y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                                y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                                log_abs_z_grad_term,
                                                                log_abs_grad_prob,
                                                                log_abs_prod_container_or_inc_array,
                                                                sign_prod_container_or_inc_array,
                                                                false, /// compute_final_scalar_grad
                                                                Model_args_as_cpp_struct);

                 }

                 { // then compute gradients on the LOG-scale, but ONLY where we have underflow or overflow.

                      //// extract previous signs and log_abs values
                      for (int t1 = 0; t1 < n_tests  ; t1++ ) {
                        
                          auto temp_array = Omega_grad_array_for_each_n[t1];
                          auto temp_array_sign = stan::math::sign(temp_array);
                          sign_Omega_grad_array_for_each_n[t1] = temp_array_sign;
                          auto temp_array_abs = stan::math::abs(temp_array);
                          log_abs_Omega_grad_array_for_each_n[t1] = fn_EIGEN_double(temp_array_abs, "log", vect_type_log);
                          
                            for (int t2 = 0; t2 <  t1 + 1; t2++ ) {
                              sign_Omega_grad_array_for_each_n[t1].col(t2)(problem_index_array[c][t1]).setOnes();
                              log_abs_Omega_grad_array_for_each_n[t1].col(t2)(problem_index_array[c][t1]).setConstant(-700.0);
                            }
                            
                      }

                      fn_MVP_compute_L_Omega_grad_log_scale(      n_problem_array[c],
                                                                  problem_index_array[c],
                                                                  U_Omega_grad_array[c],
                                                                  sign_Omega_grad_array_for_each_n,
                                                                  log_abs_Omega_grad_array_for_each_n,
                                                                  log_abs_Bound_Z[c], // not in other grads
                                                                  sign_Bound_Z[c], // not in other grads
                                                                  log_Z_std_norm[c], // not in other grads
                                                                  sign_Z_std_norm, // not in other grads
                                                                  L_Omega_double[c],
                                                                  log_abs_L_Omega_double[c],
                                                                  log_phi_Z_recip[c],
                                                                  y1_log_prob[c],
                                                                  log_prob_rowwise_prod_temp,
                                                                  log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                                  sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                                  log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                                  sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                                  log_common_grad_term_1,
                                                                  log_abs_z_grad_term,
                                                                  sign_z_grad_term,
                                                                  log_abs_grad_prob,
                                                                  sign_grad_prob,
                                                                  log_abs_prod_container_or_inc_array,
                                                                  sign_prod_container_or_inc_array,
                                                                  log_abs_prod_container_or_inc_array_comp,
                                                                  sign_prod_container_or_inc_array_comp,
                                                                  log_abs_derivs_chain_container_vec_comp,
                                                                  sign_derivs_chain_container_vec_comp,
                                                                  log_sum_result,
                                                                  sign_sum_result,
                                                                  log_terms,
                                                                  sign_terms,
                                                                  log_abs_a,
                                                                  log_abs_b,
                                                                  sign_a,
                                                                  sign_b,
                                                                  container_max_logs,
                                                                  container_sum_exp_signed,
                                                                  Model_args_as_cpp_struct);

                   for (int t1 = 0; t1 < n_tests  ; t1++ ) {
                      for (int t2 = 0; t2 <  t1 + 1; t2++ ) {
                           LogSumVecSingedResult log_sum_vec_signed_struct = log_sum_vec_signed_v1(log_abs_Omega_grad_array_for_each_n[t1].col(t2),  // entire array as carrying fwd the non-log-scale grad previously computed
                                                                                                   sign_Omega_grad_array_for_each_n[t1].col(t2), // entire array as carrying fwd the non-log-scale grad previously computed
                                                                                                   vect_type);

                           U_Omega_grad_array[c](t1, t2) +=   stan::math::exp(log_sum_vec_signed_struct.log_sum) * log_sum_vec_signed_struct.sign;
                     }
                   }

                         sign_z_grad_term.setOnes();
                         sign_grad_prob.setOnes();
                         sign_prod_container_or_inc_array.setOnes();
                         sign_sum_result.setOnes();
                         sign_terms.setOnes();
                         sign_a.setOnes();
                         sign_b.setOnes();
                         sign_prod_container_or_inc_array_comp.setOnes();
                         sign_derivs_chain_container_vec_comp.setOnes();

                         log_abs_z_grad_term.setConstant(-700.0);
                         log_abs_grad_prob.setConstant(-700.0);
                         log_abs_prod_container_or_inc_array.setConstant(-700.0);
                         log_sum_result.setConstant(-700.0);
                         log_terms.setConstant(-700.0);
                         log_abs_a.setConstant(-700.0);
                         log_abs_b.setConstant(-700.0);
                         container_max_logs.setConstant(-700.0);
                         log_abs_prod_container_or_inc_array_comp.setConstant(-700.0);
                         log_abs_derivs_chain_container_vec_comp.setConstant(-700.0);

            }

          }

          if (n_class > 1) { /// prevelance only estimated for latent class models

              if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only" ) ) {

                         {
                             rowwise_log_sum = y1_log_prob[c].rowwise().sum();
                             log_abs_prev_grad_array_col_for_each_n   =    log_prob_n_recip + rowwise_log_sum ;
                             sign_prev_grad_array_col_for_each_n.setOnes(); //// just a vector of +1's since probs are always positive

                             // final scalar grad using log-sum-exp
                             LogSumVecSingedResult log_sum_vec_signed_struct = log_sum_vec_signed_v1( log_abs_prev_grad_array_col_for_each_n,
                                                                                                      sign_prev_grad_array_col_for_each_n,
                                                                                                      vect_type);

                             prev_grad_vec(c)  +=   stan::math::exp(log_sum_vec_signed_struct.log_sum) * log_sum_vec_signed_struct.sign;
                         }

              }

          }

        }

      } /// end of chunk block
  
    }

    //////////////////////// gradients for latent class membership probabilitie(s) (i.e. disease prevalence)
    if (n_class > 1) {
        for (int c = 0; c < n_class; c++) {
          prev_unconstrained_grad_vec(c)  =   prev_grad_vec(c) * deriv_p_wrt_pu_double ;
        }
        prev_unconstrained_grad_vec(0) = prev_unconstrained_grad_vec(1) - prev_unconstrained_grad_vec(0) - 2 * tanh_u_prev[1];
        prev_unconstrained_grad_vec_out(0) = prev_unconstrained_grad_vec(0);
    }

    log_prob_out += out_mat.segment(1 + n_params, N).sum();       // log_prob_out += log_lik.sum();
    if (exclude_priors == false)  log_prob_out += prior_densities;
    log_prob_out +=  log_jac_u;
    log_prob_out +=  log_jac_p_double;

    int i = 0; // probs_all_range.prod() cancels out
    for (int c = 0; c < n_class; c++ ) {
      for (int t = 0; t < n_tests; t++) {
        for (int k = 0; k <  n_covariates_per_outcome_vec(c, t); k++) {
          if (exclude_priors == false) {
            beta_grad_array[c](k, t) +=  - ((beta_double_array[c](k, t) - prior_coeffs_mean[c](k, t)) / prior_coeffs_sd[c](k, t) ) * (1.0/prior_coeffs_sd[c](k, t) ) ;     // add normal prior density derivative to gradient
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
    out_mat.segment(1 + n_us, n_corrs).array() += U_Omega_grad_vec.array() ;        //.cast<float>()  ;
    out_mat.segment(1 + n_us + n_corrs, n_covariates_total) = beta_grad_vec ; //.cast<float>() ;
    out_mat(1 + n_us + n_corrs + n_covariates_total) = ((grad_prev_AD +  prev_unconstrained_grad_vec_out(0)));
    
  }

//  return(out_mat);

}



















 
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













 



 
 
 
 
 
 