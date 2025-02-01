functions {
  
  
   matrix corr_to_chol(real x, int J) {
        matrix[J, J] cor = add_diag(rep_matrix(x, J, J), 1 - x);
        return cholesky_decompose(cor); 
  }
 
 
 
 matrix lb_ub_lp (matrix y,
                  real lb, 
                  real ub,
                  int log_det_J) {
 
      int N = rows(y); 
      int M = cols(y); 
      matrix[N, M] tanh_y; 
   
      tanh_y = tanh(y);
      
      if (log_det_J == 1)  target +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh_y))) ;
      return lb +  (ub - lb) *  0.5 * (1 + tanh_y) ;   
    
  }  
  
   vector lb_ub_lp (vector y, 
                    real lb,
                    real ub,
                    int log_det_J) {
 
      int N = num_elements(y); 
      vector[N] tanh_y; 
   
      tanh_y = tanh(y);
      
      if (log_det_J == 1)  target +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh_y))) ;
      return lb +  (ub - lb) *  0.5 * (1 + tanh_y) ;   
    
  }  
 
  real lb_ub_lp (real y, 
                 real lb, 
                 real ub, 
                 int log_det_J) {
      
        real  tanh_y = tanh(y);
   
        if (log_det_J == 1) target +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh_y))) ;
        return lb +  (ub - lb) *  0.5 * (1 + tanh_y) ; 
    
  } 
  
  
  
   matrix cholesky_corr_constrain_outer_lp( // matrix Omega_raw,
                                           vector col_one_raw, 
                                           vector off_raw,
                                           real lb, 
                                           real ub,
                                           int log_det_J) {
                                             
   // segment(col(Omega_raw, 1), 2, K - 1);
   
    int K = num_elements(col_one_raw) + 1;
    vector[K - 1] z = lb_ub_lp(col_one_raw, lb, ub, log_det_J);
    matrix[K, K] L = diag_matrix(rep_vector(1, K));
    vector[K] D;
    D[1] = 1;
    L[2:K, 1] = z[1:K - 1];
    D[2] = 1 - L[2, 1]^2;
    int cnt = 1;
   
    for (i in 3:K) {
      
       D[i] = 1 - L[i, 1]^2; 
       L[i, 2:i - 1] = rep_row_vector(1 - L[i, 1]^2, i - 2);
       real l_ij_old = L[i, 2];
       
      for (j in 2:i - 1) {
              //  real l_ij_old = L[i, j];
                real b1 = dot_product(L[j, 1:(j - 1)], D[1:j - 1]' .* L[i, 1:(j - 1)]);
                  
                  // how to derive the bounds
                  // we know that the correlation value C is bound by
                  // b1 - Ljj * Lij_old <= C <= b1 + Ljj * Lij_old
                  // Now we want our bounds to be enforced too so
                  // max(lb, b1 - Ljj * Lij_old) <= C <= min(ub, b1 + Ljj * Lij_old)
                  // We have the Lij_new = (C - b1) / Ljj
                  // To get the bounds on Lij_new is
                  // (bound - b1) / Ljj 
                  
                  real low = max({-sqrt(l_ij_old) * D[j], lb - b1});
                  real up = min({sqrt(l_ij_old) * D[j], ub - b1}); 
                  
                  real x = lb_ub_lp(off_raw[cnt], low, up, log_det_J);
                  L[i, j] = x / D[j]; 
        
                  if (log_det_J == 1)  target += -0.5 * log(D[j]);
                  
                   l_ij_old *= 1 - (D[j] * L[i, j]^2) / l_ij_old;
                  
                 // real mul = 1 - (D[j] * L[i, j]^2) / l_ij_old;
                 // L[i, (j + 1):i - 1] *= mul;
                  //D[i] *= mul;
                  cnt += 1;
        }
        
        D[i] = l_ij_old;
        
      }
      
        return diag_post_multiply(L, sqrt(D));
        
  }
  
  
  
   
   // real   Stan_wrapper_lp_fn(     vector theta_main_vec, 
   //                                vector theta_us_vec, 
   //                                matrix y, 
   //                                int  n_chunks_target, 
   //                                real overflow_threshold, // 5
   //                                real underflow_threshold, 
   //                                real prev_prior_a, 
   //                                real prev_prior_b, 
   //                                vector lkj_cholesky_eta, 
   //                                matrix n_covariates_per_outcome_vec,  // 10
   //                                array[] matrix prior_coeffs_mean, 
   //                                array[] matrix prior_coeffs_sd, 
   //                                array[] matrix prior_for_corr_a,  
   //                                array[] matrix prior_for_corr_b, 
   //                                array[] matrix lb_corr,  // 15
   //                                array[] matrix ub_corr,  
   //                                array[] matrix known_values,
   //                                array[] matrix known_values_indicator,
   //                                array[,] matrix X); // 19
                                
                                
                                
////// this is the entire log_posterior fn for LC-MVP / MVP / latent_trait models w/ manual gradients and AVX-512 (or AVX-2 or if not available then using Stan's built-in fns)                                  
real   Stan_wrapper_lp_fn_LC_MVP_var( int Model_type_int, 
                                      int force_autodiff_int, 
                                      int force_PartialLog_int,
                                      int multi_attempts_int,
                                      vector theta_main_vec,  // 5
                                      vector theta_us_vec, 
                                      matrix y, 
                                      int  n_chunks_target, 
                                      real overflow_threshold, 
                                      real underflow_threshold,  // 10
                                      real prev_prior_a, 
                                      real prev_prior_b, 
                                      vector lkj_cholesky_eta, 
                                      matrix n_covariates_per_outcome_vec,   
                                      array[] matrix prior_coeffs_mean,  // 15
                                      array[] matrix prior_coeffs_sd, 
                                      array[] matrix prior_for_corr_a,  
                                      array[] matrix prior_for_corr_b, 
                                      array[] matrix lb_corr,   
                                      array[] matrix ub_corr,   // 20
                                      array[] matrix known_values,
                                      array[] matrix known_values_indicator,
                                      array[,] matrix X); // 27
                                  
                                    
 
 
}

   
    

data {

      int<lower=1> N;
      int<lower=2> n_tests;
      matrix<lower=0>[N, n_tests]   y;  //////// data
      int<lower=2> n_class;
      int<lower=1> n_pops;
      array[N] int pop;
      int n_covariates_max;
      //array[n_class, n_tests] matrix[N, n_covariates_max] X; /////// covariate array (can have  DIFFERENT NUMBERS of covariates for each  outcome - fill rest of array with 999999 if they vary between outcomes)
      array[n_class, n_tests] int n_covs_per_outcome;
      int corr_force_positive;
      array[n_class] matrix[n_tests, n_tests] lb_corr;
      array[n_class] matrix[n_tests, n_tests] ub_corr;
      array[n_class] matrix[n_tests, n_tests] known_values;
      array[n_class] matrix[n_tests, n_tests] known_values_indicator;
      real overflow_threshold;
      real underflow_threshold;
      ///// priors 
      array[n_class, n_covariates_max, n_tests]  real prior_beta_mean;
      array[n_class, n_covariates_max, n_tests]  real<lower=0> prior_beta_sd;
      vector[n_class] lkj_cholesky_eta;
      vector<lower=0>[n_pops] prior_p_alpha;
      vector<lower=0>[n_pops] prior_p_beta;
      ///// other
      int n_params_main;
      int n_chunks_target;
      int<lower=0, upper=(n_tests * (n_tests - 1)) %/% 2> known_num;
      int priors_via_Stan;
      int Model_type_int; ////  1 for MVP, 2 for LC_MVP, and 3 for latent_trait 
      int multi_attempts_int;
      int force_autodiff_int;
      int force_PartialLog_int;
  
}



transformed data {
  
      //  int log_det_J_main_indicator = 0; //// note: we are EXCLUDING log_det_J for main (i.e. corr's and prev) because this is handled in C++
   
        vector[n_class] lkj_cholesky_eta_flat = rep_vector(1, n_class);
        array[n_class] matrix[n_covariates_max, n_tests] prior_beta_mean_flat;
        array[n_class] matrix[n_covariates_max, n_tests] prior_beta_sd_flat;
        vector<lower=0>[n_pops] prior_p_alpha_flat = rep_vector(1, n_pops);
        vector<lower=0>[n_pops] prior_p_beta_flat  = rep_vector(1, n_pops);
        
        int k_choose_2 = (n_tests * (n_tests - 1)) / 2;
        int km1_choose_2 = ((n_tests - 1) * (n_tests - 2)) / 2;
        
        int n_covariates_total_nd =    (sum( (n_covs_per_outcome[1,])));
        int n_covariates_total_d =     (sum( (n_covs_per_outcome[2,])));
        int n_covariates_total =       n_covariates_total_nd + n_covariates_total_d;
        
       /////  int n_params_main = n_class*k_choose_2 + n_covariates_total + (n_class - 1);
        int n_us = n_tests * N;
  
        array[n_class, n_tests] matrix[N, n_covariates_max] X;
  
        matrix[n_class, n_tests] n_covariates_per_outcome_vec = to_matrix(n_covs_per_outcome);
        
        array[n_class] matrix[n_covariates_max, n_tests] prior_beta_mean_as_array_of_mat;
        array[n_class] matrix[n_covariates_max, n_tests] prior_beta_sd_as_array_of_mat;
        
        array[n_class] matrix[n_tests, n_tests] prior_for_corr_a_dummy;
        array[n_class] matrix[n_tests, n_tests] prior_for_corr_b_dummy;
        
        vector[n_us] theta_nuisance_vec_as_data = rep_vector(0.01, n_us);
        vector[n_params_main] theta_main_vec_as_data = rep_vector(0.01, n_params_main);
        
        real<lower=-1, upper=1> lb;  ////// TEMP 
        real<lower=lb, upper=1> ub = 1.0; ////// TEMP 

        if (corr_force_positive == 1)  lb = 0; ////// TEMP 
        else lb = -1.0; ////// TEMP 
        
        
        for (c in 1:n_class) {
          prior_beta_mean_flat[c,,] = rep_matrix(0.0, n_covariates_max, n_tests);
          prior_beta_sd_flat[c,,] =   rep_matrix(1.0, n_covariates_max, n_tests);
        }
        
             
        for (c in 1:n_class) { 
            
             prior_for_corr_a_dummy[c,,] = rep_matrix(0, n_tests, n_tests);
             prior_for_corr_b_dummy[c,,] = rep_matrix(0, n_tests, n_tests);
             
            for (t in 1:n_tests) {
                prior_beta_mean_as_array_of_mat[c,,t] = to_vector(prior_beta_mean[c,,t]);
                prior_beta_sd_as_array_of_mat[c,,t] =   to_vector(prior_beta_sd[c,,t]);
                
                X[c, t] = rep_matrix(1, N, 1);
            }
          
        }
        
                                                  

}


parameters {
  
       //  put nuisance parameters FIRST (NOTE: doesnt have to be on "raw" scale to work as grad is computed w.r.t unconstrained anyway!)
       vector[n_us] theta_nuisance_vec; 
       vector[n_params_main] theta_main_vec;
       
}


 


transformed parameters {

     ///// arrays / matrices / vectors of UNCONSTRAINED parameters, categorised by parameter type (e.g. coeffs, etc)
     array[n_class] matrix[n_tests, n_tests] Omega_unconstrained_var;
     vector[n_covariates_total] beta_vec;
     vector[n_pops*(n_class - 1)] prev_unconstrained_vec;
     ///////
     array[n_class] vector[n_tests - 1] col_one_raw;
     array[n_class] vector[km1_choose_2 - known_num] off_raw;
     array[n_class] matrix[n_tests, n_tests] Omega;
     array[n_class] matrix[n_tests, n_tests] L_Omega;
     array[n_class] matrix[n_covariates_max, n_tests] beta;
     matrix[n_pops, n_class]  prev_unconstrained_matrix;
     matrix<lower=0, upper=1>[n_pops, n_class]  prev_var_matrix;
     /// vector[N] log_lik  = rep_vector(0.0, N);
     real log_posterior = 0.0;



     {
           //// set counter for main params
           int counter = 1;
           
           if (Model_type_int == 2) { ////  if LC_MVP
               //// CORRELATIONS are the first n_class * T * (T - 1) / 2 elements of main parameter vector
               for (c in 1:n_class) {
                 for (t1 in 2:n_tests) {
                   for (t2 in 1:(t1 - 1)) {
                       Omega_unconstrained_var[c, t1, t2] = theta_main_vec[counter];
                       counter += 1;
                   }
                 }
               }
           } else  if (Model_type_int == 3) { ////  if latent_trait
               for (c in 1:n_class) {
                     for (t in 1:n_tests) {
                           counter += 1;
                       }
                     }
           }
           
           ///// COEFFICIENTS are the next set of parameters
           int counter_beta = 1;
           for (c in 1:n_class) {
                 for (t in 1:n_tests) {
                    for (k in 1:n_covs_per_outcome[c, t]) {
                        beta_vec[counter_beta] = theta_main_vec[counter];
                        counter_beta += 1;
                        counter += 1;
                    }
                 }
           }

           ///// PREVELANCE is the last parameter (if model is latent class, otherwise done)
           int n_prevs = n_pops*(n_class - 1);
           for (i in 1:n_prevs) {
               prev_unconstrained_vec[i] = theta_main_vec[counter];
               counter += 1;
           }


     }



     if (n_class > 1) {
         prev_unconstrained_matrix[1, 2] = prev_unconstrained_vec[1];
         prev_unconstrained_matrix[1, 1] = 1.0 - prev_unconstrained_vec[1];
         prev_var_matrix = lb_ub_lp(prev_unconstrained_matrix, 0.0, 1.0, priors_via_Stan);
     }

     for (c in 1:n_class) {

           col_one_raw[c, 1:(n_tests - 1)] = to_vector(Omega_unconstrained_var[c, 2:n_tests, 1]);

           {

               int counter = 1;
               for (t1 in 3:n_tests) {  //// t1 > t2
                 for (t2 in 2:(t1 - 1)) {
                   if (counter <= km1_choose_2) {
                      off_raw[c, counter] = Omega_unconstrained_var[c, t1, t2];
                      counter += 1;
                   }
                 }
               }

           }

     }


    {
            int counter = 1;
            for (c in 1:n_class) {
                      for (t in 1:n_tests) {
                        for (k in 1:n_covs_per_outcome[c, t]) {
                           beta[c, k, t] = beta_vec[counter];
                           counter += 1;
                        }
                      }
                    L_Omega[c, :  ] =   cholesky_corr_constrain_outer_lp( col_one_raw[c, :], off_raw[c, :], lb, ub, priors_via_Stan);
                    Omega[c, :  ] = multiply_lower_tri_self_transpose(L_Omega[c, :]);
            }
      }
      
      
      
              if (priors_via_Stan == 1) {

                   log_posterior =     Stan_wrapper_lp_fn_LC_MVP_var(                    Model_type_int, /// model_type (1 for MVP, 2 for LC_MVP, 3 for latent_trait)
                                                                                  force_autodiff_int,
                                                                                  force_PartialLog_int,
                                                                                  multi_attempts_int, /// multi_attempts 
                                                                                  theta_main_vec,
                                                                                  theta_nuisance_vec,
                                                                                  y,
                                                                                  n_chunks_target,
                                                                                  overflow_threshold,
                                                                                  underflow_threshold,
                                                                                  prior_p_alpha_flat[1],  //// FLAT prior as managing priors via Stan
                                                                                  prior_p_beta_flat[1],  //// FLAT prior as managing priors via Stan
                                                                                  lkj_cholesky_eta_flat, //// FLAT prior as managing priors via Stan
                                                                                  n_covariates_per_outcome_vec,
                                                                                  prior_beta_mean_flat,  //// FLAT prior as managing priors via Stan
                                                                                  prior_beta_sd_flat, //// FLAT prior as managing priors via Stan
                                                                                  prior_for_corr_a_dummy,
                                                                                  prior_for_corr_b_dummy,
                                                                                  lb_corr,
                                                                                  ub_corr,
                                                                                  known_values,
                                                                                  known_values_indicator,
                                                                                  X);
                                                                        
                  // log_lik = tail(lp_and_log_lik, N);
                  // log_posterior = head(lp_and_log_lik, 1)[1];


              } else { ///// if managing priors directly via C++
                
                   log_posterior     =  Stan_wrapper_lp_fn_LC_MVP_var(                     Model_type_int, /// model_type (1 for MVP, 2 for LC_MVP, 3 for latent_trait)
                                                                                    force_autodiff_int,
                                                                                    force_PartialLog_int,
                                                                                    multi_attempts_int, /// multi_attempts  
                                                                                    theta_main_vec,
                                                                                    theta_nuisance_vec,
                                                                                    y,
                                                                                    n_chunks_target,
                                                                                    overflow_threshold, 
                                                                                    underflow_threshold,
                                                                                    prior_p_alpha[1],
                                                                                    prior_p_beta[1],
                                                                                    lkj_cholesky_eta,
                                                                                    n_covariates_per_outcome_vec,  
                                                                                    prior_beta_mean_as_array_of_mat,
                                                                                    prior_beta_sd_as_array_of_mat,
                                                                                    prior_for_corr_a_dummy,
                                                                                    prior_for_corr_b_dummy,
                                                                                    lb_corr,  
                                                                                    ub_corr,
                                                                                    known_values,
                                                                                    known_values_indicator,
                                                                                    X);
                                                                        
                    // log_lik = tail(lp_and_log_lik, N);
                    // log_posterior = head(lp_and_log_lik, 1)[1];
                                
          
               }
               

               



}
 


 


model {
  

   if (priors_via_Stan == 1) {
    
           for (c in 1 : n_class) {
                for (t in 1 : n_tests) {
                     for (k in 1 : n_covs_per_outcome[c, t]) {
                       target +=  normal_lpdf(beta[c, k, t]  | prior_beta_mean_as_array_of_mat[c, k, t], prior_beta_sd_as_array_of_mat[c, k, t]);
                    }
                }
                 target += lkj_corr_cholesky_lpdf(L_Omega[c,,] | lkj_cholesky_eta[c]);
            }
    
    
            for (g in 1 : n_pops) {
              target += beta_lpdf(prev_var_matrix[g] | prior_p_alpha[g], prior_p_beta[g]);
            }

    }

              
              
                 target += log_posterior;


 
                

}



  



generated quantities {

    vector[n_tests] Se_bin;
    vector[n_tests] Sp_bin;
    vector[n_tests] Fp_bin;


     for (c in 1:n_class) {


        for (t in 1:n_tests) { // for binary tests

            if (n_class == 2) { // summary Se and Sp only calculated if n_class = 2 (i.e. the "standard" # of classes for DTA)
                  Se_bin[t]  =        Phi(   beta[2, 1, t]   );
                  Sp_bin[t]  =    1 - Phi(   beta[1, 1, t]   );
                  Fp_bin[t]  =    1 - Sp_bin[t];
            }
      }



  }


}




