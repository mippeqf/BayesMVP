R_fn_EHMC_find_initial_eps        <- function( Model_type.,
                                               theta., 
                                                indiactor_mixed_M.,
                                                HMC.,
                                                index_subset.,
                                                index_subset_main_dense.,
                                                y.,
                                                X.,
                                                L.,
                                                eps.,
                                                momentum_0.,
                                                velocity_0.,
                                                testing.,
                                                M_diag_vec.,
                                                M_main_dense.,
                                                M_Chol_main_dense.,
                                                M_inv_dense_main.,
                                                grad_x_initial.,
                                                log_posterior_initial.,
                                                n_us.,
                                                n_params_main.,
                                                other_lp_and_grad_args.
                                                )  {  
  
  
  
  
  # grad_x_initial. <- grad
  # log_posterior_initial. <- log_posterior
  
  grad_x_initial. <- grad_x_initial.
  log_posterior <- log_posterior_initial.
  
  log_tempratio <-  -10000
  
  tempratio <- 1000000
  a <- 1
  
  
  eps  <-  1
  
  
  n_params <- length(theta.)
  momentum_0 <- momentum <-  rnorm(n = n_params, 0,  1)
 
  eps_vec <- c()
  
  index_main <- (n_us.+1):n_params
  index_us <- 1:n_us.
  
  p_jump <-  0 
  
  velocity_0 <- velocity_0. # lkj_cholesky_eta
  
  for (kkkk in 1:25) {
    
                log_posterior <- NA
                
             try({ 
                Rcpp_leapfrog_outs <- BayesMVPv2::Rcpp_fn_sampling_single_iter_burnin( Model_type = Model_type.,
                                                                                       theta_main_array = theta.[index_main],
                                                                                       theta_us_array = theta.[index_us],
                                                                                       y = y.,
                                                                                       X = X.,
                                                                                       other_args = other_lp_and_grad_args.,
                                                                                       L = 1,
                                                                                       eps = eps,
                                                                                       log_posterior = log_posterior_initial.,
                                                                                       M_inv_us_array = (1 / M_diag_vec.[index_us]), 
                                                                                       M_dense_main  = diag(n_params_main.),
                                                                                       M_inv_dense_main = diag(n_params_main.),
                                                                                       M_inv_dense_main_chol  = diag(n_params_main.) )
                
                                                                                                        
                
                log_posterior_initial. =  Rcpp_leapfrog_outs[2, 1] 
                log_posterior =           Rcpp_leapfrog_outs[3, 1] ### Rcpp_leapfrog_outs[[5]]
                
             }, silent = TRUE)
                  
 
                
             
            
                
                if (!(is.na(log_posterior))) {
                  
                  
                          if (p_jump < 0.80) {
                            eps <- 0.5 * eps
                          }
                  
 
                          
             
                        p_jump <-   Rcpp_leapfrog_outs[5, 1] # min(1, exp(  + tempratio))
 
                } else {
             
                            eps <- 0.5 * eps
 
                     
                  
                }
                
                eps_vec[kkkk]  = eps
                
                if (kkkk > 1) {
                  if ( eps_vec[kkkk] ==  eps_vec[kkkk - 1] ) { 
                    break
                  }
                }
                
 
    
  }
  
  
  return(eps)
  
  
  
}
  
  
  
  
  
