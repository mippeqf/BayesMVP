


#' init_inits
#' @keywords internal
#' @export
init_inits    <- function(init_model_outs,
                          init_lists_per_chain,
                          compile,
                          force_recompile,
                          cmdstanr_model_fit_obj,
                          n_chains_burnin,
                          n_params_main,
                          n_nuisance,
                          N,
                          Stan_model_file_path,
                          Stan_data_list,
                          Stan_cpp_user_header,
                          sample_nuisance,
                          ...) {
  
  
  
  ## NOTE:        metric_shape_nuisance = "diag" is the only option for nuisance !!
  
  if (is.null(init_model_outs)) {
    warning("init_model_outs not specified - please create init_model_outs using init_model() and then pass as an argument to init_burnin()")
  }
  
  Model_type <- init_model_outs$Model_type
 
  n_params <- n_params_main + n_nuisance
  
  y <- init_model_outs$y
  
  if (is.null(init_lists_per_chain)) { 
    warning("initial values per chain (init_lists_per_chain) not supplied - using defaults")
  }
  
   
  
  if (Model_type == "latent_trait") { 
    n_tests <- ncol(y)
    n_params_main <- 1 + sum(n_covariates_per_outcome_mat) + 2 * n_tests 
    n_nuisance <- n_tests * N
  } else if (Model_type == "LC_MVP") { 
    n_tests <- ncol(y)
    n_params_main <- 1 + sum(n_covariates_per_outcome_mat) + 2 * choose(n_tests, 2)
    n_nuisance <- n_tests * N
  } else if (Model_type == "MVP") { 
    n_tests <- ncol(y)
    n_params_main <-   sum(n_covariates_per_outcome_mat) + choose(n_tests, 2)
    n_nuisance <- n_tests * N
  } else { 
    
    if (is.null(n_params_main)) {
      warning("n_params_main not specified - will compute from Stan model")
    }

    if (is.null(n_nuisance)) {
      stop("n_nuisance not specified - please specify")
    }
    
  }
  
  
  
  
  
  if (Model_type != "Stan") { 
    if (Model_type == "MVP") {
      n_class <- 1
    } else {
      n_class <- 2
    }
  }

 
 ####  n_chains_burnin <- 8 
  
  ##  ----------------------------   starting values - set defaults if user does not supply --- this is only for NON-Stan models - for Stan models inits are specified the same as they are for Stan
  
  
  if (Model_type == "Stan") { 
    
           dummy_json_file_path <- NULL
           dummy_model_so_file <- NULL
           
            # //////   ----- if Stan model, then all 3 of: 
            # Stan_model_file_path,
            # Stan_data_list, and 
            # init_lists_per_chain need to be user-supplied
            
            bs_model <- init_model_outs$bs_model
            json_file_path <- init_model_outs$json_file_path
         
              
            if (compile == TRUE) {  # re-compile the user-supplied Stan model to extract model methods and make init's  

                      if (is.null(Stan_cpp_user_header)) {
                        
                        mod <- cmdstanr::cmdstan_model(Stan_model_file_path, 
                                                       compile_model_methods = TRUE,
                                                       force_recompile = force_recompile
                        )
                        
                      } else {
                        
                        mod <- cmdstanr::cmdstan_model(Stan_model_file_path, 
                                                       compile_model_methods = TRUE,
                                                       force_recompile = force_recompile,
                                                       user_header =  Stan_cpp_user_header)  
                      }

              
            } else {  ### use the inputted cmdstanr_model_fit_obj to re-initialise the model
              
              # model_fit <- cmdstanr_model_fit_obj
              mod <- cmdstanr_model_fit_obj
              
            }
            
            
            
            
            model_fit <- mod$sample(  data = Stan_data_list,
                                      seed = 123,
                                      chains = n_chains_burnin,
                                      parallel_chains = n_chains_burnin,
                                      iter_warmup = 1,
                                      iter_sampling = 1,
                                      init = init_lists_per_chain,
                                      adapt_delta = 0.10,
                                      max_treedepth = 1)
            
              
              
              cmdstanr_model_out <- model_fit$summary()
              param_names <- cmdstanr_model_out$variable
              
              
              unconstrained_vec_per_chain <- list()
              for (kk in 1:n_chains_burnin) {
                unconstrained_vec_per_chain[[kk]] <- model_fit$unconstrain_variables(init_lists_per_chain[[kk]])
              }
              
              # ?StanModel
              # print(unconstrained_vec_per_chain)
              
              n_params <- length( unconstrained_vec_per_chain[[kk]])
              if (sample_nuisance == TRUE) { 
                n_params_main <- n_params - n_nuisance
              } else { 
                n_params_main <- n_params
              }
              
 
    
 
  } else {
    
            if (Model_type == "LC_MVP") {
          
                  # prior_for_corr_a <- init_model_outs$prior_for_corr_a
                  # prior_for_corr_b <- init_model_outs$prior_for_corr_b
                  
                  X <- init_model_outs$X
                  n_covariates_per_outcome_mat <- init_model_outs$model_args_list$model_args_list$n_covariates_per_outcome_mat
                  n_covariates_max_nd <- max(n_covariates_per_outcome_mat[1, ])
                  n_covariates_max_d <-  max(n_covariates_per_outcome_mat[2, ])  
                  n_covariates_max <- max(n_covariates_max_nd, n_covariates_max_d)
                  X_nd <- X[[1]]
                  X_d <-  X[[2]]
              
                  prior_coeffs_mean_mat <- init_model_outs$model_args_list$model_args_list$prior_coeffs_mean_mat
                  prior_coeffs_sd_mat <- init_model_outs$model_args_list$model_args_list$prior_coeffs_sd_mat
          
                  ub_corr <- init_model_outs$model_args_list$model_args_list$ub_corr
                  lb_corr <- init_model_outs$model_args_list$model_args_list$lb_corr
                  lkj_cholesky_eta <- init_model_outs$model_args_list$model_args_list$lkj_cholesky_eta
                  
                  prev_prior_a <- init_model_outs$model_args_list$model_args_list$prev_prior_a
                  prev_prior_b <- init_model_outs$model_args_list$model_args_list$prev_prior_b
                  
                  known_num <- 0
                  known_values_indicator_list <- init_model_outs$model_args_list$model_args_list$known_values_indicator_list
                  known_values_list <- init_model_outs$model_args_list$model_args_list$known_values_list
                  
                  ## these vars need conversion to different types compatible w/ Stan
                  corr_force_positive <- as.integer(init_model_outs$model_args_list$model_args_list$corr_force_positive)
                  prior_only <-  as.integer(init_model_outs$model_args_list$model_args_list$prior_only)

                  
                  overflow_threshold <- init_model_outs$model_args_list$model_args_list$overflow_threshold
                  underflow_threshold <- init_model_outs$model_args_list$model_args_list$underflow_threshold
 
                  
                  
                  for (c in 1:n_class) {
                    for (t in 1:n_tests) {
                      X[[c]][[t]][1:N, 1:n_covariates_max] <- 1
                    }
                  }
                  
                  n_pops <- 1 
                  
                  print(prior_coeffs_mean_mat)
                  print(prior_coeffs_sd_mat)
                  
                 #  print(X)
                  for (c in 1:n_class) {
                    for (t in 1:n_tests) {
                       print( is.numeric(X[[c]][[t]]))
                    }
                  }
                  
                  Stan_data_list <- list(N = N,
                                         n_tests = n_tests,
                                         y = y,
                                         n_class = 2,
                                         n_pops =  1,  ## multi-pop not supported yet (currently only in Stan version)
                                         pop =  (rep(1, N)),
                                         #####
                                         n_covariates_max_nd = n_covariates_max_nd,
                                         n_covariates_max_d = n_covariates_max_d,
                                         n_covariates_max = n_covariates_max,
                                         # X_nd = X_nd,
                                         # X_d = X_d,
                                         #X = list(X_nd, X_d),
                                         n_covs_per_outcome = n_covariates_per_outcome_mat,
                                         #####
                                         corr_force_positive = corr_force_positive,
                                         known_num = known_num,
                                         # lb_corr = lb_corr,
                                         # ub_corr = ub_corr,
                                         overflow_threshold =  overflow_threshold,
                                         underflow_threshold = underflow_threshold,
                                         #### priors
                                         prior_only = prior_only,
                                         prior_beta_mean = prior_coeffs_mean_mat,
                                         prior_beta_sd = prior_coeffs_sd_mat,
                                         prior_LKJ = lkj_cholesky_eta,
                                         prior_p_alpha =  array(rep(prev_prior_a, n_pops)),
                                         prior_p_beta =  array(rep(prev_prior_b, n_pops)))
 
                  
                  
                  outs_init_bs_model <- init_bs_model(Stan_data_list = Stan_data_list,
                                                      Stan_model_name = "PO_LC_MVP_bin.stan")
                  
                  bs_model <- outs_init_bs_model$bs_model
                  json_file_path <- outs_init_bs_model$json_file_path         
                  Stan_model_file_path <- outs_init_bs_model$Stan_model_file_path  
                  
                  dummy_json_file_path <- json_file_path
                  dummy_model_so_file <- transform_stan_path(Stan_model_file_path)
                  
        } else if (Model_type == "MVP") {
          
                  # prior_for_corr_a <- init_model_outs$prior_for_corr_a
                  # prior_for_corr_b <- init_model_outs$prior_for_corr_b
                  
                  X <- init_model_outs$X
                  n_covariates_per_outcome_mat <- init_model_outs$model_args_list$model_args_list$n_covariates_per_outcome_mat
                  n_covariates_max_nd <- 999999 #max(n_covariates_per_outcome_mat[[1]])
                  n_covariates_max_d <-  999999 #max(n_covariates_per_outcome_mat[[2]])  
                  n_covariates_max <- max(unlist(n_covariates_per_outcome_mat))
                  ##X_nd <- X[[1]]
                 # X_d <-  X[[2]]
                  
                  prior_coeffs_mean_mat <- init_model_outs$model_args_list$model_args_list$prior_coeffs_mean_mat
                  prior_coeffs_sd_mat <- init_model_outs$model_args_list$model_args_list$prior_coeffs_sd_mat
                  
                  ub_corr <- init_model_outs$model_args_list$model_args_list$ub_corr
                  lb_corr <- init_model_outs$model_args_list$model_args_list$lb_corr
                  lkj_cholesky_eta <- init_model_outs$model_args_list$model_args_list$lkj_cholesky_eta
                  
                  prev_prior_a <- init_model_outs$model_args_list$model_args_list$prev_prior_a
                  prev_prior_b <- init_model_outs$model_args_list$model_args_list$prev_prior_b
                  
                  known_num <- 0
                  known_values_indicator_list <- init_model_outs$model_args_list$model_args_list$known_values_indicator_list
                  known_values_list <- init_model_outs$model_args_list$model_args_list$known_values_list
                  
                  ## these vars need conversion to different types compatible w/ Stan
                  corr_force_positive <- as.integer(init_model_outs$model_args_list$model_args_list$corr_force_positive)
                  prior_only <-  as.integer(init_model_outs$model_args_list$model_args_list$prior_only)
                  
                  
                  overflow_threshold <- init_model_outs$model_args_list$model_args_list$overflow_threshold
                  underflow_threshold <- init_model_outs$model_args_list$model_args_list$underflow_threshold
 
                  # for (c in 1:n_class) {
                  #   for (t in 1:n_tests) {
                  #     X[[c]][[t]][1:N, 1:n_covariates_max] <- 1
                  #   }
                  # }
                  
                  n_pops <- 1 
 
                  Stan_data_list <- list(N = N,
                                         n_tests = n_tests,
                                         ## y = y,
                                         #####
                                         n_covariates_max = n_covariates_max,
                                         #X = list(X_nd, X_d),
                                         n_covs_per_outcome = n_covariates_per_outcome_mat[1, ],
                                         #####
                                         corr_force_positive = corr_force_positive,
                                         known_num = known_num,
                                         # lb_corr = lb_corr,
                                         # ub_corr = ub_corr,
                                         overflow_threshold =  overflow_threshold,
                                         underflow_threshold = underflow_threshold,
                                         #### priors
                                         prior_only = prior_only,
                                         prior_beta_mean = prior_coeffs_mean_mat[[1]],
                                         prior_beta_sd = prior_coeffs_sd_mat[[1]],
                                         prior_LKJ =  lkj_cholesky_eta)
                  
                  
                  
                  
                  outs_init_bs_model <- init_bs_model(Stan_data_list = Stan_data_list,
                                                      Stan_model_name = "PO_MVP_bin.stan")
                  
                  bs_model <- outs_init_bs_model$bs_model
                  json_file_path <- outs_init_bs_model$json_file_path         
                  Stan_model_file_path <- outs_init_bs_model$Stan_model_file_path  
 
                  
                  dummy_json_file_path <- json_file_path
                  dummy_model_so_file <- transform_stan_path(Stan_model_file_path)
          
        } else if (Model_type == "latent_trait") {
          
                  overflow_threshold <- init_model_outs$model_args_list$model_args_list$overflow_threshold
                  underflow_threshold <- init_model_outs$model_args_list$model_args_list$underflow_threshold
                  
                  LT_b_priors_shape <- init_model_outs$model_args_list$model_args_list$LT_b_priors_shape
                  LT_b_priors_scale <- init_model_outs$model_args_list$model_args_list$LT_b_priors_scale
                  LT_known_bs_values <- init_model_outs$model_args_list$model_args_list$LT_known_bs_values
                  LT_known_bs_indicator <- init_model_outs$model_args_list$model_args_list$LT_known_bs_indicator
                  
                  prior_coeffs_mean_mat <- init_model_outs$model_args_list$model_args_list$prior_coeffs_mean_mat
                  prior_coeffs_sd_mat <- init_model_outs$model_args_list$model_args_list$prior_coeffs_sd_mat
 
                  prev_prior_a <- init_model_outs$model_args_list$model_args_list$prev_prior_a
                  prev_prior_b <- init_model_outs$model_args_list$model_args_list$prev_prior_b
 
                  ## these vars need conversion to different types compatible w/ Stan
                  corr_force_positive <- as.integer(init_model_outs$model_args_list$model_args_list$corr_force_positive)
                  prior_only <-  as.integer(init_model_outs$model_args_list$model_args_list$prior_only)
  
                  ##  print(Stan_data_list)
                  
                  n_pops <- 1
                  
                  print(LT_b_priors_shape) ; print(LT_b_priors_scale) ; 
                  print(LT_known_bs_values) ;  print(LT_known_bs_indicator) ; 
                  
                  Stan_data_list <- list(N = N,
                                         n_tests = n_tests,
                                         y = y,
                                         n_class = 2,
                                         n_pops =  1,  ## multi-pop not supported yet (currently only in Stan version)
                                         pop =   (rep(1, N)),
                                         corr_force_positive = corr_force_positive,
                                         overflow_threshold =  overflow_threshold,
                                         underflow_threshold = underflow_threshold,
                                         #### priors
                                         prior_only = prior_only,
                                         prior_beta_mean = prior_coeffs_mean_mat,
                                         prior_beta_sd = prior_coeffs_sd_mat,
                                         ####
                                         LT_b_priors_shape = LT_b_priors_shape,
                                         LT_b_priors_scale = LT_b_priors_scale,
                                         LT_known_bs_values = LT_known_bs_values,
                                         LT_known_bs_indicator = LT_known_bs_indicator, 
                                         ####
                                         prior_p_alpha =  array(rep(prev_prior_a, n_pops)),
                                         prior_p_beta =  array(rep(prev_prior_b, n_pops))) 
         
                  
                  outs_init_bs_model <- init_bs_model(Stan_data_list = Stan_data_list,
                                                      Stan_model_name = "PO_latent_trait_bin.stan")
                  
                  bs_model <- outs_init_bs_model$bs_model
                  json_file_path <- outs_init_bs_model$json_file_path         
                  Stan_model_file_path <- outs_init_bs_model$Stan_model_file_path  
                  
                  dummy_json_file_path <- json_file_path
                  dummy_model_so_file <- transform_stan_path(Stan_model_file_path)
       
          
        }
          
          
          # re-compile the user-supplied Stan model to extract model methods and make init's vector and JSON data file 
    

    param_names <- NULL
    

    
    
    if (compile == TRUE) {  # re-compile the user-supplied Stan model to extract model methods and make init's  
      
            if (is.null(Stan_cpp_user_header)) {
              
              mod <- cmdstanr::cmdstan_model(Stan_model_file_path, 
                                             compile_model_methods = TRUE,
                                             force_recompile = force_recompile
              )
              
            } else {
              
              mod <- cmdstanr::cmdstan_model(Stan_model_file_path, 
                                             compile_model_methods = TRUE,
                                             force_recompile = force_recompile,
                                             user_header =  Stan_cpp_user_header)  
            }

      
    } else {  ### use the inputted cmdstanr_model_fit_obj to re-initialise the model
      
      # model_fit <- cmdstanr_model_fit_obj
      mod <- cmdstanr_model_fit_obj
      
    }
    
    
    model_fit <- mod$sample(  data = Stan_data_list,
                              seed = 123,
                              chains = n_chains_burnin,
                              parallel_chains = n_chains_burnin,
                              iter_warmup = 1,
                              iter_sampling = 1,
                              init = init_lists_per_chain,
                              adapt_delta = 0.10,
                              max_treedepth = 1)
    
    
          cmdstanr_model_out <- model_fit$summary()
          param_names <- cmdstanr_model_out$variable
          
          unconstrained_vec_per_chain <- list()
          for (kk in 1:n_chains_burnin) {
            unconstrained_vec_per_chain[[kk]] <- model_fit$unconstrain_variables(init_lists_per_chain[[kk]])
          }
          
          bs_model <- NULL
          bs_model <- init_model_outs$bs_model
    
    
  }
  
  
   model_so_file <- transform_stan_path(Stan_model_file_path)
  
  
 
    ##  --------------------------   End of starting values 
    print(n_chains_burnin)
    print(n_params_main)
    print(n_nuisance)
    
    theta_main_vectors_all_chains_input_from_R  <- array(0, dim = c(n_params_main, n_chains_burnin))
    theta_us_vectors_all_chains_input_from_R    <- array(0, dim = c(n_nuisance, n_chains_burnin))
    
    n_params <- n_nuisance + n_params_main
    index_nuisance <- 1:n_nuisance
    index_main <- (1 + n_nuisance):n_params
  
    for (kk in 1:n_chains_burnin) {
      theta_main_vectors_all_chains_input_from_R[, kk] <-     unconstrained_vec_per_chain[[kk]][index_main]
      theta_us_vectors_all_chains_input_from_R[, kk] <-     unconstrained_vec_per_chain[[kk]][index_nuisance]
    }

  
  
  
  Model_args_as_Rcpp_List <- init_model_outs$Model_args_as_Rcpp_List
  
  
  
  
  return(list(  cmdstanr_model_fit_obj = mod,
                bs_model = bs_model,
                json_file_path = json_file_path,
                model_so_file = model_so_file,
                dummy_json_file_path = dummy_json_file_path,
                dummy_model_so_file = dummy_model_so_file,
                param_names = param_names,
                Stan_data_list = Stan_data_list,
                Stan_model_file_path = Stan_model_file_path,
                Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                unconstrained_vec_per_chain = unconstrained_vec_per_chain,
                theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R, 
                theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R))
  
}




