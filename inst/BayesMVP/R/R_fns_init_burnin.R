




#' init_EHMC_Metric_as_Rcpp_List
#' @keywords internal
#' @export
init_EHMC_Metric_as_Rcpp_List   <- function(  n_params_main,
                                              n_nuisance,
                                              metric_shape_main) {
  
      n_params <- n_params_main + n_nuisance
      try({  
        index_nuisance <- 1:n_nuisance
      })
      
      M_diag_vec <- rep(1, n_params)
      
      M_dense_main <- diag(n_params_main)
      M_inv_dense_main <- M_dense_main
      M_inv_dense_main_chol <- M_dense_main
      
      M_inv_main_vec <- matrix(c(diag(M_inv_dense_main)))
      M_main_vec <- matrix(c(1 / M_inv_main_vec))
      
      M_inv_us_vec <- matrix(1 / M_diag_vec[index_nuisance])
      M_us_vec <- matrix(M_diag_vec[index_nuisance])
      
      
      
      EHMC_Metric_as_Rcpp_List <- list( 
          ### for main params
          M_dense_main = M_dense_main,
          M_inv_dense_main = M_inv_dense_main,
          M_inv_dense_main_chol = M_inv_dense_main_chol,
          M_inv_main_vec = M_inv_main_vec,
          M_main_vec = M_main_vec,
          ### for nuisance
          M_inv_us_vec = M_inv_us_vec, 
          M_us_vec = M_us_vec,
          ### shape of main metric
          metric_shape_main = metric_shape_main)
   

   return(EHMC_Metric_as_Rcpp_List)

}




#' init_EHMC_args_as_Rcpp_List
#' @keywords internal
#' @export
init_EHMC_args_as_Rcpp_List   <- function(diffusion_HMC) {
  
        tau_main <- 1
        tau_main_ii <- 1
        eps_main <- 1
        tau_us <- 1
        tau_us_ii <- 1
        eps_us <- 1

        EHMC_args_as_Rcpp_List <- list(
          ### for main params
          tau_main = tau_main,
          tau_main_ii = tau_main_ii,
          eps_main = eps_main,
          ### for nuisance
          tau_us = tau_us,
          tau_us_ii = tau_us_ii,
          eps_us = eps_us,
          diffusion_HMC = diffusion_HMC)
        
      return(EHMC_args_as_Rcpp_List)

}



#' init_EHMC_burnin_as_Rcpp_List
#' @keywords internal
#' @export
init_EHMC_burnin_as_Rcpp_List   <- function(n_params_main,
                                            n_nuisance,
                                            adapt_delta,
                                            LR_main,
                                            LR_us) {
  
      n_params <- n_params_main + n_nuisance
      index_main <- (1 + n_nuisance):n_params
      try({  
        index_nuisance <- 1:n_nuisance
      })
  
  ##### set ADAM-related params (initialise)
      ### for main params
      adapt_delta_main <- adapt_delta
      LR_main <- LR_main
      eps_m_adam_main <-  1.0
      eps_v_adam_main <-  1.0
      tau_m_adam_main <-  1.0
      tau_v_adam_main <-  1.0
      index_main <- index_main
      M_dense_sqrt <- matrix(c(diag(n_params_main)))
      snaper_m_vec_main  <- matrix(c(rep(1, n_params_main)))
      snaper_w_vec_main  <- matrix(c(rep(0.01, n_params_main)))
      
      
      ### for NUISANCE params
      adapt_delta_us <- adapt_delta
      LR_us <- LR_us
      eps_m_adam_us <-  1.0
      eps_v_adam_us <-  1.0
      tau_m_adam_us <-  1.0
      tau_v_adam_us <-  1.0
      index_nuisance <- index_nuisance
      sqrt_M_us_vec <- matrix(rep(1, n_nuisance))
      snaper_m_vec_us  <- matrix(rep(1, n_nuisance))
      snaper_w_vec_us  <- matrix(rep(0.01, n_nuisance))
      
      
      ### for main params
      adapt_delta_main <- adapt_delta_main
      LR_main <- LR_main
      eps_m_adam_main <- eps_m_adam_main
      eps_v_adam_main <- eps_v_adam_main
      tau_m_adam_main <- tau_m_adam_main
      tau_v_adam_main <- tau_v_adam_main
      index_main <- index_main
      M_dense_sqrt <- M_dense_sqrt
      snaper_m_vec_main <- snaper_m_vec_main
      snaper_w_vec_main <- snaper_w_vec_main
      eigen_max_main <- 0
      eigen_vector_main <- matrix(c(rep(0, n_params_main)))
      
      ### for NUISANCE params
      adapt_delta_us <- adapt_delta_us
      LR_us <- LR_us
      eps_m_adam_us <- eps_m_adam_us
      eps_v_adam_us <- eps_v_adam_us
      tau_m_adam_us <- tau_m_adam_us
      tau_v_adam_us <- tau_v_adam_us
      index_nuisance <- index_nuisance
      M_dense_sqrt <- M_dense_sqrt
      snaper_m_vec_us <- snaper_m_vec_us
      snaper_w_vec_us <- snaper_w_vec_us
      eigen_max_us <- 0 
      eigen_vector_us <- matrix(c(rep(0, n_nuisance)))
      
      # ------------ put in the list to put into C++
      EHMC_burnin_as_Rcpp_List <- list(   ### for main params
                                          adapt_delta_main = adapt_delta_main,
                                          LR_main = LR_main,
                                          eps_m_adam_main = eps_m_adam_main,
                                          eps_v_adam_main = eps_v_adam_main,
                                          tau_m_adam_main = tau_m_adam_main,
                                          tau_v_adam_main = tau_v_adam_main,
                                          eigen_max_main =  eigen_max_main,
                                          index_main = index_main,
                                          M_dense_sqrt = M_dense_sqrt,
                                          snaper_m_vec_main = snaper_m_vec_main,
                                          snaper_w_vec_main = snaper_w_vec_main,
                                          eigen_vector_main = eigen_vector_main,
                                          ### for nuisance
                                          adapt_delta_us = adapt_delta_us,
                                          LR_us = LR_us,
                                          eps_m_adam_us = eps_m_adam_us,
                                          eps_v_adam_us = eps_v_adam_us,
                                          tau_m_adam_us = tau_m_adam_us,
                                          tau_v_adam_us = tau_v_adam_us,
                                          eigen_max_us =  eigen_max_us,
                                          index_us = index_nuisance,
                                          sqrt_M_us_vec = sqrt_M_us_vec,
                                          snaper_m_vec_us = snaper_m_vec_us,
                                          snaper_w_vec_us = snaper_w_vec_us,
                                          eigen_vector_us = eigen_vector_us)
      
      
      return(EHMC_burnin_as_Rcpp_List)
      
}




#' init_and_run_burnin
#' @keywords internal
#' @export
init_and_run_burnin   <- function(  Model_type,
                                    init_object,
                                    Stan_data_list,
                                    # model_args_list,
                                    y,
                                    N,
                                    parallel_method,
                                    ##
                                    manual_tau,
                                    tau_if_manual,
                                    ##
                                    n_chains_burnin,
                                    sample_nuisance,
                                    Phi_type,
                                    n_params_main,
                                    n_nuisance,
                                    seed,
                                    n_burnin,
                                    adapt_delta,
                                    LR_main,
                                    LR_us,
                                    n_adapt,
                                    partitioned_HMC,
                                    diffusion_HMC,
                                    clip_iter,
                                    gap ,
                                    metric_type_main,
                                    metric_shape_main,
                                    metric_type_nuisance,
                                    metric_shape_nuisance,
                                    max_eps_main,
                                    max_eps_us,
                                    max_L,
                                    tau_mult,
                                    ratio_M_us,
                                    ratio_M_main,
                                    interval_width_main,
                                    interval_width_nuisance,
                                    force_autodiff,
                                    force_PartialLog,
                                    multi_attempts,
                                    n_nuisance_to_track,
                                    Model_args_as_Rcpp_List) {
  
  
  # lp_grad_outs <- fn_Rcpp_wrapper_fn_lp_grad( Model_type = "latent_trait",
  #                                             force_autodiff = FALSE,
  #                                             force_PartialLog = FALSE,
  #                                             theta_main_vec = theta_vec[index_main],
  #                                             theta_us_vec = theta_vec[index_us],
  #                                             y = y,
  #                                             grad_option = "all",
  #                                             Model_args_as_Rcpp_List = Model_args_as_Rcpp_List)

  ###  print( c(lp_grad_outs[1], tail(lp_grad_outs, 21)) )

  ## NOTE:        metric_shape_nuisance = "diag" is the only option for nuisance !!
  
  n_params <- n_params_main + n_nuisance
  index_us <- 1:n_nuisance
  index_main <- (n_nuisance + 1):n_nuisance
 
  
  {  # ---------------------------------------------------------------- list for EHMC params / EHMC struct in C++
    
        EHMC_Metric_as_Rcpp_List <- init_EHMC_Metric_as_Rcpp_List(   n_params_main = n_params_main, 
                                                                     n_nuisance = n_nuisance, 
                                                                     metric_shape_main = metric_shape_main)  
    
  }
  
  
  
  { # ----------------------------------------------------------------- list for EHMC params / EHMC struct in C++  #
    
        EHMC_args_as_Rcpp_List <- init_EHMC_args_as_Rcpp_List(diffusion_HMC = diffusion_HMC) 
    
  }

 
  {  # ----------------------------------------------------------------- list for EHMC params / EHMC struct in C++
      
        EHMC_burnin_as_Rcpp_List <- init_EHMC_burnin_as_Rcpp_List( n_params_main = n_params_main,
                                                                   n_nuisance = n_nuisance,
                                                                   adapt_delta = adapt_delta,
                                                                   LR_main = LR_main,
                                                                   LR_us = LR_us)
    
  }
  

  theta_main_vectors_all_chains_input_from_R <- init_object$theta_main_vectors_all_chains_input_from_R
  theta_us_vectors_all_chains_input_from_R <-   init_object$theta_us_vectors_all_chains_input_from_R
  
  # print(theta_main_vectors_all_chains_input_from_R)
  # print(theta_us_vectors_all_chains_input_from_R)
  
  ###  theta_vec_mean <- rowMeans(rbind(theta_us_vectors_all_chains_input_from_R, theta_main_vectors_all_chains_input_from_R))
  
 
 # print(theta_main_vectors_all_chains_input_from_R)
  

  
  EHMC_args_as_Rcpp_List$diffusion_HMC <- diffusion_HMC
  
  

 ##  Model_args_as_Rcpp_List$Model_args_strings[1, 2] <-  Phi_type
  
  results_burnin <-  BayesMVP:::R_fn_EHMC_SNAPER_ADAM_burnin( Model_type = Model_type,
                                                              sample_nuisance = sample_nuisance,
                                                              parallel_method = parallel_method,
                                                              n_params_main = n_params_main,
                                                              n_nuisance = n_nuisance,
                                                              Stan_data_list = Stan_data_list,
                                                              # model_args_list = model_args_list,
                                                              y = y,
                                                              N = N,
                                                              ##
                                                              manual_tau = manual_tau, ##
                                                              tau_if_manual = tau_if_manual, ##
                                                              ##
                                                              n_chains_burnin = n_chains_burnin,
                                                              seed = seed,
                                                              n_burnin = n_burnin,
                                                              LR_main = LR_main,
                                                              LR_us = LR_us,
                                                              n_adapt = n_adapt,
                                                              partitioned_HMC = partitioned_HMC,
                                                              clip_iter = clip_iter,
                                                              gap = gap,
                                                              metric_type_main = metric_type_main,
                                                              metric_shape_main = metric_shape_main,
                                                              metric_type_nuisance = metric_type_nuisance,
                                                              metric_shape_nuisance = metric_shape_nuisance,
                                                              max_eps_main = max_eps_main,
                                                              max_eps_us = max_eps_us,
                                                              max_L = max_L,
                                                              tau_mult = tau_mult,
                                                              ratio_M_us = ratio_M_us,
                                                              ratio_M_main = ratio_M_main,
                                                              interval_width_main = interval_width_main,
                                                              interval_width_nuisance = interval_width_nuisance,
                                                              force_autodiff = force_autodiff,
                                                              force_PartialLog = force_PartialLog,
                                                              multi_attempts = multi_attempts,
                                                              theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                                              theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R,
                                                              n_nuisance_to_track = n_nuisance_to_track,
                                                              Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                              EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                                                              EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                              EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List)
  
  
  ### update Rcpp lists to pass onto sampling / post-burnin function 
  time_burnin <- results_burnin$time_burnin
  EHMC_args_as_Rcpp_List <- results_burnin$EHMC_args_as_Rcpp_List
  EHMC_Metric_as_Rcpp_List <- results_burnin$EHMC_Metric_as_Rcpp_List
  EHMC_burnin_as_Rcpp_List <- results_burnin$EHMC_burnin_as_Rcpp_List
  
  ## for init values for sampling phase
  theta_main_vectors_all_chains_input_from_R <- results_burnin$theta_main_vectors_all_chains_input_from_R
  theta_us_vectors_all_chains_input_from_R <- results_burnin$theta_us_vectors_all_chains_input_from_R
  
  return(list(time_burnin = time_burnin,
              Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
              EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
              EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
              EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
              theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
              theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R))
  
}






















