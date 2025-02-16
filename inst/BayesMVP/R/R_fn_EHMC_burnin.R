




## helper function:
#' is_valid
#' @keywords internal
#' @export
is_valid <- function(x) {
    return( !(is.null(x)) && all(!is.na(x)) && all(!is.nan(x)) && all(is.finite(x)) )
}


## helper function:
#' if_not_NA_or_INF_else
#' @keywords internal
#' @export
if_not_NA_or_INF_else <- function(x, alt_x) { 
  
    success_indicator <- TRUE
    if (is_valid(x) == TRUE) {
      return(x)
    } else { 
      return(alt_x)
    }
  
}


# Burnin - using R fn (which calls Rcpp fn @ each iter)  ------------------------------------------------------------------------------------------------------------------ 


#' R_fn_EHMC_SNAPER_ADAM_burnin
#' @keywords internal
#' @export
R_fn_EHMC_SNAPER_ADAM_burnin <-    function(    Model_type,
                                                vect_type,
                                                parallel_method,
                                                Stan_data_list,
                                                y,
                                                N,
                                                sample_nuisance,
                                                n_params_main,
                                                n_nuisance,
                                                n_chains_burnin,
                                                seed,
                                                ##
                                                manual_tau, ##
                                                tau_if_manual, ##
                                                ##
                                                n_burnin,
                                                LR_main,
                                                LR_us,
                                                n_adapt,
                                                partitioned_HMC,
                                                clip_iter,
                                                gap,
                                                metric_type_main,# = "Hessian",
                                                metric_shape_main,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ape_main,# = "dense",
                                                metric_type_nuisance ,#= "Empirical",
                                                metric_shape_nuisance ,#= "diag",
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
                                                theta_main_vectors_all_chains_input_from_R,
                                                theta_us_vectors_all_chains_input_from_R,
                                                n_nuisance_to_track,
                                                Model_args_as_Rcpp_List,
                                                EHMC_args_as_Rcpp_List,
                                                EHMC_Metric_as_Rcpp_List,
                                                EHMC_burnin_as_Rcpp_List) { 
  
  tictoc::tic()

  debug <- FALSE ## BOOKMARK
  
  message("Printing from R_fn_EHMC_SNAPER_ADAM_burnin:")

  RcppParallel::setThreadOptions(numThreads = n_chains_burnin);

  Model_type_R <- Model_type
  
  ## Fixed SNAPER-HMC / ADAM constants: 
  beta1_adam = 0.00
  beta2_adam = 0.95
  eps_adam = 1e-8
  ##
  kappa = 8.0
  eta_w <- 3
  
  if (sample_nuisance == TRUE) { 
    n_params <- n_params_main + n_nuisance
    index_nuisance <- 1:n_nuisance
    index_main <- (1 + n_nuisance):n_params
  } else { 
    n_params <- n_params_main  
    index_nuisance <- 1
    index_main <- 1:n_params
    n_nuisance <- 1
  }

  Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
  Model_args_as_Rcpp_List$n_params_main <- n_params_main

  if (Model_type != "Stan") {
    n_tests <- ncol(y)
    Model_args_as_Rcpp_List$Model_args_bools[15, 1] <- FALSE # debug
  }
  
  
  n_chains <- n_chains_burnin
  
  print(paste("n_params_main = ", n_params_main))
  print(paste("n_nuisance = ", n_nuisance))
 
  if (sample_nuisance == TRUE) { 
     theta_vec_mean <-  rowMeans( rbind(theta_us_vectors_all_chains_input_from_R, theta_main_vectors_all_chains_input_from_R))
  } else { 
     #### theta_main_vectors_all_chains_input_from_R[,] <- 0 
     theta_vec_mean <-  rowMeans( rbind(theta_main_vectors_all_chains_input_from_R))
  }
  

 #   theta_vec <- c(theta_us_vectors_all_chains_input_from_R[, 1], theta_main_vectors_all_chains_input_from_R[, 1]) # using inits from chain 1

  iter_seq_burnin <- seq(from = 1, to = n_burnin, by = 1)

  M_dense_main_non_scaled <-   EHMC_Metric_as_Rcpp_List$M_dense_main
  M_inv_dense_main_non_scaled <- EHMC_Metric_as_Rcpp_List$M_inv_dense_main
  M_inv_dense_main_chol_non_scaled <-  EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol
  
 
  {
      
          # --------  SNAPER stuff for main params (to load into C++ structs)
          EHMC_burnin_as_Rcpp_List$snaper_m_vec_main <-    theta_vec_mean[index_main]
          EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical <- rep(1, n_params_main)
          EHMC_burnin_as_Rcpp_List$snaper_w_vec_main <- rep(0.01, n_params_main)
          EHMC_burnin_as_Rcpp_List$eigen_max_main <- sqrt(sum(EHMC_burnin_as_Rcpp_List$snaper_w_vec_main^2) )
          EHMC_burnin_as_Rcpp_List$eigen_vector_main =   EHMC_burnin_as_Rcpp_List$snaper_w_vec_main/sqrt(abs(EHMC_burnin_as_Rcpp_List$eigen_max_main))
      
          # --------  SNAPER stuff for nuisance (to load into C++ structs)
          EHMC_burnin_as_Rcpp_List$snaper_m_vec_us <-     theta_vec_mean[index_nuisance]
          EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical  <- rep(1, n_nuisance)
          EHMC_burnin_as_Rcpp_List$snaper_w_vec_us <- rep(0.01, n_nuisance)
          EHMC_burnin_as_Rcpp_List$eigen_max_us  <- sqrt(sum(  EHMC_burnin_as_Rcpp_List$snaper_w_vec_us^2) )
          EHMC_burnin_as_Rcpp_List$eigen_vector_us =     EHMC_burnin_as_Rcpp_List$snaper_w_vec_us/sqrt(abs( EHMC_burnin_as_Rcpp_List$eigen_max_us ))
          
          EHMC_Metric_as_Rcpp_List$M_us_vec <-   1.0 / EHMC_Metric_as_Rcpp_List$M_inv_us_vec  
          
          EHMC_burnin_as_Rcpp_List$M_dense_sqrt <- pracma::sqrtm(EHMC_Metric_as_Rcpp_List$M_dense_main)[[1]]
          
          snaper_m_vec_all <-    theta_vec_mean
          snaper_s_vec_all_empirical <- rep(1, n_params)
          snaper_w_vec_all <- rep(0.01, n_params)
          eigen_max_all <- sqrt(sum(snaper_w_vec_all^2) )
          eigen_vector_all =   snaper_w_vec_all/sqrt(abs(eigen_max_all))

  }
  
  
 
 {
 
   if (Model_type == "Stan") { 
     n_class <- 0 
   } else { 
     n_class <-  Model_args_as_Rcpp_List$Model_args_ints[2]
   }
   
   print(paste("n_class = ", n_class))
   
  EHMC_args_as_Rcpp_List$eps_main <- 0.01 # just in case fn_find_initial_eps_main_and_us fails
  EHMC_args_as_Rcpp_List$eps_us <- 0.01 # just in case fn_find_initial_eps_main_and_us fails

  
  ### INITIAL VALUE(S) FOR EPSILON (I.E. THE HMC STEP-SIZE(S)) ---- BOOKMARK ------------------------------------------------------------------------------:
  try({

        if (sample_nuisance == TRUE) {
          theta_us_vec <-   c(theta_vec_mean[index_nuisance])
        } else {
          theta_us_vec <-   c(rep(1, n_nuisance))
        }

        par_res <- (BayesMVP:::fn_find_initial_eps_main_and_us(        theta_main_vec_initial_ref = matrix(c(theta_vec_mean[index_main]), ncol = 1),
                                                                       theta_us_vec_initial_ref = matrix(c(theta_us_vec), ncol = 1),
                                                                       partitioned_HMC = partitioned_HMC,
                                                                       seed = seed,
                                                                       Model_type = Model_type,
                                                                       force_autodiff = force_autodiff,
                                                                       force_PartialLog = force_PartialLog,
                                                                       multi_attempts = multi_attempts,
                                                                       y_ref = y,
                                                                       Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                                       EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                                                                       EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List))

        #par_res <- parallel::mccollect(par_proc)
        par_res <- par_res  ## [[1]]

        message(paste("step_sizes = "))
        ## main
        EHMC_args_as_Rcpp_List$eps_main <- min(0.50, par_res[[1]])
        cat(paste("eps_main = "))
        print(EHMC_args_as_Rcpp_List$eps_main)
        
        ## nuisance
        if (partitioned_HMC == TRUE) {
          
              EHMC_args_as_Rcpp_List$eps_us <-   min(0.50, par_res[[2]])
              cat(paste("eps_us = "))
              print(EHMC_args_as_Rcpp_List$eps_us)
              
        } else if (partitioned_HMC == FALSE) {
          
              EHMC_args_as_Rcpp_List$eps_us <-  EHMC_args_as_Rcpp_List$eps_main
            
        }

  })
   
  
  {
      # --------  eps for main (to load into C++ structs)
      EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- EHMC_args_as_Rcpp_List$eps_main
      EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
  
      # --------  eps for nuisance (to load into C++ structs)
      EHMC_burnin_as_Rcpp_List$eps_m_adam_us <- EHMC_args_as_Rcpp_List$eps_us
      EHMC_burnin_as_Rcpp_List$eps_v_adam_us  <- 0
  
      # --------  tau for main (to load into C++ structs)
      EHMC_args_as_Rcpp_List$tau_main  <-  EHMC_args_as_Rcpp_List$eps_main
      EHMC_burnin_as_Rcpp_List$tau_m_adam_main <- EHMC_args_as_Rcpp_List$tau_main
      EHMC_burnin_as_Rcpp_List$tau_v_adam_main <- 0
  
      # --------  tau for nuisance (to load into C++ structs)
      EHMC_args_as_Rcpp_List$tau_us  <-  EHMC_args_as_Rcpp_List$eps_us
      EHMC_burnin_as_Rcpp_List$tau_m_adam_us <- EHMC_args_as_Rcpp_List$tau_us
      EHMC_burnin_as_Rcpp_List$tau_v_adam_us <- 0
  
      # --------  M for nuisance (to load into C++ structs)
      M_us_vec <-   rep(1, n_nuisance)
      EHMC_Metric_as_Rcpp_List$M_inv_us_vec <- 1 / M_us_vec
      EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec <- rep(1, n_nuisance)
  
      # -------- M for main (to load into C++ structs)
      EHMC_Metric_as_Rcpp_List$M_inv_dense_main <- diag(rep(1, n_params_main))
      EHMC_Metric_as_Rcpp_List$M_dense_main <- diag(rep(1, n_params_main))
      EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- diag(rep(1, n_params_main))
  }
 
   


  div_main <- p_jump_per_chain <- div_us <- p_jump_us_per_chain <- c()
  
  time_burnin <- 0 

  
  ####  print(theta_main_vectors_all_chains_input_from_R)
  
  shrinkage_factor <- 1
  
  L_main_during_burnin_vec <- c()
  L_us_during_burnin_vec <-   c()
  
  
 ####  Start burnin   ------------------------------------------------------------------------------------------------------------------------------------------------
 for (ii in iter_seq_burnin) {
          
              if (ii %% 10 == 0) {
                    print(ii)
              }
            
              # if (ii %% 50 == 0) {
              #   gc(reset = TRUE)
              # }
          
               if (manual_tau == FALSE) {
                    try({
              
                      vec_points <-      c(  gap:(gap+1),
                                             max(c(round( 0.35 * n_adapt, 0) ), gap),
                                             max(c(round( 0.45 * n_adapt, 0) ), gap),
                                             max(c(round( 0.55 * n_adapt, 0) ), gap),
                                             max(c(round( 0.65 * n_adapt, 0) ), gap)
                                          )
              
                      if (ii %in% vec_points) {
              
                                tau_main_prop <-  tau_mult * sqrt(EHMC_burnin_as_Rcpp_List$eigen_max_main)
                                EHMC_args_as_Rcpp_List$tau_main  <-   if_not_NA_or_INF_else(tau_main_prop, 5.0 *   EHMC_args_as_Rcpp_List$eps_main)
                                ##
                                tau_us_prop <- tau_mult * sqrt(EHMC_burnin_as_Rcpp_List$eigen_max_us)
                                ## tau_us_prop <- 3 
                                EHMC_args_as_Rcpp_List$tau_us  <-   if_not_NA_or_INF_else(tau_us_prop, 5.0 * EHMC_args_as_Rcpp_List$eps_us)
                    
                      }
                    })
               }
         
               if (manual_tau == FALSE) {
                           if (ii %in% c(clip_iter:gap)) {
                                       
                                               if (ii < round((clip_iter + gap) / 4.0) )   {
                                                 
                                                     tau_main_prop  <-  2.0 *   EHMC_args_as_Rcpp_List$eps_main
                                                     tau_us_prop  <-    2.0 *   EHMC_args_as_Rcpp_List$eps_us
                                                 
                                               } else if (ii %in% c( round((clip_iter + gap) / 4):round((clip_iter + gap) / 2)  )) {
                                                 
                                                     tau_main_prop  <-  5.0 *   EHMC_args_as_Rcpp_List$eps_main
                                                     tau_us_prop  <-    5.0 *   EHMC_args_as_Rcpp_List$eps_us
                                                 
                                               } else if (ii %in% c( round((clip_iter + gap) / 2):round((clip_iter + gap) * 0.75 )  )) {
                                                 
                                                     tau_main_prop  <-  10.0 *   EHMC_args_as_Rcpp_List$eps_main
                                                     tau_us_prop  <-    10.0 *   EHMC_args_as_Rcpp_List$eps_us
                                                 
                                               } else {
                                                 
                                                     tau_main_prop  <-  0.75 *  tau_mult *   sqrt(EHMC_burnin_as_Rcpp_List$eigen_max_main)
                                                     tau_us_prop  <-    0.75 *  tau_mult *   sqrt(EHMC_burnin_as_Rcpp_List$eigen_max_us)
                                                
                                               }
                                               
                                               EHMC_args_as_Rcpp_List$tau_main  <-   if_not_NA_or_INF_else(tau_main_prop, 5.0 * EHMC_args_as_Rcpp_List$eps_main)
                                               EHMC_args_as_Rcpp_List$tau_us    <-   if_not_NA_or_INF_else(tau_us_prop,   5.0 * EHMC_args_as_Rcpp_List$eps_us)
                                               
                           }
               }
     
               # ### If not using partitioned_HMC (and therefore also NOT using diffusion_HMC), then set the path length to be the mean of tau_main and tau_us 
               # if (partitioned_HMC == FALSE) { # if sampling all parameters at once
               #         
               #         EHMC_args_as_Rcpp_List$tau_main <- mean( EHMC_args_as_Rcpp_List$tau_main, EHMC_args_as_Rcpp_List$tau_us)
               #   
               # }
     
               if (manual_tau == TRUE) {
                   if (length(tau_if_manual) == 1) {
                         EHMC_args_as_Rcpp_List$tau_main <- tau_if_manual
                         EHMC_args_as_Rcpp_List$tau_us   <- tau_if_manual
                   } else if (length(tau_if_manual) == 2) { 
                         EHMC_args_as_Rcpp_List$tau_main <- tau_if_manual[1]
                         EHMC_args_as_Rcpp_List$tau_us   <- tau_if_manual[2]
                   }
               }
          
               if ( EHMC_args_as_Rcpp_List$tau_main  > 50) {
                 EHMC_args_as_Rcpp_List$tau_main  <- 50
               }
               if ( EHMC_args_as_Rcpp_List$tau_us  > 50) {
                 EHMC_args_as_Rcpp_List$tau_us  <- 50
               }
     
               ## Make sure this comes last in the tau adaptation above:
               if (ii < clip_iter) {
                 EHMC_args_as_Rcpp_List$tau_us  <-    1 * EHMC_args_as_Rcpp_List$eps_us;
                 EHMC_args_as_Rcpp_List$tau_main  <-  1 * EHMC_args_as_Rcpp_List$eps_main;
               }
   
               #### ---------------------------------------------------------------------------------------------------------------------------------------------------------------
               #### current mean theta across all K chains:
               if (sample_nuisance == TRUE) { 
                        theta_vec_current_mean <- rowMeans(rbind(theta_us_vectors_all_chains_input_from_R, theta_main_vectors_all_chains_input_from_R))
                        theta_vec_current_us <-   theta_vec_current_mean[index_nuisance]
                        theta_vec_current_main <- theta_vec_current_mean[index_main]
               } else { 
                        theta_vec_current_mean <- rowMeans(rbind(theta_main_vectors_all_chains_input_from_R))
                        theta_vec_current_main <- theta_vec_current_mean
               }
               
               
               if (ii < 0.5 * n_burnin)  {
                        shrinkage_factor <- 0.75
                        main_vec_for_Hessian <- theta_vec_current_main
               } else {
                        shrinkage_factor <- 0.25
                        main_vec_for_Hessian <- EHMC_burnin_as_Rcpp_List$snaper_m_vec_main
               }
              
               #### ---------------------------------------------------------------------------------------------------------------------------------------------------------------
               if ( (ii < n_adapt) && (manual_tau == FALSE)) {
              
                     {
                        
                                  #### ////  updates for MAIN: --------------------------------------------------------------------------------------------------------------------
                                  ## update snaper_m and snaper_s_empirical (for MAIN):
                                  try({
                                  outs_update_snaper_m_and_s <- BayesMVP:::fn_update_snaper_m_and_s( EHMC_burnin_as_Rcpp_List$snaper_m_vec_main, 
                                                                                                     EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical, 
                                                                                                     theta_vec_current_main,  
                                                                                                     ii)
                                  EHMC_burnin_as_Rcpp_List$snaper_m_vec_main <- outs_update_snaper_m_and_s[,1]
                                  EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical <- outs_update_snaper_m_and_s[,2]
                                  })
                                  #### update snaper_w (for MAIN):
                                  try({
                                  # outs_update_eigen_max_and_eigen_vec <-  BayesMVP:::fn_update_snaper_w_dense_M(   snaper_w_vec = EHMC_burnin_as_Rcpp_List$snaper_w_vec_main,
                                  #                                                                                  eigen_vector = EHMC_burnin_as_Rcpp_List$eigen_vector_main,
                                  #                                                                                  eigen_max = EHMC_burnin_as_Rcpp_List$eigen_max_main,
                                  #                                                                                  theta_vec = theta_vec_current_main,
                                  #                                                                                  snaper_m_vec = EHMC_burnin_as_Rcpp_List$snaper_m_vec_main,
                                  #                                                                                  ii = ii,
                                  #                                                                                  M_dense_sqrt = EHMC_burnin_as_Rcpp_List$M_dense_sqrt)
                                  outs_update_eigen_max_and_eigen_vec <-  BayesMVP:::fn_update_snaper_w_diag_M(    snaper_w_vec = EHMC_burnin_as_Rcpp_List$snaper_w_vec_main,
                                                                                                                   eigen_vector = EHMC_burnin_as_Rcpp_List$eigen_vector_main,
                                                                                                                   eigen_max = EHMC_burnin_as_Rcpp_List$eigen_max_main,
                                                                                                                   theta_vec = theta_vec_current_main,
                                                                                                                   snaper_m_vec = EHMC_burnin_as_Rcpp_List$snaper_m_vec_main,
                                                                                                                   ii = ii,
                                                                                                                   sqrt_M_vec = c(diag(EHMC_burnin_as_Rcpp_List$M_dense_sqrt)))
                                  EHMC_burnin_as_Rcpp_List$snaper_w_vec_main <- outs_update_eigen_max_and_eigen_vec
                                  if (ii < 10) { 
                                     EHMC_burnin_as_Rcpp_List$snaper_w_vec_main <- abs(EHMC_burnin_as_Rcpp_List$snaper_w_vec_main)
                                  }
                                  })
                                  ##### update eigen_max and eigen_vec (for MAIN):
                                  try({
                                    outs_update_eigen_max_and_eigen_vec <-  BayesMVP:::fn_update_eigen_max_and_eigen_vec(snaper_w_vec = EHMC_burnin_as_Rcpp_List$snaper_w_vec_main)
                                    EHMC_burnin_as_Rcpp_List$eigen_max_main <- max(0.0001, outs_update_eigen_max_and_eigen_vec[1])
                                    EHMC_burnin_as_Rcpp_List$eigen_vector_main <- tail(outs_update_eigen_max_and_eigen_vec, n_params_main)
                                  })
                                  # {
                                  #     message(paste("snaper_m_vec_main:"))
                                  #     snaper_m_vec_main <-   EHMC_burnin_as_Rcpp_List$snaper_m_vec_main
                                  #     message(cat("length = ", length(snaper_m_vec_main)))
                                  #     message(cat("head = ", head(snaper_m_vec_main, 10)))
                                  #     ##
                                  #     message(paste("snaper_s_vec_main_empirical:"))
                                  #     snaper_s_vec_main_empirical <-   EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical
                                  #     message(cat("length = ", length(snaper_s_vec_main_empirical)))
                                  #     message(cat("head = ", head(snaper_s_vec_main_empirical, 10)))
                                  #     ##
                                  #     message(paste("snaper_w_vec_main:"))
                                  #     snaper_w_vec_main <-   EHMC_burnin_as_Rcpp_List$snaper_w_vec_main
                                  #     message(cat("length = ", length(snaper_w_vec_main)))
                                  #     message(cat("head = ", head(snaper_w_vec_main, 10)))
                                  #     ##
                                  #     message(paste("eigen_max_main:"))
                                  #     eigen_max_main <-   EHMC_burnin_as_Rcpp_List$eigen_max_main
                                  #     message(cat("length = ", length(eigen_max_main)))
                                  #     message(cat("head = ", head(eigen_max_main, 10)))
                                  #     ##
                                  #     message(paste("eigen_vector_main:"))
                                  #     eigen_vector_main <-   EHMC_burnin_as_Rcpp_List$eigen_vector_main
                                  #     message(cat("length = ", length(eigen_vector_main)))
                                  #     message(cat("head = ", head(eigen_vector_main, 10)))
                                  # }
                     }
                            #### ////  updates for NUISANCE: --------------------------------------------------------------------------------------------------------------------
                        if (partitioned_HMC == TRUE) {
                            if (sample_nuisance == TRUE) { 
                                  ## update snaper_m and snaper_s_empirical (for NUISANCE):
                                  try({
                                  outs_update_snaper_m_and_s <-   BayesMVP:::fn_update_snaper_m_and_s( EHMC_burnin_as_Rcpp_List$snaper_m_vec_us,  
                                                                                                       EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical, 
                                                                                                       theta_vec_current_us, 
                                                                                                       ii)
                                  EHMC_burnin_as_Rcpp_List$snaper_m_vec_us <- outs_update_snaper_m_and_s[,1]
                                  EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical <- outs_update_snaper_m_and_s[,2]
                                  })
                                  # update snaper_w (for NUISANCE):
                                  try({
                                    outs_update_eigen_max_and_eigen_vec <-   BayesMVP:::fn_update_snaper_w_diag_M(   snaper_w_vec = EHMC_burnin_as_Rcpp_List$snaper_w_vec_us,
                                                                                                                     eigen_vector = EHMC_burnin_as_Rcpp_List$eigen_vector_us,
                                                                                                                     eigen_max = EHMC_burnin_as_Rcpp_List$eigen_max_us,
                                                                                                                     theta_vec = theta_vec_current_us,
                                                                                                                     snaper_m_vec = EHMC_burnin_as_Rcpp_List$snaper_m_vec_us,
                                                                                                                     ii = ii,
                                                                                                                     sqrt_M_vec =  EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec)
                                    EHMC_burnin_as_Rcpp_List$snaper_w_vec_us <- outs_update_eigen_max_and_eigen_vec
                                  })
                                  # update eigen_max and eigen_vec (for NUISANCE):
                                  try({
                                  outs_update_eigen_max_and_eigen_vec <-   BayesMVP:::fn_update_eigen_max_and_eigen_vec(snaper_w_vec = EHMC_burnin_as_Rcpp_List$snaper_w_vec_us)
                                  EHMC_burnin_as_Rcpp_List$eigen_max_us <-  max(0.0001, outs_update_eigen_max_and_eigen_vec[1] )
                                  EHMC_burnin_as_Rcpp_List$eigen_vector_us <- tail(outs_update_eigen_max_and_eigen_vec, n_nuisance)
                                  })
                            }
                        
                      }
                      if (partitioned_HMC == FALSE) {
                                  # 
                                  # theta_vec_mean_all <- theta_vec_current_mean
                                  # ##
                                  # snaper_m_vec_all <-  c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_us, EHMC_burnin_as_Rcpp_List$snaper_m_vec_main)
                                  # snaper_s_vec_all_empirical <- c(EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical, EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical)
                                  # ##
                                  # eigen_vector_all <- c(EHMC_burnin_as_Rcpp_List$eigen_vector_us, EHMC_burnin_as_Rcpp_List$eigen_vector_main)
                                  # snaper_w_vec_all <- c(EHMC_burnin_as_Rcpp_List$snaper_w_vec_us, EHMC_burnin_as_Rcpp_List$snaper_w_vec_main)
                                  # ##
                                  # sqrt_M_vec_all <- c(EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec, diag(EHMC_burnin_as_Rcpp_List$M_dense_sqrt))
                                  # ##
                                  # ## update snaper_m and snaper_s_empirical (for ALL PARAMS):
                                  # try({
                                  #       outs_update_snaper_m_and_s <-   BayesMVP:::fn_update_snaper_m_and_s( snaper_m_vec_all,  
                                  #                                                                            snaper_s_vec_all_empirical, 
                                  #                                                                            theta_vec_mean_all, 
                                  #                                                                            ii)
                                  #       ## Update snaper_m:
                                  #       snaper_m_vec_all <- outs_update_snaper_m_and_s[,1]
                                  #       EHMC_burnin_as_Rcpp_List$snaper_m_vec_us <-   head(snaper_m_vec_all, n_nuisance)
                                  #       EHMC_burnin_as_Rcpp_List$snaper_m_vec_main <- tail(snaper_m_vec_all, n_params_main)
                                  #       ## Update snaper_s:
                                  #       snaper_s_vec_all_empirical <- outs_update_snaper_m_and_s[,2]
                                  #       EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical <-   head(snaper_s_vec_all_empirical, n_nuisance)
                                  #       EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical <- tail(snaper_s_vec_all_empirical, n_params_main)
                                  # })
                                  # # update snaper_w (for ALL PARAMS):
                                  # try({
                                  #   outs_update_eigen_max_and_eigen_vec <-   BayesMVP:::fn_update_snaper_w_diag_M(   snaper_w_vec = snaper_w_vec_all,
                                  #                                                                                    eigen_vector = eigen_vector_all,
                                  #                                                                                    eigen_max = eigen_max_all,
                                  #                                                                                    theta_vec = theta_vec_mean_all,
                                  #                                                                                    snaper_m_vec = snaper_m_vec_all,
                                  #                                                                                    ii = ii,
                                  #                                                                                    sqrt_M_vec =  sqrt_M_vec_all)
                                  #   ## Update snaper_w:
                                  #   snaper_w_vec_all <- outs_update_eigen_max_and_eigen_vec
                                  #   EHMC_burnin_as_Rcpp_List$snaper_w_vec_us <-   head(snaper_w_vec_all, n_nuisance)
                                  #   EHMC_burnin_as_Rcpp_List$snaper_w_vec_main <- tail(snaper_w_vec_all, n_params_main)
                                  # })
                                  # # update eigen_max and eigen_vec (for ALL PARAMS):
                                  # try({
                                  #   outs_update_eigen_max_and_eigen_vec <- BayesMVP:::fn_update_eigen_max_and_eigen_vec(snaper_w_vec = snaper_w_vec_all)
                                  #   ## Update eigen_max:
                                  #   eigen_max_all <-  max(0.0001, outs_update_eigen_max_and_eigen_vec[1] )
                                  #   EHMC_burnin_as_Rcpp_List$eigen_max_main <- eigen_max_all
                                  #   EHMC_burnin_as_Rcpp_List$eigen_max_us   <- eigen_max_all
                                  #   ## Update eigen_vector's:
                                  #   eigen_vector_all <- tail(outs_update_eigen_max_and_eigen_vec, n_params)
                                  #   EHMC_burnin_as_Rcpp_List$eigen_vector_us <-   head(eigen_vector_all, n_nuisance)
                                  #   EHMC_burnin_as_Rcpp_List$eigen_vector_main <- tail(eigen_vector_all, n_params_main)
                                  # })
                        
                      }
                
        
               }
    
          if (ii < 2) {
            empicical_cov_main <- diag(EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical)
          }
    
          ## update covariance using Welford's algorithm
          if ( (metric_type_main == "Empirical") && (metric_shape_main == "dense") ) {
        
                  cov_outs <- BayesMVP:::update_cov_Welford( new_sample = theta_vec_current_main,
                                                  ii = ii,
                                                  mean_vec = EHMC_burnin_as_Rcpp_List$snaper_m_vec_main,
                                                  cov_mat = empicical_cov_main)
                  empicical_cov_main <- (ii/(ii-1)) * cov_outs$cov_mat # for unbiased estimate
    
          }


          { ## //// Update metric(s):
            
                    if  ( (ii >  (- 1 + round(n_burnin/5))) && (ii %% interval_width_main == 0) && (ii < (n_burnin - round(n_burnin/10))) )  {
            
                           if (partitioned_HMC == FALSE) { 
                                            
                                            if (metric_shape_main == "unit") { ## when partitioned_HMC is FALSE, setting "metric_shape_main" to "unit" results in the ENTIRE metric being a unit metric!!
                                              
                                                      metric_diag_vec <- rep(1, n_params)
                                                      
                                                      ## Metrics for main:
                                                      EHMC_Metric_as_Rcpp_List$M_inv_main_vec <- matrix(rep(1, n_params_main), ncol = 1)
                                                      EHMC_Metric_as_Rcpp_List$M_inv_dense_main <-  diag(c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec))
                                                      EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <-  EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                                                      EHMC_Metric_as_Rcpp_List$M_dense_sqrt <-           EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                                                      ## Metrics for nuisance:
                                                      EHMC_Metric_as_Rcpp_List$M_inv_us_vec <-  matrix(rep(1, n_nuisance), ncol = 1)
                                                      EHMC_Metric_as_Rcpp_List$M_us_vec <-      matrix(rep(1, n_nuisance), ncol = 1)
                                                      EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec <- matrix(rep(1, n_nuisance), ncol = 1)
                                              
                                              
                                            } else if (metric_shape_main == "diag") { 
                                              
                                                      ## First get the value of the current metric:
                                                      M_inv_us_vec_current <- EHMC_Metric_as_Rcpp_List$M_inv_us_vec
                                                      M_inv_main_vec_current <- EHMC_Metric_as_Rcpp_List$M_inv_main_vec
                                                 
                                                      if (metric_type_main == "Empirical") {
                                                        
                                                               snaper_s_vec_to_use_main <-     EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical
                                                          
                                                      } else if (metric_type_main == "Hessian") {  ## then use the inverse-negative Hessian (diagonal) computed above
                                                          
                                                               outs <-     BayesMVP:::fn_Rcpp_compute_PD_Hessian_main(   shrinkage_factor = shrinkage_factor,
                                                                                                                         num_diff_e = 0.00001,
                                                                                                                         Model_type = Model_type,
                                                                                                                         force_autodiff = force_autodiff,
                                                                                                                         force_PartialLog = force_PartialLog,
                                                                                                                         multi_attempts = multi_attempts,
                                                                                                                         theta_main_vec = main_vec_for_Hessian,
                                                                                                                         theta_us_vec = EHMC_burnin_as_Rcpp_List$snaper_m_vec_us,
                                                                                                                         y = y,
                                                                                                                         Model_args_as_Rcpp_List = Model_args_as_Rcpp_List)
                                                               Hessian <- outs$Hessian
                                                               diag_Hessian <- c(diag(Hessian))
                                                               snaper_s_vec_to_use_main <-     matrix(c(1 / diag_Hessian), ncol = 1)
                                                          
                                                      }
                                                      
                                                      if (metric_type_nuisance == "Empirical") {
                                                        
                                                               snaper_s_vec_to_use_nuisance <- EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical
                                                          
                                                      }
                                                      
                                                      ## Now compute ADJUSTED snaper_s_vec_all:
                                                      snaper_s_vec_all <- c(snaper_s_vec_to_use_nuisance, snaper_s_vec_to_use_main)
                                                      max_var_s <- max(snaper_s_vec_all)
                                                      adj_snaper_s_vec_all <- max_var_s * (1 / snaper_s_vec_all)
                                                      adj_snaper_s_vec_to_use_nuisance <- adj_snaper_s_vec_all[index_nuisance]
                                                      adj_snaper_s_vec_to_use_main <- adj_snaper_s_vec_all[index_main]
                                                      
                                                      ## now update the OVERALL metric and then ADJUST AGAIN:
                                                      M_inv_us_vec <-       ratio_M_us *   adj_snaper_s_vec_to_use_nuisance +     (1 - ratio_M_us) *   M_inv_us_vec_current
                                                      M_inv_main_vec <-     ratio_M_main * adj_snaper_s_vec_to_use_main +         (1 - ratio_M_main) * M_inv_main_vec_current
                                                      
                                                      ## Final ADJUSTED-OVERALL metric (for main + nuisance):
                                                      M_inv_all_vec <- c(M_inv_us_vec, M_inv_main_vec)
                                                      max_var_new <- max(M_inv_all_vec)
                                                      adj_M_inv_all_vec <- max_var_new * (1 / M_inv_all_vec)
                                                      
                                                      ## Then replace "M_inv_all_vec" with the adjusted version:
                                                      M_inv_all_vec <- adj_M_inv_all_vec
                                                      M_all_vec <- 1.0 / M_inv_all_vec
                                                      sqrt_M_all_vec <- sqrt(M_all_vec)
                                                      
                                                      #### ---- Now update the C++ lists:
                                                      #### ---- for main:
                                                      EHMC_Metric_as_Rcpp_List$M_inv_main_vec <-  matrix(M_inv_all_vec[index_main], ncol = 1)
                                                      ## Now update the "dense" params:
                                                      EHMC_Metric_as_Rcpp_List$M_dense_main <- diag(c(1 / c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec)))
                                                      EHMC_Metric_as_Rcpp_List$M_inv_dense_main <-  diag(c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec))
                                                      EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <-  t(chol(EHMC_Metric_as_Rcpp_List$M_inv_dense_main))
                                                      EHMC_burnin_as_Rcpp_List$M_dense_sqrt <- pracma::sqrtm(EHMC_Metric_as_Rcpp_List$M_dense_main)[[1]]
                                                      #### ---- for nuisance:
                                                      EHMC_Metric_as_Rcpp_List$M_inv_us_vec <-  matrix(M_inv_all_vec[index_nuisance], ncol = 1)
                                                      EHMC_Metric_as_Rcpp_List$M_us_vec <-      matrix(M_all_vec[index_nuisance], ncol = 1)
                                                      EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec <- matrix(sqrt_M_all_vec[index_nuisance], ncol = 1)
                                              
                                            } else if (metric_shape_main == "dense") { ## NOTE: when partitioned_HMC<-FALSE, setting "metric_shape_main<-dense" results in a dense MAIN metric and a diagonal NUISANCE metric.
                                                      
                                                      ## First get current metric:
                                                      M_inv_us_vec_current <-     EHMC_Metric_as_Rcpp_List$M_inv_us_vec
                                                      M_inv_main_DENSE_current <- EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                                                      
                                                      if (metric_type_main == "Empirical") {
                                                        
                                                            snaper_s_vec_to_use_main <-     EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical
                                                            snaper_s_vec_to_use_main_DENSE <- empicical_cov_main
                                                            snaper_s_vec_to_use_main_DENSE_inv <- BayesMVP:::Rcpp_solve(snaper_s_vec_to_use_main_DENSE)
                                                          
                                                      } else if (metric_type_main == "Hessian") {  ## then use the inverse-negative Hessian (diagonal) computed above
                                                        
                                                            outs <-     BayesMVP:::fn_Rcpp_compute_PD_Hessian_main(   shrinkage_factor = shrinkage_factor,
                                                                                                                      num_diff_e = 0.00001,
                                                                                                                      Model_type = Model_type,
                                                                                                                      force_autodiff = force_autodiff,
                                                                                                                      force_PartialLog = force_PartialLog,
                                                                                                                      multi_attempts = multi_attempts,
                                                                                                                      theta_main_vec = main_vec_for_Hessian,
                                                                                                                      theta_us_vec = EHMC_burnin_as_Rcpp_List$snaper_m_vec_us,
                                                                                                                      y = y,
                                                                                                                      Model_args_as_Rcpp_List = Model_args_as_Rcpp_List)
                                                            Hessian <- outs$Hessian
                                                            diag_Hessian <- c(diag(Hessian))
                                                            snaper_s_vec_to_use_main <-     matrix(diag_Hessian, ncol = 1)
                                                            snaper_s_vec_to_use_main_DENSE <- Hessian
                                                            snaper_s_vec_to_use_main_DENSE_inv <- BayesMVP:::Rcpp_solve(snaper_s_vec_to_use_main_DENSE)
                                                        
                                                      }
                                                      
                                                      if (metric_type_nuisance == "Empirical") {
                                                        
                                                            snaper_s_vec_to_use_nuisance <- EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical
                                                        
                                                      }
                                                      
                                                      ## Find the OVERALL maximum variance (comparing diagonal and dense):
                                                      max_var_s <- max(max(snaper_s_vec_to_use_nuisance), max(snaper_s_vec_to_use_main))
                                                      
                                                      ## Adjust both metrics using the SAME max_var_s:
                                                      adj_snaper_s_vec_to_use_main_DENSE <- max_var_s * snaper_s_vec_to_use_main_DENSE_inv
                                                      adj_snaper_s_vec_to_use_nuisance <-   max_var_s * (1 / snaper_s_vec_to_use_nuisance)
                                                      
                                                      #### Update both metrics with ratio:
                                                      ## Main:
                                                      M_inv_dense_main <- BayesMVP:::Rcpp_near_PD(ratio_M_main * adj_snaper_s_vec_to_use_main_DENSE + (1 - ratio_M_main) * M_inv_main_DENSE_current)
                                                      M_dense_main <-     BayesMVP:::Rcpp_solve(M_inv_dense_main)
                                                      ## nuisance:
                                                      M_inv_us_vec <-     ratio_M_us * adj_snaper_s_vec_to_use_nuisance +   (1 - ratio_M_us) * M_inv_us_vec_current
  
                                                      ## Adjust both again using same OVERALL maximum variance:
                                                      max_var_new <- max(max(M_inv_us_vec), max(diag(M_inv_dense_main)))
                                                      adj_M_inv_dense_main <- max_var_new * M_dense_main
                                                      adj_M_inv_us_vec <-     max_var_new * (1 / M_inv_us_vec)
                                                      
                                                      ## Update the C++ lists:
                                                      ## For main:
                                                      EHMC_Metric_as_Rcpp_List$M_inv_dense_main <- adj_M_inv_dense_main
                                                      EHMC_Metric_as_Rcpp_List$M_dense_main <- solve(adj_M_inv_dense_main)
                                                      EHMC_Metric_as_Rcpp_List$M_inv_main_vec <- matrix(diag(adj_M_inv_dense_main), ncol = 1)
                                                      EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- t(chol(adj_M_inv_dense_main))
                                                      EHMC_burnin_as_Rcpp_List$M_dense_sqrt <- pracma::sqrtm(EHMC_Metric_as_Rcpp_List$M_dense_main)[[1]]
                                                      
                                                      ## For nuisance:
                                                      EHMC_Metric_as_Rcpp_List$M_inv_us_vec <- matrix(adj_M_inv_us_vec, ncol = 1)
                                                      EHMC_Metric_as_Rcpp_List$M_us_vec <- matrix(1.0 / adj_M_inv_us_vec, ncol = 1)
                                                      EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec <- matrix(sqrt(1.0 / adj_M_inv_us_vec), ncol = 1)
                                              
                                            }
                             
                           } else if (partitioned_HMC == TRUE) { 
                                   
                                   if (metric_shape_main == "unit") { 
                                     
                                           EHMC_Metric_as_Rcpp_List$M_inv_main_vec <- matrix(rep(1, n_params_main), ncol = 1)
                                           EHMC_Metric_as_Rcpp_List$M_inv_dense_main <-  diag(c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec))
                                           EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <-  EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                                           EHMC_Metric_as_Rcpp_List$M_dense_sqrt <-           EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                                     
                                   } else { 
                                     
                                           if (metric_type_main == "Hessian") {
                                             
                                                       ## //////////// update M_dense (for main param) using Hessian (num_diff)
                                                       try({
                                                         
                                                         message(paste("updating Hessian (for main params)"))
                                                         
                                                         Rcpp_update_M_dense_using_Hessian_num_diff <-     BayesMVP:::fn_Rcpp_wrapper_update_M_dense_main_Hessian( M_dense_main = M_dense_main_non_scaled,
                                                                                                                                                                   M_inv_dense_main = M_inv_dense_main_non_scaled,
                                                                                                                                                                   M_inv_dense_main_chol = M_inv_dense_main_chol_non_scaled,
                                                                                                                                                                   shrinkage_factor = shrinkage_factor,
                                                                                                                                                                   ratio_Hess_main = ratio_M_main,
                                                                                                                                                                   interval_width = interval_width_main,
                                                                                                                                                                   num_diff_e = 0.00001,
                                                                                                                                                                   Model_type = Model_type,
                                                                                                                                                                   force_autodiff = force_autodiff,
                                                                                                                                                                   force_PartialLog = force_PartialLog,
                                                                                                                                                                   multi_attempts = multi_attempts,
                                                                                                                                                                   theta_main_vec = main_vec_for_Hessian,
                                                                                                                                                                   theta_us_vec = EHMC_burnin_as_Rcpp_List$snaper_m_vec_us,
                                                                                                                                                                   y = y,
                                                                                                                                                                   Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                                                                                                                                   ii = ii,
                                                                                                                                                                   n_burnin = n_adapt,
                                                                                                                                                                   metric_type = "Hessian")
                                                         
                                                         M_dense_main_non_scaled <-     Rcpp_update_M_dense_using_Hessian_num_diff[[1]]
                                                         M_inv_dense_main_non_scaled <- Rcpp_update_M_dense_using_Hessian_num_diff[[2]]
                                                         M_inv_dense_main_chol_non_scaled <- Rcpp_update_M_dense_using_Hessian_num_diff[[3]]
                                                         
                                                         ## re-scale by largest variance:
                                                         max_var <-  1  ## max(diag(M_inv_dense_main_non_scaled))
                                                         EHMC_Metric_as_Rcpp_List$M_dense_main <-  max_var * M_dense_main_non_scaled
                                                         EHMC_Metric_as_Rcpp_List$M_inv_dense_main <-  BayesMVP:::Rcpp_solve(EHMC_Metric_as_Rcpp_List$M_dense_main)
                                                         EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <-  BayesMVP:::Rcpp_Chol(EHMC_Metric_as_Rcpp_List$M_inv_dense_main)
                                                         EHMC_burnin_as_Rcpp_List$M_dense_sqrt <- pracma::sqrtm(EHMC_Metric_as_Rcpp_List$M_dense_main)[[1]]
                                                         
                                                         if (metric_shape_main == "diag") {
                                                           
                                                           EHMC_Metric_as_Rcpp_List$M_inv_main_vec <-  matrix(diag(EHMC_Metric_as_Rcpp_List$M_inv_dense_main), ncol = 1)
                                                           EHMC_Metric_as_Rcpp_List$M_inv_dense_main <- diag(c(diag(EHMC_Metric_as_Rcpp_List$M_inv_dense_main)))
                                                           EHMC_Metric_as_Rcpp_List$M_dense_main <- diag(c(diag(  1 /   EHMC_Metric_as_Rcpp_List$M_dense_main  )))
                                                           EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- t(chol((EHMC_Metric_as_Rcpp_List$M_inv_dense_main)))
                                                           
                                                         } else if (metric_shape_main == "dense") {
                                                           
                                                           EHMC_Metric_as_Rcpp_List$M_inv_main_vec <- matrix((diag(EHMC_Metric_as_Rcpp_List$M_inv_dense_main)), ncol = 1)
                                                           
                                                         } else if (metric_shape_main == "unit") { 
                                                           
                                                           EHMC_Metric_as_Rcpp_List$M_inv_main_vec <- matrix(rep(1, n_params_main), ncol = 1)
                                                           EHMC_Metric_as_Rcpp_List$M_inv_dense_main <-  diag(c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec))
                                                           EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <-  EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                                                           EHMC_Metric_as_Rcpp_List$M_dense_sqrt <-           EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                                                           
                                                         }
                                                         
                                                       })
                                             
                                           } else if (metric_type_main == "Empirical") {
                                             
                                                      message(paste("updating Euclidean metric (for main params)"))
                                                     
                                                     if (metric_shape_main == "diag") {
                                                           
                                                                 max_var <- 1 #  max(EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical)
                                                                 M_as_inv_snaper_s_vec_main_empirical <-   1 / EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical
                                                                 M_as_inv_snaper_s_vec_main_empirical_scaled <- max_var * M_as_inv_snaper_s_vec_main_empirical
                                                                 M_inv_as_snaper_s_vec_main_empirical_scaled <- 1 / M_as_inv_snaper_s_vec_main_empirical_scaled
                                                                 
                                                                 ## now update M_inv_nuisance
                                                                 M_inv_main_vec_current <- EHMC_Metric_as_Rcpp_List$M_inv_main_vec
                                                                 EHMC_Metric_as_Rcpp_List$M_inv_main_vec <-  ratio_M_main * M_inv_as_snaper_s_vec_main_empirical_scaled + (1 - ratio_M_main) * M_inv_main_vec_current
                                                                 EHMC_burnin_as_Rcpp_List$sqrt_M_main_vec <- EHMC_Metric_as_Rcpp_List$M_inv_main_vec
                                                       
                                                     } else if (metric_shape_main == "dense") {
                                                           
                                                                 max_var <-  1 # max(diag(empicical_cov_main))
                                                                 inv_empicical_cov_main <-   BayesMVP:::Rcpp_solve(empicical_cov_main)
                                                                 inv_empicical_cov_main_scaled <- max_var * inv_empicical_cov_main
                                                                 #  empicical_cov_main_scaled <- BayesMVP:::Rcpp_solve(inv_empicical_cov_main_scaled) #   1 / inv_empicical_cov_main_scaled
                                                                 
                                                                 EHMC_Metric_as_Rcpp_List$M_dense_main <-  ratio_M_main * inv_empicical_cov_main_scaled    +    (1 - ratio_M_main) *  EHMC_Metric_as_Rcpp_List$M_dense_main
                                                                 EHMC_Metric_as_Rcpp_List$M_inv_dense_main <- BayesMVP:::Rcpp_solve(EHMC_Metric_as_Rcpp_List$M_dense_main)
                                                                 EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <-  BayesMVP:::Rcpp_Chol(  EHMC_Metric_as_Rcpp_List$M_inv_dense_main )
                                                                 EHMC_burnin_as_Rcpp_List$M_dense_sqrt <-     pracma::sqrtm(EHMC_Metric_as_Rcpp_List$M_dense_main)[[1]]
                                                                 ## And update the diagonal:
                                                                 EHMC_Metric_as_Rcpp_List$M_inv_main_vec <- matrix(c(diag(EHMC_Metric_as_Rcpp_List$M_inv_dense_main)), ncol = 1)
                                                       
                                                     }
                                                     
                                           }
                                     
                                   }
                                   
                                   if  ( (ii < n_adapt) &&  (sample_nuisance == TRUE)  ) {
                                       
                                       try({
                                         
                                         if (metric_shape_nuisance == "unit") { 
                                           
                                           #### ---- for unit metric --------------------------------------
                                           EHMC_Metric_as_Rcpp_List$M_inv_us_vec <-  matrix(rep(1, n_nuisance), ncol = 1)
                                           EHMC_Metric_as_Rcpp_List$M_us_vec <-      matrix(rep(1, n_nuisance), ncol = 1)
                                           EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec <- matrix(rep(1, n_nuisance), ncol = 1)
                                           
                                         } else {
                                           
                                           ## ------------------------------------------------------------------------------------------------------------------------------------------- 
                                           max_var <- 1 #max(EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical)
                                           M_as_inv_snaper_s_vec_us_empirical <-   1 / EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical
                                           M_as_inv_snaper_s_vec_us_empirical_scaled <- max_var * M_as_inv_snaper_s_vec_us_empirical
                                           M_inv_as_snaper_s_vec_us_empirical_scaled <- 1 / M_as_inv_snaper_s_vec_us_empirical_scaled
                                           ## now update M_inv_nuisance:
                                           EHMC_Metric_as_Rcpp_List$M_inv_us_vec <- ratio_M_us * M_inv_as_snaper_s_vec_us_empirical_scaled + (1 - ratio_M_us) * EHMC_Metric_as_Rcpp_List$M_inv_us_vec
                                           EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec <- sqrt(1 /  EHMC_Metric_as_Rcpp_List$M_inv_us_vec)
                                           EHMC_Metric_as_Rcpp_List$M_us_vec <- 1.0 / EHMC_Metric_as_Rcpp_List$M_inv_us_vec
                                           
                                         }
                                         
                                       })
                                     
                                   }
                             
                           }
                      
                    } ## // end of "if  ( (ii >  (- 1 + round(n_burnin/5))) && (ii %% interval_width_main == 0) && (ii < (n_burnin - round(n_burnin/10))) )  {" loop
            
          }  ## // end of metric block
   
          ## //////////////////   --------------------------------  Perform iteration  ------------------------------------------------------------------------------
          try({
            
                                        if (parallel_method == "RcppParallel")
                                          fn <- BayesMVP:::fn_R_RcppParallel_EHMC_single_iter_burnin
                                        else {  ### OpenMP
                                          fn <- BayesMVP:::fn_R_OpenMP_EHMC_single_iter_burnin
                                        }
            
                                        result <-   fn(      n_threads_R = n_chains_burnin,
                                                             seed_R = seed + ii, ## seed + ii
                                                             sample_nuisance_R = sample_nuisance,
                                                             n_nuisance_to_track = n_nuisance_to_track,
                                                             n_iter_R = 1,
                                                             current_iter_R = ii,
                                                             n_adapt  = 0,
                                                             partitioned_HMC_R = partitioned_HMC,
                                                             clip_iter = clip_iter,
                                                             gap =  gap,
                                                             burnin_indicator = FALSE,
                                                             metric_type_nuisance = metric_type_nuisance,
                                                             metric_type_main = metric_type_main,
                                                             shrinkage_factor = shrinkage_factor,
                                                             max_eps_main = max_eps_main,
                                                             max_eps_us = max_eps_us,
                                                             tau_main_target = 0, # dummy arg
                                                             tau_us_target = 0,   # dummy arg
                                                             main_L_manual = FALSE,  # dummy arg
                                                             L_main_if_manual = 0,   # dummy arg
                                                             us_L_manual = FALSE,    # dummy arg
                                                             L_us_if_manual = 0,     # dummy arg
                                                             max_L = max_L,
                                                             tau_mult = tau_mult,
                                                             ratio_M_us = ratio_M_us,
                                                             ratio_Hess_main = ratio_M_main,
                                                             M_interval_width = interval_width_main,
                                                             Model_type_R =  Model_type,
                                                             force_autodiff_R = force_autodiff,
                                                             force_PartialLog_R = force_PartialLog ,
                                                             multi_attempts_R = multi_attempts,
                                                             theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                                             theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R,
                                                             y_Eigen_R =  y,
                                                             Model_args_as_Rcpp_List =  Model_args_as_Rcpp_List,
                                                             EHMC_args_as_Rcpp_List =   EHMC_args_as_Rcpp_List,
                                                             EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                             EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List)

                           #  result <- parallel::mccollect(result)

                            theta_main_vectors_all_chains_input_from_R <-                  result[[2]]
                            theta_main_0_burnin_tau_adapt_all_chains_input_from_R <-       result[[6]]
                            theta_main_prop_burnin_tau_adapt_all_chains_input_from_R <-    result[[7]]
                            velocity_main_0_burnin_tau_adapt_all_chains_input_from_R <-    result[[8]]
                            velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R <- result[[9]]
                            tau_main_ii_vec <- result[[3]][6,]

                         if (sample_nuisance == TRUE) { 
                           
                                theta_us_vectors_all_chains_input_from_R <-                  result[[4]]
                                theta_us_0_burnin_tau_adapt_all_chains_input_from_R <-       result[[10]]
                                theta_us_prop_burnin_tau_adapt_all_chains_input_from_R <-    result[[11]]
                                velocity_us_0_burnin_tau_adapt_all_chains_input_from_R <-    result[[12]]
                                velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R <- result[[13]]
                                tau_us_ii_vec <-   result[[5]][6,]
                                
                         }
 
          })
                        try({
                          
                              div_main <- div_us <- c()
                              p_jump_per_chain <- p_jump_us_per_chain <- c()
                              
                              for (kk in 1:n_chains_burnin) {
                                div_main[kk] <- result[[3]][, kk][2]
                                p_jump_per_chain[kk] <-   result[[3]][, kk][1]
                                if (sample_nuisance == TRUE) { 
                                  div_us[kk] <- result[[5]][, kk][2]
                                  p_jump_us_per_chain[kk] <-   result[[5]][, kk][1]
                                }
                              }
                          
                        })
  
                        p_jump_main <- mean(p_jump_per_chain, na.rm = TRUE)
                        if (sample_nuisance == TRUE) { 
                           p_jump_us <- mean(p_jump_us_per_chain, na.rm = TRUE)
                        }

                        if (ii < n_adapt) {

                        ## //////////////////   --------------------------------  update eps (step-size) for main ------------------------------------------------------------------------------
                                        adapt_eps_outs <-  BayesMVP:::fn_Rcpp_wrapper_adapt_eps_ADAM( EHMC_args_as_Rcpp_List$eps_main,
                                                                                           EHMC_burnin_as_Rcpp_List$eps_m_adam_main,
                                                                                           EHMC_burnin_as_Rcpp_List$eps_v_adam_main,
                                                                                           ii, n_adapt,
                                                                                           EHMC_burnin_as_Rcpp_List$LR_main,
                                                                                           p_jump_main, EHMC_burnin_as_Rcpp_List$adapt_delta_main,
                                                                                           beta1_adam, beta2_adam, eps_adam)

                                        EHMC_args_as_Rcpp_List$eps_main <-    min(1.0, adapt_eps_outs[1])   ;  EHMC_args_as_Rcpp_List$eps_main
                                        EHMC_burnin_as_Rcpp_List$eps_m_adam_main <-  adapt_eps_outs[2]   ; EHMC_burnin_as_Rcpp_List$eps_m_adam_main
                                        EHMC_burnin_as_Rcpp_List$eps_v_adam_main <-  adapt_eps_outs[3]   ;    EHMC_burnin_as_Rcpp_List$eps_v_adam_main

                        ## //////////////////   --------------------------------  update eps (step-size) for nuisance ------------------------------------------------------------------------------
                                        if  (   (sample_nuisance == TRUE)  ) {
                                            adapt_eps_outs <-  BayesMVP:::fn_Rcpp_wrapper_adapt_eps_ADAM(  EHMC_args_as_Rcpp_List$eps_us,
                                                                                                EHMC_burnin_as_Rcpp_List$eps_m_adam_us,
                                                                                                EHMC_burnin_as_Rcpp_List$eps_v_adam_us ,
                                                                                                ii, n_adapt, EHMC_burnin_as_Rcpp_List$LR_us,
                                                                                                p_jump_us, EHMC_burnin_as_Rcpp_List$adapt_delta_us,
                                                                                                beta1_adam, beta2_adam, eps_adam)
    
                                            EHMC_args_as_Rcpp_List$eps_us <-          min(2.5,   adapt_eps_outs[1] )
                                            EHMC_burnin_as_Rcpp_List$eps_m_adam_us  <- adapt_eps_outs[2]
                                            EHMC_burnin_as_Rcpp_List$eps_v_adam_us <- adapt_eps_outs[3]
                                        }

                        }

                       ## //////////////////   --------------------------------  update tau for main ------------------------------------------------------------------------------
                         
                               if ( (ii < n_adapt) && (manual_tau == FALSE)) {
                                 
                                       tau_ADAM_main_per_chain_prop <- list()
                                       tau_ADAM_main_per_chain <- list()
                                   #    M_dense_sqrt <- as.matrix(forceSymmetric(M_dense_sqrt))
        
                                     try({
                                      for (kk in 1:n_chains_burnin) {
        
                                        tau_ADAM_main_per_chain_prop[[kk]] <- BayesMVP:::fn_Rcpp_wrapper_update_tau_w_diag_M_ADAM(    eigen_vector =  EHMC_burnin_as_Rcpp_List$eigen_vector_main, # shared between chains
                                                                                                                                      eigen_max = EHMC_burnin_as_Rcpp_List$eigen_max_main, # shared between chains
                                                                                                                                      theta_vec_initial = theta_main_0_burnin_tau_adapt_all_chains_input_from_R[, kk], # VARIES between chains
                                                                                                                                      theta_vec_prop = theta_main_prop_burnin_tau_adapt_all_chains_input_from_R[, kk], # VARIES between chains
                                                                                                                                      snaper_m_vec = EHMC_burnin_as_Rcpp_List$snaper_m_vec_main, # shared between chains
                                                                                                                                      velocity_prop = velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R[, kk],  # VARIES between chains
                                                                                                                                      velocity_0 = velocity_main_0_burnin_tau_adapt_all_chains_input_from_R[, kk], # VARIES between chains
                                                                                                                                      tau = EHMC_args_as_Rcpp_List$tau_main, # shared between chains
                                                                                                                                      LR = EHMC_burnin_as_Rcpp_List$LR_main, # shared between chains
                                                                                                                                      ii = ii, # shared between chains
                                                                                                                                      n_burnin = n_adapt, # shared between chains
                                                                                                                                      sqrt_M_vec =  (diag(EHMC_burnin_as_Rcpp_List$M_dense_sqrt)), # shared between chains
                                                                                                                                      tau_m_adam = EHMC_burnin_as_Rcpp_List$tau_m_adam_main, # shared between chains
                                                                                                                                      tau_v_adam = EHMC_burnin_as_Rcpp_List$tau_v_adam_main, # shared between chains
                                                                                                                                      tau_ii = tau_main_ii_vec[kk]) # VARIES between chains
                                                                                                                                    
        
                                          if (!(any(is.nan( tau_ADAM_main_per_chain_prop[[kk]])))) {
                                            tau_ADAM_main_per_chain[[kk]] <- tau_ADAM_main_per_chain_prop[[kk]]
                                          } else {
                                            tau_ADAM_main_per_chain[[kk]][1] <- EHMC_args_as_Rcpp_List$tau_main
                                            tau_ADAM_main_per_chain[[kk]][2] <- EHMC_burnin_as_Rcpp_List$tau_m_adam_main
                                            tau_ADAM_main_per_chain[[kk]][3] <- EHMC_burnin_as_Rcpp_List$tau_v_adam_main
                                          }
        
                                      }
        
                                      tau_ADAM_main_per_chain <- Filter(is_valid, tau_ADAM_main_per_chain)
                                      tau_ADAM_main_avg <-  Reduce("+",  tau_ADAM_main_per_chain) / length( tau_ADAM_main_per_chain)
        
                                      #### update lists:
                                      if  (length(tau_ADAM_main_avg) > 0) {
                                        EHMC_args_as_Rcpp_List$tau_main <- tau_ADAM_main_avg[[1]]  # ifelse( is.nan( tau_ADAM_main_avg[[1]] ), EHMC_args_as_Rcpp_List$tau_main, tau_ADAM_main_avg[[1]])
                                        EHMC_burnin_as_Rcpp_List$tau_m_adam_main <- tau_ADAM_main_avg[[2]] # ifelse( is.nan( tau_ADAM_main_avg[[2]]), EHMC_burnin_as_Rcpp_List$tau_m_adam_main, tau_ADAM_main_avg[[2]])
                                        EHMC_burnin_as_Rcpp_List$tau_v_adam_main <-  tau_ADAM_main_avg[[3]] # ifelse( is.nan( tau_ADAM_main_avg[[3]]), EHMC_burnin_as_Rcpp_List$tau_v_adam_main, tau_ADAM_main_avg[[3]])
                                      }
        
                                     })
                                       
                                       L_main_iter_ii <-  EHMC_args_as_Rcpp_List$tau_main /  EHMC_args_as_Rcpp_List$eps_main 
                                       L_main_during_burnin_vec[ii] <- L_main_iter_ii
                                       
                               }
                                              
        
                               ## //////////////////   --------------------------------  update tau for us ------------------------------------------------------------------------------
                        if (partitioned_HMC == TRUE) {
                               if  ((ii < n_adapt) && (sample_nuisance == TRUE) && (manual_tau == FALSE)) {
                                 
                                        tau_ADAM_us_per_chain_prop <- list()
                                        tau_ADAM_us_per_chain<- list()
        
                                        try({
                                           for (kk in 1:n_chains_burnin) {
        
                                             tau_ADAM_us_per_chain_prop[[kk]] <-    BayesMVP:::fn_Rcpp_wrapper_update_tau_w_diag_M_ADAM(   eigen_vector = EHMC_burnin_as_Rcpp_List$eigen_vector_us,
                                                                                                                                           eigen_max = EHMC_burnin_as_Rcpp_List$eigen_max_us,
                                                                                                                                           theta_vec_initial = theta_us_0_burnin_tau_adapt_all_chains_input_from_R[, kk], ####
                                                                                                                                           theta_vec_prop = theta_us_prop_burnin_tau_adapt_all_chains_input_from_R[, kk], ####
                                                                                                                                           snaper_m_vec = EHMC_burnin_as_Rcpp_List$snaper_m_vec_us,
                                                                                                                                           velocity_prop = velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R[, kk],  ####
                                                                                                                                           velocity_0 = velocity_us_0_burnin_tau_adapt_all_chains_input_from_R[, kk], ####
                                                                                                                                           tau = EHMC_args_as_Rcpp_List$tau_us,
                                                                                                                                           LR = EHMC_burnin_as_Rcpp_List$LR_us,
                                                                                                                                           ii = ii,
                                                                                                                                           n_burnin = n_adapt,
                                                                                                                                           sqrt_M_vec = EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec,
                                                                                                                                           tau_m_adam = EHMC_burnin_as_Rcpp_List$tau_m_adam_us,
                                                                                                                                           tau_v_adam = EHMC_burnin_as_Rcpp_List$tau_v_adam_us,
                                                                                                                                           tau_ii = tau_us_ii_vec[kk])
        
                                             if (!(any(is.nan( tau_ADAM_us_per_chain_prop[[kk]])))) {
                                               tau_ADAM_us_per_chain[[kk]] <- tau_ADAM_us_per_chain_prop[[kk]]
                                             }
        
                                           }
        
        
                                            tau_ADAM_us_per_chain <- Filter(is_valid, tau_ADAM_us_per_chain)
                                            tau_ADAM_us_avg <-  Reduce("+",  tau_ADAM_us_per_chain)/ length( tau_ADAM_us_per_chain)
        
                                            #### update lists:
                                            if  (length(tau_ADAM_us_avg) > 0) {
                                                EHMC_args_as_Rcpp_List$tau_us <- tau_ADAM_us_avg[[1]]
                                                EHMC_burnin_as_Rcpp_List$tau_m_adam_us <- tau_ADAM_us_avg[[2]]
                                                EHMC_burnin_as_Rcpp_List$tau_v_adam_us <- tau_ADAM_us_avg[[3]]
                                            }
        
                                          })
                                        
                                      L_us_iter_ii <-  EHMC_args_as_Rcpp_List$tau_us /  EHMC_args_as_Rcpp_List$eps_us 
                                      L_us_during_burnin_vec[ii] <- L_us_iter_ii
        
                               }
                        }
                            ## //////////////////   --------------------------------  update tau for ALL PARAMS ------------------------------------------------------------------------------
                         if (partitioned_HMC == FALSE) {
                             
                                   # if  ((ii < n_adapt) && (sample_nuisance == TRUE) && (manual_tau == FALSE)) {
                                   #   
                                   #           tau_ADAM_all_per_chain_prop <- list()
                                   #           tau_ADAM_all_per_chain <- list()
                                   #           
                                   #           if (sample_nuisance == TRUE) {
                                   #             
                                   #                 theta_vec_initial_all <-    rbind(theta_us_0_burnin_tau_adapt_all_chains_input_from_R,       theta_main_0_burnin_tau_adapt_all_chains_input_from_R)
                                   #                 theta_vec_prop_all <-       rbind(theta_us_prop_burnin_tau_adapt_all_chains_input_from_R,    theta_main_prop_burnin_tau_adapt_all_chains_input_from_R)
                                   #                 velocity_prop_all <-        rbind(velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R, velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R)
                                   #                 velocity_0_all <-           rbind(velocity_us_0_burnin_tau_adapt_all_chains_input_from_R,    velocity_main_0_burnin_tau_adapt_all_chains_input_from_R)
                                   #               
                                   #           } else if (sample_nuisance == FALSE) { 
                                   #               
                                   #                 theta_vec_initial_all <-    theta_main_0_burnin_tau_adapt_all_chains_input_from_R
                                   #                 theta_vec_prop_all <-       theta_main_prop_burnin_tau_adapt_all_chains_input_from_R
                                   #                 velocity_prop_all <-        velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R
                                   #                 velocity_0_all <-           velocity_main_0_burnin_tau_adapt_all_chains_input_from_R
                                   #               
                                   #           }
                                   #           
                                   #           try({
                                   #                   for (kk in 1:n_chains_burnin) {
                                   #                           
                                   #                           tau_ADAM_all_per_chain_prop[[kk]] <-    BayesMVP:::fn_Rcpp_wrapper_update_tau_w_diag_M_ADAM(    eigen_vector = eigen_vector_all,
                                   #                                                                                                                           eigen_max = eigen_max_all,
                                   #                                                                                                                           theta_vec_initial = theta_vec_initial_all[, kk], ####
                                   #                                                                                                                           theta_vec_prop = theta_vec_prop_all[, kk], ####
                                   #                                                                                                                           snaper_m_vec = snaper_m_vec_all,
                                   #                                                                                                                           velocity_prop = velocity_prop_all[, kk],  ####
                                   #                                                                                                                           velocity_0 = velocity_0_all[, kk], ####
                                   #                                                                                                                           tau = EHMC_args_as_Rcpp_List$tau_main,
                                   #                                                                                                                           LR =  EHMC_burnin_as_Rcpp_List$LR_main,
                                   #                                                                                                                           ii = ii,
                                   #                                                                                                                           n_burnin = n_adapt,
                                   #                                                                                                                           sqrt_M_vec = sqrt_M_vec_all,
                                   #                                                                                                                           tau_m_adam = EHMC_burnin_as_Rcpp_List$tau_m_adam_main,
                                   #                                                                                                                           tau_v_adam = EHMC_burnin_as_Rcpp_List$tau_v_adam_main,
                                   #                                                                                                                           tau_ii = tau_main_ii_vec[kk])
                                   #                           
                                   #                           if (!(any(is.nan( tau_ADAM_all_per_chain_prop[[kk]])))) {
                                   #                             tau_ADAM_all_per_chain[[kk]] <- tau_ADAM_all_per_chain_prop[[kk]]
                                   #                           }
                                   #                     
                                   #                   }
                                   #                   
                                   #                   tau_ADAM_all_per_chain <- Filter(is_valid, tau_ADAM_all_per_chain)
                                   #                   tau_ADAM_all_avg <-  Reduce("+",  tau_ADAM_all_per_chain)/ length( tau_ADAM_all_per_chain)
                                   #                   
                                   #                   #### update lists:
                                   #                   if  (length(tau_ADAM_all_avg) > 0) {
                                   #                     EHMC_args_as_Rcpp_List$tau_main <- tau_ADAM_all_avg[[1]]
                                   #                     EHMC_burnin_as_Rcpp_List$tau_m_adam_main <- tau_ADAM_all_avg[[2]]
                                   #                     EHMC_burnin_as_Rcpp_List$tau_v_adam_main <- tau_ADAM_all_avg[[3]]
                                   #                   }
                                   #             
                                   #           })
                                   #           
                                   #           L_main_iter_ii <-  EHMC_args_as_Rcpp_List$tau_main /  EHMC_args_as_Rcpp_List$eps_main
                                   #           L_main_during_burnin_vec[ii] <- L_main_iter_ii
                                   #           
                                   # }
                             
                           }
                                      
                           if (ii %% 25 == 0) {
      
                                        if (partitioned_HMC == TRUE) { # if NOT sampling all parameters at once
                                                  cat(colourise(    (paste("p_jump_main = ", round(p_jump_main, 3)))          , "green"), "\n")
                                                  cat(colourise(    (paste("eps_main = ", round(EHMC_args_as_Rcpp_List$eps_main, 3)))          , "blue"), "\n")
                                                  comment(print(paste("tau_main = ",  round(EHMC_args_as_Rcpp_List$tau_main, 3))))
                                                  comment(print(paste("L_main = ",  round(ceiling(EHMC_args_as_Rcpp_List$tau_main/EHMC_args_as_Rcpp_List$eps_main), 0))))
                                                  cat(colourise(    (paste("div_main = ", sum(div_main)))          , "red"), "\n")
                                        } else {
                                                  cat(colourise(    (paste("p_jump = ", round(p_jump_main, 3)))          , "green"), "\n")
                                                  cat(colourise(    (paste("eps = ", round(EHMC_args_as_Rcpp_List$eps_main, 3)))          , "blue"), "\n")
                                                  comment(print(paste("tau = ",  round(EHMC_args_as_Rcpp_List$tau_main, 3))))
                                                  comment(print(paste("L = ",  round(ceiling(EHMC_args_as_Rcpp_List$tau_main/EHMC_args_as_Rcpp_List$eps_main), 0))))
                                                  cat(colourise(    (paste("div = ", sum(div_main)))          , "red"), "\n")
                                        }


                                        if (partitioned_HMC == TRUE) { # if NOT sampling all parameters at once
                                            if     (sample_nuisance == TRUE)   {
                                                cat(colourise(    (paste("p_jump_us = ", round(p_jump_us, 3)))          , "green"), "\n")
                                                cat(colourise(    (paste("eps_us = ", round(EHMC_args_as_Rcpp_List$eps_us, 3)))          , "blue"), "\n")
                                                comment(print(paste("tau_us = ",  round(EHMC_args_as_Rcpp_List$tau_us, 3))))
                                                comment(print(paste("L_us = ",  round(ceiling(EHMC_args_as_Rcpp_List$tau_us / EHMC_args_as_Rcpp_List$eps_us), 0))))
                                                cat(colourise(    (paste("div_us = ", sum(div_us)))          , "red"), "\n")
                                            }
                                        }
                                      
                           }
                                    
                            if (ii == n_burnin) {

                                    try({
                                        print(tictoc::toc(log = TRUE))
                                        log.txt <- tictoc::tic.log(format = TRUE)
                                        tictoc::tic.clearlog()
                                        time_burnin <- unlist(log.txt)
                                        ##
                                        extract_numeric_string <- stringr::str_extract(time_burnin, "\\d+\\.\\d+")   
                                        time_burnin <- as.numeric(extract_numeric_string)
                                    })
                              
                            } 


    }




 }
 
    L_main_during_burnin <- mean(L_main_during_burnin_vec, na.rm = TRUE)
    L_us_during_burnin <-   mean(L_us_during_burnin_vec, na.rm = TRUE)
  
    out <- list( n_chains_burnin = n_chains_burnin,
                  n_burnin = n_burnin,
                  time_burnin = time_burnin,
                  eps_main =  EHMC_args_as_Rcpp_List$eps_main,
                  tau_main =  EHMC_args_as_Rcpp_List$tau_main,
                  eps_us =  EHMC_args_as_Rcpp_List$eps_us,
                  tau_us =  EHMC_args_as_Rcpp_List$tau_us,
                  L_main_during_burnin_vec = L_main_during_burnin_vec,
                  L_us_during_burnin_vec = L_us_during_burnin_vec,
                  L_main_during_burnin = L_main_during_burnin,
                  L_us_during_burnin = L_us_during_burnin,
                  theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                  theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R,
                  EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                  EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                  EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List)
    
    return(out)
  
  
}
  


              
   









