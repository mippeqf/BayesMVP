# # |  |  | -------------  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



BayesMVP_sample <- function(seed,
                            Model_type,
                            y, 
                            X,
                            Phi_approx = "Phi",
                            exp_fast,
                            log_fast,
                            save_individual_log_lik = FALSE,
                            LOO = FALSE,
                            forked = FALSE, # only on linux or MAC and no progress bar available 
                            autodiff. = TRUE,
                            num_chunks = 1,
                            metric_type = "Hessian",
                            use_eigen_main_only = TRUE,
                            use_old_eigen_main_method = TRUE ,
                            n_chains = parallel::detectCores(logical = FALSE), 
                            n_chains_burnin = NULL, 
                            n_burnin = 125, 
                            n_post_adapt_burnin = 25,
                            n_iter = 1000,
                            learning_rate_main = 0.10,
                            adapt_delta = 0.80, 
                            clip_iter = n_burnin * 0.2, 
                            adapt_interval_width = 25,
                            corr_param,
                            prior_only = FALSE,
                            corr_force_positive,
                            corr_pos_offset,
                            prior_mean_vec,
                            prior_sd_vec,
                            lkj_cholesky_eta,
                            corr_prior_beta = FALSE,
                            corr_prior_normal = FALSE,
                            prior_for_corr_a,
                            prior_for_corr_b,
                            known_values_indicator_list = NULL,
                            known_values_list = NULL,
                            ub_corr =NULL,
                            lb_corr = NULL,
                            LT_b_priors_shape = NULL,
                            LT_b_priors_scale = NULL,
                            LT_known_bs_indicator = NULL,
                            LT_known_bs_values = NULL,
                            prev_prior_a = 1.5,
                            prev_prior_b = 1.5,
                            partial_float = FALSE, 
                            NT_us = TRUE, 
                            lb_phi_approx = -5,
                            ub_phi_approx = +5, 
                            rough_approx = FALSE,  
                            Phi_exact_indicator_if_not_using_rough_approx = TRUE, 
                            HMC = FALSE, 
                            Euclidean_M_main = TRUE, 
                            dense_G_indicator = FALSE, 
                            smooth_M_main = FALSE, 
                            kappa_val = 8,
                            fix_rho = FALSE, 
                            override_Euclidean_main = TRUE, 
                            learning_rate_us = learning_rate_main, 
                            max_eps = 0.50,
                            max_depth = 10,
                            adapt_M_main = TRUE,
                            soft_abs = FALSE,
                            soft_abs_alpha = NA, 
                            eps = 1, 
                            tau_main = 1, 
                            main_L_manual = FALSE, 
                            L_main_if_manual = 1, 
                            diag_metric = TRUE, 
                            MALA_main = FALSE, 
                            max_depth_main = 10, 
                            tau_main_jittered = TRUE, ## 
                            u_Euclidean_metric_const = TRUE, ## 
                            smooth_M_us = FALSE, ##
                            adapt_M_us = TRUE,  ## 
                            main_eps_manual = FALSE, 
                            L_main = 1, 
                            override_Euclidean_us = TRUE  
                            ) {
                            
 
  
  Phi_type = Phi_approx
  
  n_cores_for_WCP = 1
     #   sampling_option = 11 # MD - original
        sampling_option = 100 # MD - new
  #   sampling_option = 10 # AD
   
    # tanh_option = 1
  tanh_option = 4
  #tanh_option = 5
   
 
   
  {
    
    
    Euclidean_M_main = TRUE
    
    if (is.null(n_chains_burnin)) {
    
      n_chains_burnin <-   min(8, parallel::detectCores(logical = FALSE))
      
    }
    

      
 
    colourise <- function(text, fg = "black", bg = NULL) {
      term <- Sys.getenv()["TERM"]
      colour_terms <- c("xterm-color","xterm-256color", "screen", "screen-256color")
      
      if(rcmd_running() || !any(term %in% colour_terms, na.rm = TRUE)) {
        return(text)
      }
      
      col_escape <- function(col) {
        paste0("\033[", col, "m")
      }
      
      col <- .fg_colours[tolower(fg)]
      if (!is.null(bg)) {
        col <- paste0(col, .bg_colours[tolower(bg)], sep = ";")
      }
      
      init <- col_escape(col)
      reset <- col_escape("0")
      paste0(init, text, reset)
    }
    
    .fg_colours <- c(
      "black" = "0;30",
      "blue" = "0;34",
      "green" = "0;32",
      "cyan" = "0;36",
      "red" = "0;31",
      "purple" = "0;35",
      "brown" = "0;33",
      "light gray" = "0;37",
      "dark gray" = "1;30",
      "light blue" = "1;34",
      "light green" = "1;32",
      "light cyan" = "1;36",
      "light red" = "1;31",
      "light purple" = "1;35",
      "yellow" = "1;33",
      "white" = "1;37"
    )
    
    .bg_colours <- c(
      "black" = "40",
      "red" = "41",
      "green" = "42",
      "brown" = "43",
      "blue" = "44",
      "purple" = "45",
      "cyan" = "46",
      "light gray" = "47"
    )
    
    rcmd_running <- function() {
      nchar(Sys.getenv('R_TESTS')) != 0
    }
  }
  
  
                            
 

{
  options(scipen = 99999)
  options(max.print = 1000000)
  options(mc.cores = parallel::detectCores())
  numerical_diff_e = 0.001
}








# |   -------------   initialise -----------------------------------------------------


{
  
  tau_main  <- 1
  
  kk <- 1
  
  n_params_main <- (n_class - 1)  + n_class * choose(n_tests, 2) + n_class * (n_covariates + 1)  * n_tests 
  n_us <- 1 * N * n_tests 
  n_params <- n_params_main + n_us 
  
  n_corrs <- n_class * 0.5 * n_tests * (n_tests - 1)
  n_coeffs <- n_class * n_tests * 1
  
  index_us <- 1:n_us
  index_main <- (n_us+1):n_params
  
       beta1_adam <- 0 ;  beta2_adam <-   0.95 ;  eps_adam <- 10^(-8)   # "long  run" settings
        # beta1_adam <- 0 ;  beta2_adam <-   0.50 ;  eps_adam <- 10^(-3) # "short run" settings
  
  accept <- c()
  snaper_w <- rep(0.01, n_params)
  snaper_s <- rep(0.01, n_params)
  snaper_m <- rep(0.01, n_params)
 #  tau <- 1
  eps_m_adam <-  eps_v_adam <- 1
  tau_m_adam <- tau_v_adam <- 1
  h_bar <- eps
  dense_M <- F
  
  eigen_max <- sqrt(sum(snaper_w^2))
  eigen_vector <- snaper_w/eigen_max # same as z 
  
  
  eigen_max_main <- sqrt(sum(snaper_w[index_main]^2))
  eigen_vector_main <- snaper_w[index_main]/eigen_max_main # same as z 
  
  error_count <- 0 
  L_main_vec <- c()
  p_jump_main_vec <- c()
  
  L_main_ii <- 1
  
  eps_us <- eps <- eps_ii <- 1
  
  index_subset <- 1:n_params
  
  min_ess_sec_vec <- c()
  min_ess_grad_vec <- min_ess_grad_sampling_vec <- c()
  
  
  m_sq <- v_sq <- cov_sq <- 1
  m_sq_us <- v_sq_us <- cov_sq_us <- 1
  
  L_us_ii = 1
  
  velocity  <- rep(0.01, n_params)
  velocity_prop <-   rep(0.01, n_params)
  velocity_0 <-   rep(0.01, n_params)

  
  p_jump_us = 1
  
  eps_m_adam_us <- eps_v_adam_us <-  eps_us
  
 
 # momentum <- momentum_0
#   grad_prop <- grad
  # 
  
  eps_ii_vec <- c()
  
  tau_m_adam_us <- 1
  tau_v_adam_us <- 0
  
  tau_m_adam_main <- 1
  tau_v_adam_main <- 0
  

  
  
  adapt_delta <-  adapt_delta
  
  
   trace_theta_test <- array(dim  = c(1 + n_burnin, n_params_main))
   trace_theta_test_all_chains <- array(dim  = c(n_chains_burnin, 1 + n_burnin, n_params_main))
   
   if  (Model_type == "LT_LC") {
     corr_force_positive = TRUE
   }
     
   theta_vec <-     rep(0.01, n_params)
  if (corr_force_positive == TRUE)  theta_vec[(n_us + 1):(n_us+n_corrs)] <- rnorm(n = n_corrs, mean = -3, sd = 0.00001)
  else    theta_vec[(n_us + 1):(n_us+n_corrs)] <- rep(0.01, n_corrs)
  
  theta_vec[   (n_us + n_corrs + 1):(n_us + n_corrs + n_coeffs/2)  ] <- rep(-1, n_coeffs/2)
  theta_vec[(n_us + n_corrs + 1 + n_coeffs/2):(n_us + n_corrs + n_coeffs)] <- rep(1, n_coeffs/2)
  #  log_posterior_iter_0 <- log_posterior[kk]
   
  theta_vec[n_params] =  -0.6931472  # this is equiv to starting val of p = 0.20!  since: 0.5 * (tanh( -0.6931472) + 1)  = -0.6931472
  
  
  
   # theta_vec <-  theta_vec
  theta_vec_prop <-  theta_vec
  theta_vec_initial <- theta_vec
  #theta_vec_inc_p <- theta_vec
  
  p_jump_vec <- c()
  # theta_mean <- theta_vec
  # theta_var <- rep(0.0001, n_params)
  
 
  
  # if (test_manifold_stuff == TRUE) 
  n_leapfrog_vec <- c()
  
  window_index <- 0 
  window_index_us <- 0
  
  M_inv_diag_vec <- rep(1, n_params)  # initial
  M_diag_vec <- rep(1, n_params)
  M_inv_diag_vec_stored <-  rep(1, n_params)
  

  
  u_trace_index <- 0
  
  window_index_us <- 0
  
  
  
  window_index_main <- 0 
  window_index_main_2 <- 0 
  
 
  
  L_main_vec <- c()
  L_us_vec <- c()
  
  error_count <- 0 
  
  eps_m_adam <-  eps
  eps_v_adam <- 0
  
  p_jump_us_vec <- c()
  p_jump_main_vec <- c()
  

  
}

 



  
  
  
  {
    
    
    
    
    
  # set lp_and_grad function arguments needed and put them in a list, so that they can be passed on to the generic EHMC-ADAM functions
  exclude_priors = FALSE
  lkj_prior_method = 2
  grad_main = TRUE
  grad_nuisance = TRUE
  CI = CI
 
   
  homog_corr = homog_corr
  lkj_cholesky = mvr_cholesky 
  
  
 
  
  prior_coeffs_mean = t(prior_mean_vec)
  prior_coeffs_sd = t(prior_sd_vec)
  n_class = n_class
  n_tests = n_tests
 
  
  N_obs_threshold_for_using_vectorised_fn = 100000
  N_obs = N * n_tests
 
  
  
  vectorised =  TRUE # with chunking (much faster and much better scaling !!)
 
  list_prior_for_corr_a <-  list_prior_for_corr_b <- list()

 
  
  for (c in 1:n_class) {
      list_prior_for_corr_a[[c]] <- prior_for_corr_a[,,c]
      list_prior_for_corr_b[[c]] <- prior_for_corr_b[,,c]
  }
  
  
 
            if (is.null(ub_corr)) {                                                                                                                          
                            ub_corr <- list()                                                                                                                            
                            for (c in 1:n_class) {                                                                                                                       
                                  ub_corr[[c]] <- diag(n_tests)                                                                                                            
                                  ub_corr[[c]][, ] <- +1                                                                                                                   
                              }                                                                                                                                            
            }
  
            if (is.null(lb_corr)) {                                                                                                                          
                  lb_corr <- list()                                                                                                                            
                  for (c in 1:n_class) {                                                                                                                       
                        lb_corr[[c]] <- diag(n_tests)                                                                                                            
                        lb_corr[[c]][, ] <- -1                                                                                                                   
                    }                                                                                                                                            
            }
  
            for (c in 1:n_class) {                                                                                                                           
                  if (corr_force_positive == TRUE)                                                                                                             
                        lb_corr[[c]][, ] <- 0                                                                                                                    
                    else lb_corr[[c]][, ] <- -1                                                                                                                  
            }
  
  
  
            if (Model_type == "LT_LC")   corr_param <- "latent_trait"
 
  
            if (corr_param == "Arch") {                                                                                                                      
                    if (prior_only == TRUE)       model_number <- 9                                                                                                                        
                    else model_number <- 4                                                                                                                       
           }     else if (corr_param == "Chol_Nump") {                                                                                                            
                    if (prior_only == TRUE)         model_number <- 6                                                                                                                        
                    else model_number <- 8                                                                                                                       
            }     else if (corr_param == "Chol_Stan") {                                                                                                            
                    if (prior_only == TRUE)       model_number <- 3                                                                                                                        
                    else model_number <- 1                                                                                                                       
            }    else if (corr_param == "Chol_Schur") {                                                                                                           
                    if (prior_only == TRUE)          model_number <- 10                                                                                                                       
                    else model_number <-     11 #  11                                                                                                                     
            }   else if (corr_param == "latent_trait") {                                                                                                         
                  model_number <- 12                                                                                                                           
            }
  
  
           #  model_number <- 20 # for testing / debug
            # 
            model_number <- 11 # Schur, AD (depending on pkg - check)
  
 
    
            
            if (is.null(known_values_indicator_list)) {                                                                                                      
                  known_values_indicator_list <- list()                                                                                                        
                  known_values_list <- list()                                                                                                                  
                  for (c in 1:n_class) {                                                                                                                       
                        known_values_indicator_list[[c]] <- diag(n_tests)                                                                                        
                        known_values_list[[c]] <- diag(n_tests)                                                                                                  
                  }      
            }
   
            
            if (is.null(LT_b_priors_shape)) {  
              LT_b_priors_shape <- array(1, dim = c(n_class, n_tests))
            }
            if (is.null(LT_b_priors_scale)) {  
              LT_b_priors_scale <- array(1, dim = c(n_class, n_tests))
            }
 
                  
            if (is.null(LT_known_bs_indicator)) {  
              LT_known_bs_indicator <- array(0, dim = c(n_class, n_tests))
            }
            if (is.null(LT_known_bs_values)) {  
              LT_known_bs_values    <- array(0.00001, dim = c(n_class, n_tests))
            }
 
            
            
            known_corr_index_vec <- rep(0, n_corrs)
            
            if (corr_param != "latent_trait") {
                      at_least_one_corr_known_indicator <- 0 
                      counter <- 1
                      corr_index = n_us + 1
                      for (c in 1:n_class) {
                        for (i in 2:n_tests) {
                          for (j in 1:(i-1)) {
                            if (known_values_indicator_list[[c]][i, j] == 1)   {
                              known_corr_index_vec[counter] <- corr_index
                              at_least_one_corr_known_indicator <- 1
                            }
                            corr_index <- corr_index + 1
                            counter <- counter + 1
                          }
                        }
                      }
            } else { 
                at_least_one_corr_known_indicator <- 0 
                    counter <- 1
                    corr_index = n_us + 1
                    for (c in 1:n_class) {
                      for (i in 1:n_tests) {
                          if (LT_known_bs_indicator[c, i] == 1)   {
                            known_corr_index_vec[counter] <- corr_index
                            at_least_one_corr_known_indicator <- 1
                          }
                          corr_index <- corr_index + 1
                          counter <- counter + 1
                      }
                    }
                    
                    known_corr_index_vec <- (n_us + 1 + n_class*n_tests):(n_us + n_corrs) # [(counter:n_corrs)] 
                    
            }
            
            
            
            index_subset_excl_known <- index_main
            for (i_known in 1:length(known_corr_index_vec)) {
              index_subset_excl_known <-   index_subset_excl_known[index_subset_excl_known != known_corr_index_vec[i_known]]
            }
 
 
            
            N = nrow(y)
            min_LC_N_estimate_class_1 <-   N -  round( sum(y)   / n_tests, 0) ; min_LC_N_estimate_class_1
            min_LC_N_estimate_class_2 <-   N -  min_LC_N_estimate_class_1 ; min_LC_N_estimate_class_2
            
            min_LC_N_estimate_class_min <- min(min_LC_N_estimate_class_1, min_LC_N_estimate_class_2) ; min_LC_N_estimate_class_min
            
            X_list <- list(X)
            
            

            
            n_cores = 1
            corr_prior_norm = FALSE
            
            
            {
                other_lp_and_grad_args <- list()
                other_lp_and_grad_args[[1]] = n_cores
                other_lp_and_grad_args[[2]] = exclude_priors
                other_lp_and_grad_args[[3]] = CI
                other_lp_and_grad_args[[4]] = lkj_cholesky_eta
                other_lp_and_grad_args[[5]] = prior_coeffs_mean
                other_lp_and_grad_args[[6]] = prior_coeffs_sd
                other_lp_and_grad_args[[7]] = n_class
                other_lp_and_grad_args[[8]] = ub_phi_approx
                other_lp_and_grad_args[[9]] = num_chunks
                other_lp_and_grad_args[[10]] = corr_force_positive
                other_lp_and_grad_args[[11]] = list_prior_for_corr_a
                other_lp_and_grad_args[[12]] = list_prior_for_corr_b
                other_lp_and_grad_args[[13]] = corr_prior_beta
                other_lp_and_grad_args[[14]] = corr_prior_norm
                other_lp_and_grad_args[[15]] = lb_corr
                other_lp_and_grad_args[[16]] = ub_corr
                other_lp_and_grad_args[[17]] = known_values_indicator_list
                other_lp_and_grad_args[[18]] = known_values_list
                other_lp_and_grad_args[[19]] = prev_prior_a
                other_lp_and_grad_args[[20]] = prev_prior_b
                other_lp_and_grad_args[[21]] = exp_fast
                other_lp_and_grad_args[[22]] = log_fast
                other_lp_and_grad_args[[23]] = Phi_approx
                other_lp_and_grad_args[[24]] = LT_b_priors_shape
                other_lp_and_grad_args[[25]] = LT_b_priors_scale
                other_lp_and_grad_args[[26]] = LT_known_bs_indicator
                other_lp_and_grad_args[[27]] = LT_known_bs_values
            }
       
             
               
            
             
            
            if (Model_type == "MVP_LC")       fn =  BayesMVPv2::fn_lp_and_grad_MVP_using_Chol_Spinkney_MD_and_AD  
            if (Model_type == "LT_LC")        fn =  BayesMVPv2::fn_lp_and_grad_latent_trait_MD_and_AD  
             
            
           # fn = fn_lp_and_grad_latent_trait_MD_and_AD  #####
 
              
            outs_manual_grad <-  fn(theta_main = theta_vec[index_main],
                                    theta_us = theta_vec[index_us],
                                   y = y, 
                                   X = X_list, 
                                   other_args = other_lp_and_grad_args)
                                   
            
            log_posterior <- outs_manual_grad[1]     
            log_posterior_initial <- log_posterior
            log_posterior                                                                                                                                    
            grad <- head(outs_manual_grad[-1], n_params)  ;   tail(grad, 50) 
            individual_log_lik = outs_manual_grad[(n_params + 2):((n_params + 1 + N))]
            
            head(grad, 50) ;  tail(grad, 50)
            


            
  
  }

  
  
 
  # -------------------------------------------------------------------------

  if (is.null(adapt_interval_width)) {

        {
          interval <- 1
 
          if (n_burnin < 301) {
            adapt_interval_width <- n_burnin / 10
          } else { 
            adapt_interval_width <- 25
          }
          
          
         #  adapt_interval_width <- n_burnin
          
            seq_burnin <- 1:n_burnin
            partitions <- split(seq_burnin, ceiling(seq_along(seq_burnin)/adapt_interval_width))
            n_adapt_intervals <-  length(partitions)  ; n_adapt_intervals
        
        }
    
  } else { 
    
    {
      interval <- 1
      
      seq_burnin <- 1:n_burnin
      partitions <- split(seq_burnin, ceiling(seq_along(seq_burnin)/adapt_interval_width))
      n_adapt_intervals <-  length(partitions)  ; n_adapt_intervals
      
    }
    
    
  }
  


{
  
  
  
  tictoc::tic("burnin timer")
  
  
  options(mc.cores = parallel::detectCores(logical = FALSE))
 
  


  
  
  {
    
    
    
              
              
              {
                try({
                  stopCluster(cl)
                }, silent = TRUE)
                
              
                if (forked == TRUE) cl <- parallel::makeForkCluster(n_chains, outfile="") # FORK only works with linux
                else     cl <- parallel::makeCluster(n_chains, outfile="") # FORK only works with linux
                
                doParallel::registerDoParallel(cl)
                
              }
              
              {
                
                
                
                comb <- function(x, ...) {
                  lapply(seq_along(x),
                         function(i) c(x[[i]], lapply(list(...), function(y) y[[i]])))
                }
                
                #   doRNG::registerDoRNG(seed = seed)
                
                
                # seed <- seed_manual  ##################################
                
                set.seed(seed)
                doRNG::registerDoRNG(seed = seed)
              }
              
    
    
      }
  
  
  
  
 
  
for (interval in 1:n_adapt_intervals) {   

  gc(reset = TRUE)
  
{



  
  # run in parallel
  
  set.seed(seed)
  
  non_BCA_burnin_outs <- doRNG::`%dorng%`(
    foreach::foreach(kk = 1:n_chains_burnin, 
                     .packages = c( # "Rcpp",
                                  # "rstan",
                                  # "lcmMVPbetav2",
                                  # "lcmMVPbetav3",
                                 # "float",
                                   "Matrix" 
                                 #  "matrixStats",
                                 #  "LaplacesDemon", 
                                 #  "bdsmatrix"
                                 ), 
                     .combine = 'comb', 
                     .multicombine = TRUE,
                     .init = list(list(),   list(),  list(),  list(), list(),
                                  list(),   list(),  list() , list(), list(),
                                  list(),   list(),  list(),  list(), list(),  
                                  list(),   list(),  list(),  list(), list(), 
                                  list(),   list(),  list(),  list(), list(),
                                  list(),   list(),  list(),  list(), list(), 
                                  list(),   list(),  list(),  list(), list(), 
                                  list(),   list(),  list(),  list(), list(), 
                                  list())
    ),
    {
      
      
      options(mc.cores = parallel::detectCores())
      
 
 
      # |   -------------   start of iteration loop  -----------------------------------------------------
      
      
      
 
      ii <- 1
      partition_width <- length(partitions[[interval]])
     
      
      
    for (ii in  partitions[[interval]][1]:partitions[[interval]][partition_width] ) {
          
          if  ((ii ==  partitions[[interval]][1]) && ii != 1) {
                  # reset initial values
                 theta_vec <- theta_initial_values_per_chain[, kk]  
                 theta_vec_initial   <- theta_vec
               #  grad <-   grad_initial_values_per_chain[, kk]  
                 log_posterior <- log_posterior_initial_values_per_chain[kk]  
                 log_posterior_initial <- log_posterior
                 velocity_0 <- velocity
          }
          
          

             
          { 
            
            

            
          }
     
            
             
             # gc(reset = TRUE)
        
             eta_w <- 3 
             
             kappa <- 8
             eta_m <- 1/(ceiling(ii/kappa)+1)
 
              
              # | -----------------   adapt M (for u's) ---------------------------------------------------------------------------------------
              
           
 
                adapt_window_apply_prop <- c(0.2, 0.3, 0.4, 0.6, 0.8) # when to apply new metric 
                adapt_window_apply <-  round(adapt_window_apply_prop * n_burnin, 0) ; adapt_window_apply  # when to apply new metric  
              
               ###  adapt_M_us = FALSE
                
 
              if   (  (adapt_M_us == TRUE) &&  (override_Euclidean_us == TRUE)  )   {
                {
                  
                  
                  try({
 
                          if  (  (ii %in% tail(  partitions[[interval]], 1) )  && (ii >= n_burnin*0.2 )  )  { 
                              M_inv_diag_vec_stored[index_us] <-   snaper_s[index_us]  # using "online" variance estimate
                          }
                      
                      if (ii %in% adapt_window_apply[c(-1)]) { 
                        
                        pct_done_notification_interval_chains  = rbinom( n = n_chains, size = 1, prob = 4 * (1/n_chains) )
                        
                        if (kk %in%  c( c(1:n_chains) * pct_done_notification_interval_chains ) )  {
                           message(cat(colourise( paste("Adapting Metric (M) for NUISANCE parameters " ), "green"), "\n"))
                        }
                        
                        # if (NT_us == FALSE) {
                          M_inv_diag_vec[index_us] <- M_inv_diag_vec_stored[index_us]
                          M_diag_vec[index_us] <-  1 / M_inv_diag_vec[index_us]
                          
                          if (u_Euclidean_metric_const == TRUE) {
                            M_diag_vec[index_us] <-  rep(median(M_diag_vec[index_us]), n_us)
                            M_inv_diag_vec[index_us] <- 1 / M_diag_vec[index_us]
                          }
                      }
                    
                  })
                }
              }
              
              
                
              
              
              # | -----------------   adapt M (for main - if EUCLIDEAN metric) ---------------------------------------------------------------------------------------
              
              if (ii < 2) { 
                
                M_dense_main <-  diag(rep(1, n_params_main))
                M_inv_dense_main <-   diag(rep(1, n_params_main)) 
                M_inv_dense_main_stored <- M_inv_dense_main
                M_dense_CHOL_main <- BayesMVP::Rcpp_Chol(((M_dense_main)))
                M_dense_main_corr_mtx <- diag(rep(1, n_params_main))
                M_dense_sqrt <-  expm::sqrtm(M_dense_main)
                M_inv_dense_main_chol <- BayesMVP::Rcpp_Chol(((M_inv_dense_main)))
                
              }
              
              
              {
 
                adapt_window_apply_prop <- c(0.2, 0.3, 0.4, 0.6, 0.8) # when to apply new metric 
                adapt_window_apply <-  round(adapt_window_apply_prop * n_burnin, 0) ; adapt_window_apply  # when to apply new metric  
                
            
                
                if   (  (adapt_M_main == TRUE)  && (metric_type == "Empirical"))  {
                  
                
                  try({

                    if (smooth_M_main == FALSE) {

                                 if  (  (ii %in% tail(  partitions[[interval]], 1) )  && (ii >= n_burnin*0.2 )  )  {

                                   
                                
                                   
                                        adapt_window_sequence_main <-    partitions[[interval]] # (adapt_window_apply[window_index_main] + 1):(adapt_window_apply[window_index_main+1]-1)

                                      ##   M_inv_diag_vec <- 1 / M_diag_vec
                                        M_inv_diag_vec_current_interval <- M_inv_diag_vec_stored

                                      
                                        M_inv_diag_vec_current_interval[index_main] <- apply( trace_theta_test_all_chains[kk, adapt_window_sequence_main, ] , c(2), sd, na.rm = TRUE)  
                                        M_inv_diag_vec_current_interval[index_main] <-      M_inv_diag_vec_current_interval[index_main]   *      M_inv_diag_vec_current_interval[index_main]  
                                        # tictoc::tic("timer 999")
                                        # tictoc::toc()
                                        
                                        

                                      #  if (dense_G_indicator == TRUE) {
                                     

                                          try({

                                          if (ii < clip_iter) {
                                               M_inv_dense_main_current_interval <-  diag(M_inv_diag_vec_current_interval[index_main]) # 
                                          } else {
                                       
                                            if ((adapt_window_sequence_main[1] - (n_burnin/10)) > 1) {
                                              try({
                                           
                                               M_inv_dense_main_current_interval <- cov(  trace_theta_test_all_chains[kk,  (adapt_window_sequence_main[1] - (n_burnin/10)):(tail(adapt_window_sequence_main, 1)),  ] ,
                                                                                          use = "complete" )
                                           
                                              }, silent = TRUE)
                                            } else {
                                            try({
                                               M_inv_dense_main_current_interval <-   diag(M_inv_diag_vec_current_interval[index_main])  
                                            }, silent = TRUE)
                                           }

                                          }
                                          })
                                
                                        
                                     
                                     #   }

                                        if  (  tail(M_inv_diag_vec_stored, 1) != 1  ) {
                                         #   weight_current <- 0.50
                                            weight_current <- 0.60
                                          # weight_current <- 0.75#
                                            M_inv_diag_vec_stored[index_main] <- (1 - weight_current)*M_inv_diag_vec_stored[index_main] + weight_current*M_inv_diag_vec_current_interval[index_main]
                                            M_inv_dense_main_stored <-           (1 - weight_current)*M_inv_dense_main_stored           + weight_current*M_inv_dense_main_current_interval
                                        } else { ##  first adaptation stage
                                          M_inv_diag_vec_stored[index_main] <- M_inv_diag_vec_current_interval[index_main]
                                          M_inv_dense_main_stored  <- M_inv_dense_main_current_interval
                                        }


                                        M_dense_main <- BayesMVP::Rcpp_solve(M_inv_dense_main)
                                        M_inv_dense_main_chol <- BayesMVP::Rcpp_Chol(((M_inv_dense_main)))

                                   


                                 }

                                     if (ii %in% adapt_window_apply[c(-1)]) {

                                       pct_done_notification_interval_chains  = rbinom( n = n_chains, size = 1, prob = 4 * (1/n_chains) )
                                       
                                       if (kk %in%  c( c(1:n_chains) * pct_done_notification_interval_chains ) )  {
                                           message(cat(colourise( paste("Adapting Metric (M) for MAIN parameters " ), "green"), "\n"))
                                       }

                                           M_inv_diag_vec[index_main] <- M_inv_diag_vec_stored[index_main]
                                           M_diag_vec[index_main] <-  1 / M_inv_diag_vec[index_main]

                                           if (dense_G_indicator == TRUE) {

                                        
                                                    M_inv_dense_main <-    M_inv_dense_main_stored
                                                    M_inv_dense_main_chol <- BayesMVP::Rcpp_Chol(((M_inv_dense_main)))

                                                 if (matrixcalc::is.positive.definite(M_inv_dense_main) == FALSE) {
                                                   M_inv_dense_main <- as.matrix(nearPD(M_inv_dense_main, keepDiag = TRUE)$mat)
                                                   M_inv_dense_main <-  as.matrix(forceSymmetric(  M_inv_dense_main ))
                                                   M_inv_dense_main_chol <- BayesMVP::Rcpp_Chol(((M_inv_dense_main)))
                                                 }
                                                
                                                 try({

                                                     M_dense_main <- BayesMVP::Rcpp_solve(M_inv_dense_main)
                                                     M_dense_main <- as.matrix(forceSymmetric(  M_dense_main ))
                                                     M_dense_main_corr_mtx <- cov2cor(M_dense_main)

                                                       #  if (ii ==  (adapt_window_apply[c(-1)][1] + 1) )  {
                                     
                                                            M_dense_sqrt <-  expm::sqrtm(M_dense_main)
                                                      
                                                       #  }

                                                 })
                                                   

                                           }
                                     }

                    } else {

                        if (ii < n_burnin) {
                          if (ii >  25) {
                                M_inv_diag_vec[index_main] <- snaper_s[index_main]  # using "online" variance estimate
                                M_diag_vec[index_main] <-  1 / M_inv_diag_vec[index_main]

                                M_inv_diag_vec_stored = M_inv_diag_vec
                          }
                        }

                   }



                  })
                  
                  
                }
              }
              
 
          
          # | -----------------------------------   PROPOSAL FOR  MAIN (or ALL) parameters-----------------------------------------------------------------------------------------------------------------------------------------
          
 
          # 
          # #
          try({
            
            
            
            # find initial epsilon
            if (ii == 1)  {
                        
                               initial_eps_attempt_success <- FALSE
# 
                               try({
                                eps <- BayesMVPv2::R_fn_EHMC_find_initial_eps(Model_type. = Model_type,
                                                                            theta. = theta_vec,
                                                                  indiactor_mixed_M. = FALSE,
                                                                  HMC. = HMC,
                                                                  index_subset. = index_subset,
                                                                  index_subset_main_dense. = index_main,
                                                                  y. = y,
                                                                  X. = X_list,
                                                                  L. = 1,
                                                                  eps. = eps,
                                                                  velocity_0. = velocity,
                                                                  momentum_0. = velocity_0,
                                                                  testing. = FALSE,
                                                                  M_diag_vec. = M_diag_vec,
                                                                  M_main_dense. = diag(n_params_main),
                                                                  M_inv_dense_main.  = diag(n_params_main),
                                                                  M_Chol_main_dense. = diag(n_params_main),
                                                                  grad_x_initial. = grad,
                                                                  log_posterior_initial. = log_posterior,
                                                                  n_us. = n_us,
                                                                  n_params_main. = n_params_main,
                                                                  other_lp_and_grad_args. = other_lp_and_grad_args)
                                
 
                                  

                                initial_eps_attempt_success <- TRUE
                                
                                
                                })
    


            }
            
            
            
            
            
            {
              
              
              
            
               tau_mult <- 1.6
               slow_tau = FALSE
              #  slow_tau = TRUE

              # gap <- round(n_burnin * 0.25, 0) ; gap
              # if (gap - clip_iter < 25) { gap = round(min(n_burnin/2, 2 * clip_iter), 0)  } 
 
              if (n_burnin < 300) gap <-  clip_iter  + round(n_burnin / 5)
              else                gap <-  clip_iter  + round(n_burnin / 10)
              
              if (MALA_main == TRUE) {
                  L_main <- L_main_ii <-  1 ; tau_main <- eps
              } else {
                
                if (ii < clip_iter) {       # clip first phase of burn-in to MALA
                  
                                  L_main_ii <- 1
                                  L_main <- 1 ###
                                  tau_main <- eps
                                  tau_main_ii <- tau_main
                                
                                  tau_target = tau_main
                      
                } else {
                  
                  
                         
                              
                           if (ii %in% c(clip_iter:gap)) { 
                                    
    
                               
                                         if (min_LC_N_estimate_class_min < 400) {
                                                  if (ii <  round((clip_iter + gap) / 4) )   { 
                                                      L_main <- 2 
                                                  } else if (ii %in% c( round((clip_iter + gap) / 4):round((clip_iter + gap) / 2)  )) {
                                                      L_main <- 4 
                                                  } else if (ii %in% c( round((clip_iter + gap) / 2):round((clip_iter + gap) * 0.75 )  )) {
                                                      L_main <- 6
                                                  } else { 
                                                      L_main <-    round(  0.75 *  tau_mult *   sqrt(eigen_max_main) /  eps ) #  round( round(  0.75 *  tau_mult *   sqrt(eigen_max_main) /  eps ) + 6 ) / 2
                                                  }
                                         } else if (min_LC_N_estimate_class_min %in% c(401:2000)) { 
                                                 if (ii <  round((clip_iter + gap) / 4) )   { 
                                                   L_main <- 4 
                                                 } else if (ii %in% c( round((clip_iter + gap) / 4):round((clip_iter + gap) / 2)  )) {
                                                   L_main <- 8 
                                                 } else if (ii %in% c( round((clip_iter + gap) / 2):round((clip_iter + gap) * 0.75 )  )) {
                                                   L_main <- 16
                                                 } else { 
                                                   L_main <-   round(  0.75 *  tau_mult *   sqrt(eigen_max_main) /  eps )
                                                 }
                                         } else { 
                                                 if (ii <  round((clip_iter + gap) / 4) )   { 
                                                   L_main <- 4 
                                                 } else if (ii %in% c( round((clip_iter + gap) / 4):round((clip_iter + gap) / 2)  )) {
                                                   L_main <- 12 
                                                 } else if (ii %in% c( round((clip_iter + gap) / 2):round((clip_iter + gap) * 0.75 )  )) {
                                                   L_main <- 36
                                                 } else { 
                                                   L_main <-      round(  0.75 *  tau_mult *   sqrt(eigen_max_main) /  eps )
                                                 }
                                         }
                             
                  
                                                 
                                                 tau_main <- eps * L_main 
                                                 tau_main_ii  <- runif(n = 1, min = 0, max = 2 * tau_main)
                                                 L_main_ii <-   ceiling(tau_main_ii / eps)
                             
                                                 tau_target = tau_main
                             
                           } else if (ii > gap) { 
                              
            



                                          if (slow_tau == TRUE) {
                                  
                                                   if (ii %in% c( gap:n_burnin ))  {
                                                     
                                                     last_slow_iter_prop <- 0.85
                                                     
                                                     if (ii %in% c( c(gap:(gap+1)), 
                                                                    c(round( 0.30 * n_burnin, 0) )  , 
                                                                    c(round( 0.35 * n_burnin, 0) )   , 
                                                                    c(round( 0.45 * n_burnin, 0) )   , 
                                                                    c(round( last_slow_iter_prop * n_burnin, 0) )   ,
                                                                  #  c(round( 0.65 * n_burnin, 0) )   ,
                                                                  #  c(round( 0.75 * n_burnin, 0) )   ,
                                                                    0
                                                                    )  )     {
                                                       
                                                                        if (use_eigen_main_only == TRUE)  {
                                                                          tau_target =  tau_mult *   sqrt(eigen_max_main)
                                                                          L_main <-   ceiling(tau_main / eps)
                                                                        } else {
                                                                          #  tau_main =  tau_mult *   sqrt(eigen_max)
                                                                          tau_target =  tau_mult *   sqrt(eigen_max_main)
                                                                          L_main <-   ceiling(tau_main / eps)
                                                                        }
                                                        }
            
                                                     
                                                       if (ii < round( last_slow_iter_prop * n_burnin, 0)) {
                                                              tau_target_interval = seq(from = tau_target/2, to = tau_target, length =  round( last_slow_iter_prop * n_burnin, 0) + 1)
                                                              tau_main = tau_target_interval[ii]
                                                       } else { 
                                                              tau_main = tau_target
                                                       }
                                                     
                                                              
                                                              
                                                              tau_main_ii  <- runif(n = 1, min = 0, max = 2 * tau_main)
                                                              L_main <-   ceiling(tau_main / eps)
                                                              L_main_ii <-   ceiling(tau_main_ii / eps)
            
                                                      }
 
                                          }  else if (slow_tau == FALSE)  {  
                                            
                                            
                                            
                                               ##   if (ii %in% c( c(gap:(gap+1)), c(round( 0.40 * n_burnin, 0) )  , c(round( 0.50 * n_burnin, 0) ) , c(round( 0.75 * n_burnin, 0) )  )  )   {
                                            
                                            if (ii %in% c( c(gap:(gap+1)), 
                                                           # c(round( 0.40 * n_burnin, 0) ) `£$%"^` , 
                                                           c(round( 0.35 * n_burnin, 0) )   , 
                                                           c(round( 0.45 * n_burnin, 0) )   , 
                                                           c(round( 0.55 * n_burnin, 0) )   ,
                                                           c(round( 0.65 * n_burnin, 0) )   ,
                                                           c(round( 0.75 * n_burnin, 0) )   ,
                                                           0
                                            )  )     {
                                              
                                                             #  if (ii %in% c( c(gap:(gap+1)), c(round( 0.30 * n_burnin, 0) )  , c(round( 0.40 * n_burnin, 0) )   , c(round( 0.45 * n_burnin, 0)   )  )  )     {
                                                            ### print(paste("| ---------- resetting tau ------------ |"))
                                                                              if (use_eigen_main_only == TRUE)  {
                                                                                 tau_main =  tau_mult *   sqrt(eigen_max_main)
                                                                              } else {
                                                                                 #  tau_main =  tau_mult *   sqrt(eigen_max)
                                                                                 tau_main =  tau_mult *   sqrt(eigen_max_main)
                                                                              }
                                                             
                                                            ### print(paste("| ----------  tau_main = ", round(tau_main, 1), " ------------ |  "))
                                                             
                                                 }
                                            
                                                  tau_target = tau_main
                                            
                                            
                                                  tau_main_ii  <- runif(n = 1, min = 0, max = 2 * tau_main)
                                                  L_main <-   ceiling(tau_main / eps)
                                                  L_main_ii <-   ceiling(tau_main_ii / eps)
                                            
                                          }
                                            
                           }
                                          
 
                  
                            tau_main_ii  <- runif(n = 1, min = 0, max = 2 * tau_main)
                            L_main <-   ceiling(tau_main / eps)
                            L_main_ii <-   ceiling(tau_main_ii / eps)
                            
                  
                  
                              
                              if  (  (main_L_manual == TRUE) )  {
                                            L_main <-    L_main_if_manual #  ceiling(tau_main / eps)
                                            tau_main <- eps * L_main 
             
                              }  else { 
                                            L_main <-   ceiling(tau_main / eps)
                              }
                              
                              if (tau_main < eps) { 
                                     tau_main_ii <- eps
                                     tau_main    <- eps 
                                     L_main_ii <- 1
                                     L_main  <- 1
                              } else { 
                                    tau_main_ii  <- runif(n = 1, min = 0, max = 2 * tau_main)
                                    L_main <-   ceiling(tau_main / eps)
                                    L_main_ii <-   ceiling(tau_main_ii / eps)
                              }
                              
                  if (tau_main > 100) { 
                     tau_main = 100
                     tau_main_ii = 100
                     L_main_ii <- ceiling(tau_main / eps)
                     L_main  <- ceiling(tau_main / eps)
                  }
                  
                              
                }
                

              }
              
              
              if (main_eps_manual == TRUE) {
                eps_main  <- eps_main_if_manual
                eps_main_ii <- eps_main
              }
              
              
              
              
              
            }
            
            
          })
          
          
          {
            
 
            try({
              if (eps > max_eps) {
                eps_ii <- eps <- max_eps
              } else {
                eps_ii <- eps
              }
            })
            
            
            
            rhmc_outs_main  <-    NA
            
          }
          
          
          try({
            
            if (L_main_ii > 2^max_depth) {
              #  break
              L_main_ii <- 2^max_depth
              L_main <- 2^max_depth
              tau_main <- L_main * eps
            }
            L_main_vec[ii] <- L_main_ii
            
          })
          
          
          
          
          grad_main = TRUE
          grad_nuisance = TRUE
          
 
          
          EHMC_single_iter_R_fn_output_main <- NA
          
          
          try({
 
            
 
              
                    largest_var <-   1  # max(1 / M_diag_vec)
                    M_diag_vec_adj <-  M_diag_vec * largest_var
                    sqrt_adj_vars <- sqrt(M_diag_vec_adj[index_main])
                    
                   
        
                        num =  25

                
                      start_adapting_Hessian <-  0   #### bookmark
 
                 
           if (metric_type == "Hessian") {
               if  (  (ii >   round(start_adapting_Hessian, 0)) && (adapt_M_main == TRUE)  ) {
 
                      
                      try({  
                                         if   ((ii %in% c( start_adapting_Hessian:(n_burnin - start_adapting_Hessian))) && 
                                               (ii %% num  == 0 )  && 
                                               ( ii < round(n_burnin * 0.90)) ) 
                                           {
                                     #   if (ii %in%  c(adapt_window_apply[c(-1)] )) {
                                              
                                              # 
                                       
                                           
                                         ###  print(index_main)
                                           #
                                               if (ii > (n_burnin/20)) {
                                                  theta_vec_us <- head(snaper_m, n_us)
                                                  theta_vec_main <- snaper_m[index_main]
                                               } else { 
                                                #    theta_vec_us <- head(theta_vec, n_us)
                                                  theta_vec_us <- theta_vec[index_us]
                                                  theta_vec_main <- theta_vec[index_main]
                                               }
                                           
   
                                      
                                      
                                             if (ii > ( n_burnin * 0.50 )) {
                                               try({
                                                # for (iiiii in 1:n_params_main) {
                                                #    theta_vec_main[iiiii] <- median(  trace_theta_test_all_chains[kk, ( round(n_burnin * 0.40)):(ii-1), iiiii], na.rm = T )
                                                # }
                                                 theta_vec_main <-  Rfast::colMedians( trace_theta_test_all_chains[kk, ( round(n_burnin * 0.40)):(ii-1), ] , na.rm = TRUE)
                                                 trace_theta_test[( round(n_burnin * 0.40)):(ii-1),  ] =  trace_theta_test_all_chains[kk, ( round(n_burnin * 0.40)):(ii-1),  ]
                                               })
                                             }
                      
                                           
                 
                                              theta_vec_main_inc <- theta_vec_main
                                              Hessian <- Hessian_prop <-  array(0, dim = c(n_params_main, n_params_main))
 
                                              
                                              
                                              outs_manual_grad <-  fn(theta_main = theta_vec_main_inc,
                                                                      theta_us = theta_vec_us, 
                                                                      y = y, 
                                                                      X = X_list,
                                                                      other_args  = other_lp_and_grad_args)
                                              
                                              
                                              grad <- -  head(outs_manual_grad[-1], n_params)
                                              numerical_diff_e = 0.001
                                              if (N < 2500)  {
                                                numerical_diff_e <- 0.01
                                              }
                                              
                                              for (i_param in 1:n_params_main) {
                                                
                                                theta_vec_main_inc[i_param] <-        theta_vec_main[i_param] + numerical_diff_e
    
                                                outs_manual_grad <-  fn(theta_main = theta_vec_main_inc,
                                                                        theta_us = theta_vec_us, 
                                                                        y = y, 
                                                                        X = X_list,
                                                                        other_args  = other_lp_and_grad_args)
 
                                                
                                                
                                                
                                                grad_numdiff <-   - head(outs_manual_grad[-1], n_params)
                                                
                                                Hessian_prop[i_param, ] <- ( grad_numdiff[index_main] - grad[index_main] ) /  numerical_diff_e
                                                
                                                theta_vec_main_inc = theta_vec_main # reset
                                                
                                              }
                                   
                                              
                                          
                                              Hessian <-  M_dense_main
                                              
                                              try({ 
                                                  
                                                  if  (  (!(any(is.na(Hessian_prop))))  &&   (matrixcalc::is.positive.definite( as.matrix(  Matrix::forceSymmetric(as.matrix(Hessian_prop)))) ) )    { 
                                                    Hessian <- as.matrix(Matrix::forceSymmetric(Hessian_prop)) 
                                                  } else { 
                                                    Hessian <-      as.matrix(nearPD(Hessian_prop)$mat) #  as.matrix(Matrix::forceSymmetric(M_dense_main)) 
                                                  }
                                              
                                              })
 
                                                Chol_neg_Hessian <- NULL
                                              
                                              try({
                                                Chol_neg_Hessian <- BayesMVP::Rcpp_Chol((Hessian))
                                              }, silent = TRUE)

                                              if (is.null(Chol_neg_Hessian)) {
                                                pd_indicator <- 0
                                              } else {
                                                pd_indicator <- 1
                                              }
                                              #

                                              try({
                                                if (pd_indicator == 0) {
                                                  Hessian <-  as.matrix(nearPD(Hessian)$mat)
                                                  Hessian <- as.matrix(Matrix::forceSymmetric(Hessian))
                                                }
                                              },silent = TRUE)

                                                #  hess_adapt_weight <-  0.50
                                                # hess_adapt_weight <-  0.75
                                                 if (ii > 0.666 * n_burnin)  { 
                                                        hess_adapt_weight <-    1  
                                                   } else    { 
                                                        hess_adapt_weight <-   0.50
                                                   }
                                                
                                                 if (!(any(is.na(Hessian))))  M_dense_main <- (1-hess_adapt_weight)*M_dense_main + hess_adapt_weight*Hessian

                                              Chol_neg_Hessian <- NULL

                                              try({
                                                Chol_neg_Hessian <- BayesMVP::Rcpp_Chol((M_dense_main))
                                              }, silent = TRUE)

                                              if (is.null(Chol_neg_Hessian)) {
                                                pd_indicator <- 0
                                              } else {
                                                pd_indicator <- 1
                                              }
                                              #

                                              try({
                                                if (pd_indicator == 0) {
                                                  M_dense_main <-  as.matrix(nearPD(M_dense_main)$mat)
                                                  M_dense_main <- as.matrix(Matrix::forceSymmetric(M_dense_main))
                                                }
                                              },silent = TRUE)
                                              # 
                                                
                                                
                                                
                                              ##   M_dense_main <- diag(diag(M_dense_main))
                                                
                                               #  print(M_dense_main)
                                                
                                              try({
                                              M_inv_dense_main <-    BayesMVP::Rcpp_solve(M_dense_main)
                                              M_inv_dense_main_chol <- BayesMVP::Rcpp_Chol((M_inv_dense_main))
                                              })
                                                
             
                                                
                                              M_dense_main_adj <-  M_dense_main # diag(sqrt_adj_vars) %*% M_dense_main_corr_mtx %*% diag(sqrt_adj_vars)
                                              M_inv_dense_main_adj <-   M_inv_dense_main
                                              
                                              # 
                                              # seq(from = 1, to = n_burnin, by = 25)
                                              #  seq(from = 1, to = n_burnin, by = 50)
                                              # 
                                              # start_dense_metric_prop* n_burnin
                                              
                                               start_dense_metric_prop <- 0 # dense only
                       
                                               # 
                                              if ( (ii > start_dense_metric_prop* n_burnin) ) {
                                                M_dense_main_adj <-  M_dense_main # diag(sqrt_adj_vars) %*% M_dense_main_corr_mtx %*% diag(sqrt_adj_vars)
                                                M_inv_dense_main_adj <-    M_inv_dense_main  #  BayesMVP::Rcpp_solve(M_dense_main_adj)
                                              } else {
                                                M_dense_main <- diag(diag(M_dense_main))
                                                M_inv_dense_main <- diag(diag(M_inv_dense_main))
                                                
                                                M_inv_dense_main_chol <- BayesMVP::Rcpp_Chol((M_inv_dense_main))
                                                M_dense_main_adj <-  M_dense_main
                                                M_inv_dense_main_adj <-   M_inv_dense_main
                                              }
                                              
                                           
                                         }  else { 
                                           
                                               M_dense_main_adj <-  M_dense_main # diag(sqrt_adj_vars) %*% M_dense_main_corr_mtx %*% diag(sqrt_adj_vars)
                                               M_inv_dense_main_adj <-   M_inv_dense_main
                                               M_inv_dense_main_chol <- BayesMVP::Rcpp_Chol((M_inv_dense_main_adj))
                                           
                                         }
                                            
                      })
                    }
                 
               }   else { 
                   M_dense_main_adj <-  M_dense_main
                   M_inv_dense_main_adj <-   M_inv_dense_main 
                   M_diag_vec <- M_diag_vec
                   M_inv_dense_main_chol <- BayesMVP::Rcpp_Chol(M_inv_dense_main)
               }
                    
                    
                 
                 
               })
          
          

            
            if (main_L_manual == TRUE) {
            
                L_main <- L_main_if_manual
                L_main_ii <-  runif(n = 1, min = 1, max = 2*L_main)
            
            } 
            
          
          
           ##  L_main_ii <- ifelse(L_main_ii < 1, 1, L_main_ii)
            L_main_vec[ii] <- L_main_ii
            

            n_attempts = 1


            div = 1;
 
                            try({
                              
                              known_corr_index_vec - n_us
# 
                              if  (Model_type == "LT_LC") {
                                    # M_dense_main[    known_corr_index_vec - n_us ,     known_corr_index_vec - n_us  ]  <- 0
                                    # M_inv_dense_main[   known_corr_index_vec - n_us ,    known_corr_index_vec - n_us  ]  <- 0
                                    # M_inv_dense_main_chol[   known_corr_index_vec - n_us ,    known_corr_index_vec - n_us  ]  <- 0
                                    # 
                                    # diag(M_dense_main[ known_corr_index_vec - n_us ,  known_corr_index_vec - n_us  ])  <-  rep(1,   length(known_corr_index_vec - n_us )  )
                                    # diag(M_inv_dense_main[ known_corr_index_vec - n_us ,  known_corr_index_vec - n_us ])  <-  rep(1,   length(known_corr_index_vec - n_us )  )
                                    # diag(M_inv_dense_main_chol[ known_corr_index_vec - n_us , known_corr_index_vec - n_us  ])  <- rep(1,   length(known_corr_index_vec - n_us )  )
                                    
                                    M_dense_main = diag(diag(M_dense_main))
                                    M_inv_dense_main = diag(diag(M_inv_dense_main))
                                    M_inv_dense_main_chol = diag(diag(M_inv_dense_main_chol))
                              }

                         
                              
                          rhmc_outs_main =    BayesMVPv2::Rcpp_fn_sampling_single_iter_burnin(                    Model_type = Model_type,
                                                                                                                  theta_main_array = theta_vec[index_main],
                                                                                                                  theta_us_array = theta_vec[index_us],
                                                                                                                  y = y,
                                                                                                                  X = X_list,
                                                                                                                  other_args = other_lp_and_grad_args,
                                                                                                                  L = L_main_ii,
                                                                                                                  eps = eps,
                                                                                                                  log_posterior = log_posterior,
                                                                                                                  M_inv_us_array = (1 / M_diag_vec[index_us]), 
                                                                                                                  M_dense_main  = M_dense_main,
                                                                                                                  M_inv_dense_main = M_inv_dense_main,
                                                                                                                  M_inv_dense_main_chol  = M_inv_dense_main_chol)
                                                                                                                  
                              

                              div = 0
 
                            })
              
 

                    
                #    if (  (any(is.na(rhmc_outs_main[, 2]))) || any(is.infinite(rhmc_outs_main[, 2])) )   div = 1
         
                    if   (    div   ==  1   )   {
                      
                                          p_jump  <-  0
                                          accept  <-  0
                                          
                                          print(paste("ERROR - MAIN PARAMS - CHAIN - DIV", kk))
                                          error_count <- error_count + 1
                                          
                                          log_posterior  <- log_posterior_initial
                                          log_posterior_prop  <- log_posterior_initial
                                          
                                          theta_vec <- theta_vec_initial
                                          theta_vec_prop <-  theta_vec_initial
                                          
                                          velocity  <- velocity_0
                                          velocity_prop  <- velocity_0
                                          
                                          next
                                          
                    } else {
                            
                                          p_jump  <- rhmc_outs_main[5, 1]
                                          accept  <- rhmc_outs_main[6, 1]
                                          
                                          log_posterior  <- rhmc_outs_main[1, 1]
                                          log_posterior_initial  <- rhmc_outs_main[2, 1]
                                          log_posterior_prop  <- rhmc_outs_main[3, 1]
                                          
                                          div  <- rhmc_outs_main[4, 1]
                                           if (div == 1) {
                                             
                                             print(paste("ERROR - MAIN PARAMS - CHAIN - DIV", kk))
                                             error_count <- error_count + 1
                                           }
                                          theta_vec  <- rhmc_outs_main[, 2]
                                          theta_vec_initial  <-  rhmc_outs_main[, 3]
                                          theta_vec_prop  <- rhmc_outs_main[, 4]
                                          
                                          velocity  <- rhmc_outs_main[, 5]
                                          velocity_0  <- rhmc_outs_main[, 6]
                                          velocity_prop  <- rhmc_outs_main[, 7]
        
                    }
                    
     
          
          
          #  # | -------------------------  Tau Adaptation - for MAIN (or ALL) parameters ------------------------------------------------------------------------

        #    try({
          
            # start_adapting_tau <- gap
            # start_adapting_tau <- clip_iter
             start_adapting_tau <- 1
            
          if  ( (ii > (round(  start_adapting_tau  , 0))) ) {
          if   ( (ii < n_burnin)  ) {
            
          
              if (MALA_main == FALSE) {
                  
                #  if (ii < 5) theta_mean_across_chains <- theta_vec
                  
                  if (ii < 2) {
                    snaper_m <- theta_vec
                  } else { 
                    snaper_m <- (1-eta_m)*snaper_m + eta_m * theta_vec
                  }
 
                  snaper_s <- (1-eta_m)*snaper_s + eta_m* ( theta_vec  - snaper_m)^2   # update s (and M, if smooth) 
 
                  sqrt_M_pos <-   sqrt(M_diag_vec)
                  sqrt_M_prop <-   sqrt_M_pos
 
 
                index_subset =  index_main
 
               # if (at_least_one_corr_known_indicator == 1) { 
                  index_subset = index_subset_excl_known
                  index_subset_known <-   index_main[is.na(pmatch(index_main, index_subset_excl_known ))]
                  theta_vec_initial[index_subset_known] <- rnorm(n = length(index_subset_known))
                  theta_vec[index_subset_known] <- rnorm(n = length(index_subset_known))
                  theta_vec_prop[index_subset_known] <- rnorm(n = length(index_subset_known))
                  velocity_prop[index_subset_known] <- rnorm(n = length(index_subset_known))
                  velocity_0[index_subset_known] <- rnorm(n = length(index_subset_known))
                  velocity[index_subset_known] <- rnorm(n = length(index_subset_known))
               # }
 
                
                
              
                M_chol <-  diag(sqrt(M_diag_vec[index_main]))

                
                
                try({  
                  
                  x_c <-  sqrt_M_pos * ( theta_vec - snaper_m) 
                  
              
                  
                  
                  {
                    
                 
                    eigen_max_prop <- sqrt(sum(snaper_w^2))
                    eigen_max_main_prop  <- sqrt(sum(snaper_w[index_main]^2))
                    eigen_max_us_prop  <- sqrt(sum(snaper_w[index_us]^2))
                    ###   eigen_vector_main  <- snaper_w[index_main]/eigen_max_main  
                    
                    if (!any(is.na(eigen_max_prop))) eigen_max  <- eigen_max_prop
                    if (!any(is.na(eigen_max_main_prop))) eigen_max_main <- eigen_max_main_prop
                    if (!any(is.na(eigen_max_us_prop))) eigen_max_us  <- eigen_max_us_prop
                    
                    if (use_eigen_main_only == TRUE)  {
                      index_subset =  index_main
                      if ( use_old_eigen_main_method == FALSE)   eigen_vector  <- snaper_w/eigen_max_main  
                      if ( use_old_eigen_main_method == TRUE )   eigen_vector  <- snaper_w/eigen_max  # old  (wrong but works well?!)
                    } else { 
                      index_subset =  c(index_us, index_main)
                      
                       eigen_vector_prop  <- snaper_w/eigen_max  
                      
                      if (!any(is.na(eigen_vector_prop))) eigen_vector <- eigen_vector_prop
                      
                    }
                    
                  }
                  
 
                  
                   if (ii > eta_w & eigen_max > 0) {
                     
                             
                                  
                                  if (dense_G_indicator == TRUE)  {
                                   #   M_chol <- BayesMVP::Rcpp_Chol(((M_dense_main))
                                 
                                    if ( (ii > 0.5 * n_burnin) ) {
                                      #   M_chol <-   diag(sqrt(M_diag_vec[index_main])) # M_dense_sqrt  # expm::sqrtm(M_dense_main)
                                        M_chol <-    M_dense_sqrt  # expm::sqrtm(M_dense_main)
                                    } else { 
                                       M_chol <-  diag(sqrt(M_diag_vec[index_main]))
                                    }
                                    
                                     x_c[index_main] <-  c(M_chol %*% ( theta_vec - snaper_m)[index_main])
                                  }
                                  current_w <- x_c * sum( x_c * eigen_vector) 
                                  snaper_w <- snaper_w * ((ii-eta_w)/(ii+1)) +  ((eta_w+1)/(ii+1)) * current_w
                                  

                                  

                                  
                      

                   } else { 
                                 snaper_w <- x_c
                   }
                  
                  
                  

                  
                  
                })
                
                
                
                if (use_eigen_main_only == TRUE)  {
                  index_subset =  index_main
                } else { 
                  index_subset =  c(index_us, index_main)
                }
                
                
 
              dense_G_indicator = FALSE
                try({    
                  
                  # update tau
                    pos_c_array  <- eigen_vector  * sqrt_M_pos * (theta_vec_initial - snaper_m)  
                    if (dense_G_indicator == TRUE)   pos_c_array[index_main]  <-  c(eigen_vector[index_main] %*% M_chol %*% (theta_vec_initial - snaper_m)[index_main])
                    pos_c_array <-  ifelse(is.infinite(pos_c_array ), 0,    pos_c_array  )  
                    pos_c_per_chain <- sum( pos_c_array[index_subset]  , na.rm = T  )   #   = sqrt( phi(x_{0})  )
                    
                    prop_c_array  <- eigen_vector * sqrt_M_prop * (theta_vec_prop - snaper_m)  
                    if (dense_G_indicator == TRUE)   prop_c_array[index_main]  <-  c(eigen_vector[index_main] %*% M_chol %*% (theta_vec_prop - snaper_m)[index_main])
                    prop_c_array <-  ifelse(is.infinite(prop_c_array), 0,    prop_c_array  ) 
                    prop_c_per_chain <- sum( prop_c_array[index_subset]  , na.rm = T  )    # = sqrt(  phi(x_{tau})  )
                    
                    diff_sq <-       ((prop_c_per_chain)^2-(pos_c_per_chain)^2) # = phi(x_{tau})  - phi(x_{0})
                    m_sq_per_chain <-  (pos_c_per_chain^2+prop_c_per_chain^2)/2 # (1-eta_m)*m_sq_per_chain + eta_m*(pos_c_per_chain[kk]^2+prop_c_per_chain[kk]^2)/2
                    
                    pos_c_grad_per_chain  <-   2 *   pos_c_per_chain * sqrt_M_pos * eigen_vector  # = grad_phi(X_{0})
                    prop_c_grad_per_chain <-   2 *   prop_c_per_chain * sqrt_M_prop * eigen_vector  # = grad_phi(X_{tau})
                    if (dense_G_indicator == TRUE) {
                      pos_c_grad_per_chain[index_main] <-    2 *   c(pos_c_per_chain[index_main]  %*% M_chol %*% eigen_vector[index_main])  # = grad_phi(X_{0})
                      prop_c_grad_per_chain[index_main] <-   2 *   c(prop_c_per_chain[index_main] %*% M_chol %*% eigen_vector[index_main])  # = grad_phi(X_{tau})
                    }
 
                    if (ii > eta_w & eigen_max > 0) {
                      
                              v_sq_per_chain <-    (0.5*((pos_c_per_chain^2-m_sq)^2+(prop_c_per_chain^2-m_sq)^2))  
                              cov_sq_per_chain <-    ((pos_c_per_chain^2-m_sq)*(prop_c_per_chain^2-m_sq))
                              
                              m_sq  <- m_sq * (1 - eta_m) +     (m_sq_per_chain) * eta_m
                              v_sq <- v_sq * ((ii-eta_w)/(ii+1)) +     (v_sq_per_chain) * ((eta_w+1)/(ii+1))
                              cov_sq <- cov_sq * ((ii-eta_w)/(ii+1)) +     (cov_sq_per_chain) * ((eta_w+1)/(ii+1))
                              
                              
                              if (fix_rho == TRUE) {
                                rho <-  1
                              } else {
                                rho <-  cov_sq   / v_sq
                              }
                              
                    }
                })
                dense_G_indicator = TRUE


                    if (ii > eta_w & eigen_max > 0) {
                               tau_noisy_grad_val = diff_sq * ( sum(prop_c_grad_per_chain[index_subset]  * velocity_prop[index_subset] )  + 
                                                                sum(pos_c_grad_per_chain[index_subset]   * velocity_0[index_subset]  ) )  - 
                                                    (0.5*(1+min(1, rho))/(tau_main_ii))*(diff_sq)^2 
                    
                    tau_noisy_grad <- ifelse(is.na(tau_noisy_grad_val), 0, tau_noisy_grad_val)
                    
                    # if ( ii %% 20 == 0) {
                    #   print(paste("mean rho = ", round(mean(rho), 3)))
                    # }
                    
                    tau_m_adam <- beta1_adam*tau_m_adam + (1-beta1_adam)*tau_noisy_grad
                    tau_v_adam <- beta2_adam*tau_v_adam + (1-beta2_adam)*tau_noisy_grad^2
                    tau_m_hat <-  tau_m_adam/(1-beta1_adam^ii)
                    tau_v_hat <-  tau_v_adam/(1-beta2_adam^ii)
                    # note: actual LR will be ~ LR at start of burnin and decrease to 0.05*LR at end of burnin (i.e. the LR will smoothly decrease as iterations decrease)
                    # current_alpha <- learning_rate_main  * (1 - (((1 - learning_rate_main)*ii)/n_burnin.) )
                    
                  #   current_alpha <- learning_rate_main  * (1 - 0.95*ii/n_burnin)
                    current_alpha <- learning_rate_main *(  1-(1 - learning_rate_main)*ii/(n_burnin) )
                    log_tau_val <- log(abs(tau_main)) + current_alpha*tau_m_hat/(sqrt(tau_v_hat) + eps_adam)
                    log_tau <- ifelse(is.na(log_tau_val), log_tau, log_tau_val)
                     if (main_L_manual == FALSE) tau_main <- exp(log_tau)
                    }
                    
                    
                    index_subset = index_main
                    
              }
                    
 
              
              
           # })
            
          }
          } else {
            theta_mean_across_chains <- theta_vec
           # tau_main <-  L_main_if_manual * eps
            L_main <- L_main_if_manual
            L_main_ii <- L_main_if_manual
          }   
 
            
            
  
          
          try({ 
            {
              p_jump_vec[ii] <- p_jump
 
              
              theta_mean <-  snaper_m
              theta_var <-  snaper_s
              
              try({
              #   trace_theta_test[kk, ii, ] <-   theta_vec[index_main] # for main params store all post-burnin iterations
                trace_theta_test[ii, ] <-   theta_vec[index_main] # for main params store all post-burnin iterations
              })
              
              
            }
          })
          
          # | ------------  Adaptation of epsilon  using ADAM ---------------------------------------------------------------
          
          try({ 
                    if (main_eps_manual == FALSE) {
                      
                      
                      adapt_delta_val <- adapt_delta
                            # 
                            # 
                            if ( (adapt_delta < 0.95) && ( (min_LC_N_estimate_class_min < 400) ) && (Phi_type == 1)  ) {
                                  if (ii < 0.333 * n_burnin)  adapt_delta_val <- adapt_delta + 0.15
                                  if (ii %in% c( round(0.333 * n_burnin):round(0.50 * n_burnin)))   adapt_delta_val <- adapt_delta + 0.125
                                  if (ii %in% c( round(0.50  * n_burnin):round(0.60 * n_burnin)))   adapt_delta_val <- adapt_delta + 0.10
                                  if (ii %in% c( round(0.60  * n_burnin):round(0.70 * n_burnin)))   adapt_delta_val <- adapt_delta + 0.075
                                  if (ii %in% c( round(0.70  * n_burnin):round(0.80 * n_burnin)))   adapt_delta_val <- adapt_delta + 0.05
                                  if (ii %in% c( round(0.80  * n_burnin):round(0.90 * n_burnin)))   adapt_delta_val <- adapt_delta + 0.025
                                  if (ii %in% c( round(0.90  * n_burnin):n_burnin))   adapt_delta_val <- adapt_delta
                            } else {
                                   adapt_delta_val <- adapt_delta
                            }
                            # 
                            # 
                      
                      
                            adapt_eps_outs <-   BayesMVP::R_fn_adapt_eps_ADAM(eps. = eps,
                                                                     eps_m_adam. = eps_m_adam,
                                                                     eps_v_adam. = eps_v_adam,
                                                                     iter. = ii,
                                                                     n_burnin. = n_burnin,
                                                                     learning_rate. = learning_rate_main, 
                                                                     p_jump. = p_jump,
                                                                     adapt_delta. = adapt_delta_val,
                                                                     beta1_adam. = beta1_adam,
                                                                     beta2_adam. = beta2_adam,
                                                                     eps_adam. = eps_adam,
                                                                     L_manual. = main_L_manual,
                                                                     L_ii. = L_main_ii)
                            
                            eps <- adapt_eps_outs$eps
                            eps_m_adam <- adapt_eps_outs$eps_m_adam
                            eps_v_adam <- adapt_eps_outs$eps_v_adam
                      
                    }
                    
                    
                    
                    p_jump_main_vec[ii] <- p_jump
                    
                    
                    eps_mean_main <- eps
            
          })
          
          try({
                  
            
             pct_done_notification_interval_chains  = rbinom( n = n_chains, size = 1, prob = 10 * (1/n_chains) )
              
             if (kk %in%  c( c(1:n_chains) * pct_done_notification_interval_chains ) )  {
               
                      if (ii %% round(n_burnin/10, 0) == 0) { 
                        
                        message(  paste( round( (ii/ (n_iter + n_burnin)) * 100, 1), "% done - chain", kk)  )
           
                      }
                      
                      if (ii %% round(n_burnin/5) == 0) { 
                        
                       ###  print(paste("mean acceptance prob (main) = ", round(mean(p_jump_main_vec), 2)))
                        
                        print(paste("tau = ", round(tau_main, 2)))
                        print(paste("eps = ", round(eps, 4)))
                        print(paste("L_mean (main) = ", ceiling(tau_main / eps )) )
                        print(paste("L_main_ii (main) = ", round(L_main_ii, 0)))
                        
                        print(paste("L_target (if using main only)  = ",   round(tau_mult * sqrt(eigen_max_main ) / eps , 2) ) )
                        print(paste("L_target (if using all params) = ",   round(tau_mult * sqrt(eigen_max      ) / eps , 2) ) )
                        
                        
                      }
               
             }
                  
          })
          
 
          
 
          
        }   # | ------------     end of iterations (ii loop) -----------------------------------------------------------       
      
      #W#     output_list <- list()
      
      try({   L_main_mean <- mean(L_main_vec, na.rm = TRUE)  })
      try({   L_main_mean_samp <- mean(L_main_vec[(n_burnin+1):(n_iter + n_burnin)] )  })
 
      try({ 
        return(list(trace_theta_test = trace_theta_test, 
                    eps_mean_main =  eps, # eps_mean_main,
                    L_main_mean =  L_main_mean,
                    1,
                    time_burnin = 0, # 5
                    time_sampling = 0,
                    L_main_mean_samp,
                    trace_theta_test_full = 0 ,
                    diag_vec_G = 0,
                    tau_main = tau_main, # 10 
                    theta_vec = theta_vec, 
                    A_dense = 0, 
                    R_diag = 0,
                    Empirical_Fisher_mat = 0,
                    G_dense_main = 0, # 15
                    L_main = L_main,
                    M_diag_vec = M_diag_vec, 
                    snaper_m = snaper_m,  #####
                    snaper_s = snaper_s, 
                    snaper_w = snaper_w, # 20
                    m_sq = m_sq, 
                    v_sq = v_sq, 
                    cov_sq = cov_sq, 
                    tau_m_adam = tau_m_adam, 
                    tau_v_adam = tau_v_adam,  # 25
                    theta_vec_initial = 0, 
                    grad = 0, 
                    log_posterior = log_posterior, 
                    M_inv_diag_vec_stored = M_inv_diag_vec_stored,
                    M_inv_dense_main_stored = M_inv_dense_main_stored, # 30
                    M_dense_main = M_dense_main,
                    M_inv_dense_main = M_inv_dense_main,
                    M_dense_main_corr_mtx = 0, 
                    Empirical_Fisher_mat_adj_to_use = 0,
                    A_dense_adj_to_use = 0, # 35
                    R_diag_adj_to_use = 0,
                    theta_mean_across_chains = 0 ,
                    eigen_vector = 0,
                    1,
                    1 ,# 40
                    tau_target = tau_target
        ))
      })
      
    } )  # | ------------    end of "foreach" loop -----------------------------------------------------------
  
  

 
  }
  
  

  {

   # tictoc::tic("timer 999")
  for (kk in 1:n_chains_burnin) {
      trace_theta_test_all_chains[kk, partitions[[interval]],   ] <-  non_BCA_burnin_outs[[1]][[kk]][partitions[[interval]],    ]
  }
  #  print(tictoc::toc())

  # non_BCA_burnin_outs[[3]]
  
  mean_L_burnin <- non_BCA_burnin_outs[[3]]
  mean_L_burnin <- mean(unlist(mean_L_burnin), na.rm = TRUE) # mean 
  
  ## reset adaptation params 
  # Get mean epsilon
  eps_per_chain <- non_BCA_burnin_outs[[2]]
  eps <- mean(unlist(eps_per_chain), na.rm = TRUE) # mean 
 #  eps <- median(unlist(eps_per_chain), na.rm = TRUE) # median 
  
  # Get mean L
  L_per_chain <-    non_BCA_burnin_outs[[16]]
  L_main <- L <- mean(unlist(L_per_chain), na.rm = TRUE) # mean 
 #  L_main <- L <- median(unlist(L_per_chain), na.rm = TRUE) # median 
  
  # Get mean tau
  # if (main_L_manual == FALSE) {
    tau_per_chain <-    non_BCA_burnin_outs[[10]]
    tau_main <- tau <- mean(unlist(tau_per_chain), na.rm = TRUE) # mean 
   #  tau_main <- tau <- median(unlist(tau_per_chain), na.rm = TRUE) # median 
  # } else { 
  #   tau_main <- tau <-  L_main * eps 
  # }
    

    # means of other ADAM adaptation params 
    snaper_m <-  Reduce("+",  non_BCA_burnin_outs[[18]])/ length( non_BCA_burnin_outs[[18]])
    snaper_s <-  Reduce("+",  non_BCA_burnin_outs[[19]])/ length( non_BCA_burnin_outs[[19]])
    snaper_w <-  Reduce("+",  non_BCA_burnin_outs[[20]])/ length( non_BCA_burnin_outs[[20]])
    m_sq <-  Reduce("+",  non_BCA_burnin_outs[[21]])/ length( non_BCA_burnin_outs[[21]])
    v_sq <-  Reduce("+",  non_BCA_burnin_outs[[22]])/ length( non_BCA_burnin_outs[[22]])
    cov_sq <-  Reduce("+",  non_BCA_burnin_outs[[23]])/ length( non_BCA_burnin_outs[[23]])
    tau_m_adam <-  Reduce("+",  non_BCA_burnin_outs[[24]])/ length( non_BCA_burnin_outs[[24]])
    tau_v_adam <-  Reduce("+",  non_BCA_burnin_outs[[25]])/ length( non_BCA_burnin_outs[[25]])
 
  
  # Get starting values for each chain
  theta_initial_values_per_chain <- array(dim = c(n_params, n_chains_burnin))
  # grad_initial_values_per_chain <- array(dim = c(n_params, n_chains))
  log_posterior_initial_values_per_chain <- rep(NA, n_chains_burnin)
  
  # str(trace_theta_test_per_chain[,,,kk])
  # str( non_BCA_burnin_outs[[1]])
  dim_1 <-  dim( non_BCA_burnin_outs[[1]][[1]])[1]
 #   trace_theta_test_per_chain <-  array(dim = c( dim_1 , n_burnin + 1, n_params_main, n_chains))
  
  for (kk in 1:n_chains_burnin) {
    theta_initial_values_per_chain[, kk] <- non_BCA_burnin_outs[[11]][[kk]]
    log_posterior_initial_values_per_chain[kk] <-  non_BCA_burnin_outs[[28]][[kk]]
  }
  
  
#  if ( interval  < n_adapt_intervals) {
    
            # Get mean  M  (diag, all params) - should NOT be adjusted for max variance
            M_diag_vec_per_chain <- non_BCA_burnin_outs[[17]]
            M_diag_vec <-  Reduce("+", M_diag_vec_per_chain)/ length(M_diag_vec_per_chain)
              
              
            # Get mean  M  (diag, all params) - should NOT be adjusted for max variance
            M_inv_diag_vec_stored_per_chain <- non_BCA_burnin_outs[[29]]  #  Empirical_Fisher_mat_adj_to_use <- Empirical_Fisher_mat_adj
            M_inv_diag_vec_stored <-  Reduce("+", M_inv_diag_vec_stored_per_chain)/ length(M_inv_diag_vec_stored_per_chain)
           
            
            M_inv_dense_main_stored_per_chain <- non_BCA_burnin_outs[[30]]
            M_inv_dense_main_stored <-  Reduce("+", M_inv_dense_main_stored_per_chain)/ length(M_inv_dense_main_stored_per_chain)
            M_inv_dense_main_stored <-  as.matrix( Matrix::forceSymmetric(M_inv_dense_main_stored) )
            
            M_dense_main_per_chain <- non_BCA_burnin_outs[[31]]
            M_dense_main <-  Reduce("+", M_dense_main_per_chain)/ length(M_dense_main_per_chain)
            M_dense_main <-  as.matrix( Matrix::forceSymmetric(M_dense_main) )
            
            M_inv_dense_main <- BayesMVP::Rcpp_solve(M_dense_main)
            M_inv_dense_main <-  as.matrix(Matrix::forceSymmetric(M_inv_dense_main))
          
           
            Chol_M_inv_dense <- NULL
            
            try({
              Chol_M_inv_dense <- BayesMVP::Rcpp_Chol((M_inv_dense_main))
            }, silent = TRUE)
            
            if (is.null(Chol_M_inv_dense)) {
              pd_indicator <- 0
            } else {
              pd_indicator <- 1
            }
            
            
            try({
              if (pd_indicator == 0) {
                M_inv_dense_main <-  as.matrix(nearPD(M_inv_dense_main)$mat)
                M_inv_dense_main <- as.matrix(Matrix::forceSymmetric(M_inv_dense_main))
              }
            },silent = TRUE)
            
            try({
              M_dense_main <-    BayesMVP::Rcpp_solve(M_inv_dense_main)
              M_inv_dense_main_chol <- BayesMVP::Rcpp_Chol((M_inv_dense_main))
            })
            
            
            
            try({  
            M_dense_sqrt <- expm::sqrtm(M_dense_main)
            })
           
            
            
            try({
            if ( start_dense_metric_prop == 1) {
              M_dense_main <-  diag(diag(M_dense_main)) # diag(sqrt_adj_vars) %*% M_dense_main_corr_mtx %*% diag(sqrt_adj_vars)
              M_inv_dense_main <-   diag(diag(M_inv_dense_main))  #  BayesMVP::Rcpp_solve(M_dense_main_adj)
              M_inv_dense_main_chol <- BayesMVP::Rcpp_Chol((M_inv_dense_main))
            }
            },silent = TRUE)
            
            
            
            largest_var <-   1  # max(1 / M_diag_vec)
            M_diag_vec_adj <-  (1 / M_inv_diag_vec_stored) * largest_var
            
            
  
 # }
 
  
  
  theta_mean_per_chain <- non_BCA_burnin_outs[[37]]
  theta_mean_across_chains <-  Reduce("+", theta_mean_per_chain)/ length(theta_mean_per_chain)

  
  
  tau_target_per_chain <- non_BCA_burnin_outs[[41]]
  tau_target <-  Reduce("+", tau_target_per_chain)/ length(tau_target_per_chain)
  
  
  print(paste("L = ",  round(L_main, 0)))
  print(paste("eps = ",  round(eps, 4)))
  print(paste("tau = ",  round(tau_main, 2)))
 

  
  #   tictoc::tic("burnin timer")
  

 
  }
  

} # end of for loop for intervals 


print(tictoc::toc(log = TRUE))
log.txt <- tictoc::tic.log(format = TRUE)
tictoc::tic.clearlog()
time_burnin <- unlist(log.txt)

try({  
  time_burnin <- as.numeric( substr(start = 0, stop = 100,  strsplit(  strsplit(time_burnin, "[:]")[[1]], "[s]")[[2]][1] ) )
})




}




  
  
# |-----------------   Get non-BCA burnin summary params ---------------------------------------
  { 
    
    
  ##   M_inv_dense_main =  BayesMVP::Rcpp_solve(M_dense_main)
    
    ## reset adaptation params 
    # Get mean epsilon
    eps_per_chain <- non_BCA_burnin_outs[[2]]
    eps <- mean(unlist(eps_per_chain), na.rm = TRUE) # mean 
      #  eps <- median(unlist(eps_per_chain), na.rm = TRUE) # median 
    
    # Get mean L
    L_per_chain <-    non_BCA_burnin_outs[[16]]
    L_main <- L <- mean(unlist(L_per_chain), na.rm = TRUE) # mean 
   #  L_main <- L <- median(unlist(L_per_chain), na.rm = TRUE) # median 
    
    # # Get mean tau
    # if (main_L_manual == FALSE) {
      tau_per_chain <-    non_BCA_burnin_outs[[10]]
     tau_main <- tau <- mean(unlist(tau_per_chain), na.rm = TRUE) # mean 
    #     # tau_main <- tau <- median(unlist(tau_per_chain), na.rm = TRUE) # median 
    # } else { 
    #   tau_main <- tau <-  L_main * eps 
    # }
    

      # Get mean  G (diag, all params) - should be adjusted for max variance
      diag_vec_G_per_chain <- non_BCA_burnin_outs[[9]]
      diag_vec_G <-  Reduce("+", diag_vec_G_per_chain)/ length(diag_vec_G_per_chain)
    #  diag_vec_G[index_us] <- 1 # shouldnt need to do this

      # Get mean  M  (diag, all params) - should NOT be adjusted for max variance
      M_diag_vec_per_chain <- non_BCA_burnin_outs[[17]]
      M_diag_vec <-  Reduce("+", M_diag_vec_per_chain)/ length(M_diag_vec_per_chain)
     #  M_diag_vec[index_us] <- 1 # shouldnt need to do this


      M_dense_main_per_chain <- non_BCA_burnin_outs[[31]]
      M_dense_main <-  Reduce("+", M_dense_main_per_chain)/ length(M_dense_main_per_chain)
      M_dense_main <-  as.matrix( Matrix::forceSymmetric(M_dense_main) )

      try({
      M_inv_dense_main <- BayesMVP::Rcpp_solve(M_dense_main)
      })
 

      Chol_M_inv_dense <- NULL

      try({
        Chol_M_inv_dense <- BayesMVP::Rcpp_Chol((M_inv_dense_main))
      }, silent = TRUE)

      if (is.null(Chol_M_inv_dense)) {
        pd_indicator <- 0
      } else {
        pd_indicator <- 1
      }


      try({
        if (pd_indicator == 0) {
          M_inv_dense_main <-  as.matrix(nearPD(M_inv_dense_main)$mat)
          M_inv_dense_main <- as.matrix(Matrix::forceSymmetric(M_inv_dense_main))
        }
      },silent = TRUE)

      try({
        M_dense_main <-    BayesMVP::Rcpp_solve(M_inv_dense_main)
        M_inv_dense_main_chol <- BayesMVP::Rcpp_Chol((M_inv_dense_main))
      },silent = TRUE)




      try({
      M_dense_sqrt <- expm::sqrtm(M_dense_main)
      })


      try({
        if ( start_dense_metric_prop == 1) {
          M_dense_main <-  diag(diag(M_dense_main)) # diag(sqrt_adj_vars) %*% M_dense_main_corr_mtx %*% diag(sqrt_adj_vars)
          M_inv_dense_main <-   diag(diag(M_inv_dense_main))  #  BayesMVP::Rcpp_solve(M_dense_main_adj)
          M_inv_dense_main_chol <- BayesMVP::Rcpp_Chol((M_inv_dense_main))
        }
      },silent = TRUE)

    # 
    
    
      
      
      theta_initial_values_per_chain <- array(dim = c(n_params, n_chains))
      

      
      kk_burn = 1
      for (kk in 1:n_chains) {
         theta_initial_values_per_chain[, kk] <- non_BCA_burnin_outs[[11]][[kk_burn]]
         kk_burn = kk_burn + 1
         if (kk_burn == n_chains_burnin) {  kk_burn = 1  }
      }
      
      
      print(paste("L = ",  round(L_main, 0)))
      print(paste("eps = ",  round(eps, 4)))
      print(paste("tau = ",  round(tau_main, 2)))
      
      
  }
    
  
  
  if (main_L_manual == TRUE) {
    
    L_main <- L_main_if_manual
    L_main_ii <- L_main_if_manual
    
  }





  
  
  

# -=  ||||||||||||||||||||||||||||||||||||||| --------------------------   Post-burnin  ------------------------------------------------------------

  try({  
  rm(velocity)
  # rm(velocity_0)
  # rm(velocity_0)
  rm(snaper_m)
  rm(snaper_s)
  rm(snaper_w)
  rm(trace_theta_test)
  })
  
  
 
  n_iter_inc_post_adapt_burn <- n_iter + 25
  
  
      use_Rcpp_for_sampling_phase = TRUE
   #  use_Rcpp_for_sampling_phase = FALSE
  
  
 if (use_Rcpp_for_sampling_phase == TRUE)  { 
{

  tictoc::tic("post-burnin timer")

  # run in parallel
 # set.seed(seed)

  xx <- doRNG::`%dorng%`(
    foreach::foreach(kk = 1:n_chains,
                     .packages = c("Rcpp",
                                   "BayesMVP")
    ),
    {


          theta_vec <-   theta_initial_values_per_chain[, kk]
      

          
          outs =     BayesMVPv2::Rcpp_fn_post_burnin_HMC_post_adaptation_phase_float_big_version(      Model_type = Model_type,
                                                                                                       theta_main_array = theta_vec[index_main],
                                                                                                       theta_us_array = theta_vec[index_us],
                                                                                                       y = y,
                                                                                                       X = X_list,
                                                                                                       other_args = other_lp_and_grad_args,
                                                                                                       tau_jittered =   TRUE,
                                                                                                       n_iter  =  n_iter_inc_post_adapt_burn,
                                                                                                       n_chain_for_loading_bar = kk,
                                                                                                       tau = tau_main,
                                                                                                       eps = eps,
                                                                                                       log_posterior = 0,
                                                                                                       M_inv_us_array = (1 / M_diag_vec[index_us]), 
                                                                                                       M_dense_main  = M_dense_main,
                                                                                                       M_inv_dense_main = M_inv_dense_main,
                                                                                                       M_inv_dense_main_chol  = M_inv_dense_main_chol)
         
     


                                                                                         

          return(outs)


          # return(list(trace_theta_main = outs[[1]],
          #             div_trace = outs[[2]]))



    })


  {
      print(tictoc::toc(log = TRUE))
      log.txt <- tictoc::tic.log(format = TRUE)
      tictoc::tic.clearlog()
      time_post_burnin <- unlist(log.txt)
  }


      try({
        time_post_burnin <- as.numeric( substr(start = 0, stop = 100,  strsplit(  strsplit(time_post_burnin, "[:]")[[1]], "[s]")[[2]][1] ) )
      })
  #
  #
}

   
   
 } else { 
#   
#   
 
  
  
  
  str(theta_initial_values_per_chain)
  
  
  apply(theta_initial_values_per_chain, c(1), mean)
  
  
  # 500 fails quickly 
  
   # iter_using_Rcpp = FALSE
     iter_using_Rcpp = TRUE
  
  {

    tictoc::tic("post-burnin timer")

    # run in parallel
    # set.seed(seed)

    xx <- doRNG::`%dorng%`(
      foreach::foreach(kk = 1:n_chains,
                       .packages = c("Rcpp",
                                     "BayesMVP")
      ),
      {


        theta_main_trace <- array(0, dim = c(n_params_main, n_iter_inc_post_adapt_burn))
        div_vec <- c()

        theta_previous_iter <-       theta_initial_values_per_chain[, kk]
        theta_vec <-             theta_initial_values_per_chain[, kk]


        Phi_type_original <- Phi_type
        
        for (ii in 1:n_iter_inc_post_adapt_burn) {
          
          
          
          if (iter_using_Rcpp == TRUE) {
            
                          
                          if (kk == 1) {
                            if (ii %% round(n_iter_inc_post_adapt_burn/100, 0) == 0) {
                              comment(print( ( 100 * (ii / n_iter_inc_post_adapt_burn ) )) )
                            }
                          }
                          
                          tau_main_ii = runif(n = 1, min = 0, max = 2 * tau_main)
                          L_main_ii = ceiling(tau_main_ii / eps)
                          L_main_ii =  ifelse(L_main_ii < 1, 1, L_main_ii)
                          
                          if   (Phi_type_original  == 3) n_attempts = 1
                          else                           n_attempts = 2
                        
                          
                        skip = FALSE
                        for (attempt in  1:n_attempts)  {
                          
                                       
                                    
                                        if (skip == TRUE) {
                                          skip <- FALSE
                                          next
                                        }
                          
                          if (attempt == 2) comment(print(paste("attempt 2")))
                                        
                                        if (attempt == 1)   { 
                                           Phi_type = Phi_type_original
                                        } else { 
                                           Phi_type = 3
                                        }
                          
                          
                                        div_vec[ii] = 1;
                          

                                        if (ii > 1) {
                                           theta_main_trace[, ii] =   theta_main_trace[, ii - 1]
                                        }
                                        
                                        
                                        
                                        velocity_0 <- rep(NA,  n_params)
                                        
                                        velocity_0[index_us] = rnorm(n = n_us, mean = rep(0, n_us),  sd = sqrt(1 / M_diag_vec[index_us] ) )
                                        velocity_0[(n_us + 1):n_params] =  M_inv_dense_main_chol %*% rnorm(n = n_params_main, 0, 1)
                                        
                                        velocity <- velocity_0
                                        velocity_prop <- velocity_0

                                        try({

                                          theta_initial = theta_vec




                                          out_mat_for_leapfrog = BayesMVP::Rcpp_fn_wo_list_ELMC_multiple_leapfrogs_EUC_EFISH_HI_with_MIXED_M_NT_us_float(   n_cores =   1,
                                                                                                                                  theta  =  theta_previous_iter, # //////////////////////////////////
                                                                                                                                  y = y,   # //////////////////////////////////
                                                                                                                                  X = X_list, # //////////////////////////////////
                                                                                                                                  dense_G_indicator = TRUE,  # 5
                                                                                                                                  numerical_diff_e = 0.0001,
                                                                                                                                  L = L_main_ii,
                                                                                                                                  eps = eps,
                                                                                                                                  log_posterior = log_posterior,
                                                                                                                                  log_M_inv_us_vec =  log(abs(1 / M_diag_vec[index_us]  )) ,   #  10  # //////////////////////////////////
                                                                                                                                  M_dense_main  = M_dense_main,
                                                                                                                                  M_inv_dense_main  = M_inv_dense_main,
                                                                                                                                  log_M_inv_dense_main = log(abs(M_inv_dense_main)),
                                                                                                                                  M_inv_dense_main_chol  = M_inv_dense_main_chol,
                                                                                                                                  log_M_inv_dense_main_chol = log(abs(M_inv_dense_main_chol)),
                                                                                                                                  n_us = n_us,
                                                                                                                                  n_params_main = n_params_main,            # 15
                                                                                                                                  exclude_priors =  exclude_priors,
                                                                                                                                  CI = CI,
                                                                                                                                  lkj_cholesky_eta = lkj_cholesky_eta,
                                                                                                                                  prior_coeffs_mean = prior_coeffs_mean,
                                                                                                                                  prior_coeffs_sd = prior_coeffs_sd,                  # 20
                                                                                                                                  n_class = n_class,
                                                                                                                                  n_tests = n_tests,
                                                                                                                                  ub_threshold_phi_approx = ub_phi_approx,
                                                                                                                                  n_chunks = num_chunks, # chunks
                                                                                                                                  corr_force_positive = corr_force_positive,           # 25
                                                                                                                                  prior_for_corr_a = list_prior_for_corr_a,
                                                                                                                                  prior_for_corr_b = list_prior_for_corr_b,
                                                                                                                                  corr_prior_beta = corr_prior_beta,
                                                                                                                                  corr_prior_norm = FALSE,
                                                                                                                                  lb_corr = lb_corr,                        # 30
                                                                                                                                  ub_corr = ub_corr,
                                                                                                                                  known_values_indicator = known_values_indicator_list,
                                                                                                                                  known_values = known_values_list,
                                                                                                                                  prev_prior_a = prev_prior_a,
                                                                                                                                  prev_prior_b = prev_prior_b,                     # 35
                                                                                                                                  Phi_type = Phi_type,
                                                                                                                                  sampling_option =   sampling_option ,  # 11)
                                                                                                                                  generate_velocity = TRUE,
                                                                                                                                  log_abs_velocity_0 = log(abs(velocity_0)), # //////////////////////////////////
                                                                                                                                  velocity_0_signs = sign(velocity_0) , # //////////////////////////////////
                                                                                                                                  M_inv_us_vec_sqrt = sqrt((1 / M_diag_vec[index_us]  )),
                                                                                                                                  log_scale = TRUE
                                                                                                                                 #  log_abs_M_dense_main = log(abs(M_dense_main)),
                                                                                                                                 #  log_M_us_vec =  log(abs( M_diag_vec[index_us]  ) )
                                        )

                                        log_posterior =  out_mat_for_leapfrog[2,1]
                                        log_posterior_prop =  out_mat_for_leapfrog[2,1]
                                        U_x = - log_posterior;

                                        log_posterior_initial = out_mat_for_leapfrog[1,1]
                                        U_x_initial =  - log_posterior_initial;

                                        velocity_0  =  exp(out_mat_for_leapfrog[,5]) * (out_mat_for_leapfrog[,6])
                                        velocity_prop  =  exp(out_mat_for_leapfrog[,7]) * (out_mat_for_leapfrog[,8])

                                         energy_old = U_x_initial ;
                                         energy_old = energy_old  +    0.5 * sum( velocity_0[(n_us+1):n_params] * BayesMVP::Rcpp_mult_mat_by_col_vec(M_dense_main,  velocity_0[(n_us+1):n_params]) )
                                         energy_old = energy_old  +    0.5 * sum(  ( head(velocity_0, n_us) ) * (  M_diag_vec[index_us]  ) )

                                         energy_new = U_x ;
                                         energy_new = energy_new  +    0.5 * sum( velocity_prop[(n_us+1):n_params] * BayesMVP::Rcpp_mult_mat_by_col_vec(M_dense_main,  velocity_prop[(n_us+1):n_params]) )
                                         energy_new = energy_new  +    0.5 * sum(  ( head(velocity_prop, n_us) ) * (  M_diag_vec[index_us]  ) )






                                        log_ratio = - energy_new + energy_old;

                                        p_jump_vec <- c()
                                        p_jump_vec[1] = 1;
                                        p_jump_vec[2] = exp(log_ratio);

                                        p_jump =  min(p_jump_vec);

                                        accept = 0;

                                        div = 0;


                                        if  ((runif(n = 1, 0, 1) > p_jump) || (div == 1)) { # // # reject proposal

                                            accept = 0;
                                            theta_vec =    theta_initial ;

                                        } else {   #  // # accept proposal


                                            accept = 1;
                                            theta_vec = exp(out_mat_for_leapfrog[,3]) *  out_mat_for_leapfrog[,4]     #   ; //  theta_prop ; // out_mat_for_leapfrog.col(1).segment(1, n_params) ;

                                        }



                                        div_vec[ii] = 0;
                                        theta_previous_iter = theta_initial #  head(single_iter_out, n_params) ;
                                        theta_main_trace[, ii] = theta_vec[(n_us + 1):(n_us + n_params_main)]
                                        })
                                        # 
                                        
                                        # 
                                        # try({
                                        #   single_iter_out =     BayesMVP::Rcpp_fn_no_list_HMC_single_iter_float(                   n_cores =   1,
                                        #                                                                                            theta_initial  =  theta_previous_iter, # //////////////////////////////////
                                        #                                                                                            y = y,   # //////////////////////////////////
                                        #                                                                                            X = X_list, # //////////////////////////////////
                                        #                                                                                            dense_G_indicator = TRUE,  # 5
                                        #                                                                                            L = L_main_ii,
                                        #                                                                                            eps = eps,
                                        #                                                                                            log_posterior_initial = log_posterior,
                                        #                                                                                            M_inv_us_vec = (1 / M_diag_vec[index_us]  ), # //////////////////////////////////
                                        #                                                                                            log_M_inv_us_vec =  log(abs(1 / M_diag_vec[index_us]  )) ,   #  10  # //////////////////////////////////
                                        #                                                                                            M_dense_main  = M_dense_main,
                                        #                                                                                            M_inv_dense_main  = M_inv_dense_main,
                                        #                                                                                            log_M_inv_dense_main = log(abs(M_inv_dense_main)),
                                        #                                                                                            M_inv_dense_main_chol  = M_inv_dense_main_chol,
                                        #                                                                                            log_M_inv_dense_main_chol = log(abs(M_inv_dense_main_chol)),
                                        #                                                                                            n_us = n_us,
                                        #                                                                                            n_params_main = n_params_main,            # 15
                                        #                                                                                            exclude_priors,
                                        #                                                                                            CI,
                                        #                                                                                            lkj_cholesky_eta,
                                        #                                                                                            prior_coeffs_mean,
                                        #                                                                                            prior_coeffs_sd,                  # 20
                                        #                                                                                            n_class,
                                        #                                                                                            n_tests,
                                        #                                                                                            ub_phi_approx,
                                        #                                                                                            num_chunks, # chunks
                                        #                                                                                            corr_force_positive,           # 25
                                        #                                                                                            list_prior_for_corr_a,
                                        #                                                                                            list_prior_for_corr_b,
                                        #                                                                                            corr_prior_beta,
                                        #                                                                                            FALSE,
                                        #                                                                                            lb_corr,                        # 30
                                        #                                                                                            ub_corr,
                                        #                                                                                            known_values_indicator_list,
                                        #                                                                                            known_values_list,
                                        #                                                                                            prev_prior_a,
                                        #                                                                                            prev_prior_b,                     # 35
                                        #                                                                                            Phi_type = Phi_type,
                                        #                                                                                            sampling_option = 100,  # 11)
                                        #                                                                                           generate_velocity = FALSE,
                                        #                                                                                           log_abs_velocity_0 = log(abs(velocity_0)), # //////////////////////////////////
                                        #                                                                                           velocity_0_signs = sign(velocity_0), # //////////////////////////////////
                                        #                                                                                           log_scale = TRUE,
                                        #                                                                                           log_abs_M_dense_main = log(abs(M_dense_main)),
                                        #                                                                                           log_M_us_vec =  log(abs( M_diag_vec[index_us] )) ,
                                        #                                                                                           M_inv_us_vec_sqrt = sqrt( (1 / M_diag_vec[index_us]  ))
                                        #   )
                                        # 
                                        # 
                                        #         div_vec[ii] = 0;
                                        #         theta_previous_iter = head(single_iter_out, n_params) ;
                                        #         theta_main_trace[, ii] = head(single_iter_out, n_params)[(n_us + 1):(n_us + n_params_main)]
                                        # 
                                        # })
                                     # 
                                        
                                        
                                      if  ( (attempt == 1) ) { 
                                              
                                              if  ( ( div_vec[ii]  == 0) && (Phi_type_original == 1)   ) { 
                                                skip = TRUE
                                                Phi_type = Phi_type_original
                                              }
                                              
                                              if  ( ( div_vec[ii]  == 1) && (Phi_type_original == 1)   ) {  # if div  on  attempt 1, try attempt 2
                                                skip = FALSE
                                                Phi_type = 3
                                              }
                                        
                                      }
                                        

                                        
                                     ##   if (attempt == 2)    Phi_type = Phi_type_original
                                        
                                        
                        }
            
          
                }  else { 

                  
                  
                  
                              if (kk == 1) {
                                if (ii %% round(n_iter_inc_post_adapt_burn/100, 0) == 0) {
                                  comment(print( ( 100 * (ii / n_iter_inc_post_adapt_burn ) )) )
                                }
                              }
                              
                              tau_main_ii = runif(n = 1, min = 0, max = 2 * tau_main)
                              L_main_ii = ceiling(tau_main_ii / eps)
                              L_main_ii =  ifelse(L_main_ii < 1, 1, L_main_ii)
                              
                              
                              if (ii > 1) {
                                theta_main_trace[, ii] =   theta_main_trace[, ii - 1]
                              }
                              
                              try( {  
                                single_iter_out =     LMC_single_iteration_wo_list_R(                         theta_previous_iter,
                                                                                                              y,
                                                                                                              X_list,
                                                                                                              TRUE,  # 5
                                                                                                              L_main_ii,
                                                                                                              eps,
                                                                                                              log_posterior,
                                                                                                              M_diag_vec[index_main] ,
                                                                                                              1 / M_diag_vec[index_us] ,   # 10
                                                                                                              M_dense_main,
                                                                                                              M_inv_dense_main,
                                                                                                              M_inv_dense_main_chol,
                                                                                                              n_us,
                                                                                                              n_params_main,            # 15
                                                                                                              exclude_priors,
                                                                                                              CI,
                                                                                                              lkj_cholesky_eta,
                                                                                                              prior_coeffs_mean,
                                                                                                              prior_coeffs_sd,                  # 20
                                                                                                              n_class,
                                                                                                              n_tests,
                                                                                                              ub_phi_approx,
                                                                                                              num_chunks, # chunks
                                                                                                              corr_force_positive,           # 25
                                                                                                              list_prior_for_corr_a,
                                                                                                              list_prior_for_corr_b,
                                                                                                              corr_prior_beta,
                                                                                                              FALSE,
                                                                                                              lb_corr,                        # 30
                                                                                                              ub_corr,
                                                                                                              known_values_indicator_list,
                                                                                                              known_values_list,
                                                                                                              prev_prior_a,
                                                                                                              prev_prior_b,                     # 35
                                                                                                              Phi_type,
                                                                                                              11)              
                                div <-  single_iter_out[n_params + N]
                                
                                if (div == 0) {
                                  div_vec[ii] = 0;
                                  theta_previous_iter = head(single_iter_out, n_params) ;
                                  theta_main_trace[, ii] = head(single_iter_out, n_params)[(n_us + 1):(n_us + n_params_main)] ; 
                                } else  { 
                                  div_vec[ii] = 1;
                                }
                                
                                
                                
                                
                              })
                              
                  
                  
                  
                            
                  }

          
        }

        out_list <- list();

        out_list[[1]] = theta_main_trace;
        out_list[[3]] = div_vec;

        return(out_list)



      })




    {
      print(tictoc::toc(log = TRUE))
      log.txt <- tictoc::tic.log(format = TRUE)
      tictoc::tic.clearlog()
      time_post_burnin <- unlist(log.txt)
    }


    try({
      time_post_burnin <- as.numeric( substr(start = 0, stop = 100,  strsplit(  strsplit(time_post_burnin, "[:]")[[1]], "[s]")[[2]][1] ) )
    })

    
    
    
  }
    
    
    
    
    
 }
    
    
    

  # 
  # 
  
  
  ##     str(xx)
       
       
   ##    xx[[1]]
      # 
      # str(xx[[1]][[1]])
  # trace_theta_main_list <- xx[[1]]$trace_theta_main

  
  {
    
    
      trace_theta_main_list <- list()
      trace_individual_log_lik_array_list <- list()
      trace_div_list <- list()
      
      for (kk in 1:n_chains) {
         trace_theta_main_list[[kk]] <- xx[[kk]][[1]] 
         trace_individual_log_lik_array_list[[kk]] <- xx[[kk]][[2]] 
         trace_div_list[[kk]] <- xx[[kk]][[3]] 
      }
      
   
  }
  
  
  
  { 
    
          time_burnin <-       time_burnin # * ( n_burnin / (n_iter + n_burnin) )
          time_sampling <-     time_post_burnin  # * ( n_iter / (n_iter + n_burnin) )
          time_total <- time_burnin + time_sampling
    
    
    
  }
  
  
  
  {
    
    chain_set_new = c(1:n_chains)
    
    pars_index_vec <- index_subset_excl_known - n_us
   # pars_index_vec <-  1:31 # (N*n_tests + 1):(N*n_tests + 31)
    n_params_main_effective <- length(pars_index_vec)
    n_corrs_effective <- n_params_main_effective - (1 + n_coeffs)
    
    trace_theta_main <- array(dim = c(n_chains, n_iter, n_params_main_effective))
    trace_individual_log_lik_array  <- array(dim = c(n_chains, n_iter, N))
    trace_div <- array(dim = c(n_chains, n_iter))
    
    try({
    for (kk in 1:n_chains) {
      for (i_par in 1:n_params_main_effective) {
         #  trace_theta_main[kk,, i_par] <-      (trace_theta_main_list[[kk]]  )[[1]][pars_index_vec[i_par] , ]
        trace_theta_main[kk, 1:(0 + n_iter), i_par] <-    trace_theta_main_list[[kk]][pars_index_vec[i_par], ][  (n_post_adapt_burnin  + 1):(n_post_adapt_burnin+ n_iter)  ] #  trace_theta_main_list[pars_index_vec[i_par] , ]
      }
      try({  
      trace_individual_log_lik_array[kk,,] <-    t(trace_individual_log_lik_array_list[[kk]][ , (n_post_adapt_burnin  + 1):(n_post_adapt_burnin+ n_iter)])  #  trace_theta_main_list[pars_index_vec[i_par] , ]
      }, silent = TRUE)
      trace_div[kk,] <-     trace_div_list[[kk]][(n_post_adapt_burnin  + 1):(n_post_adapt_burnin+ n_iter)]
    }
    })
    
    
    trace_theta_main = ifelse(trace_theta_main == 0, Inf, trace_theta_main)
    
    
    n_divergent =  sum(trace_div)  ; n_divergent
    pct_divergent = (  n_divergent  / (n_chains * n_iter) ) * 100  ; pct_divergent
    
    
  }
  
  
  
  {
    
 
    
    
  }
  
  
  
  {
    
    
    
    # ESS and R-hat for MAIN parameters
    ess <- rhat_vec <- nexted_rhat_vec <-  c()
    
    
    superchain_ids = seq(from = 1, to = n_chains, by = 1)
    if (n_chains > 4)  superchain_ids = c(rep(1, n_chains/2), rep(2, n_chains/2))
    if (n_chains > 15)  superchain_ids = c(rep(1, n_chains/4), rep(2, n_chains/4), rep(3, n_chains/4), rep(4, n_chains/4))
    if (n_chains > 47)  superchain_ids = c(rep(1, n_chains/8), rep(2, n_chains/8), rep(3, n_chains/8), rep(4, n_chains/8),
                                           rep(5, n_chains/8), rep(6, n_chains/8), rep(7, n_chains/8), rep(8, n_chains/8))
    
    try({
      for (i in 1:n_params_main_effective) {   # c(2, 4,5,7,8:16)
        ess[i] <-  sort(rstan::ess_bulk((t(   trace_theta_main[ chain_set_new, 1:(0 + n_iter), i]  )))) ;  ess[i]
        rhat_vec[i] <-   rstan::Rhat(t((trace_theta_main[chain_set_new, 1:(0 + n_iter), i])))
        nexted_rhat_vec[i] <-   posterior::rhat_nested(t((trace_theta_main[chain_set_new, 1:(0 + n_iter), i])), superchain_ids = superchain_ids)
      }
    })
    #
    
    {
      print(min(ess))
      print(max(rhat_vec))
      print(max(nexted_rhat_vec))
    }
    

    
  }
  
  
  {
    
    
 
  
        total_time_mins <- time_total/60
        total_time_hours <- total_time_mins/60
        pb_time_seconds <- time_sampling
        pb_time_mins <- pb_time_seconds/60
        pb_time_hours <- pb_time_mins/60
 
        L_main_mean_cumul <- 0
        L_samp_main_mean_cumul <- 0

        eps_main_mean_cumul <- 0
 
        
  }
  
  
  
 
  {
    
   

        try({
          stopCluster(cl)
        }, silent = TRUE)


                try({
                 
 
    
                      print(sort(round(ess, 0)))
    
                      Max_rhat <- max(rhat_vec, na.rm = TRUE) ; Max_rhat
                      Max_rhat_nested <- max(nexted_rhat_vec, na.rm = TRUE) ; Max_rhat
    
    
                      Min_ESS <- min(ess) ; # print(Min_ESS)
                      Mean_ESS <- mean(ess) ; # print(Mean_ESS)
                      Min_ESS_per_sec <- Min_ESS / time_total
                      Min_ESS_per_sec_sampling <- Min_ESS / time_sampling
    
                      if (MALA_main == TRUE) {     L_main_mean <- 1 ; L_main <- 1    }
                      n_grad_evals_main_total  <-  mean_L_burnin * n_chains  * n_burnin  + L_main * n_chains *  n_iter
                      n_grad_evals_main_sampling <-   n_iter  * n_chains * L_main
    
                      Min_ESS_per_grad <- Min_ESS / n_grad_evals_main_total
                      Min_ESS_per_grad_sampling <-  Min_ESS / n_grad_evals_main_sampling ;# print(round(Min_ESS_per_grad_sampling * 1000, 3))
    
            
                })
    
    
  }

  


 


              try({
              file_list <- list(

                # basic  info
                paste("seed = ", seed),
                paste("N = ", N),
                paste("n_chains = ", n_chains),
                (paste("n_burnin = ", n_burnin)),
                (paste("n_iter = ", n_iter)),
                (paste("JITT = ", tau_main_jittered)),
                print(paste("adapt_delta  = ", adapt_delta))  ,
                (paste("learning_rate  = ", learning_rate_main)),
                (paste("fix_rho  = ", fix_rho)),
                (paste("clip_iter = ", clip_iter)),
                (paste("prLKJ_ = ", lkj_cholesky_eta)),
                (paste("Euclidean (main) = ", Euclidean_M_main)),
                (paste("Smooth (main) = ", smooth_M_main)),
                (paste("G_dense_ (main) = ", dense_G_indicator)),
                (paste("G_us_const (u's) = ", u_Euclidean_metric_const)),

                (paste( "NT_us  = ", NT_us))  ,
                (paste( "rough_approx  = ", rough_approx))    ,
                (paste( "lb_phi_approx  = ", lb_phi_approx))    ,
                (paste( "ub_phi_approx  = ", ub_phi_approx))   ,


                # efficiency stats
                (paste("Min ESS (main) = ", round(Min_ESS, 3))),
                (paste("Max Rhat (main params) = ", round(Max_rhat, 3))),

                (paste("time (total) = ", round(min(time_total), 0))),
                (paste("time (burnin) = ", round(min(time_burnin), 0))),
                (paste("time (sampling) = ", round(min(time_sampling), 0))),


                paste("total time =", round(time_total, 0), "seconds"),
                paste("total time =", floor(total_time_mins), "minutes and ", round(((total_time_mins - floor(total_time_mins))*60), 0), "seconds"),
                paste("total time =", floor(total_time_hours), "hours and ", round(((total_time_hours - floor(total_time_hours))*60), 0), "minutes"),

                paste("Sampling (post-burnin) time =", round(pb_time_seconds, 0), "seconds"),
                paste("Sampling (post-burnin) time =", floor(pb_time_mins), "minutes and ", round(((pb_time_mins - floor(pb_time_mins))*60), 0), "seconds"),
                paste("Sampling (post-burnin) time =", floor(pb_time_hours), "hours and ", round(((pb_time_hours - floor(pb_time_hours))*60), 0), "minutes"), # 20

                (paste("Min ESS / sec = ", round((Min_ESS_per_sec), 4)))        ,
                (paste("Min ESS / sec (sampling only)  = ", round((Min_ESS_per_sec_sampling), 4)))         ,

                (paste("Min ESS / grad = (total) ", round((Min_ESS_per_grad*1000), 2))),
                (paste("Min ESS / grad (sampling only) = ", round((Min_ESS_per_grad_sampling*1000), 2)))     ,

                # other stats
                (paste("eps (sampling) = ", eps)),
                (paste("L (sampling) = ", L_main)),
                (paste("divergences = ", n_divergent))

              )

             ###  saveRDS(file_list, file = paste0("Eff_info_", file_name))
              })

 


 

              try({
              {

                print(paste("seed = ", seed))
                print(paste("| ---------------   N = ", N))
                print(paste("Phi_type = ", Phi_type))
                print(paste("clip_iter = ", clip_iter))
                print(paste("n_chains (burnin only) = ", n_chains_burnin))
                print(paste("n_chains (sampling only) = ", n_chains))
                print(paste("n_burnin = ", n_burnin))
                print(paste("n_iter = ", n_iter))


                print(paste("learning_rate (main) = ", learning_rate_main))
                print(paste("adapt_interval_width   = ", adapt_interval_width))
                try({   print(paste("adapt_delta (main) = ", adapt_delta))  })


                try({  
                print(paste("n_divergent = ", round(n_divergent, 0)))
                print(paste("pct_divergent = ", round(pct_divergent, 0)))
                })
                
                

                # print(paste("SoftAbs  = ", soft_abs))
                # print(paste("SoftAbs_alpha_ = ", soft_abs_alpha))

                print(paste("corr_force_positive = ", corr_force_positive))
                print(paste("priorLKJ = ", lkj_cholesky_eta))


               # # cat(colourise(      (paste( "NT_us  = ", NT_us))      , "blue"), "\n")
               #  cat(colourise(      (paste( "rough_approx  = ", rough_approx))      , "blue"), "\n")
               #  cat(colourise(      (paste( "lb_phi_approx  = ", lb_phi_approx))      , "blue"), "\n")
                cat(colourise(      (paste( "ub_phi_approx  = ", ub_phi_approx))      , "blue"), "\n")

                cat(colourise(      (paste( "Euclidean (main) = ", Euclidean_M_main))      , "blue"), "\n")
                cat(colourise(      (paste( "L_jitter (main) = ", tau_main_jittered))      , "blue"), "\n")

                cat(colourise(    (paste("mean L (main - burnin) = ", round(mean_L_burnin, 0)))        , "blue"), "\n")
                cat(colourise(    (paste("mean L (main - sampling) = ", round(L_main, 0)))        , "blue"), "\n")

                cat(colourise(    (paste("mean epsilon (main) = ", round(eps, 4)))          , "blue"), "\n")

                cat(colourise(    paste( "Dense_G (main) = ", dense_G_indicator )      , "blue"), "\n")
                cat(colourise(    paste( "smooth_M_main = ",  smooth_M_main)      , "blue"), "\n")

                cat(colourise(    paste( "adapt_M_us = ", adapt_M_us)      , "green"), "\n")
                # cat(colourise(    paste( "smooth_M_us = ",  smooth_M_main)      , "green"), "\n")
                cat(colourise(    paste( "u_Euclidean_metric_const (median(Var(U's))) = ", u_Euclidean_metric_const)      , "green"), "\n")

                print(paste("Min ESS (main) = ", round(Min_ESS, 0)))
                print(paste("Max Rhat (main params) = ", round(Max_rhat, 3)))
                print(paste("Max n-Rhat (main params) = ", round(Max_rhat_nested, 3)))

                print(paste("time (total) = ", round((time_total/60), 3)))
                print(paste("time (sampling) = ", round((time_sampling/60), 3)))

                cat(colourise(        (paste("Min ESS / sec (total) = ", round((Min_ESS_per_sec), 4)))            , "purple"), "\n")
                cat(colourise(        (paste("Min ESS / sec (sampling only)  = ", round((Min_ESS_per_sec_sampling), 4)))            , "purple"), "\n")

                cat(colourise(       (paste("Min ESS / grad (total) = ", round((Min_ESS_per_grad*1000), 4)))             , "red"), "\n")
                cat(colourise(       (paste("Min ESS / grad (sampling only) = ", round((Min_ESS_per_grad_sampling*1000), 4)))             , "red"), "\n")

              }
              })
 


 
        if (save_individual_log_lik == TRUE) { 
          trace_individual_log_lik_array <- trace_individual_log_lik_array
        } else { 
          trace_individual_log_lik_array <- "Please set: 'save_individual_log_lik = TRUE' to save this log_lik object"
        }
        
        

try({  
  return(list(trace_array = trace_theta_main,
              trace_individual_log_lik_array = trace_individual_log_lik_array,
              L_burnin = round(mean_L_burnin, 0),
              L_sampling = round(L_main, 0),
              min_ESS = round(Min_ESS, 0),
              Max_rhat = Max_rhat,
              Max_rhat_nested = Max_rhat_nested,
              time_total = round((time_total/60), 3),
              time_sampling = round((time_sampling/60), 3),
              Min_ESS_per_sec_overall = round((Min_ESS_per_sec), 4),
              Min_ESS_per_sec_sampling = round((Min_ESS_per_sec_sampling), 4),
              Min_ESS_per_grad_overall = round((Min_ESS_per_grad*1000), 4),
              Min_ESS_per_grad_sampling = round((Min_ESS_per_grad_sampling*1000), 4),
              time_burnin = round((time_total/60), 3) - round((time_sampling/60), 3),
             n_divs =  round(n_divergent, 0),
             trace_div = trace_div
  ))
})
  
}







