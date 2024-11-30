
########## -------- EXAMPLE 2 -------------------------------------------------------------------------------------------------------------------------------- 
######### Running the BUILT-IN (i.e. manual gradients) LC-MVP and/or 
######### latent trait models for the analysis of test accuracy data 
######### without a gold standard (using simulated data). 
         


## d::document("~/Documents/Work/PhD_work/R_packages/BayesMVP")   # Rebuild documentation

# Sys.setenv(CCACHE_DEBUG=1)  # Enable debug logging
# Sys.setenv(CCACHE_DEBUGDIR="C:/Users/enzoc/ccache-debug")   
# Sys.setenv(CCACHE_LOGFILE="C:/Users/enzoc/ccache.log")

if (.Platform$OS.type == "windows") {
  
    # Sys.setenv(CCACHE_DIR = "C:\\rtools44\\x86_64-w64-mingw32.static.posix\\installed")
    # Sys.setenv(CCACHE_MAXSIZE = "10G")  # Set cache size limit
  
    try({  setwd("C:/users/enzoc/Documents/BayesMVP/examples")    }, silent = TRUE)
   pkg_dir <- "C:/users/enzoc/Documents/BayesMVP"
  
} else { 
  
   try({  setwd("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/examples")   }, silent = TRUE)
   try({  setwd("/home/enzo/Documents/Work/PhD_work/R_packages/BayesMVP/examples")    }, silent = TRUE)
  pkg_dir <- "~/Documents/Work/PhD_work/R_packages/BayesMVP"
  
}

options(buildtools.check = function(action) TRUE ) 

 

options(warning.length =  8000)

require(cmdstanr)
require(bridgestan)
require(BayesMVP)

#          C:\Users\enzoc\.cmdstan\cmdstan-2.35.0\stan\lib\stan_math\lib\tbb_2020.3\include\tbb

 
rstudioapi::restartSession()
remove.packages("BayesMVP")
unlink("C:/Users/enzoc/AppData/Local/R/win-library/4.4/00LOCK-BayesMVP", recursive = TRUE)
devtools::clean_dll(pkg_dir)  
Rcpp::compileAttributes(pkg_dir)  
devtools::document(pkg_dir)  
## Install
devtools::clean_dll(pkg_dir)
Rcpp::compileAttributes(pkg_dir)
devtools::install(pkg_dir)
require(BayesMVP)
BayesMVP::install_BayesMVP()
require(BayesMVP) # this works (after installation using "--no-test-load" flag)
 



remotes::install_github("https://github.com/CerulloE1996/BayesMVP", force = TRUE)
require(BayesMVP)
BayesMVP::install_BayesMVP()
require(BayesMVP) # this works (after installation using "--no-test-load" flag)





BayesMVP::detect_vectorization_support()


 

{
  
    # Set working direcory ---------------

    try({  setwd("C:/users/enzoc/Documents")    }, silent = TRUE)
    #  options(repos = c(CRAN = "http://cran.rstudio.com"))
    
    # options ------------------------------------------------------------------- 
    #  totalCores = 8
    rstan::rstan_options(auto_write = TRUE)
    options(scipen = 99999)
    options(max.print = 1000000000)
    #  rstan_options(auto_write = TRUE)
    options(mc.cores = parallel::detectCores())
 
}

require(BayesMVP)

if (.Platform$OS.type == "windows") {
  pkg_dir <- "C:/users/enzoc/Documents/BayesMVP"  ### TEMP - remove for final version of file 
} else { 
  pkg_dir <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP" ### TEMP - remove for final version of file 
  # pkg_dir <- "/home/enzo/Documents/Work/PhD_work/R_packages/BayesMVP" # ## TEMP - remove for final version of file 
}


 

 



Model_type <- "LC_MVP"
##   Model_type <- "latent_trait"


source(file.path(pkg_dir, "examples/BayesMVP_LC_MVP_prep.R"))



###  ------  select sample size (one of: 500, 1000, 2500, 5000, 12500, 25000)
N <- 500



{
   
  
{
  if (N == 500)   y <-  y_master_list_seed_123_datasets[[1]]  
  if (N == 1000)  y <-  y_master_list_seed_123_datasets[[2]]  
  if (N == 2500)  y <-  y_master_list_seed_123_datasets[[3]]  
  if (N == 5000)  y <-  y_master_list_seed_123_datasets[[4]] 
  if (N == 12500) y <-  y_master_list_seed_123_datasets[[5]] 
  if (N == 25000) y <-  y_master_list_seed_123_datasets[[6]]  
}


  ## Set important variables 
    n_tests <- ncol(y)
    n_class <- 2
    n_covariates <- 1
    n_covariates_max <- 1
    n_nuisance <- N * n_tests
    n_params_main <-  choose(n_tests, 2) * 2 + 1 + n_tests * n_covariates * 2

  
    seed_dataset <- 123 
    
    n_covariates_per_outcome_mat <- array(n_covariates_max, dim = c(n_class, n_tests))

  # ## X is user-supplied
  # n_covariates_per_outcome_mat <- array(1, dim = c(n_class, n_tests))
  # X_per_class <- array(1, dim = c(n_tests, 1, N))
  # X_list <- list()
  # 
  # for (c in 1:n_class) {
  #   X_list[[c]] <- list()
  #   for (t in 1:n_tests) {
  #     X_list[[c]][[t]] <- array(999999, dim = c(N, n_covariates_per_outcome_mat[c, t] ))
  #     for (k in 1:n_covariates_per_outcome_mat[c, t] ) {
  #       for (n in 1:N) {
  #         X_list[[c]][[t]][n,  k] <- X_per_class[t, k, n]
  #       }
  #     }
  #   }
  # }
  # 
  # 
  # X <- X_list
  # # check X right format 
  # str(X)
  #  
  
  
  if (Model_type == "latent_trait") { 
    n_params_main <- (n_class - 1) + sum(unlist(n_covariates_per_outcome_mat)) + n_class * n_tests 
  } else if (Model_type != "latent_trait")  { 
    n_params_main <- (n_class - 1) + sum(unlist(n_covariates_per_outcome_mat)) + n_class * choose(n_tests, 2)
  }
  
  
  
  
  prior_a_mean  <-  vector("list", length = n_class)
  prior_a_sd <-  vector("list", length = n_class)

  prior_a_mean_mat  <-   array(0,  dim = c(n_covariates_max, n_tests))
  prior_a_sd_mat  <-    array(1,  dim = c(n_covariates_max, n_tests))

  for (c in 1:n_class) {
    prior_a_mean[[c]] <- prior_a_mean_mat
    prior_a_sd[[c]] <- prior_a_sd_mat
  }

  # intercepts / coeffs prior means
  prior_a_mean[[1]][1, 1] <- -2.10
  prior_a_sd[[1]][1, 1] <- 0.45

  prior_a_mean[[2]][1, 1] <- +0.40
  prior_a_sd[[2]][1, 1] <-  0.375
  
  k_choose_2 <- choose(n_tests, 2)
  km1_choose_2 = 0.5 * (n_tests - 2) * (n_tests - 1)
  known_num = 0
  
  beta_vec_init <- rep(0, n_class * n_tests)
  beta_vec_init[1:n_tests] <- - 1   # tests 1-T, class 1 (D-)
  beta_vec_init[(n_tests + 1):(2*n_tests)] <- + 1   # tests 1-T, class 1 (D+)
  
  n_obs <- N * n_tests
  prev_est <- sum(y) / (n_obs)
  p_raw <- atanh(2*prev_est - 1) 
  off_raw <- list()
  col_one_raw <- list()
  
  for (i in 1:2) {
    off_raw[[i]] <-  (c(rep(0.01, km1_choose_2 - known_num)))
    col_one_raw[[i]] <-  (c(rep(0.01, n_tests - 1)))
  }
  
  u_raw <- array(0.01, dim = c(N, n_tests))
  
  
  if (Model_type == "LC_MVP") {
      
          init = list(
            u_raw = (u_raw),
            p_raw =  (-0.6931472), # equiv to 0.20 on p
            beta_vec = beta_vec_init,
            off_raw = off_raw,
            col_one_raw =  col_one_raw
          )
        
  } else if (Model_type == "latent_trait") { 
    
          LT_a_vec_init <- rep(0, n_class * n_tests)
          LT_a_vec_init[1:n_tests] <- - 1   # tests 1-T, class 1 (D-)
          LT_a_vec_init[(n_tests + 1):(2*n_tests)] <- + 1   # tests 1-T, class 1 (D+)
          
          init = list(
            u_raw = (u_raw),
            p_raw =  (-0.6931472), # equiv to 0.20 on p
            LT_a_vec = LT_a_vec_init,
            LT_b_raw_vec = rep(-3.00, n_class * n_tests) 
          )
        
  }
  
  
  
  
}

  



## -----------  initialise model / inits etc
# based on (informal) testing, more than 8 burnin chains seems unnecessary 
# and probably not worth the extra overhead (even on a 96-core AMD EPYC Genoa CPU)
  n_chains_burnin <- 2
  init_lists_per_chain <- rep(list(init), n_chains_burnin) 
  
   
  
  ## make model_args_list (note: Stan models don't need this)
  if (Model_type == "LC_MVP") {
    
      model_args_list  <- list(       lkj_cholesky_eta = c(12, 3), 
                                      n_covariates_per_outcome_mat = n_covariates_per_outcome_mat,  # only needed if want to include covariates
                                      #X = X, # only needed if want to include covariates
                                      num_chunks =   BayesMVP:::find_num_chunks_MVP(N, n_tests),
                                      prior_coeffs_mean_mat = prior_a_mean,
                                      prior_coeffs_sd_mat =    prior_a_sd, 
                                      prev_prior_a = 1, # show how to change this later
                                      prev_prior_b = 1  # show how to change this later
                               )
  
  } else if (Model_type == "latent_trait") { 
    
    model_args_list <- list(         LT_b_priors_shape = LT_b_priors_shape,
                                     LT_b_priors_scale = LT_b_priors_scale,
                                     prior_coeffs_mean_mat = prior_a_mean,
                                     prior_coeffs_sd_mat =    prior_a_sd, 
                                     prev_prior_a = 1,  # show how to change this later
                                     prev_prior_b = 1   # show how to change this later
                            ) 
  }
  
  
  
 
  
 #   source(file.path(pkg_dir, "examples/load_R_packages.R"))
  
  #  setup_env()
  
  ###  -----------  Compile + initialise the model using "MVP_model$new(...)" 
  model_obj <- BayesMVP::MVP_model$new(   Model_type = Model_type,
                                          y = y,
                                          N = N,
                                          model_args_list = model_args_list, # this arg is only needed for BUILT-IN (not Stan) models
                                          init_lists_per_chain = init_lists_per_chain,
                                          sample_nuisance = TRUE,
                                          n_chains_burnin = n_chains_burnin,
                                          n_params_main = n_params_main,
                                          n_nuisance = n_nuisance)
  
  
  
  
  ## ----------- Set soe basic sampler settings
  {
    ### seed <- 123
    n_chains_sampling <- 1
    n_superchains <- 1 ## round(n_chains_sampling / n_chains_burnin) # Each superchain is a "group" or "nest" of chains. If using ~8 chains or less, set this to 1. 
    n_iter <- 1000                                 
    n_burnin <- 500
    adapt_delta <- 0.80
    learning_rate <- 0.05
    diffusion_HMC <- FALSE
    n_nuisance_to_track <- 10 # set to some small number (< 10) if don't care about making inference on nuisance params (which is most of the time!)
  }
  
  
  
  #### ------ sample model using "  model_obj$sample()" --------- 
  ##  NOTE: You can also use "model_obj$sample()" to update the model.
  ## For example, if using the same model but a new/different dataset
  ## (so new y and N, and n_nuisance needed), you can do:
  ## model_obj$sample(y = y, N = N, n_nuisance = n_nuisance, ...)
  ## You can also update model_args_list. 
  ##  for example, let's say I wanted to change the prior for disease prevalence
  ## to be informative s.t. prev ~ beta(5, 10). 
  ##  I could do this by modifying model_args_list:
  model_args_list$prev_prior_a <- 5
  model_args_list$prev_prior_b <- 10
  
 #   model_obj$init_object$Model_args_as_Rcpp_List$Model_args_ints
  
  ###  parallel::mcparallel  
  # then update model if needed parameters changed
  
  
  RcppParallel::setThreadOptions(numThreads = n_chains_sampling);
  
  
  model_samples <-            safe_test_wrapper_1(  n_covariates_per_outcome_mat = n_covariates_per_outcome_mat,
                                                    setup_env = setup_env,
                                                    update_model = BayesMVP:::update_model,
                                                    model_obj = model_obj,
                                                    n_iter = n_iter,
                                                    y = y,
                                                    N = N,
                                                    init_lists_per_chain = init_lists_per_chain,
                                                    n_params_main = n_params_main,
                                                    n_nuisance = n_nuisance,
                                                    model_args_list = model_args_list,
                                                    n_burnin = n_burnin,
                                                    n_chains_burnin = n_chains_burnin,
                                                    n_superchains = n_superchains,
                                                    diffusion_HMC = diffusion_HMC,
                                                    adapt_delta = adapt_delta,
                                                    learning_rate = learning_rate,
                                                    n_nuisance_to_track = n_nuisance_to_track,
                                                    
                                                  expr = {    setup_env() 
                                                    
                                       model_samples <-   ( model_obj$sample(   seed = 1,
                                                                                interval_width_main = 500,
                                                      n_iter = n_iter,
                                                      clip_iter = 500,
                                                      y = y,
                                                      N = N,
                                                      tau_mult = 2.0,
                                                #     force_recompile = force_recompile,
                                                      init_lists_per_chain = init_lists_per_chain,
                                                      n_params_main = n_params_main,
                                                      n_nuisance = n_nuisance,
                                                      model_args_list = model_args_list,
                                                      partitioned_HMC = TRUE,
                                                      vect_type = "Stan",
                                                     #  vect_type = "AVX512",
                                                     #  vect_type = "AVX2",
                                                      ratio_M_us = 0.25,
                                                      ratio_M_main = 0.25,
                                                      parallel_method = "RcppParallel",
                                                      n_burnin = n_burnin,
                                                      n_chains_burnin = n_chains_burnin,
                                                      sample_nuisance = TRUE,
                                                      n_chains_sampling = n_chains_sampling,
                                                      n_superchains = n_superchains,
                                                       diffusion_HMC = diffusion_HMC,
                                                      adapt_delta = adapt_delta,
                                                      learning_rate = learning_rate,
                                                      force_autodiff = TRUE,
                                                      force_PartialLog = TRUE,
                                                      multi_attempts = FALSE,
                                                      metric_shape_main = "dense",
                                                      metric_type_main = "Hessian",
                                                      n_nuisance_to_track = n_nuisance_to_track)  ) 
                                       
                                       } )
  
  model_samples
  
       ## model_samples <- parallel::mccollect(model_samples)
  
  ##    source(file.path(pkg_dir, "examples/load_R_packages.R"))
  
  #### --- MODEL RESULTS SUMMARY + DIAGNOSTICS -------------------------------------------------------------
  # after fitting, call the "summary()" method to compute + extract e.g. model summaries + traces + plotting methods 
  # model_fit <- model_samples$summary() # to call "summary()" w/ default options 
  require(bridgestan)
  model_fit <- model_samples$summary(save_log_lik_trace = FALSE, 
                                     compute_nested_rhat = FALSE
                                    # compute_transformed_parameters = FALSE
                                     ) 
  
  
  # x <- matrix(runif(n = 100, min = 0, max = 1))
  # BayesMVP:::Rcpp_wrapper_EIGEN_double(x = x, fn = "log", vect_type = "AVX2", skip_checks = FALSE) - 
  # BayesMVP:::Rcpp_wrapper_EIGEN_double(x = x, fn = "log", vect_type = "AVX512", skip_checks = FALSE)
  # 
  # 
   
  # extract # divergences + % of sampling iterations which have divergences
  model_fit$get_divergences()
  
  # HMC_info <- model_fit$get_HMC_info()
  # L_main <- HMC_info$tau_main/HMC_info$eps_main
  # L_us <- HMC_info$tau_us/HMC_info$eps_us
  # 
  # n_grad_evals_main_sampling <- n_iter * n_chains_sampling * L_main
  # n_grad_evals_us_sampling <- n_iter * n_chains_sampling * L_us
  # n_grad_evals_total_sampling <- n_grad_evals_us_sampling + n_grad_evals_main_sampling
  # 
  # Min_ESS <- model_fit$get_efficiency_metrics()$Min_ESS_main
  # 
  # 
  # 
  # ESS_per_grad_total_sampling <- Min_ESS / (n_grad_evals_total_sampling / 2) ; ESS_per_grad_total_sampling
  # 1.32 / (ESS_per_grad_total_sampling * 1000)
  # 
  

  ###### --- TRACE PLOTS  ----------------------------------------------------------------------------------
  # trace_plots_all <- model_samples$plot_traces() # if want the trace for all parameters 
  trace_plots <- model_fit$plot_traces(params = c("beta", "Omega", "p"), 
                                       batch_size = 12)
  
  # you can extract parameters by doing: "trace$param_name()". 
  # For example:
  # display each panel for beta and Omega ("batch_size" controls the # of plots per panel. Default = 9)
  trace_plots$beta[[1]] # 1st (and only) panel
  
  
  # display each panel for Omega ("batch_size" controls the # of plots per panel.  Default = 9)
  trace_plots$Omega[[1]] # 1st panel
  trace_plots$Omega[[2]] # 2nd panel
  trace_plots$Omega[[3]] # 3rd panel
  trace_plots$Omega[[4]] # 4th panel
  trace_plots$Omega[[5]] # 5th (and last) panel
  
 
  
  ###### --- POSTERIOR DENSITY PLOTS -------------------------------------------------------------------------
  # density_plots_all <- model_samples$plot_densities() # if want the densities for all parameters 
  # Let's plot the densities for: sensitivity, specificity, and prevalence 
  density_plots <- model_fit$plot_densities(params = c("Se_bin", "Sp_bin", "p"), 
                                            batch_size = 12)
  
  
  # you can extract parameters by doing: "trace$param_name()". 
  # For example:
  # display each panel for beta and Omega ("batch_size" controls the # of plots per panel. Default = 9)
  density_plots$Se[[1]] # Se - 1st (and only) panel
  density_plots$Sp[[1]] # Sp - 1st (and only) panel
  density_plots$p[[1]] # p (prevelance) - 1st (and only) panel
  
  
  
  ###### --- OTHER FEATURES -------------------------------------------------------------------------
  ## The "model_summary" object (created using the "$summary()" method) contains many useful objects. 
  ## For example:
  require(dplyr)
  # nice summary tibble for main parameters, includes ESS/Rhat, etc
  model_fit$get_summary_main() %>% print(n = 50) 
  # nice summary tibble for transformed parameters, includes ESS/Rhat, etc
  model_fit$get_summary_transformed() %>% print(n = 150) 
  # nice summary tibble for generated quantities, includes ESS/Rhat, etc (for LC-MVP this includes Se/Sp/prevalence)
  model_fit$get_summary_generated_quantities () %>% print(n = 150) 
 
  ## users can also easily use the "posterior" R package to compute their own statistics. 
  ## For example:
  # let's say we want to compute something not included in the default
  # "$summary()" method of BayesMVP, such as tail-ESS.
  # We can just use the posterior R package to compute this:
  require(posterior)  
  ## first extract the trace array object (note: already in a posterior-compatible format!)
  posterior_draws <- model_fit$get_posterior_draws()
  # then compute tail-ESS using posterior::ess_tail:
  posterior::ess_tail(posterior_draws[,,"Se_bin[1]"])
  
  ## You can also get the traces as tibbles (stored in seperate tibbles for main params, 
  ## transformed params, and generates quantities) using the "$get_posterior_draws_as_tibbles()" method:
  tibble_traces  <- model_fit$get_posterior_draws_as_tibbles()
  tibble_trace_main <- tibble_traces$trace_as_tibble_main_params
  tibble_trace_transformed_params <- tibble_traces$trace_as_tibble_transformed_params
  tibble_trace_generated_quantities <- tibble_traces$trace_as_tibble_generated_quantities
  
  ## You can also easily extract model run time / efficiency information using the "$get_efficiency_metrics()" method:
  model_efficiency_metrics <- model_fit$get_efficiency_metrics()
  time_burnin <- model_efficiency_metrics$time_burnin  ; time_burnin
  time_sampling <- model_efficiency_metrics$time_sampling ; time_sampling
  time_total_MCMC <- model_efficiency_metrics$time_total_MCMC  ; time_total_MCMC
  time_total_inc_summaries <- model_efficiency_metrics$time_total_inc_summaries ; time_total_inc_summaries # note this includes time to compute R-hat, etc 
  
  # We can also extract some more specific efficiency info, again using the "$get_efficiency_metrics()" method:
  Min_ESS_main_params <- model_efficiency_metrics$Min_ESS_main   ; Min_ESS_main_params
  Min_ESS_per_sec_sampling <- model_efficiency_metrics$Min_ESS_per_sec_samp ; Min_ESS_per_sec_sampling
  Min_ESS_per_sec_overall <- model_efficiency_metrics$Min_ESS_per_sec_total ; Min_ESS_per_sec_overall
  
  Min_ESS_per_grad <- model_efficiency_metrics$Min_ESS_per_grad_sampling ; Min_ESS_per_grad
  grad_evals_per_sec <- model_efficiency_metrics$grad_evals_per_sec ; grad_evals_per_sec
  
  ## extract the "time to X ESS" - these are very useful for knowing how long to
  #  run your model for. 
  est_time_to_100_ESS <- model_efficiency_metrics$est_time_to_100_ESS_inc_summaries ; est_time_to_100_ESS
  est_time_to_1000_ESS <- model_efficiency_metrics$est_time_to_1000_ESS_inc_summaries ; est_time_to_1000_ESS
  est_time_to_10000_ESS <- model_efficiency_metrics$est_time_to_10000_ESS_inc_summaries; est_time_to_10000_ESS
  
  ##  You can also use the "model_samples$time_to_ESS()" method to estimate 
  ## "time to X ESS" for general X:
  ##  For example let's say we determined our target (min) ESS to be ~5000:
  est_time_5000_ESS <- model_fit$time_to_target_ESS(target_ESS = 5000) ; est_time_5000_ESS
  est_time_5000_ESS
  
  ### You can also extract the log_lik trace (note: for Stan models this will only work )
  log_lik_trace <- model_fit$get_log_lik_trace()
  str(log_lik_trace) # will be NULL unless you specify  "save_log_lik_trace = TRUE" in the "$summary()" method
  ## can then use log_lik_trace e.g. to compute LOO-IC using the loo package 
  
  
  # 
  # 
  # sample_obj <- model_samples$model_results
  # 
  #   model_summary_outs <-      create_stan_summary(   model_results = sample_obj,
  #                                                     init_model_and_vals_object = init_model_and_vals_object,
  #                                                     n_nuisance = n_nuisance,
  #                                                     compute_main_params = TRUE,
  #                                                     compute_generated_quantities = TRUE,
  #                                                     compute_transformed_parameters = TRUE,
  #                                                     save_log_lik_trace = TRUE,
  #                                                     save_nuisance_trace = FALSE)
  #   
 
# #       
# #       
# #       
#
#
#
#     # compute main_params, generated quantities AND transformed parameters
#   model_summary_outs <-      create_stan_summary(   model_results = sample_obj,
#                                                     init_model_and_vals_object = init_model_and_vals_object,
#                                                     n_nuisance = n_nuisance,
#                                                     compute_main_params = TRUE,
#                                                     compute_generated_quantities = TRUE,
#                                                     compute_transformed_parameters = TRUE,
#                                                     save_log_lik_trace = TRUE,
#                                                     save_nuisance_trace = FALSE)


  
  

 ##   Rcpp::sourceCpp("~/Documents/Work/PhD_work/R_packages/BayesMVP/src/main_v9.cpp")

 
  
  
  
  {
    
    set.seed(1)
    
    init_model_and_vals_object <- model_obj$init_object
    
    
    {
       # sample_obj <- model_samples$model_results
        #model_results = sample_obj
        init_object <- model_obj$init_object
        compute_main_params = TRUE
        compute_generated_quantities = TRUE
        compute_transformed_parameters = TRUE
        save_log_lik_trace = TRUE
        save_nuisance_trace = FALSE
    }
    
    options(scipen = 99999)
     

    

        n_us <- N*n_tests
        n_params <- n_us + n_params_main
        index_us <- 1:n_us
        index_main <- (1 + n_us):n_params

        if (Model_type == "latent_trait") {
          n_corrs <- n_tests * 2
        } else {
          n_corrs <- 2 * choose(n_tests, 2)
        }

        theta_vec <- rep(0.01, n_params)

        Model_args_as_Rcpp_List <- init_model_and_vals_object$Model_args_as_Rcpp_List

        num_chunks <- 1
        Model_args_as_Rcpp_List$Model_args_ints[4, 1] <- num_chunks
        Model_args_as_Rcpp_List$Model_args_ints
        Model_args_as_Rcpp_List$Model_args_doubles
        Model_args_as_Rcpp_List$Model_args_strings
        Model_args_as_Rcpp_List$Model_args_bools

        Model_args_as_Rcpp_List$Model_args_doubles[3, 1] <- +0.5
        Model_args_as_Rcpp_List$Model_args_doubles[4, 1] <- -0.5

        Model_args_as_Rcpp_List$Model_args_strings[c(1, 4,5,7,8,9,10,11), 1]  <- "Stan"
        
        Model_args_as_Rcpp_List$Model_args_bools["debug",1] <- FALSE

        Model_args_as_Rcpp_List$Model_args_strings[2, 1] <- "Phi"
        Model_args_as_Rcpp_List$Model_args_strings[3, 1] <- "inv_Phi"

        Model_args_as_Rcpp_List$Model_args_strings[13,1] <- "Phi"

       # Model_args_as_Rcpp_List$Model_args_col_vecs_double[[1]] <- matrix(Model_args_as_Rcpp_List$Model_args_col_vecs_double[[1]])
        Model_args_as_Rcpp_List$Model_args_vecs_of_mats_double

        Model_args_as_Rcpp_List$Model_args_vecs_of_mats_int
        Model_args_as_Rcpp_List$Model_args_vecs_of_col_vecs_int

        theta_vec <- rnorm(n = n_params, mean = 0, sd = 1)
 
        theta_vec[(n_us + 1):n_params] <- rnorm(n = n_params_main, mean = 0, sd = 1 )

        Model_args_as_Rcpp_List$Model_args_doubles[3, 1] <- +5 # overflow - this one seems OK 
        Model_args_as_Rcpp_List$Model_args_doubles[4, 1] <- -5 # underflow - need to fix 

  }
 

  
  
   ## Rcpp::sourceCpp("~/Documents/Work/PhD_work/R_packages/BayesMVP/src/main.cpp")
  
   ##  dyn.load("C:\\Users\\enzoc\\Documents\\BayesMVP\\inst\\dummy_stan_model_win_model.dll")
   
  #   tic()
  # for (i in 1:1000)
   
   BayesMVP::detect_vectorization_support()
   
   
  require(BayesMVP)
  lp_grad_outs <-   safe_test_wrapper_1 ( theta_vec = theta_vec,
                                        index_main = index_main,
                                        index_us = index_us,
                                        y = y,
                                        Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
    
    expr = {
      
      require(BayesMVP)
      # 
      # Sys.setenv(BRIDGESTAN="C:/Users/enzoc/.bridgestan/bridgestan-2.5.0")
      # 
      # dummy_data_N <-  100
      # dummy_data_vec <- rnorm(dummy_data_N)
      # dummy_data <- list(N = dummy_data_N, y = dummy_data_vec )
      # # convert data to JSON format (use cmdstanr::write_stan_json NOT jsonlite::tload_and_run_log_prob_grad_all_StanoJSON)
      # r_data_list <- dummy_data
      # r_data_JSON <- tempfile(fileext = ".json")
      # cmdstanr::write_stan_json(r_data_list, r_data_JSON)
      # 
      # Sys.setenv(STAN_THREADS="true")
      # model <- bridgestan::StanModel$new("C:/users/enzoc/Downloads/dummy_stan_model_win.stan",
      #                                    data = r_data_JSON, 
      #                                    seed = 1234
      # )
      # print(paste0("This model's name is ", model$name(), "."))
      # print(paste0("This model has ", model$param_num(), " parameters."))
      # 
      # res <- model$log_density_gradient(1, jacobian = TRUE)
      
      
     # cat("  PATH: ", Sys.getenv("PATH"), "\n")
     # cat("  libPaths: ", paste(.libPaths(), collapse = "; "), "\n")
     # cat("  Working Directory: ", getwd(), "\n")
      
      cat("Preloading critical DLLs for BayesMVP package\n")
      
      # List of DLLs to preload
      dll_paths <- c(
        # "C:/Users/enzoc/Documents/BayesMVP/inst/tbb12.dll",
        "C:/Users/enzoc/Documents/BayesMVP/inst/BayesMVP/inst/tbb_stan/tbb.dll",
        "C:/Users/enzoc/Documents/BayesMVP/inst/BayesMVP/inst/dummy_stan_model_win_model.so",
        "C:/Users/enzoc/Documents/BayesMVP/inst/BayesMVP/inst/dummy_stan_model_win_model.dll",
        "C:/Users/enzoc/Documents/BayesMVP/inst/BayesMVP/inst/BayesMVP.dll"
      )
      
      # Attempt to load each DLL
      for (dll in dll_paths) {
        
        tryCatch(
          {
            dyn.load(dll)
            cat("  Loaded:", dll, "\n")
          },
          error = function(e) {
            cat("  Failed to load:", dll, "\n  Error:", e$message, "\n")
          }
        )
        
      }
      
      
   
    
    
    BayesMVP::Rcpp_wrapper_fn_lp_grad( Model_type = "LC_MVP",
                                                                     force_autodiff = TRUE,
                                                                     force_PartialLog = TRUE,
                                                                     multi_attempts = FALSE,
                                                                     theta_main_vec = theta_vec[index_main],
                                                                     theta_us_vec = theta_vec[index_us],
                                                                     y = y,
                                                                     grad_option = "all",
                                                                     Model_args_as_Rcpp_List = Model_args_as_Rcpp_List) 
    
    } )
  
 
  
  lp_grad_outs
  # lp_grad_outs$job
  
  
 ## toc()

  x <- matrix(runif(n = 100))
  safe_test_wrapper_1( x =  x, expr = BayesMVP::Rcpp_wrapper_EIGEN_double(x = x, fn  = "log", vect_type = "AVX2", skip_checks = FALSE))
  safe_test_wrapper_1( x =  x, expr = BayesMVP::Rcpp_wrapper_EIGEN_double(x = x, fn  = "log", vect_type = "Stan", skip_checks = FALSE))

  outs_AD <- parallel::mccollect(lp_grad_outs)
  outs_AD
  
  
  Model_args_as_Rcpp_List$Model_args_bools
  Model_args_as_Rcpp_List$Model_args_ints
  Model_args_as_Rcpp_List$Model_args_doubles
  Model_args_as_Rcpp_List$Model_args_strings
  
  ## Model_args_as_Rcpp_List$Model_args_col_vecs_double[[1]] <- matrix(Model_args_as_Rcpp_List$Model_args_col_vecs_double[[1]])
  Model_args_as_Rcpp_List$Model_args_col_vecs_double
  Model_args_as_Rcpp_List$Model_args_mats_double
  Model_args_as_Rcpp_List$Model_args_mats_int
  
  Model_args_as_Rcpp_List$Model_args_vecs_of_mats_double
  Model_args_as_Rcpp_List$Model_args_vecs_of_mats_int

 sum(unlist(Model_args_as_Rcpp_List$Model_args_2_later_vecs_of_mats_double[[1]]))
 
  lp_grad_outs <-   parallel::mcparallel(  Rcpp_wrapper_fn_lp_grad( Model_type = Model_type,
                                                                     force_autodiff = FALSE,
                                                                     force_PartialLog = FALSE,
                                                                     multi_attempts = F,
                                                                     theta_main_vec = theta_vec[index_main],
                                                                     theta_us_vec = theta_vec[index_us],
                                                                     y = y,
                                                                     grad_option = "none",
                                                                     Model_args_as_Rcpp_List = Model_args_as_Rcpp_List))
  
  
  y
 
  

  out <- parallel::mccollect(lp_grad_outs)
  
  
   #out
 ###  head(out[[1]][-1], 2500)

  out[[1]][1]
  outs_AD[[1]][1]

  -17796.95
  
  out[[1]][1] - outs_AD[[1]][1] # log_prob diff

  (out[[1]][1] ) > abs( outs_AD[[1]][1] )

  sum(  head(out[[1]][-1], 1000000000)  -  head(outs_AD[[1]][-1], 1000000000) )  # u'#s grad diff (sum)

  head(out[[1]][-1], 50000)  -  head(outs_AD[[1]][-1], 50000)  ### 
  
  abs(  head(out[[1]][-1], 50000)  -  head(outs_AD[[1]][-1], 50000) ) /   head(outs_AD[[1]][-1], 50000)
  
  tail(head(out[[1]][-1], 50000), 31) 
  tail(head(outs_AD[[1]][-1], 50000), 31)
  
 
  
  tail(head(out[[1]][-1], 50000), 31)  - 
  tail(head(outs_AD[[1]][-1], 50000), 31)
  
  # sum(out[[1]])
  # sum(outs_AD[[1]])

  grad_us_manual <- head(out[[1]][-1], n_us)
  grad_us_AD <- head(outs_AD[[1]][-1], n_us)
  
  sum( (grad_us_manual - grad_us_AD) )
  
  
  grad_us_manual - grad_us_AD
  sum(grad_us_manual - grad_us_AD)
  
  # head(grad_us_AD, 1000)
  # head(grad_us_manual, 1000)
  
  sum(grad_us_AD)
  
  head(grad_us_manual - grad_us_AD, 1000000)
  
  sum( (grad_us_manual - grad_us_AD)[1501:2500] )
  
  sum( abs(grad_us_manual - grad_us_AD) )
  
  # (grad_us_manual - grad_us_AD)
  # 
  # head(grad_us_manual - grad_us_AD, 2000)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  test_log_abs_sum_exp <- function() {
    
    # Test case 1: Simple numbers
    log_abs1 <- matrix(c(log(2), log(3),
                         log(4), log(5)), nrow=2, byrow=TRUE)  # Note byrow=TRUE
    signs1 <- matrix(c(1, 1,
                       1, -1), nrow=2, byrow=TRUE)
    
    # Direct calculation for test case 1
    row1_sum <- 2 + 3     # = 5
    row2_sum <- 4 - 5     # = -1
    expected1 <- list(
      log_sum = c(log(abs(row1_sum)), log(abs(row2_sum))),  # c(log(5), log(1))
      signs = c(sign(row1_sum), sign(row2_sum))             # c(1, -1)
    )
    
    # Test case 2: Potential cancellations
    log_abs2 <- matrix(c(log(1), log(1), log(1),
                         log(1000), log(1000), log(1000)), nrow=2, byrow=TRUE)
    signs2 <- matrix(c(1, -1, 1,
                       1, 1, -1), nrow=2, byrow=TRUE)
    
    # Direct calculation for test case 2
    row1_sum2 <- 1 - 1 + 1     # = 1
    row2_sum2 <- 1000 + 1000 - 1000  # = 1000
    expected2 <- list(
      log_sum = c(log(abs(row1_sum2)), log(abs(row2_sum2))),
      signs = c(sign(row1_sum2), sign(row2_sum2))
    )
    
    # Test case 3: Large numbers
    log_abs3 <- matrix(c(700, 700, 700,
                         -700, -700, -700), nrow=2, byrow=TRUE)
    signs3 <- matrix(c(1, -1, 1,
                       1, 1, -1), nrow=2, byrow=TRUE)
    
    # Call the wrapper function
    result1 <- BayesMVP::fn_Rcpp_wrapper_log_abs_sum_exp_general(log_abs1, signs1, "Loop", "Loop")
    result2 <- BayesMVP::fn_Rcpp_wrapper_log_abs_sum_exp_general(log_abs2, signs2, "Loop", "Loop")
    result3 <- BayesMVP::fn_Rcpp_wrapper_log_abs_sum_exp_general(log_abs3, signs3, "Loop", "Loop")
    
    # Print results with more detail
    cat("Test Case 1 (Simple numbers):\n")
    cat("Input matrix log_abs:\n")
    print(log_abs1)
    cat("Input matrix signs:\n")
    print(signs1)
    cat("Expected row 1: sum=", row1_sum, "log=", log(abs(row1_sum)), "sign=", sign(row1_sum), "\n")
    cat("Expected row 2: sum=", row2_sum, "log=", log(abs(row2_sum)), "sign=", sign(row2_sum), "\n")
    cat("Got log_sums:", result1[[1]], "\n")
    cat("Got signs   :", result1[[2]], "\n\n")
    
    cat("Test Case 2 (Cancellations):\n")
    cat("Input matrix log_abs:\n")
    print(log_abs2)
    cat("Input matrix signs:\n")
    print(signs2)
    cat("Expected row 1: sum=", row1_sum2, "log=", log(abs(row1_sum2)), "sign=", sign(row1_sum2), "\n")
    cat("Expected row 2: sum=", row2_sum2, "log=", log(abs(row2_sum2)), "sign=", sign(row2_sum2), "\n")
    cat("Got log_sums:", result2[[1]], "\n")
    cat("Got signs   :", result2[[2]], "\n\n")
    
    return(list(
      test1 = list(result = result1, expected = expected1),
      test2 = list(result = result2, expected = expected2),
      test3 = result3,
      inputs = list(
        case1 = list(log_abs = log_abs1, signs = signs1),
        case2 = list(log_abs = log_abs2, signs = signs2),
        case3 = list(log_abs = log_abs3, signs = signs3)
      )
    ))
    
  }
  
 
  
  # Run tests
  results <- test_log_abs_sum_exp()
  
  # Additional numerical checks
  check_results <- function(results) {
    cat("\nNumerical Checks:\n")
    
    # Test 1 differences
    cat("Test 1 differences:\n")
    cat("Log sums:", results$test1$result[[1]] - results$test1$expected$log_sum, "\n")
    cat("Signs   :", results$test1$result[[2]] - results$test1$expected$signs, "\n")
    
    # Test 2 differences
    cat("\nTest 2 differences:\n")
    cat("Log sums:", results$test2$result[[1]] - results$test2$expected$log_sum, "\n")
    cat("Signs   :", results$test2$result[[2]] - results$test2$expected$signs, "\n")
  }
  
  # Run numerical checks
  check_results(results)
 
  

#   vec_1 <-   head(grad_us_manual, 12500)
#   vec_2 <-   head(grad_us_AD, 12500)
# 
#    vec_1 - vec_2
# #
#   tail(out[[1]][-1], 11)[1:10]  -  tail(outs_AD[[1]][-1], 11)[1:10] # coeffs grad diffs
#   tail(out[[1]][-1], n_params_main)[1:n_corrs]  -  tail(outs_AD[[1]][-1], n_params_main)[1:n_corrs] # corrs grad diffs
#   tail(out[[1]][-1], 1)   -  tail(outs_AD[[1]][-1], 1) # prevelance grad diff
#
#   tail(out[[1]][-1], 11)[1:10]
#   tail(outs_AD[[1]][-1], 11)[1:10]

# 
#   n_params_main
#   
#   tail(out[[1]][-1], n_params_main)[1:n_corrs] 
#   tail(outs_AD[[1]][-1], n_params_main)[1:n_corrs] 
#   
#   tail(out[[1]][-1], n_params_main)[1:n_corrs]  -  tail(outs_AD[[1]][-1], n_params_main)[1:n_corrs]
# 
#   tail(out[[1]][-1], 1)   -  tail(outs_AD[[1]][-1], 1)
# 
#   tail(out[[1]][-1], n_params_main)[1:n_corrs]
#   tail(outs_AD[[1]][-1], n_params_main)[1:n_corrs]
# 
#   # tail(outs_AD[[1]][-1], 11)[1:5]
#   # tail(out[[1]][-1], 11)[1:5]
#   #
#   # tail(outs_AD[[1]][-1], 11)[6:10]
#   # tail(out[[1]][-1], 11)[6:10]
#   
#   
#   Rcpp::sourceCpp("~/Documents/Work/PhD_work/R_packages/BayesMVP/src/main_v9.cpp")
#   
#  
#   
#   # sample_obj <- parallel::mccollect(sample_obj)[[1]]
#   # sample_obj
#   
# 
#   
#   
#   # print main params ('parameters' block)
#  # print(model_summary_outs$tibble_summary_df_wo_nuisance_and_log_lik, n = n_params_main)
# 
#   
#   
#   
#   
#   # Rcpp::sourceCpp(file.path(pkg_dir, "/src/main_v9.cpp"))
#   
#   
#   
#   
#   
#   
#   
#   # inits_list <- list(inits_coeffs =   init_burnin_object$inits_coeffs_vecs,
#   #                    inits_raw_corrs =   init_burnin_object$inits_raw_corrs_vecs,
#   #                   #  inits_LT_bs =   init_burnin_object$init_raw_prev_list, # for latent trait only
#   #                    inits_prev =   init_burnin_object$init_raw_prev_list
#   #                    )
#   # 
#   # 
#   
#   k_choose_2 <- choose(n_tests, 2)
#   km1_choose_2 = 0.5 * (n_tests - 2) * (n_tests - 1)
#   known_num = 0
#   
#   beta_vec_init <- rep(0, n_class * n_tests)
#   beta_vec_init[1:n_tests] <- - 1   # tests 1-T, class 1 (D-)
#   beta_vec_init[(n_tests + 1):(2*n_tests)] <- + 1   # tests 1-T, class 1 (D+)
#   
#   n_obs <- N * n_tests
#   prev_est <- sum(y) / (n_obs)
#   prev_raw_init <- atanh(2*prev_est - 1) 
#   # note: inits DONT need to be in any specific order - however the parameter declaration block in Stan does (nuisance first, rest later)
#   init = list(
#     u_raw = (array(0.01, dim = c(N, n_tests))),
#     beta_vec = beta_vec_init,
#     L_Omega_raw = array(0.01, dim = c(n_class, k_choose_2 - known_num)), 
#     p_raw = prev_raw_init   
#   )
#   
#  
#   init_lists_per_chain <- rep(list(init), n_chains_burnin) 
#   
#     hard_coded_model_outs <- init_hard_coded_model(Model_type = "LC_MVP", y = y, X = X, model_args_list = model_args_list)
#     
#     hard_coded_model_outs$Model_args_as_Rcpp_List$Model_args_bools
#     
#     hard_coded_model_outs$model_args_list$n_covariates_per_outcome_vec
#    # hard_coded_model_outs$model_args_list$prior_coeffs_mean_vec
#   
#   init_model_object <- init_model(Model_type = "LC_MVP",
#                                   y = y, 
#                                   X = X, 
#                                   model_args_list = model_args_list)
#   
#   init_model_object$model_args_list$model_args_list$prior_coeffs_mean_vec
#   init_model_object$model_args_list$model_args_list$prior_coeffs_sd_vec
#   init_model_object$model_args_list$model_args_list$n_covariates_per_outcome_vec
#   init_model_object$model_args_list$model_args_list$lkj_cholesky_eta
#   init_model_object$model_args_list$model_args_list$prev_prior_a
#   
#   #  (init_model_object$X)
#   
#   init_vals_object <- init_inits(init_model_outs = init_model_object,
#                                  init_lists_per_chain = init_lists_per_chain,
#                                  n_chains_burnin = 8,
#                                  n_params_main = n_params_main,
#                                  n_nuisance = n_nuisance)
#   
# 
#      Rcpp::sourceCpp("~/Documents/Work/PhD_work/R_packages/BayesMVP/src/main_v9.cpp")
#   
#   #init_vals_object$
#   
#     init_vals_object$Model_args_as_Rcpp_List$Model_args_bools
#     init_vals_object$Model_args_as_Rcpp_List$Model_args_ints
#     init_vals_object$Model_args_as_Rcpp_List$Model_args_doubles
#     init_vals_object$Model_args_as_Rcpp_List$Model_args_strings
#     init_vals_object$Model_args_as_Rcpp_List$Model_args_strings[2,1] <-  "Phi"
#     init_vals_object$Model_args_as_Rcpp_List$Model_args_strings[3,1] <-  "inv_Phi"
#     
#     init_vals_object$Model_args_as_Rcpp_List$Model_args_col_vecs_double
#     init_vals_object$Model_args_as_Rcpp_List$Model_args_vecs_of_col_vecs_int[[1]]
#     init_vals_object$Model_args_as_Rcpp_List$Model_args_vecs_of_mats_double
#     init_vals_object$Model_args_as_Rcpp_List$Model_args_vecs_of_mats_int
#     
#     init_vals_object$Model_args_as_Rcpp_List$Model_args_2_later_vecs_of_mats_double
#     
#     
#     
#  
#      # parallel::mcparallel
#     
#   init_burnin_object <-    (init_and_run_burnin(  init_vals_object, 
#                                         n_chains_burnin = 8, 
#                                         n_params_main = n_params_main, 
#                                         n_nuisance = n_nuisance, 
#                                        # shrinkage_factor = 1,
#                                         diffusion_HMC = TRUE, 
#                                         metric_shape_main = "dense",
#                                        seed_R = 123,
#                                        n_burnin = 750,
#                                        adapt_delta = 0.85,
#                                        LR_main = 0.05,
#                                        LR_us = 0.05,
#                                        y = y,
#                                        Model_type = Model_type
#                                        ))
#   
#     init_burnin_object <- parallel::mccollect(init_burnin_object)
#    
#    init_burnin_object[[1]]$EHMC_args_as_Rcpp_List
#   
#   {
#     
#   theta_main_vectors_all_chains_input_from_R <- init_burnin_object$theta_main_vectors_all_chains_input_from_R
#   theta_us_vectors_all_chains_input_from_R <- init_burnin_object$theta_us_vectors_all_chains_input_from_R
#   
#   Model_args_as_Rcpp_List <- init_burnin_object$Model_args_as_Rcpp_List
#   EHMC_args_as_Rcpp_List <- init_burnin_object$EHMC_args_as_Rcpp_List
#   EHMC_Metric_as_Rcpp_List <- init_burnin_object$EHMC_Metric_as_Rcpp_List
#   EHMC_burnin_as_Rcpp_List <- init_burnin_object$EHMC_burnin_as_Rcpp_List
#   
#   }
#   
#   EHMC_Metric_as_Rcpp_List$M_dense_main
#   
#  # EHMC_Metric_as_Rcpp_List$M_inv_us_vec
#   min(EHMC_Metric_as_Rcpp_List$M_inv_us_vec)
#   
#   n_superchains <- 8
#   n_chains_sampling <- 64
#   partitioned_HMC <- TRUE
#   n_iter = 1000
#   seed = 123
#   n_nuisance_to_track <- 10
#   
#   
#   {
#     
#   
#     
#     source(file.path(pkg_dir, "/R/R_fns_post_burnin_prep_inits.R"))
#     
#     post_burnin_prep_inits <-  R_fn_post_burnin_prep_for_sampling(n_chains_sampling = n_chains_sampling,
#                                                                   n_superchains = n_superchains,
#                                                                   n_params_main = Model_args_as_Rcpp_List$n_params_main,
#                                                                   n_nuisance = Model_args_as_Rcpp_List$n_nuisance, 
#                                                                   theta_main_vectors_all_chains_input_from_R, 
#                                                                   theta_us_vectors_all_chains_input_from_R)
#     
#     theta_main_vectors_all_chains_input_from_R <- post_burnin_prep_inits$theta_main_vectors_all_chains_input_from_R
#     theta_us_vectors_all_chains_input_from_R <- post_burnin_prep_inits$theta_us_vectors_all_chains_input_from_R
#     
#     
#     
#   }
#    
#   
#   
#   
#   
#   {
#      
#   #  gc(reset = TRUE)
#     rm(result)
#     
#     
#     Model_args_as_Rcpp_List$Model_args_strings[c(1, 4,5,6,7,8,9,10,11),1] <- "AVX512"
#     # Model_args_as_Rcpp_List$Model_args_strings[c(1, 4,5,6,7,8,9,10,11),1] <- "AVX2"
#     #Model_args_as_Rcpp_List$Model_args_strings[c(1, 4,5,6,7,8,9,10,11),1] <- "Stan"
#     
#     Model_args_as_Rcpp_List$Model_args_strings[6,1] <- "Stan" # need to fix LSE for AVX
#     
# 
#     RcppParallel::setThreadOptions(numThreads = n_chains_sampling);
#     
#     tictoc::tic("post-burnin timer")
#    
#     
#     
#     result <-     (Rcpp_fn_RcppParallel_EHMC_sampling(  n_threads_R = n_chains_sampling,
#                                                         n_nuisance_to_track = n_nuisance_to_track,
#                                                         seed_R = seed,
#                                                         iter_one_by_one = FALSE,
#                                                         n_iter_R = n_iter,
#                                                         partitioned_HMC_R = partitioned_HMC,
#                                                         Model_type_R = Model_type,
#                                                         force_autodiff_R = FALSE,
#                                                         force_PartialLog = TRUE,
#                                                         multi_attempts_R = TRUE,
#                                                         theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
#                                                         theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R,
#                                                         y =  y, # rep(list(y), n_chains_sampling),
#                                                         Model_args_as_Rcpp_List =  Model_args_as_Rcpp_List,
#                                                         EHMC_args_as_Rcpp_List =   EHMC_args_as_Rcpp_List,
#                                                         EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List))
#     
#     
#     
#     # result <- parallel::mccollect(result) ; result <- result[[1]]
#     
#     
#     try({ 
#       {
#         print(tictoc::toc(log = TRUE))
#         log.txt <- tictoc::tic.log(format = TRUE)
#         tictoc::tic.clearlog()
#         time_sampling <- unlist(log.txt)
#       }
#     })
#     
#    # gc()
#     
#     try({
#       time_sampling <- as.numeric( substr(start = 0, stop = 100,  strsplit(  strsplit(time_sampling, "[:]")[[1]], "[s]")[[2]][1] ) )
#     })
#     
#     print(paste("sampling time = ",  time_sampling) )
#   #  print(paste("total time = ", time_burnin + time_sampling) )
#     
#   }
#    
#   
#  
#            
#            # Rcpp::sourceCpp(file.path(pkg_dir, "/src/main_v9.cpp"))
#      
# ### ---- Perform BURNIN  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
#          #  source(file.path(pkg_dir, "/R/R_fn_EHMC_burnin_v2.R"))
#  
#            
# ### -----------  sampling  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 
#   #  Rcpp::sourceCpp("~/Documents/Work/PhD_work/R_packages/BayesMVP/src/main_v9.cpp")
# 
#             
#  
#   
# ### -----------  summarise results   --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  
#   
# #{
# 
#   
# #   str(result)
# 
#  #  {
#     
#  
#       
#     {
#      # theta_trace_as_3D_array_pre <- result[[6]]
#       theta_trace_as_3D_array_pre <- result[[1]]
#       theta_nuisance_as_3D_array_pre <- result[[3]]
#      # str(theta_trace_as_3D_array_pre)
#       
#       theta_trace_3D_array <- array(dim = c(n_params_main, n_iter, n_chains_sampling))
#       theta_nuisance_as_3D_array <- array(dim = c(10, n_iter, n_chains_sampling))
#       
#       for (param in 1:n_params_main) {
#         for (kk in 1:n_chains_sampling) {
#         # theta_trace_3D_array[param,  , kk] <- theta_trace_as_3D_array_pre[[param]][kk,  ]
#          theta_trace_3D_array[param,  , kk] <- theta_trace_as_3D_array_pre[[kk]][param,  ]
#         }
#       }
#       
#       for (param in 1:10) {
#         for (kk in 1:n_chains_sampling) {
#           theta_nuisance_as_3D_array[param,  , kk] <- theta_nuisance_as_3D_array_pre[[kk]][param,  ]
#         }
#       }
#       
#   
#       
#  
#        div_trace  <- result[[2]]
#        
#        n_divs <- sum(unlist(div_trace)) ; n_divs
#        pct_divs <- 100 *  (n_divs / (length(div_trace) * n_iter) ) ; pct_divs
#     }
#              
#             try({ 
#                   ess_vec <- c()
#                   for (i in 1:n_params_main) {
#                     #ess_vec[i] <-  rstan::ess_bulk(  t(theta_trace_as_3D_array_pre[[i]]) )
#                     ess_vec[i] <-  rstan::ess_bulk(  (theta_trace_3D_array[i,,]))
#                   }
#                   ess_vec_nuisance <- c()
#                   for (i in 1:n_nuisance_to_track) {
#                       ess_vec_nuisance[i] <-  rstan::ess_bulk(theta_nuisance_as_3D_array[i,,])
#                   }
#             })
#             try({ 
#               Min_ess <- min(ess_vec)  ;  print(min(ess_vec))
#             #   Min_ess_nuisance_subset <- min(ess_vec_nuisance)  ;  print(min(ess_vec_nuisance))
#             })
#             try({ 
#               Min_ess_per_sec <-   (Min_ess/time_sampling)   ;  
#               print((paste("Min ESS / sec = ", signif(Min_ess_per_sec, 3))))
#               print((paste("Min ESS / sec [64 ch. estimated] = ", signif(Min_ess_per_sec * (64 / n_chains_sampling), 3))))
#             })
#             try({ 
#               n_grad_evals_sampling_main <-  (EHMC_args_as_Rcpp_List$tau_main / EHMC_args_as_Rcpp_List$eps_main)  * n_iter * n_chains_sampling
#               Min_ess_per_grad_main <-  Min_ess / n_grad_evals_sampling_main
#             })
#             try({
#               n_grad_evals_sampling_us <-  (EHMC_args_as_Rcpp_List$tau_us / EHMC_args_as_Rcpp_List$eps_us)  * n_iter * n_chains_sampling
#               Min_ess_per_grad_us <-  Min_ess / n_grad_evals_sampling_us
#             })
#             try({ 
#               weight_nuisance_grad <- 0.20
#               weight_main_grad <- 0.80
#               Min_ess_per_grad_overall_weighted <- (weight_nuisance_grad * Min_ess_per_grad_us + weight_main_grad * Min_ess_per_grad_main) / (weight_nuisance_grad + weight_main_grad)
#               comment(print(signif( 1000 * Min_ess_per_grad_overall_weighted, 3)))
#               print((paste("Min ESS / grad (weighted) = ", signif(1000 *  Min_ess_per_grad_overall_weighted, 3))))
#             })
#     
#        
#             
#                # n_grad_evals_main_total  <-  mean_L_burnin * n_chains_sampling  * n_burnin  + L_main * n_chains_sampling *  n_iter
#                # n_grad_evals_main_sampling <-   n_iter  * n_chains_sampling * L_main
#                # 
#                # Min_ESS_per_grad <- Min_ESS / n_grad_evals_main_total
#                # Min_ESS_per_grad_sampling <-  Min_ESS / n_grad_evals_main_sampling ;# print(round(Min_ESS_per_grad_sampling * 1000, 3))
#        
#        
#        
#   }
#   
#  
#   
#   {
#     
#     
#     theta_trace_3D_array_2 <- array(dim = c(n_params_main, n_chains_sampling, n_iter))
#     
#     for (i in 1:n_params_main) {
#       for (kk in 1:n_chains_sampling) {
#         for (ii in 1:n_iter) {
#          # theta_trace_3D_array_2[i, kk, ii] <- theta_trace_3D_array[[i]][ii, kk]
#           theta_trace_3D_array_2[i, kk, ii] <- theta_trace_3D_array[i, ii, kk]
#         }
#       }
#     }
#     
#     theta_trace_3D_array_pnorm <- pnorm(theta_trace_3D_array_2)
#     posterior_means_pnorm  <-  apply(apply(theta_trace_3D_array_pnorm, FUN = mean, c(1,2)) , FUN = mean,  1)
#     
#     theta_trace_3D_array_1m_pnorm <- 1 - pnorm(theta_trace_3D_array_2)
#     posterior_means_1m_pnorm  <-  apply(apply(theta_trace_3D_array_1m_pnorm, FUN = mean, c(1,2)) , FUN = mean,  1)
#     
#     theta_trace_3D_array_tanh_p1_div_2 <- (tanh( tail(theta_trace_3D_array_2, 1)) + 1)/2
#     posterior_means_tanh_p1_div_2  <-  apply(apply(theta_trace_3D_array_tanh_p1_div_2, FUN = mean, c(1,2)) , FUN = mean,  1)
#  
#     
#     print(100 * round( (tail(posterior_means_pnorm, n_tests * 2 + 1)[6:10]), 3))
#     print(100 * round( (tail(theta_trace_3D_array_1m_pnorm, n_tests * 2 + 1)[1:5]), 3))
#     print(100 * round( tail(posterior_means_tanh_p1_div_2, 1) , 3) )
#     
#      pct_divs <- sum(100 * n_divs) / length(div_trace)
#     print(paste("N_divergences = ",    round(n_divs) ))
#     print(paste("% divergences = ",    round(pct_divs, 3) ))
#     
#     # plot density of nuisance 
#     sd_1st_unc_u <- sd(nuisance_trace_3D_array[[1]][,1])
#     
#     par(mfrow = c(4, 4))
#     for (i in 1:9)  {
#       plot(density(nuisance_trace_3D_array[[i]][,]), xlim = c(-0.5, 0.5))
#     }
#     
#     Min_ess_nuisance_subset
#   
#     sd_1st_unc_u
#     
#     random_param_index <- sample(x = c(1:31), size = 7)
#     for (i in random_param_index)  {
#        plot(theta_trace_3D_array[[i]][,1])
#     }
#     
#     
#   }
#   
#   
#   ess_vec
#   
#   
#   
#   
#   
# }
# 
#           
#          
#                  Se_true_observed_list[[123]]  *100
#                  Sp_true_observed_list[[123]]  *100
#               prev_true_observed_list[[123]]*100
#  
# 
# 
#               
#               
#               
#               
#               
#                
#               
#               
#               
#               
#               
#               
#               