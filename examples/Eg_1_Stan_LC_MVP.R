
 
         
{
  
  rstan::rstan_options(auto_write = TRUE)
  options(scipen = 99999)
  options(max.print = 1000000000)
  #  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
  
}



# Run LC-MVP model (manual-gradients using SNAPER-diffusion-space HMC)   --------------------------------------------------------------------------------------------------------------------------------------------------
      
pkg_dir <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP"
   
## first sprecify model type (for Stan models see other Stan example file)
Model_type <- "Stan"

 
source(file.path(pkg_dir, "examples/BayesMVP_LC_MVP_prep.R"))
 

# 
# - | ----------  prepare Stan data and inits --------------------------------------------------------------------



## select N to use 
N <- 500
    
    
    {
      
      if (N == 500)     dataset_index = 1
      if (N == 1000)    dataset_index = 2
      if (N == 2500)    dataset_index = 3
      if (N == 5000)    dataset_index = 4
      if (N == 12500)   dataset_index = 5
      if (N == 25000)   dataset_index = 6
      
      y = y_master_list_seed_123_datasets [[dataset_index]]
      
      
      ## Set important variables
      n_tests <- ncol(y)
      n_class <- 2
      n_covariates <- 1
      n_nuisance <- N * n_tests
      n_params_main <-  choose(n_tests, 2) * 2 + 1 + n_tests * n_covariates * 2
      
      
      
      N_sims <- 123
      
    ###  model_type <- "LC_MVP"
      fully_vectorised <- 1  ;   GHK_comp_method <- 2 ;       handle_numerical_issues <- 1 ;     overflow_threshold <- +5 ;    underflow_threshold <- -5  #  MAIN MODEL SETTINGS 
      
       Phi_type <- 2 # using Phi_approx and inv_Phi_approx - in  Stan these are slower than Phi() and inv_Phi() !!!!!!!! (as log/exp are slow)
      #  Phi_type <- 1 # using Phi and inv_Phi
      
      corr_param <- "Sean" ; prior_lkj <- c(12, 3) ;  corr_prior_beta =  0 ;  corr_force_positive <-  0  ;  prior_param_3 <- 0 ; uniform_indicator <- 0 
      #  corr_param <- "Sean" ; prior_lkj <- c(10, 2) ;  corr_prior_beta =  0 ;  corr_force_positive <-  0  ;  prior_param_3 <- 0 ; uniform_indicator <- 0 
      
      CI <- 0
      prior_only <-  0
      corr_prior_norm = 0
      gamma_indicator <-  0
      skew_norm_indicator <- 0
      
      prior_lkj_skewed_diseased <-   prior_lkj
      prior_lkj_skewed_non_diseased <-  prior_lkj
      
      #  tailored_corr_priors <- TRUE # priors which are more consistent with posteior (should recover parameters better, especially for lower N and in the smaller latent class)
      tailored_corr_priors <- FALSE # priors which are more consistent with posteior (should recover parameters better, especially for lower N and in the smaller latent class)
      
      n_class <- 2
      n_covariates <- 1
      n_tests <-   n_tests 
      n_ordinal_tests <- 0
      n_binary_tests <- n_tests
      Thr =   rep(1, n_tests)
      #N <- nrow(y)
      
      prior_a_mean <-   array(0,  dim = c(n_class, n_tests, n_covariates))
      prior_a_sd  <-    array(1,  dim = c(n_class, n_tests, n_covariates))
      
      
      
      # intercepts / coeffs prior means
      prior_a_mean[1,1,1] <- -2.10
      prior_a_sd[1,1,1] <- 0.45
      
      prior_a_mean[2,1,1] <- +0.40
      prior_a_sd[2,1,1] <-  0.375
      
      n_pops <- 1
      group <- rep(1, N)
    
      
      n_covs_per_outcome = array(1, dim = c(n_class, n_tests))
      n_covariates_total_nd  =    (sum( (n_covs_per_outcome[1,])));
      n_covariates_total_d   =     (sum( (n_covs_per_outcome[2,])));
      n_covariates_total  =       n_covariates_total_nd + n_covariates_total_d;
      
      k_choose_2   = (n_tests * (n_tests - 1)) / 2;
      km1_choose_2 =  ((n_tests - 1) * (n_tests - 2)) / 2;
      
      i = 123
      
      
      n_covariates_max <- 1
      n_covariates_max_nd <- 1
      n_covariates_max_d <- 1
      
      X_nd <- list()
      for(t in 1:n_tests) {
        X_nd[[t]] <- matrix(1, nrow=N, ncol=n_covariates_max_nd)
      }
      
      X_d <- list()
      for(t in 1:n_tests) {
        X_d[[t]] <- matrix(1, nrow=N, ncol=n_covariates_max_nd)
      }
      
      
      {
        
        
        # if (prior_only == 0 ) { 
        stan_data = list(  N =   (N), # length(corr_prior_dist_LKJ_4), #  N, 
                           n_tests = n_tests,
                           y = (y_master_list_seed_123_datasets[[dataset_index]]),
                           n_class = 2,
                           n_pops = 1, #  n_pops,
                           pop = group,
                           n_covariates_max_nd = n_covariates_max_nd,
                           n_covariates_max_d = n_covariates_max_d,
                           n_covariates_max = n_covariates_max,
                           X_nd = X_nd,
                           X_d =  X_d,
                           n_covs_per_outcome = n_covs_per_outcome, 
                           corr_force_positive = corr_force_positive,
                           known_num = 0,
                           overflow_threshold = overflow_threshold,
                           underflow_threshold = underflow_threshold,
                           prior_only = prior_only, ######
                           prior_beta_mean =   prior_a_mean ,
                           prior_beta_sd  =   prior_a_sd ,
                           prior_LKJ = prior_lkj, 
                           prior_p_alpha = array(rep(5, n_pops)),
                           prior_p_beta =  array(rep(10, n_pops)),
                           Phi_type = Phi_type,
                           handle_numerical_issues = handle_numerical_issues,
                           fully_vectorised = fully_vectorised,
                           corr_prior_beta = corr_prior_beta,
                           corr_prior_norm = corr_prior_norm,
                           k_choose_2 = k_choose_2,
                           km1_choose_2 = km1_choose_2,
                           GHK_comp_method = GHK_comp_method)
        
        # stan_data_list[[i]] <- stan_data
        
      }
      
      
      # model files  
        file <- (file.path(pkg_dir, "inst/stan_models/LC_MVP_bin_PartialLog_v5.stan"))
        if (prior_only == 1 )  file <-  (file.path(pkg_dir, "inst/stan_models/PO_LC_MVP_bin.stan"))   
      
      
     #  ### modify cmdstanr path if necessary to include custom C++ header files
     #  cmdstanr::cmdstan_path()
     # 
     #  cmdstan_make_local(dir = cmdstan_path(), cpp_options = list("CXXFLAGS= -O3  -march=native  -mtune=native  -D_REENTRANT  -mfma  -mavx512f -mavx512vl -mavx512dq",
     #                                                              "CPPFLAGS= -O3  -march=native  -mtune=native  -D_REENTRANT  -mfma  -mavx512f -mavx512vl -mavx512dq"
     #                                                              ), append = FALSE)
     # 
     #  ### Then, if modified cmdstanr w/ custom C++ header files, rebuild the library using  cmdstanr::rebuild_cmdstan()
     # # cmdstanr::rebuild_cmdstan()
     #  
     #  
      mod <- cmdstan_model(file) # ,  include_paths = c("/home/enzocerullo/.cmdstan/cmdstan-2.35.0/stan/lib/stan_math/stan/math/rev/", 
                                                       # "/home/enzocerullo/.cmdstan/cmdstan-2.35.0/stan/lib/stan_math/"))
 
      # #  source("load_R_packages.R")
      # require(cmdstanr)
      # cmdstanr_example(
      #   example = "logistic",
      #   # example = c("logistic", "schools", "schools_ncp"),,
      #   force_recompile = getOption("cmdstanr_force_recompile", default = FALSE)
      # )
      
      # -------------------------------------------------------------------------
      
      u_raw <- array(0.01, dim = c(N, n_tests))
      
      km1_choose_2 = 0.5 * (n_tests - 2) * (n_tests - 1)
      known_num = 0
      
      beta_vec_init <- rep(0.01, n_class * n_tests)
      beta_vec_init[1:n_tests] <- - 1   # tests 1-T, class 1 (D-)
      beta_vec_init[(n_tests + 1):(2*n_tests)] <- + 1   # tests 1-T, class 1 (D+)
      
      off_raw <- list()
      col_one_raw <- list()
      
      for (i in 1:2) {
        off_raw[[i]] <-  (c(rep(0.01, km1_choose_2 - known_num)))
        col_one_raw[[i]] <-  (c(rep(0.01, n_tests - 1)))
      }
      
      init = list(
        u_raw = (u_raw),
        p_raw =  (-0.6931472), # equiv to 0.20 on p
        beta_vec = beta_vec_init,
        off_raw = off_raw,
        col_one_raw =  col_one_raw
        #L_Omega_raw = array(0.01, dim = c(n_class, choose(n_tests, 2)))
      )
      
      Stan_init_list <- init
      
      
      stan_data$prior_beta_mean
      stan_data$prior_beta_sd
      stan_data$prior_LKJ
      stan_data$Phi_type
 
      
      
    }
    
    






    
  

    
  # make lists of lists for inits 
  n_chains_burnin <- 8 
  init_lists_per_chain <- rep(list(Stan_init_list), n_chains_burnin) 
 
  
  ## set Stan model file path for your Stan model (replace with your path)
   Stan_model_file_path <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/stan_models/LC_MVP_bin_PartialLog_v5.stan" ;    Stan_cpp_user_header <- NULL
   #  Stan_model_file_path <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/stan_models/LC_MVP_bin_PartialLog_w_cpp.stan" ;    Stan_cpp_user_header <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/mvp_exp_approx.hpp"
   

   
   
    ## initialise model 
    init_model_and_vals_object <- initialise_model(  Model_type = "Stan",
                                                     sample_nuisance = TRUE,
                                                     init_lists_per_chain = init_lists_per_chain,
                                                    ##  model_args_list = NULL, ## this arg is only for MANUAL models
                                                     Stan_data_list = stan_data,
                                                     Stan_model_file_path = Stan_model_file_path,
                                                     n_chains_burnin = n_chains_burnin,
                                                    ## Stan_cpp_user_header  = Stan_cpp_user_header, ## optional C++ user-header file (if want to include C++ fns in Stan model)
                                                     n_params_main = n_params_main,
                                                     n_nuisance = n_nuisance)

    
     # parallel::mcparallel
    
    
    ## ----------- Set basic sampler settings
    {
      seed <- 123
      n_chains_sampling <- 64 
      n_superchains <- 4 # Each superchain is a "group" or "nest" of chains. If using ~8 chains or less, set this to 1. 
      n_iter = 1000
      n_burnin <- 500
      adapt_delta <- 0.80
      LR_main <- 0.05
      LR_us <- 0.05
    }
    
    
    
    Rcpp::sourceCpp("~/Documents/Work/PhD_work/R_packages/BayesMVP/src/main_v9.cpp") # , verbose = TRUE)
    
    
    ## sample / run model
    sample_obj <-   ( sample_model(Model_type = "Stan",
                                   sample_nuisance = TRUE,
                               init_model_and_vals_object = init_model_and_vals_object,
                               n_iter = n_iter,
                               n_burnin = n_burnin,
                               seed = seed,
                               y = y,
                               n_chains_burnin = n_chains_burnin,
                               n_chains_sampling = n_chains_sampling,
                               n_superchains = n_superchains,
                               diffusion_HMC = TRUE,
                            #   metric_shape_main = "diag",
                               metric_shape_main = "dense",
                              metric_type_main = "Hessian",
                             #   metric_type_main = "Emprical",
                             ##  n_nuisance_to_track = 5,
                               n_params_main = n_params_main,
                               n_nuisance = n_nuisance) )
    
    ### parallel::mccollect(sample_obj)
    
    
    ##### results summary 
    model_summary_outs <-    create_stan_summary(   model_results = sample_obj, 
                                                    init_model_and_vals_object = init_model_and_vals_object,
                                                    n_nuisance = n_nuisance, 
                                                    compute_main_params = TRUE, 
                                                    compute_generated_quantities = TRUE, 
                                                    compute_transformed_parameters = FALSE, 
                                                    save_log_lik_trace = FALSE, 
                                                    save_nuisance_trace = FALSE)
    
    
 
  
  
  
  
  
  
  
  
  
 
    
    
    
    
    
    
    
    
    
  
  
  