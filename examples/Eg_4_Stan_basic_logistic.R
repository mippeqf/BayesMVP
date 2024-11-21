
 
{
  
  rstan::rstan_options(auto_write = TRUE)
  options(scipen = 99999)
  options(max.print = 1000000000)
  #  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
  
}




{
  
  # Set working direcory ---------------
  try({  setwd("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/examples")   }, silent = TRUE)
  try({  setwd("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/examples")    }, silent = TRUE)
  #  options(repos = c(CRAN = "http://cran.rstudio.com"))
  
  # options ------------------------------------------------------------------- 
  #  totalCores = 8
  rstan::rstan_options(auto_write = TRUE)
  options(scipen = 99999)
  options(max.print = 1000000000)
  #  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
  
}

          

# Run LC-MVP model (manual-gradients using SNAPER-diffusion-space HMC)   --------------------------------------------------------------------------------------------------------------------------------------------------
      
pkg_dir <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP"
   
source("load_R_packages.R")
 
## source(file.path(pkg_dir, "examples/BayesMVP_LC_MVP_prep.R"))
 
 

# 
# - | ----------  prepare Stan data and inits --------------------------------------------------------------------


  ## ### source(file.path(pkg_dir, "examples/Stan_LC_MVP_prepare_data.R"))


      Model_type <- "Stan" # specify Model_type as "Stan" if using a Stan model

      ### input N to use 
      N <- 10000

      ### generate data
      n_covs <- 10
      X <- array(NA, dim = c(N, n_covs))

      set.seed(1)
      for (i in 1:n_covs) {
         #  X[1:N, i]  <-  rnorm(n = N, mean = 1, sd = 1)  # cts covariates
           X[1:N, i]  <-  rlogis(n = N)  # cts covariates
      }
      
      true_intercept <- 0.0
      true_beta_vals <- c(-1.0, -0.50, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5) # , rep(1, 20))
      length(true_beta_vals)
      true_Xbeta_vals <- X %*%  t(t(true_beta_vals))
      
       # latent_results <- rnorm(n = N, mean =  true_intercept + true_Xbeta_vals, sd = 1) # for probit regression
       latent_results <- rlogis(n = N, location =  true_intercept + true_Xbeta_vals) # for logistic regression
      
      ## simulated dataset
      y <- ifelse(latent_results > 0, 1, 0)
  
      ### Stan model data list 
      stan_data = list(  N =  N, 
                         y = c(y),
                         n_covs = n_covs, 
                         X = X)
      
      ### Stan model file (put path to your Stan model)
      file <- (file.path(pkg_dir, "inst/stan_models/basic_logistic.stan"))
      mod <- cmdstan_model(file)
      
      
      # # OR using the version w/ fast math C++ functions:
      # file <- (file.path(pkg_dir, "inst/stan_models/basic_logistic_w_fast_fns.stan"))
      # ## and then input the path to the corresponding C++ files:
      # path_to_cpp_user_header <- file.path(pkg_dir, "src/approx_cpp_fns_for_Stan.hpp")
      # 
      # 
      
      mod <- cmdstan_model(file, 
                          # force_recompile = TRUE,
                          # user_header = path_to_cpp_user_header,
                           cpp_options = list(
                                               "CXXFLAGS =    -O3  -march=native  -mtune=native -fPIC -D_REENTRANT  -mfma  -mavx512f -mavx512vl -mavx512dq" ,
                                               "CPPFLAGS =    -O3  -march=native  -mtune=native -fPIC -D_REENTRANT  -mfma  -mavx512f -mavx512vl -mavx512dq"
                                               ))



 
 
      #### Stan initial values list 
      alpha <- 0.001
     ##  beta <- rep(0.001, n_covs)
      beta <- true_beta_vals
      
      init = list(
        alpha = alpha, 
        beta = beta,
        u_raw = rep(0.001, N)
      )
      
      Stan_init_list <- init
      
      # make lists of lists for inits 
      n_chains <- 64
      init_lists_per_chain <- rep(list(Stan_init_list), n_chains) 
      
      
      ### run model using Stan directly (cmdstanr) first 
      {
            tictoc::tic()
        
            Stan_model_out <- mod$sample(data = stan_data, 
                                         seed = 3,
                                         refresh = 50,
                                         init = init_lists_per_chain,
                                         chains = n_chains,
                                         parallel_chains = n_chains,
                                         max_treedepth = 10,
                                         iter_warmup = 500,
                                         iter_sampling = 1000)
          
            
            try({
              {
                print(tictoc::toc(log = TRUE))
                log.txt <- tictoc::tic.log(format = TRUE)
                tictoc::tic.clearlog()
                total_time <- unlist(log.txt)
              }
            })
            
            try({
              total_time <- as.numeric( substr(start = 0, stop = 100,  strsplit(  strsplit(total_time, "[:]")[[1]], "[s]")  [[1]][1] ) )
            })
        
      }
      
      
      (Stan_model_out$time())
      
     ###  Stan_model_out$unconstrain_variables()
      
      
      ### results summary
      Stan_model_results <- Stan_model_out$summary()
      print(Stan_model_results, n = 100)
      
      
      ### some quick efficiency stats 
      {
        Stan_sampler_diagnostics <- as.array(Stan_model_out$sampler_diagnostics())
        Stan_treedepth_array <- Stan_sampler_diagnostics[,,1]
        Stan_avg_treedepth_during_sampling <- sum(Stan_treedepth_array) / length(Stan_treedepth_array)
        Stan_avg_L_during_sampling <- 2^Stan_avg_treedepth_during_sampling
        Stan_n_grad_evals_sampling <- n_chains * n_iter * Stan_avg_L_during_sampling
        
        Stan_times_per_chain <- Stan_model_out$time()
        time_sampling <- max(Stan_times_per_chain$chains$sampling)
        
        Stan_min_ESS <- min(round(Stan_model_results$ess_bulk[(0 + 1):(0 + n_covs - 1)]))
   
        Stan_min_ESS_per_sec <-  Stan_min_ESS / total_time ; Stan_min_ESS_per_sec
        Stan_min_ESS_per_sec_sampling <-  Stan_min_ESS / time_sampling ; Stan_min_ESS_per_sec_sampling
        Stan_min_ESS_per_grad_sampling <-   ( Stan_min_ESS / Stan_n_grad_evals_sampling )
        Stan_min_ESS_per_grad_sampling_x_1000 <- 1000 * Stan_min_ESS_per_grad_sampling
        Stan_grad_evals_per_sec <- Stan_min_ESS_per_sec_sampling /  Stan_min_ESS_per_grad_sampling 
        Stan_grad_evals_per_sec_div_1000 <- Stan_grad_evals_per_sec / 1000
        
        
        print(Stan_min_ESS_per_sec)
        print(Stan_min_ESS_per_sec_sampling)
        print(Stan_min_ESS_per_grad_sampling_x_1000)
        print(Stan_grad_evals_per_sec_div_1000)
        
      }
 
      
      ##### using Stan's fns:
      ### N = 10,000: 
      # run 1: ESS/sec = 767, samp = 1710, ESS/grad (samp) = 4.70  grad/sec =  363
      # run 2: ESS/sec = 699, samp = 1660, ESS/grad (samp) = 4.39  grad/sec =  378
      # run 3: ESS/sec = 795, samp = 1720, ESS/grad (samp) = 4.87  grad/sec =  353
      # AVG:
      ### N = 20,000
      # run 1: ESS/sec = 229, samp =  367, ESS/grad (samp) = 4.92  grad/sec =  74.6
      # run 2: ESS/sec = 212, samp =  327, ESS/grad (samp) = 4.73  grad/sec =  69.1
      # run 3: ESS/sec = 212, samp =  329, ESS/grad (samp) = 4.68  grad/sec =  70.3
      
      
      ##### using Stan but w/ custom C++ fns:
      ### N = 10,000: 
      # run 1: ESS/sec = 823, samp = 2001, ESS/grad (samp) = 4.82  grad/sec = 415
      # run 2: ESS/sec = 785, samp = 1849, ESS/grad (samp) = 4.67  grad/sec = 396
      # run 3: ESS/sec = 770, samp = 1785, ESS/grad (samp) = 4.50  grad/sec = 396
      ### N = 20,000
      # run 1: 
      # run 2: ESS/sec = ..., samp =  ..., ESS/grad (samp) = ...  grad/sec =  ...
      # run 3: 
      
      
    

 
  
    ## set Stan model file path for your Stan model (replace with your path)
    Stan_model_file_path <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/stan_models/basic_logistic.stan"     

    # ## or using the version w/ fast math C++ functions: 
    # Stan_model_file_path <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/stan_models/basic_logistic_w_fast_fns.stan" 
    # ## and the path to the corresponding C++ user header containing the fast C++ math functions (downloaded seperately from R package)
    # Stan_cpp_user_header <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/stan_models/basic_logistic_w_fast_fns.stan" 
    # 

    
    sample_nuisance <- FALSE
    
    
    # make lists of lists for inits 
    n_chains_burnin <- 8
    init_lists_per_chain <- rep(list(Stan_init_list), n_chains_burnin) 
   
    ## initialise model 
    n_params_main <- n_covs + 1
    init_model_and_vals_object <- initialise_model( Model_type = "Stan",
                                                    N = N,
                                                    sample_nuisance = sample_nuisance, # make sure FALSE if model has no latent variables / nuisance parameters 
                                                    init_lists_per_chain = init_lists_per_chain,
                                                    Stan_data_list = stan_data,
                                                    Stan_model_file_path = Stan_model_file_path,
                                                    n_chains_burnin = n_chains_burnin,
                                                 ###     Stan_cpp_user_header  = Stan_cpp_user_header, ## optional C++ user-header file (if want to include C++ fns in Stan model)
                                                    n_params_main = n_params_main,
                                                    n_nuisance = n_nuisance)
    
    
    init_model_and_vals_object$model_so_file

    
     # parallel::mcparallel
    
    
    
    
    ## ----------- Set basic sampler settings
    {
          seed <- 123
          n_chains_sampling <- 64 
          n_superchains <- 4 # Each superchain is a "group" or "nest" of chains. If using ~8 chains or less, set this to 1. 
          n_iter = 10000
          n_burnin <- 500
          adapt_delta <- 0.80
          LR_main <- 0.075
          LR_us <- 0.075
    }
    
    
  #####  Rcpp::sourceCpp("~/Documents/Work/PhD_work/R_packages/BayesMVP/src/main_v9.cpp") # , verbose = TRUE)
    
    
    ## sample / run model
    # parallel::mcparallel
    
    sample_obj <-  ( sample_model(Model_type = "Stan",
                                  init_object = init_model_and_vals_object,
                              # init_model_and_vals_object = ,
                               sample_nuisance = sample_nuisance,
                               n_iter = n_iter,
                               n_burnin = n_burnin,
                               seed = seed,
                               y = t(y),
                               n_chains_burnin = n_chains_burnin,
                               n_chains_sampling = n_chains_sampling,
                               n_superchains = n_superchains,
                               diffusion_HMC = TRUE,
                               #  metric_shape_main = "diag",
                              metric_shape_main = "dense",
                                 metric_type_main = "Hessian",
                            #   metric_type_main = "Empirical",
                               n_params_main = n_params_main,
                               n_nuisance = n_nuisance) )
    
    
    sample_obj$time_total
    
    ###  parallel::mccollect(sample_obj)
    
    str(sample_obj)
    
    str(sample_obj$result[[1]])
    
    str(sample_obj$result[[1]])
    
    trace_main <- sample_obj$result[[1]]
    
    trace_main_2 <- array(dim = c(n_params_main, n_iter, n_chains_sampling))
    for (kk in 1:n_chains) {
      trace_main_2[,,kk] <- trace_main[[kk]]
    }
    
    str(trace_main_2)
    
    trace_main_2_between_chains <- apply(trace_main_2, FUN = mean, c(1,2), na.rm = TRUE)
    trace_main_2_posterior_summary <- apply(trace_main_2_between_chains, FUN = mean, 1, na.rm = TRUE)
    
    str(trace_main_2)
    
    trace_main_2_posterior_summary
    
    ess_vec <- c()
    for (i in 1:n_params_main) { 
      ess_vec[i] <- ess_bulk(trace_main_2[i,,])
    }
    
    ess_vec
    min_ess <- min(ess_vec)
    min_ess
    

    n_params_main
    
  
    sample_obj$time_total
    min_ess / sample_obj$time_sampling
    min_ess / sample_obj$time_total
    
    total_time
    Stan_min_ESS_per_sec_sampling
    Stan_min_ESS_per_sec
 
    
    
    
    
    str(trace_main_2)
    
    trace_main_2[1,,16]
    
    
    
    trace_main_2_posterior_summary
    
    str(trace_main_2_posterior_summary)
    
    n_corrs <- choose(n_tests, 2)
    
    signif(trace_main_2_posterior_summary, 3)[(n_corrs + 1):n_params_main]
    true_betas_all_outcomes_vec
    
    
    signif(trace_main_2_posterior_summary, 3)[(n_corrs + 1):n_params_main] - true_betas_all_outcomes_vec
    
    
    trace_main_2_posterior_summary[(n_corrs + 1):(n_corrs + sum(n_covariates_per_outcome_vec))]
    
    
    
    
    ##### results summary 
    model_summary_outs <-    create_stan_summary(   model_results = sample_obj, 
                                                   #  = 
                                                      init_object = init_model_and_vals_object,
                                                    n_nuisance = 1, 
                                                    compute_main_params = TRUE, 
                                                    compute_generated_quantities = FALSE, 
                                                    compute_transformed_parameters = FALSE, 
                                                    save_log_lik_trace = FALSE, 
                                                    save_nuisance_trace = FALSE)
    
    
 
  
  
  
  
  
  
  
  
  
 
    
    
    
    
    
    
    
    
    
  
  
  