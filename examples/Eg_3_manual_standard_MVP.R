

########## -------- EXAMPLE 3 -------------------------------------------------------------------------------------------------------------------------------- 
######### Running the BUILT-IN (i.e. manual gradients) MVP model


{
  
  # Set working direcory ---------------
  try({  setwd("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/examples")   }, silent = TRUE)
  try({  setwd("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/examples")    }, silent = TRUE)
  #  options(repos = c(CRAN = "http://cran.rstudio.com"))
  
  # options ------------------------------------------------------------------- 
  #  totalCores = 8
  rstan::rstan_options(auto_write = TRUE)
  options(scipen = 999999)
  options(max.print = 1000000000)
  #  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
  
}


source("load_R_packages.R")

pkg_dir <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP" # ## TEMP - remove for final version of file 

 
Model_type <- "MVP"




# ------------------------------  Set basic specs / options
{
  options(scipen = 99999)
  options(max.print = 1000000)
  options(mc.cores = parallel::detectCores())
  numerical_diff_e = 0.001
  
  Model_type <- "MVP"
  set.seed(1, kind = "L'Ecuyer-CMRG")
  
  N <-  5000
  
  n_outcomes <- n_tests <- 4
  
  n_class <- 1 # NON-LCM model so only 1 class. 
  
  n_chains = round(parallel::detectCores()/2)
  n_iter = 500
}

 


# ------------------------------- SIMULATE data (assumed n_outcomes = 4 and non-LCM - modify for other values)
{
  
# n_covariates_per_outcome_vec <-  vector("list", length = n_class)
# n_covariates_per_outcome_vec[[1]] <- rep(NA, n_outcomes)
n_covariates_per_outcome_vec <- array(NA, dim = c(n_class, n_outcomes))
# first 3 outcomes have SINGLE BINARY COVARIATE and a SINGLE CONTINUOUS COVARIATE  [put cts covs FIRST and then binary covs SECOND and then cat covs THIRD]
n_covariates_per_outcome_vec[1, 1:3] <-  2
X_for_outcomes_w_single_bin_and_single_cts_cov <-  array(dim  = c(3, N, n_covariates_per_outcome_vec[1, 1]))

for (t in 1:3) {  ######  TEMP (change to cat w/ 3 categories!!)
  set.seed(t, kind = "L'Ecuyer-CMRG")
  X_for_outcomes_w_single_bin_and_single_cts_cov[t, 1:N, 1] <-  rnorm(n = N, mean = 1, sd = 1)  # cts covariates
  X_for_outcomes_w_single_bin_and_single_cts_cov[t, 1:N, 2] <-  rbinom(n = N, size = 1, prob = 0.25) # binary covariates
}
set.seed(1, kind = "L'Ecuyer-CMRG") # reset seed to 1

true_betas_for_outcome_1 <- c(-2.50, +0.25)
true_betas_for_outcome_2 <- true_betas_for_outcome_3  <- true_betas_for_outcome_1

# 3rd outcome has 1 CATEGORICAL COVARIATE w/ 3 categories (remember - MVP  has NO INTERCEPT if there are covariates)
# n_covariates_per_outcome_vec[[1]][4] <-    1
# X_for_outcome_w_single_cat_cov <- rbinom(n = N, size = 1, prob = 0.75)
# true_betas_for_outcome_4_cat_0_vs_cat_1 <- -0.50
# true_betas_for_outcome_4_cat_0_vs_cat_2 <- +0.50
 


# 4th outcome has 8 CONTINUOUS COVARIATES
n_covariates_per_outcome_vec[1, 4] <-    8
X_for_outcome_w_8_continuous_covs <- array(dim = c(N, n_covariates_per_outcome_vec[1, 4]))


for (k in 1:n_covariates_per_outcome_vec[1, 4]) {
  set.seed(k, kind = "L'Ecuyer-CMRG")
   kk = k - n_covariates_per_outcome_vec[1, 4]/2
   X_for_outcome_w_8_continuous_covs[1:N, k] <- rnorm(n = N, mean =  0.25*kk, sd = 1.0)
}
set.seed(1, kind = "L'Ecuyer-CMRG") # reset seed to 1

true_betas_for_outcome_4 <- c(-0.5, 1.5, 0.5, -0.25, -1.0, 0.5, 0, 0)

true_betas_all_outcomes_vec <- c(true_betas_for_outcome_1,  true_betas_for_outcome_2, true_betas_for_outcome_3, true_betas_for_outcome_4) # , true_betas_for_outcome_5)

# now make X (FULL covariate data array)
n_covariates_max <- max(n_covariates_per_outcome_vec)
X <- list(vector("list", n_tests))

str(X_for_outcomes_w_single_bin_and_single_cts_cov[1,,])

X[[1]][[1]] <- X_for_outcomes_w_single_bin_and_single_cts_cov[1,,]
X[[1]][[2]] <- X_for_outcomes_w_single_bin_and_single_cts_cov[2,,]
X[[1]][[3]] <- X_for_outcomes_w_single_bin_and_single_cts_cov[3,,]
X[[1]][[4]] <- X_for_outcome_w_8_continuous_covs

# check X correct format (should be a list of lists, and then each inner element is a matrix with N columns)
str(X)

# X  <- array(999999, dim = c(n_tests, n_covariates_max, N))
# X[1:3, 1:n_covariates_per_outcome_vec[1, 1], 1:N] <- X_for_outcomes_w_single_bin_and_single_cts_cov
# X[4, 1:n_covariates_per_outcome_vec[1, 4], 1:N] <- t(X_for_outcome_w_8_continuous_covs)


true_beta_vals <- array(999999, dim = c(n_tests, n_covariates_max))

true_beta_vals[1, 1:n_covariates_per_outcome_vec[1, 1]] <- true_betas_for_outcome_1
true_beta_vals[2, 1:n_covariates_per_outcome_vec[1, 2]] <- true_betas_for_outcome_2
true_beta_vals[3, 1:n_covariates_per_outcome_vec[1, 3]] <- true_betas_for_outcome_3
true_beta_vals[4, 1:n_covariates_per_outcome_vec[1, 4]] <- true_betas_for_outcome_4
## true_beta_vals[5, 1:n_covariates_per_outcome_vec[5]] <- true_betas_for_outcome_5


true_Xbeta_vals <- array(dim = c(N, n_tests))
for (t in 1:n_tests) {
  true_Xbeta_vals[1:N, t] <-  (X[[1]][[t]][1:N, 1:n_covariates_per_outcome_vec[1, t]]) %*% (true_beta_vals[t, 1:n_covariates_per_outcome_vec[1, t]])
}
 
 
}

str(X)




# ------------------------------  generate data
{
  
  y_master_list_seed_123_datasets <- list()
  
  N_datasets <- 123
  
  y_list <- list()
  
  for (ii in 1:N_datasets) {
    
    df_sim_seed <- ii
    
    set.seed(df_sim_seed, kind = "L'Ecuyer-CMRG")
    
    Sigma_highly_varied <- matrix(c(1,  0,     0,        0,       
                                    0,  1,     0.50,     0.25,    
                                    0,  0.50,  1,        0.40,   
                                    0,  0.25,  0.40,     1    ),   
                                  n_tests, n_tests)
 
  
      
      Sigma  <- 1.5  * Sigma_highly_varied # / 2
      diag(Sigma) <- rep(1, n_tests)
    
    
    latent_results <- LaplacesDemon::rmvn(n = N, mu = (true_Xbeta_vals), Sigma = Sigma)
    results <- ifelse(latent_results > 0, 1, 0)
    y <- results
    
    print(Sigma)

    y_list[[ii]] <- y
    
    
  }
  
  if (N == 500)   y_master_list_seed_123_datasets[[1]] <- y_list[[123]]
  if (N == 1000)  y_master_list_seed_123_datasets[[2]] <- y_list[[123]]
  if (N == 2500)  y_master_list_seed_123_datasets[[3]] <- y_list[[123]]
  if (N == 5000)  y_master_list_seed_123_datasets[[4]] <- y_list[[123]]
  if (N == 12500) y_master_list_seed_123_datasets[[5]] <- y_list[[123]]
  if (N == 25000) y_master_list_seed_123_datasets[[6]] <- y_list[[123]]
  
}




# -------------------------------  Set PRIORS
# any priors not specified will be set to defaults. 
{
  beta_prior_mean_vec <- beta_prior_sd_vec <-  vector("list", length = n_class)
  beta_prior_mean_vec[[1]] <- array(0, dim = c(n_covariates_max, n_tests))
  beta_prior_sd_vec[[1]] <- array(5, dim = c(n_covariates_max, n_tests))
  
  lkj_cholesky_eta = 2  ;  corr_force_positive = 0
}





 
 

# ------------------------------  Set INITIAL VALUES ("inits") -----------------------------------------------------------------------
# any inits not specified will be set to defaults. 
{
  
  init_vals_beta <-  vector("list", length = n_class)
  init_vals_beta[[1]] <- array(0, dim = c(n_tests, n_covariates_max))
  init_vals_beta[[1]] <-  ifelse(true_beta_vals == 999999.0, 0, true_beta_vals)
  init_vals_beta[[1]] <- t(init_vals_beta[[1]])
  
  # make beta vector (make sure to loop through class first, then outcome/test, then n_covs_per_outcome)
  beta_vec_init <- c()
  counter <- 1
  for (c in 1 : n_class) {
      for (t in 1:n_tests) {
        for (k in 1:n_covariates_per_outcome_vec[c, t]) {
          beta_vec_init[counter] <- init_vals_beta[[c]][k, t]
          counter <- counter + 1
        }
      }
  }
  
  ### init values for correlations 
  k_choose_2 <- choose(n_tests, 2)
  km1_choose_2 = 0.5 * (n_tests - 2) * (n_tests - 1)
  known_num = 0
  
  for (i in 1:n_class) {
    off_raw<-  (c(rep(0.01, km1_choose_2 - known_num)))
    col_one_raw <-  (c(rep(0.01, n_tests - 1)))
  }
  
  ## init values for nuisance parameters 
  u_raw <- array(0.01, dim = c(N, n_tests))
  
}


## init values list 
init = list(
  u_raw = (u_raw),
  beta_vec = beta_vec_init,
  off_raw = off_raw,
  col_one_raw =  col_one_raw
)

## number of parameters
n_params_main <-   sum(n_covariates_per_outcome_vec) + choose(n_tests, 2)
n_nuisance <- N * n_tests

 





# ------------------------------ initialise + run model  ------------------------------------------------------------------------------------

Rcpp::sourceCpp("~/Documents/Work/PhD_work/R_packages/BayesMVP/src/main_v9.cpp")# , verbose = TRUE)


### model args / inputs 
model_args_list  <- list(n_covariates_per_outcome_vec = (n_covariates_per_outcome_vec),
                         lkj_cholesky_eta = lkj_cholesky_eta,
                         X = X,
                         prior_coeffs_mean_vec = beta_prior_mean_vec,
                         prior_coeffs_sd_vec =    beta_prior_sd_vec)


 
### initialise the model using "initialise_model" function
n_chains_burnin <- 8 # based on (informal) testing, more than 8 burnin chains seems unnecessary and probably not worth the extra overhead. 
init_lists_per_chain <- rep(list(init), n_chains_burnin) 

init_model_and_vals_object <- initialise_model(  init_lists_per_chain = init_lists_per_chain, 
                                                 Model_type = Model_type,
                                                 model_args_list = model_args_list, # this arg is only for MANUAL models
                                                 y = y,
                                                 n_chains_burnin = n_chains_burnin,
                                                 n_params_main = n_params_main,
                                                 n_nuisance = n_nuisance)

# mat <- init_model_and_vals_object$init_model_object$Model_args_as_Rcpp_List$Model_args_mats_int[[1]]
# is.matrix(mat)
   


## ----------- Set basic sampler settings
{
  ### seed <- 123
  n_chains_sampling <- 64 
  n_superchains <- 4 ## round(n_chains_sampling / n_chains_burnin) # Each superchain is a "group" or "nest" of chains. If using ~8 chains or less, set this to 1. 
  n_iter = 1000
  n_burnin <- 500
  adapt_delta <- 0.80
  LR_main <- 0.05
  LR_us <- 0.05
  n_nuisance_to_track <- 5
}




Rcpp::sourceCpp("~/Documents/Work/PhD_work/R_packages/BayesMVP/src/main_v9.cpp") # , verbose = TRUE)


# 
# init_model_and_vals_object$init_model_object$Model_args_as_Rcpp_List$Model_args_ints[4, 1] <- 1 # num_chunks
# 
# init_model_and_vals_object$init_model_object$Model_args_as_Rcpp_List$Model_args_mats_int[[1]]
# 
# str(init_model_and_vals_object$init_model_object$Model_args_as_Rcpp_List$Model_args_2_later_vecs_of_mats_double )
# 
# init_model_and_vals_object$y <- y

# str(X)
# 
# init_model_and_vals_object$init_model_object$Model_args_as_Rcpp_List$Model_args_strings

###  parallel::mcparallel 

sample_obj <-        (sample_model(   init_model_and_vals_object = init_model_and_vals_object,
                                      Model_type = Model_type,
                                      n_iter = n_iter,
                                      n_burnin = n_burnin,
                                      seed = 2,
                                      y = y,
                                      n_chains_burnin = n_chains_burnin,
                                      n_chains_sampling = n_chains_sampling,
                                      n_superchains = n_superchains,
                                      diffusion_HMC = TRUE,
                                      adapt_delta = adapt_delta,
                                      LR_main = LR_main,
                                      LR_us = LR_us,
                                      # metric_shape_main = "diag",
                                      metric_shape_main = "dense",
                                      metric_type_main = "Hessian",
                                      #  metric_type_main = "Emprical",
                                      n_params_main = n_params_main,
                                      force_autodiff = FALSE,
                                      force_PartialLog = FALSE,
                                      multi_attempts = FALSE,
                                      n_nuisance = n_nuisance, 
                                      n_nuisance_to_track = n_nuisance_to_track))


str(sample_obj$result[[1]])

trace_main <- sample_obj$result[[1]]

trace_main_2 <- array(dim = c(n_params_main, n_iter, n_chains_sampling))
for (kk in 1:n_chains) {
  trace_main_2[,,kk] <- trace_main[[kk]]
}

str(trace_main_2)

trace_main_2_between_chains <- apply(trace_main_2, FUN = mean, c(1,2), na.rm = TRUE)
trace_main_2_posterior_summary <- apply(trace_main_2_between_chains, FUN = mean, 1, na.rm = TRUE)

str(trace_main_2_posterior_summary)

n_corrs <- choose(n_tests, 2)

signif(trace_main_2_posterior_summary, 3)[(n_corrs + 1):n_params_main]
true_betas_all_outcomes_vec


signif(trace_main_2_posterior_summary, 3)[(n_corrs + 1):n_params_main] - true_betas_all_outcomes_vec


trace_main_2_posterior_summary[(n_corrs + 1):(n_corrs + sum(n_covariates_per_outcome_vec))]

 
 #####  ---------------- model summary  
  model_summary_outs <-    create_stan_summary(   model_results = sample_obj, 
                                                  init_model_and_vals_object = init_model_and_vals_object,
                                                  n_nuisance = n_nuisance, 
                                                  compute_main_params = TRUE, 
                                                  compute_generated_quantities = FALSE, 
                                                  compute_transformed_parameters = TRUE, 
                                                  save_log_lik_trace = FALSE, 
                                                  save_nuisance_trace = FALSE)
  
  
  print(model_summary_outs$ESS_per_sec_samp / (1000 * model_summary_outs$Min_ess_per_grad_samp_weighted) )
  
 

























 
# 
# 
# 
# 
# n_us <- N*n_tests
# n_params <- n_us + n_params_main
# index_us <- 1:n_us
# index_main <- (1 + n_us):n_params
# 
# if (Model_type == "latent_trait") {
#   n_corrs <- n_tests * 2
# } else { 
#   n_corrs <- 2 * choose(n_tests, 2)
# }
# 
# 
# theta_vec <- rep(0.01, n_params)
# 
# 
# init_model_object <- init_model_and_vals_object$init_model_object
# init_vals_object <- init_model_and_vals_object$init_vals_object
# 
# Model_args_as_Rcpp_List <- init_vals_object$Model_args_as_Rcpp_List
# 
# 
# 
# 
# 
# 
# num_chunks <- 1
# Model_args_as_Rcpp_List$Model_args_ints[4, 1] <- num_chunks
# Model_args_as_Rcpp_List$Model_args_ints
# Model_args_as_Rcpp_List$Model_args_doubles
# Model_args_as_Rcpp_List$Model_args_strings
# Model_args_as_Rcpp_List$Model_args_bools
# 
# Model_args_as_Rcpp_List$Model_args_doubles[3, 1] <- +0.5
# Model_args_as_Rcpp_List$Model_args_doubles[4, 1] <- -0.5
# 
# Model_args_as_Rcpp_List$Model_args_strings[c(1, 4,5,7,8,9,10,11), 1]  <- "Stan"
# 
# Model_args_as_Rcpp_List$Model_args_strings[2, 1] <- "Phi"
# Model_args_as_Rcpp_List$Model_args_strings[3, 1] <- "inv_Phi"
# 
# Model_args_as_Rcpp_List$Model_args_strings[13,1] <- "Phi"
# 
# # Model_args_as_Rcpp_List$Model_args_col_vecs_double[[1]] <- matrix(Model_args_as_Rcpp_List$Model_args_col_vecs_double[[1]])
# Model_args_as_Rcpp_List$Model_args_vecs_of_mats_double
# 
# Model_args_as_Rcpp_List$Model_args_vecs_of_mats_int
# Model_args_as_Rcpp_List$Model_args_vecs_of_col_vecs_int
# 
# # theta_vec <- rnorm(n = n_params, mean = 0, sd = 0.10)
# 
# # theta_vec[2501:2520] <-   0.01 ## corrs
# # theta_vec[2521:2530] <-   0.0001 ## coeffs
# 
# 
# ### Rcpp::sourceCpp("~/Documents/Work/PhD_work/R_packages/BayesMVP/src/main_v9.cpp")
# 
# theta_vec[(n_us + 1):n_params] <- rnorm(n = n_params_main, mean = 0, sd = 0.25)
# 
# Model_args_as_Rcpp_List$Model_args_doubles[3, 1] <- 5
# Model_args_as_Rcpp_List$Model_args_doubles[4, 1] <- -5
# 
# 
# 
# 
# Rcpp::sourceCpp("~/Documents/Work/PhD_work/R_packages/BayesMVP/src/main_v9.cpp") # , verbose = TRUE)
# 
# 
# tic()
# # for (i in 1:1000)
# lp_grad_outs <-   parallel::mcparallel (fn_Rcpp_wrapper_fn_lp_grad( Model_type = Model_type,
#                                                                     force_autodiff = TRUE,
#                                                                     force_PartialLog = TRUE,
#                                                                     theta_main_vec = theta_vec[index_main],
#                                                                     theta_us_vec = theta_vec[index_us],
#                                                                     y = y,
#                                                                     grad_option = "all",
#                                                                     Model_args_as_Rcpp_List = Model_args_as_Rcpp_List))
# toc()
# 
# 
# 
# outs_AD <- parallel::mccollect(lp_grad_outs)
# 
# 
# 
# 
# # parallel::mcparallel
# Model_args_as_Rcpp_List$Model_args_doubles[3, 1] <- +5
# Model_args_as_Rcpp_List$Model_args_doubles[4, 1] <- -5
# lp_grad_outs <-   parallel::mcparallel(fn_Rcpp_wrapper_fn_lp_grad( Model_type = Model_type,
#                                                                    force_autodiff = FALSE,
#                                                                    force_PartialLog = FALSE,
#                                                                    theta_main_vec = theta_vec[index_main],
#                                                                    theta_us_vec = theta_vec[index_us],
#                                                                    y = y,
#                                                                    grad_option = "all",
#                                                                    Model_args_as_Rcpp_List = Model_args_as_Rcpp_List))
# 
# 
# 
# 
# out <- parallel::mccollect(lp_grad_outs)
# #out
# ###  head(out[[1]][-1], 2500)
# 
# out[[1]][1]
# outs_AD[[1]][1]
# 
# 
# out[[1]][1] - outs_AD[[1]][1] # log_prob diff
# 
# abs(out[[1]][1] ) > abs( outs_AD[[1]][1] )
# 
# sum(  head(out[[1]][-1], 25000)  -  head(outs_AD[[1]][-1], 25000)   )  # u'#s grad diff (sum)
# 
# head(out[[1]][-1], 5000)  -  head(outs_AD[[1]][-1], 5000)
# 
# grad_us_manual <- head(out[[1]][-1], n_us)
# grad_us_AD <- head(outs_AD[[1]][-1], n_us)
# 
# vec_1 <-   head(grad_us_manual, 12500)
# vec_2 <-   head(grad_us_AD, 12500)
# 
# vec_1 - vec_2
# 
# tail(out[[1]][-1], 11)[1:10]  -  tail(outs_AD[[1]][-1], 11)[1:10] # coeffs grad diffs
# tail(out[[1]][-1], n_params_main)[1:n_corrs]  -  tail(outs_AD[[1]][-1], n_params_main)[1:n_corrs] # corrs grad diffs
# tail(out[[1]][-1], 1)   -  tail(outs_AD[[1]][-1], 1) # prevelance grad diff
# 
# tail(out[[1]][-1], 11)[1:10]
# tail(outs_AD[[1]][-1], 11)[1:10]
# 
# tail(out[[1]][-1], n_params_main)
# tail(outs_AD[[1]][-1], n_params_main)
# 
# tail(out[[1]][-1], n_params_main) - 
# tail(outs_AD[[1]][-1], n_params_main)
# 
# n_params_main
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
# 
# 
# 
