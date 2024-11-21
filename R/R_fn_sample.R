








#' sample_model
#' @keywords internal
#' @export
sample_model  <-    function(     Model_type,
                                  init_object,
                                  parallel_method = "RcppParallel",
                                  y = NULL,
                                  N = NULL,
                                  sample_nuisance,
                                  diffusion_HMC = TRUE,
                                  partitioned_HMC  = TRUE,
                                  vect_type = NULL,
                                  Phi_type = NULL,
                                  inv_Phi_type = NULL,
                                  n_params_main,
                                  n_nuisance,
                                  n_chains_burnin,
                                  n_chains_sampling,
                                  n_superchains,
                                  seed,
                                  n_burnin = 500,
                                  n_iter = 1000,
                                  adapt_delta = 0.80,
                                  LR_main = NULL,
                                  LR_us = NULL,
                                  n_adapt  = NULL,
                                  clip_iter = NULL,
                                  gap = NULL,
                                  metric_type_main = "Hessian",
                                  metric_shape_main = "dense",
                                  metric_type_nuisance = "Euclidean",
                                  metric_shape_nuisance = "diag",
                                  shrinkage_factor = NULL,
                                  max_eps_main = 1.0, 
                                  max_eps_us = 2.5,
                                  tau_main_target = 0,
                                  tau_us_target = 0,
                                  main_L_manual = FALSE,
                                  L_main_if_manual = 0,
                                  us_L_manual = FALSE,
                                  L_us_if_manual = 0,
                                  max_L = 1024,
                                  tau_mult = 1.60,
                                  ratio_M_us = 0.25,
                                  ratio_M_main = 0.25,
                                  interval_width_main  = NULL,
                                  interval_width_nuisance  = NULL,
                                  force_autodiff = FALSE,  
                                  force_PartialLog = FALSE,
                                  multi_attempts = FALSE,
                                  n_nuisance_to_track = 5, 
                                  ...) { 
  
  
                if (sample_nuisance == FALSE) { 
                  n_nuisance <- 9 # dummy
                }
 
                
                # init_model_object <- init_object$init_model_object
                # init_vals_object <- init_object$init_vals_object
 
                Model_args_as_Rcpp_List <- init_object$Model_args_as_Rcpp_List
                
                if (Model_type != "Stan")  {
                    n_params_main <- Model_args_as_Rcpp_List$n_params_main # <- n_params_main
                    n_nuisance <- Model_args_as_Rcpp_List$n_nuisance #<- n_nuisance
                } else { 
                    Model_args_as_Rcpp_List$n_params_main <- n_params_main
                    Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
                }
                
 
                # Model_args_as_Rcpp_List$model_so_file <- init_object$Model_args_as_Rcpp_List$model_so_file
                # print(   Model_args_as_Rcpp_List$model_so_file)
                
                print(paste("n_nuisance = ", n_nuisance))
  
                
                 if (Model_type != "Stan") {

                                # Model_args_as_Rcpp_List$Model_args_mats_int[[1]] <- (Model_args_as_Rcpp_List$Model_args_mats_int[[1]])
                                #
                                # print((Model_args_as_Rcpp_List$Model_args_mats_int[[1]]))
                                # ##  print(paste("hello", Model_args_as_Rcpp_List$Model_args_mats_int[[1]]))
                                #
                                # Model_args_as_Rcpp_List$Model_args_ints[4, 1] <- num_chunks # num_chunks
                                #
                                # Model_args_as_Rcpp_List$Model_args_doubles[3, 1] <- +5
                                # Model_args_as_Rcpp_List$Model_args_doubles[4, 1] <- -5
                                #
                                # Model_args_bools <- matrix(rep(NA, 15))
                                # Model_args_bools[1:14, 1] <-  Model_args_as_Rcpp_List$Model_args_bools
                                # Model_args_as_Rcpp_List$Model_args_bools <- Model_args_bools
                                #
                                # Model_args_as_Rcpp_List$Model_args_bools[15, 1] <- FALSE # debug

                                 # nuisance transformation
                                 Model_args_as_Rcpp_List$Model_args_strings[13, 1] <- "Phi"
                   
                                  # try({
                                  # Model_args_as_Rcpp_List$Model_args_strings[2,1] <-    "Phi"
                                  # Model_args_as_Rcpp_List$Model_args_strings[3,1] <-    "inv_Phi"
                                 
                                 
                                  try({
                                    Model_args_as_Rcpp_List$Model_args_strings[c(1, 4,5,6,7,8,9,10,11),1] <-     vect_type
                                    Model_args_as_Rcpp_List$Model_args_strings[6,1] <-  vect_type
                                  }, silent = TRUE)

                              # Model_args_as_Rcpp_List$Model_args_strings[1,1] <-  "AVX512" # general  - fine (clang)
                              # Model_args_as_Rcpp_List$Model_args_strings[4,1] <-  "AVX512" # exp  - fine (clang)
                              # Model_args_as_Rcpp_List$Model_args_strings[5,1] <-  "AVX512" # log    - fine (clang)
                              # Model_args_as_Rcpp_List$Model_args_strings[6,1] <-  "AVX512" # lse
                              # Model_args_as_Rcpp_List$Model_args_strings[7,1] <-  "AVX512" # tanh     - fine (clang)
                              # Model_args_as_Rcpp_List$Model_args_strings[8,1] <-  "AVX512" # Phi  - not working w/ chunking   ?!
                              # Model_args_as_Rcpp_List$Model_args_strings[9,1] <-  "AVX512" # log_Phi - not working w/ chunking  ?!
                              # Model_args_as_Rcpp_List$Model_args_strings[10,1] <- "AVX512" # inv_Phi   - fine (clang)
                              # Model_args_as_Rcpp_List$Model_args_strings[11,1] <- "AVX512" # inv_Phi_approx_from_logit_prob - fine (clang)





                            #
                            #       # Model_args_as_Rcpp_List$Model_args_doubles[[3]] <- +0.0000001
                            #       # Model_args_as_Rcpp_List$Model_args_doubles[[4]] <- -0.0000001
                            #
                                  # #  set.seed(123)
                                  # n_params <-  n_nuisance + n_params_main
                                  # index_main <- (n_nuisance + 1):n_params
                                  # index_us <- 1:n_nuisance
                                  # theta_vec <- rnorm(n = n_params, mean = 0, sd = 0.01)
                                  #
                                  #


                 }
                
                     
                      
                      n_class <- 0 
                      
                      if (Model_type == "Stan") { 
                        n_class <- 0 
                      } else { 
                        n_class <-  Model_args_as_Rcpp_List$Model_args_ints[2]
                      }
                      
                      print(paste("n_class = ", n_class))
                      
                      
                      if (n_class == 1) {
                            
                               lkj_cholesky_eta <- Model_args_as_Rcpp_List$Model_args_col_vecs_double[[1]]
                            if (is.matrix(lkj_cholesky_eta) == FALSE) {
                               lkj_cholesky_eta <- matrix(lkj_cholesky_eta)
                            }
                               
                            Model_args_as_Rcpp_List$Model_args_col_vecs_double[[1]] <- lkj_cholesky_eta
                           # Model_args_as_Rcpp_List$Model_args_2_later_vecs_of_mats_double[[1]] <-  (Model_args_as_Rcpp_List$Model_args_2_later_vecs_of_mats_double[[1]])
                            
                      }
                      
                      
                      
                
                # Model_args_as_Rcpp_List$model_so_file <-  "none"
                # Model_args_as_Rcpp_List$json_file_path <- "none"
                      
                      # print(Model_args_as_Rcpp_List$Model_args_col_vecs_double)
                      # print(paste("marker 1"))
                      #  (str(Model_args_as_Rcpp_List$Model_args_2_later_vecs_of_mats_double[[1]]))
                      # print(paste("marker 2"))
                      # 
                      # Model_args_as_Rcpp_List$Model_args_2_later_vecs_of_mats_double[[1]] <- list(Model_args_as_Rcpp_List$Model_args_2_later_vecs_of_mats_double[[1]])
                      # 
                      # print(paste("marker 1"))
                      # (str(Model_args_as_Rcpp_List$Model_args_2_later_vecs_of_mats_double[[1]]))
                      # print(paste("marker 2"))
                #       
                #       str((Model_args_as_Rcpp_List$Model_args_2_later_vecs_of_mats_double))
                #       str((Model_args_as_Rcpp_List$Model_args_2_later_vecs_of_mats_double[[1]]))
                #       
                # theta_vec <- rep(0.01, n_params)
                # print(n_params_main)
                # print(n_nuisance)
                # lp_grad_outs <- parallel::mcparallel(fn_Rcpp_wrapper_fn_lp_grad( Model_type = "MVP",
                #                                             force_autodiff = FALSE,
                #                                             force_PartialLog = FALSE,
                #                                             theta_main_vec = matrix(theta_vec[index_main]),
                #                                             theta_us_vec = matrix(theta_vec[index_us]),
                #                                             y = y,
                #                                             grad_option = "main_only",
                #                                             Model_args_as_Rcpp_List = Model_args_as_Rcpp_List))
                # #
                # 
                # lp_grad_outs <- parallel::mccollect(lp_grad_outs)
 
                
                # # parallel::mcparallel
                
                RcppParallel::setThreadOptions(numThreads = n_chains_burnin);
                
                init_burnin_object <-                (          init_and_run_burnin( Model_type = Model_type,
                                                                                     sample_nuisance = sample_nuisance,
                                                                                     y = y,
                                                                                     N = N,
                                                                                     init_object = init_object,
                                                                                     n_chains_burnin = n_chains_burnin,
                                                                                     n_params_main = n_params_main,
                                                                                     n_nuisance = n_nuisance,
                                                                                     shrinkage_factor = shrinkage_factor,
                                                                                     diffusion_HMC = diffusion_HMC,
                                                                                     metric_type_main = metric_type_main,
                                                                                     metric_shape_main = metric_shape_main,
                                                                                     metric_type_nuisance = metric_type_nuisance,
                                                                                     metric_shape_nuisance = metric_shape_nuisance,
                                                                                     seed = seed,
                                                                                     n_burnin = n_burnin,
                                                                                     adapt_delta = adapt_delta,
                                                                                     LR_main = LR_main,
                                                                                     LR_us = LR_us,
                                                                                     n_adapt = n_adapt,
                                                                                     partitioned_HMC = partitioned_HMC,
                                                                                     clip_iter = clip_iter,
                                                                                     gap =gap,
                                                                                     max_eps_main = max_eps_main,
                                                                                     max_eps_us = max_eps_us,
                                                                                     tau_main_target = tau_main_target,
                                                                                     tau_us_target = tau_us_target,
                                                                                     main_L_manual = main_L_manual,
                                                                                     L_main_if_manual = L_main_if_manual,
                                                                                     us_L_manual = us_L_manual,
                                                                                     L_us_if_manual = L_us_if_manual,
                                                                                     max_L = max_L,
                                                                                     ratio_M_us = ratio_M_us,
                                                                                     ratio_M_main = ratio_M_main,
                                                                                     interval_width_main = interval_width_main,
                                                                                     interval_width_nuisance = interval_width_nuisance,
                                                                                     tau_mult = tau_mult,
                                                                                     n_nuisance_to_track = 10,
                                                                                     force_autodiff = force_autodiff,
                                                                                     force_PartialLog = force_PartialLog,
                                                                                     multi_attempts = multi_attempts,
                                                                                     Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                                                     ...
                                                                                     ))
        

                {

                  theta_main_vectors_all_chains_input_from_R <- init_burnin_object$theta_main_vectors_all_chains_input_from_R
                  theta_us_vectors_all_chains_input_from_R <- init_burnin_object$theta_us_vectors_all_chains_input_from_R

                  Model_args_as_Rcpp_List <- init_burnin_object$Model_args_as_Rcpp_List
                  EHMC_args_as_Rcpp_List <- init_burnin_object$EHMC_args_as_Rcpp_List
                  EHMC_Metric_as_Rcpp_List <- init_burnin_object$EHMC_Metric_as_Rcpp_List
                  EHMC_burnin_as_Rcpp_List <- init_burnin_object$EHMC_burnin_as_Rcpp_List

                  time_burnin <- init_burnin_object$time_burnin

                }

                {


                  post_burnin_prep_inits <-  R_fn_post_burnin_prep_for_sampling(n_chains_sampling = n_chains_sampling,
                                                                                n_superchains = n_superchains,
                                                                                n_params_main = Model_args_as_Rcpp_List$n_params_main,
                                                                                n_nuisance = Model_args_as_Rcpp_List$n_nuisance,
                                                                                theta_main_vectors_all_chains_input_from_R,
                                                                                theta_us_vectors_all_chains_input_from_R)

                  theta_main_vectors_all_chains_input_from_R <- post_burnin_prep_inits$theta_main_vectors_all_chains_input_from_R
                  theta_us_vectors_all_chains_input_from_R <- post_burnin_prep_inits$theta_us_vectors_all_chains_input_from_R



                }


                {

                  gc(reset = TRUE)


                  tictoc::tic("post-burnin timer")

                  if (Model_type != "Stan") {
                      Model_args_as_Rcpp_List$model_so_file <- "none"
                      Model_args_as_Rcpp_List$json_file_path <- "none"
                  }


                  RcppParallel::setThreadOptions(numThreads = n_chains_sampling);

                  ## Rcpp_fn_RcppParallel_EHMC_sampling
                  # Rcpp_fn_openMP_EHMC_sampling
                  # parallel::mcparallel

                  # if (Model_type != "Stan")  {
                  #      Model_args_as_Rcpp_List$Model_args_ints[4, 1] <- num_chunks
                  # }

                  if (parallel_method == "OpenMP") { 
                    Rcpp_parallel_sampling_fn <- Rcpp_fn_openMP_EHMC_sampling
                  } else { ###  use RcppParallel
                    Rcpp_parallel_sampling_fn <- Rcpp_fn_RcppParallel_EHMC_sampling
                  }
                  
                  
                  ### Call C++ parallel sampling function
                  result <-       (Rcpp_parallel_sampling_fn(   n_threads_R = n_chains_sampling,
                                                                sample_nuisance_R = sample_nuisance,
                                                                n_nuisance_to_track = n_nuisance_to_track,
                                                                seed_R = seed,
                                                                iter_one_by_one = FALSE,
                                                                n_iter_R = n_iter,
                                                                partitioned_HMC_R = partitioned_HMC,
                                                                Model_type_R = Model_type,
                                                                force_autodiff_R = force_autodiff,
                                                                force_PartialLog = force_PartialLog,
                                                                multi_attempts_R = multi_attempts,
                                                                theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                                                theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R,
                                                                y =  y,  
                                                                Model_args_as_Rcpp_List =  Model_args_as_Rcpp_List,
                                                                EHMC_args_as_Rcpp_List =   EHMC_args_as_Rcpp_List,
                                                                EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List))



                   # result <- parallel::mccollect(result) ; result <- result[[1]]


                  try({
                    {
                      print(tictoc::toc(log = TRUE))
                      log.txt <- tictoc::tic.log(format = TRUE)
                      tictoc::tic.clearlog()
                      time_sampling <- unlist(log.txt)
                    }
                  })

                  # gc()

                  try({
                    time_sampling <- as.numeric( substr(start = 0, stop = 100,  strsplit(  strsplit(time_sampling, "[:]")[[1]], "[s]")[[2]][1] ) )
                  })

                  print(paste("sampling time = ",  time_sampling) )
                  #  print(paste("total time = ", time_burnin + time_sampling) )

                }
                
                time_total <- time_sampling + time_burnin

  out_list <- list(time_burnin = time_burnin,
                   time_sampling = time_sampling,
                   time_total = time_total,
                   result = result, 
                   init_burnin_object = init_burnin_object)
  
  
  return(out_list)





}




