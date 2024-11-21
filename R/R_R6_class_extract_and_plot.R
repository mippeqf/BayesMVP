
require(R6)

#' @title MVP extract and plot Class
#' @description Class for handling MVP model summaries, samples (trace) and diagnostics.
#'  
#' 
#' @details 
#' This class is called via the main "MVP_model" R6 class.
#' 
#' @section Model Types:
#' * MVP: Multivariate probit model
#' * LC-MVP: Latent class multivariate probit model
#' * latent_trait: Latent trait model
#' * Stan: User-supplied Stan models (files extension must be .stan). 
#' 
#' 
#' @section Class Relationship:
#' This class is not meant to be instantiated/called directly by users. Instead, it is created 
#' and returned by the MVP_model$summary() method. It provides methods for:
#' * Extracting MCMC diagnostics (divergences, ESS, etc.)
#' * Creating trace and density plots
#' * Computing efficiency metrics
#' * Accessing posterior draws in various formats
#' 
#' 
#' @section Common Usage Patterns:
#' \describe{
#'   \item{Basic Diagnostics}{get_divergences() -> get_efficiency_metrics()}
#'   \item{Parameter Summaries}{get_summary_main() -> get_summary_transformed()}
#'   \item{Visualization}{plot_traces() -> plot_densities()}
#'   \item{Custom Analysis}{get_posterior_draws() -> Your analysis}
#' }
#' 
#' 
#' 
#' 
#' @export
MVP_class_extract_and_plot <- R6Class("MVP_class_extract_and_plot",
                             
                               public = list(
                                 
                                    #' @field summary_object The summary object, obtained from the main "MVP_model" R6 class. 
                                    summary_object = NULL,
                                    #' @field init_object Initialization object, obtained from the main "MVP_model" R6 class. 
                                    init_object = NULL,
                                    #' @field n_nuisance The total number of nuisance parameters, please see the "MVP_model" R6 class documentation for more details. 
                                    n_nuisance = NULL,
                                    
                                    #' @param model_summary The summary object, obtained from the main "MVP_model" R6 class. 
                                    #' @param init_object The Initialization object, obtained from the main "MVP_model" R6 class. 
                                    #' @param n_nuisance The total number of nuisance parameters, please see the "MVP_model" R6 class documentation for more details. 
                                    initialize = function(model_summary,  
                                                          init_object,
                                                          n_nuisance) {   
                                      
                                              self$summary_object <- model_summary   
                                              self$init_object <- init_object
                                              self$n_nuisance <- n_nuisance
                                          
                                    },
                                    
                                    
                                    ## Convenience methods for summaries (inc. divergences, R-hat, ESS, posterior means, etc)
                                    #'@return Returns self$summary_object$summaries$divergences obtained from the main "MVP_model" R6 class. 
                                    #'self$summary_object$summaries$divergences contains divergence information (e.g. % of transitions which
                                    #'diverged). 
                                    get_divergences = function() {
                                         self$summary_object$summaries$divergences
                                    },
                                    
                                    #'@return Returns main parameter summaries as a tibble.
                                    get_summary_main = function() {
                                         self$summary_object$summaries$summary_tibbles$summary_tibble_main_params
                                    },
                                    
                                    #'@return Returns transformed parameter summaries as a tibble.
                                    get_summary_transformed = function() {
                                         self$summary_object$summaries$summary_tibbles$summary_tibble_transformed_parameters
                                    },
                                    
                                    #'@return Returns generated quantities as a tibble.
                                    get_summary_generated_quantities = function() {
                                         self$summary_object$summaries$summary_tibbles$summary_tibble_generated_quantities
                                    },
                                  
                                    #'Convenience method for posterior draws (as 3D arrays)
                                    #'@return Returns posterior draws as a 3D array (iterations, chains, parameters). 
                                    get_posterior_draws = function() {
                                      self$summary_object$traces$traces_as_arrays$draws_array
                                    },
                                    
                                    #'Convenience method for all traces (as 3D arrays)
                                    #'@return Returns posterior traces as a 3D arrays (iterations, chains, parameters). 
                                    get_all_traces = function() {
                                      self$summary_object$traces
                                    },
                                    
                                    #'Convenience method for posterior draws for log_lik (as 3D arrays)
                                    #'@return Returns posterior draws for log_lik as a 3D array (iterations, chains, parameters). 
                                    get_log_lik_trace = function() {
                                      self$summary_object$traces$log_lik_trace
                                    },
                                    
                                    #'Convenience method for posterior draws for nuisance (as 3D arrays)
                                    #'@return Returns posterior draws for nuisance as a 3D array (iterations, chains, parameters). 
                                    get_nuisance_trace = function() {
                                      self$summary_object$traces$nuisance_trace
                                    },
                                    
                                    #'Convenience method for posterior draws (as tibbles)
                                    #'@return Returns trace as tibbles. The trace gets returned as 3 seperate tibbles (one tibble for the 
                                    #'main parameters, one for the transformed parameters, and one for generated quantities). 
                                    get_posterior_draws_as_tibbles = function() {
                                      list(
                                         trace_as_tibble_main_params = self$summary_object$traces$traces_as_tibbles$trace_params_main_tibble,
                                         trace_as_tibble_transformed_params =  self$summary_object$traces$traces_as_tibbles$trace_transformed_params_tibble,
                                         trace_as_tibble_generated_quantities =  self$summary_object$traces$traces_as_tibbles$trace_generated_quantities_tibble
                                      )
                                    },
                                    
                                    
                                    ## Convenience methods for efficiency metrics
                                    #' @return Returns a list containing efficiency metrics including:
                                    #' \itemize{
                                    #'   \item time_burnin: Time spent in burnin phase
                                    #'   \item time_sampling: Time spent sampling
                                    #'   \item time_total_MCMC: Total time spent doing MCMC/HMC sampling, excluding time to compute summaries and diagnostics.
                                    #'   \item time_total_inc_summaries: Total time spent doing MCMC/HMC sampling, including time to compute summaries and diagnostics.
                                    #'   \item Min_ESS_main: The minimum ESS of the main model parameters. 
                                    #'   \item Min_ESS_per_sec_sampling: The minimum ESS per second for the sampling phase. 
                                    #'   \item Min_ESS_per_sec_total: The minimum ESS per second for the total model run time, including any time spent computing
                                    #'   summaries and diagnostics. 
                                    #'   \item Min_ESS_per_grad_sampling:  The minimum ESS per gradient evaluation for the sampling phase. 
                                    #'   \item grad_evals_per_sec: The number of gradient evaluations performed per second. 
                                    #'   \item est_time_to_100_ESS_sampling: The estimated sampling time to reach a minimum ESS of 100.
                                    #'   \item est_time_to_1000_ESS_sampling: The estimated sampling time to reach a minimum ESS of 1000.
                                    #'   \item est_time_to_10000_ESS_sampling: The estimated sampling time to reach a minimum ESS of 10,000.
                                    #'   \item est_time_to_100_ESS_wo_summaries: The estimated total time (expluding time to compute model summaries and diagnostics)
                                    #'   to reach a minimum ESS of 100.
                                    #'   \item est_time_to_1000_ESS_wo_summaries: The estimated total time (expluding time to compute model summaries and diagnostics)
                                    #'   to reach a minimum ESS of 1000.
                                    #'   \item est_time_to_10000_ESS_wo_summaries: The estimated total time (expluding time to compute model summaries and diagnostics)
                                    #'   to reach a minimum ESS of 10,000.
                                    #'   \item est_time_to_100_ESS_inc_summaries: The estimated total time (including time spent computing model summaries and 
                                    #'   diagnostics) to reach a minimum ESS of 100.
                                    #'   \item est_time_to_1000_ESS_inc_summaries: The estimated total time (including time spent computing model summaries and 
                                    #'   diagnostics) to reach a minimum ESS of 1000.
                                    #'   \item est_time_to_10000_ESS_inc_summaries: The estimated total time (including time spent computing model summaries and 
                                    #'   diagnostics) to reach a minimum ESS of 10,000.
                                    #' }
                                    get_efficiency_metrics = function() {
                                      list(
                                        ## times
                                        time_burnin = self$summary_object$summaries$efficiency_info$time_burnin,
                                        time_sampling = self$summary_object$summaries$efficiency_info$time_sampling,
                                        time_total_MCMC = self$summary_object$summaries$efficiency_info$time_total_wo_summaries,
                                        time_total_inc_summaries = self$summary_object$summaries$efficiency_info$time_total,
                                        # Now extract some more specific efficiency info:
                                        Min_ESS_main = self$summary_object$summaries$efficiency_info$Min_ESS_main,
                                        Min_ESS_per_sec_sampling = self$summary_object$summaries$efficiency_info$ESS_per_sec_samp,
                                        Min_ESS_per_sec_total = self$summary_object$summaries$efficiency_info$ESS_per_sec_total,
                                        Min_ESS_per_grad_sampling = self$summary_object$summaries$efficiency_info$Min_ess_per_grad_samp_weighted,
                                        grad_evals_per_sec = self$summary_object$summaries$efficiency_info$grad_evals_per_sec,
                                        ## extract the "time to X ESS" (for sampling time)
                                        est_time_to_100_ESS_sampling = self$summary_object$summaries$efficiency_info$sampling_time_to_100_ESS,
                                        est_time_to_1000_ESS_sampling = self$summary_object$summaries$efficiency_info$sampling_time_to_1000_ESS,
                                        est_time_to_10000_ESS_sampling = self$summary_object$summaries$efficiency_info$sampling_time_to_10000_ESS,
                                        ## extract the "time to X ESS" (for total time w/o summaries)
                                        est_time_to_100_ESS_wo_summaries = self$summary_object$summaries$efficiency_info$total_time_to_100_ESS_wo_summaries,
                                        est_time_to_1000_ESS_wo_summaries = self$summary_object$summaries$efficiency_info$total_time_to_1000_ESS_wo_summaries,
                                        est_time_to_10000_ESS_wo_summaries = self$summary_object$summaries$efficiency_info$total_time_to_1000_ESS_wo_summaries,
                                        ## extract the "time to X ESS" (for total time inc. summaries)
                                        est_time_to_100_ESS_inc_summaries = self$summary_object$summaries$efficiency_info$total_time_to_100_ESS_with_summaries,
                                        est_time_to_1000_ESS_inc_summaries = self$summary_object$summaries$efficiency_info$total_time_to_1000_ESS_with_summaries,
                                        est_time_to_10000_ESS_inc_summaries = self$summary_object$summaries$efficiency_info$total_time_to_10000_ESS_with_summaries
                                        
                                      )
                                    },
                                    
                                    
                                    
                                    ## -- Convenience fn to compute "time to X ESS" using the "$time_to_ESS()" method --------------------------------------
                                    #'@param target_ESS The target ESS. 
                                    #'@return Returns a list called "target_ESS_times" which contains the estimated sampling time to reach the target ESS 
                                    #'("sampling_time_to_target_ESS"), the estimated total time excluding the time it takes to compute model summaries 
                                    #'("total_time_to_target_ESS_wo_summaries"), and the total estimated time to reach the 
                                    #'target ESS ("total_time_to_target_ESS_with_summaries").
                                    time_to_target_ESS = function(target_ESS) {
                                      
                                      if (is.null(self$summary_object)) {
                                        stop("No summary object available")   
                                      }
                                      
                                      # get required efficiency info first
                                      Min_ESS_main   <- self$summary_object$summaries$efficiency_info$Min_ESS_main
                                      time_burnin    <- self$summary_object$summaries$efficiency_info$time_burnin
                                      time_sampling  <- self$summary_object$summaries$efficiency_info$time_sampling
                                      time_summaries <- self$summary_object$summaries$efficiency_info$time_summaries
                                      
                                      sampling_time_to_Min_ESS <- time_sampling
                                      
                                      ## est. sampling time to target_ESS
                                      sampling_time_to_target_ESS <- (target_ESS / Min_ESS_main) * sampling_time_to_Min_ESS
                                      
                                      ## total w/o summary time
                                      total_time_to_target_ESS_wo_summaries <-   time_burnin + sampling_time_to_target_ESS
                                      
                                      ## total w/ summary time
                                      summary_time_to_Min_ESS <- time_summaries
                                      summary_time_to_target_ESS <- (target_ESS / Min_ESS_main) * summary_time_to_Min_ESS
                                      
                                      total_time_to_target_ESS_with_summaries <-   time_burnin + sampling_time_to_target_ESS   + summary_time_to_target_ESS
                                      
                                      target_ESS_times <- list(sampling_time_to_target_ESS = sampling_time_to_target_ESS,
                                                               total_time_to_target_ESS_wo_summaries = total_time_to_target_ESS_wo_summaries,
                                                               total_time_to_target_ESS_with_summaries = total_time_to_target_ESS_with_summaries)
                                      
                                      return(target_ESS_times)
                                      
                                      
                                    },
 
                                    
                                    ## -- trace plots method --------------------------------------------------------------------------------------------------
                                    #'@param params The parameters to generate trace plots for. This is a character vector - e.g. to plot trace plots for 
                                    #'beta: params = c("beta"). 
                                    #'The default is NULL and the trace plots for all model parameters (which have a trace array) will be plotted. 
                                    #'@param batch_size The number of trace plots to display per panel. Default is 9. 
                                    #'@return If no parameters specified (i.e. params = NULL), then this will return an object containing the trace plots for all
                                    #'model parameters  which have a trace array.
                                    plot_traces = function(params = NULL, 
                                                           batch_size = 9
                                                           ) {
                                      
                                                    if (is.null(self$summary_object)) {
                                                      stop("No summary object available")   
                                                    }
                                                    
                                                    if (!is.null(params) && !is.character(params)) {
                                                      stop("params must be NULL or a character vector")
                                                    }
                                      
                                                    if (!is.numeric(batch_size) || batch_size < 1) {
                                                      stop("batch_size must be a positive integer")
                                                    }
                                                    
                                                    # # Debug print
                                                    # cat("Summary object structure:\n")
                                                    # str(self$summary_object)
                                                    
                                                    # Get the draws array
                                                    draws_array <- self$summary_object$traces$traces_as_arrays$draws_array 
                                                    
                                                    if (is.null(draws_array)) {
                                                      stop("draws_array is NULL in summary_object")
                                                    }
                                      
                                                    if (is.null(params)) { # plot all params
                                                      
                                                          bayesplot::mcmc_trace(draws_array)
                                                      
                                                    } else {   # plot specific params using custom "plot_multiple_params_batched" fn - mimics Stan's method but uses bayesplot 
                                                      
                                                                    plot_multiple_params_batched( draws_array = draws_array, 
                                                                                                  param_prefixes = params, 
                                                                                                  plot_type = "trace", 
                                                                                                  batch_size = batch_size)
                                                    }
                                      
                                    },
                                    
                                    ## -- density plots method -------------------------------------------------------------------------------------------------
                                    #'@param params The parameters to generate posterior density plots for. This is a character vector - e.g. to plot the 
                                    #'posterior density plots for beta: params = c("beta"). 
                                    #'The default is NULL and the posterior density plots for all model parameters (which have a trace array) will be plotted. 
                                    #'@param batch_size The number of posterior density plots to display per panel. Default is 9. 
                                    #'@return If no parameters specified (i.e. params = NULL), then this will return an object containing the posterior density
                                    #'plots for all model parameters which have a trace array.
                                    plot_densities = function(params = NULL, 
                                                              batch_size =  9
                                                              ) {
                                      
                                                      if (is.null(self$summary_object)) {
                                                        stop("No summary object available")  
                                                      }
                                      
                                                      if (!is.null(params) && !is.character(params)) {
                                                        stop("params must be NULL or a character vector")
                                                      }
                                                      
                                                      if (!is.numeric(batch_size) || batch_size < 1) {
                                                        stop("batch_size must be a positive integer")
                                                      }
                                      
                                                      # Get the draws array
                                                      draws_array <- self$summary_object$traces$traces_as_arrays$draws_array
                                                      
                                                      if (is.null(params)) { # plot all params
                                                        
                                                            bayesplot::mcmc_dens(draws_array)
                                                        
                                                      } else {   # plot specific params using custom "plot_multiple_params_batched" fn - mimics Stan's method but uses bayesplot
                                                            
                                                                      plot_multiple_params_batched( draws_array = draws_array, 
                                                                                                    param_prefixes = params, 
                                                                                                    plot_type = "density", 
                                                                                                    batch_size = batch_size)
                                                      }
                                                        
                                    }

                
                        
                      )
)
























