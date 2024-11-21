


#' initialise_model
#' @keywords internal
#' @export
initialise_model  <-    function(     Model_type,
                                      compile = TRUE,
                                      cmdstanr_model_fit_obj = NULL,
                                      y = NULL,
                                      N,
                                      n_params_main,
                                      n_nuisance,
                                      init_lists_per_chain,
                                      sample_nuisance = NULL,
                                      n_chains_burnin = NULL,
                                      model_args_list = NULL,
                                      Stan_data_list = NULL,
                                      Stan_model_file_path = NULL,
                                      Stan_cpp_user_header = NULL,
                                      ...) { 
  
  
                if (sample_nuisance == FALSE) { 
                  n_nuisance <- 9 # dummy
                }
  
                # user inputs Stan_model_file_path (for Stan)
                init_model_object <- init_model(Model_type = Model_type,
                                                          y = y, 
                                                          N = N,
                                                          n_params_main = n_params_main,
                                                          n_nuisance = n_nuisance,
                                                          model_args_list = model_args_list, # this arg is only for MANUAL models
                                                          Stan_data_list = Stan_data_list,
                                                          Stan_cpp_user_header = Stan_cpp_user_header,
                                                          Stan_model_file_path = Stan_model_file_path,
                                                          ...)
                
                
                
                init_vals_object <- init_inits(          init_model_outs = init_model_object,
                                                         compile = compile,
                                                         cmdstanr_model_fit_obj = cmdstanr_model_fit_obj,
                                                         sample_nuisance = sample_nuisance,
                                                         init_lists_per_chain = init_lists_per_chain,
                                                         Stan_model_file_path = Stan_model_file_path,
                                                         Stan_cpp_user_header = Stan_cpp_user_header,
                                                         Stan_data_list = Stan_data_list,
                                                         n_chains_burnin = n_chains_burnin,
                                                         N = N,
                                                         n_params_main = n_params_main,
                                                         n_nuisance = n_nuisance,
                                                         ...)
    
                param_names <- init_vals_object$param_names
                Stan_model_file_path <- init_vals_object$Stan_model_file_path
                
                json_file_path <- init_vals_object$json_file_path
                model_so_file <- init_vals_object$model_so_file
                
                dummy_json_file_path <- init_vals_object$dummy_json_file_path
                dummy_model_so_file <- init_vals_object$dummy_model_so_file
                
                if (compile == FALSE) {  # use the inputted cmdstanr_model_fit_obj
                  cmdstanr_model_fit_obj <- cmdstanr_model_fit_obj
                } else { 
                  cmdstanr_model_fit_obj <- init_vals_object$cmdstanr_model_fit_obj
                }
                
                # init_vals_object$Model_args_as_Rcpp_List$json_file_path <-  json_file_path ##
                # init_vals_object$Model_args_as_Rcpp_List$model_so_file <-   model_so_file ##
                
                Model_args_as_Rcpp_List <-   init_vals_object$Model_args_as_Rcpp_List
                
                theta_us_vectors_all_chains_input_from_R <- init_vals_object$theta_us_vectors_all_chains_input_from_R
                theta_main_vectors_all_chains_input_from_R <- init_vals_object$theta_main_vectors_all_chains_input_from_R
                
                init_object <- list(Model_type = Model_type,
                                     y = y,
                                     N = N,
                                     n_params_main = n_params_main,
                                     n_nuisance = n_nuisance,
                                     sample_nuisance = sample_nuisance,
                                     init_lists_per_chain = init_lists_per_chain,
                                     n_chains_burnin = n_chains_burnin,
                                    ### model_args_list = model_args_list,
                                     Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                    ### Stan_data_list = Stan_data_list,
                                     param_names = param_names,
                                     theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R,
                                     theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                     json_file_path = json_file_path,
                                     model_so_file = model_so_file,
                                     dummy_model_so_file = dummy_model_so_file,
                                     dummy_json_file_path = dummy_json_file_path,
                                     Stan_model_file_path = Stan_model_file_path,
                                     Stan_cpp_user_header = Stan_cpp_user_header,
                                     cmdstanr_model_fit_obj = cmdstanr_model_fit_obj)
 
  
  return(init_object)

 

}




