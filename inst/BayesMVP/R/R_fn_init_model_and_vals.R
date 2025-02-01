


#' initialise_model
#' @keywords internal
#' @export
initialise_model  <-    function(     Model_type,
                                      compile,
                                      force_recompile,
                                      cmdstanr_model_fit_obj ,
                                      y,
                                      N,
                                      n_params_main,
                                      n_nuisance,
                                      init_lists_per_chain,
                                      sample_nuisance,
                                      n_chains_burnin,
                                      model_args_list, # this arg is only for MANUAL models
                                      Stan_data_list,
                                      Stan_model_file_path,
                                      Stan_cpp_user_header,
                                      Stan_cpp_flags,
                                      ...) { 
  
  
                if (sample_nuisance == FALSE) { 
                  n_nuisance <- 9 # dummy
                }
  
                # user inputs Stan_model_file_path (for Stan)
                init_model_object <- init_model(          Model_type = Model_type,
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
                                                         force_recompile = force_recompile,
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
                                                         Stan_cpp_flags = Stan_cpp_flags,
                                                         ...)
                
                if (Model_type != "Stan") {
                  Stan_data_list <- init_vals_object$Stan_data_list
                }
    
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
                
                bs_model <- init_vals_object$bs_model
                
                Model_args_as_Rcpp_List <-   init_vals_object$Model_args_as_Rcpp_List
                
                theta_us_vectors_all_chains_input_from_R <- init_vals_object$theta_us_vectors_all_chains_input_from_R
                theta_main_vectors_all_chains_input_from_R <- init_vals_object$theta_main_vectors_all_chains_input_from_R
                
                init_object <- list( Model_type = Model_type,
                                     y = y,
                                     N = N,
                                     n_params_main = n_params_main,
                                     n_nuisance = n_nuisance,
                                     sample_nuisance = sample_nuisance,
                                     ##
                                     bs_model = bs_model,
                                     ##
                                     init_lists_per_chain = init_lists_per_chain,
                                     n_chains_burnin = n_chains_burnin,
                                     ##
                                     param_names = param_names,
                                     ## Initial value vectors (for each chain):
                                     theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R,
                                     theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                     ##
                                     Stan_model_file_path = Stan_model_file_path,
                                     model_so_file = model_so_file,
                                     dummy_model_so_file = dummy_model_so_file,
                                     ##
                                     model_args_list = model_args_list, # this arg is only for MANUAL models
                                     Model_args_as_Rcpp_List = Model_args_as_Rcpp_List, ## MOSTLY only used for manual models (except for n_nuisance and n_params_main)
                                     ##
                                     Stan_data_list = Stan_data_list,
                                     json_file_path = json_file_path,
                                     dummy_json_file_path = dummy_json_file_path,
                                     ##
                                     Stan_cpp_user_header = Stan_cpp_user_header,
                                     Stan_cpp_flags = Stan_cpp_flags,
                                     cmdstanr_model_fit_obj = cmdstanr_model_fit_obj)
 
  
  return(init_object)

 

}




