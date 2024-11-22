

init_bs_model <- function(Stan_data_list, 
                          Stan_model_name,
                          ...) {
    
        ###  Stan_model_name <- "PO_LC_MVP_bin.stan"  #### TEMP
       ###   pkg_dir <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP" #### TEMP
        
        Sys.setenv(STAN_THREADS = "true")
  
        # Get package directory paths
        pkg_dir <- system.file(package = "BayesMVP")
        stan_dir <- file.path(pkg_dir, "/stan_models")
        data_dir <- file.path(pkg_dir, "/stan_data")  # directory to store data inc. JSON data files
        
        # Create data directory if it doesn't exist
        if (!dir.exists(data_dir)) {
          dir.create(data_dir, recursive = TRUE)
        }
        
        # Stan model path
        Stan_model_file_path <- file.path(stan_dir, Stan_model_name)
        Stan_model <- file.path(Stan_model_file_path)
        
        # make persistent (non-temp) JSON data file path with unique identifier
        data_hash <- digest::digest(Stan_data_list)  # Hash the data to create unique identifier
        json_filename <- paste0("data_", data_hash, ".json")
        json_file_path <- file.path(data_dir, json_filename)
        
        # write JSON data using cmdstanr
        cmdstanr::write_stan_json(Stan_data_list, json_file_path)
        
        # # put in list for C++ struct
        # Model_args_as_Rcpp_List$json_file_path <- json_file_path
        
        # Create bridgestan model
        bs_model <- bridgestan::StanModel$new(
          lib = Stan_model,
          data = json_file_path,
          seed = 123)
        
        # Return both model and data path
        return(list(
          bs_model = bs_model,
          json_file_path = json_file_path,
          Stan_model_file_path = Stan_model_file_path))
  
}








