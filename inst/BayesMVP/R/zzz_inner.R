


#' cmdstanr_path
#' @export
cmdstanr_path <- function() {
          
          # Check if the CMDSTAN environment variable is set
          try({
            cmdstan_env <- cmdstanr::cmdstan_path()
            if (!(cmdstan_env %in% c("", " ", "  "))) {
              message("Using CMDSTAN path: ", cmdstan_env)
              return(cmdstan_env) # Use the value from the environment variable
            }
          }, silent = TRUE)
          
          # Get the user's home directory
          home_dir <- Sys.getenv(if (.Platform$OS.type == "windows") "USERPROFILE" else "HOME")
          
          # Check for .cmdstan directory
          cmdstan_dirs <- Sys.glob(file.path(home_dir, ".cmdstan", "cmdstan-*"))
          
          if (length(cmdstan_dirs) > 0) {
            # Sort directories by version (assumes lexicographical sorting works for version strings)
            recent_dir <- cmdstan_dirs[order(cmdstan_dirs, decreasing = TRUE)][1]
            message("Found latest cmdstan in .cmdstan: ", recent_dir)
            return(recent_dir)
          }
          
          # Check for cmdstan directory directly under HOME
          cmdstan_dir <- file.path(home_dir, "cmdstan")
          if (dir.exists(cmdstan_dir)) {
            message("Found cmdstan in home directory: ", cmdstan_dir)
            return(cmdstan_dir)
          }
          
          # If no valid path is found
          stop("CmdStan directory not found. Please install CmdStan or set the CMDSTAN environment variable.")
  
}






#' bridgestan_path
#' @export
bridgestan_path <- function() {
  
          # Check if the BRIDGESTAN environment variable is already set
          if (!(Sys.getenv("BRIDGESTAN") %in% c("", " ", "  "))) {
            message(paste(cat("Bridgestan path found at:"), Sys.getenv("BRIDGESTAN")))
            return(Sys.getenv("BRIDGESTAN")) # Use the value from the environment variable
          }
          
          # Get the user's home directory
          home_dir <- Sys.getenv(if (.Platform$OS.type == "windows") "USERPROFILE" else "HOME")
          
          # Define the default paths for BridgeStan v2.5.0
          default_path <- file.path(home_dir, ".bridgestan", "bridgestan-2.5.0")
          
          # Check if the default path exists
          if (dir.exists(default_path) == TRUE) {
            Sys.setenv(BRIDGESTAN=default_path)
            message(paste(cat("Bridgestan path found at:"), default_path))
            return(default_path)
          }
          
          
          # If v2.5.0 not available, then fallback to finding the most recent bridgestan directory
          search_pattern <- file.path(home_dir, ".bridgestan" , "bridgestan-*")
          available_dirs <- Sys.glob(search_pattern)
          
          # Filter for valid version directories and sort by version
          if (length(available_dirs) > 0) {
            recent_dir <- available_dirs[order(available_dirs, decreasing = TRUE)][1]
            Sys.setenv(BRIDGESTAN=recent_dir)
            message(paste(cat("Bridgestan path found at:"), recent_dir))
            return(recent_dir)
          }
          
          ## If nothing, then look for plain "bridgestan" dir (i.e. w/o a ".")
          search_pattern_wo_dot <- file.path(home_dir, "bridgestan")
          available_dir_wo_dot <- Sys.glob(search_pattern_wo_dot) ; available_dir_wo_dot
          return(available_dir_wo_dot)
          
          
          # If no valid path is found
          stop("BridgeStan directory not found.")
  
}






#' setup_env_post_install
#' @export
setup_env_post_install <- function() {
  
  
          # Set brigestan and cmdstanr environment variables / directories
          ## bs_dir <- bridgestan_path()
          ## cmdstan_dir <- cmdstanr_path()
          
          mvp_user_dir <- file.path(Sys.getenv("USERPROFILE"), ".BayesMVP")
          
          if (.Platform$OS.type == "windows") {
            
                    TBB_STAN_DLL <- TBB_CMDSTAN_DLL <- DUMMY_MODEL_SO <- DUMMY_MODEL_DLL <- NULL
                    
                    cat("Setting up BayesMVP Environment for Windows:\n")
                    
                    cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                    try({   TBB_STAN_DLL <- file.path(mvp_user_dir, "tbb.dll") })
                    ## try({   TBB_CMDSTAN_DLL <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "tbb.dll") }) # prioritise user's installed tbb dll/so
                    try({   DUMMY_MODEL_SO <- file.path(mvp_user_dir, "dummy_stan_modeL_win_model.so") })
                    try({   DUMMY_MODEL_DLL <- file.path(mvp_user_dir, "dummy_stan_modeL_win_model.dll") })
                    
                    dll_paths <- c(TBB_STAN_DLL,
                                   ## TBB_CMDSTAN_DLL,
                                   DUMMY_MODEL_SO,
                                   DUMMY_MODEL_DLL)
            
          }  else {  ### if Linux or Mac
            
                    TBB_STAN_SO <- TBB_CMDSTAN_SO <- DUMMY_MODEL_SO <- NULL
                    
                    cat("Setting up BayesMVP Environment for Linux / Mac OS:\n")
                    
                    cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                    try({  TBB_STAN_SO <- file.path(mvp_user_dir, "libtbb.so.2") })
                    ## try({  TBB_CMDSTAN_SO <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "libtbb.so.2") })  # prioritise user's installed tbb dll/so
                    try({  DUMMY_MODEL_SO <- file.path(mvp_user_dir, "dummy_stan_model_model.so") })
                    
                    dll_paths <- c(TBB_STAN_SO,
                                   ## TBB_CMDSTAN_SO,
                                   DUMMY_MODEL_SO)
            
          }
  
  
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
          

          
          
  
}







#' .onLoad
#' @export
.onLoad <- function(libname, 
                    pkgname) {
  
      is_windows <- .Platform$OS.type == "windows"
      
      dll_path <- file.path(libname, pkgname)
      
      try({  comment(print(paste(dll_path))) })
      try({  comment(print(dll_path)) })
      
      if (is_windows == TRUE) {
        
       ## try({ dyn.load(file.path(dll_path, "tbb.dll")) })
        try({ dyn.load(file.path(dll_path, "dummy_stan_model_win_model.so")) })
        try({ dyn.load(file.path(dll_path, "dummy_stan_model_win_model.dll")) })
        ## try({ dyn.load(file.path(dll_path, "R.dll")) })
        try({ dyn.load(file.path(dll_path, "BayesMVP.dll")) })
        
      } else { 
        # dyn.load(file.path(dll_path, "dummy_stan_model_model.so"))
        ##    setup_env_post_install() 
        ##   try({  .make_libs(libname, pkgname) }, silent = TRUE)
      }
      

  
}



#' .onAttach
#' @export
.onAttach <- function(libname, 
                      pkgname) {

   setup_env_post_install()  
  
}


#' .First.lib
#' @export
.First.lib <- function(libname, 
                       pkgname) {
 
   setup_env_post_install()  
  
}





