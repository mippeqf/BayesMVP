


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
          
          # Define the default path for BridgeStan
          default_path <- file.path(home_dir, ".bridgestan", "bridgestan-2.5.0")
          
          # Check if the default path exists
          if (dir.exists(default_path) == TRUE) {
            Sys.setenv(BRIDGESTAN=default_path)
            message(paste(cat("Bridgestan path found at:"), default_path))
            return(default_path)
          }
          
          
          # Fallback to finding the most recent bridgestan directory
          search_pattern <- file.path(home_dir, ".bridgestan", "bridgestan-*")
          available_dirs <- Sys.glob(search_pattern)
          
          # Filter for valid version directories and sort by version
          if (length(available_dirs) > 0) {
            recent_dir <- available_dirs[order(available_dirs, decreasing = TRUE)][1]
            Sys.setenv(BRIDGESTAN=recent_dir)
            message(paste(cat("Bridgestan path found at:"), recent_dir))
            return(recent_dir)
          }
          
          # Additional fallback for Windows-specific path
          if (.Platform$OS.type == "windows") {
            
                  windows_default_path <- "C:/.bridgestan/bridgestan-2.5.0"
                  
                  if (dir.exists(windows_default_path)) {
                      Sys.setenv(BRIDGESTAN=windows_default_path)
                    message(paste(cat("Bridgestan path found at:"), windows_default_path))
                      return(windows_default_path)
                  }
                  
                  
                  ## otherwise look for paths without a "." if user installed w/o R
                  windows_search_pattern <- "C:/bridgestan-*"
                  available_dirs <- Sys.glob(windows_search_pattern)
                  
                  if (length(available_dirs) > 0) {
                    recent_dir <- available_dirs[order(available_dirs, decreasing = TRUE)][1]
                    Sys.setenv(BRIDGESTAN=recent_dir)
                    message(paste(cat("Bridgestan path found at:"), recent_dir))
                    return(recent_dir)
                  }
                  
          }
          
       
         
          
          # If no valid path is found
          stop("BridgeStan directory not found.")
  
}



#' setup_env
#' @export
setup_env <- function() {
  
  
                    try({ 
                      options(devtools.install.args = c("--no-test-load"))
                    })
                            
                    ## Set brigestan and cmdstanr environment variables / directories 
                    bs_dir <- bridgestan_path()
                    cmdstan_dir <- cmdstanr_path()
            
            if (.Platform$OS.type == "windows") {
              
                      cat("Setting up BayesMVP Environment for Windows:\n")
              
                      cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                      TBB_STAN_DLL <- file.path(system.file(package = "BayesMVP"), "tbb_stan", "tbb.dll")
                      TBB_CMDSTAN_DLL <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "tbb.dll")  # prioritise user's installed tbb dll/so
                      DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"), "dummy_stan_modeL_win_model.so")
                      DUMMY_MODEL_DLL <- file.path(system.file(package = "BayesMVP"), "dummy_stan_modeL_win_model.dll")
                      
                      dll_paths <- c(TBB_STAN_DLL, 
                                     TBB_CMDSTAN_DLL,
                                     DUMMY_MODEL_SO, 
                                     DUMMY_MODEL_DLL)
                      
            }  else {  ### if Linux or Mac
              
                      cat("Setting up BayesMVP Environment for Linux / Mac OS:\n")
                      
                      cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                      TBB_STAN_SO <- file.path(system.file(package = "BayesMVP"), "tbb_stan", "libtbb.so.2")
                      TBB_CMDSTAN_SO <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "libtbb.so.2")  # prioritise user's installed tbb dll/so
                      DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"), "dummy_stan_modeL_win_model.so")
                      
                      dll_paths <- c(TBB_STAN_SO, 
                                     TBB_CMDSTAN_SO,
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

  
  
 


#' install_BayesMVP
#' @export
install_BayesMVP <- function() {
  
          require(cmdstanr)
          require(bridgestan)
    
          BayesMVP:::setup_env()
          
          try({ 
            options(devtools.install.args = c("--no-test-load"))
          })
          
          devtools::install_github(repo = "https://github.com/CerulloE1996/BayesMVP", 
                                   force = TRUE, 
                                   build_opts  = c("--no-test-load"))
          
          BayesMVP:::setup_env() # setup env again before loading 
          
          require(BayesMVP)
          
 
  
}





