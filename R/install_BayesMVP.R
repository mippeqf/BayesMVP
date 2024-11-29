


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



#' find_install_temp_dir
#' @export
find_install_temp_dir <- function() {

  # Get the library path being used for installation
  library_path <- .libPaths()[1]  # Typically the first entry is the target library path

  # Construct the expected 00LOCK directory path
  lock_dir <- file.path(library_path, paste0("00LOCK-", "BayesMVP"))

  # Check if the directory exists
  if (dir.exists(lock_dir)) {
    return(normalizePath(lock_dir))
  } else {
    stop("Temporary installation directory not found: ", lock_dir)
  }

}


 



#' setup_env_during_install
#' @export
setup_env_during_install <- function() {


          # try({
          #   options(devtools.install.args = c("--no-test-load"))
          # })

          ## Set brigestan and cmdstanr environment variables / directories
          bs_dir <- bridgestan_path()
          cmdstan_dir <- cmdstanr_path()

          temp_dir <- file.path(find_install_temp_dir(), "00new", "BayesMVP")

          paste(cat("temp_dir = "), temp_dir)


          if (.Platform$OS.type == "windows") {

            cat("Setting up BayesMVP Environment for Windows:\n")

            cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
            TBB_STAN_DLL <- file.path(temp_dir,   "tbb_stan", "tbb.dll")
            TBB_CMDSTAN_DLL <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "tbb.dll")  # prioritise user's installed tbb dll/so
            DUMMY_MODEL_SO <- file.path(temp_dir,   "dummy_stan_modeL_win_model.so")
            DUMMY_MODEL_DLL <- file.path(temp_dir,   "dummy_stan_modeL_win_model.dll")

            dll_paths <- c(TBB_STAN_DLL,
                           TBB_CMDSTAN_DLL,
                           DUMMY_MODEL_SO,
                           DUMMY_MODEL_DLL)

          }  else {  ### if Linux or Mac

            cat("Setting up BayesMVP Environment for Linux / Mac OS:\n")

            cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
            TBB_STAN_SO <- file.path(temp_dir,  "tbb_stan", "libtbb.so.2")
            TBB_CMDSTAN_SO <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "libtbb.so.2")  # prioritise user's installed tbb dll/so
            DUMMY_MODEL_SO <- file.path(temp_dir,   "dummy_stan_modeL_win_model.so")

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



 


#' setup_env_pre_install
#' @export
setup_env_pre_install <- function() {
  
  
        # Set brigestan and cmdstanr environment variables / directories
        bs_dir <- bridgestan_path()
        cmdstan_dir <- cmdstanr_path()
        
        if (.Platform$OS.type == "windows") {
                  
                  TBB_STAN_DLL <- TBB_CMDSTAN_DLL <- DUMMY_MODEL_SO <- DUMMY_MODEL_DLL <- NULL
                  
                  cat("Setting up BayesMVP Environment for Windows:\n")
                  
                  cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                  try({   TBB_STAN_DLL <- file.path(system.file(package = "BayesMVP"), "BayesMVP", "inst", "tbb_stan", "tbb.dll") })
                  try({   TBB_CMDSTAN_DLL <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "tbb.dll") }) # prioritise user's installed tbb dll/so
                  try({   DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"),  "BayesMVP", "inst", "dummy_stan_modeL_win_model.so") })
                  try({   DUMMY_MODEL_DLL <- file.path(system.file(package = "BayesMVP"),  "BayesMVP", "inst", "dummy_stan_modeL_win_model.dll") })
                  
                  dll_paths <- c(TBB_STAN_DLL,
                                 # TBB_CMDSTAN_DLL,
                                 DUMMY_MODEL_SO,
                                 DUMMY_MODEL_DLL)
          
        }  else {  ### if Linux or Mac
                  
                  TBB_STAN_SO <- TBB_CMDSTAN_SO <- DUMMY_MODEL_SO <- NULL
                  
                  cat("Setting up BayesMVP Environment for Linux / Mac OS:\n")
                  
                  cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                  try({  TBB_STAN_SO <- file.path(system.file(package = "BayesMVP"),  "BayesMVP", "inst", "tbb_stan", "libtbb.so.2") })
                  try({  TBB_CMDSTAN_SO <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "libtbb.so.2") })  # prioritise user's installed tbb dll/so
                  try({  DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"),  "BayesMVP", "inst", "dummy_stan_modeL_win_model.so") })
                  
                  dll_paths <- c(TBB_STAN_SO,
                                 # TBB_CMDSTAN_SO,
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





#' setup_env_post_install
#' @export
setup_env_post_install <- function() {
  
  
          # Set brigestan and cmdstanr environment variables / directories
          bs_dir <- bridgestan_path()
          cmdstan_dir <- cmdstanr_path()
          
          ## temp_dir <- Sys.getenv("TEMP")
          
          if (.Platform$OS.type == "windows") {
            
                    TBB_STAN_DLL <- TBB_CMDSTAN_DLL <- DUMMY_MODEL_SO <- DUMMY_MODEL_DLL <- NULL
                    
                    cat("Setting up BayesMVP Environment for Windows:\n")
                    
                    cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                    try({   TBB_STAN_DLL <- file.path(system.file(package = "BayesMVP"), "tbb_stan", "tbb.dll") })
                    try({   TBB_CMDSTAN_DLL <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "tbb.dll") }) # prioritise user's installed tbb dll/so
                    try({   DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"), "dummy_stan_modeL_win_model.so") })
                    try({   DUMMY_MODEL_DLL <- file.path(system.file(package = "BayesMVP"), "dummy_stan_modeL_win_model.dll") })
                    
                    dll_paths <- c(TBB_STAN_DLL,
                                   # TBB_CMDSTAN_DLL,
                                   DUMMY_MODEL_SO,
                                   DUMMY_MODEL_DLL)
            
          }  else {  ### if Linux or Mac
                    
                    TBB_STAN_SO <- TBB_CMDSTAN_SO <- DUMMY_MODEL_SO <- NULL
                    
                    cat("Setting up BayesMVP Environment for Linux / Mac OS:\n")
                    
                    cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                    try({  TBB_STAN_SO <- file.path(system.file(package = "BayesMVP"), "tbb_stan", "libtbb.so.2") })
                    try({  TBB_CMDSTAN_SO <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "libtbb.so.2") })  # prioritise user's installed tbb dll/so
                    try({  DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"), "dummy_stan_modeL_win_model.so") })
                    
                    dll_paths <- c(TBB_STAN_SO,
                                   # TBB_CMDSTAN_SO,
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
  
        try({  setup_env_pre_install() })  # setup env (BEFORE removing wrapper pkg)
  
        # Get inner package from inst/ and copy to temp dir
        try({  
            inner_pkg <- system.file("BayesMVP", package = "BayesMVP")
            temp_dir <- file.path(tempdir(), paste0("BayesMVP_temp_", format(Sys.time(), "%H%M%S")))
            dir.create(temp_dir, recursive = TRUE)
            file.copy(inner_pkg, temp_dir, recursive = TRUE)
        })
        
        # Unload installer namespace
        try(unloadNamespace("BayesMVP"), silent = TRUE)
  
        # Uninstall current package
        try(outer_pkg <- system.file(package = "BayesMVP"), silent = TRUE)
        try(devtools::uninstall(outer_pkg), silent = TRUE)
        try(detach("package:BayesMVP", unload = TRUE), silent = TRUE)
        try(remove.packages("BayesMVP"), silent = TRUE)

  
        Sys.sleep(5) 
        
        
        # Remove lock directory
        lock_dir <- file.path(.libPaths()[1], "00LOCK-BayesMVP")
        if (dir.exists(lock_dir)) {
          unlink(lock_dir, recursive = TRUE, force = TRUE)
        }
        
        # Detach and unload the namespace
        if ("BayesMVP" %in% loadedNamespaces()) {
          try(unloadNamespace("BayesMVP"), silent = TRUE)
        }
        
        # Detach the package if loaded
        if ("package:BayesMVP" %in% search()) {
          try(detach("package:BayesMVP", unload = TRUE), silent = TRUE)
        }
        
        # Remove the installed package directory
        pkg_dir <- file.path(.libPaths()[1], "BayesMVP")
        if (dir.exists(pkg_dir)) {
          unlink(pkg_dir, recursive = TRUE, force = TRUE)
        }
        
        # Install fresh from temp copy
        options(devtools.install.args = c("--no-test-load", "--no-staged-install"))
        devtools::install(file.path(temp_dir, "BayesMVP"), 
                          args = c("--no-test-load", "--no-staged-install"),
                          dependencies = FALSE,  # Prevent namespace cycling
                          quiet = FALSE, 
                          force = TRUE)
        
        
        try({   setup_env_post_install()  }) # setup env again before loading 
        
 
  
}























