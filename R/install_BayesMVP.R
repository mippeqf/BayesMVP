


#' setup_env_pre_install
#' @export
setup_env_pre_install <- function() {


        ## Set brigestan and cmdstanr environment variables / directories
        #### bs_dir <- BayesMVP:::bridgestan_path()
        cmdstan_dir <- BayesMVP:::cmdstanr_path()

        if (.Platform$OS.type == "windows") {

                  TBB_STAN_DLL <- TBB_CMDSTAN_DLL <- DUMMY_MODEL_SO <- DUMMY_MODEL_DLL <- NULL

                  cat("Setting up BayesMVP Environment for Windows:\n")

                  cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                  try({   TBB_STAN_DLL <- file.path(system.file(package = "BayesMVP"), "BayesMVP", "inst", "tbb_stan", "tbb.dll") })
                  try({   TBB_CMDSTAN_DLL <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "tbb.dll") }) # prioritise user's installed tbb dll/so
                  try({   DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"),  "BayesMVP", "inst", "dummy_stan_modeL_win_model.so") })
                  try({   DUMMY_MODEL_DLL <- file.path(system.file(package = "BayesMVP"),  "BayesMVP", "inst", "dummy_stan_modeL_win_model.dll") })

                  dll_paths <- c(TBB_STAN_DLL,
                                 TBB_CMDSTAN_DLL,
                                 DUMMY_MODEL_SO,
                                 DUMMY_MODEL_DLL)

        }  else {  ### if Linux or Mac

                  TBB_STAN_SO <- TBB_CMDSTAN_SO <- DUMMY_MODEL_SO <- NULL

                  cat("Setting up BayesMVP Environment for Linux / Mac OS:\n")

                  cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                  try({  TBB_STAN_SO <- file.path(system.file(package = "BayesMVP"),  "BayesMVP", "inst", "tbb_stan", "libtbb.so.2") })
                  try({  TBB_CMDSTAN_SO <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "libtbb.so.2") })  # prioritise user's installed tbb dll/so
                  try({  DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"),  "BayesMVP", "inst", "dummy_stan_model_model.so") })

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





#' setup_env_post_install
#' @export
setup_env_post_install <- function() {


          # Set brigestan and cmdstanr environment variables / directories
          #### bs_dir <- bridgestan_path()
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
                                   TBB_CMDSTAN_DLL,
                                   DUMMY_MODEL_SO,
                                   DUMMY_MODEL_DLL)

          }  else {  ### if Linux or Mac

                    TBB_STAN_SO <- TBB_CMDSTAN_SO <- DUMMY_MODEL_SO <- NULL

                    cat("Setting up BayesMVP Environment for Linux / Mac OS:\n")

                    cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                    try({  TBB_STAN_SO <- file.path(system.file(package = "BayesMVP"), "tbb_stan", "libtbb.so.2") })
                    try({  TBB_CMDSTAN_SO <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "libtbb.so.2") })  # prioritise user's installed tbb dll/so
                    try({  DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"), "dummy_stan_model_model.so") })

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
  
          mvp_user_dir <- file.path(Sys.getenv("USERPROFILE"), ".BayesMVP")
          dir.create(mvp_user_dir, showWarnings = FALSE, recursive = TRUE)
          
          pkg_dir <- system.file(package = "BayesMVP", "BayesMVP", "inst")
          # pkg_dir <- file.path(libname, pkgname, "inst", "BayesMVP", "inst")    
          
          # Copy TBB
          tbb_file <- if (.Platform$OS.type == "windows") "tbb.dll" else "libtbb.so"
          file.copy(
            from =  file.path(pkg_dir, "tbb_stan", tbb_file),
            to = file.path(mvp_user_dir, tbb_file), 
            overwrite = TRUE
          )
          
          if (.Platform$OS.type == "windows") {
            
                  # Copy dummy model SO
                  file.copy(
                    from = file.path(pkg_dir, "dummy_stan_model_win_model.so"),
                    to = file.path(mvp_user_dir, "dummy_stan_model_win_model.so"),
                    overwrite = TRUE
                  )
                  
                  # Copy dummy model DLL
                  file.copy(
                    from = file.path(pkg_dir, "dummy_stan_model_win_model.dll"),
                    to = file.path(mvp_user_dir, "dummy_stan_model_win_model.dll"),
                    overwrite = TRUE
                  )
            
          } else { 
            
                  # Copy dummy model SO
                  file.copy(
                    from = file.path(pkg_dir, "dummy_stan_model_model.so"),
                    to = file.path(mvp_user_dir, "dummy_stan_model_model.so"),
                    overwrite = TRUE
                  )
                  
          }
  
  
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
        path <- file.path(temp_dir, "BayesMVP")
        ## Rcpp::compileAttributes(path)  
        devtools::install(path, 
                          args = c("--no-test-load", "--no-staged-install"),
                          dependencies = FALSE,  # Prevent namespace cycling
                          quiet = FALSE, 
                          force = TRUE
                         #   quick = TRUE
                          )
        
        
        try({   setup_env_post_install()  }) # setup env again before loading 
        
 
  
}























