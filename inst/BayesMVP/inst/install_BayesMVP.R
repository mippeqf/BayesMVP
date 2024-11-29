


setup_env_post_install <- function() {
  
  
  # try({ 
  #   options(devtools.install.args = c("--no-test-load"))
  # })
  
  ## Set brigestan and cmdstanr environment variables / directories 
  ##  bs_dir <- bridgestan_path()
  # cmdstan_dir <- cmdstanr_path()
  
  ## temp_dir <- Sys.getenv("TEMP")
  # 
  # if (.Platform$OS.type == "windows") {
  #   
  #          # cat("Setting up BayesMVP Environment for Windows:\n")
  #   
  #         #  cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
  #           TBB_STAN_DLL <- file.path(system.file(package = "BayesMVP"), "tbb_stan", "tbb.dll")
  #          # TBB_CMDSTAN_DLL <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "tbb.dll")  # prioritise user's installed tbb dll/so
  #           DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"), "dummy_stan_modeL_win_model.so")
  #           DUMMY_MODEL_DLL <- file.path(system.file(package = "BayesMVP"), "dummy_stan_modeL_win_model.dll")
  #           
  #           dll_paths <- c(TBB_STAN_DLL, 
  #                         # TBB_CMDSTAN_DLL,
  #                          DUMMY_MODEL_SO, 
  #                          DUMMY_MODEL_DLL)
  #           
  # }  else {  ### if Linux or Mac
  #   
  #         #  cat("Setting up BayesMVP Environment for Linux / Mac OS:\n")
  #           
  #         #  cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
  #           TBB_STAN_SO <- file.path(system.file(package = "BayesMVP"), "tbb_stan", "libtbb.so.2")
  #        #   TBB_CMDSTAN_SO <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "libtbb.so.2")  # prioritise user's installed tbb dll/so
  #           DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"), "dummy_stan_modeL_win_model.so")
  #           
  #           dll_paths <- c(TBB_STAN_SO, 
  #                         # TBB_CMDSTAN_SO,
  #                          DUMMY_MODEL_SO)
  #   
  # }
  # 
  #           
  #           # Attempt to load each DLL
  #           for (dll in dll_paths) {
  #             
  #             tryCatch(
  #               {
  #                 dyn.load(dll)
  #                # cat("  Loaded:", dll, "\n")
  #               },
  #               error = function(e) {
  #                # cat("  Failed to load:", dll, "\n  Error:", e$message, "\n")
  #               }
  #             )
  #             
  #           }
  
}

setup_env_post_install()


