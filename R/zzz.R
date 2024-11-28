 

 

#' setup_env_onload
#' @export
setup_env_onload <- function(libname, 
                             pkgname) {
  
            try({ 
              options(devtools.install.args = c("--no-test-load"))
            })
            
            ## Set brigestan and cmdstanr environment variables / directories 
            bs_dir <- bridgestan_path()
            cmdstan_dir <- BayesMVP:::cmdstanr_path()
  
            if (.Platform$OS.type == "windows") {
              
                          ### TBB_BS <- file.path(system.file(package = "BayesMVP"), "tbb_bs", "tbb.dll") # not needed?
                          TBB_STAN_1 <- file.path(libname, pkgname, "inst", "tbb_stan", "tbb.dll")
                          TBB_STAN_2 <- file.path(libname, pkgname, "tbb_stan", "tbb.dll")
                          TBB_CMDSTAN_DLL <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "tbb.dll")  # prioritise user's installed tbb dll/so
                          DUMMY_MODEL_SO_1 <- file.path(libname, pkgname, "inst", "dummy_stan_modeL_win_model.so")
                          DUMMY_MODEL_SO_2 <- file.path(libname, pkgname, "dummy_stan_modeL_win_model.so")
                          DUMMY_MODEL_DLL_1 <- file.path(libname, pkgname, "inst", "dummy_stan_modeL_win_model.dll")
                          DUMMY_MODEL_DLL_2 <- file.path(libname, pkgname, "dummy_stan_modeL_win_model.dll")
                          
                          dll_paths <- c(TBB_STAN_1, TBB_STAN_2, TBB_CMDSTAN_DLL,
                                         DUMMY_MODEL_SO_1, DUMMY_MODEL_SO_2,
                                         DUMMY_MODEL_DLL_1, DUMMY_MODEL_DLL_2)
                    
              
            } else {  ### if Linux or Mac
                          
                          TBB_STAN_1 <- file.path(libname, pkgname, "inst", "tbb_stan", "libtbb.so.2")
                          TBB_STAN_2 <- file.path(libname, pkgname, "tbb_stan", "libtbb.so.2")
                          TBB_CMDSTAN_SO <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "libtbb.so.2")  # prioritise user's installed tbb dll/so
                          DUMMY_MODEL_SO_1 <- file.path(libname, pkgname, "inst", "dummy_stan_modeL_win_model.so")
                          DUMMY_MODEL_SO_2 <- file.path(libname, pkgname, "dummy_stan_modeL_win_model.so")
                          
                          dll_paths <- c(TBB_STAN_1, TBB_STAN_2, TBB_CMDSTAN_SO,
                                         DUMMY_MODEL_SO_1, DUMMY_MODEL_SO_2)
                        
              
            }
            
            
                      # Attempt to load each DLL / SO
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
  

  return(BayesMVP:::setup_env_onload(libname, pkgname))
  
}



#' .onAttach
#' @export
.onAttach <- function(libname, 
                      pkgname) {
  
  
  return(BayesMVP:::setup_env_onload(libname, pkgname))
  
}


#' .First.lib
#' @export
.First.lib <- function(libname, 
                       pkgname) {
  
  
  return(BayesMVP:::setup_env_onload(libname, pkgname))
  
}

 

 



 





 
