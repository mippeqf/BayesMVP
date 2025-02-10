

# ##
# rm(list = ls())
# ##
rstudioapi::restartSession()
# ##
## rm(list = ls())


##

{
    ## First remove any possible package fragments:
    ## Find user_pkg_install_dir:
    user_pkg_install_dir <- Sys.getenv("R_LIBS_USER")
    print(paste("user_pkg_install_dir = ", user_pkg_install_dir))
    ##
    ## Find pkg_install_path + pkg_temp_install_path:
    pkg_install_path <- file.path(user_pkg_install_dir, "BayesMVP")
    pkg_temp_install_path <- file.path(user_pkg_install_dir, "00LOCK-BayesMVP") 
    ##
    ## Remove any (possible) BayesMVP package fragments:
    remove.packages("BayesMVP")
    unlink(pkg_install_path, recursive = TRUE, force = TRUE)
    unlink(pkg_temp_install_path, recursive = TRUE, force = TRUE)
}





if (.Platform$OS.type == "windows") {
  local_pkg_dir <-  "C:\\Users\\enzoc\\Documents\\Work\\PhD_work\\R_packages\\BayesMVP"
  local_INNER_pkg_dir <-  "C:\\Users\\enzoc\\Documents\\Work\\PhD_work\\R_packages\\BayesMVP\\inst\\BayesMVP"
} else {
  if (parallel::detectCores() > 16) {   ## if on local HPC
      local_pkg_dir <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP"  
      local_INNER_pkg_dir <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP"
  } else {  ## if on laptop
      local_pkg_dir <- "/home/enzo/Documents/Work/PhD_work/R_packages/BayesMVP"  
      local_INNER_pkg_dir <- "/home/enzo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP"
  }
}



# #### -------- INNER pkg stuff:
# ## Only works if do this first (on Linux)!! :
# source("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP/src/R_script_load_OMP_Linux.R")
# ## Document INNER package:
# devtools::clean_dll(local_INNER_pkg_dir)
# Rcpp::compileAttributes(local_INNER_pkg_dir)
# ### devtools::document(local_INNER_pkg_dir)
# roxygen2::roxygenize(local_INNER_pkg_dir)


#### -------- ACTUAL (LOCAL) INSTALL: 
## Document:
devtools::clean_dll(local_pkg_dir)
#### devtools::document(local_pkg_dir)
roxygen2::roxygenize(local_pkg_dir)
##
## Install (outer pkg):
devtools::clean_dll(local_pkg_dir)
devtools::install(local_pkg_dir, upgrade = "never")
##
## * MIGHT * have to restart session:
#### rstudioapi::restartSession()
##
{
    install_success <- FALSE
    try({  
      ## Only works if do this first (on Linux)!! :
      source("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP/src/R_script_load_OMP_Linux.R")
      ##
      ## Install (inner pkg):
      require(BayesMVP)
      BayesMVP::install_BayesMVP() ## CUSTOM_FLAGS = CUSTOM_FLAGS)
      require(BayesMVP)
      install_success <- TRUE
    })
    
    rstudioapi::restartSession()
}




# CUSTOM_FLAGS <- list(  "CCACHE_PATH = ccache",
#                        "CXX_COMPILER = g++",
#                        "CPP_COMPILER = gcc", 
#                        "CXX_STD = CXX17",
#                        "CPU_BASE_FLAGS = -O3 -march=native -mtune=native",
#                        "FMA_FLAGS = -mfma",
#                        "AVX_FLAGS = -mavx -mavx2 -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl")
#                        # "OMP_FLAGS",
#                        # "OMP_LIB_PATH",
#                        # "OMP_LIB_FLAGS",
#                        # "PKG_CPPFLAGS",
#                        # "PKG_CXXFLAGS",
#                        # "CPPFLAGS",
#                        # "CXXFLAGS",
#                        # "PKG_LIBS")
# 
# 
