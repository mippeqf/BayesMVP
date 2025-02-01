

##
rm(list = ls())
##
rstudioapi::restartSession()
##
rm(list = ls())
##

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
## Document INNER package:
devtools::clean_dll(local_INNER_pkg_dir)
Rcpp::compileAttributes(local_INNER_pkg_dir)
devtools::document(local_INNER_pkg_dir)


#### -------- ACTUAL (LOCAL) INSTALL: 
## Document:
devtools::clean_dll(local_pkg_dir)
devtools::document(local_pkg_dir)
## Install (outer pkg):
devtools::clean_dll(local_pkg_dir)
devtools::install(local_pkg_dir, upgrade = "never")
## * MIGHT * have to restart session:
rstudioapi::restartSession()
##
## Install (inner pkg):
require(BayesMVP)
BayesMVP::install_BayesMVP()
require(BayesMVP)





