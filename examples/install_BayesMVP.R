# install_BayesMVP.R


require(cmdstanr)
require(bridgestan)

require(callr)




options(buildtools.check = function(action) TRUE ) 
options(warning.length =  8000)

  










# Sys.getenv("PWD")
# Sys.getenv("HOME")




## set_pkg_example_path_and_wd <- function(pkg_example_path = NULL) {
set_pkg_example_path_and_wd <- function() {
        
                ## Find user_root_dir:
                ## user_root_dir <- Sys.getenv("PWD")
                user_root_dir <- Sys.getenv("HOME")
                print(paste("user_root_dir = ", user_root_dir))
                
                user_BayesMVP_dir <- file.path(user_root_dir, "BayesMVP")
                print(paste("user_BayesMVP_dir = ", user_BayesMVP_dir))
        
                pkg_example_path <- file.path(user_BayesMVP_dir, "examples")
                print(paste("pkg_example_path = ", pkg_example_path))
                
                ## Set working directory:
                setwd(pkg_example_path)
                message(paste("Workind directory set to: ", pkg_example_path))
                
                # ## Find user_pkg_install_dir:
                # user_pkg_install_dir <- Sys.getenv("R_LIBS_USER")
                # print(paste("user_pkg_install_dir = ", user_pkg_install_dir))
                # 
                # ## Find pkg_install_path:
                # pkg_install_path <- file.path(user_pkg_install_dir, "BayesMVP")
                # print(paste("pkg_install_path = ", pkg_install_path))
                
             outs <- list(user_root_dir = user_root_dir,
                          user_BayesMVP_dir = user_BayesMVP_dir,
                          pkg_example_path = pkg_example_path)
             
             return(outs)
  
}



  





# Function to run before restart:
before_restart <- function(pkg_dir) {
  
  remove.packages("BayesMVP")
  pkg_install_path <- "/home/enzocerullo/R/x86_64-pc-linux-gnu-library/4.4/BayesMVP"
  pkg_temp_install_path <- "/home/enzocerullo/R/x86_64-pc-linux-gnu-library/4.4/00LOCK-BayesMVP"
  unlink(pkg_install_path, recursive = TRUE, force = TRUE)
  unlink(pkg_temp_install_path, recursive = TRUE, force = TRUE)
  
  devtools::clean_dll(pkg_dir)  
  Rcpp::compileAttributes(pkg_dir)  
  devtools::document(pkg_dir)  
  
  devtools::clean_dll(pkg_dir)
  Rcpp::compileAttributes(pkg_dir)
  devtools::install(pkg_dir, upgrade = "never")
  
}

# Function to run after restart:
after_restart <- function(local = FALSE) {
  require(BayesMVP)
  BayesMVP::install_BayesMVP()
  require(BayesMVP)
}

# Main execution
library(callr)






### ---------------------------
try({ 
  ## Run first part in external process (using callr)
  callr::r(func = before_restart, 
           args = list(pkg_dir = pkg_dir))
  })


# # Restart R in a new process and run second part with error handling (using callr)
# result <- try(r(after_restart, error = TRUE))
# if (inherits(result, "try-error")) {
#   cat("\nError in new R session:\n")
#   cat(attr(result, "condition")$message, "\n")
# }


## Install (inner pkg):
require(BayesMVP)
BayesMVP::install_BayesMVP()
require(BayesMVP)
