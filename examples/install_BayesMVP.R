# install_BayesMVP.R

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
after_restart <- function() {
  require(BayesMVP)
  BayesMVP::install_BayesMVP()
  require(BayesMVP)
}

# Main execution
library(callr)

