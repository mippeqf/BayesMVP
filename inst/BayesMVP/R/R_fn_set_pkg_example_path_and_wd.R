


#' set_pkg_example_path_and_wd
#' @export
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

