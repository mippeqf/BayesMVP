 


#' .make_user_dir
#' @export
.make_user_dir <- function(libname, 
                           pkgname) {
          
  
              user_dir <- path.expand("~/BayesMVP")
              if (!dir.exists(user_dir)) {
                dir.create(user_dir)
              }
              cat("user_dir =", user_dir, "\n")
              
              ## -------------- USER Examples dir:
              user_examples_dir <- file.path(user_dir, "examples")
              cat("user_examples_dir =", user_examples_dir, "\n")
              
              system_examples_dir <- system.file("examples", package = "BayesMVP")
              cat("system_examples_dir =", system_examples_dir, "\n")
              
              if (!dir.exists(user_examples_dir)) {
                dir.create(user_examples_dir)
                ## Copy the entire examples directory:
                file.copy(
                  from = list.files(system_examples_dir, full.names = TRUE),
                  to = user_examples_dir,
                  recursive = TRUE
                )
              }
              
              ## -------------- USER src dir:
              user_src_dir <- file.path(user_dir, "src")
              cat("user_src_dir =", user_src_dir, "\n")
              
              system_src_dir <- system.file("BayesMVP/src", package = "BayesMVP")
              cat("system_src_dir =", system_src_dir, "\n")
    
              if (!dir.exists(user_src_dir)) {
                dir.create(user_src_dir)
                ## Copy the entire src directory:
                file.copy(
                  from = list.files(system_src_dir, full.names = TRUE),
                  to = user_src_dir,
                  recursive = TRUE
                )
              }
              
              ## -------------- USER stan_models dir:
              user_stan_models_dir <- file.path(user_dir, "stan_models")
              cat("user_stan_models_dir =", user_stan_models_dir, "\n")
              
              system_stan_models_dir <- system.file("BayesMVP/inst/stan_models/", package = "BayesMVP")
              cat("system_stan_models_dir =", system_stan_models_dir, "\n")
              
              if (!dir.exists(user_stan_models_dir)) {
                dir.create(user_stan_models_dir)
                ## Copy the entire src directory:
                file.copy(
                  from = list.files(system_stan_models_dir, full.names = TRUE),
                  to = user_stan_models_dir,
                  recursive = TRUE
                )
              }
          
              
              # ## -------------- USER TBB files:
              # user_stan_models_dir <- file.path(user_dir, "stan_models")
              # cat("user_stan_models_dir =", user_stan_models_dir, "\n")
              # 
              # system_stan_models_dir <- system.file("BayesMVP/inst/stan_models/", package = "BayesMVP")
              # cat("system_stan_models_dir =", system_stan_models_dir, "\n")
              # 
              # if (!dir.exists(user_stan_models_dir)) {
              #   dir.create(user_stan_models_dir)
              #   ## Copy the entire src directory:
              #   file.copy(
              #     from = list.files(system_stan_models_dir, full.names = TRUE),
              #     to = user_stan_models_dir,
              #     recursive = TRUE
              #   )
              # }
              # 
              # 
              # mvp_user_dir <- file.path(Sys.getenv("USERPROFILE"), ".BayesMVP")
              # dir.create(mvp_user_dir, showWarnings = FALSE, recursive = TRUE)
              # 
              # ## Copy TBB:
              # tbb_file <- if (.Platform$OS.type == "windows") "tbb.dll" else "libtbb.so"
              # file.copy(
              #   from =  file.path(pkg_dir, "tbb_stan", tbb_file),
              #   to = file.path(mvp_user_dir, tbb_file), 
              #   overwrite = TRUE
              # )
              # 
              # ## Copy dummy model files (for bridgestan):
              # if (.Platform$OS.type == "windows") {
              #   
              #   # Copy dummy model SO
              #   file.copy(
              #     from = file.path(pkg_dir, "dummy_stan_model_win_model.so"),
              #     to = file.path(mvp_user_dir, "dummy_stan_model_win_model.so"),
              #     overwrite = TRUE
              #   )
              #   
              #   # Copy dummy model DLL
              #   file.copy(
              #     from = file.path(pkg_dir, "dummy_stan_model_win_model.dll"),
              #     to = file.path(mvp_user_dir, "dummy_stan_model_win_model.dll"),
              #     overwrite = TRUE
              #   )
              #   
              # } else { 
              #   
              #   # Copy dummy model SO
              #   file.copy(
              #     from = file.path(pkg_dir, "dummy_stan_model_model.so"),
              #     to = file.path(mvp_user_dir, "dummy_stan_model_model.so"),
              #     overwrite = TRUE
              #   )
              #   
              # }
              
              
              # ## -------------- USER DUMMY MODEL files:
              # mvp_user_dir <- file.path(Sys.getenv("USERPROFILE"), ".BayesMVP")
              # dir.create(mvp_user_dir, showWarnings = FALSE, recursive = TRUE)
              # 
              # ## Copy TBB:
              # tbb_file <- if (.Platform$OS.type == "windows") "tbb.dll" else "libtbb.so"
              # file.copy(
              #   from =  file.path(pkg_dir, "tbb_stan", tbb_file),
              #   to = file.path(mvp_user_dir, tbb_file), 
              #   overwrite = TRUE
              # )
              # 
              # ## Copy dummy model files (for bridgestan):
              # if (.Platform$OS.type == "windows") {
              #   
              #   # Copy dummy model SO
              #   file.copy(
              #     from = file.path(pkg_dir, "dummy_stan_model_win_model.so"),
              #     to = file.path(mvp_user_dir, "dummy_stan_model_win_model.so"),
              #     overwrite = TRUE
              #   )
              #   
              #   # Copy dummy model DLL
              #   file.copy(
              #     from = file.path(pkg_dir, "dummy_stan_model_win_model.dll"),
              #     to = file.path(mvp_user_dir, "dummy_stan_model_win_model.dll"),
              #     overwrite = TRUE
              #   )
              #   
              # } else { 
              #   
              #   # Copy dummy model SO
              #   file.copy(
              #     from = file.path(pkg_dir, "dummy_stan_model_model.so"),
              #     to = file.path(mvp_user_dir, "dummy_stan_model_model.so"),
              #     overwrite = TRUE
              #   )
              #   
              # }
              
              
}
