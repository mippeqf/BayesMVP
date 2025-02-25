


omp_path_1 <- " " ## /usr/lib/llvm-14/lib"
omp_path_2 <- " " ##"/usr/lib/x86_64-linux-gnu/libomp.so.5"
omp_path_3 <- " " ##"/usr/lib/llvm-14/lib/libomp.so.5"
omp_path_4 <- " " ##"/opt/AMD/aocc-compiler-5.0.0/lib/libomp.so"
omp_path_5 <- " " ##"/opt/AMD/aocc-compiler-4.2.2/lib/libomp.so"
omp_path_6 <- "/usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so"
##
omp_path_vec <- c(omp_path_1, omp_path_2, 
                  omp_path_3, omp_path_4,
                  omp_path_5, omp_path_6)

counter <- 0
for (i in 1:length(omp_path_vec)) {
      if (dir.exists(omp_path_vec[i])) {
          counter <- counter + 1
          if (counter == 1) {
            current_ld_path <- Sys.getenv("LD_LIBRARY_PATH")
            ## new_ld_path <- omp_path_vec[i]
            new_ld_path <- if (current_ld_path == "") omp_path_vec[i] else paste0(omp_path_vec[i], ":", current_ld_path)
            Sys.setenv(LD_LIBRARY_PATH = new_ld_path)
          }
      }
}



writeLines(as.character(Sys.getenv("LD_LIBRARY_PATH")))

