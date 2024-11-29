

 
find_num_chunks_MVP  <- function(N, 
                                 n_tests) {


   n_obs <- N * n_tests
        
          if (parallel::detectCores() < 17) { # laptop / consumer-level computer's
            
                if (n_obs <= 2500)            num_chunks <-  1
                if (n_obs %in% c(2501,  5000))    num_chunks <-   2
                if (n_obs %in% c(5001,  12500))   num_chunks <-   2
                if (n_obs %in% c(12501, 25000))   num_chunks <-   10
                if (n_obs %in% c(25001, 62500))   num_chunks <-   20 # 25
                if (n_obs > 62500)   num_chunks <-   40
                if (n_obs > 125000)  num_chunks <-  125 # untested
                if (n_obs > 250000)  num_chunks <-  250 # untested
                if (n_obs > 500000)  num_chunks <-  500 # untested
                
          } else { # HPC's / server's 
            
                if (n_obs <= 2500)            num_chunks <-  1
                if (n_obs %in% c(2501,  5000))    num_chunks <-   1
                if (n_obs %in% c(5001,  12500))   num_chunks <-   2
                if (n_obs %in% c(12501, 25000))   num_chunks <-   10
                if (n_obs %in% c(25001, 62500))   num_chunks <-   20  
                if (n_obs > 62500)   num_chunks <-   40
                if (n_obs > 125000)  num_chunks <-  125 # untested
                if (n_obs > 250000)  num_chunks <-  250 # untested
                if (n_obs > 500000)  num_chunks <-  500 # untested
                
          }
   
   
   return(num_chunks)
      
  
}






