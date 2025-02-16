



#' #' update_cov_Welford
#' @keywords internal
#' @export
update_cov_Welford <- function( new_sample,
                                ii, 
                                mean_vec,
                                cov_mat) {
  
  ii <- ii + 1
  
  # Compute delta between new sample and old mean
  delta <- new_sample - mean_vec
  
  # # Update mean
  # mean_vec <- mean_vec + delta/ii
  
  # Compute delta between new sample and new mean
  delta2 <- new_sample - mean_vec
  
  # Update covariance
  cov_mat <- ((ii-1)/ii) * cov_mat + (1/ii) * (delta %*% t(delta2))
  
  return(list(ii=ii, 
              mean_vec=mean_vec,
              cov_mat=cov_mat))
  
}


