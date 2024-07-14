R_fn_adapt_eps_ADAM <- function(eps.,
                                eps_m_adam.,
                                eps_v_adam.,
                                iter.,
                                n_burnin.,
                                learning_rate., 
                                p_jump.,
                                adapt_delta.,
                                beta1_adam.,
                                beta2_adam.,
                                eps_adam.,
                                L_manual.,
                                L_ii.
                                )  {


             override_Euclidean <- TRUE
  
     
                  if (iter. < n_burnin.) {
                    
                            try({
                              
                                  ### adapt step size (epsilon) using ADAM
                                  # using mean
                                  if (iter. < n_burnin.*0.3) {
                                       adapt_delta_val <-   adapt_delta. # 0.95  # else adapt_delta_val <- adapt_delta. 
                                   # adapt_delta_val <- adapt_delta.
                                  } else { 
                                    adapt_delta_val <-  adapt_delta.
                                  }
                                  eps_noisy_grad_across_chains <-   ( p_jump. - adapt_delta_val )
                                  
                                  eps_m_adam. <- beta1_adam.*eps_m_adam.+(1-beta1_adam.)*eps_noisy_grad_across_chains
                                  eps_v_adam. <- beta2_adam.*eps_v_adam.+(1-beta2_adam.)*eps_noisy_grad_across_chains^2
                                  eps_m_hat <- eps_m_adam./(1-beta1_adam.^iter.)
                                  eps_v_hat <- eps_v_adam./(1-beta2_adam.^iter.)
                                  
                                  current_alpha <- learning_rate. *(  1-(1 - learning_rate.)*iter./(n_burnin.) )
                                  log_h <-  log(eps.) + current_alpha*eps_m_hat/(sqrt(eps_v_hat)+eps_adam.)
                                  eps. <- exp(log_h)
                            
                            })
                    
                          return(list(eps = eps.,
                                      eps_m_adam = eps_m_adam.,
                                      eps_v_adam = eps_v_adam.))
                    
                  } else {   
                          
                  }
                  
             return(list(eps = eps.,
                         eps_m_adam = eps_m_adam.,
                         eps_v_adam = eps_v_adam.))
             
    
}







