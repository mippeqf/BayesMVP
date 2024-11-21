


#pragma once

// [[Rcpp::depends(StanHeaders)]] 
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppParallel)]] 
// [[Rcpp::depends(RcppEigen)]]
 

#include <Eigen/Dense>
 


// [[Rcpp::plugins(cpp17)]]



 
using namespace Eigen;

 

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
 

 
// adapt step size (eps) using ADAM
Eigen::Matrix<double, -1, 1>        adapt_eps_ADAM(double &eps,   //// updating this (pass by reference)
                                                   double &eps_m_adam,   //// updating this (pass by reference)
                                                   double &eps_v_adam,  //// updating this (pass by reference)
                                                   const double &iter, 
                                                   const double &n_burnin, 
                                                   const double &LR,  /// ADAM learning rate
                                                   const double &p_jump, 
                                                   const double &adapt_delta, 
                                                   const double &beta1_adam, 
                                                   const double &beta2_adam, 
                                                   const double &eps_adam) {
 
        
        // try {
       
          const double eps_noisy_grad_across_chains = p_jump - adapt_delta;
          
          // update moving avg's for ADAM
          eps_m_adam = beta1_adam * eps_m_adam + (1.0 - beta1_adam) * eps_noisy_grad_across_chains;  ///// update 
          eps_v_adam = beta2_adam * eps_v_adam + (1.0 - beta2_adam) * std::pow(eps_noisy_grad_across_chains, 2.0);  ///// update 
          
          // calc. bias-corrected estimates (local to this fn)
          // const double iter_dbl = (double) iter;
          // const double n_burnin_dbl = (double) n_burnin;
          const double eps_m_hat = eps_m_adam / (1.0 - std::pow(beta1_adam, iter));
          const double eps_v_hat = eps_v_adam / (1.0 - std::pow(beta2_adam, iter));
          
          const double current_alpha = LR * (  1.0 - (1.0 - LR)*iter/(n_burnin) );
          
          const double  log_h =   stan::math::log(eps) + current_alpha * eps_m_hat/(stan::math::sqrt(eps_v_hat) + eps_adam);
          eps = stan::math::exp(log_h);  ///// update  
          
          Eigen::Matrix<double, -1, 1>  out_vec(3);
          out_vec(0) = eps;
          out_vec(1) = eps_m_adam;
          out_vec(2) = eps_v_adam;
          
          return out_vec;
          
        // } catch (...) {
        //  // std::cerr << "Error in epsilon adaptation with ADAM." << std::endl; // bookmark - thread-safe??
        //   
        //   Eigen::Matrix<double, 3, 1>  out_vec(3);
        //   out_vec(0) = eps;
        //   out_vec(1) = eps_m_adam;
        //   out_vec(2) = eps_v_adam;
        //   
        //   return out_vec;
        //   
        // }
 
  
}











