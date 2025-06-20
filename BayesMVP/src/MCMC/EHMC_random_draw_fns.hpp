
#pragma once

 


#include <random>

  

#include <Eigen/Dense>
 
 
 

#include <unsupported/Eigen/SpecialFunctions>
 
 
 
 
using namespace Eigen;
 
 
 
 
 
 


 

 

 
// HMC sampler functions   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 
 


template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE   Eigen::Matrix<double, -1, 1>  generate_random_std_norm_vec( int n_params, 
                                                                            T &rng) {
   
         //// Initialise at zero:
         Eigen::Matrix<double, -1, 1> std_norm_vec = Eigen::Matrix<double, -1, 1>::Zero(n_params);
         
         std::normal_distribution<double> dist(0.0, 1.0); 
         
         //// Fill vector:
         for (int d = 0; d < n_params; d++) {
             double norm_draw = dist(rng);
             std_norm_vec(d) = norm_draw;
         }
         
         return std_norm_vec;
   
}
 
 
 
 
 
 
template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE    void generate_random_std_norm_vec_InPlace( Eigen::Ref<Eigen::Matrix<double, -1, 1>> std_norm_vec,
                                                            T &rng) {
  
         const int n_params = std_norm_vec.size();
    
         std::normal_distribution<double> dist(0.0, 1.0); 
         
         //// Initialise at zero:
         std_norm_vec.setZero();
         
         //// Fill vector:
         for (int d = 0; d < n_params; d++) {
             double norm_draw = dist(rng);
             std_norm_vec(d) = norm_draw; 
         }
   
}
 
 
 
  
template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE  double generate_random_std_uniform(T &rng) {
  
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const double rand_std_unif = dist(rng);
        return rand_std_unif;
  
}


 
template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE  double generate_random_tau_ii(  double tau, 
                                               T &rng) {
  
        std::uniform_real_distribution<double> dist(0.0, 2.0 * tau);
        const double tau_ii = dist(rng);
        return tau_ii;

}





 
 
 

 
 
 
 
 
 
 
 
 
 

 
 