
#pragma once

 
 


#include <random>

  

#include <Eigen/Dense>
 
 
 

#include <unsupported/Eigen/SpecialFunctions>
 
 
 
 
using namespace Eigen;

 
 
 
 
 
 
 
// HMC sampler functions   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



Eigen::Matrix<double, -1, 1> generate_random_std_norm_vec_Return(   int n_params, 
                                                                    std::mt19937  &rng) {
  
    Eigen::Matrix<double, -1, 1> std_norm_vec(n_params);
    
    std::normal_distribution<double> dist(0.0, 1.0); 
    for (int d = 0; d < n_params; d++) {
      std_norm_vec(d) = dist(rng);
    }
    
    return std_norm_vec;
  
}



void generate_random_std_norm_vec( Eigen::Matrix<double, -1, 1> &std_norm_vec,
                                                                  int n_params, 
                                                                  std::mt19937  &rng) {

  std::normal_distribution<double> dist(0.0, 1.0); 
  for (int d = 0; d < n_params; d++) {
    std_norm_vec(d) = dist(rng);
  }

}


 


// Another function using the same RNG
void generate_random_tau_ii(  double tau, 
                                                             double &tau_ii,  // ref because assigning
                                                             std::mt19937  &rng) {

  std::uniform_real_distribution<double> dist(0.0, 2.0 * tau);
  tau_ii = dist(rng); // assign random value

}







bool check_divergence_Eigen(                  const HMCResult &result_input,
                                              const Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                              const double hamiltonian_energy,
                                              const double previous_hamiltonian_energy) {
  
  /// first check is any NaN or Inf values 
  if ( (is_NaN_or_Inf_Eigen(lp_and_grad_outs) == true)  )  { 
 
        // for (int i = 0; i < lp_and_grad_outs.size(); ++i) {
        //   if (std::isnan(lp_and_grad_outs(i)) || std::isinf(lp_and_grad_outs(i))) {
        //     std::cout  << "NaN or Inf detected at indices: "  << "\n"   ;
        //     std::cout  << i << " "  << "\n"    << std::endl; 
        //   }
        // }
        
        return true;
    
  }
  
 

  
  // //  now check for large energy changes
  // {
  //     const double energy_diff = std::fabs(hamiltonian_energy - previous_hamiltonian_energy);
  //     const double energy_threshold = 100000000.0; // arbitrary
  //   
  //     if (energy_diff > energy_threshold) {
  //       return true;
  //     }
  // }

  return false; // If no issues detected
  
} 




double compute_kinetic_energy_diag(const Eigen::Matrix<double, -1, 1> &velocity, 
                                   const Eigen::Matrix<double, -1, 1> &M_vec) {
  
  double energy = 0.0;
  const int N = velocity.size();
  
  for (int i = 0; i < N; ++i) {
    double term = 0.5 * velocity(i) * velocity(i) * M_vec(i);
    // Use Kahan summation for better numerical stability
    double y = term - ((energy * 1e-10) * 1e10);
    double t = energy + y;
    energy = t;
  }
  
  return energy;
  
}
 
 
double compute_kinetic_energy_dense( const Eigen::Matrix<double, -1, 1>  &velocity, 
                                     const Eigen::Matrix<double, -1, -1> &M_dense) {
   
   // Use M * v first to minimize roundoff error accumulation
   Eigen::Matrix<double, -1, 1> Mv = M_dense * velocity;
   
   // Compute v'Mv using Kahan summation for better precision
   double sum = 0.0;
   double c = 0.0;  // Running compensation for lost low-order bits
   
   for (int i = 0; i < velocity.size(); ++i) {
     double y = velocity(i) * Mv(i) - c;
     double t = sum + y;
     c = (t - sum) - y;
     sum = t;
   }
   
   return 0.5 * sum;
   
}

 

bool proposal_div(    const double log_ratio, 
                      const double energy_old, 
                      const double energy_new) {
   
   if (!std::isfinite(log_ratio)) {
     return true;  // Reject non-finite proposals
   }
   
   if (std::abs(energy_new - energy_old) > 1e3) {
     return true;  // Reject suspiciously large energy differences
   }
   
   return false;
   
}


void check_numeric_stability(const Eigen::Matrix<double, -1, 1> &vec, 
                             const std::string &name) {
  
  if (vec.hasNaN()) {
    throw std::runtime_error("NaN detected in " + name);
  }
  if (!vec.allFinite()) {
    throw std::runtime_error("Inf detected in " + name);
  }
  
}








// ALWAYS_INLINE  void leapfrog_integrator_dense_M_standard_HMC_main_InPlace(    Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
//                                                                               Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
//                                                                               Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
//                                                                               const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
//                                                                               const Eigen::Matrix<double, -1, -1> &M_inv_dense_main,
//                                                                               const Eigen::Matrix<int, -1, -1> &y_ref,
//                                                                               const int L_ii,  
//                                                                               const double eps,
//                                                                               const std::string &Model_type,
//                                                                               const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
//                                                                               const std::string &grad_option,
//                                                                               const Model_fn_args_struct &Model_args_as_cpp_struct,
//                                                                               const Stan_model_struct &Stan_model_as_cpp_struct,
//                                                                               std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
//                                                                                                  const std::string &,
//                                                                                                  const bool, const bool, const bool,
//                                                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
//                                                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
//                                                                                                  const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
//                                                                                                  const std::string &,
//                                                                                                  const Model_fn_args_struct &,
//                                                                                                  const Stan_model_struct &)> fn_lp_grad_InPlace
//                                                                                 
// ) {
//   
//       const int N = Model_args_as_cpp_struct.N;
//       const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
//       const int n_params_main = Model_args_as_cpp_struct.n_params_main;
//       const int n_params = n_params_main + n_nuisance;
//       
//       Eigen::Matrix<double, -1, 1> grad_main =  lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);
//       
//       for (int l = 0; l < L_ii; l++) {
//         
//             // Update velocity (first half step)
//             Eigen::Matrix<double, -1, 1> temp_1 = M_inv_dense_main * grad_main;
//             velocity_main_vec_proposed_ref.array() +=    0.5 * eps * temp_1.array() ; 
//             
//             //// updae params by full step
//             theta_main_vec_proposed_ref.array()  +=  eps *     velocity_main_vec_proposed_ref.array() ;
//             
//             // Update lp and gradients
//             fn_lp_grad_InPlace(lp_and_grad_outs, 
//                                Model_type, force_autodiff, force_PartialLog, multi_attempts,
//                                theta_main_vec_proposed_ref, theta_us_vec_initial_ref, y_ref, grad_option,  
//                                Model_args_as_cpp_struct,  
//                                Stan_model_as_cpp_struct);
//             grad_main =   lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);
//             
//             // Update velocity (second half step)
//             Eigen::Matrix<double, -1, 1> temp_2 = M_inv_dense_main * grad_main;
//             velocity_main_vec_proposed_ref.array() +=    0.5 * eps * temp_2.array() ;
//         
//       } // End of leapfrog steps  
//   
// }
// 
// 
// 
//  
// ALWAYS_INLINE   void leapfrog_integrator_diag_M_standard_HMC_main_InPlace(       Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
//                                                                                                 Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
//                                                                                                 Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
//                                                                                                 const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
//                                                                                                 const Eigen::Matrix<double, -1, -1> &M_inv_main_vec,
//                                                                                                 const Eigen::Matrix<int, -1, -1> &y_ref,
//                                                                                                 const int L_ii,  
//                                                                                                 const double eps,
//                                                                                                 const std::string &Model_type,
//                                                                                                 const bool force_autodiff, const bool force_PartialLog,  const bool multi_attempts,
//                                                                                                 const std::string &grad_option,
//                                                                                                 const Model_fn_args_struct &Model_args_as_cpp_struct,
//                                                                                                 const Stan_model_struct &Stan_model_as_cpp_struct,
//                                                                                                 std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
//                                                                                                                    const std::string &,
//                                                                                                                    const bool, const bool, const bool,
//                                                                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
//                                                                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
//                                                                                                                    const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
//                                                                                                                    const std::string &,
//                                                                                                                    const Model_fn_args_struct &,
//                                                                                                                    const Stan_model_struct & )> fn_lp_grad_InPlace
//                                                                       
// ) {
//   
//       const int N = Model_args_as_cpp_struct.N;
//       const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
//       const int n_params_main = Model_args_as_cpp_struct.n_params_main;
//       const int n_params = n_params_main + n_nuisance;
//       
//       Eigen::Matrix<double, -1, 1> grad_main =  ( lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array()).matrix();
//       
//       for (int l = 0; l < L_ii; l++) {
//         
//             // Update velocity (first half step)
//             velocity_main_vec_proposed_ref.array() +=  ( 0.5 * eps * M_inv_main_vec.array() *  grad_main.array() ).array() ; 
//             
//             //// updae params by full step
//             theta_main_vec_proposed_ref.array()  +=  eps *     velocity_main_vec_proposed_ref.array() ;
//             
//             // Update lp and gradients
//             fn_lp_grad_InPlace(lp_and_grad_outs, 
//                                Model_type, force_autodiff, force_PartialLog, multi_attempts,
//                                theta_main_vec_proposed_ref, theta_us_vec_initial_ref, y_ref, grad_option, 
//                                Model_args_as_cpp_struct,  
//                                Stan_model_as_cpp_struct);
//             grad_main =  ( lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array()).matrix();
//             
//             // Update velocity (second half step)
//             velocity_main_vec_proposed_ref.array() +=  ( 0.5 * eps * M_inv_main_vec.array() *  grad_main.array() ).array() ;  
//         
//       } // End of leapfrog steps  
//   
// }
// 
// 
// 
// 
// 






void                                        fn_standard_HMC_main_only_single_iter_InPlace_process(   HMCResult &result_input,
                                                                                                     const bool  burnin, 
                                                                                                     std::mt19937  &rng,
                                                                                                     const int seed,
                                                                                                     const std::string &Model_type,
                                                                                                     const bool  force_autodiff,
                                                                                                     const bool  force_PartialLog,
                                                                                                     const bool  multi_attempts,
                                                                                                     const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                                                     const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                                     EHMC_fn_args_struct  &EHMC_args_as_cpp_struct, /// pass by ref. to modify (???)
                                                                                                     const EHMC_Metric_struct   &EHMC_Metric_struct_as_cpp_struct,
                                                                                                     const Stan_model_struct &Stan_model_as_cpp_struct
) {


    //// important params
    const int N = Model_args_as_cpp_struct.N;
    const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
    const int n_params_main = Model_args_as_cpp_struct.n_params_main;
    const int n_params = n_params_main + n_nuisance;
  
    const std::string grad_option = "all"; ///////////////////// DEBUG
     
    const std::string metric_shape_main = EHMC_Metric_struct_as_cpp_struct.metric_shape_main;
    
    double U_x_initial =  0.0 ; 
    double U_x_prop =  0.0 ; 
    double log_posterior_prop =  0.0 ; 
    double log_posterior_0  =   0.0 ;  
    double log_ratio = 0.0;
    double energy_old = 0.0;
    double energy_new = 0.0;
    
    result_input.main_theta_vec_0 =  result_input.main_theta_vec;
    result_input.us_theta_vec_0 =  result_input.us_theta_vec;

  {

      {
          Eigen::Matrix<double, -1, 1>  std_norm_vec_main(n_params_main);
          generate_random_std_norm_vec(std_norm_vec_main, n_params_main, rng);
          if (metric_shape_main == "dense") result_input.main_velocity_0_vec  = EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main_chol * std_norm_vec_main;
          if (metric_shape_main == "diag")  result_input.main_velocity_0_vec.array() = std_norm_vec_main.array() *  (EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec).array().sqrt() ; 
      }
      
   

    {
      
 
      { 
        
      try {
        
              result_input.main_velocity_vec_proposed  =   result_input.main_velocity_0_vec; // set initial velocity
              result_input.main_theta_vec_proposed =       result_input.main_theta_vec_0;   // set initial theta   
      
                ////---------------------------------------------------------------------------------------------------------------///    Perform L leapfrogs   ///-----------------------------------------------------------------------------------------------------------------------------------------
                generate_random_tau_ii(   EHMC_args_as_cpp_struct.tau_main,    EHMC_args_as_cpp_struct.tau_main_ii, rng);
                int    L_ii = std::ceil(  EHMC_args_as_cpp_struct.tau_main_ii / EHMC_args_as_cpp_struct.eps_main );
                if (L_ii < 1) { L_ii = 1 ; }
                
                //// initial lp  
                fn_lp_grad_InPlace(     result_input.lp_and_grad_outs, 
                                        Model_type, 
                                        force_autodiff, force_PartialLog, multi_attempts,
                                        result_input.main_theta_vec,  result_input.us_theta_vec, 
                                        y_ref, 
                                        grad_option,
                                        Model_args_as_cpp_struct,  
                                        Stan_model_as_cpp_struct);
                
                log_posterior_0 =  result_input.lp_and_grad_outs(0);
                U_x_initial = - log_posterior_0; //// initial energy
                
                // if (metric_shape_main == "dense") {
                // 
                //             // leapfrog_integrator_dense_M_standard_HMC_main_InPlace(    result_input.main_velocity_vec_proposed, 
                //             //                                                           result_input.main_theta_vec_proposed, 
                //             //                                                           result_input.lp_and_grad_outs, 
                //             //                                                           result_input.us_theta_vec,
                //             //                                                           EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main,
                //             //                                                           y_ref,
                //             //                                                           L_ii, EHMC_args_as_cpp_struct.eps_main,
                //             //                                                           Model_type, 
                //             //                                                           force_autodiff, force_PartialLog, multi_attempts, 
                //             //                                                           grad_option,
                //             //                                                           Model_args_as_cpp_struct, 
                //             //                                                           Stan_model_as_cpp_struct, 
                //             //                                                           fn_lp_grad_InPlace);
                //   
                // } else if (metric_shape_main == "diag") {
                //   
                //       leapfrog_integrator_diag_M_standard_HMC_main_InPlace(     result_input.main_velocity_vec_proposed, 
                //                                                                 result_input.main_theta_vec_proposed, 
                //                                                                 result_input.lp_and_grad_outs, 
                //                                                                 result_input.us_theta_vec,
                //                                                                 EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec,
                //                                                                 y_ref,
                //                                                                 L_ii, EHMC_args_as_cpp_struct.eps_main,
                //                                                                 Model_type, 
                //                                                                 force_autodiff, force_PartialLog, multi_attempts, 
                //                                                                 grad_option,
                //                                                                 Model_args_as_cpp_struct, 
                //                                                                 Stan_model_as_cpp_struct, 
                //                                                                 fn_lp_grad_InPlace);
                // }
                
                
                // Eigen::Matrix<double, -1, 1> grad_main =  result_input.lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);
                // 
                // for (int l = 0; l < L_ii; l++) {
                //   
                //       // Update velocity (first half step)
                //       Eigen::Matrix<double, -1, 1> temp_1 = EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main * grad_main;
                //       result_input.main_velocity_vec_proposed.array() +=    0.5 * EHMC_args_as_cpp_struct.eps_main * temp_1.array() ; 
                //       
                //       //// updae params by full step
                //       result_input.main_theta_vec_proposed.array()  +=  EHMC_args_as_cpp_struct.eps_main * result_input.main_velocity_vec_proposed.array() ;
                //       
                //       // Update lp and gradients
                //       fn_lp_grad_InPlace(  result_input.lp_and_grad_outs, 
                //                            Model_type, force_autodiff, force_PartialLog, multi_attempts,
                //                            result_input.main_theta_vec_proposed, result_input.us_theta_vec, y_ref, grad_option,  
                //                            Model_args_as_cpp_struct,  
                //                            Stan_model_as_cpp_struct);
                //       grad_main =   result_input.lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);
                //       
                //       // Update velocity (second half step)
                //       Eigen::Matrix<double, -1, 1> temp_2 = EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main * grad_main;
                //       result_input.main_velocity_vec_proposed.array() +=    0.5 * EHMC_args_as_cpp_struct.eps_main * temp_2.array() ;
                //   
                // } // End of leapfrog steps  
                
                Eigen::Matrix<double, -1, 1> grad_main =  result_input.lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);
                Eigen::Matrix<double, -1, 1> grad_scaled = EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main * grad_main;
                double step_size = EHMC_args_as_cpp_struct.eps_main;
                double half_step_size = 0.5 * EHMC_args_as_cpp_struct.eps_main;
 
                for (int l = 0; l < L_ii; l++) {
                      
                      // First half step for velocity
                      result_input.main_velocity_vec_proposed.array() += half_step_size * grad_scaled.array();
                      check_numeric_stability(result_input.main_velocity_vec_proposed, "velocity_half_step");
                      
                      // Full step for position
                      result_input.main_theta_vec_proposed.array() += step_size * result_input.main_velocity_vec_proposed.array();
                      check_numeric_stability(result_input.main_theta_vec_proposed, "position_step");
                      
                      // Update gradients
                      // Update lp and gradients
                      fn_lp_grad_InPlace(  result_input.lp_and_grad_outs, 
                                           Model_type, force_autodiff, force_PartialLog, multi_attempts,
                                           result_input.main_theta_vec_proposed, result_input.us_theta_vec, y_ref, grad_option,  
                                           Model_args_as_cpp_struct,  
                                           Stan_model_as_cpp_struct);
                      grad_main =   result_input.lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);
                      grad_scaled = EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main * grad_main;
                      check_numeric_stability(grad_main, "gradient_update");
                      
                      // Second half step for velocity 
                      result_input.main_velocity_vec_proposed.array() += half_step_size * grad_scaled.array();
                      check_numeric_stability(result_input.main_velocity_vec_proposed, "velocity_final");
                  
                }
                
                
                //// proposed lp  
                log_posterior_prop =  result_input.lp_and_grad_outs(0);
                U_x_prop = - log_posterior_prop; // initial energy
              
                //////////////////////////////////////////////////////////////////    M-H acceptance step  (i.e, Accept/Reject step)
                if (metric_shape_main == "dense") {
                    
                    energy_old = U_x_initial + compute_kinetic_energy_dense(result_input.main_velocity_0_vec, EHMC_Metric_struct_as_cpp_struct.M_dense_main);
                    energy_new = U_x_prop +    compute_kinetic_energy_dense(result_input.main_velocity_vec_proposed, EHMC_Metric_struct_as_cpp_struct.M_dense_main);
                    log_ratio = - energy_new + energy_old;
                    
                } else if (metric_shape_main == "diag") {
                  
                    energy_old = U_x_initial + compute_kinetic_energy_diag(result_input.main_velocity_0_vec, EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec);
                    energy_new = U_x_prop +    compute_kinetic_energy_diag(result_input.main_velocity_vec_proposed, EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec);
                    log_ratio = - energy_new + energy_old;
                  
                }
       
              if  ( (check_divergence_Eigen(  result_input,  
                                              result_input.lp_and_grad_outs, 
                                              energy_old, 
                                              energy_new) == true) || (proposal_div(log_ratio, energy_old, energy_new) == true) )      {     /// if main_div, reject proposal 
                  
                      result_input.main_p_jump = 0.0;
                      result_input.main_theta_vec  =   result_input.main_theta_vec_0;    
                      result_input.main_velocity_vec = result_input.main_velocity_0_vec;
                      result_input.main_div = 1;  // and set main_div indiator to 1
                
              }  else {  // if no main_div, carry on with MH step 
                
                          result_input.main_div = 0;
                          result_input.main_p_jump = std::min(1.0, stan::math::exp(log_ratio));
                          
                          std::uniform_real_distribution<double> unif(0.0, 1.0);
                          double acceptance_rand = unif(rng); 
                      
                      if   (acceptance_rand > result_input.main_p_jump)   {  // # reject proposal
                            result_input.main_theta_vec  =   result_input.main_theta_vec_0;  
                            result_input.main_velocity_vec = result_input.main_velocity_0_vec;
                      } else {   // # accept proposal
                            result_input.main_theta_vec  =    result_input.main_theta_vec_proposed ; 
                            result_input.main_velocity_vec =  result_input.main_velocity_vec_proposed ;
                      }
                  
                     // return result_input;
                      
              }

      } catch (...) {
        
                // std::cout << "  Could not evaluate lp_grad function when sampling main parameters " << ")\n";
          
                result_input.main_div = 1;
                result_input.main_p_jump = 0.0;
                result_input.main_theta_vec  =   result_input.main_theta_vec_0; 
                result_input.main_velocity_vec = result_input.main_velocity_0_vec;
                // return result_input;
              
      }
      


    }   
      
    }
    

  }
  
 // return result_input;


}








 
 
 
 
 
 
 
 
 
 
 