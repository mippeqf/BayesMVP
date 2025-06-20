
#pragma once


 
 
#include <random>

 

#include <Eigen/Dense>
 
 

#include <unsupported/Eigen/SpecialFunctions>
 
 
 
 
 
using namespace Eigen;
 
 

  
  
 
 
 
// Diffusion-HMC sampler functions   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





//  
// ALWAYS_INLINE   void leapfrog_integrator_diag_M_standard_HMC_nuisance_InPlace(  Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
//                                                                                 Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
//                                                                                 Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
//                                                                                 const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
//                                                                                 const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
//                                                                                 const Eigen::Matrix<int, -1, -1> &y_ref,
//                                                                                 const int L_ii, 
//                                                                                 const double eps,
//                                                                                 const std::string &Model_type,
//                                                                                 const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts, 
//                                                                                 const std::string &grad_option,
//                                                                                 const Model_fn_args_struct &Model_args_as_cpp_struct,
//                                                                                 const Stan_model_struct &Stan_model_as_cpp_struct,
//                                                                                 std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
//                                                                                                    const std::string &,
//                                                                                                    const bool, const bool, const bool,
//                                                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
//                                                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
//                                                                                                    const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
//                                                                                                    const std::string &,
//                                                                                                    const Model_fn_args_struct &,
//                                                                                                    const Stan_model_struct &)> fn_lp_grad_InPlace
//                                                               
// ) {
//   
//       const int n_nuisance = velocity_us_vec_proposed_ref.size();
//       Eigen::Matrix<double, -1, 1> grad_us =  lp_and_grad_outs.segment(1, n_nuisance);
//       
//       for (int l = 0; l < L_ii; l++) {
//         
//               // Update velocity (first half step)
//               Eigen::Matrix<double, -1, 1> temp_1 = grad_us.array()  * M_inv_us_vec.array();
//               velocity_us_vec_proposed_ref.array() += 0.5 * eps * temp_1.array();
//               
//               //// updae params by full step
//               theta_us_vec_proposed_ref.array()  +=  eps *     velocity_us_vec_proposed_ref.array() ;
//               
//               // Update lp and gradients
//               fn_lp_grad_InPlace(lp_and_grad_outs, Model_type, force_autodiff, force_PartialLog, multi_attempts,
//                                  theta_main_vec_initial_ref, theta_us_vec_proposed_ref, y_ref, grad_option, 
//                                  Model_args_as_cpp_struct,  
//                                  Stan_model_as_cpp_struct);
//               grad_us =  lp_and_grad_outs.segment(1, n_nuisance);
//               
//               // Update velocity (second half step)
//               Eigen::Matrix<double, -1, 1> temp_2 = grad_us.array()  * M_inv_us_vec.array();
//               velocity_us_vec_proposed_ref.array() += 0.5 * eps * temp_2.array();
//         
//       } // End of leapfrog steps
//   
// }
// 
// 
// 
// 
// 
// 
// 
// 
// 
//  
// ALWAYS_INLINE    void leapfrog_integrator_diag_M_diffusion_HMC_nuisance_InPlace(  Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
//                                                                         Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
//                                                                         Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
//                                                                         Eigen::Matrix<double, -1, 1> &theta_us_vec_current_segment,
//                                                                         Eigen::Matrix<double, -1, 1> &velocity_us_vec_current_segment,
//                                                                         const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
//                                                                         const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
//                                                                         const Eigen::Matrix<int, -1, -1> &y_ref,
//                                                                         const int L_ii,  
//                                                                         const double eps_1, const double cos_eps_2, const double sin_eps_2,
//                                                                         const std::string &Model_type, 
//                                                                         const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts, 
//                                                                         const std::string &grad_option,
//                                                                         const Model_fn_args_struct &Model_args_as_cpp_struct,
//                                                                         const Stan_model_struct &Stan_model_as_cpp_struct,
//                                                                         std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
//                                                                                            const std::string &,
//                                                                                            const bool, const bool, const bool,
//                                                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
//                                                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
//                                                                                            const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
//                                                                                            const std::string &,
//                                                                                            const Model_fn_args_struct &,
//                                                                                            const Stan_model_struct &)> fn_lp_grad_InPlace
//                                                                         
// ) {
//   
//   const int n_nuisance = velocity_us_vec_proposed_ref.size();
//   Eigen::Matrix<double, -1, 1> grad_us =  lp_and_grad_outs.segment(1, n_nuisance);
//   
//   for (int l = 0; l < L_ii; l++) {
//     
//           // Update velocity (first half step)
//           Eigen::Matrix<double, -1, 1> temp_1 = grad_us.array()  * M_inv_us_vec.array();
//           velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * temp_1.array();
//           
//           // Full step for position + update velocity again (only for Gaussian)
//           theta_us_vec_current_segment.array() = theta_us_vec_proposed_ref.array();
//           velocity_us_vec_current_segment.array() = velocity_us_vec_proposed_ref.array();
//           theta_us_vec_proposed_ref.array() =    (cos_eps_2 * theta_us_vec_current_segment.array()     + sin_eps_2 * velocity_us_vec_current_segment.array());
//           velocity_us_vec_proposed_ref.array() = (cos_eps_2 * velocity_us_vec_current_segment.array()  - sin_eps_2 * theta_us_vec_current_segment.array());
//           
//           // Update lp and gradients
//           fn_lp_grad_InPlace(lp_and_grad_outs, Model_type, force_autodiff, force_PartialLog, multi_attempts,
//                              theta_main_vec_initial_ref, theta_us_vec_proposed_ref, y_ref, grad_option, 
//                              Model_args_as_cpp_struct,  
//                              Stan_model_as_cpp_struct);
//           
//           // Update velocity (second half step)
//           Eigen::Matrix<double, -1, 1> temp_2 = grad_us.array()  * M_inv_us_vec.array();
//           velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * temp_2.array();
//     
//   } // End of leapfrog steps
//   
// }
// 
// 
// 
// 









 
 ALWAYS_INLINE  void         fn_Diffusion_HMC_nuisance_only_single_iter_InPlace_process(              HMCResult &result_input,
                                                                                               const bool burnin,
                                                                                               std::mt19937  &rng,
                                                                                               const int seed,
                                                                                               const std::string &Model_type,
                                                                                               const bool  force_autodiff,
                                                                                               const bool  force_PartialLog,
                                                                                               const bool  multi_attempts, 
                                                                                               const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                                               const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                               //MVP_ThreadLocalWorkspace &MVP_workspace,
                                                                                               EHMC_fn_args_struct  &EHMC_args_as_cpp_struct,
                                                                                               const EHMC_Metric_struct  &EHMC_Metric_struct_as_cpp_struct,
                                                                                               const Stan_model_struct &Stan_model_as_cpp_struct
) {

      //// important params
      const int N = Model_args_as_cpp_struct.N;
      const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
      const int n_params_main = Model_args_as_cpp_struct.n_params_main;
      const int n_params = n_params_main +  n_nuisance;
  
      const std::string grad_option = "all";
      
      // const double h = EHMC_args_as_cpp_struct.eps_us; /// HMC-equiv step-size ??
      const double eps_1 = EHMC_args_as_cpp_struct.eps_us; ; // std::sqrt(h);
      const double eps_1_sq = eps_1 * eps_1; /// h_sq is equiv. to MALA step-size  ??
      const double cos_eps_2 = (1.0 - 0.25 * eps_1_sq) / (1.0 + 0.25 * eps_1_sq);
      const double sin_eps_2 = std::sqrt(1.0 - (cos_eps_2 * cos_eps_2) ); 
  
      double U_x_initial =  0.0; 
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
       Eigen::Matrix<double, -1, 1> std_norm_vec_us(n_nuisance); // testing if static thread_local makes more efficient
       generate_random_std_norm_vec(std_norm_vec_us, n_nuisance, rng);
       result_input.us_velocity_0_vec.array() = ( std_norm_vec_us.array() *  (EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec).array().sqrt() );  //.cast<float>() ;  
    }
     
    { 
      
      { 
 

      try {
        
        
            result_input.us_velocity_vec_proposed =            result_input.us_velocity_0_vec ;  
            result_input.us_theta_vec_proposed =  result_input.us_theta_vec_0;  
    
            // ---------------------------------------------------------------------------------------------------------------///    Perform L leapfrogs   ///-----------------------------------------------------------------------------------------------------------------------------------------
              generate_random_tau_ii(   EHMC_args_as_cpp_struct.tau_us,    EHMC_args_as_cpp_struct.tau_us_ii, rng);
            
              int    L_ii;
              if (EHMC_args_as_cpp_struct.diffusion_HMC == true)   L_ii = std::ceil(  EHMC_args_as_cpp_struct.tau_us_ii /  eps_1 );
              if (EHMC_args_as_cpp_struct.diffusion_HMC == false)  L_ii = std::ceil(  EHMC_args_as_cpp_struct.tau_us_ii /  EHMC_args_as_cpp_struct.eps_us );
              if (L_ii < 1) { L_ii = 1 ; }
              
              //// initial lp  (and grad)
              fn_lp_grad_InPlace(     result_input.lp_and_grad_outs, 
                                      Model_type, 
                                      force_autodiff, force_PartialLog, multi_attempts,
                                      result_input.main_theta_vec,  result_input.us_theta_vec, 
                                      y_ref,  grad_option,
                                      Model_args_as_cpp_struct, 
                                      Stan_model_as_cpp_struct);
              
              log_posterior_0 =  result_input.lp_and_grad_outs(0);
              U_x_initial = - log_posterior_0; // initial energy
    
           // if (EHMC_args_as_cpp_struct.diffusion_HMC == true) {
           //   
           //         //// make params/containers for sampling u's using advanced HMC
           //         Eigen::Matrix<double, -1, 1>  theta_us_vec_current_segment =     result_input.us_theta_vec;
           //         Eigen::Matrix<double, -1, 1>  velocity_us_vec_current_segment =  result_input.us_velocity_vec;
           //   
           //         leapfrog_integrator_diag_M_diffusion_HMC_nuisance_InPlace(   result_input.us_velocity_vec_proposed, 
           //                                                                      result_input.us_theta_vec_proposed, 
           //                                                                      result_input.lp_and_grad_outs, 
           //                                                                      theta_us_vec_current_segment, 
           //                                                                      velocity_us_vec_current_segment,
           //                                                                      result_input.main_theta_vec,
           //                                                                      EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec,
           //                                                                      y_ref,
           //                                                                      L_ii,
           //                                                                      eps_1, cos_eps_2, sin_eps_2,
           //                                                                      Model_type, 
           //                                                                      force_autodiff, force_PartialLog, multi_attempts, 
           //                                                                      grad_option,
           //                                                                      Model_args_as_cpp_struct,
           //                                                                      Stan_model_as_cpp_struct,
           //                                                                      fn_lp_grad_InPlace);
           //        
           // } else  {
           //      
           //        // leapfrog_integrator_diag_M_standard_HMC_nuisance_InPlace(   result_input.us_velocity_vec_proposed, 
           //        //                                                             result_input.us_theta_vec_proposed, 
           //        //                                                             result_input.lp_and_grad_outs, 
           //        //                                                             result_input.main_theta_vec,
           //        //                                                             EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec,
           //        //                                                             y_ref,
           //        //                                                             L_ii,
           //        //                                                             EHMC_args_as_cpp_struct.eps_us,
           //        //                                                             Model_type, 
           //        //                                                             force_autodiff, force_PartialLog, multi_attempts, 
           //        //                                                             grad_option,
           //        //                                                             Model_args_as_cpp_struct,
           //        //                                                             Stan_model_as_cpp_struct,
           //        //                                                             fn_lp_grad_InPlace);
           //        
           // 
           //         
           // }
           
           
           
           Eigen::Matrix<double, -1, 1> grad_us =  result_input.lp_and_grad_outs.segment(1, n_nuisance);
           
           //// Update velocity (first half step)
           Eigen::Matrix<double, -1, 1> temp_1 = grad_us.array()  * EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec.array();
           result_input.us_velocity_vec_proposed.array() += 0.5 * EHMC_args_as_cpp_struct.eps_us * temp_1.array();
           
           //// updae params by full step
           result_input.us_theta_vec_proposed.array()  +=  EHMC_args_as_cpp_struct.eps_us *     result_input.us_velocity_vec_proposed.array() ;
           
           //// Update lp and gradients
           fn_lp_grad_InPlace(result_input.lp_and_grad_outs, Model_type, force_autodiff, force_PartialLog, multi_attempts, 
                              result_input.main_theta_vec, result_input.us_theta_vec_proposed, y_ref, grad_option, 
                              Model_args_as_cpp_struct,  
                              Stan_model_as_cpp_struct);
           grad_us =  result_input.lp_and_grad_outs.segment(1, n_nuisance);
           
           //// Update velocity (second half step)
           Eigen::Matrix<double, -1, 1> temp_2 = grad_us.array()  * EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec.array();
           result_input.us_velocity_vec_proposed.array() += 0.5 * EHMC_args_as_cpp_struct.eps_us * temp_2.array();
           
                  
                  log_posterior_prop =   result_input.lp_and_grad_outs(0);
                  U_x_prop = - log_posterior_prop;
                  
            //////////////////////////////////////////////////////////////////    M-H acceptance step  (i.e, Accept/Reject step)
            {
   
                energy_old  = U_x_initial +  0.5 * (  result_input.us_velocity_0_vec.array()  *  result_input.us_velocity_0_vec.array()          * ( EHMC_Metric_struct_as_cpp_struct.M_us_vec ).array() ).sum() ; 
                energy_new  = U_x_prop +   0.5 * (    result_input.us_velocity_vec_proposed.array()  * result_input.us_velocity_vec_proposed.array()    * ( EHMC_Metric_struct_as_cpp_struct.M_us_vec   ).array() ).sum() ;
                
                if (EHMC_args_as_cpp_struct.diffusion_HMC == true)  {
                  energy_old +=   0.5 * (     result_input.us_theta_vec_0.array()   * result_input.us_theta_vec_0.array()   *   ( EHMC_Metric_struct_as_cpp_struct.M_us_vec   ).array() ).sum() ; 
                  energy_new +=   0.5 * (     result_input.us_theta_vec_proposed.array()     *  result_input.us_theta_vec_proposed.array()  * ( EHMC_Metric_struct_as_cpp_struct.M_us_vec  ).array() ).sum() ;
                }
      
                log_ratio = - energy_new + energy_old;
    
            }
            
          
          if  (check_divergence_Eigen(result_input, result_input.lp_and_grad_outs, energy_old, energy_new) == true)      {
                  
                  /// if us_div, reject proposal 
                  result_input.us_div = 1;
                  result_input.us_p_jump = 0.0;
                  
                  result_input.us_theta_vec  =        result_input.us_theta_vec_0;
                  result_input.us_velocity_vec =      result_input.us_velocity_0_vec ;
            
          }  else {  // if no us_div, carry on with MH step 
            
                  result_input.us_div = 0;
                  result_input.us_p_jump = std::min(1.0, stan::math::exp(log_ratio));
                    
                    if  ( (R::runif(0.0, 1.0) >  result_input.us_p_jump) ) {  // # reject proposal
                      result_input.us_theta_vec  =     result_input.us_theta_vec_0;
                      result_input.us_velocity_vec =   result_input.us_velocity_0_vec;
                    } else {   // # accept proposal
                      result_input.us_theta_vec  =   result_input.us_theta_vec_proposed ;  
                      result_input.us_velocity_vec = result_input.us_velocity_vec_proposed;
                    }
                  
          }

      } catch (...) { // if iteration fails (recorded as us_div)
        
              //  std::cout << "  Could not evaluate lp_grad function when sampling nuisance parameters " << ")\n";

                result_input.us_div = 1;
                result_input.us_p_jump = 0.0;
                
                result_input.us_theta_vec  =     result_input.us_theta_vec_0;
                result_input.us_velocity_vec =   result_input.us_velocity_0_vec;

      }
      
    }  
 

  }

 
}
  
  
  //return result_input;
  
  
}


 
 
 
  


 
