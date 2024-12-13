#pragma once

 
 

 

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <tbb/concurrent_vector.h>

 
 
#include <chrono> 
#include <unordered_map>
#include <memory>
#include <thread>
#include <functional>

 
 
//// ANSI codes for different colors
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"

 
using namespace Rcpp;
using namespace Eigen;
 
 
static std::mutex print_mutex;  //// global mutex 
static std::mutex result_mutex_1; //// global mutex 
static std::mutex result_mutex_2; //// global mutex 






//template<typename T = std::unique_ptr<dqrng::random_64bit_generator>>
template<typename T = std::mt19937>
ALWAYS_INLINE  void                    fn_sample_HMC_multi_iter_single_thread(      HMC_output_single_chain &HMC_output_single_chain_i,
                                                                                    HMCResult &result_input,
                                                                                    const bool burnin_indicator,
                                                                                    const int chain_id,
                                                                                    int current_iter,
                                                                                    const int seed,
                                                                                    T &rng,
                                                                                    const int n_iter,
                                                                                    const bool partitioned_HMC,
                                                                                    const std::string &Model_type,
                                                                                    const bool sample_nuisance,
                                                                                    const bool force_autodiff,
                                                                                    const bool force_PartialLog,
                                                                                    const bool multi_attempts,
                                                                                    const int n_nuisance_to_track,
                                                                                    const Eigen::Matrix<int, -1, -1> &y_Eigen_i,
                                                                                    const Model_fn_args_struct &Model_args_as_cpp_struct,  ///// ALWAYS read-only
                                                                                    EHMC_fn_args_struct  &EHMC_args_as_cpp_struct,
                                                                                    const EHMC_Metric_struct   &EHMC_Metric_as_cpp_struct, 
                                                                                    const Stan_model_struct    &Stan_model_as_cpp_struct)  {
  
 
     const int N =  Model_args_as_cpp_struct.N;
     const int n_us =  Model_args_as_cpp_struct.n_nuisance;
     const int n_params_main = Model_args_as_cpp_struct.n_params_main;
     const int n_params = n_params_main + n_us;
     
     const bool burnin = false; 
 
         ///////////////////////////////////////// perform iterations for adaptation interval
         ////// main iteration loop
         for (int ii = 0; ii < n_iter; ++ii) {
           
                     int seed_ii;
           
                     if (burnin_indicator == false) {
                         current_iter = ii;
                         // seed_ii = seed;
                         seed_ii = seed + 1000*(chain_id + 1) * 1000*(current_iter + n_iter + 1);
                     } else {  //// burnin 
                         //// leave "current_iter" as it is if burnin (current_iter will be given in via R)
                         seed_ii = seed + 1000*(chain_id + 1) * 1000*(current_iter + 1);
                     }
                     
                     rng.seed(seed_ii);  // change the seed
                     //const int stream_ii = (chain_id + 1) * current_iter + 1;
                     // std::unique_ptr<dqrng::random_64bit_generator> rng_ii;
                 
                     if (partitioned_HMC == true) {
                   
                               //////////////////////////////////////// sample nuisance (GIVEN main)
                               if (sample_nuisance == true)   {
                                 
                                             // auto rng_ii = dqrng::generator<pcg64_unique>(seed_ii);
                                             // std::mt19937 rng_ii(seed_ii); 
                                             
                                             stan::math::start_nested();
                                             fn_Diffusion_HMC_nuisance_only_single_iter_InPlace_process(    result_input,    
                                                                                                            burnin,  rng, seed_ii,
                                                                                                            Model_type, 
                                                                                                            force_autodiff, force_PartialLog,  multi_attempts, 
                                                                                                            y_Eigen_i,
                                                                                                            Model_args_as_cpp_struct,  
                                                                                                            EHMC_args_as_cpp_struct, EHMC_Metric_as_cpp_struct, 
                                                                                                            Stan_model_as_cpp_struct);
                                             stan::math::recover_memory_nested(); 
                                            
                                             HMC_output_single_chain_i.diagnostics_p_jump_us()(ii) =  result_input.us_p_jump();
                                             HMC_output_single_chain_i.diagnostics_div_us()(ii) =  result_input.us_div();
                                   
                                 } /// end of nuisance-part of iteration
                                 
                                 { /// sample main GIVEN u's
                                   
                                             // auto rng_ii = dqrng::generator<pcg64_unique>(seed_ii);
                                             // std::mt19937 rng_ii(seed_ii); 
                                   
                                             stan::math::start_nested();
                                             fn_standard_HMC_main_only_single_iter_InPlace_process(      result_input,   
                                                                                                         burnin,  rng, seed_ii,
                                                                                                         Model_type,  
                                                                                                         force_autodiff, force_PartialLog,  multi_attempts,
                                                                                                         y_Eigen_i,
                                                                                                         Model_args_as_cpp_struct, 
                                                                                                         EHMC_args_as_cpp_struct, EHMC_Metric_as_cpp_struct, 
                                                                                                         Stan_model_as_cpp_struct);
                                             stan::math::recover_memory_nested(); 
                                             
                                             HMC_output_single_chain_i.diagnostics_p_jump_main()(ii) =  result_input.main_p_jump();
                                             HMC_output_single_chain_i.diagnostics_div_main()(ii) =  result_input.main_div();
                                   
                                 } /// end of main_params part of iteration
                     
                     } else {  //// sample all params at once 
                       
                                         // auto rng_ii = dqrng::generator<pcg64_unique>(seed_ii);
                                         // std::mt19937 rng_ii(seed_ii); 
                       
                                         stan::math::start_nested();
                                         fn_standard_HMC_dual_single_iter_InPlace_process(    result_input,    
                                                                                              burnin,  rng, seed_ii,
                                                                                              Model_type, 
                                                                                              force_autodiff, force_PartialLog,  multi_attempts, 
                                                                                              y_Eigen_i,
                                                                                              Model_args_as_cpp_struct,   
                                                                                              EHMC_args_as_cpp_struct, EHMC_Metric_as_cpp_struct, 
                                                                                              Stan_model_as_cpp_struct);
                                         stan::math::recover_memory_nested(); 
                                         
                                         HMC_output_single_chain_i.diagnostics_p_jump_us()(ii) =  result_input.us_p_jump();
                                         HMC_output_single_chain_i.diagnostics_div_us()(ii) =  result_input.us_div();
                                         HMC_output_single_chain_i.diagnostics_p_jump_main()(ii) =  result_input.main_p_jump();
                                         HMC_output_single_chain_i.diagnostics_div_main()(ii) =  result_input.main_div();
                                         
                     }
                     
                     //// store iteration ii 
                     {
                         
                         std::lock_guard<std::mutex> lock(result_mutex_1);  
                         
                         // HMC_output_single_chain_i.store_iteration(ii, sample_nuisance);
                         HMC_output_single_chain_i.trace_main().col(ii) = result_input.main_theta_vec(); 
  
                         if (sample_nuisance == true) {
                              HMC_output_single_chain_i.trace_div()(0, ii) =  (0.50 * (result_input.main_div() + result_input.us_div()));
                              HMC_output_single_chain_i.trace_nuisance().col(ii) = result_input.us_theta_vec();  
                         } else {
                              HMC_output_single_chain_i.trace_div()(0, ii) = result_input.main_div();
                         }
                         
                     }
                       
         
         
                 if (burnin_indicator == false) {
                   if (ii %  static_cast<int>(std::round(static_cast<double>(n_iter)/4.0)) == 0) {
                     std::lock_guard<std::mutex> lock(print_mutex);
                     double pct_complete = 100.0 * (static_cast<double>(ii) / static_cast<double>(n_iter));
                     std::cout << "Chain #" << chain_id << " - Sampling is around " << pct_complete << " % complete" << "\n";
                   }
                 }
                               


           
         } ////////////////////// end of iteration(s)
         
         
    {
           std::lock_guard<std::mutex> lock(result_mutex_2);
           HMC_output_single_chain_i.result_input() = result_input;
    }
     
}
     
     
     
     
     
     
     
     
     
     

     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
