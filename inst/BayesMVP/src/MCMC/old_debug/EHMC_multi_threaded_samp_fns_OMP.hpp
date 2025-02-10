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
 
 
 

     
     
     
    
     
//// --------------------------------- OpenMP functions -- BURNIN fn ------------------------------------------------------------------------------------------------------------------------------------------- 
     
     
     
     
//// OpenMP
void EHMC_burnin_OpenMP(    const int  n_threads,
                            const int  seed,
                            const int  n_iter,
                            const int  current_iter,
                            const bool partitioned_HMC,
                            const std::string &Model_type,
                            const bool sample_nuisance,
                            const bool force_autodiff,
                            const bool force_PartialLog,
                            const bool multi_attempts,
                            ///// inputs
                            const Eigen::Matrix<double, -1, -1>  &theta_main_vectors_all_chains_input_from_R_RcppPar,
                            const Eigen::Matrix<double, -1, -1>  &theta_us_vectors_all_chains_input_from_R_RcppPar,
                            ///// other outputs  
                            Eigen::Matrix<double, -1, -1>   &other_main_out_vector_all_chains_output_to_R_RcppPar,
                            Eigen::Matrix<double, -1, -1>   &other_us_out_vector_all_chains_output_to_R_RcppPar,
                            //// data
                            const std::vector<Eigen::Matrix<int, -1, -1>>   &y_copies,
                            //////////////  input structs
                            std::vector<Model_fn_args_struct> &Model_args_as_cpp_struct_copies,
                            std::vector<EHMC_fn_args_struct>  &EHMC_args_as_cpp_struct_copies,
                            std::vector<EHMC_Metric_struct>   &EHMC_Metric_as_cpp_struct_copies,
                            std::vector<EHMC_burnin_struct>   &EHMC_burnin_as_cpp_struct_copies,
                            //// nuisance outputs
                            Eigen::Matrix<double, -1, -1>  &theta_main_vectors_all_chains_output_to_R_RcppPar,
                            Eigen::Matrix<double, -1, -1>  &theta_us_vectors_all_chains_output_to_R_RcppPar,
                            //// main
                            Eigen::Matrix<double, -1, -1>  &theta_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar,
                            Eigen::Matrix<double, -1, -1>  &theta_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar,
                            Eigen::Matrix<double, -1, -1>  &velocity_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar,
                            Eigen::Matrix<double, -1, -1>  &velocity_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar,
                            //// nuisance
                            Eigen::Matrix<double, -1, -1>  &theta_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar,
                            Eigen::Matrix<double, -1, -1>  &theta_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar,
                            Eigen::Matrix<double, -1, -1>  &velocity_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar,
                            Eigen::Matrix<double, -1, -1>  &velocity_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar
) {
  
       omp_set_num_threads(n_threads);
       omp_set_dynamic(0);  
       
       const int N =  Model_args_as_cpp_struct_copies[0].N;
       const int n_us =  Model_args_as_cpp_struct_copies[0].n_nuisance;
       const int n_params_main = Model_args_as_cpp_struct_copies[0].n_params_main;
       const int n_nuisance_to_track = 1;
       
       std::vector<HMC_output_single_chain> HMC_outputs;
       HMC_outputs.reserve(n_threads);
       for (int i = 0; i < n_threads; ++i) {
         HMC_output_single_chain HMC_output_single_chain(n_iter, n_nuisance_to_track, n_params_main, n_us, N);
         HMC_outputs.emplace_back(HMC_output_single_chain);
       }
       
       std::vector<HMCResult> HMC_inputs;
       HMC_inputs.reserve(n_threads);
       for (int i = 0; i < n_threads; ++i) {
         HMCResult HMCResult(n_params_main, n_us, N);
         HMC_inputs.emplace_back(HMCResult);
       } 

       //// parallel for-loop
       #pragma omp parallel for shared(HMC_outputs, HMC_inputs)
       for (int i = 0; i < n_threads; i++) {  
         
             const int chain_id = i;
         
             const int N =  Model_args_as_cpp_struct_copies[i].N;
             const int n_us =  Model_args_as_cpp_struct_copies[i].n_nuisance;
             const int n_params_main = Model_args_as_cpp_struct_copies[i].n_params_main;
             const int n_params = n_params_main + n_us;
             const bool burnin_indicator = false;
             const int n_nuisance_to_track = 1;
             
             stan::math::ChainableStack ad_tape;
             stan::math::nested_rev_autodiff nested;
             
             pcg64 rng(seed, chain_id); // each chain gets its own RNG stream
             
         {
           
           {
             
             ///////////////////////////////////////// perform iterations for adaptation interval
             // HMCResult result_input(n_params_main, n_us, N);
             HMC_inputs[i].main_theta_vec() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
             HMC_inputs[i].main_theta_vec_0() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
             
             if (sample_nuisance == true)  {
               HMC_inputs[i].us_theta_vec() = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
               HMC_inputs[i].us_theta_vec_0() = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
             }
             
             {
               
               //////////////////////////////// perform iterations for chain i
               if (Model_type == "Stan") {  
                 
                     Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct_copies[i].model_so_file,
                                                                                                Model_args_as_cpp_struct_copies[i].json_file_path, 
                                                                                                seed + chain_id);
                     
                     fn_sample_HMC_multi_iter_single_thread(    HMC_outputs[i],
                                                                HMC_inputs[i], 
                                                                burnin_indicator, 
                                                                chain_id,  
                                                                current_iter, 
                                                                seed, 
                                                                rng,
                                                                n_iter,
                                                                partitioned_HMC,
                                                                Model_type, sample_nuisance,
                                                                force_autodiff, force_PartialLog,  multi_attempts,  n_nuisance_to_track, 
                                                                y_copies[i], 
                                                                Model_args_as_cpp_struct_copies[i], 
                                                                EHMC_args_as_cpp_struct_copies[i], 
                                                                EHMC_Metric_as_cpp_struct_copies[i], 
                                                                Stan_model_as_cpp_struct);
                     
                     fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct); //// destroy Stan model object
                 
               } else { 
                 
                     Stan_model_struct Stan_model_as_cpp_struct; ////  dummy struct
                     
                     fn_sample_HMC_multi_iter_single_thread(    HMC_outputs[i],
                                                                HMC_inputs[i], 
                                                                burnin_indicator, 
                                                                chain_id, 
                                                                current_iter,
                                                                seed, 
                                                                rng,
                                                                n_iter,
                                                                partitioned_HMC,
                                                                Model_type, sample_nuisance,
                                                                force_autodiff, force_PartialLog,  multi_attempts,  n_nuisance_to_track, 
                                                                y_copies[i], 
                                                                Model_args_as_cpp_struct_copies[i], 
                                                                EHMC_args_as_cpp_struct_copies[i], 
                                                                EHMC_Metric_as_cpp_struct_copies[i], 
                                                                Stan_model_as_cpp_struct);
                 
                 
               }
             
             if (sample_nuisance == true)  {
               //////// Write results back to the shared array once half-iteration completed
               theta_us_vectors_all_chains_output_to_R_RcppPar.col(i) =         HMC_inputs[i].us_theta_vec();
               //// for burnin / ADAM-tau adaptation only
               theta_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) =  HMC_inputs[i].us_theta_vec_0();
               theta_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) = HMC_inputs[i].us_theta_vec_proposed() ;
               velocity_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) = HMC_inputs[i].us_velocity_0_vec();
               velocity_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) =  HMC_inputs[i].us_velocity_vec_proposed();
             }
             
             //////// Write results back to the shared array once half-iteration completed
             theta_main_vectors_all_chains_output_to_R_RcppPar.col(i) =    HMC_inputs[i].main_theta_vec();
             ///// for burnin / ADAM-tau adaptation only
             theta_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) =  HMC_inputs[i].main_theta_vec_0();
             theta_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) = HMC_inputs[i].main_theta_vec_proposed();
             velocity_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) =  HMC_inputs[i].main_velocity_0_vec();
             velocity_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) = HMC_inputs[i].main_velocity_vec_proposed();
             
             //////// compute summaries at end of iterations from each chain
             //// other outputs (once all iterations finished) - main
             other_main_out_vector_all_chains_output_to_R_RcppPar(0, i) = HMC_outputs[i].diagnostics_p_jump_main().sum() / static_cast<double>(n_iter);
             other_main_out_vector_all_chains_output_to_R_RcppPar(1, i) = HMC_outputs[i].diagnostics_div_main().sum();
             //// other outputs (once all iterations finished) - nuisance
             if (sample_nuisance == true)  {
               other_us_out_vector_all_chains_output_to_R_RcppPar(0, i) =  HMC_outputs[i].diagnostics_p_jump_us().sum() / static_cast<double>(n_iter);
               other_us_out_vector_all_chains_output_to_R_RcppPar(1, i) =  HMC_outputs[i].diagnostics_div_us().sum();
             }
             
             /////////////////  ---- burnin-specific stuff -----
             //// other outputs (once all iterations finished) - main
             other_main_out_vector_all_chains_output_to_R_RcppPar(2, i) = EHMC_burnin_as_cpp_struct_copies[i].tau_m_adam_main;
             other_main_out_vector_all_chains_output_to_R_RcppPar(3, i) = EHMC_burnin_as_cpp_struct_copies[i].tau_v_adam_main;
             other_main_out_vector_all_chains_output_to_R_RcppPar(4, i) = EHMC_args_as_cpp_struct_copies[i].tau_main;
             other_main_out_vector_all_chains_output_to_R_RcppPar(5, i) = EHMC_args_as_cpp_struct_copies[i].tau_main_ii;
             ////
             other_main_out_vector_all_chains_output_to_R_RcppPar(6, i) = EHMC_burnin_as_cpp_struct_copies[i].eps_m_adam_main;
             other_main_out_vector_all_chains_output_to_R_RcppPar(7, i) = EHMC_burnin_as_cpp_struct_copies[i].eps_v_adam_main;
             other_main_out_vector_all_chains_output_to_R_RcppPar(8, i) = EHMC_args_as_cpp_struct_copies[i].eps_main;
             //// other outputs (once all iterations finished) - nuisance
             if (sample_nuisance == true)  {
               ////
               other_us_out_vector_all_chains_output_to_R_RcppPar(2, i) = EHMC_burnin_as_cpp_struct_copies[i].tau_m_adam_us;
               other_us_out_vector_all_chains_output_to_R_RcppPar(3, i) = EHMC_burnin_as_cpp_struct_copies[i].tau_v_adam_us;
               other_us_out_vector_all_chains_output_to_R_RcppPar(4, i) = EHMC_args_as_cpp_struct_copies[i].tau_us;
               other_us_out_vector_all_chains_output_to_R_RcppPar(5, i) = EHMC_args_as_cpp_struct_copies[i].tau_us_ii;
               ////
               other_us_out_vector_all_chains_output_to_R_RcppPar(6, i) = EHMC_burnin_as_cpp_struct_copies[i].eps_m_adam_us;
               other_us_out_vector_all_chains_output_to_R_RcppPar(7, i) = EHMC_burnin_as_cpp_struct_copies[i].eps_v_adam_us;
               other_us_out_vector_all_chains_output_to_R_RcppPar(8, i) = EHMC_args_as_cpp_struct_copies[i].eps_us;
             }

           }
           
         }  
         
       }  //// end of parallel OpenMP loop
       
     }


}









 







      



//// --------------------------------- OpenMP  functions  -- SAMPLING fn ------------------------------------------------------------------------------------------------------------------------------------------- 



 

//// OpenMP  - SAMPLING
void EHMC_sampling_OpenMP(    const int  n_threads,
                              const int  seed,
                              const int  n_iter,
                              const bool partitioned_HMC,
                              const std::string &Model_type,
                              const bool sample_nuisance,
                              const bool force_autodiff,
                              const bool force_PartialLog,
                              const bool multi_attempts,
                              ///// inputs
                              const Eigen::Matrix<double, -1, -1>  &theta_main_vectors_all_chains_input_from_R_RcppPar,
                              const Eigen::Matrix<double, -1, -1>  &theta_us_vectors_all_chains_input_from_R_RcppPar,
                              //// output (main)
                              std::vector<Eigen::Matrix<double, -1, -1>> &trace_output,
                              //// data
                              const std::vector<Eigen::Matrix<int, -1, -1>>   &y_copies,
                              //////////////  input structs
                              std::vector<Model_fn_args_struct> &Model_args_as_cpp_struct_copies,
                              std::vector<EHMC_fn_args_struct>  &EHMC_args_as_cpp_struct_copies,
                              std::vector<EHMC_Metric_struct>   &EHMC_Metric_as_cpp_struct_copies,
                              /////////////
                              std::vector<Eigen::Matrix<double, -1, -1>> &trace_divs,
                              const int &n_nuisance_to_track_R,
                              std::vector<Eigen::Matrix<double, -1, -1>> &trace_nuisance,
                              std::vector<Eigen::Matrix<double, -1, -1>> &trace_log_lik
) {

  //// local storage 
  const int N = Model_args_as_cpp_struct_copies[0].N;
  const int n_us =  Model_args_as_cpp_struct_copies[0].n_nuisance;
  const int n_params_main = Model_args_as_cpp_struct_copies[0].n_params_main;
  const int n_nuisance_to_track = 1;
  
  std::vector<HMC_output_single_chain> HMC_outputs;
  HMC_outputs.reserve(n_threads);
  for (int i = 0; i < n_threads; ++i) {
    HMC_output_single_chain HMC_output_single_chain(n_iter, n_nuisance_to_track, n_params_main, n_us, N);
    HMC_outputs.emplace_back(HMC_output_single_chain);
  }
  
  std::vector<HMCResult> HMC_inputs;
  HMC_inputs.reserve(n_threads);
  for (int i = 0; i < n_threads; ++i) {
    HMCResult HMCResult(n_params_main, n_us, N);
    HMC_inputs.emplace_back(HMCResult);
  } 
  
  omp_set_num_threads(n_threads);
  omp_set_dynamic(0);

  //// parallel for-loop
  #pragma omp parallel for shared(HMC_outputs)
  for (int i = 0; i < n_threads; i++) {  
    
        const int chain_id = i;
    
        const int N =  Model_args_as_cpp_struct_copies[i].N;
        const int n_us =  Model_args_as_cpp_struct_copies[i].n_nuisance;
        const int n_params_main = Model_args_as_cpp_struct_copies[i].n_params_main;
        const int n_params = n_params_main + n_us;
        
        const bool burnin_indicator = false;
        const int n_nuisance_to_track = 1;
        
        thread_local stan::math::ChainableStack ad_tape;
        thread_local stan::math::nested_rev_autodiff nested;
        
        pcg64 rng(seed, chain_id); // each chain gets its own RNG stream
        
        int current_iter = 0; // gets assigned later for post-burnin
        
        ///////////////////////////////////////// perform iterations for adaptation interval
        HMC_inputs[i].main_theta_vec() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
        HMC_inputs[i].main_theta_vec_0() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
        
        if (sample_nuisance == true)  {
          HMC_inputs[i].us_theta_vec() = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
          HMC_inputs[i].us_theta_vec_0() = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
        }
        
          //////////////////////////////// perform iterations for chain i
          if (Model_type == "Stan") {  
            
                Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct_copies[i].model_so_file,
                                                                                           Model_args_as_cpp_struct_copies[i].json_file_path, 
                                                                                           seed + chain_id);
                
                fn_sample_HMC_multi_iter_single_thread(    HMC_outputs[i],
                                                           HMC_inputs[i], 
                                                           burnin_indicator, 
                                                           chain_id, 
                                                           current_iter,
                                                           seed, 
                                                           rng,
                                                           n_iter,
                                                           partitioned_HMC,
                                                           Model_type, sample_nuisance,
                                                           force_autodiff, force_PartialLog,  multi_attempts,  n_nuisance_to_track, 
                                                           y_copies[i], 
                                                           Model_args_as_cpp_struct_copies[i], 
                                                           EHMC_args_as_cpp_struct_copies[i], 
                                                           EHMC_Metric_as_cpp_struct_copies[i], 
                                                           Stan_model_as_cpp_struct);
                
                fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct); //// destroy Stan model object
            
          } else { 
            
              Stan_model_struct Stan_model_as_cpp_struct; ////  dummy struct
              
              fn_sample_HMC_multi_iter_single_thread(    HMC_outputs[i],
                                                         HMC_inputs[i], 
                                                         burnin_indicator, 
                                                         chain_id, 
                                                         current_iter,
                                                         seed, 
                                                         rng,
                                                         n_iter,
                                                         partitioned_HMC,
                                                         Model_type, sample_nuisance,
                                                         force_autodiff, force_PartialLog,  multi_attempts,  n_nuisance_to_track, 
                                                         y_copies[i], 
                                                         Model_args_as_cpp_struct_copies[i], 
                                                         EHMC_args_as_cpp_struct_copies[i], 
                                                         EHMC_Metric_as_cpp_struct_copies[i], 
                                                         Stan_model_as_cpp_struct);
            
          }
    
  }  //// end of parallel OpenMP loop
  
  #pragma omp barrier  // Make sure all threads are done before copying
  #pragma omp flush(HMC_outputs)  // Ensure all writes to HMC_outputs are visible
  
  // At end of sampling, copy results directly to R matrices
    for (int i = 0; i < n_threads; ++i) {
      
            // Copy main trace
            for (int ii = 0; ii < HMC_outputs[i].trace_main().cols(); ++ii) {
              for (int param = 0; param < HMC_outputs[i].trace_main().rows(); ++param) {
                trace_output[i](param, ii) =  (HMC_outputs[i].trace_main()(param, ii));
              }
            }
            
            // Copy divs
            for (int ii = 0; ii < HMC_outputs[i].trace_div().cols(); ++ii) {
              for (int param = 0; param < HMC_outputs[i].trace_div().rows(); ++param) {
                trace_divs[i](param, ii) =  (HMC_outputs[i].trace_div()(param, ii));
              }
            }
            
            // Copy nuisance  
            if (sample_nuisance) {
              for (int ii = 0; ii < HMC_outputs[i].trace_nuisance().cols(); ++ii) {
                for (int param = 0; param < HMC_outputs[i].trace_nuisance().rows(); ++param) {
                  trace_nuisance[i](param, ii) =  (HMC_outputs[i].trace_nuisance()(param, ii));
                }
              }
            }
            
            // Copy log-lik   (for built-in models only)
            if (Model_type != "Stan") {
              for (int ii = 0; ii < HMC_outputs[i].trace_log_lik().cols(); ++ii) {
                for (int param = 0; param < HMC_outputs[i].trace_log_lik().rows(); ++param) {
                  trace_log_lik[i](param, ii) =  (HMC_outputs[i].trace_log_lik()(param, ii));
                }
              }
            }
    }
 
}



 





  