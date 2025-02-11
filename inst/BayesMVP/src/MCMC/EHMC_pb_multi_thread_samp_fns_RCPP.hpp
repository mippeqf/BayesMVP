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
 
 
 
 
 




     
 

// --------------------------------- RcpParallel  functions  -- SAMPLING fn ------------------------------------------------------------------------------------------------------------------------------------------- 





 
 
class RcppParallel_EHMC_sampling : public RcppParallel::Worker {
  
public:
  
          //// Clear all Eigen matrices:
          void reset_Eigen() {
                theta_main_vectors_all_chains_input_from_R_RcppPar.resize(0, 0);
                theta_us_vectors_all_chains_input_from_R_RcppPar.resize(0, 0);
          }
          
          //// Clear all tbb concurrent vectors
          void reset_tbb() { 
                HMC_outputs.clear();
                HMC_inputs.clear();
                y_copies.clear();
                Model_args_as_cpp_struct_copies.clear();
                EHMC_args_as_cpp_struct_copies.clear();
                EHMC_Metric_as_cpp_struct_copies.clear();
          }
          
          //// Clear all:
          void reset() {
                reset_tbb();
                reset_Eigen();
          } 
          
          //////////////////// ---- declare variables
          const uint64_t  global_seed;
          const int  n_threads;
          const int  n_iter;
          const bool partitioned_HMC;
          const std::string Model_type;
          const bool sample_nuisance;
          const bool force_autodiff;
          const bool force_PartialLog;
          const bool multi_attempts; 
          
          //// local storage:
          tbb::concurrent_vector<HMC_output_single_chain> HMC_outputs;
          tbb::concurrent_vector<HMCResult> HMC_inputs;
          
          //// Input data (to read)
          Eigen::Matrix<double, -1, -1>  theta_main_vectors_all_chains_input_from_R_RcppPar;
          Eigen::Matrix<double, -1, -1>  theta_us_vectors_all_chains_input_from_R_RcppPar;
          
          //// data:
          tbb::concurrent_vector<Eigen::Matrix<int, -1, -1>> y_copies;  
          
          //// input structs:
          tbb::concurrent_vector<Model_fn_args_struct>   Model_args_as_cpp_struct_copies;  
          tbb::concurrent_vector<EHMC_fn_args_struct>    EHMC_args_as_cpp_struct_copies;  
          tbb::concurrent_vector<EHMC_Metric_struct>     EHMC_Metric_as_cpp_struct_copies;  
          
          //////////////////// ---- declare SAMPLING-SPECIFIC variables:
          //// references to R trace matrices:
          const int n_nuisance_to_track;
          std::vector<Rcpp::NumericMatrix> &trace_output;
          std::vector<Rcpp::NumericMatrix> &trace_output_divs;
          std::vector<Rcpp::NumericMatrix> &trace_output_nuisance;
          //// this only gets used for built-in models, for Stan models log_lik must be defined in the "transformed parameters" block.
          std::vector<Rcpp::NumericMatrix> &trace_output_log_lik;   
          
  ////////////// Constructor (initialise these with the SOURCE format)
  RcppParallel_EHMC_sampling(  const int  &n_threads_R,
                               const uint64_t  &global_seed_R,
                               const int  &n_iter_R,
                               const bool &partitioned_HMC_R,
                               const std::string &Model_type_R,
                               const bool &sample_nuisance_R,
                               const bool &force_autodiff_R,
                               const bool &force_PartialLog_R,
                               const bool &multi_attempts_R,
                               //// inputs
                               const Eigen::Matrix<double, -1, -1> &theta_main_vectors_all_chains_input_from_R,
                               const Eigen::Matrix<double, -1, -1> &theta_us_vectors_all_chains_input_from_R,
                               ////  data:
                               const std::vector<Eigen::Matrix<int, -1, -1>> &y_copies_R,
                               //////////////  input structs:
                               const std::vector<Model_fn_args_struct>  &Model_args_as_cpp_struct_copies_R,   // READ-ONLY
                               std::vector<EHMC_fn_args_struct>  &EHMC_args_as_cpp_struct_copies_R,  
                               const std::vector<EHMC_Metric_struct>  &EHMC_Metric_as_cpp_struct_copies_R, // READ-ONLY
                               //// -------------- For POST-BURNIN only:
                               const int &n_nuisance_to_track_R,
                               std::vector<Rcpp::NumericMatrix> &trace_output_,
                               std::vector<Rcpp::NumericMatrix> &trace_output_divs_,
                               std::vector<Rcpp::NumericMatrix> &trace_output_nuisance_,
                               std::vector<Rcpp::NumericMatrix> &trace_output_log_lik_
                               )
    :
    n_threads(n_threads_R),
    global_seed(global_seed_R),
    n_iter(n_iter_R),
    partitioned_HMC(partitioned_HMC_R),
    Model_type(Model_type_R),
    sample_nuisance(sample_nuisance_R),
    force_autodiff(force_autodiff_R),
    force_PartialLog(force_PartialLog_R),
    multi_attempts(multi_attempts_R) ,
    //// inputs:
    theta_main_vectors_all_chains_input_from_R_RcppPar(theta_main_vectors_all_chains_input_from_R),
    theta_us_vectors_all_chains_input_from_R_RcppPar(theta_us_vectors_all_chains_input_from_R),
    //// -------------- For POST-BURNIN only:
    n_nuisance_to_track(n_nuisance_to_track_R), 
    trace_output(trace_output_),
    trace_output_divs(trace_output_divs_),
    trace_output_nuisance(trace_output_nuisance_),
    trace_output_log_lik(trace_output_log_lik_)
  {
            y_copies = convert_std_vec_to_concurrent_vector(y_copies_R, y_copies);
            Model_args_as_cpp_struct_copies = convert_std_vec_to_concurrent_vector(Model_args_as_cpp_struct_copies_R, Model_args_as_cpp_struct_copies);
            EHMC_args_as_cpp_struct_copies = convert_std_vec_to_concurrent_vector(EHMC_args_as_cpp_struct_copies_R, EHMC_args_as_cpp_struct_copies);
            EHMC_Metric_as_cpp_struct_copies = convert_std_vec_to_concurrent_vector(EHMC_Metric_as_cpp_struct_copies_R, EHMC_Metric_as_cpp_struct_copies);
            
            const int N = Model_args_as_cpp_struct_copies[0].N;
            const int n_us =  Model_args_as_cpp_struct_copies[0].n_nuisance;
            const int n_params_main = Model_args_as_cpp_struct_copies[0].n_params_main;
    
            HMC_outputs.reserve(n_threads_R);
            for (int i = 0; i < n_threads_R; ++i) {
              HMC_output_single_chain HMC_output_single_chain(n_iter_R, n_nuisance_to_track, n_params_main, n_us, N);
              HMC_outputs.emplace_back(HMC_output_single_chain);
            }
            
            HMC_inputs.reserve(n_threads_R);
            for (int i = 0; i < n_threads_R; ++i) {
              HMCResult HMCResult(n_params_main, n_us, N);
              HMC_inputs.emplace_back(HMCResult);
            } 
        
  }

  ////////////// RcppParallel Parallel operator
  void operator() (std::size_t begin, std::size_t end) {
 
              const int global_seed_int = static_cast<int>(global_seed);
              #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
                  dqrng::xoshiro256plus global_rng(global_seed_int);  
              #endif
      
              //// Process all chains from begin to end:
              for (std::size_t i = begin; i < end; ++i) {
                    
                          const int chain_id_int = static_cast<int>(i);
                          const int seed_i = global_seed_int + n_iter*(1 + chain_id_int);
                           
                          #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
                                     thread_local dqrng::xoshiro256plus rng_i(global_rng);      // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                                     rng_i.long_jump(n_iter*(1 + chain_id_int));  
                          #elif RNG_TYPE_CPP_STD == 1
                                     thread_local std::mt19937 rng_i;  // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                                     rng_i.seed(seed_i); 
                          #elif RNG_TYPE_pcg64 == 1
                                     pcg_extras::seed_seq_from<std::random_device> global_seed;
                                     thread_local pcg64 rng_i(global_seed, n_iter*(1 + chain_id_int)); // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                          #elif RNG_TYPE_pcg32 == 1
                                     pcg_extras::seed_seq_from<std::random_device> global_seed;
                                     thread_local pcg32 rng_i(global_seed, n_iter*(1 + chain_id_int)); // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                          #endif
                          
                          const int N = Model_args_as_cpp_struct_copies[i].N;
                          const int n_us =  Model_args_as_cpp_struct_copies[i].n_nuisance;
                          const int n_params_main = Model_args_as_cpp_struct_copies[i].n_params_main;
                          const int n_params = n_params_main + n_us;
                          
                          thread_local stan::math::ChainableStack ad_tape;     // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                          thread_local stan::math::nested_rev_autodiff nested; // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                           
                          const bool burnin_indicator = false;
                          const int current_iter = 0; // gets assigned later for post-burnin
                      
                    
                          {
                    
                              ///////////////////////////////////////// perform iterations for adaptation interval
                              HMC_inputs[i].main_theta_vec() =   theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
                              HMC_inputs[i].main_theta_vec_0() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
                              HMC_inputs[i].us_theta_vec() =     theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
                              HMC_inputs[i].us_theta_vec_0() =   theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
                
                              
                              if (Model_type == "Stan") {  
                   
                                            Stan_model_struct  Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(    Model_args_as_cpp_struct_copies[i].model_so_file,
                                                                                                                          Model_args_as_cpp_struct_copies[i].json_file_path, 
                                                                                                                          seed_i);
                                      
                                           //////////////////////////////// perform iterations for chain i:
                                           fn_sample_HMC_multi_iter_single_thread(   HMC_outputs[i] ,
                                                                                     HMC_inputs[i], 
                                                                                     burnin_indicator, 
                                                                                     chain_id_int, 
                                                                                     current_iter,
                                                                                     seed_i, 
                                                                                     rng_i,
                                                                                     n_iter,
                                                                                     partitioned_HMC,
                                                                                     Model_type,  sample_nuisance,
                                                                                     force_autodiff, force_PartialLog,  multi_attempts,  
                                                                                     n_nuisance_to_track, 
                                                                                     y_copies[i], 
                                                                                     Model_args_as_cpp_struct_copies[i],  
                                                                                     EHMC_args_as_cpp_struct_copies[i],
                                                                                     EHMC_Metric_as_cpp_struct_copies[i], 
                                                                                     Stan_model_as_cpp_struct);
                                            ////////////////////////////// end of iteration(s)
                                            
                                            //// destroy Stan model object:
                                            fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);  
                                            
                                          
                        
                              } else  { 
                                
                                            Stan_model_struct Stan_model_as_cpp_struct; ///  dummy struct
                                
                                            //////////////////////////////// perform iterations for chain i:
                                            fn_sample_HMC_multi_iter_single_thread(  HMC_outputs[i],   
                                                                                     HMC_inputs[i], 
                                                                                     burnin_indicator, 
                                                                                     chain_id_int, 
                                                                                     current_iter,
                                                                                     seed_i, 
                                                                                     rng_i,
                                                                                     n_iter,
                                                                                     partitioned_HMC,
                                                                                     Model_type,  sample_nuisance, 
                                                                                     force_autodiff, force_PartialLog,  multi_attempts,  
                                                                                     n_nuisance_to_track, 
                                                                                     y_copies[i], 
                                                                                     Model_args_as_cpp_struct_copies[i], 
                                                                                     EHMC_args_as_cpp_struct_copies[i], 
                                                                                     EHMC_Metric_as_cpp_struct_copies[i], 
                                                                                     Stan_model_as_cpp_struct);
                                            ////////////////////////////// end of iteration(s)
                                            
                              }
                              
                
                            
                
                          
                        } // end of parallel stuff
      
          }

    }  //// end of all parallel work// Definition of static thread_local variable
  
 
  
  // Copy results directly to R matrices
  void copy_results_to_output() {
    for (int i = 0; i < n_threads; ++i) {

          // Copy main trace
          for (int ii = 0; ii < HMC_outputs[i].trace_main().cols(); ++ii) {
            for (int param = 0; param < HMC_outputs[i].trace_main().rows(); ++param) {
              trace_output[i](param, ii) = (HMC_outputs[i].trace_main()(param, ii));
            }
          }

          // Copy divs
          for (int ii = 0; ii < HMC_outputs[i].trace_div().cols(); ++ii) {
            for (int param = 0; param < HMC_outputs[i].trace_div().rows(); ++param) {
              trace_output_divs[i](param, ii) =  (HMC_outputs[i].trace_div()(param, ii));
            }
          }

          // Copy nuisance  
          if (sample_nuisance) {
            for (int ii = 0; ii < HMC_outputs[i].trace_nuisance().cols(); ++ii) {
              for (int param = 0; param < HMC_outputs[i].trace_nuisance().rows(); ++param) {
                trace_output_nuisance[i](param, ii) =  (HMC_outputs[i].trace_nuisance()(param, ii));
              }
            }
          }

          // Copy log-lik   (for built-in models only)
          if (Model_type != "Stan") {
            for (int ii = 0; ii < HMC_outputs[i].trace_log_lik().cols(); ++ii) {
              for (int param = 0; param < HMC_outputs[i].trace_log_lik().rows(); ++param) {
                trace_output_log_lik[i](param, ii) =   (HMC_outputs[i].trace_log_lik()(param, ii));
              }
            }
          }

    }
  }



};



 
 
 
 





  