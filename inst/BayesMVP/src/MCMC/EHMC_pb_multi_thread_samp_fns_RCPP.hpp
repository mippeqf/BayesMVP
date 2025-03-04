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
 
 
 
 
 




#include <tbb/task_scheduler_observer.h>
#include <tbb/task_arena.h>
 
 class PinningObserver : public tbb::task_scheduler_observer {
   
       std::vector<int> core_ids;
   
     public:
       PinningObserver(const std::vector<int>& cores) 
         : core_ids(cores) {
         observe(true); // Activate the observer
       }
       
       void on_scheduler_entry(bool worker) override {
         int thread_id = tbb::this_task_arena::current_thread_index();
         if (thread_id < core_ids.size()) {
           cpu_set_t cpuset;
           CPU_ZERO(&cpuset);
           CPU_SET(core_ids[thread_id], &cpuset);
           pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
         }
       }
   
 };
 
 
 
 
 

// --------------------------------- RcpParallel  functions  -- SAMPLING fn ------------------------------------------------------------------------------------------------------------------------------------------- 





 
 
class RcppParallel_EHMC_sampling : public RcppParallel::Worker {
  
public:
  
          
          //////////////////// ---- declare variables
          // #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
          //       dqrng::xoshiro256plus global_rng_main;
          //       dqrng::xoshiro256plus global_rng_nuisance;
          // #endif
          
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
          std::vector<HMC_output_single_chain> HMC_outputs;
          std::vector<HMCResult> HMC_inputs;
          
          //// Input data (to read)
          const Eigen::Matrix<double, -1, -1>  &theta_main_vectors_all_chains_input_from_R_RcppPar;
          const Eigen::Matrix<double, -1, -1>  &theta_us_vectors_all_chains_input_from_R_RcppPar;
          
          //// data:
          const std::vector<Eigen::Matrix<int, -1, -1>> &y_copies;  
          
          //// input structs:
          const std::vector<Model_fn_args_struct>   &Model_args_as_cpp_struct_copies;  
          std::vector<EHMC_fn_args_struct>          &EHMC_args_as_cpp_struct_copies;  //// These have to be modifiable
          const std::vector<EHMC_Metric_struct>     &EHMC_Metric_as_cpp_struct_copies;  
          
          //////////////////// ---- declare SAMPLING-SPECIFIC variables:
          //// references to R trace matrices:
          const int n_nuisance_to_track;
          std::vector<Eigen::Matrix<double, -1, -1>> &trace_output;
          std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_divs;
          std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_nuisance;
          //// this only gets used for built-in models, for Stan models log_lik must be defined in the "transformed parameters" block.
          std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_log_lik;   
          
          
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
                               //// data:
                               const std::vector<Eigen::Matrix<int, -1, -1>> &y_copies_,
                               //// input structs:
                               const std::vector<Model_fn_args_struct>  &Model_args_as_cpp_struct_copies_,   // READ-ONLY
                               std::vector<EHMC_fn_args_struct>  &EHMC_args_as_cpp_struct_copies_,  
                               const std::vector<EHMC_Metric_struct>  &EHMC_Metric_as_cpp_struct_copies_, // READ-ONLY
                               //// -------------- For POST-BURNIN only:
                               const int &n_nuisance_to_track_R,
                               std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_,
                               std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_divs_,
                               std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_nuisance_,
                               std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_log_lik_
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
    ////  data:
    y_copies(y_copies_),
    //// input structs:
    Model_args_as_cpp_struct_copies(Model_args_as_cpp_struct_copies_),
    EHMC_args_as_cpp_struct_copies(EHMC_args_as_cpp_struct_copies_),
    EHMC_Metric_as_cpp_struct_copies(EHMC_Metric_as_cpp_struct_copies_),
    //// -------------- For POST-BURNIN only:
    n_nuisance_to_track(n_nuisance_to_track_R), 
    trace_output(trace_output_),
    trace_output_divs(trace_output_divs_),
    trace_output_nuisance(trace_output_nuisance_),
    trace_output_log_lik(trace_output_log_lik_)
  {
    
            // #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
            //     global_rng_main.seed(global_seed_R);
            //     global_rng_nuisance.seed(global_seed_R + 1e6);
            // #endif
            
            const int N = Model_args_as_cpp_struct_copies[0].N;
            const int n_us =  Model_args_as_cpp_struct_copies[0].n_nuisance;
            const int n_params_main = Model_args_as_cpp_struct_copies[0].n_params_main;
    
            HMC_outputs.reserve(n_threads_R);
            HMC_inputs.reserve(n_threads_R);
            for (int i = 0; i < n_threads_R; ++i) {
                HMC_outputs.emplace_back(n_iter_R, n_nuisance_to_track, n_params_main, n_us, N);
                HMC_inputs.emplace_back(n_params_main, n_us, N);
            }
        
  }

  ////////////// RcppParallel Parallel operator
  void operator() (std::size_t begin, std::size_t end) {
           
              const uint64_t global_seed_main =     global_seed;
              const uint64_t global_seed_nuisance = global_seed + 1e6;
              const int global_seed_main_int =      static_cast<int>(global_seed_main);
              const int global_seed_nuisance_int =  static_cast<int>(global_seed_nuisance);
      
              //// Process all chains from begin to end:
              for (std::size_t i = begin; i < end; ++i) {
                    
                           const int chain_id_int = static_cast<int>(i);
                           const int seed_main_int_i =     global_seed_main_int +     n_iter*(1 + chain_id_int);
                           const int seed_nuisance_int_i = global_seed_nuisance_int + n_iter*(1 + chain_id_int);
                           
                           #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
                                 dqrng::xoshiro256plus rng_main_i; //(global_rng_main);      // make thread local copy of rng 
                                 dqrng::xoshiro256plus rng_nuisance_i; //(global_rng_nuisance);      // make thread local copy of rng 
                                 rng_main_i.seed(seed_main_int_i);  // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                                 rng_nuisance_i.seed(seed_nuisance_int_i);  // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                           #elif RNG_TYPE_CPP_STD == 1
                                 std::mt19937 rng_main_i;  // Fresh RNG // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                                 std::mt19937 rng_nuisance_i;  // Fresh RNG // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                                 rng_main_i.seed(seed_main_int_i); // set / re-set the seed
                                 rng_nuisance_i.seed(seed_nuisance_int_i); // set / re-set the seed
                           #endif
                          
                           const int N = Model_args_as_cpp_struct_copies[i].N;
                           const int n_us =  Model_args_as_cpp_struct_copies[i].n_nuisance;
                           const int n_params_main = Model_args_as_cpp_struct_copies[i].n_params_main;
                           const int n_params = n_params_main + n_us;
                          
                           stan::math::ChainableStack ad_tape;     // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                           //// stan::math::nested_rev_autodiff nested; // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                           
                           const bool burnin_indicator = false;
                           const int current_iter = 0; // gets assigned later for post-burnin
                      
                    
                          {
                    
                              ///////////////////////////////////////// perform iterations for adaptation interval
                              HMC_inputs[i].main_theta_vec() =   theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
                              HMC_inputs[i].main_theta_vec_0() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
                              HMC_inputs[i].us_theta_vec() =     theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
                              HMC_inputs[i].us_theta_vec_0() =   theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
                
                              
                              if (Model_type == "Stan") {  
                   
                                           Stan_model_struct  Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(     Model_args_as_cpp_struct_copies[i].model_so_file,
                                                                                                                          Model_args_as_cpp_struct_copies[i].json_file_path, 
                                                                                                                          seed_main_int_i);
                                      
                                           //////////////////////////////// perform iterations for chain i:
                                           fn_sample_HMC_multi_iter_single_thread(   HMC_outputs[i] ,
                                                                                     HMC_inputs[i], 
                                                                                     burnin_indicator, 
                                                                                     chain_id_int, 
                                                                                     current_iter,
                                                                                     seed_main_int_i, 
                                                                                     seed_nuisance_int_i,
                                                                                     rng_main_i,
                                                                                     rng_nuisance_i,
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
                                                                                     seed_main_int_i, 
                                                                                     seed_nuisance_int_i,
                                                                                     rng_main_i,
                                                                                     rng_nuisance_i,
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
    
          // const int n_nuisance = HMC_outputs[0].trace_nuisance().rows();
          const int n_iter = HMC_outputs[0].trace_main().cols();
          
          for (int i = 0; i < n_threads; ++i) {
            
                      // Copy main trace
                      for (int ii = 0; ii < n_iter; ++ii) {
                  
                          trace_output[i].col(ii) = HMC_outputs[i].trace_main().col(ii);
                          trace_output_divs[i].col(ii) =  HMC_outputs[i].trace_div().col(ii);
                          if (sample_nuisance == true) {
                            trace_output_nuisance[i].col(ii) =  HMC_outputs[i].trace_nuisance().col(ii);
                          }
                          if (Model_type != "Stan") {
                            trace_output_log_lik[i].col(ii) =   HMC_outputs[i].trace_log_lik().col(ii);
                          }
                       
                      }
      
          }
    
  }



};



 
 
 
 





  