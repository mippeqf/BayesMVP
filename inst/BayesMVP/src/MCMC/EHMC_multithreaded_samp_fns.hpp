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
 
 
   

   

template<typename T = std::unique_ptr<dqrng::random_64bit_generator>>
ALWAYS_INLINE  void                    fn_sample_HMC_multi_iter_single_thread(                    HMC_output_single_chain &HMC_output_single_chain_i,
                                                                                                  HMCResult &result_input,
                                                                                                  const bool burnin_indicator,
                                                                                                  const int chain_id,
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
           
                     //// make a copy of rng for iteration ii + advance rng by ii
                     auto rng_ii = rng->clone(chain_id * n_iter + ii + 1);    
                     // auto rng_ii = rng->clone(0);  // Get a copy
                     // rng_ii->advance(chain_id * n_iter + ii + 1);  // Then advance it
                 
                     if (partitioned_HMC == true) {
                       
                               stan::math::start_nested();
                   
                               //////////////////////////////////////// sample nuisance (GIVEN main)
                               if (sample_nuisance == true)   {
                                     
                                           fn_Diffusion_HMC_nuisance_only_single_iter_InPlace_process(    result_input,    
                                                                                                          burnin,  rng_ii, seed,
                                                                                                          Model_type, 
                                                                                                          force_autodiff, force_PartialLog,  multi_attempts, 
                                                                                                          y_Eigen_i,
                                                                                                          Model_args_as_cpp_struct,  
                                                                                                          EHMC_args_as_cpp_struct, EHMC_Metric_as_cpp_struct, 
                                                                                                          Stan_model_as_cpp_struct);
                                        
                                         
                                             HMC_output_single_chain_i.diagnostics_p_jump_us()(ii) =  result_input.us_p_jump();
                                             HMC_output_single_chain_i.diagnostics_div_us()(ii) =  result_input.us_div();
                                   
                                 } /// end of nuisance-part of iteration
                                 
                                 { /// sample main GIVEN u's
                                     
                                         fn_standard_HMC_main_only_single_iter_InPlace_process(      result_input,   
                                                                                                     burnin,  rng_ii, seed,
                                                                                                     Model_type,  
                                                                                                     force_autodiff, force_PartialLog,  multi_attempts,
                                                                                                     y_Eigen_i,
                                                                                                     Model_args_as_cpp_struct, 
                                                                                                     EHMC_args_as_cpp_struct, EHMC_Metric_as_cpp_struct, 
                                                                                                     Stan_model_as_cpp_struct);
                                
                                       
                                         HMC_output_single_chain_i.diagnostics_p_jump_main()(ii) =  result_input.main_p_jump();
                                         HMC_output_single_chain_i.diagnostics_div_main()(ii) =  result_input.main_div();
                                   
                                 } /// end of main_params part of iteration
                               
                                 stan::math::recover_memory_nested(); 
                     
                     } else {  //// sample all params at once 
                       
                                         stan::math::start_nested();
                                         fn_standard_HMC_dual_single_iter_InPlace_process(    result_input,    
                                                                                              burnin,  rng_ii, seed,
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
     
     
     
     
     
     
     
     
     
     

     
     
     
     
//// --------------------------------- OpenMP  functions  -- BURNIN fn ------------------------------------------------------------------------------------------------------------------------------------------- 
     
     
     
     
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
                            std::vector<HMC_output_single_chain> &HMC_outputs,
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
       
       // auto global_rng = dqrng::generator<dqrng::xoshiro256plusplus>(seed); // base RNG
       
       //// parallel for-loop
       #pragma omp parallel for
       for (int i = 0; i < n_threads; i++) {  
            
             // auto rng = global_rng->clone(i + current_iter + 1);  // Create thread local copy and advance it by thread number + 1 jumps
             auto rng = dqrng::generator<pcg64>(seed + current_iter, i);
         
             const int N =  Model_args_as_cpp_struct_copies[i].N;
             const int n_us =  Model_args_as_cpp_struct_copies[i].n_nuisance;
             const int n_params_main = Model_args_as_cpp_struct_copies[i].n_params_main;
             const int n_params = n_params_main + n_us;
             const bool burnin_indicator = false;
             const int n_nuisance_to_track = 1;
             
             stan::math::ChainableStack ad_tape;
             stan::math::nested_rev_autodiff nested;
         
         {
           
           {
             
             ///////////////////////////////////////// perform iterations for adaptation interval
             HMCResult result_input(n_params_main, n_us, N);
             result_input.main_theta_vec() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
             result_input.main_theta_vec_0() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
             
             if (sample_nuisance == true)  {
               result_input.us_theta_vec() = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
               result_input.us_theta_vec_0() = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
             }
             
             {
               
               //////////////////////////////// perform iterations for chain i
               if (Model_type == "Stan") {  
                 
                     Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct_copies[i].model_so_file,
                                                                                                Model_args_as_cpp_struct_copies[i].json_file_path, 
                                                                                                seed + i * 1000);
                     
                     fn_sample_HMC_multi_iter_single_thread(    HMC_outputs[i],
                                                                result_input, 
                                                                burnin_indicator, 
                                                                i, seed + i * 1000, rng, n_iter,
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
                                                                result_input, 
                                                                burnin_indicator, 
                                                                i, seed + i * 1000, rng, n_iter,
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
               theta_us_vectors_all_chains_output_to_R_RcppPar.col(i) =         result_input.us_theta_vec();
               //// for burnin / ADAM-tau adaptation only
               theta_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) =  result_input.us_theta_vec_0();
               theta_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) = result_input.us_theta_vec_proposed() ;
               velocity_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) = result_input.us_velocity_0_vec();
               velocity_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) =  result_input.us_velocity_vec_proposed();
             }
             
             //////// Write results back to the shared array once half-iteration completed
             theta_main_vectors_all_chains_output_to_R_RcppPar.col(i) =    result_input.main_theta_vec();
             ///// for burnin / ADAM-tau adaptation only
             theta_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) =  result_input.main_theta_vec_0();
             theta_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) = result_input.main_theta_vec_proposed();
             velocity_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) =  result_input.main_velocity_0_vec();
             velocity_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) = result_input.main_velocity_vec_proposed();
             
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










     
// --------------------------------- RcpParallel  functions  -- BURNIN fn ------------------------------------------------------------------------------------------------------------------------------------------- 
     
     
 
  
  
class RcppParallel_EHMC_burnin : public RcppParallel::Worker {
    
//private:
//  std::unique_ptr<dqrng::random_64bit_generator> global_rng = std::unique_ptr<dqrng::random_64bit_generator>(dqrng::generator<dqrng::xoshiro256plusplus>(seed)); // seeded from R's RNG
  
public:
       void reset_Eigen() {
         theta_main_vectors_all_chains_input_from_R_RcppPar.resize(0, 0);
         theta_us_vectors_all_chains_input_from_R_RcppPar.resize(0, 0);
       }
       
       // Clear all tbb concurrent vectors
       void reset_tbb() { 
         y_copies.clear();
         Model_args_as_cpp_struct_copies.clear();
         EHMC_args_as_cpp_struct_copies.clear();
         EHMC_Metric_as_cpp_struct_copies.clear();
         EHMC_burnin_as_cpp_struct_copies.clear();
         HMC_outputs.clear();
       }
       
       void reset() {
         reset_tbb();
         reset_Eigen();
       } 
       
       //////////////////// ---- declare variables
       int seed;
       int n_threads;
       int n_iter;
       int current_iter;
       bool partitioned_HMC;
       
       //// Input data (to read)
       Eigen::Matrix<double, -1, -1>  theta_main_vectors_all_chains_input_from_R_RcppPar;
       Eigen::Matrix<double, -1, -1>  theta_us_vectors_all_chains_input_from_R_RcppPar;
       
       //// other main outputs;
       RcppParallel::RMatrix<double>  other_main_out_vector_all_chains_output_to_R_RcppPar;
       RcppParallel::RMatrix<double>  other_us_out_vector_all_chains_output_to_R_RcppPar;
       
       //// data
       tbb::concurrent_vector<Eigen::Matrix<int, -1, -1>>  y_copies;
       
       //////////////  input structs
       tbb::concurrent_vector<Model_fn_args_struct> Model_args_as_cpp_struct_copies;
       tbb::concurrent_vector<EHMC_fn_args_struct>  EHMC_args_as_cpp_struct_copies;
       tbb::concurrent_vector<EHMC_Metric_struct>   EHMC_Metric_as_cpp_struct_copies;
       tbb::concurrent_vector<EHMC_burnin_struct>   EHMC_burnin_as_cpp_struct_copies;
       
       //// other args (read only)
       std::string Model_type;
       bool sample_nuisance; 
       bool force_autodiff;
       bool force_PartialLog;
       bool multi_attempts;
       
       //// local storage 
       tbb::concurrent_vector<HMC_output_single_chain> HMC_outputs;
       
       //////////////// burnin-specific containers (i.e., these containers are not in the corresponding sampling fn)
       //// nuisance outputs
       RcppParallel::RMatrix<double>  theta_main_vectors_all_chains_output_to_R_RcppPar;
       RcppParallel::RMatrix<double>  theta_us_vectors_all_chains_output_to_R_RcppPar;
       //// main
       RcppParallel::RMatrix<double>  theta_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar;
       RcppParallel::RMatrix<double>  theta_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar;
       RcppParallel::RMatrix<double>  velocity_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar;
       RcppParallel::RMatrix<double>  velocity_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar;
       //// nuisance
       RcppParallel::RMatrix<double>  theta_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar; 
       RcppParallel::RMatrix<double>  theta_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar;
       RcppParallel::RMatrix<double>  velocity_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar;
       RcppParallel::RMatrix<double>  velocity_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar;
       
       ////////////// Constructor (initialise these with the SOURCE format)
       RcppParallel_EHMC_burnin(    int &n_threads_R,
                                    int &seed_R,
                                    int &n_iter_R,
                                    int &current_iter_R,
                                    bool &partitioned_HMC_R,
                                    std::string &Model_type_R,
                                    bool &sample_nuisance_R,
                                    bool &force_autodiff_R,
                                    bool &force_PartialLog_R,
                                    bool &multi_attempts_R,
                                    //// inputs
                                    const Eigen::Matrix<double, -1, -1> &theta_main_vectors_all_chains_input_from_R,
                                    const Eigen::Matrix<double, -1, -1> &theta_us_vectors_all_chains_input_from_R,
                                    //// outputs (main)
                                    NumericMatrix &other_main_out_vector_all_chains_output_to_R,
                                    //// outputs (nuisance)
                                    NumericMatrix &other_us_out_vector_all_chains_output_to_R,
                                    //////////////   data
                                    const std::vector<Eigen::Matrix<int, -1, -1>> &y_copies_R,
                                    //////////////  input structs
                                    std::vector<Model_fn_args_struct> &Model_args_as_cpp_struct_copies_R,
                                    std::vector<EHMC_fn_args_struct> &EHMC_args_as_cpp_struct_copies_R,
                                    std::vector<EHMC_Metric_struct> &EHMC_Metric_as_cpp_struct_copies_R,
                                    std::vector<EHMC_burnin_struct> &EHMC_burnin_as_cpp_struct_copies_R,
                                    ///////////////////// burnin-specific stuff
                                    NumericMatrix  &theta_main_vectors_all_chains_output_to_R,
                                    NumericMatrix  &theta_us_vectors_all_chains_output_to_R,
                                    NumericMatrix  &theta_main_0_burnin_tau_adapt_all_chains_input_from_R,
                                    NumericMatrix  &theta_main_prop_burnin_tau_adapt_all_chains_input_from_R,
                                    NumericMatrix  &velocity_main_0_burnin_tau_adapt_all_chains_input_from_R,
                                    NumericMatrix  &velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R,
                                    NumericMatrix  &theta_us_0_burnin_tau_adapt_all_chains_input_from_R,
                                    NumericMatrix  &theta_us_prop_burnin_tau_adapt_all_chains_input_from_R,
                                    NumericMatrix  &velocity_us_0_burnin_tau_adapt_all_chains_input_from_R,
                                    NumericMatrix  &velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R
       )
         
         :
         n_threads(n_threads_R),
         seed(seed_R),
         n_iter(n_iter_R),
         current_iter(current_iter_R),
         partitioned_HMC(partitioned_HMC_R),
         ////////////// inputs
         theta_main_vectors_all_chains_input_from_R_RcppPar(theta_main_vectors_all_chains_input_from_R),
         theta_us_vectors_all_chains_input_from_R_RcppPar(theta_us_vectors_all_chains_input_from_R),
         //////////////  outputs (main)
         other_main_out_vector_all_chains_output_to_R_RcppPar(other_main_out_vector_all_chains_output_to_R),
         //////////////  outputs (nuisance)
         other_us_out_vector_all_chains_output_to_R_RcppPar(other_us_out_vector_all_chains_output_to_R),
         ////////////// other args (read only)
         Model_type(Model_type_R),
         sample_nuisance(sample_nuisance_R),
         force_autodiff(force_autodiff_R),
         force_PartialLog(force_PartialLog_R),
         multi_attempts(multi_attempts_R) ,
         ///////////////////// burnin-specific stuff
         theta_main_vectors_all_chains_output_to_R_RcppPar(theta_main_vectors_all_chains_output_to_R),
         theta_us_vectors_all_chains_output_to_R_RcppPar(theta_us_vectors_all_chains_output_to_R),
         theta_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar(theta_main_0_burnin_tau_adapt_all_chains_input_from_R),
         theta_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar(theta_main_prop_burnin_tau_adapt_all_chains_input_from_R),
         velocity_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar(velocity_main_0_burnin_tau_adapt_all_chains_input_from_R),
         velocity_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar(velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R),
         theta_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar(theta_us_0_burnin_tau_adapt_all_chains_input_from_R),
         theta_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar(theta_us_prop_burnin_tau_adapt_all_chains_input_from_R),
         velocity_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar(velocity_us_0_burnin_tau_adapt_all_chains_input_from_R),
         velocity_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar(velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R)
       { 
         
             ////////////// data 
             y_copies = convert_std_vec_to_concurrent_vector(y_copies_R, y_copies);
             
             //////////////  input structs
             Model_args_as_cpp_struct_copies =  convert_std_vec_to_concurrent_vector(Model_args_as_cpp_struct_copies_R, Model_args_as_cpp_struct_copies);
             EHMC_args_as_cpp_struct_copies =   convert_std_vec_to_concurrent_vector(EHMC_args_as_cpp_struct_copies_R, EHMC_args_as_cpp_struct_copies);
             EHMC_Metric_as_cpp_struct_copies = convert_std_vec_to_concurrent_vector(EHMC_Metric_as_cpp_struct_copies_R, EHMC_Metric_as_cpp_struct_copies);
             EHMC_burnin_as_cpp_struct_copies = convert_std_vec_to_concurrent_vector(EHMC_burnin_as_cpp_struct_copies_R, EHMC_burnin_as_cpp_struct_copies);
             
             const int N = Model_args_as_cpp_struct_copies[0].N;
             const int n_us =  Model_args_as_cpp_struct_copies[0].n_nuisance;
             const int n_params_main = Model_args_as_cpp_struct_copies[0].n_params_main;
             const int n_nuisance_to_track = 1;
             
             HMC_outputs.reserve(n_threads_R);
             for (int i = 0; i < n_threads_R; ++i) {
               HMC_output_single_chain HMC_output_single_chain(n_iter_R, n_nuisance_to_track, n_params_main, n_us, N);
               HMC_outputs.emplace_back(HMC_output_single_chain);
             }
         
       }
       
   ////////////// RcppParallel Parallel operator
   void operator() (std::size_t begin, std::size_t end) {

         std::size_t i = begin;  //// each thread processes only the chain at index `i`
         {
           
           // auto rng = global_rng->clone(i + 1);  // Create thread local copy and advance it by thread number + 1 jumps
           auto rng = dqrng::generator<pcg64>(seed + current_iter, i);
           
           const int N = Model_args_as_cpp_struct_copies[i].N;
           const int n_us =  Model_args_as_cpp_struct_copies[i].n_nuisance;
           const int n_params_main = Model_args_as_cpp_struct_copies[i].n_params_main;
           const int n_params = n_params_main + n_us;
           const bool burnin_indicator = true;
           const int n_nuisance_to_track = 1;
           
           stan::math::ChainableStack ad_tape;
           stan::math::nested_rev_autodiff nested;
           
           
          {
            
            ///////////////////////////////////////// perform iterations for adaptation interval
            HMCResult result_input(n_params_main, n_us, N);
            result_input.main_theta_vec() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
            result_input.main_theta_vec_0() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
            
            if (sample_nuisance == true)  {
              result_input.us_theta_vec() = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
              result_input.us_theta_vec_0() = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
            }
            
            {
              
              //////////////////////////////// perform iterations for chain i
              if (Model_type == "Stan") {  
                
                Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct_copies[i].model_so_file,
                                                                                           Model_args_as_cpp_struct_copies[i].json_file_path, 
                                                                                           seed + i * 1000);
                
                fn_sample_HMC_multi_iter_single_thread(    HMC_outputs[i],
                                                           result_input, 
                                                           burnin_indicator, 
                                                           i, seed + i * 1000, rng, n_iter,
                                                           partitioned_HMC,
                                                           Model_type, sample_nuisance,
                                                           force_autodiff, force_PartialLog,  multi_attempts,  n_nuisance_to_track, 
                                                           y_copies[i], 
                                                           Model_args_as_cpp_struct_copies[i], 
                                                           EHMC_args_as_cpp_struct_copies[i], EHMC_Metric_as_cpp_struct_copies[i], 
                                                           Stan_model_as_cpp_struct);
                //// destroy Stan model object
                fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
                
              } else { 
                
                Stan_model_struct Stan_model_as_cpp_struct; ///  dummy struct
                
                fn_sample_HMC_multi_iter_single_thread(    HMC_outputs[i],
                                                           result_input, 
                                                           burnin_indicator, 
                                                           i, seed + i * 1000, rng, n_iter,
                                                           partitioned_HMC,
                                                           Model_type, sample_nuisance,
                                                           force_autodiff, force_PartialLog,  multi_attempts,  n_nuisance_to_track, 
                                                           y_copies[i], 
                                                           Model_args_as_cpp_struct_copies[i], 
                                                           EHMC_args_as_cpp_struct_copies[i], EHMC_Metric_as_cpp_struct_copies[i], 
                                                           Stan_model_as_cpp_struct);
                
                
              }
              
              
              /////////////////////////////////////////// end of iteration(s)
              if (sample_nuisance == true)  {
                //////// Write results back to the shared array once half-iteration completed
                theta_us_vectors_all_chains_output_to_R_RcppPar.column(i) =         fn_convert_EigenColVec_to_RMatrixColumn(  result_input.us_theta_vec() ,  theta_us_vectors_all_chains_output_to_R_RcppPar.column(i));
                ///// for burnin / ADAM-tau adaptation only
                theta_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn( result_input.us_theta_vec_0(),      theta_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
                theta_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn(  result_input.us_theta_vec_proposed() ,     theta_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
                velocity_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn( result_input.us_velocity_0_vec(),         velocity_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
                velocity_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn( result_input.us_velocity_vec_proposed(),  velocity_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
              }
              
              //////// Write results back to the shared array once half-iteration completed
              theta_main_vectors_all_chains_output_to_R_RcppPar.column(i) =          fn_convert_EigenColVec_to_RMatrixColumn(result_input.main_theta_vec(),  theta_main_vectors_all_chains_output_to_R_RcppPar.column(i));
              ///// for burnin / ADAM-tau adaptation only
              theta_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn( result_input.main_theta_vec_0(),      theta_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
              theta_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn(result_input.main_theta_vec_proposed(),     theta_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
              velocity_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn( result_input.main_velocity_0_vec(),         velocity_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
              velocity_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn( result_input.main_velocity_vec_proposed(),  velocity_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
              
              //////// compute summaries at end of iterations from each chain
              // other outputs (once all iterations finished) - main
              other_main_out_vector_all_chains_output_to_R_RcppPar(0, i) = HMC_outputs[i].diagnostics_p_jump_main().sum() / n_iter;
              other_main_out_vector_all_chains_output_to_R_RcppPar(1, i) = HMC_outputs[i].diagnostics_div_main().sum();
              // other outputs (once all iterations finished) - nuisance
              if (sample_nuisance == true)  {
                other_us_out_vector_all_chains_output_to_R_RcppPar(0, i) =  HMC_outputs[i].diagnostics_p_jump_us().sum() / n_iter;
                other_us_out_vector_all_chains_output_to_R_RcppPar(1, i) =  HMC_outputs[i].diagnostics_div_us().sum();
              }
              
              /////////////////  ---- burnin-specific stuff -----
              // other outputs (once all iterations finished) - main
              ////
              other_main_out_vector_all_chains_output_to_R_RcppPar(2, i) = EHMC_burnin_as_cpp_struct_copies[i].tau_m_adam_main;
              other_main_out_vector_all_chains_output_to_R_RcppPar(3, i) = EHMC_burnin_as_cpp_struct_copies[i].tau_v_adam_main;
              other_main_out_vector_all_chains_output_to_R_RcppPar(4, i) = EHMC_args_as_cpp_struct_copies[i].tau_main;
              other_main_out_vector_all_chains_output_to_R_RcppPar(5, i) = EHMC_args_as_cpp_struct_copies[i].tau_main_ii;
              ////
              other_main_out_vector_all_chains_output_to_R_RcppPar(6, i) = EHMC_burnin_as_cpp_struct_copies[i].eps_m_adam_main;
              other_main_out_vector_all_chains_output_to_R_RcppPar(7, i) = EHMC_burnin_as_cpp_struct_copies[i].eps_v_adam_main;
              other_main_out_vector_all_chains_output_to_R_RcppPar(8, i) = EHMC_args_as_cpp_struct_copies[i].eps_main;
              // other outputs (once all iterations finished) - nuisance
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
              
            } /// end of big local block
            
          }
                 
                     
              }  //// end of all parallel work// Definition of static thread_local variable
                 
                   
  } /// end of void RcppParallel operator
       
    
};












     
 

// --------------------------------- RcpParallel  functions  -- SAMPLING fn ------------------------------------------------------------------------------------------------------------------------------------------- 





 
 
class RcppParallel_EHMC_sampling : public RcppParallel::Worker {
  
//private:
//  std::unique_ptr<dqrng::random_64bit_generator> global_rng = std::unique_ptr<dqrng::random_64bit_generator>(dqrng::generator<dqrng::xoshiro256plusplus>(seed)); // seeded from R's RNG
  
public:
          void reset_Eigen() {
                theta_main_vectors_all_chains_input_from_R_RcppPar.resize(0, 0);
                theta_us_vectors_all_chains_input_from_R_RcppPar.resize(0, 0);
          }
          
          // Clear all tbb concurrent vectors
          void reset_tbb() { 
                y_copies.clear();
                Model_args_as_cpp_struct_copies.clear();
                EHMC_args_as_cpp_struct_copies.clear();
                EHMC_Metric_as_cpp_struct_copies.clear();
                HMC_outputs.clear();
          }
          
          void reset() {
                reset_tbb();
                reset_Eigen();
          } 
          
          //////////////////// ---- declare variables
          const int  seed;
          const int  n_threads;
          const int  n_iter;
          const bool partitioned_HMC;
          
          //// local storage 
          tbb::concurrent_vector<HMC_output_single_chain> HMC_outputs;
          
          //// references to R trace matrices
          std::vector<Rcpp::NumericMatrix> &R_trace_output;
          std::vector<Rcpp::NumericMatrix> &R_trace_divs;
          const int n_nuisance_to_track;
          std::vector<Rcpp::NumericMatrix> &R_trace_nuisance;
          
          //// this only gets used for built-in models, for Stan models log_lik must be defined in the "transformed parameters" block.
          std::vector<Rcpp::NumericMatrix> &R_trace_log_lik;     
          
          //// Input data (to read)
          Eigen::Matrix<double, -1, -1>  theta_main_vectors_all_chains_input_from_R_RcppPar;
          Eigen::Matrix<double, -1, -1>  theta_us_vectors_all_chains_input_from_R_RcppPar;
          
          //// data
          tbb::concurrent_vector<Eigen::Matrix<int, -1, -1>>   y_copies;  
          
          //// input structs
          tbb::concurrent_vector<Model_fn_args_struct>   Model_args_as_cpp_struct_copies;  
          tbb::concurrent_vector<EHMC_fn_args_struct>    EHMC_args_as_cpp_struct_copies;  
          tbb::concurrent_vector<EHMC_Metric_struct>     EHMC_Metric_as_cpp_struct_copies;  
          
          //// other args (read only)
          const std::string Model_type;
          const bool sample_nuisance;
          const bool force_autodiff;
          const bool force_PartialLog;
          const bool multi_attempts; 
          
      
  ////////////// Constructor (initialise these with the SOURCE format)
  RcppParallel_EHMC_sampling(  const int  &n_threads_R,
                               const int  &seed_R,
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
                               //// output (main)
                               std::vector<Rcpp::NumericMatrix> &trace_output,
                               //////////////   data
                               const std::vector<Eigen::Matrix<int, -1, -1>> &y_copies_R,
                               //////////////  input structs
                               const std::vector<Model_fn_args_struct>  &Model_args_as_cpp_struct_copies_R,   // READ-ONLY
                               std::vector<EHMC_fn_args_struct>  &EHMC_args_as_cpp_struct_copies_R,  
                               const std::vector<EHMC_Metric_struct>  &EHMC_Metric_as_cpp_struct_copies_R, // READ-ONLY
                               /////////////
                               std::vector<Rcpp::NumericMatrix> &trace_output_divs,
                               const int &n_nuisance_to_track_R,
                               std::vector<Rcpp::NumericMatrix> &trace_output_nuisance,
                               std::vector<Rcpp::NumericMatrix> &trace_output_log_lik
                               )
    :
    n_threads(n_threads_R),
    seed(seed_R),
    n_iter(n_iter_R),
    partitioned_HMC(partitioned_HMC_R),
    ////////////// inputs
    theta_main_vectors_all_chains_input_from_R_RcppPar(theta_main_vectors_all_chains_input_from_R),
    theta_us_vectors_all_chains_input_from_R_RcppPar(theta_us_vectors_all_chains_input_from_R),
    //////////////  
    Model_type(Model_type_R),
    sample_nuisance(sample_nuisance_R),
    force_autodiff(force_autodiff_R),
    force_PartialLog(force_PartialLog_R),
    multi_attempts(multi_attempts_R) ,
    ///////////// trace outputs
    n_nuisance_to_track(n_nuisance_to_track_R), 
    R_trace_output(trace_output),
    R_trace_divs(trace_output_divs),
    R_trace_nuisance(trace_output_nuisance),
    R_trace_log_lik(trace_output_log_lik)
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
  }

  ////////////// RcppParallel Parallel operator
  void operator() (std::size_t begin, std::size_t end) {
    
    std::size_t i = begin;  // each thread processes only the chain at index `i`
    
    // auto rng = global_rng->clone(i + 1);  // Create thread local copy and advance it by thread number + 1 jumps
    auto rng = dqrng::generator<pcg64>(seed, i);
    
    const int N = Model_args_as_cpp_struct_copies[i].N;
    const int n_us =  Model_args_as_cpp_struct_copies[i].n_nuisance;
    const int n_params_main = Model_args_as_cpp_struct_copies[i].n_params_main;
    const int n_params = n_params_main + n_us;
    
    const bool burnin_indicator = false;
    
    {
    
      static  thread_local stan::math::ChainableStack ad_tape;
      static  thread_local stan::math::nested_rev_autodiff nested;
      
      HMCResult result_input(n_params_main, n_us, N); // JUST putting this as thread_local doesnt fix the "lagging chain 0" issue. 
    
          {
    
              ///////////////////////////////////////// perform iterations for adaptation interval
              result_input.main_theta_vec() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
              result_input.main_theta_vec_0() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
              result_input.us_theta_vec() = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
              result_input.us_theta_vec_0() = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);

              
            if (Model_type == "Stan") {  
 
                          Stan_model_struct  Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct_copies[i].model_so_file,
                                                                                                        Model_args_as_cpp_struct_copies[i].json_file_path, 
                                                                                                        seed + i * 1000);
                    
                         //////////////////////////////// perform iterations for chain i
                         fn_sample_HMC_multi_iter_single_thread(   HMC_outputs[i] ,
                                                                   result_input, 
                                                                   burnin_indicator, 
                                                                   i, seed + i * 1000, rng, n_iter,
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
                          //// destroy Stan model object
                          fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);  
                          
                        
      
            } else  { 
              
                          Stan_model_struct Stan_model_as_cpp_struct; ///  dummy struct
              
                          //////////////////////////////// perform iterations for chain i
                          fn_sample_HMC_multi_iter_single_thread(  HMC_outputs[i],   
                                                                   result_input, 
                                                                   burnin_indicator, 
                                                                   i, seed + i * 1000, rng, n_iter,
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
    for (size_t i = 0; i < n_threads; ++i) {

          // Copy main trace
          for (int ii = 0; ii < HMC_outputs[i].trace_main().cols(); ++ii) {
            for (int param = 0; param < HMC_outputs[i].trace_main().rows(); ++param) {
              R_trace_output[i](param, ii) = (HMC_outputs[i].trace_main()(param, ii));
            }
          }

          // Copy divs
          for (int ii = 0; ii < HMC_outputs[i].trace_div().cols(); ++ii) {
            for (int param = 0; param < HMC_outputs[i].trace_div().rows(); ++param) {
              R_trace_divs[i](param, ii) =  (HMC_outputs[i].trace_div()(param, ii));
            }
          }

          // Copy nuisance  
          if (sample_nuisance) {
            for (int ii = 0; ii < HMC_outputs[i].trace_nuisance().cols(); ++ii) {
              for (int param = 0; param < HMC_outputs[i].trace_nuisance().rows(); ++param) {
                R_trace_nuisance[i](param, ii) =  (HMC_outputs[i].trace_nuisance()(param, ii));
              }
            }
          }

          // Copy log-lik   (for built-in models only)
          if (Model_type != "Stan") {
            for (int ii = 0; ii < HMC_outputs[i].trace_log_lik().cols(); ++ii) {
              for (int param = 0; param < HMC_outputs[i].trace_log_lik().rows(); ++param) {
                R_trace_log_lik[i](param, ii) =   (HMC_outputs[i].trace_log_lik()(param, ii));
              }
            }
          }

    }
  }



};



 
 




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
                              std::vector<Rcpp::NumericMatrix> &trace_output,
                              //// data
                              const std::vector<Eigen::Matrix<int, -1, -1>>   &y_copies,
                              //////////////  input structs
                              std::vector<Model_fn_args_struct> &Model_args_as_cpp_struct_copies,
                              std::vector<EHMC_fn_args_struct>  &EHMC_args_as_cpp_struct_copies,
                              std::vector<EHMC_Metric_struct>   &EHMC_Metric_as_cpp_struct_copies,
                              /////////////
                              std::vector<Rcpp::NumericMatrix> &trace_divs,
                              const int &n_nuisance_to_track_R,
                              std::vector<Rcpp::NumericMatrix> &trace_nuisance,
                              std::vector<Rcpp::NumericMatrix> &trace_log_lik
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
  
  omp_set_num_threads(n_threads);
  omp_set_dynamic(0);
  
  // auto global_rng = dqrng::generator<dqrng::xoshiro256plusplus>(seed); // base RNG
  
  //// parallel for-loop
  #pragma omp parallel for shared(HMC_outputs)
  for (int i = 0; i < n_threads; i++) {  
        
        // auto rng = global_rng->clone(i + 1);  // Create thread local copy and advance it by thread number + 1 jumps
        auto rng = dqrng::generator<pcg64>(seed, i);
    
        const int N =  Model_args_as_cpp_struct_copies[i].N;
        const int n_us =  Model_args_as_cpp_struct_copies[i].n_nuisance;
        const int n_params_main = Model_args_as_cpp_struct_copies[i].n_params_main;
        const int n_params = n_params_main + n_us;
        
        const bool burnin_indicator = false;
        const int n_nuisance_to_track = 1;
        
        static thread_local stan::math::ChainableStack ad_tape;
        static thread_local stan::math::nested_rev_autodiff nested;
        
        ///////////////////////////////////////// perform iterations for adaptation interval
        HMCResult result_input(n_params_main, n_us, N);
        result_input.main_theta_vec() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
        result_input.main_theta_vec_0() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
        
        if (sample_nuisance == true)  {
          result_input.us_theta_vec() = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
          result_input.us_theta_vec_0() = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
        }
        
          //////////////////////////////// perform iterations for chain i
          if (Model_type == "Stan") {  
            
            Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct_copies[i].model_so_file,
                                                                                       Model_args_as_cpp_struct_copies[i].json_file_path, 
                                                                                       seed + i * 1000);
            
            fn_sample_HMC_multi_iter_single_thread(    HMC_outputs[i],
                                                       result_input, 
                                                       burnin_indicator, 
                                                       i, seed + i * 1000, rng, n_iter,
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
                                                       result_input, 
                                                       burnin_indicator, 
                                                       i, seed + i * 1000, rng, n_iter,
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



 





  