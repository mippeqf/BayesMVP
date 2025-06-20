#pragma once

 

using namespace Rcpp;
using namespace Eigen; 

 

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <tbb/concurrent_vector.h>

 
 
#include <chrono> 
#include <unordered_map>
#include <memory>
#include <thread>
#include <functional>



 
 
// ANSI codes for different colors
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"

 
 

 
static std::mutex print_mutex; //// global mutex 

 
   
   
   
   

void                    fn_sample_HMC_multi_iter_single_thread(                    HMC_output_single_chain &HMC_output_single_chain_i,
                                                                                                  HMCResult &result_input,
                                                                                                  const bool burnin_indicator,
                                                                                                  const int chain_id,
                                                                                                  const int seed,
                                                                                                  std::mt19937 &rng,
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
                     
                     //// reset initial theta's
                     result_input.main_theta_vec_0 =  result_input.main_theta_vec;
                     result_input.us_theta_vec_0 =  result_input.us_theta_vec;
           
                     // if (burnin_indicator == false) {
                     //     if (ii %  static_cast<int>(std::round(static_cast<double>(n_iter)/4.0)) == 0) {
                     //         std::lock_guard<std::mutex> lock(print_mutex);
                     //         double pct_complete = 100.0 * (static_cast<double>(ii) / static_cast<double>(n_iter));
                     //         std::cout << "Chain #" << chain_id << " - Sampling is around " << pct_complete << " % complete" << "\n";
                     //     }
                     // }
                     
                     
                     if (partitioned_HMC == true) {
                       
                               stan::math::start_nested();
                   
                               //////////////////////////////////////// sample nuisance (GIVEN main)
                               if (sample_nuisance == true)   {
                                     
                                           fn_Diffusion_HMC_nuisance_only_single_iter_InPlace_process(    result_input,    
                                                                                                          burnin,  rng, seed,
                                                                                                          Model_type, 
                                                                                                          force_autodiff, force_PartialLog,  multi_attempts, 
                                                                                                          y_Eigen_i,
                                                                                                          Model_args_as_cpp_struct, // MVP_workspace, 
                                                                                                          EHMC_args_as_cpp_struct, EHMC_Metric_as_cpp_struct, 
                                                                                                          Stan_model_as_cpp_struct);
                                        
                                         
                                           HMC_output_single_chain_i.diagnostics.p_jump_us(ii) =  result_input.us_p_jump;
                                           HMC_output_single_chain_i.diagnostics.div_us(ii) =  result_input.us_div;
                                   
                                 } /// end of nuisance-part of iteration
                                 
                                 { /// sample main GIVEN u's
                                     
                                         fn_standard_HMC_main_only_single_iter_InPlace_process(      result_input,   
                                                                                                     burnin,  rng, seed,
                                                                                                     Model_type,  
                                                                                                     force_autodiff, force_PartialLog,  multi_attempts,
                                                                                                     y_Eigen_i,
                                                                                                     Model_args_as_cpp_struct, //MVP_workspace,
                                                                                                     EHMC_args_as_cpp_struct, EHMC_Metric_as_cpp_struct, 
                                                                                                     Stan_model_as_cpp_struct);
                                
                                       
                                         HMC_output_single_chain_i.diagnostics.p_jump_main(ii) =  result_input.main_p_jump;
                                         HMC_output_single_chain_i.diagnostics.div_main(ii) =  result_input.main_div;
                                   
                                 } /// end of main_params part of iteration
                               
                                 stan::math::recover_memory_nested(); 
                     
                     } else {  //// sample all params at once 
                                         stan::math::start_nested();
                                         fn_standard_HMC_dual_single_iter_InPlace_process(    result_input,    
                                                                                              burnin,  rng, seed,
                                                                                              Model_type, 
                                                                                              force_autodiff, force_PartialLog,  multi_attempts, 
                                                                                              y_Eigen_i,
                                                                                              Model_args_as_cpp_struct,  //MVP_workspace, 
                                                                                              EHMC_args_as_cpp_struct, EHMC_Metric_as_cpp_struct, 
                                                                                              Stan_model_as_cpp_struct);
                                         stan::math::recover_memory_nested(); 
                                         
                                         HMC_output_single_chain_i.diagnostics.p_jump_us(ii) =  result_input.us_p_jump;
                                         HMC_output_single_chain_i.diagnostics.div_us(ii) =  result_input.us_div;
                                         HMC_output_single_chain_i.diagnostics.p_jump_main(ii) =  result_input.main_p_jump;
                                         HMC_output_single_chain_i.diagnostics.div_main(ii) =  result_input.main_div;
                     }
                       
                       // Perform MCMC sampling for the i-th chain and store the results in the thread-local buffer
                       HMC_output_single_chain_i.traces.main.col(ii) = result_input.main_theta_vec;// .cast<float>() ;
                     
                       if (sample_nuisance == true) {
                            HMC_output_single_chain_i.traces.div(0, ii) =  (0.50 * (result_input.main_div + result_input.us_div));  
                            HMC_output_single_chain_i.traces.nuisance.col(ii) = result_input.us_theta_vec; /// .cast<float>();
                       } else { 
                            HMC_output_single_chain_i.traces.div(0, ii) = result_input.main_div; 
                       }
                       
                       // if (Model_type != "Stan") {
                       //   HMC_output_single_chain_i.traces.log_lik.col(ii) = result_input.lp_and_grad_outs.tail(N).cast<float>(); 
                       // }
                       

                       
                       //// reset initial theta's
                       result_input.main_theta_vec_0 =  result_input.main_theta_vec;
                       result_input.us_theta_vec_0 =  result_input.us_theta_vec;

           
         } ////////////////////// end of iteration(s)
         
         
    HMC_output_single_chain_i.result_input = result_input;
    
   /// return HMC_output_single_chain_i;
     
   }
     
     
     
     
     
     
     
     
     
     

     
     
     
     
// --------------------------------- OpenMP  functions  -- BURNIN fn ------------------------------------------------------------------------------------------------------------------------------------------- 
     
     
     
     
////// OpenMP -- instead of a Worker struct, make it a function
void EHMC_burnin_OpenMP(    const int  n_threads,
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
       
       //  #ifdef _OPENMP
       omp_set_num_threads(n_threads);
       omp_set_dynamic(0);     
       //  #endif
       
       //// parallel for-loop
       ///#pragma omp parallel for
       for (int i = 0; i < n_threads; i++) {   
         
             const int N =  Model_args_as_cpp_struct_copies[i].N;
             const int n_us =  Model_args_as_cpp_struct_copies[i].n_nuisance;
             const int n_params_main = Model_args_as_cpp_struct_copies[i].n_params_main;
             const int n_params = n_params_main + n_us;
             const bool burnin_indicator = false;
             const int n_nuisance_to_track = 1;
             
             thread_local stan::math::ChainableStack ad_tape;
             //stan::math::nested_rev_autodiff nested;
             thread_local std::mt19937 rng(static_cast<unsigned int>(seed + 1000*i));
         
         {
           
           {
             
             ///////////////////////////////////////// perform iterations for adaptation interval
             HMCResult result_input(n_params_main, n_us, N);
             result_input.main_theta_vec = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
             result_input.main_theta_vec_0 = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
             
             if (sample_nuisance == true)  {
               result_input.us_theta_vec = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
               result_input.us_theta_vec_0 = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
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
               theta_us_vectors_all_chains_output_to_R_RcppPar.col(i) =         result_input.us_theta_vec;
               //// for burnin / ADAM-tau adaptation only
               theta_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) =  result_input.us_theta_vec_0;
               theta_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) = result_input.us_theta_vec_proposed ;
               velocity_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) = result_input.us_velocity_0_vec;
               velocity_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) =  result_input.us_velocity_vec_proposed;
             }
             
             //////// Write results back to the shared array once half-iteration completed
             theta_main_vectors_all_chains_output_to_R_RcppPar.col(i) =    result_input.main_theta_vec;
             ///// for burnin / ADAM-tau adaptation only
             theta_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) =  result_input.main_theta_vec_0;
             theta_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) = result_input.main_theta_vec_proposed;
             velocity_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) =  result_input.main_velocity_0_vec;
             velocity_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.col(i) = result_input.main_velocity_vec_proposed;
             
             //////// compute summaries at end of iterations from each chain
             //// other outputs (once all iterations finished) - main
             other_main_out_vector_all_chains_output_to_R_RcppPar(0, i) = HMC_outputs[i].diagnostics.p_jump_main.sum() / n_iter;
             other_main_out_vector_all_chains_output_to_R_RcppPar(1, i) = HMC_outputs[i].diagnostics.div_main.sum();
             //// other outputs (once all iterations finished) - nuisance
             if (sample_nuisance == true)  {
               other_us_out_vector_all_chains_output_to_R_RcppPar(0, i) =  HMC_outputs[i].diagnostics.p_jump_us.sum() / n_iter;
               other_us_out_vector_all_chains_output_to_R_RcppPar(1, i) =  HMC_outputs[i].diagnostics.div_us.sum();
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
     
     
     
     
struct RcppParallel_EHMC_burnin: public RcppParallel::Worker {
       
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
           
           const int N = Model_args_as_cpp_struct_copies[i].N;
           const int n_us =  Model_args_as_cpp_struct_copies[i].n_nuisance;
           const int n_params_main = Model_args_as_cpp_struct_copies[i].n_params_main;
           const int n_params = n_params_main + n_us;
           const bool burnin_indicator = true;
           const int n_nuisance_to_track = 1;
           
           // RNG not thread_local to avoid Windows TLS issues
#ifdef _WIN32   // Windows 
           stan::math::ChainableStack ad_tape;
           stan::math::nested_rev_autodiff nested;
           std::mt19937 rng(static_cast<unsigned int>(seed + i * 1000));
#else  // Linux version 
           thread_local stan::math::ChainableStack ad_tape;
           thread_local stan::math::nested_rev_autodiff nested;
           thread_local std::mt19937 rng(static_cast<unsigned int>(seed + i * 1000)); // Declare and initialize in one line
#endif
           
{
  
  ///////////////////////////////////////// perform iterations for adaptation interval
  HMCResult result_input(n_params_main, n_us, N);
  result_input.main_theta_vec = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
  result_input.main_theta_vec_0 = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
  
  if (sample_nuisance == true)  {
    result_input.us_theta_vec = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
    result_input.us_theta_vec_0 = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
  }
  
  {
    
    //////////////////////////////// perform iterations for chain i
    if (Model_type == "Stan") {  
      
      thread_local Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct_copies[i].model_so_file,
                                                                                 Model_args_as_cpp_struct_copies[i].json_file_path, 
                                                                                 seed + i * 1000);
      
      fn_sample_HMC_multi_iter_single_thread(    HMC_outputs[i],
                                                 result_input, 
                                                 burnin_indicator, i, seed + i * 1000, rng, n_iter,
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
      
      thread_local Stan_model_struct Stan_model_as_cpp_struct; ///  dummy struct
      
      fn_sample_HMC_multi_iter_single_thread(    HMC_outputs[i],
                                                 result_input, 
                                                 burnin_indicator, i, seed + i * 1000, rng, n_iter,
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
      theta_us_vectors_all_chains_output_to_R_RcppPar.column(i) =         fn_convert_EigenColVec_to_RMatrixColumn(  result_input.us_theta_vec ,  theta_us_vectors_all_chains_output_to_R_RcppPar.column(i));
      ///// for burnin / ADAM-tau adaptation only
      theta_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn( result_input.us_theta_vec_0,      theta_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
      theta_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn(  result_input.us_theta_vec_proposed ,     theta_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
      velocity_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn( result_input.us_velocity_0_vec,         velocity_us_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
      velocity_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn( result_input.us_velocity_vec_proposed,  velocity_us_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
    }
    
    //////// Write results back to the shared array once half-iteration completed
    theta_main_vectors_all_chains_output_to_R_RcppPar.column(i) =          fn_convert_EigenColVec_to_RMatrixColumn(result_input.main_theta_vec,  theta_main_vectors_all_chains_output_to_R_RcppPar.column(i));
    ///// for burnin / ADAM-tau adaptation only
    theta_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn( result_input.main_theta_vec_0,      theta_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
    theta_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn(result_input.main_theta_vec_proposed,     theta_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
    velocity_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn( result_input.main_velocity_0_vec,         velocity_main_0_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
    velocity_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i) = fn_convert_EigenColVec_to_RMatrixColumn( result_input.main_velocity_vec_proposed,  velocity_main_prop_burnin_tau_adapt_all_chains_output_to_R_RcppPar.column(i));
    
    //////// compute summaries at end of iterations from each chain
    // other outputs (once all iterations finished) - main
    other_main_out_vector_all_chains_output_to_R_RcppPar(0, i) = HMC_outputs[i].diagnostics.p_jump_main.sum() / n_iter;
    other_main_out_vector_all_chains_output_to_R_RcppPar(1, i) = HMC_outputs[i].diagnostics.div_main.sum();
    // other outputs (once all iterations finished) - nuisance
    if (sample_nuisance == true)  {
      other_us_out_vector_all_chains_output_to_R_RcppPar(0, i) =  HMC_outputs[i].diagnostics.p_jump_us.sum() / n_iter;
      other_us_out_vector_all_chains_output_to_R_RcppPar(1, i) =  HMC_outputs[i].diagnostics.div_us.sum();
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





 
 
struct RcppParallel_EHMC_sampling : public RcppParallel::Worker {

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
    
    const int N = Model_args_as_cpp_struct_copies[i].N;
    const int n_us =  Model_args_as_cpp_struct_copies[i].n_nuisance;
    const int n_params_main = Model_args_as_cpp_struct_copies[i].n_params_main;
    const int n_params = n_params_main + n_us;
    
    const bool burnin_indicator = false;
    
    {
    
      static thread_local stan::math::ChainableStack ad_tape;
      static thread_local stan::math::nested_rev_autodiff nested;
      
      thread_local std::mt19937 rng(static_cast<unsigned int>(seed + i + 1));
      
      HMCResult result_input(n_params_main, n_us, N); // JUST putting this as thread_local doesnt fix the "lagging chain 0" issue. 
      
      const Eigen::Matrix<int, -1, -1> &y_copy = y_copies[i]; /// make local copy of y
      const Model_fn_args_struct &Model_args_as_cpp_struct = Model_args_as_cpp_struct_copies[i];
      const EHMC_Metric_struct &EHMC_Metric_as_cpp_struct = EHMC_Metric_as_cpp_struct_copies[i];
    
          {
    
              ///////////////////////////////////////// perform iterations for adaptation interval
              result_input.main_theta_vec = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
              result_input.main_theta_vec_0 = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
              result_input.us_theta_vec = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
              result_input.us_theta_vec_0 = theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);

              
            if (Model_type == "Stan") {  
                
                            // //// For Stan models:  Initialize bs_model* pointer and void* handle
                            Stan_model_struct  Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct.model_so_file,
                                                                                                                     Model_args_as_cpp_struct.json_file_path, 
                                                                                                                     seed + i);
                          //  }
                      
                         //////////////////////////////// perform iterations for chain i
                         fn_sample_HMC_multi_iter_single_thread(   HMC_outputs[i] ,
                                                                   result_input, 
                                                                   burnin_indicator, 
                                                                   i, seed + i + 1, rng, n_iter,
                                                                   partitioned_HMC,
                                                                   Model_type,  sample_nuisance,
                                                                   force_autodiff, force_PartialLog,  multi_attempts,  
                                                                   n_nuisance_to_track, 
                                                                   y_copy, 
                                                                   Model_args_as_cpp_struct,  
                                                                   EHMC_args_as_cpp_struct_copies[i],
                                                                   EHMC_Metric_as_cpp_struct, 
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
                                                                   i, seed + i + 1, rng, n_iter,
                                                                   partitioned_HMC,
                                                                   Model_type,  sample_nuisance, 
                                                                   force_autodiff, force_PartialLog,  multi_attempts,  
                                                                   n_nuisance_to_track, 
                                                                   y_copy, 
                                                                   Model_args_as_cpp_struct, 
                                                                   EHMC_args_as_cpp_struct_copies[i], 
                                                                   EHMC_Metric_as_cpp_struct, 
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
          for (int ii = 0; ii < HMC_outputs[i].traces.main.cols(); ++ii) {
            for (int param = 0; param < HMC_outputs[i].traces.main.rows(); ++param) {
              R_trace_output[i](param, ii) = std::move(HMC_outputs[i].traces.main(param, ii));
            }
          }

          // Copy divs
          for (int ii = 0; ii < HMC_outputs[i].traces.div.cols(); ++ii) {
            for (int param = 0; param < HMC_outputs[i].traces.div.rows(); ++param) {
              R_trace_divs[i](param, ii) = std::move(HMC_outputs[i].traces.div(param, ii));
            }
          }

          // Copy nuisance if needed
          if (sample_nuisance) {
            for (int ii = 0; ii < HMC_outputs[i].traces.nuisance.cols(); ++ii) {
              for (int param = 0; param < HMC_outputs[i].traces.nuisance.rows(); ++param) {
                R_trace_nuisance[i](param, ii) = std::move(HMC_outputs[i].traces.nuisance(param, ii));
              }
            }
          }

          // Copy log-lik if needed (for built-in models only)
          if (Model_type != "Stan") {
            for (int ii = 0; ii < HMC_outputs[i].traces.log_lik.cols(); ++ii) {
              for (int param = 0; param < HMC_outputs[i].traces.log_lik.rows(); ++param) {
                R_trace_log_lik[i](param, ii) =  std::move(HMC_outputs[i].traces.log_lik(param, ii));
              }
            }
          }

    }
  }



};



 
//// std::atomic<bool> RcppParallel_EHMC_sampling::thread_locals_need_reset{false};













  