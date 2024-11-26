

// [[Rcpp::depends(StanHeaders)]]  
// [[Rcpp::depends(BH)]] 
// [[Rcpp::depends(RcppParallel)]] 
// [[Rcpp::depends(RcppEigen)]]
   
 


#include "eigen_config.hpp"
 
 
#include "pch.hpp" 
  


#if defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#else
#define ALWAYS_INLINE inline
#endif
 

 
// #include "bridgestan.h"
// #include "version.hpp"
// #include "model_rng.hpp" 
 
 
   
#include <sstream>
#include <stdexcept>    
#include <complex>
#include <dlfcn.h> // For dynamic loading  
#include <map>
#include <vector>   
#include <string>  
#include <stdexcept>
#include <stdio.h>  
#include <iostream>
#include <algorithm>
#include <cmath>
   
     
#include <stan/model/model_base.hpp>  
  
#include <stan/io/array_var_context.hpp> 
#include <stan/io/var_context.hpp> 
#include <stan/io/dump.hpp>  
  
   
#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/rapidjson_parser.hpp>   
   


   
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
   
   
// #if __has_include("omp.h")
//     #include "omp.h"  
// #endif
    
    
    
    
// //// determine vect_type to use
// #if defined(__AVX__) && ( !(defined(__AVX2__)  && defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)) ) // use AVX2
//     static const std::string VECT = "AVX";
// #elif defined(__AVX2__) && ( !(defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)) ) // use AVX2
//     static const std::string VECT = "AVX2";
// #elif defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__) // use AVX-512
//     static const std::string VECT = "AVX512"; 
// #else
//     static const std::string VECT = "Stan";
// #endif
 

 
    
  
///// General functions (e.g. fast exp() and log() approximations). Most of these are not model-specific. 
#include "general_functions/var_fns.hpp"
#include "general_functions/double_fns.hpp" 


#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_Stan.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_Loop.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fast_and_approx_AVX2_fns.hpp" // will only compile if  AVX2 is available
#include "general_functions/fns_SIMD_and_wrappers/fast_and_approx_AVX512_fns.hpp" // will only compile if  AVX-512 is available 
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_SIMD_AVX2.hpp" // will only compile if AVX2 is available 
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_SIMD_AVX512.hpp" // will only compile if AVX-512 is available 
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_overall.hpp" 
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_log_sum_exp_dbl.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_log_sum_exp_SIMD.hpp"


#include "general_functions/array_creators_Eigen_fns.hpp"
#include "general_functions/array_creators_other_fns.hpp"
#include "general_functions/structures.hpp"

#include "general_functions/misc_helper_fns_1.hpp"
#include "general_functions/misc_helper_fns_2.hpp"
#include "general_functions/compute_diagnostics.hpp"

//////// #include "BayesMVP_Stan_fast_approx_fns.hpp"

////// Now load in (mostly) MVP-specific (and MVP-LC) model functions
#include "MVP_functions/MVP_manual_grad_calc_fns.hpp" 
#include "MVP_functions/MVP_log_scale_grad_calc_fns.hpp"
#include "MVP_functions/MVP_manual_trans_and_J_fns.hpp"
#include "MVP_functions/MVP_lp_grad_AD_fns.hpp"
#include "MVP_functions/MVP_lp_grad_MD_AD_fns.hpp"
#include "MVP_functions/MVP_lp_grad_log_scale_MD_AD_fns.hpp"
#include "MVP_functions/MVP_lp_grad_multi_attempts.hpp"
//////// #include "MVP_functions/MVP_manual_Hessian_calc_fns.hpp"


#define COMPILE_LATENT_TRAIT 0
#define COMPILE_MCMC_MAIN 1

#if COMPILE_LATENT_TRAIT
    ////// Latent trait stuff
    #include "LC_LT_functions/LC_LT_manual_grad_calc_fns.hpp"
    #include "LC_LT_functions/LC_LT_log_scale_grad_calc_fns.hpp"
    #include "LC_LT_functions/LC_LT_lp_grad_AD_fns.hpp"
    #include "LC_LT_functions/LC_LT_lp_grad_MD_AD_fns.hpp"
    #include "LC_LT_functions/LC_LT_lp_grad_log_scale_MD_AD_fns.hpp"
    #include "LC_LT_functions/LT_LC_lp_grad_multi_attempts.hpp"
#endif

    
    

////// general lp_grad fn / manual/Stan model selector
#if __has_include("bridgestan.h")
    #define HAS_BRIDGESTAN_H 1
    #include "bridgestan.h"
    #include "version.hpp"
    #include "model_rng.hpp"
#else
    #define HAS_BRIDGESTAN_H 0
#endif


// #undef HAS_BRIDGESTAN_H //// override
// #define HAS_BRIDGESTAN_H 0 //// override

 


#if HAS_BRIDGESTAN_H
    #include "general_functions/Stan_model_helper_fns.hpp"
#endif


    
    
#include "general_functions/lp_grad_model_selector.hpp"


////////// ADAM / SNAPER-HMC general functions
#include "general_functions/MCMC/EHMC_adapt_eps_fn.hpp"
#include "general_functions/MCMC/EHMC_adapt_tau_fns.hpp"
#include "general_functions/MCMC/EHMC_adapt_M_Hessian_fns.hpp"

////////// EHMC sampler functions
#if COMPILE_MCMC_MAIN
  #include "general_functions/MCMC/EHMC_main_sampler_fns.hpp"
  #include "general_functions/MCMC/EHMC_nuisance_sampler_fns.hpp"
  #include "general_functions/MCMC/EHMC_dual_sampler_fns.hpp"
  #include "general_functions/MCMC/EHMC_find_initial_eps_fns.hpp"
  #include "general_functions/MCMC/EHMC_multithreaded_samp_fns.hpp"
#endif


 
 
 
 


#include <dlfcn.h> // For dynamic loading

 
 
using namespace Rcpp;
using namespace Eigen;




#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE


using std_vec_of_EigenVecs_dbl = std::vector<Eigen::Matrix<double, -1, 1>>;
using std_vec_of_EigenVecs_int = std::vector<Eigen::Matrix<int, -1, 1>>;
using std_vec_of_EigenMats_dbl = std::vector<Eigen::Matrix<double, -1, -1>>;
using std_vec_of_EigenMats_int = std::vector<Eigen::Matrix<int, -1, -1>>;

using two_layer_std_vec_of_EigenVecs_dbl = std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>;
using two_layer_std_vec_of_EigenVecs_int = std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>;
using two_layer_std_vec_of_EigenMats_dbl = std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>;
using two_layer_std_vec_of_EigenMats_int = std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>;


using three_layer_std_vec_of_EigenVecs_dbl =  std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>>;
using three_layer_std_vec_of_EigenVecs_int =  std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>>;
using three_layer_std_vec_of_EigenMats_dbl = std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>>; 
using three_layer_std_vec_of_EigenMats_int = std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>>;
 

 
 
 
// ANSI codes for different colors
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"









Model_fn_args_struct   convert_R_List_to_Model_fn_args_struct(Rcpp::List R_List) {

     // handles empty fields by using defaults
     const int N = R_List["N"];
     const int n_nuisance = R_List["n_nuisance"];
     const int n_params_main = R_List["n_params_main"];
    
     const std::string model_so_file = R_List["model_so_file"];
     const std::string json_file_path = R_List["json_file_path"];
    
     const Eigen::Matrix<bool, -1, 1> Model_args_bools = Rcpp::as<Eigen::Matrix<bool, -1, 1>>(R_List.containsElementNamed("Model_args_bools") ? R_List["Model_args_bools"] : Rcpp::LogicalMatrix(0));
     const Eigen::Matrix<int, -1, 1> Model_args_ints = Rcpp::as<Eigen::Matrix<int, -1, 1>>(R_List.containsElementNamed("Model_args_ints") ? R_List["Model_args_ints"] : Rcpp::IntegerMatrix(0));
     const Eigen::Matrix<double, -1, 1> Model_args_doubles = Rcpp::as<Eigen::Matrix<double, -1, 1>>(R_List.containsElementNamed("Model_args_doubles") ? R_List["Model_args_doubles"] : Rcpp::NumericMatrix(0));
     const Eigen::Matrix<std::string, -1, 1> Model_args_strings = Rcpp::as<Eigen::Matrix<std::string, -1, 1>>(R_List.containsElementNamed("Model_args_strings") ? R_List["Model_args_strings"] : Rcpp::StringMatrix(0));
    
     const std_vec_of_EigenVecs_dbl Model_args_col_vecs_double = Rcpp::as<std_vec_of_EigenVecs_dbl>(R_List.containsElementNamed("Model_args_col_vecs_double") ? R_List["Model_args_col_vecs_double"] : Rcpp::List(0));
     const std_vec_of_EigenVecs_int Model_args_col_vecs_int = Rcpp::as<std_vec_of_EigenVecs_int>(R_List.containsElementNamed("Model_args_col_vecs_int") ? R_List["Model_args_col_vecs_int"] : Rcpp::List(0));
     const std_vec_of_EigenMats_dbl Model_args_mats_double = Rcpp::as<std_vec_of_EigenMats_dbl>(R_List.containsElementNamed("Model_args_mats_double") ? R_List["Model_args_mats_double"] : Rcpp::List(0));
     const std_vec_of_EigenMats_int Model_args_mats_int = Rcpp::as<std_vec_of_EigenMats_int>(R_List.containsElementNamed("Model_args_mats_int") ? R_List["Model_args_mats_int"] : Rcpp::List(0));
    
     const two_layer_std_vec_of_EigenVecs_dbl Model_args_vecs_of_col_vecs_double = Rcpp::as<two_layer_std_vec_of_EigenVecs_dbl>(R_List.containsElementNamed("Model_args_vecs_of_col_vecs_double") ? R_List["Model_args_vecs_of_col_vecs_double"] : Rcpp::List(0));
     const two_layer_std_vec_of_EigenVecs_int Model_args_vecs_of_col_vecs_int = Rcpp::as<two_layer_std_vec_of_EigenVecs_int>(R_List.containsElementNamed("Model_args_vecs_of_col_vecs_int") ? R_List["Model_args_vecs_of_col_vecs_int"] : Rcpp::List(0));
     const two_layer_std_vec_of_EigenMats_dbl Model_args_vecs_of_mats_double = Rcpp::as<two_layer_std_vec_of_EigenMats_dbl>(R_List.containsElementNamed("Model_args_vecs_of_mats_double") ? R_List["Model_args_vecs_of_mats_double"] : Rcpp::List(0));
     const two_layer_std_vec_of_EigenMats_int Model_args_vecs_of_mats_int = Rcpp::as<two_layer_std_vec_of_EigenMats_int>(R_List.containsElementNamed("Model_args_vecs_of_mats_int") ? R_List["Model_args_vecs_of_mats_int"] : Rcpp::List(0));
    
     const three_layer_std_vec_of_EigenVecs_dbl Model_args_2_later_vecs_of_col_vecs_double = Rcpp::as<three_layer_std_vec_of_EigenVecs_dbl>(R_List.containsElementNamed("Model_args_2_later_vecs_of_col_vecs_double") ? R_List["Model_args_2_later_vecs_of_col_vecs_double"] : Rcpp::List(0));
     const three_layer_std_vec_of_EigenVecs_int Model_args_2_later_vecs_of_col_vecs_int = Rcpp::as<three_layer_std_vec_of_EigenVecs_int>(R_List.containsElementNamed("Model_args_2_later_vecs_of_col_vecs_int") ? R_List["Model_args_2_later_vecs_of_col_vecs_int"] : Rcpp::List(0));
     const three_layer_std_vec_of_EigenMats_dbl Model_args_2_later_vecs_of_mats_double = Rcpp::as<three_layer_std_vec_of_EigenMats_dbl>(R_List.containsElementNamed("Model_args_2_later_vecs_of_mats_double") ? R_List["Model_args_2_later_vecs_of_mats_double"] : Rcpp::List(0));
     const three_layer_std_vec_of_EigenMats_int Model_args_2_later_vecs_of_mats_int = Rcpp::as<three_layer_std_vec_of_EigenMats_int>(R_List.containsElementNamed("Model_args_2_later_vecs_of_mats_int") ? R_List["Model_args_2_later_vecs_of_mats_int"] : Rcpp::List(0));
    
     return Model_fn_args_struct(
       N,
       n_nuisance,
       n_params_main,
       model_so_file,
       json_file_path,
       Model_args_bools,
       Model_args_ints,
       Model_args_doubles,
       Model_args_strings,
       Model_args_col_vecs_double,
       Model_args_col_vecs_int,
       Model_args_mats_double,
       Model_args_mats_int,
       Model_args_vecs_of_col_vecs_double,
       Model_args_vecs_of_col_vecs_int,
       Model_args_vecs_of_mats_double,
       Model_args_vecs_of_mats_int,
       Model_args_2_later_vecs_of_col_vecs_double,
       Model_args_2_later_vecs_of_col_vecs_int,
       Model_args_2_later_vecs_of_mats_double,
       Model_args_2_later_vecs_of_mats_int
     );
     
}








//////  Function to convert from R List -> C++ struct (so can call fn's from R without having to use Rcpp::List as not thread-safe, and slower etc)
EHMC_Metric_struct convert_R_List_EHMC_Metric_struct(const Rcpp::List &R_List) {

  const Eigen::Matrix<double, -1, -1> M_dense_main = Rcpp::as<Eigen::Matrix<double, -1, -1>>(R_List["M_dense_main"]);
  const Eigen::Matrix<double, -1, -1> M_inv_dense_main = Rcpp::as<Eigen::Matrix<double, -1, -1>>(R_List["M_inv_dense_main"]);
  const Eigen::Matrix<double, -1, -1> M_inv_dense_main_chol = Rcpp::as<Eigen::Matrix<double, -1, -1>>(R_List["M_inv_dense_main_chol"]);
  
  const Eigen::Matrix<double, -1, 1>  M_inv_main_vec = Rcpp::as<Eigen::Matrix<double, -1, 1>>(R_List["M_inv_main_vec"]);
  const Eigen::Matrix<double, -1, 1>  M_inv_us_vec = Rcpp::as<Eigen::Matrix<double, -1, 1>>(R_List["M_inv_us_vec"]);
  const Eigen::Matrix<double, -1, 1>  M_us_vec = Rcpp::as<Eigen::Matrix<double, -1, 1>>(R_List["M_us_vec"]);
  
  const std::string metric_shape_main = R_List["metric_shape_main"]  ;
  
  return EHMC_Metric_struct(M_dense_main, 
                            M_inv_dense_main,
                            M_inv_dense_main_chol, 
                            M_inv_main_vec,
                            M_inv_us_vec, 
                            M_us_vec,
                            metric_shape_main);
}
 




 
 
 
// /////  Function to convert from R List -> C++ struct (so can call fn's from R without having to use Rcpp::List as not thread-safe, and slower etc)
EHMC_fn_args_struct convert_R_List_EHMC_fn_args_struct(Rcpp::List R_List) {

        // Convert the R list elements to the appropriate C++ types
        // for main params
        double tau_main = (R_List["tau_main"]);
        double tau_main_ii = (R_List["tau_main_ii"]);
        double eps_main = (R_List["eps_main"]);

        // for nuisance params
        double tau_us = (R_List["tau_us"]);
        double tau_us_ii = (R_List["tau_us_ii"]);
        double eps_us = (R_List["eps_us"]);
        
        // general
        bool diffusion_HMC = (R_List["diffusion_HMC"]);

        return EHMC_fn_args_struct( tau_main,
                                    tau_main_ii,
                                    eps_main,
                                    // for nuisance params
                                    tau_us,
                                    tau_us_ii,
                                    eps_us,
                                    diffusion_HMC);

}








// /////  Function to convert from R List -> C++ struct (so can call fn's from R without having to use Rcpp::List as not thread-safe, and slower etc)
EHMC_burnin_struct convert_R_List_EHMC_burnin_struct(Rcpp::List R_List) {

  // Convert the R list elements to the appropriate C++ types
  // for main params
  double adapt_delta_main = (R_List["adapt_delta_main"]);
  double LR_main = (R_List["LR_main"]);
  double eps_m_adam_main = (R_List["eps_m_adam_main"]);
  double eps_v_adam_main = (R_List["eps_v_adam_main"]);
  double tau_m_adam_main = (R_List["tau_m_adam_main"]);
  double tau_v_adam_main = (R_List["tau_v_adam_main"]);
  double eigen_max_main =  (R_List["eigen_max_main"]);
  Eigen::VectorXi index_main = (R_List["index_main"]);
  Eigen::Matrix<double, -1, -1> M_dense_sqrt = (R_List["M_dense_sqrt"]);
  Eigen::Matrix<double, -1, 1>  snaper_m_vec_main = (R_List["snaper_m_vec_main"]);
  Eigen::Matrix<double, -1, 1>  snaper_s_vec_main_empirical = (R_List["snaper_s_vec_main_empirical"]);
  Eigen::Matrix<double, -1, 1>  snaper_w_vec_main = (R_List["snaper_w_vec_main"]);
  Eigen::Matrix<double, -1, 1>  eigen_vector_main = (R_List["eigen_vector_main"]);

  // for nuisance params
  double adapt_delta_us = (R_List["adapt_delta_us"]);
  double LR_us = (R_List["LR_us"]);
  double eps_m_adam_us = (R_List["eps_m_adam_us"]);
  double eps_v_adam_us = (R_List["eps_v_adam_us"]);
  double tau_m_adam_us = (R_List["tau_m_adam_us"]);
  double tau_v_adam_us = (R_List["tau_v_adam_us"]);
  double eigen_max_us =  (R_List["eigen_max_us"]);
  Eigen::VectorXi index_us = (R_List["index_us"]);
  Eigen::Matrix<double, -1, 1>  sqrt_M_us_vec = (R_List["sqrt_M_us_vec"]);
  Eigen::Matrix<double, -1, 1>  snaper_m_vec_us = (R_List["snaper_m_vec_us"]);
  Eigen::Matrix<double, -1, 1>  snaper_s_vec_us_empirical = (R_List["snaper_s_vec_us_empirical"]);
  Eigen::Matrix<double, -1, 1>  snaper_w_vec_us = (R_List["snaper_w_vec_us"]);
  Eigen::Matrix<double, -1, 1>  eigen_vector_us = (R_List["eigen_vector_us"]);

  return EHMC_burnin_struct(     adapt_delta_main,
                                 LR_main,
                                 eps_m_adam_main,
                                 eps_v_adam_main,
                                 tau_m_adam_main, 
                                 tau_v_adam_main,
                                 eigen_max_main,
                                 index_main,
                                 M_dense_sqrt,
                                 snaper_m_vec_main,
                                 snaper_s_vec_main_empirical,
                                 snaper_w_vec_main,
                                 eigen_vector_main,
                                 /////// nuisance
                                 adapt_delta_us,
                                 LR_us,
                                 eps_m_adam_us,
                                 eps_v_adam_us,
                                 tau_m_adam_us,
                                 tau_v_adam_us,
                                 eigen_max_us,
                                 index_us,
                                 sqrt_M_us_vec,
                                 snaper_m_vec_us,
                                 snaper_s_vec_us_empirical,
                                 snaper_w_vec_us,
                                 eigen_vector_us);

}



#include <RcppParallel.h>
#include <vector>

struct WarmUp : public RcppParallel::Worker {
  void operator()(std::size_t begin, std::size_t end) override {
    // Perform a dummy operation
    for (std::size_t i = begin; i < end; ++i) {
      volatile double x = i * 0.1; // Prevent compiler optimization
    }
  }
};

// Call this before starting your main function
void warmUpThreads(std::size_t nThreads) {
  WarmUp warmUpTask;
  RcppParallel::parallelFor(0, nThreads, warmUpTask);
}




// [[Rcpp::export]]
Rcpp::List Rcpp_compute_chain_stats(const std::vector<Rcpp::NumericMatrix> mcmc_3D_array,
                                    const std::string stat_type,
                                    const int n_threads) {
  
  const int n_params = mcmc_3D_array.size();
  Rcpp::NumericMatrix output(n_params, 3);
  
  ComputeStatsParallel parallel_worker(n_params,
                                       stat_type,
                                       mcmc_3D_array,
                                       output);
  
  RcppParallel::parallelFor(0, n_params, parallel_worker);
  
  return Rcpp::List::create(Rcpp::Named("statistics") = output);
  
}




// [[Rcpp::export]]
Rcpp::List  Rcpp_compute_MCMC_diagnostics(     const std::vector<Rcpp::NumericMatrix> mcmc_3D_array,
                                               const std::string diagnostic,
                                               const int n_threads
) {
  
      const int n_params = mcmc_3D_array.size(); 
      Rcpp::NumericMatrix  output(n_params, 2);
      
      //// Create the parallel worker   
      ComputeDiagnosticParallel parallel_worker(n_params,
                                                diagnostic,
                                                mcmc_3D_array, 
                                                output);
      
      //// Run parallelFor
      RcppParallel::parallelFor(0, n_params, parallel_worker); // RCppParallel will distribute the load across the n_threads 
      
      //// output
      return Rcpp::List::create(Rcpp::Named("diagnostics") = output);
  
}  







// [[Rcpp::export]]
Rcpp::String  detect_vectorization_support() {
  
#if defined(__AVX__) && !(defined(__AVX2__) && defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__))
  return "AVX";
#elif defined(__AVX2__) && !(defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__))
  return "AVX2";
#elif defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__)
  return "AVX512";
#else
  return "Stan";
#endif
  
}







  



// [[Rcpp::export]]
Eigen::Matrix<double, -1, -1>     Rcpp_wrapper_EIGEN_double(              Eigen::Matrix<double, -1, -1> x,
                                                                             const std::string fn,
                                                                             const std::string vect_type,
                                                                             const bool skip_checks
) {
  

  Eigen::Matrix<double, -1, -1> out_mat =    fn_EIGEN_double(x, fn, vect_type, skip_checks);
  
  return out_mat;
  
  
}








 
// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>        Rcpp_wrapper_fn_lp_grad(             const std::string Model_type,
                                                                         const bool force_autodiff,
                                                                         const bool force_PartialLog,
                                                                         const bool multi_attempts,
                                                                         const Eigen::Matrix<double, -1, 1> theta_main_vec,
                                                                         const Eigen::Matrix<double, -1, 1> theta_us_vec,
                                                                         const Eigen::Matrix<int, -1, -1>  y,
                                                                         const std::string grad_option,
                                                                         const Rcpp::List Model_args_as_Rcpp_List
) {

  const int N = y.rows();
  const int n_us = theta_us_vec.rows()  ;
  const int n_params_main =  theta_main_vec.rows()  ;
  const int n_params = n_params_main + n_us;

  /// convert to Eigen
  const Eigen::Matrix<double, -1, 1> theta_main_vec_Ref =  theta_main_vec;
  const Eigen::Matrix<double, -1, 1> theta_us_vec_Ref =  theta_us_vec;
  const Eigen::Matrix<int, -1, -1>   y_Ref =  y;

  const Model_fn_args_struct Model_args_as_cpp_struct = convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List);


   Eigen::Matrix<double, -1, 1> lp_grad_outs = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);

   Stan_model_struct Stan_model_as_cpp_struct;
   
   
   const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
   const int desired_n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
   const int vec_size = 8;
   ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, desired_n_chunks);
   int chunk_size = chunk_size_info.chunk_size;
   const int n_tests = y.cols();
   
 //  MVP_ThreadLocalWorkspace MVP_workspace(chunk_size, n_tests, n_class);

      // stan::math::start_nested();
       fn_lp_grad_InPlace(   lp_grad_outs,
                             Model_type,
                             force_autodiff, force_PartialLog, multi_attempts,
                             theta_main_vec_Ref, theta_us_vec_Ref,
                             y_Ref,
                             grad_option,
                             Model_args_as_cpp_struct,//MVP_workspace, 
                             Stan_model_as_cpp_struct);
      // stan::math::recover_memory_nested();

   return lp_grad_outs;

}
 


 
 
 
 
 
 
 
 
 


// 
// // [[Rcpp::export]]
// Eigen::Matrix<double, -1, 1>     fn_Rcpp_wrapper_fn_Hessian_diag_nuisance(   const std::string Model_type,
//                                                                              const Eigen::Matrix<double, -1, 1> theta_main_vec,
//                                                                              const Eigen::Matrix<double, -1, 1> theta_us_vec,
//                                                                              const Eigen::Matrix<int, -1, -1>  y,
//                                                                              const Rcpp::List Model_args_as_Rcpp_List
// ) {
// 
//   const int N = y.rows();
//   const int n_us = theta_us_vec.rows()  ;
//   const int n_params_main =  theta_main_vec.rows()  ;
//   const int n_params = n_params_main + n_us;
// 
//   /// convert to Eigen
//   const Eigen::Matrix<double, -1, 1> theta_main_vec_Eigen =  theta_main_vec;
//   const Eigen::Matrix<double, -1, 1> theta_us_vec_Eigen =  theta_us_vec;
//   const Eigen::Matrix<int, -1, -1>   y_Eigen =  y;
// 
//   const Model_fn_args_struct Model_args_as_cpp_struct = convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List);
// 
//   return fn_diag_hessian_us_only_manual(   theta_main_vec_Eigen, theta_us_vec_Eigen, y_Eigen, Model_args_as_cpp_struct);
// 
// }













// [[Rcpp::export]]
Rcpp::List    fn_Rcpp_wrapper_update_M_dense_main_Hessian(            Eigen::Matrix<double, -1, -1> M_dense_main,  /// to be updated
                                                                      Eigen::Matrix<double, -1, -1> M_inv_dense_main, /// to be updated
                                                                      Eigen::Matrix<double, -1, -1> M_inv_dense_main_chol, /// to be updated
                                                                      const double shrinkage_factor,
                                                                      const double ratio_Hess_main,
                                                                      const int interval_width,
                                                                      const double num_diff_e,
                                                                      const std::string  Model_type,
                                                                      const bool force_autodiff,
                                                                      const bool force_PartialLog,
                                                                      const bool multi_attempts,
                                                                      const Eigen::Matrix<double, -1, 1> theta_main_vec,
                                                                      const Eigen::Matrix<double, -1, 1> theta_us_vec,
                                                                      const Eigen::Matrix<int, -1, -1> y,
                                                                      const Rcpp::List Model_args_as_Rcpp_List,
                                                                      const double   ii,
                                                                      const double   n_burnin,
                                                                      const std::string metric_type
) {

 const int N = y.rows();
 const int n_us = theta_us_vec.rows()  ;
 const int n_params_main =  theta_main_vec.rows()  ;
 const int n_params = n_params_main + n_us;

 const Model_fn_args_struct Model_args_as_cpp_struct = convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List);

 Stan_model_struct Stan_model_as_cpp_struct;

 
#if HAS_BRIDGESTAN_H 
 if (Model_args_as_cpp_struct.model_so_file != "none") {

   Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(Model_args_as_cpp_struct.model_so_file,
                                                          Model_args_as_cpp_struct.json_file_path,
                                                          123);

 }
#endif


 Eigen::Matrix<double, -1, -1> M_dense_main_copy = M_dense_main;
 Eigen::Matrix<double, -1, -1> M_inv_dense_main_copy = M_inv_dense_main;
 Eigen::Matrix<double, -1, -1> M_inv_dense_main_chol_copy = M_inv_dense_main_chol;
 
 const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
 const int desired_n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
 const int vec_size = 8;
 ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, desired_n_chunks);
 int chunk_size = chunk_size_info.chunk_size;
 const int n_tests = y.cols();

 update_M_dense_main_Hessian_InPlace(    M_dense_main_copy,
                                         M_inv_dense_main_copy,
                                         M_inv_dense_main_chol_copy,
                                         shrinkage_factor,
                                         ratio_Hess_main,
                                         interval_width,
                                         num_diff_e,
                                         Model_type,
                                         force_autodiff,
                                         force_PartialLog,
                                         multi_attempts,
                                         theta_main_vec,
                                         theta_us_vec,
                                         y,
                                         Model_args_as_cpp_struct,
                                         ii,
                                         n_burnin,
                                         metric_type);



 return Rcpp::List::create(
   Rcpp::Named("M_dense_main_copy") = M_dense_main_copy,
   Rcpp::Named("M_inv_dense_main_copy") = M_inv_dense_main_copy,
   Rcpp::Named("M_inv_dense_main_chol_copy") = M_inv_dense_main_chol_copy
 );



}














// // [[Rcpp::export]]
// Eigen::Matrix<double, -1, -1>     fn_Rcpp_wrapper_compute_main_Hessian_num_diff(     const double num_diff_e,
//                                                                                      const double shrinkage_factor,
//                                                                                      const std::string  Model_type,
//                                                                                      const bool force_autodiff,
//                                                                                      const bool force_PartialLog,
//                                                                                      const bool multi_attempts,
//                                                                                      const Eigen::Matrix<double, -1, 1> theta_main_vec,
//                                                                                      const Eigen::Matrix<double, -1, 1> theta_us_vec,
//                                                                                      const Eigen::Matrix<int, -1, -1> y,
//                                                                                      const Rcpp::List Model_args_as_Rcpp_List,
//                                                                                      const double   ii,
//                                                                                      const double   n_burnin,
//                                                                                      const std::string metric_type
// ) {
// 
//   const int N = y.rows();
//   const int n_us = theta_us_vec.rows()  ;
//   const int n_params_main =  theta_main_vec.rows()  ;
//   const int n_params = n_params_main + n_us;
// 
//   const Model_fn_args_struct Model_args_as_cpp_struct = convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List);
// 
//   Eigen::Matrix<double, -1, -1> Hessian(n_params_main, n_params_main);
// 
//   Stan_model_struct Stan_model_as_cpp_struct;
// 
// #if HAS_BRIDGESTAN_H 
//   if (Model_args_as_cpp_struct.model_so_file != "none") {
// 
//     Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(Model_args_as_cpp_struct.model_so_file,
//                                                            Model_args_as_cpp_struct.json_file_path,
//                                                            123);
// 
//   }
// #endif
// 
//   Hessian = num_diff_Hessian_main_given_nuisance(   num_diff_e,
//                                                     shrinkage_factor,
//                                                     Model_type,
//                                                     force_autodiff,
//                                                     force_PartialLog,
//                                                     multi_attempts,
//                                                     theta_main_vec,
//                                                     theta_us_vec,
//                                                     y,
//                                                     Model_args_as_cpp_struct,
//                                                     Stan_model_as_cpp_struct);
// 
//   return Hessian;
// 
// }
// 
// 
// 












//////////////////////   ADAM / SNAPER-HMC wrapper fns  -------------------------------------------------------------------------------------------------------------------------------



// [[Rcpp::export]]
Rcpp::List                         fn_find_initial_eps_main_and_us(      Eigen::Matrix<double, -1, 1> theta_main_vec_initial_ref,
                                                                              Eigen::Matrix<double, -1, 1> theta_us_vec_initial_ref,
                                                                              const double seed,
                                                                              const std::string Model_type,
                                                                              const bool  force_autodiff,
                                                                              const bool  force_PartialLog,
                                                                              const bool  multi_attempts,
                                                                              Eigen::Matrix<int, -1, -1> y_ref,
                                                                              const Rcpp::List Model_args_as_Rcpp_List,
                                                                              Rcpp::List  EHMC_args_as_Rcpp_List, /// pass by ref. to modify (???)
                                                                              const Rcpp::List   EHMC_Metric_as_Rcpp_List
) {
  
  
      const bool burnin = false;
      const int n_params_main = theta_main_vec_initial_ref.rows();
      const int n_us = theta_us_vec_initial_ref.rows();
      const int n_params = n_params_main + n_us;
      const int N = y_ref.rows();
      
      HMCResult result_input(n_params_main, n_params, N);
      result_input.main_theta_vec = theta_main_vec_initial_ref;
      result_input.main_theta_vec_0 = theta_main_vec_initial_ref;
      result_input.main_theta_vec_proposed = theta_main_vec_initial_ref;
      result_input.main_velocity_0_vec = theta_main_vec_initial_ref;
      result_input.main_velocity_vec_proposed = theta_main_vec_initial_ref;
      result_input.main_velocity_vec = theta_main_vec_initial_ref;
      
      // convert Rcpp::List to cpp structs and pass by reference
      const Model_fn_args_struct Model_args_as_cpp_struct = convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List);
      EHMC_fn_args_struct  EHMC_args_as_cpp_struct =  convert_R_List_EHMC_fn_args_struct(EHMC_args_as_Rcpp_List);
      const EHMC_Metric_struct   EHMC_Metric_as_cpp_struct =  convert_R_List_EHMC_Metric_struct(EHMC_Metric_as_Rcpp_List);
      
      const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
      const int desired_n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
      const int vec_size = 8;
      ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, desired_n_chunks);
      int chunk_size = chunk_size_info.chunk_size;
      const int n_tests = y_ref.cols();
      
     // MVP_ThreadLocalWorkspace MVP_workspace(chunk_size, n_tests, n_class);
      
      std::vector<double> eps_pair =  fn_find_initial_eps_main_and_us(   result_input,
                                                                         seed, burnin,  Model_type,  
                                                                         force_autodiff, force_PartialLog, multi_attempts, 
                                                                         y_ref,
                                                                         Model_args_as_cpp_struct, // MVP_workspace, 
                                                                         EHMC_args_as_cpp_struct, EHMC_Metric_as_cpp_struct);
      
      Rcpp::List outs(2);
      outs(0) = eps_pair[0];
      outs(1) = eps_pair[1];
      
      return outs;
      
}




 
 
 




// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>     fn_Rcpp_wrapper_adapt_eps_ADAM(   double eps,   //// updating this
                                                                   double eps_m_adam,   //// updating this
                                                                   double eps_v_adam,  //// updating this
                                                                   const int iter,
                                                                   const int n_burnin,
                                                                   const double LR,  /// ADAM learning rate
                                                                   const double p_jump,
                                                                   const double adapt_delta,
                                                                   const double beta1_adam,
                                                                   const double beta2_adam,
                                                                   const double eps_adam
) {


  Eigen::Matrix<double, -1, 1>  out_vec  =    adapt_eps_ADAM(  eps,
                                                               eps_m_adam,
                                                               eps_v_adam,
                                                               iter,
                                                               n_burnin,
                                                               LR,
                                                               p_jump,
                                                               adapt_delta,
                                                               beta1_adam,
                                                               beta2_adam,
                                                               eps_adam);


  out_vec(0) = eps;
  out_vec(1) = eps_m_adam;
  out_vec(2) = eps_v_adam;

  return out_vec;

}













// [[Rcpp::export]]
Eigen::Matrix<double, -1, -1>  fn_update_snaper_m_and_s(     Eigen::Matrix<double, -1, 1> snaper_m,  // to be updated
                                                             Eigen::Matrix<double, -1, 1> snaper_s_empirical,   // to be updated
                                                             const Eigen::Matrix<double, -1, 1>  theta_vec_mean,  // mean theta_vec across all K chains
                                                             const double ii
) {


  const double kappa  =  8.0;
  const double eta_m  =  1.0 / (std::ceil(static_cast<double>(ii)/kappa) + 1.0);

  // update snaper_m
  if (static_cast<int>(ii) < 2) {
    snaper_m = theta_vec_mean;
  } else {
    snaper_m = (1.0 - eta_m)*snaper_m + eta_m*theta_vec_mean;
  }

  // update posterior variances (snaper_s_empirical)
  Eigen::Matrix<double, -1, 1> theta_vec_mean_m_snaper_m = (theta_vec_mean  - snaper_m);
  Eigen::Matrix<double, -1, 1> current_variances = ( theta_vec_mean_m_snaper_m.array() * theta_vec_mean_m_snaper_m.array() ).matrix() ;
  snaper_s_empirical = (1.0 - eta_m)*snaper_s_empirical   +   eta_m*current_variances;


  Eigen::Matrix<double, -1, -1> out_mat(snaper_m.rows(), 2);
  out_mat.col(0) = snaper_m;
  out_mat.col(1) = snaper_s_empirical;
  return out_mat;

}











// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1> fn_update_eigen_max_and_eigen_vec(       double eigen_max,   //// NOT const as updating!
                                                                      Eigen::Matrix<double, -1, 1> eigen_vector,  //// NOT const as updating!
                                                                      const Eigen::Matrix<double, -1, 1>  snaper_w_vec // this is const here
) {


  // compute L2-norm of W (sum of elements squared)
  double w_norm_sq =  snaper_w_vec.array().square().sum();
  double w_norm  =  stan::math::sqrt(w_norm_sq) ; // snaper_w_vec.array().square().sum();

  //// update eigen max
  eigen_max = w_norm;

  //// update eigen vector
  if (eigen_max > 0) {
    eigen_vector = snaper_w_vec / w_norm;
  }

  Eigen::Matrix<double, -1, 1> out_vec(eigen_vector.size() + 1);
  out_vec(0) = eigen_max;
  out_vec.tail(eigen_vector.size()) = eigen_vector;

  return out_vec;

}





 



// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1> fn_update_snaper_w_dense_M(    Eigen::Matrix<double, -1, 1>  snaper_w_vec,    //// NOT const as updating!
                                                            const Eigen::Matrix<double, -1, 1>  eigen_vector,
                                                            const double eigen_max,
                                                            const Eigen::Matrix<double, -1, 1>  theta_vec,
                                                            const Eigen::Matrix<double, -1, 1>  snaper_m_vec,
                                                            const double ii,
                                                            const Eigen::Matrix<double, -1, -1> M_dense_sqrt
) {


  const int eta_w = 3;

  //// update W (for DENSE M) - this part varies from the diag_M tau-adaptation function!
  Eigen::Matrix<double, -1, 1> x_c = M_dense_sqrt * (theta_vec - snaper_m_vec); // this is the only part which is different from diag (and of course the inputs).
  if    (eigen_max > 0.0)    {
    double x_c_eigen_vector_dot_prod =  (x_c.array() * eigen_vector.array()).sum() ;
    Eigen::Matrix<double, -1, 1> current_w =   x_c * x_c_eigen_vector_dot_prod  ;
    snaper_w_vec = ( snaper_w_vec.array() * ((ii - eta_w) / (ii + 1.0)) + ((eta_w + 1.0) / (ii + 1.0)) * current_w.array() ).matrix() ; /// update snaper_w_vec
  } else {
    snaper_w_vec = x_c;
  }

  return snaper_w_vec;

}






 


// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>  fn_update_snaper_w_diag_M(       Eigen::Matrix<double, -1, 1>  snaper_w_vec,    //// NOT const as updating!
                                                               const Eigen::Matrix<double, -1, 1>  eigen_vector,
                                                               const double eigen_max,
                                                               const Eigen::Matrix<double, -1, 1>  theta_vec,
                                                               const Eigen::Matrix<double, -1, 1>  snaper_m_vec,
                                                               const double ii,
                                                               const Eigen::Matrix<double, -1, 1>  sqrt_M_vec
) {


  const int eta_w = 3;

  //// update W (for DIAG M)
  const Eigen::Matrix<double, -1, 1> x_c = ( sqrt_M_vec.array() * (theta_vec - snaper_m_vec).array() ).matrix() ; // this is the only part which is different from diag (and of course the inputs).
  if    (eigen_max > 0.0)    {
    Eigen::Matrix<double, -1, 1> current_w = ( x_c.array() * (x_c.array() * eigen_vector.array()).sum() ).matrix() ;
    snaper_w_vec = ( snaper_w_vec.array() * ((ii - eta_w) / (ii + 1)) + ((eta_w + 1) / (ii + 1)) * current_w.array() ).matrix() ; /// update snaper_w_vec
  } else {
    snaper_w_vec = x_c;
  }

  return snaper_w_vec;

}





 

 // [[Rcpp::export]]
 Eigen::Matrix<double, -1, 1> fn_Rcpp_wrapper_update_tau_w_diag_M_ADAM(    const Eigen::Matrix<double, -1, 1> eigen_vector,
                                                                           const double eigen_max,
                                                                           const Eigen::Matrix<double, -1, 1> theta_vec_initial,
                                                                           const Eigen::Matrix<double, -1, 1> theta_vec_prop,
                                                                           const Eigen::Matrix<double, -1, 1> snaper_m_vec,
                                                                           const Eigen::Matrix<double, -1, 1> velocity_prop,
                                                                           const Eigen::Matrix<double, -1, 1> velocity_0,
                                                                           double tau,  /// updating this
                                                                           const double LR,
                                                                           const double ii,
                                                                           const double n_burnin,
                                                                           const Eigen::Matrix<double, -1, 1> sqrt_M_vec,
                                                                           double tau_m_adam,   /// updating this
                                                                           double tau_v_adam,  /// updating this
                                                                           const double tau_ii
 ) {

     //  Eigen::Matrix<double, -1, 1> out_vec(3);

       ////// for main (this also updates snaper_w)
    return   fn_update_tau_w_diag_M_ADAM(
                                         eigen_vector,
                                         eigen_max,
                                         theta_vec_initial,
                                         theta_vec_prop,
                                         snaper_m_vec,   // READ-ONLY in this function
                                         velocity_prop,
                                         velocity_0,
                                         tau,  // being modified !!!
                                         LR,
                                         ii,
                                         n_burnin,
                                         sqrt_M_vec,  // READ-ONLY in this function
                                         tau_m_adam,
                                         tau_v_adam,  // being modified !!!
                                         tau_ii
                                       );


 }



 

 // [[Rcpp::export]]
 Eigen::Matrix<double, -1, 1> fn_Rcpp_wrapper_update_tau_w_dense_M_ADAM(   const Eigen::Matrix<double, -1, 1> eigen_vector,
                                                                           const double eigen_max,
                                                                           const Eigen::Matrix<double, -1, 1> theta_vec_initial,
                                                                           const Eigen::Matrix<double, -1, 1> theta_vec_prop,
                                                                           const Eigen::Matrix<double, -1, 1> snaper_m_vec,
                                                                           const Eigen::Matrix<double, -1, 1> velocity_prop,
                                                                           const Eigen::Matrix<double, -1, 1> velocity_0,
                                                                           double tau,   /// updating this
                                                                           const double LR,
                                                                           const double ii,
                                                                           const double n_burnin,
                                                                           const Eigen::Matrix<double, -1, -1> M_dense_sqrt,
                                                                           double tau_m_adam,   /// updating this
                                                                           double tau_v_adam,  /// updating this
                                                                           const double tau_ii
 ) {

    // Eigen::Matrix<double, -1, 1> out_vec(3);

     ////// for main (this also updates snaper_w)
     return fn_update_tau_w_diag_M_ADAM(
                                         eigen_vector,
                                         eigen_max,
                                         theta_vec_initial,
                                         theta_vec_prop,
                                         snaper_m_vec,   // READ-ONLY in this function
                                         velocity_prop,
                                         velocity_0,
                                         tau,  // being modified !!!
                                         LR,
                                         ii,
                                         n_burnin,
                                         M_dense_sqrt,  // READ-ONLY in this function
                                         tau_m_adam,
                                         tau_v_adam,  // being modified !!!
                                         tau_ii
                                       );

}





  




// Some R / C++ helper functions -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


// [[Rcpp::export]]
double   Rcpp_det(const Eigen::Matrix<double, -1, -1>  &mat) {

  return(    (mat).determinant()   ) ;

}




// [[Rcpp::export]]
double   Rcpp_log_det(const Eigen::Matrix<double, -1, -1>  &mat) {

  return (  stan::math::log( stan::math::abs(  (mat).determinant())  )  ) ;

}





// [[Rcpp::export]]
Eigen::Matrix<double, -1, -1>    Rcpp_solve(const Eigen::Matrix<double, -1, -1>  &mat) {

  return mat.inverse(); //  fn_convert_EigenMat_to_RcppMat_dbl(fn_convert_RcppMat_to_EigenMat(mat).inverse());

}




// [[Rcpp::export]]
Eigen::Matrix<double, -1, -1>  Rcpp_Chol(const Eigen::Matrix<double, -1, -1>  &mat) {

  Eigen::Matrix<double, -1, -1>    res_Eigen = (  (mat).llt().matrixL() ).toDenseMatrix().matrix() ;
  return  (res_Eigen);

}




// 
// // [[Rcpp::export]]
// Eigen::Matrix<double, -1, -1> sqrtm(const Eigen::Matrix<double, -1, -1> &M) {
// 
//   // make sure the input matrix is symmetric
//   Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, -1, -1>> solver(M);
// 
//   if (solver.info() != Eigen::Success) {
//     throw std::runtime_error("Eigenvalue decomposition failed!");
//   }
// 
//   // extract eigenvalues and eigenvectors
//   Eigen::Matrix<double, -1, -1>  D = solver.eigenvalues().array().sqrt().matrix().asDiagonal(); // sqrt of eigenvalues
//   Eigen::Matrix<double, -1, -1>  V = solver.eigenvectors(); // eigenvectors
// 
//   // Reconstruct the square root of the matrix
//   Eigen::Matrix<double, -1, -1> M_sqrt = V * D *  V.transpose();
// 
//   return M_sqrt;
// 
// }
// 
// 



// 
// 
// inline Eigen::Matrix<double, -1, -1>  fn_update_empirical_covariance(Eigen::Ref<Eigen::Matrix<double, -1, -1>>  empicical_cov_main,
//                                                               Eigen::Ref<Eigen::Matrix<double, -1, 1>>  snaper_m_vec,
//                                                               Eigen::Ref<Eigen::Matrix<double, -1, 1>>  theta_vec,
//                                                               double ii) {
//   double ii_p1 = ii + 1.0;
// 
//   Eigen::Matrix<double, -1, 1> delta = theta_vec - snaper_m_vec;
//   Eigen::Matrix<double, -1, -1> delta_x_self_transpose = delta * delta.transpose();
//   empicical_cov_main.array() =  (  (ii_p1 - 1.0) * empicical_cov_main.array()  + delta_x_self_transpose.array() *  ((ii_p1  -  1.0) / ii_p1) ) / ii_p1 ;
// 
//   empicical_cov_main = near_PD(empicical_cov_main);
// 
//   return empicical_cov_main;
// 
// 
// }







template<typename T>
inline bool is_multiple(T ii, T interval) {
  if constexpr (std::is_integral<T>::value) {
    return ii % interval == 0;
  } else {
    return std::fmod(ii, interval) < 1e-6;
  }
}






// Function to clean up NaN/Inf and outlier elements in the Eigen vector
inline void clean_vector(Eigen::Matrix<double, -1, 1> &vec) {

  Eigen::Array<bool, -1, 1> valid_mask = vec.array().isFinite();


  Eigen::Matrix<double, -1, 1> valid_elements = vec(valid_mask);

  if (valid_elements.size() == 0) {
    throw std::runtime_error("All elements are NaN and/or Inf!");
  }


  double mean = valid_elements.mean();
  double stddev = std::sqrt((valid_elements.array() - mean).square().mean());


  for (int i = 0; i < vec.size(); ++i) {
    if (!std::isfinite(vec(i))) {
      vec(i) = mean;
    }
  }

  for (int i = 0; i < vec.size(); ++i) {
    if (std::abs(vec(i) - mean) > 10 * stddev) {
      if (vec(i) > mean) {
        vec(i) = valid_elements(valid_elements.array() <= mean + 10 * stddev).maxCoeff();
      } else {
        vec(i) = valid_elements(valid_elements.array() >= mean - 10 * stddev).minCoeff();
      }
    }
  }

}




// 
// // [[Rcpp::export]]
// std::vector<Eigen::Matrix<double, -1, -1>> fn_2D_to_3D_array_Eigen(  const Eigen::Matrix<double, -1, -1> theta_trace_as_2D_array,
//                                                                      const int n_params_main,
//                                                                      const int n_chains,
//                                                                      const int n_iter) {
// 
// 
//   std::vector<Eigen::Matrix<double, -1, -1>> theta_trace_3D_array = vec_of_mats(n_params_main, n_chains, n_iter);
//   std::vector<Eigen::Matrix<double, -1, -1>> theta_trace_3D_array_out = vec_of_mats(n_iter, n_chains, n_params_main);
// 
//   for (int ii = 0; ii < n_iter; ++ii) {
//     for (int kk = 0; kk < n_chains; ++kk) {
//       for (int j = 0; j < n_params_main; ++j) {
//         int row_index = j + kk * n_params_main;
//         try {
//           theta_trace_3D_array[ii](j, kk) = theta_trace_as_2D_array(row_index, ii);
//           theta_trace_3D_array_out[j](ii, kk) =  theta_trace_3D_array[ii](j, kk);
//         } catch (...) {
//           // Handle error
//         }
//       }
//     }
//   }
// 
// 
//   return theta_trace_3D_array_out;
// 
// }
// 



//
//
//
// // [[Rcpp::export]]
// Rcpp::List     Rcpp_wrapper_fn_sample_HMC_multi_iter_single_thread(    const int chain_id,
//                                                                        const int seed,
//                                                                        const int n_iter,
//                                                                        const bool partitioned_HMC,
//                                                                        const std::string Model_type,
//                                                                        const bool sample_nuisance,
//                                                                        const bool force_autodiff,
//                                                                        const bool force_PartialLog,
//                                                                        const bool multi_attempts,
//                                                                        const int n_nuisance_to_track,
//                                                                        const Eigen::Matrix<double, -1, 1>  theta_main_vector_from_single_chain_input_from_R,
//                                                                        const Eigen::Matrix<double, -1, 1>  theta_us_vector_from_single_chain_input_from_R,
//                                                                        const Eigen::Matrix<int, -1, -1> y_Eigen_i,
//                                                                        const Rcpp::List  &Model_args_as_Rcpp_List,  ///// ALWAYS read-only
//                                                                        const Rcpp::List  &EHMC_args_as_Rcpp_List,
//                                                                        const Rcpp::List  &EHMC_Metric_as_Rcpp_List)  {
//
//   int N = y_Eigen_i.rows();
//   int n_params_main = theta_main_vector_from_single_chain_input_from_R.size();
//   int n_us = theta_us_vector_from_single_chain_input_from_R.size();
//   int n_params = n_params_main + n_us;
//
//   const bool burnin_indicator = false;
//
//   HMCResult result_input(n_params_main, n_us, N);
//
//   //// convert lists to C++ structs
//   const Model_fn_args_struct     Model_args_as_cpp_struct =   convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List); ///// ALWAYS read-only
//   EHMC_fn_args_struct      EHMC_args_as_cpp_struct =    convert_R_List_EHMC_fn_args_struct(EHMC_args_as_Rcpp_List);
//   const EHMC_Metric_struct       EHMC_Metric_as_cpp_struct =  convert_R_List_EHMC_Metric_struct(EHMC_Metric_as_Rcpp_List);
//
//   //auto rng = RNGManager::get_thread_local_rng(seed);
//   static thread_local std::mt19937 rng(static_cast<unsigned int>(seed));
//
//   thread_local Stan_model_struct Stan_model_as_cpp_struct;
//
//   if (Model_args_as_cpp_struct.model_so_file != "none" && Stan_model_as_cpp_struct.bs_model_ptr == nullptr) {
//
//       Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(Model_args_as_cpp_struct.model_so_file,
//                                                              Model_args_as_cpp_struct.json_file_path,
//                                                              123);
//
//   }
//
//   HMC_output_single_chain HMC_output_single_chain_i =  fn_sample_HMC_multi_iter_single_thread(result_input,
//                                                                                               burnin_indicator,
//                                                                                               rng,
//                                                                                               n_iter,
//                                                                                               partitioned_HMC,
//                                                                                               Model_type,
//                                                                                               sample_nuisance,
//                                                                                               force_autodiff,
//                                                                                               force_PartialLog,
//                                                                                               multi_attempts,
//                                                                                               n_nuisance_to_track,
//                                                                                               y_Eigen_i,
//                                                                                               Model_args_as_cpp_struct,
//                                                                                               EHMC_args_as_cpp_struct,
//                                                                                               EHMC_Metric_as_cpp_struct,
//                                                                                               Stan_model_as_cpp_struct);
//
//   // destroy Stan model object
//   if (Model_args_as_cpp_struct.model_so_file != "none" && Stan_model_as_cpp_struct.bs_model_ptr != nullptr) {
//     fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
//   }
//
//
//    Rcpp::List out_list(5);
//
//    out_list(0) = HMC_output_single_chain_i.Eigen_thread_local_trace_buffer;
//    out_list(1) = HMC_output_single_chain_i.Eigen_thread_local_trace_buffer_div;
//    out_list(2) = HMC_output_single_chain_i.Eigen_thread_local_trace_buffer_nuisance;
//
//
//    return  out_list;
//
//
// }
//
//









#if HAS_BRIDGESTAN_H


struct ParamConstrainWorker : public RcppParallel::Worker {

      // Inputs
      const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_main;
      const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_nuisance;
      const Eigen::VectorXi &pars_indicies_to_track;
      const int n_params_full;
      const int n_nuisance;
      const int n_params_main;
      const bool include_nuisance;
      const std::string &model_so_file;
      const std::string &json_file_path;

      /// Output uses tbb container
      tbb::concurrent_vector<RcppParallel::RMatrix<double>> all_param_outs_trace_concurrent; // (n_chains);


  // Constructor
  ParamConstrainWorker( const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_main,
                        const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_nuisance,
                        const Eigen::VectorXi  &pars_indicies_to_track,
                        const int &n_params_full,
                        const int &n_nuisance,
                        const int &n_params_main,
                        const bool &include_nuisance,
                        const std::string &model_so_file,
                        const std::string &json_file_path,
                        std::vector<Rcpp::NumericMatrix>   &all_param_outs_trace)
                        :
                          unc_params_trace_input_main(unc_params_trace_input_main),
                          unc_params_trace_input_nuisance(unc_params_trace_input_nuisance),
                          pars_indicies_to_track(pars_indicies_to_track),
                          n_params_full(n_params_full),
                          n_nuisance(n_nuisance),
                          n_params_main(n_params_main),
                          include_nuisance(include_nuisance),
                          model_so_file(model_so_file),
                          json_file_path(json_file_path)
    {

        // Initialize concurrent vector of RMatrix
        all_param_outs_trace_concurrent = convert_vec_of_RcppMat_to_concurrent_vector(all_param_outs_trace, all_param_outs_trace_concurrent);

    }

        // Thread-local processing
        void operator()(std::size_t begin, std::size_t end) {


                  // Process assigned chains
                    std::size_t kk = begin;
                    {


                      char* error_msg = nullptr;
                      const int n_iter = unc_params_trace_input_main[0].cols();
                      const int n_params = n_nuisance + n_params_main;
                      const int n_params_to_track = pars_indicies_to_track.size();

                      thread_local stan::math::ChainableStack ad_tape;
                      thread_local stan::math::nested_rev_autodiff nested;

                      // Initialize matrix for this chain
                      Eigen::Matrix<double, -1, -1> chain_output(n_params_to_track, n_iter);

                        thread_local Stan_model_struct Stan_model_as_cpp_struct;
                        thread_local bs_rng* bs_rng_object = nullptr;

                        // Initialize model and RNG once per thread

                        if (model_so_file != "none" && Stan_model_as_cpp_struct.bs_model_ptr == nullptr) {
                          Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(model_so_file, json_file_path, 123);
                          bs_rng_object = bs_rng_construct(123, &error_msg);
                        }


                        for (int ii = 0; ii < n_iter; ii++) {

                          Eigen::Matrix<double, -1, 1> theta_unc_full_input(n_params);
                          Eigen::Matrix<double, -1, 1> theta_constrain_full_output(n_params_full);

                              theta_unc_full_input.tail(n_params_main) = unc_params_trace_input_main[kk].col(ii);

                              if (include_nuisance) {
                                theta_unc_full_input.head(n_nuisance) = unc_params_trace_input_nuisance[kk].col(ii);
                              } else {
                                theta_unc_full_input.head(n_nuisance).array() = 0.0;
                              }

                              int result = Stan_model_as_cpp_struct.bs_param_constrain( Stan_model_as_cpp_struct.bs_model_ptr,
                                                                                        true,
                                                                                        true,
                                                                                        theta_unc_full_input.data(),
                                                                                        theta_constrain_full_output.data(),
                                                                                        bs_rng_object,
                                                                                        &error_msg);

                              if (result != 0) {
                                throw std::runtime_error("computation failed: " +
                                                         std::string(error_msg ? error_msg : "Unknown error"));
                              }

                              chain_output.col(ii) = theta_constrain_full_output(pars_indicies_to_track);

                        }

                        // Store completed chain
                        all_param_outs_trace_concurrent[kk] =  fn_convert_EigenMat_to_RMatrix(chain_output,  all_param_outs_trace_concurrent[kk]);

                        // Clean up thread-local resources if neede
                        if (bs_rng_object != nullptr) {
                          bs_rng_destruct(bs_rng_object);
                         }

                       fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);


                  }



        }


};


#endif




// [[Rcpp::export]]
std::vector<Rcpp::NumericMatrix>     fn_compute_param_constrain_from_trace_parallel(  const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_main,
                                                                                      const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_nuisance,
                                                                                      const Eigen::VectorXi &pars_indicies_to_track,
                                                                                      const int &n_params_full,
                                                                                      const int &n_nuisance,
                                                                                      const int &n_params_main,
                                                                                      const bool &include_nuisance,
                                                                                      const std::string &model_so_file,
                                                                                      const std::string &json_file_path) {


  const int n_chains = unc_params_trace_input_main.size();
  const int n_iter = unc_params_trace_input_main[0].cols();
  const int n_params_to_track = pars_indicies_to_track.size();

  std::vector<Rcpp::NumericMatrix> all_param_outs_trace_std_vec = vec_of_mats_Rcpp(n_params_to_track, n_iter, n_chains);


#if HAS_BRIDGESTAN_H

  // Create worker
  ParamConstrainWorker worker(
      unc_params_trace_input_main,
      unc_params_trace_input_nuisance,
      pars_indicies_to_track,
      n_params_full,
      n_nuisance,
      n_params_main,
      include_nuisance,
      model_so_file,
      json_file_path,
      all_param_outs_trace_std_vec);

  // Run parallel chains
  RcppParallel::parallelFor(0, n_chains, worker);


#endif


  return all_param_outs_trace_std_vec;

}










// [[Rcpp::export]]
std::vector<Eigen::Matrix<double, -1, -1>>  fn_compute_param_constrain_from_trace(    const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_main,
                                                                                      const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_nuisance,
                                                                                      const Eigen::VectorXi &pars_indicies_to_track,
                                                                                      const int &n_params_full,
                                                                                      const int &n_nuisance,
                                                                                      const int &n_params_main,
                                                                                      const bool  &include_nuisance,
                                                                                      const std::string &model_so_file,
                                                                                      const std::string &json_file_path) {


  
#if HAS_BRIDGESTAN_H
  
  char* error_msg = nullptr;
  unsigned int seed = 123;

  bs_rng* bs_rng_object = bs_rng_construct(seed, &error_msg); /// bs rng object

  // //// For Stan models:  Initialize bs_model* pointer and void* handle
  Stan_model_struct Stan_model_as_cpp_struct;

  // Initialize model
  Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(model_so_file,
                                                         json_file_path,
                                                         seed);
  
#endif
 


  /// trace to store output
  const int n_chains = unc_params_trace_input_main.size();
  const int n_iter = unc_params_trace_input_main[0].cols();
  const int n_params_to_track = pars_indicies_to_track.size();
  const int n_params = n_nuisance + n_params_main;

  std::vector<Eigen::Matrix<double, -1, -1>> all_param_outs_trace = vec_of_mats(n_params_to_track, n_iter, n_chains);
  
#if HAS_BRIDGESTAN_H

  /// make storage containers
  Eigen::Matrix<double, -1, 1> theta_unc_full_input(n_params);
  Eigen::Matrix<double, -1, 1> theta_constrain_full_output(n_params_full);


  for (int kk = 0; kk <  n_chains; kk += 1) {
    for (int ii = 0; ii <  n_iter; ii += 1) {

      theta_unc_full_input.tail(n_params_main) =   unc_params_trace_input_main[kk].col(ii);

       if (include_nuisance == true) {
          theta_unc_full_input.head(n_nuisance) =      unc_params_trace_input_nuisance[kk].col(ii);
       } else {
         theta_unc_full_input.head(n_nuisance).array() = 0.0; // set to zero if ignoring nuisance
       }


          int result = Stan_model_as_cpp_struct.bs_param_constrain(   Stan_model_as_cpp_struct.bs_model_ptr,
                                                                      true,
                                                                      true,
                                                                      theta_unc_full_input.data(),
                                                                      theta_constrain_full_output.data(), //  all_param_outs_trace[kk].col(ii).data(),
                                                                      bs_rng_object,
                                                                      &error_msg);
 

       all_param_outs_trace[kk].col(ii) = theta_constrain_full_output(pars_indicies_to_track);

          if (result != 0) {
            throw std::runtime_error("computation failed: " +
                                     std::string(error_msg ? error_msg : "Unknown error"));
          }

    }
  }

  // destroy Stan model object
  if (model_so_file != "none" && Stan_model_as_cpp_struct.bs_model_ptr != nullptr) {
    fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
  }

#endif


  return all_param_outs_trace;

}









// --------------------------------- RcpParallel  functions  ----------------------------------------------------------------------------------------------------------------------------------------------------------


// [[Rcpp::export]]
Rcpp::List                                   Rcpp_fn_RcppParallel_EHMC_sampling(  const int n_threads_R,
                                                                                  const int seed_R,
                                                                                  const int n_iter_R,
                                                                                  const bool iter_one_by_one,
                                                                                  const bool partitioned_HMC_R,
                                                                                  const std::string Model_type_R,
                                                                                  const bool sample_nuisance_R,
                                                                                  const bool force_autodiff_R,
                                                                                  const bool force_PartialLog_R,
                                                                                  const bool multi_attempts_R,
                                                                                  const int n_nuisance_to_track,
                                                                                  const Eigen::Matrix<double, -1, -1>  theta_main_vectors_all_chains_input_from_R,
                                                                                  const Eigen::Matrix<double, -1, -1>  theta_us_vectors_all_chains_input_from_R,
                                                                                  const Eigen::Matrix<int, -1, -1> y_Eigen_R,
                                                                                  const Rcpp::List Model_args_as_Rcpp_List,  ///// ALWAYS read-only
                                                                                  const Rcpp::List EHMC_args_as_Rcpp_List,
                                                                                  const Rcpp::List EHMC_Metric_as_Rcpp_List
) {

  //// key dimensions
  const int n_params_main = theta_main_vectors_all_chains_input_from_R.rows();
  const int n_us = theta_us_vectors_all_chains_input_from_R.rows();

  //// create EMPTY OUTPUT / containers* to be filled (each col filled from different thread w/ each col corresponding to a different chain)
  Rcpp::NumericMatrix  theta_main_vectors_all_chains_output_to_R =  fn_convert_EigenMat_to_RcppMat_dbl(theta_main_vectors_all_chains_input_from_R);   // write to this

  //// nuisance
  Rcpp::NumericMatrix  theta_us_vectors_all_chains_output_to_R  = fn_convert_EigenMat_to_RcppMat_dbl(theta_us_vectors_all_chains_input_from_R);

  //// convert lists to C++ structs
  const Model_fn_args_struct     Model_args_as_cpp_struct =   convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List); ///// ALWAYS read-only
  const EHMC_fn_args_struct      EHMC_args_as_cpp_struct =    convert_R_List_EHMC_fn_args_struct(EHMC_args_as_Rcpp_List);
  const EHMC_Metric_struct       EHMC_Metric_as_cpp_struct =  convert_R_List_EHMC_Metric_struct(EHMC_Metric_as_Rcpp_List);
  //// replicate these structs for thread-safety as we will be modifying them for burnin
  std::vector<Model_fn_args_struct> Model_args_as_cpp_struct_copies_R =     replicate_Model_fn_args_struct( Model_args_as_cpp_struct,  n_threads_R); // read-only
  std::vector<EHMC_fn_args_struct>  EHMC_args_as_cpp_struct_copies_R =      replicate_EHMC_fn_args_struct(  EHMC_args_as_cpp_struct,   n_threads_R); // need to edit these !!
  std::vector<EHMC_Metric_struct>   EHMC_Metric_as_cpp_struct_copies_R =    replicate_EHMC_Metric_struct(   EHMC_Metric_as_cpp_struct, n_threads_R); // read-only
  
  ///// Traces
  const int N = Model_args_as_cpp_struct.N;
  std::vector<Rcpp::NumericMatrix> trace_output =  vec_of_mats_Rcpp(n_params_main, n_iter_R, n_threads_R);
  std::vector<Rcpp::NumericMatrix> trace_output_divs =  vec_of_mats_Rcpp(1, n_iter_R, n_threads_R);
  std::vector<Rcpp::NumericMatrix> trace_output_nuisance =  vec_of_mats_Rcpp(n_nuisance_to_track, n_iter_R, n_threads_R);
//  std::vector<Rcpp::NumericMatrix> trace_output_log_lik = vec_of_mats_Rcpp(N, n_iter_R, n_threads_R);  //// possibly dummy
  
  ///// data copies
  std::vector<Eigen::Matrix<int, -1, -1>> y_copies_R = vec_of_mats<int>(y_Eigen_R.rows(), y_Eigen_R.cols(), n_threads_R);
  for (int kk = 0; kk < n_threads_R; ++kk) {
     y_copies_R[kk] = y_Eigen_R;
  }
  
  //std::vector<HMC_output_single_chain> HMC_outputs_R(n_threads_R, HMC_output_single_chain(n_iter_R, n_nuisance_to_track, n_params_main, n_us, N));
///  std::vector<HMC_output_single_chain>  HMC_outputs_R(n_threads_R);
  // for (int i = 0; i < n_threads_R; ++i) {
  //    HMC_output_single_chain HMC_output_single_chain_i(n_iter_R, n_nuisance_to_track, n_params_main, n_us, N);
  //    HMC_outputs_R[i] =  (HMC_output_single_chain_i);
  // }
  
   // tbb::task_scheduler_init init(n_threads_R);
   warmUpThreads(n_threads_R);
   
   //// create worker
   RcppParallel_EHMC_sampling      parallel_hmc_sampling(  n_threads_R,
                                                           seed_R,
                                                           n_iter_R,
                                                           partitioned_HMC_R,
                                                           Model_type_R,
                                                           sample_nuisance_R,
                                                           force_autodiff_R,
                                                           force_PartialLog_R,
                                                           multi_attempts_R,
                                                           ///// inputs
                                                           theta_main_vectors_all_chains_input_from_R,
                                                           theta_us_vectors_all_chains_input_from_R,
                                                           ///// outputs (main)
                                                           trace_output,
                                                           ///// data
                                                           y_copies_R,
                                                           ///// structs
                                                           Model_args_as_cpp_struct_copies_R, ///// ALWAYS read-only
                                                           EHMC_args_as_cpp_struct_copies_R,
                                                           EHMC_Metric_as_cpp_struct_copies_R,
                                                           ///// traces
                                                           trace_output_divs,
                                                           n_nuisance_to_track,
                                                           trace_output_nuisance
                                                         //  HMC_outputs_R
                                                         );

   //// Call parallelFor
   RcppParallel::parallelFor(0, n_threads_R, parallel_hmc_sampling);

   ////  copy / store trace
   parallel_hmc_sampling.copy_results_to_output(); 

   // //// parallel_hmc_sampling.reset();
   // parallel_hmc_sampling.reset_Eigen();
   // 
   // //// clear TBB concurrent vectors
   // parallel_hmc_sampling.reset_tbb();

   //  init.terminate();
 
   //// Reset everything
   parallel_hmc_sampling.reset();

   //// Return results
   return Rcpp::List::create(trace_output,
                             trace_output_divs,
                             trace_output_nuisance,
                             theta_main_vectors_all_chains_output_to_R,
                             theta_us_vectors_all_chains_output_to_R
                             // trace_output_log_lik
                             );




}











// 
//  // [[Rcpp::export]]
//  Rcpp::List                                   Rcpp_fn_openMP_EHMC_sampling(                const int n_threads_R,
//                                                                                            const int seed_R,
//                                                                                            const int n_iter_R,
//                                                                                            const bool iter_one_by_one,
//                                                                                            const bool partitioned_HMC_R,
//                                                                                            const std::string Model_type_R,
//                                                                                            const bool sample_nuisance_R,
//                                                                                            const bool force_autodiff_R,
//                                                                                            const bool force_PartialLog_R,
//                                                                                            const bool multi_attempts_R,
//                                                                                            const int n_nuisance_to_track,
//                                                                                            const Eigen::Matrix<double, -1, -1>  &theta_main_vectors_all_chains_input_from_R,
//                                                                                            const Eigen::Matrix<double, -1, -1>  &theta_us_vectors_all_chains_input_from_R,
//                                                                                            const Eigen::Matrix<int, -1, -1> &y_Eigen_R,
//                                                                                            const Rcpp::List &Model_args_as_Rcpp_List,  ///// ALWAYS read-only
//                                                                                            const Rcpp::List &EHMC_args_as_Rcpp_List,
//                                                                                            const Rcpp::List &EHMC_Metric_as_Rcpp_List
//  ) {
// 
// 
// 
//    // key dimensions
//    const int n_params_main = theta_main_vectors_all_chains_input_from_R.rows();
//    const int n_us = theta_us_vectors_all_chains_input_from_R.rows();
// 
//    // create EMPTY OUTPUT / containers* to be filled (each col filled from different thread w/ each col corresponding to a different chain)
//    Eigen::Matrix<double, -1, -1>  theta_main_vectors_all_chains_output_to_R =   (theta_main_vectors_all_chains_input_from_R);   // write to this
//    Eigen::Matrix<double, -1, -1>  other_main_out_vector_all_chains_output_to_R(10, n_threads_R);  // write to this
// 
//    /// nuisance
//    Eigen::Matrix<double, -1, -1>  theta_us_vectors_all_chains_output_to_R  =  (theta_us_vectors_all_chains_input_from_R) ; //// .cast<double>();
//    Eigen::Matrix<double, -1, -1>  other_us_out_vector_all_chains_output_to_R(10, n_threads_R);  // write to this
// 
//    //// convert lists to C++ structs
//    const Model_fn_args_struct     Model_args_as_cpp_struct =   convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List); ///// ALWAYS read-only
//    const EHMC_fn_args_struct      EHMC_args_as_cpp_struct =    convert_R_List_EHMC_fn_args_struct(EHMC_args_as_Rcpp_List);
//    const EHMC_Metric_struct       EHMC_Metric_as_cpp_struct =  convert_R_List_EHMC_Metric_struct(EHMC_Metric_as_Rcpp_List);
//    // ////// replicate these structs for thread-safety as we will be modifying them for burnin
//    std::vector<EHMC_fn_args_struct> EHMC_args_as_cpp_struct_copies_R =  replicate_EHMC_fn_args_struct(EHMC_args_as_cpp_struct, n_threads_R);
// 
//    std::vector<Eigen::Matrix<double, -1, -1>> trace_output = vec_of_mats<double>(n_params_main, n_iter_R, n_threads_R);
//    std::vector<Eigen::Matrix<double, -1, -1>> trace_output_divs = vec_of_mats<double>(1, n_iter_R, n_threads_R);
//    std::vector<Eigen::Matrix<double, -1, -1>> trace_output_nuisance = vec_of_mats<double>(n_nuisance_to_track, n_iter_R, n_threads_R);
//    
//    const int N = Model_args_as_cpp_struct.N;
//    std::vector<Eigen::Matrix<float, -1, -1>> trace_output_log_lik = vec_of_mats<float>(N, n_iter_R, n_threads_R);
//    
//          // call openmp function
//           EHMC_sampling_openmp(    n_threads_R,
//                                    seed_R,
//                                    n_iter_R,
//                                    partitioned_HMC_R,
//                                    Model_type_R,
//                                    sample_nuisance_R,
//                                    force_autodiff_R,
//                                    force_PartialLog_R,
//                                    multi_attempts_R,
//                                    ///// inputs
//                                    theta_main_vectors_all_chains_output_to_R,
//                                    theta_us_vectors_all_chains_output_to_R,
//                                    ///// outputs (main)
//                                    trace_output,
//                                    other_main_out_vector_all_chains_output_to_R,
//                                    ///// outputs (nuisance)
//                                    other_us_out_vector_all_chains_output_to_R,
//                                    ///// data
//                                    y_Eigen_R,
//                                    ///// structs
//                                    Model_args_as_cpp_struct, ///// ALWAYS read-only
//                                    EHMC_Metric_as_cpp_struct,
//                                    EHMC_args_as_cpp_struct_copies_R,
//                                    trace_output_divs,
//                                    n_nuisance_to_track,
//                                    trace_output_nuisance,
//                                    trace_output_log_lik);
// 
//          // Return results
//          return Rcpp::List::create(trace_output,
//                                    trace_output_divs,
//                                    trace_output_nuisance,
//                                    theta_main_vectors_all_chains_output_to_R,
//                                    theta_us_vectors_all_chains_output_to_R,
//                                    trace_output_log_lik);
// 
// 
//  }
// 











// [[Rcpp::export]]
Rcpp::List                                        fn_R_RcppParallel_EHMC_single_iter_burnin(  int n_threads_R,
                                                                                              int seed_R,
                                                                                              int n_iter_R,
                                                                                              int n_adapt,
                                                                                              const bool burnin_indicator,
                                                                                              std::string Model_type_R,
                                                                                              bool sample_nuisance_R,
                                                                                              bool force_autodiff_R,
                                                                                              bool force_PartialLog_R,
                                                                                              bool multi_attempts_R,
                                                                                              const int n_nuisance_to_track,
                                                                                              const double max_eps_main,
                                                                                              const double max_eps_us,
                                                                                              bool partitioned_HMC_R,
                                                                                              const std::string metric_type_main,
                                                                                              double shrinkage_factor,
                                                                                              const std::string metric_type_nuisance,
                                                                                              const double tau_main_target,
                                                                                              const double tau_us_target,
                                                                                              const int clip_iter,
                                                                                              const int gap,
                                                                                              const bool main_L_manual,
                                                                                              const bool us_L_manual,
                                                                                              const int L_main_if_manual,
                                                                                              const int L_us_if_manual,
                                                                                              const int max_L,
                                                                                              const double tau_mult,
                                                                                              const double ratio_M_us,
                                                                                              const double ratio_Hess_main,
                                                                                              const int M_interval_width,
                                                                                              Eigen::Matrix<double, -1, -1>  theta_main_vectors_all_chains_input_from_R,
                                                                                              Eigen::Matrix<double, -1, -1>  theta_us_vectors_all_chains_input_from_R,
                                                                                              const Eigen::Matrix<int, -1, -1> y_Eigen_R,
                                                                                              const Rcpp::List Model_args_as_Rcpp_List,  ///// ALWAYS read-only
                                                                                              Rcpp::List EHMC_args_as_Rcpp_List,
                                                                                              Rcpp::List EHMC_Metric_as_Rcpp_List,
                                                                                              Rcpp::List EHMC_burnin_as_Rcpp_List
) {



  // key dimensions
  const int n_params_main = theta_main_vectors_all_chains_input_from_R.rows();
  const int n_us = theta_us_vectors_all_chains_input_from_R.rows();

  // create EMPTY OUTPUT / containers* to be filled (each col filled from different thread w/ each col corresponding to a different chain)
  NumericMatrix  theta_main_vectors_all_chains_output_to_R =  fn_convert_EigenMat_to_RcppMat_dbl(theta_main_vectors_all_chains_input_from_R);   // write to this
  NumericMatrix  other_main_out_vector_all_chains_output_to_R(10, n_threads_R);  // write to this
  double p_jump_main_R = 0.0;
  int div_main_R = 0;

  /// nuisance
  NumericMatrix  theta_us_vectors_all_chains_output_to_R  = fn_convert_EigenMat_to_RcppMat_dbl(theta_us_vectors_all_chains_input_from_R);
  NumericMatrix  other_us_out_vector_all_chains_output_to_R(10, n_threads_R);  // write to this
  double p_jump_us_R = 0.0;
  int div_us_R = 0;

  //// convert lists to C++ structs
  Model_fn_args_struct     Model_args_as_cpp_struct =   convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List); ///// ALWAYS read-only
  EHMC_fn_args_struct      EHMC_args_as_cpp_struct =    convert_R_List_EHMC_fn_args_struct(EHMC_args_as_Rcpp_List);
  EHMC_Metric_struct       EHMC_Metric_as_cpp_struct =  convert_R_List_EHMC_Metric_struct(EHMC_Metric_as_Rcpp_List);
  EHMC_burnin_struct       EHMC_burnin_as_cpp_struct  = convert_R_List_EHMC_burnin_struct(EHMC_burnin_as_Rcpp_List);

  ////// replicate these structs for thread-safety as we will be modifying them for burnin
  std::vector<EHMC_fn_args_struct> EHMC_args_as_cpp_struct_copies =  replicate_EHMC_fn_args_struct(EHMC_args_as_cpp_struct, n_threads_R);
 // std::vector<EHMC_burnin_struct>  EHMC_burnin_as_cpp_struct_copies_R = replicate_EHMC_burnin_struct(EHMC_burnin_as_cpp_struct, n_threads_R);

  /////// containers for burnin outputs ONLY (not needed for sampling) - stores: theta_0, theta_prop, velocity_0, velocity_prop
  NumericMatrix  theta_main_0_burnin_tau_adapt_all_chains_input_from_R(n_params_main, n_threads_R);
  NumericMatrix  theta_main_prop_burnin_tau_adapt_all_chains_input_from_R(n_params_main, n_threads_R);
  NumericMatrix  velocity_main_0_burnin_tau_adapt_all_chains_input_from_R(n_params_main, n_threads_R);
  NumericMatrix  velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R(n_params_main, n_threads_R);
  NumericMatrix  theta_us_0_burnin_tau_adapt_all_chains_input_from_R(n_us, n_threads_R);
  NumericMatrix  theta_us_prop_burnin_tau_adapt_all_chains_input_from_R(n_us, n_threads_R);
  NumericMatrix  velocity_us_0_burnin_tau_adapt_all_chains_input_from_R(n_us, n_threads_R);
  NumericMatrix  velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R(n_us, n_threads_R);

  int n_iter_for_fn_call = n_iter_R;
  if (burnin_indicator == true) n_iter_for_fn_call = 1;

  ////// make trace containers (as 2D matrix using 3D mapping functions)
  std::vector<NumericMatrix> trace_output = vec_of_mats_Rcpp(n_params_main, n_iter_R, n_threads_R);

  int one =  1;

  // tbb::task_scheduler_init init(n_threads_R);

  // // create worker
  RcppParallel_EHMC_burnin parallel_hmc_test(         n_threads_R,
                                                      seed_R,
                                                      one,
                                                      partitioned_HMC_R,
                                                      Model_type_R,
                                                      sample_nuisance_R,
                                                      force_autodiff_R,
                                                      force_PartialLog_R,
                                                      multi_attempts_R,
                                                      ///// inputs
                                                      theta_main_vectors_all_chains_input_from_R,
                                                      theta_us_vectors_all_chains_input_from_R,
                                                      ///// outputs (main)
                                                    //  trace_output,
                                                      other_main_out_vector_all_chains_output_to_R,
                                                      ///// outputs (nuisance)
                                                      other_us_out_vector_all_chains_output_to_R,
                                                      ///// data
                                                      y_Eigen_R,
                                                      /////// structs
                                                      Model_args_as_cpp_struct, ///// ALWAYS read-only
                                                      EHMC_Metric_as_cpp_struct,
                                                      EHMC_args_as_cpp_struct_copies,
                                                      ////////// burnin-specific stuff
                                                      theta_main_vectors_all_chains_output_to_R,
                                                      theta_us_vectors_all_chains_output_to_R,
                                                      theta_main_0_burnin_tau_adapt_all_chains_input_from_R,
                                                      theta_main_prop_burnin_tau_adapt_all_chains_input_from_R,
                                                      velocity_main_0_burnin_tau_adapt_all_chains_input_from_R,
                                                      velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R,
                                                      theta_us_0_burnin_tau_adapt_all_chains_input_from_R,
                                                      theta_us_prop_burnin_tau_adapt_all_chains_input_from_R,
                                                      velocity_us_0_burnin_tau_adapt_all_chains_input_from_R,
                                                      velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R,
                                                      EHMC_burnin_as_cpp_struct);

 // parallel_hmc_test.reset();


  parallelFor(0, n_threads_R, parallel_hmc_test);     //// Call parallelFor

  // if (Model_type_R == "Stan") {
  //   //// destroy model to free memory once finished (destroys dummy model if not using BridgeStan)
  //   fn_bs_destroy_Stan_model(Model_args_as_cpp_struct.bs_model_ptr,
  //                            Model_args_as_cpp_struct.bs_handle);
  // }


  if (   (burnin_indicator == false)   ) {


    // Return results
    return Rcpp::List::create(
      ////// main outputs for main params & nuisance
      Rcpp::wrap(trace_output), // theta
      Rcpp::wrap(theta_main_vectors_all_chains_output_to_R),
      Rcpp::wrap(other_main_out_vector_all_chains_output_to_R),
      Rcpp::wrap(theta_us_vectors_all_chains_output_to_R), // 3 // theta
      Rcpp::wrap(other_us_out_vector_all_chains_output_to_R),
      //////
      theta_main_0_burnin_tau_adapt_all_chains_input_from_R,
      theta_main_prop_burnin_tau_adapt_all_chains_input_from_R, // 10
      velocity_main_0_burnin_tau_adapt_all_chains_input_from_R,
      velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R, // 12
      theta_us_0_burnin_tau_adapt_all_chains_input_from_R,
      theta_us_prop_burnin_tau_adapt_all_chains_input_from_R, // 14
      velocity_us_0_burnin_tau_adapt_all_chains_input_from_R,
      velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R, // 16
      //////
      EHMC_Metric_as_cpp_struct.M_dense_main,
      EHMC_Metric_as_cpp_struct.M_inv_dense_main,
      EHMC_Metric_as_cpp_struct.M_inv_dense_main_chol,
      EHMC_Metric_as_cpp_struct.M_inv_us_vec
    );


  }


 //
 //
 // if (burnin_indicator == true) {
 //
 //
 //
 //    ////////////   ------  Begin burnin / warmup   -------------------------------------------------------------------------
 //    //// make containers to re-use in iteration (ii) loop
 //    Eigen::Matrix<double, -1, -1>  theta_main_vectors_all_chains_Eigen =   Rcpp::as<Eigen::Matrix<double, -1, -1>>(theta_main_vectors_all_chains_output_to_R)  ;
 //    Eigen::Matrix<double, -1, 1>   theta_vec_current_main  = theta_main_vectors_all_chains_Eigen.rowwise().mean();
 //    Eigen::Matrix<double, -1, 1>   snaper_m_vec_main = theta_vec_current_main;
 //    Eigen::Matrix<double, -1, 1>   snaper_s_vec_main_empirical =  EHMC_burnin_as_cpp_struct.snaper_s_vec_main_empirical;
 //    Eigen::Matrix<double, -1, 1>   snaper_w_vec_main  = EHMC_burnin_as_cpp_struct.snaper_w_vec_main;
 //    double eigen_max_main =  EHMC_burnin_as_cpp_struct.eigen_max_main;
 //    Eigen::Matrix<double, -1, 1>   eigen_vector_main  =  EHMC_burnin_as_cpp_struct.eigen_vector_main;
 //
 //
 //    Eigen::Matrix<double, -1, -1>  theta_us_vectors_all_chains_Eigen =     Rcpp::as<Eigen::Matrix<double, -1, -1>>(theta_us_vectors_all_chains_output_to_R)  ;
 //    Eigen::Matrix<double, -1, 1>   theta_vec_current_us  = theta_us_vectors_all_chains_Eigen.rowwise().mean();
 //    Eigen::Matrix<double, -1, 1>   snaper_m_vec_us =  theta_vec_current_us;
 //    Eigen::Matrix<double, -1, 1>   snaper_s_vec_us_empirical  =  EHMC_burnin_as_cpp_struct.snaper_s_vec_us_empirical;
 //    Eigen::Matrix<double, -1, 1>   snaper_w_vec_us  =  EHMC_burnin_as_cpp_struct.snaper_w_vec_us;
 //    double eigen_max_us =   EHMC_burnin_as_cpp_struct.eigen_max_us;
 //    Eigen::Matrix<double, -1, 1>   eigen_vector_us  = EHMC_burnin_as_cpp_struct.eigen_vector_us;
 //
 //    Eigen::Matrix<double, -1, 1>   tau_main_per_chain_to_avg_vec  = Eigen::Matrix<double, -1, 1>::Zero(n_threads_R);
 //    Eigen::Matrix<double, -1, 1>   tau_m_adam_main_per_chain_to_avg_vec  = Eigen::Matrix<double, -1, 1>::Zero(n_threads_R);
 //    Eigen::Matrix<double, -1, 1>   tau_v_adam_main_per_chain_to_avg_vec  = Eigen::Matrix<double, -1, 1>::Zero(n_threads_R);
 //
 //    Eigen::Matrix<double, -1, 1>   tau_us_per_chain_to_avg_vec  = Eigen::Matrix<double, -1, 1>::Zero(n_threads_R);
 //    Eigen::Matrix<double, -1, 1>   tau_m_adam_us_per_chain_to_avg_vec  = Eigen::Matrix<double, -1, 1>::Zero(n_threads_R);
 //    Eigen::Matrix<double, -1, 1>   tau_v_adam_us_per_chain_to_avg_vec  = Eigen::Matrix<double, -1, 1>::Zero(n_threads_R);
 //
 //    ///// set ADAM hyper-params for eps / tau adaptation
 //    const double beta1_adam = 0.00; // ADAM hyperparameter 1
 //    const double beta2_adam = 0.95; // ADAM hyperparameter 2
 //    const double eps_adam = 1e-8; // ADAM "eps" for numerical stability
 //    const double kappa = 8.0;
 //    //double eta_m = 1.0/(std::ceil(1.0/kappa) + 1.0);
 //
 //    int n_iter_adaptation_window = 1; /// doing BCA @ every iteration!
 //
 //    ///// for main
 //    const double LR_main = EHMC_burnin_as_cpp_struct.LR_main;  ///   shared between the K chains
 //    const double adapt_delta_main =    EHMC_burnin_as_cpp_struct.adapt_delta_main; ///   shared between the K chains
 //
 //    Eigen::Matrix<double, -1, -1> M_dense_main = EHMC_Metric_as_cpp_struct.M_dense_main;
 //    Eigen::Matrix<double, -1, -1> M_inv_dense_main = EHMC_Metric_as_cpp_struct.M_inv_dense_main;
 //    Eigen::Matrix<double, -1, -1> M_inv_dense_main_chol = EHMC_Metric_as_cpp_struct.M_inv_dense_main_chol;
 //    Eigen::Matrix<double, -1, -1>  M_dense_main_Chol = ( M_dense_main.llt().matrixL() ).toDenseMatrix().matrix() ;
 //    Eigen::Matrix<double, -1, -1>  M_dense_sqrt = M_dense_main_Chol;// sqrtm(M_dense_main); // compute mass matrix sqrt (for burnin adaptation)
 //
 //    ///// for nuisance
 //    const double LR_us = EHMC_burnin_as_cpp_struct.LR_us;  ///   shared between the K chains
 //    const double adapt_delta_us =    EHMC_burnin_as_cpp_struct.adapt_delta_us; ///   shared between the K chains
 //
 //
 //    double tau_us_prop, tau_main_prop;
 //    double L_main, L_us;
 //
 //    Eigen::Matrix<double, -1, 1> M_inv_us_vec =  EHMC_Metric_as_cpp_struct.M_inv_us_vec;
 //    Eigen::Matrix<double, -1, 1> sqrt_M_us_vec = EHMC_burnin_as_cpp_struct.sqrt_M_us_vec;
 //
 //    const int N = y_Eigen_R.rows();
 //    const int n_tests = y_Eigen_R.cols();
 //
 //    double divs_main = 0.0;
 //    double p_jump_main_mean = 0.0;
 //
 //    Eigen::Matrix<double, -1, -1> empicical_cov_main = EHMC_Metric_as_cpp_struct.M_inv_dense_main;
 //
 //    double tau_main, tau_m_adam_main, tau_v_adam_main;
 //    double eps_main, eps_m_adam_main, eps_v_adam_main;
 //    double tau_us, tau_m_adam_us, tau_v_adam_us;
 //    double eps_us, eps_m_adam_us, eps_v_adam_us;
 //
 //    tau_main = EHMC_args_as_cpp_struct.tau_main;
 //    eps_main = EHMC_args_as_cpp_struct.eps_main;
 //    tau_us =   EHMC_args_as_cpp_struct.tau_us;
 //    eps_us = EHMC_args_as_cpp_struct.eps_us;
 //
 //    tau_m_adam_main = EHMC_burnin_as_cpp_struct.tau_m_adam_main;
 //    tau_v_adam_main = EHMC_burnin_as_cpp_struct.tau_v_adam_main;
 //    eps_m_adam_main = EHMC_burnin_as_cpp_struct.eps_m_adam_main;
 //    eps_v_adam_main = EHMC_burnin_as_cpp_struct.eps_v_adam_main;
 //
 //    tau_m_adam_us = EHMC_burnin_as_cpp_struct.tau_m_adam_us;
 //    tau_v_adam_us = EHMC_burnin_as_cpp_struct.tau_v_adam_us;
 //    eps_m_adam_us = EHMC_burnin_as_cpp_struct.eps_m_adam_us;
 //    eps_v_adam_us = EHMC_burnin_as_cpp_struct.eps_v_adam_us;
 //
 //    const std::string Hessian_string =  "Hessian";
 //    const std::string Empirical_string =  "Empirical";
 //
 //
 //    const int n_burnin = n_iter_R;
 //    if (n_adapt == 0) {
 //      n_adapt =  n_burnin - std::round(n_burnin/10);
 //    }
 //
 //
 //    ///// --------  start of burnin iterations   ----------------------------------------------------------------------------------------------------------------
 //    for (int ii = 0; ii < n_iter_R; ++ii) {
 //
 //      if (ii < n_adapt) {
 //
 //              if (ii < clip_iter) { /// MALA up until clip_iter
 //
 //                tau_us =   1.0 * eps_us;
 //                tau_main = 1.0 * eps_main;
 //
 //                L_main = tau_main / eps_main;
 //                L_us = tau_us / eps_us;
 //
 //              } else  {
 //
 //                      try {
 //
 //                              double clip_iter_dbl = static_cast<double>(clip_iter);
 //                              double gap_dbl =       static_cast<double>(gap);
 //                              double ii_dbl =   static_cast<double>(ii);
 //
 //                              double clip_iter_p_gap_div_4 =    static_cast<double>(std::round( (clip_iter_dbl + gap_dbl/4.0 )) );
 //                              double clip_iter_p_gap_div_2 =    static_cast<double>(std::round( (clip_iter_dbl + gap_dbl/2.0 )) );
 //                              double clip_iter_p_gap_div_1p33 = static_cast<double>(std::round( (clip_iter_dbl + gap_dbl/1.3333333333333333333333333 )) );
 //                              double clip_iter_p_gap_m1 = clip_iter + gap - 1.0;
 //
 //                              double tau_inc_1_main =  ( (0.50 * tau_mult * std::sqrt(eigen_max_main) ));
 //                              double tau_inc_2_main =  ( (0.75 * tau_mult * std::sqrt(eigen_max_main) ));
 //                              double tau_inc_3_main =  ( (tau_mult * std::sqrt(eigen_max_main) ));
 //
 //                              double tau_inc_1_us =   (0.50 * tau_mult * std::sqrt(eigen_max_us) );
 //                              double tau_inc_2_us =   (0.75 * tau_mult * std::sqrt(eigen_max_us) );
 //                              double tau_inc_3_us =   (tau_mult * std::sqrt(eigen_max_us) );
 //
 //                              eps_us =   EHMC_args_as_cpp_struct_copies[0].eps_us;
 //                              eps_main = EHMC_args_as_cpp_struct_copies[0].eps_main;
 //
 //
 //                              if (ii_dbl <= clip_iter_p_gap_div_4) {
 //                                tau_main =  5.0 * eps_main;
 //                                tau_us   =  5.0   * eps_us;
 //                              } else if ( (ii_dbl > clip_iter_p_gap_div_4)  &&  (ii_dbl <= clip_iter_p_gap_div_2) )  {
 //                                tau_main =  10.0  * eps_main;
 //                                tau_us   =  10.0    * eps_us;
 //                              } else if ( (ii_dbl > clip_iter_p_gap_div_2) && (ii_dbl <= clip_iter_p_gap_div_1p33) ) {
 //                                tau_main = tau_inc_1_main;
 //                                tau_us   = tau_inc_1_us;
 //                                if (tau_main_target != 0) {
 //                                  tau_main = tau_main_target;
 //                                }
 //                                if (tau_us_target != 0) {
 //                                  tau_us = tau_us_target;
 //                                }
 //                              } else if ( (ii_dbl > clip_iter_p_gap_div_1p33) && (ii_dbl <= clip_iter_p_gap_m1) ) {
 //                                tau_main = tau_inc_2_main;
 //                                tau_us   = tau_inc_2_us;
 //                                if (tau_main_target != 0) {
 //                                  tau_main = tau_main_target;
 //                                }
 //                                if (tau_us_target != 0) {
 //                                  tau_us = tau_us_target;
 //                                }
 //                              } else if ( (ii_dbl > clip_iter_p_gap_m1) && (ii_dbl <= clip_iter_p_gap_m1 + 50) ) {
 //                                tau_main =  tau_inc_3_main;
 //                                tau_us =  tau_inc_3_us;
 //                                if (tau_main_target != 0) {
 //                                  tau_main = tau_main_target;
 //                                }
 //                                if (tau_us_target != 0) {
 //                                  tau_us = tau_us_target;
 //                                }
 //                              } else {
 //                                tau_main = EHMC_args_as_cpp_struct_copies[0].tau_main;
 //                                tau_us =   EHMC_args_as_cpp_struct_copies[0].tau_us;
 //                              }
 //
 //                      } catch (...) {
 //
 //                      }
 //
 //                      L_main = tau_main / eps_main;
 //                      L_us = tau_us / eps_us;
 //
 //              }
 //
 //              //// limit max tau / L (like Stan's "max_treedepth)
 //              if (tau_main > static_cast<double>(max_L)*eps_main) tau_main =   static_cast<double>(max_L)*eps_main;
 //              if (tau_us >   static_cast<double>(max_L)*eps_us) tau_us       = static_cast<double>(max_L)*eps_us;
 //
 //              if (main_L_manual == true) {
 //                tau_main = L_main_if_manual * eps_main;
 //              }
 //              if (us_L_manual == true) {
 //                tau_us = L_us_if_manual * eps_us;
 //              }
 //
 //              if (tau_main < eps_main) tau_main = eps_main;
 //              if (tau_us < eps_us)     tau_us = eps_us;
 //
 //              for (int kk = 0; kk < n_threads_R; ++kk) {
 //                EHMC_args_as_cpp_struct_copies[kk].tau_us = tau_us;
 //                EHMC_args_as_cpp_struct_copies[kk].tau_main = tau_main;
 //              }
 //
 //      }
 //
 //              if (ii % 10 == 0) {
 //
 //                double pct_complete = 100.0 * (static_cast<double>(ii) / static_cast<double>(n_iter_R));
 //                pct_complete = std::round(pct_complete);  // round to nearest integer if needed
 //              //  Rcpp::Rcout << "Burn-in is " <<  pct_complete << "% complete"  << std::endl;
 //                Rcpp::Rcout << BLUE     <<  "Burn-in is" <<  RESET  <<  pct_complete <<  BLUE << "% complete"  << RESET << std::endl;
 //
 //                Rcpp::Rcout << "L_main = "  << L_main  << std::endl;
 //                Rcpp::Rcout << YELLOW     << "eps_main = "  <<  RESET  << eps_main << std::endl;
 //
 //                Rcpp::Rcout << "L_us = "  << L_us  << std::endl;
 //                Rcpp::Rcout << "eps_us = "  << eps_us  << std::endl;
 //
 //                double sqrt_eigen_max_main = tau_mult * stan::math::sqrt(eigen_max_main);
 //                double sqrt_eigen_max_us = tau_mult *  stan::math::sqrt(eigen_max_us);
 //                Rcpp::Rcout << "tau_mult * sqrt_eigen_max_main = "  <<   sqrt_eigen_max_main  << std::endl;
 //                Rcpp::Rcout << "tau_mult *  sqrt_eigen_max_us = "  <<     sqrt_eigen_max_us  << std::endl;
 //
 //                Rcpp::Rcout << "n_divs (main) = "  <<     divs_main   << std::endl;
 //                Rcpp::Rcout << "p_jump_main_mean = "  <<     p_jump_main_mean   << std::endl;
 //                Rcpp::Rcout << " snaper_m_vec_main(30) = "  <<     snaper_m_vec_main(30)   << std::endl;
 //
 //              }
 //
 //              ////////////////////////////   ----------------  update snaper_m, snaper_s, snaper_w, and eigen_vec/eigen_max  for MAIN PARAMS  ------------------------------------------------------------
 //              if (ii < n_adapt) {
 //                  //// convert to Eigen
 //                  theta_main_vectors_all_chains_Eigen =   Rcpp::as<Eigen::Matrix<double, -1, -1>>(theta_main_vectors_all_chains_output_to_R)  ; /// current value of theta(s)
 //                  theta_vec_current_main =  theta_main_vectors_all_chains_Eigen.rowwise().mean(); /// current value of theta(s) [mean between K chains]
 //                  if (ii < 3) {
 //                    snaper_m_vec_main = theta_vec_current_main;
 //                  }
 //                  /////////   update snaper_m and snaper_s_empirical for MAIN
 //                  Eigen::Matrix<double, -1, -1>   outs_update_snaper_m_and_s_main =       fn_update_snaper_m_and_s( snaper_m_vec_main,  snaper_s_vec_main_empirical,
 //                                                                                                                    theta_vec_current_main, static_cast<double>(ii));
 //
 //                  // update values
 //                  if (is_NaN_or_Inf_Eigen(outs_update_snaper_m_and_s_main) == false) {
 //                    snaper_m_vec_main = outs_update_snaper_m_and_s_main.col(0);
 //                    snaper_s_vec_main_empirical = outs_update_snaper_m_and_s_main.col(1);
 //                  }
 //
 //                  ////////  update snaper_w   for MAIN
 //                  Eigen::Matrix<double, -1, 1> snaper_w_vec_main_prop           =  fn_update_snaper_w_dense_M(      snaper_w_vec_main, eigen_vector_main, eigen_max_main,
 //                                                                                                                    theta_vec_current_main,  snaper_m_vec_main,
 //                                                                                                                    static_cast<double>(ii),
 //                                                                                                                    M_dense_sqrt);
 //                   if (!(is_NaN_or_Inf_Eigen(snaper_w_vec_main_prop))) {
 //                     snaper_w_vec_main = snaper_w_vec_main_prop;
 //                   }
 //
 //                  Eigen::Matrix<double, -1, 1>  outs_update_eigen_max_and_eigen_vec_main =  fn_update_eigen_max_and_eigen_vec( eigen_max_main,   eigen_vector_main,  snaper_w_vec_main);
 //                  // update values
 //                  if (!(is_NaN_or_Inf_Eigen(outs_update_eigen_max_and_eigen_vec_main))) {
 //                      eigen_max_main =  std::max(0.0001, outs_update_eigen_max_and_eigen_vec_main(0));
 //                      eigen_vector_main = outs_update_eigen_max_and_eigen_vec_main.tail(n_params_main);
 //                  }
 //              }
 //              ////////////////////////////   ----------------   update snaper_m, snaper_s, snaper_w, and eigen_vec/eigen_max  for NUISANCE PARAMS  ------------------------------------------------------------
 //              if (ii < n_adapt) {
 //                  //// convert to Eigen
 //                  theta_us_vectors_all_chains_Eigen =     Rcpp::as<Eigen::Matrix<double, -1, -1>>(theta_us_vectors_all_chains_output_to_R)  ; /// current value of theta(s)
 //                  theta_vec_current_us =  theta_us_vectors_all_chains_Eigen.rowwise().mean(); /// current value of theta(s) [mean between K chains]
 //                  if (ii < 3) {
 //                    snaper_m_vec_us = theta_vec_current_us;
 //                  }
 //                  /////////   update snaper_m and snaper_s_empirical for us
 //                  Eigen::Matrix<double, -1, -1>  outs_update_snaper_m_and_s_us =       fn_update_snaper_m_and_s( snaper_m_vec_us,  snaper_s_vec_us_empirical,
 //                                                                                                                 theta_vec_current_us, static_cast<double>(ii));
 //
 //                  // update values
 //                  if (is_NaN_or_Inf_Eigen(outs_update_snaper_m_and_s_us) == false) {
 //                      snaper_m_vec_us = outs_update_snaper_m_and_s_us.col(0);
 //                      snaper_s_vec_us_empirical = outs_update_snaper_m_and_s_us.col(1);
 //                  }
 //
 //                  ////////  update snaper_w   for us
 //                  Eigen::Matrix<double, -1, 1>  snaper_w_vec_us_prop          =  fn_update_snaper_w_diag_M(   snaper_w_vec_us, eigen_vector_us, eigen_max_us,
 //                                                                                                              theta_vec_current_us, snaper_m_vec_us,
 //                                                                                                              static_cast<double>(ii),
 //                                                                                                              sqrt_M_us_vec);
 //                  if (is_NaN_or_Inf_Eigen(snaper_w_vec_us_prop) == false) {
 //                    snaper_w_vec_us = snaper_w_vec_us_prop;
 //                  }
 //                  Eigen::Matrix<double, -1, 1>   outs_update_eigen_max_and_eigen_vec_us =  fn_update_eigen_max_and_eigen_vec( eigen_max_us, eigen_vector_us, snaper_w_vec_us);
 //                    // update values
 //                    if (!(is_NaN_or_Inf_Eigen(outs_update_eigen_max_and_eigen_vec_us))) {
 //                      eigen_max_us =  std::max(0.0001, outs_update_eigen_max_and_eigen_vec_us(0));
 //                      eigen_vector_us = outs_update_eigen_max_and_eigen_vec_us.tail(n_us);
 //                    }
 //              }
 //              ////////  ----- Adapt mass matrix (M) for MAIN PARAMS  -------------------------------------------------------------------------------------------------------
 //              if (ii < n_adapt) {
 //                if (ii > 2) {
 //                  empicical_cov_main =  fn_update_empirical_covariance(empicical_cov_main, snaper_m_vec_main, theta_vec_current_main, static_cast<double>(ii));
 //                } else {
 //                  empicical_cov_main = M_inv_dense_main;
 //                }
 //              }
 //
 //              if (ii < n_adapt) {
 //                if  ( is_multiple(ii, M_interval_width) )  {    //  &&  (ii < static_cast<int>(std::round( static_cast<double>(n_adapt) * 0.90))  ) ) ) ) {
 //
 //                  if (metric_type_main ==  Hessian_string) {
 //
 //
 //                    if (ii < 0.5 * n_burnin)  {
 //                      shrinkage_factor = 1.00;
 //                    } else {
 //                      shrinkage_factor = 0.50;
 //                    }
 //
 //
 //                          try {
 //
 //                            Eigen::Matrix<double, -1, -1> M_dense_main = EHMC_Metric_as_cpp_struct.M_dense_main;
 //                            Eigen::Matrix<double, -1, -1> M_inv_dense_main = EHMC_Metric_as_cpp_struct.M_inv_dense_main;
 //                            Eigen::Matrix<double, -1, -1> M_inv_dense_main_chol = EHMC_Metric_as_cpp_struct.M_inv_dense_main_chol;
 //
 //                            update_M_dense_main_Hessian_InPlace(   M_dense_main,  M_inv_dense_main,  M_inv_dense_main_chol,
 //                                                                   shrinkage_factor,
 //                                                                   ratio_Hess_main,
 //                                                                   M_interval_width,
 //                                                                   0.0001,
 //                                                                   Model_type_R,
 //                                                                   true, // force_ad
 //                                                                   false, // force_partialLog
 //                                                                   snaper_m_vec_main,  snaper_m_vec_us,
 //                                                                   y_Eigen_R,
 //                                                                   Model_args_as_cpp_struct,
 //                                                                   static_cast<double>(ii),
 //                                                                   static_cast<double>(n_adapt),
 //                                                                   metric_type_main);
 //
 //                            if (is_NaN_or_Inf_Eigen(M_dense_main) == false) {
 //
 //                              EHMC_Metric_as_cpp_struct.M_dense_main =          M_dense_main;
 //                              EHMC_Metric_as_cpp_struct.M_inv_dense_main =      M_inv_dense_main;
 //                              EHMC_Metric_as_cpp_struct.M_inv_dense_main_chol = M_inv_dense_main_chol;
 //                              /// now update sqrt of M_dense_main
 //                              Eigen::Matrix<double, -1, -1> M_dense_main_Chol =  Rcpp_Chol(M_dense_main) ; // (  EHMC_Metric_as_cpp_struct.M_dense_main.llt().matrixL() ).toDenseMatrix().matrix() ;
 //                              Eigen::Matrix<double, -1, -1> M_dense_sqrt =   M_dense_main_Chol; // sqrtm( EHMC_Metric_as_cpp_struct.M_dense_main);//  M_dense_main_Chol;//   compute mass matrix sqrt (for burnin adaptation)
 //                              EHMC_burnin_as_cpp_struct.M_dense_sqrt  = M_dense_sqrt;
 //                            }
 //
 //                          } catch (...) {
 //
 //                            Rcpp::Rcout << "Failed to update M_dense_main using Hessian (num diff)! - will attempt using empirical (co)variance" << std::endl;
 //
 //                          }
 //
 //                  } else if (metric_type_main == Empirical_string)  {
 //
 //                    try {
 //                              Eigen::Matrix<double, -1, -1> M_dense_main = EHMC_Metric_as_cpp_struct.M_dense_main;
 //                              Eigen::Matrix<double, -1, -1> M_inv_dense_main = EHMC_Metric_as_cpp_struct.M_inv_dense_main;
 //                              Eigen::Matrix<double, -1, -1> M_inv_dense_main_chol = EHMC_Metric_as_cpp_struct.M_inv_dense_main_chol;
 //
 //                              M_inv_dense_main =   (1.0 - ratio_Hess_main)  * M_inv_dense_main + ratio_Hess_main *  empicical_cov_main ;
 //                              M_inv_dense_main = near_PD(M_inv_dense_main);
 //                              M_dense_main = M_inv_dense_main.inverse();
 //                              M_inv_dense_main_chol = Rcpp_Chol(M_inv_dense_main);
 //
 //                              if (is_NaN_or_Inf_Eigen(M_dense_main) == false) {
 //
 //                                EHMC_Metric_as_cpp_struct.M_dense_main =          M_dense_main;
 //                                EHMC_Metric_as_cpp_struct.M_inv_dense_main =      M_inv_dense_main;
 //                                EHMC_Metric_as_cpp_struct.M_inv_dense_main_chol = M_inv_dense_main_chol;
 //                                /// now update sqrt of M_dense_main
 //                                Eigen::Matrix<double, -1, -1> M_dense_main_Chol =  Rcpp_Chol(M_dense_main) ; // (  EHMC_Metric_as_cpp_struct.M_dense_main.llt().matrixL() ).toDenseMatrix().matrix() ;
 //                                Eigen::Matrix<double, -1, -1> M_dense_sqrt =   M_dense_main_Chol; // sqrtm( EHMC_Metric_as_cpp_struct.M_dense_main);//  M_dense_main_Chol;//   compute mass matrix sqrt (for burnin adaptation)
 //                                EHMC_burnin_as_cpp_struct.M_dense_sqrt  = M_dense_sqrt;
 //                              }
 //
 //                            } catch (...) {
 //
 //                              Rcpp::Rcout << "Failed to update M_dense_main using Hessian (num diff)! - will attempt using empirical (co)variance" << std::endl;
 //
 //                            }
 //
 //                  } else  {   //// unit metric
 //
 //
 //                  }
 //                }
 //              }
 //             ////////  ----- Adapt mass matrix (M) for NUISANCE PARAMS  (using diagonal M w/ empirical variances) -------------------------------------------------------
 //             if (ii < n_adapt) {
 //               if  ( is_multiple(ii, M_interval_width) )  {
 //
 //                     if (metric_type_nuisance ==  Empirical_string) {
 //
 //                           // double max_s_empirical_us =   snaper_s_vec_us_empirical.array().maxCoeff();
 //                            M_inv_us_vec.array()  =    (1.0 - ratio_M_us) * M_inv_us_vec.array()   +    ratio_M_us *   snaper_s_vec_us_empirical.array();
 //                            sqrt_M_us_vec.array() = (1.0 / sqrt_M_us_vec.array()).sqrt();
 //
 //                            //// update structs
 //                            EHMC_Metric_as_cpp_struct.M_inv_us_vec = M_inv_us_vec;
 //                            EHMC_burnin_as_cpp_struct.sqrt_M_us_vec = sqrt_M_us_vec;
 //
 //                     } else if (metric_type_nuisance ==  Hessian_string) {
 //
 //                       Eigen::Matrix<double, -1, 1>  Hessian_diag_nuisance = fn_diag_hessian_us_only_manual(snaper_m_vec_main, snaper_m_vec_us, y_Eigen_R, Model_args_as_cpp_struct);
 //                       Hessian_diag_nuisance.array() =   Hessian_diag_nuisance.array().abs();
 //                       clean_vector(Hessian_diag_nuisance);
 //                       Eigen::Matrix<double, -1, 1>  Hessian_inv_diag_nuisance = ( 1.0 / Hessian_diag_nuisance.array() ).matrix() ;
 //
 //                       M_inv_us_vec.array()  =    (1.0 - ratio_M_us) * M_inv_us_vec.array()   +   ratio_M_us *   Hessian_inv_diag_nuisance.array();
 //                       sqrt_M_us_vec.array() = (1.0 / sqrt_M_us_vec.array()).sqrt();
 //
 //                       //// update structs
 //                       EHMC_Metric_as_cpp_struct.M_inv_us_vec = M_inv_us_vec;
 //                       EHMC_burnin_as_cpp_struct.sqrt_M_us_vec = sqrt_M_us_vec;
 //
 //                     } else  {  //// unit metric
 //
 //                     }
 //                 }
 //              }
 //              ////////////////////////////   ----------------  adapt epsilon for MAIN-----------------------------------------------------------------------------
 //              if (ii < n_adapt) {
 //
 //                p_jump_main_mean = 0.0; // taking the MEAN between the K chains
 //                divs_main  = 0.0; // taking the MEAN between the K chains
 //                for (int kk = 0; kk < n_threads_R; ++kk) {
 //                  p_jump_main_mean += other_main_out_vector_all_chains_output_to_R(0, kk) ;
 //                  divs_main   += other_main_out_vector_all_chains_output_to_R(1, kk) ;
 //                }
 //                p_jump_main_mean = p_jump_main_mean / static_cast<double>(n_threads_R); /// compute mean
 //
 //                Eigen::Matrix<double, -1, 1>   out_vec_eps_main  = adapt_eps_ADAM(  eps_main,
 //                                                                                    eps_m_adam_main,
 //                                                                                    eps_v_adam_main,
 //                                                                                    static_cast<double>(ii),   static_cast<double>(n_burnin) ,
 //                                                                                    LR_main,
 //                                                                                    p_jump_main_mean,
 //                                                                                    adapt_delta_main,
 //                                                                                    beta1_adam, beta2_adam, eps_adam);
 //                if (is_NaN_or_Inf_Eigen(out_vec_eps_main) == false) {
 //                    eps_main =         out_vec_eps_main(0);
 //                    eps_m_adam_main =  out_vec_eps_main(1);
 //                    eps_v_adam_main =  out_vec_eps_main(2);
 //                    /// update values in the struct (same for all chains)
 //                    for (int kk = 0; kk < n_threads_R; ++kk) {
 //                      EHMC_args_as_cpp_struct_copies[kk].eps_main =             std::min(max_eps_main, out_vec_eps_main(0));
 //                    }
 //                }
 //              }
 //              // //////////////////////////   ----------------  adapt epsilon   for nuisance-----------------------------------------------------------------------------
 //              if (ii < n_adapt) {
 //
 //                double p_jump_us_mean = 0.0; // taking the MEAN between the K chains
 //                for (int kk = 0; kk < n_threads_R; ++kk) {
 //                  p_jump_us_mean += other_us_out_vector_all_chains_output_to_R(0, kk) ;
 //                }
 //                p_jump_us_mean = p_jump_us_mean / static_cast<double>(n_threads_R); /// compute mean
 //
 //                Eigen::Matrix<double, -1, 1>  out_vec_eps_us = adapt_eps_ADAM(  eps_us,
 //                                                                                eps_m_adam_us,
 //                                                                                eps_v_adam_us,
 //                                                                                static_cast<double>(ii),   static_cast<double>(n_burnin) ,
 //                                                                                LR_us,
 //                                                                                p_jump_us_mean,
 //                                                                                adapt_delta_us,
 //                                                                                beta1_adam, beta2_adam, eps_adam);
 //                if (is_NaN_or_Inf_Eigen(out_vec_eps_us) == false) {
 //                    eps_us =         out_vec_eps_us(0);
 //                    eps_m_adam_us =  out_vec_eps_us(1);
 //                    eps_v_adam_us =  out_vec_eps_us(2);
 //                    /// update values in the struct (same for all chains)
 //                    for (int kk = 0; kk < n_threads_R; ++kk) {
 //                      EHMC_args_as_cpp_struct_copies[kk].eps_us =            std::min(max_eps_us, out_vec_eps_us(0));
 //                    }
 //                }
 //
 //              }
 //              //////// ----------------  Now update tau for MAIN -----------------------------------------------------------------------------
 //              if (ii < n_adapt) {
 //
 //                tau_main_per_chain_to_avg_vec.setConstant(tau_main);
 //                tau_m_adam_main_per_chain_to_avg_vec.setConstant(tau_m_adam_main);
 //                tau_v_adam_main_per_chain_to_avg_vec.setConstant(tau_v_adam_main);
 //
 //                ////// update tau for main
 //                for (int kk = 0; kk < n_threads_R; ++kk) {
 //
 //                  double tau_main_kk =  other_main_out_vector_all_chains_output_to_R(5, kk);
 //
 //                  Eigen::Matrix<double, -1, 1> ADAM_tau_outs_main   =  fn_update_tau_w_dense_M_ADAM(   eigen_vector_main,
 //                                                                                                       eigen_max_main,
 //                                                                                                       fn_convert_RCppNumMat_Column_to_EigenColVec(theta_main_0_burnin_tau_adapt_all_chains_input_from_R.column(kk)),
 //                                                                                                       fn_convert_RCppNumMat_Column_to_EigenColVec(theta_main_prop_burnin_tau_adapt_all_chains_input_from_R.column(kk)),
 //                                                                                                       snaper_m_vec_main,
 //                                                                                                       fn_convert_RCppNumMat_Column_to_EigenColVec(velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R.column(kk)),
 //                                                                                                       fn_convert_RCppNumMat_Column_to_EigenColVec(velocity_main_0_burnin_tau_adapt_all_chains_input_from_R.column(kk)),
 //                                                                                                       tau_main,  //  being modified
 //                                                                                                       LR_main,
 //                                                                                                       static_cast<double>(ii),   static_cast<double>(n_burnin) ,
 //                                                                                                       M_dense_sqrt,
 //                                                                                                       tau_m_adam_main,  //  being modified
 //                                                                                                       tau_v_adam_main, //  being modified
 //                                                                                                       tau_main_kk);
 //
 //                  if (is_NaN_or_Inf_Eigen(ADAM_tau_outs_main) == false) {
 //                    tau_main_per_chain_to_avg_vec(kk) =        ADAM_tau_outs_main(0);
 //                    tau_m_adam_main_per_chain_to_avg_vec(kk) = ADAM_tau_outs_main(1);
 //                    tau_v_adam_main_per_chain_to_avg_vec(kk) = ADAM_tau_outs_main(2);
 //                  }
 //
 //                }
 //                // compute means between the K chains
 //                tau_main = tau_main_per_chain_to_avg_vec.mean();
 //                tau_m_adam_main = tau_m_adam_main_per_chain_to_avg_vec.mean();
 //                tau_v_adam_main = tau_v_adam_main_per_chain_to_avg_vec.mean();
 //                // update the struct(s) in each chain (to carry over to next iter)
 //
 //                if (main_L_manual == true) {
 //                  tau_main = L_main_if_manual * eps_main;
 //                }
 //
 //                for (int kk = 0; kk < n_threads_R; ++kk) {
 //                  EHMC_args_as_cpp_struct_copies[kk].tau_main = tau_main;
 //                }
 //              }
 //              //////// ----------------  Now update tau for us -----------------------------------------------------------------------------
 //              if (ii < n_adapt) {
 //
 //                tau_us_per_chain_to_avg_vec.setConstant(tau_us);
 //                tau_m_adam_us_per_chain_to_avg_vec.setConstant(tau_m_adam_us);
 //                tau_v_adam_us_per_chain_to_avg_vec.setConstant(tau_v_adam_us);
 //
 //                ////// update tau for us
 //                for (int kk = 0; kk < n_threads_R; ++kk) {
 //
 //                  double tau_us_kk =  other_us_out_vector_all_chains_output_to_R(5, kk);
 //
 //                  Eigen::Matrix<double, -1, 1> ADAM_tau_outs_us   = fn_update_tau_w_diag_M_ADAM( eigen_vector_us,
 //                                                                                                 eigen_max_us,
 //                                                                                                 fn_convert_RCppNumMat_Column_to_EigenColVec(theta_us_0_burnin_tau_adapt_all_chains_input_from_R.column(kk)),
 //                                                                                                 fn_convert_RCppNumMat_Column_to_EigenColVec(theta_us_prop_burnin_tau_adapt_all_chains_input_from_R.column(kk)),
 //                                                                                                 snaper_m_vec_us,
 //                                                                                                 fn_convert_RCppNumMat_Column_to_EigenColVec(velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R.column(kk)),
 //                                                                                                 fn_convert_RCppNumMat_Column_to_EigenColVec(velocity_us_0_burnin_tau_adapt_all_chains_input_from_R.column(kk)),
 //                                                                                                 tau_us,  //  being modified
 //                                                                                                 LR_us,
 //                                                                                                 static_cast<double>(ii),
 //                                                                                                 static_cast<double>(n_burnin) ,
 //                                                                                                 sqrt_M_us_vec,
 //                                                                                                 tau_m_adam_us,  //  being modified
 //                                                                                                 tau_v_adam_us, //  being modified
 //                                                                                                 tau_us_kk);
 //
 //                  if (is_NaN_or_Inf_Eigen(ADAM_tau_outs_us) == false) {
 //                    tau_us_per_chain_to_avg_vec(kk) =        ADAM_tau_outs_us(0);
 //                    tau_m_adam_us_per_chain_to_avg_vec(kk) = ADAM_tau_outs_us(1);
 //                    tau_v_adam_us_per_chain_to_avg_vec(kk) = ADAM_tau_outs_us(2);
 //                  }
 //
 //                }
 //                // compute means between the K chains
 //                tau_us = tau_us_per_chain_to_avg_vec.mean();
 //                tau_m_adam_us = tau_m_adam_us_per_chain_to_avg_vec.mean();
 //                tau_v_adam_us = tau_v_adam_us_per_chain_to_avg_vec.mean();
 //
 //                // update the struct(s) in each chain (to carry over to next iter)
 //                if (us_L_manual == true) {
 //                  tau_us = L_us_if_manual * eps_us;
 //                }
 //                for (int kk = 0; kk < n_threads_R; ++kk) {
 //                  EHMC_args_as_cpp_struct_copies[kk].tau_us = tau_us;
 //                }
 //
 //              }
 //
 //
 //              // // ///////////////////////////   ---------------- PERFORM ITERATION  ------------------------------------------------------------------------------------
 //              theta_main_vectors_all_chains_input_from_R =    (theta_main_vectors_all_chains_Eigen);
 //              theta_us_vectors_all_chains_input_from_R =      (theta_us_vectors_all_chains_Eigen);
 //
 //              try {
 //
 //                int one = 1;
 //
 //                // // create worker
 //                RcppParallel_EHMC_burnin parallel_hmc_test(         n_threads_R,
 //                                                                    seed_R,
 //                                                                    one,
 //                                                                    partitioned_HMC_R,
 //                                                                    Model_type_R,
 //                                                                    sample_nuisance_R,
 //                                                                    force_autodiff_R,
 //                                                                    force_PartialLog_R,
 //                                                                    multi_attempts_R,
 //                                                                    ///// inputs
 //                                                                    theta_main_vectors_all_chains_input_from_R,
 //                                                                    theta_us_vectors_all_chains_input_from_R,
 //                                                                    ///// outputs (main)
 //                                                                    other_main_out_vector_all_chains_output_to_R,
 //                                                                    ///// outputs (nuisance)
 //                                                                    other_us_out_vector_all_chains_output_to_R,
 //                                                                    ///// data
 //                                                                    y_Eigen_R,
 //                                                                    /////// structs
 //                                                                    Model_args_as_cpp_struct, ///// ALWAYS read-only
 //                                                                    EHMC_Metric_as_cpp_struct,
 //                                                                    EHMC_args_as_cpp_struct_copies,
 //                                                                    ////////// burnin-specific stuff
 //                                                                    theta_main_vectors_all_chains_output_to_R,
 //                                                                    theta_us_vectors_all_chains_output_to_R,
 //                                                                    theta_main_0_burnin_tau_adapt_all_chains_input_from_R,
 //                                                                    theta_main_prop_burnin_tau_adapt_all_chains_input_from_R,
 //                                                                    velocity_main_0_burnin_tau_adapt_all_chains_input_from_R,
 //                                                                    velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R,
 //                                                                    theta_us_0_burnin_tau_adapt_all_chains_input_from_R,
 //                                                                    theta_us_prop_burnin_tau_adapt_all_chains_input_from_R,
 //                                                                    velocity_us_0_burnin_tau_adapt_all_chains_input_from_R,
 //                                                                    velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R,
 //                                                                    EHMC_burnin_as_cpp_struct);
 //
 //
 //                  parallelFor(0, n_threads_R, parallel_hmc_test);     //// Call parallelFor
 //
 //              } catch (...) {
 //
 //                Rcpp::Rcout << "Iteration" << static_cast<double>(ii) << "Rejected!" << std::endl;
 //
 //              }
 //
 //
 //
 //    }  ////////// end of iterations / end of adaptation
 //
 //    //// shut down parallel processes
 //
 //      // if (Model_type_R == "Stan") {
 //      //     //// destroy model to free memory once finished (destroys dummy model if not using BridgeStan)
 //      //     fn_bs_destroy_Stan_model(Model_args_as_cpp_struct.bs_model_ptr,
 //      //                              Model_args_as_cpp_struct.bs_handle);
 //      // }
 //
 //
 //   {  /// burnin only
 //
 //          // Return results
 //          return Rcpp::List::create(
 //            ////// main outputs for main params & nuisance
 //            Named("trace_output") = Rcpp::wrap(trace_output), // theta
 //            Named("theta_main_vectors_all_chains_output_to_R") = Rcpp::wrap(theta_main_vectors_all_chains_output_to_R),
 //            Named("other_main_out_vector_all_chains_output_to_R") = Rcpp::wrap(other_main_out_vector_all_chains_output_to_R), // 3
 //            Named("theta_us_vectors_all_chains_output_to_R") = Rcpp::wrap(theta_us_vectors_all_chains_output_to_R),
 //            Named("other_us_out_vector_all_chains_output_to_R") = Rcpp::wrap(other_us_out_vector_all_chains_output_to_R),
 //            //////
 //            theta_main_0_burnin_tau_adapt_all_chains_input_from_R,
 //            theta_main_prop_burnin_tau_adapt_all_chains_input_from_R, // 10
 //            velocity_main_0_burnin_tau_adapt_all_chains_input_from_R,
 //            velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R, // 12
 //            theta_us_0_burnin_tau_adapt_all_chains_input_from_R,
 //            theta_us_prop_burnin_tau_adapt_all_chains_input_from_R, // 14
 //            velocity_us_0_burnin_tau_adapt_all_chains_input_from_R,
 //            velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R, // 16
 //            //////
 //            Named("M_dense_main") = EHMC_Metric_as_cpp_struct.M_dense_main,
 //            Named("M_inv_dense_main") = EHMC_Metric_as_cpp_struct.M_inv_dense_main,
 //            Named("M_inv_dense_main_chol") = EHMC_Metric_as_cpp_struct.M_inv_dense_main_chol,
 //            Named("M_inv_us_vec") = EHMC_Metric_as_cpp_struct.M_inv_us_vec
 //          );
 //
 //    }
 //
 //
 // }
 //

 // Return results
 return Rcpp::List::create(
   ////// main outputs for main params & nuisance
   Rcpp::wrap(trace_output), // theta
   Rcpp::wrap(theta_main_vectors_all_chains_output_to_R),
   Rcpp::wrap(other_main_out_vector_all_chains_output_to_R),
   Rcpp::wrap(theta_us_vectors_all_chains_output_to_R), // 3 // theta
   Rcpp::wrap(other_us_out_vector_all_chains_output_to_R),
   //////
   theta_main_0_burnin_tau_adapt_all_chains_input_from_R,
   theta_main_prop_burnin_tau_adapt_all_chains_input_from_R, // 10
   velocity_main_0_burnin_tau_adapt_all_chains_input_from_R,
   velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R, // 12
   theta_us_0_burnin_tau_adapt_all_chains_input_from_R,
   theta_us_prop_burnin_tau_adapt_all_chains_input_from_R, // 14
   velocity_us_0_burnin_tau_adapt_all_chains_input_from_R,
   velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R, // 16
   //////
   EHMC_Metric_as_cpp_struct.M_dense_main,
   EHMC_Metric_as_cpp_struct.M_inv_dense_main,
   EHMC_Metric_as_cpp_struct.M_inv_dense_main_chol,
   EHMC_Metric_as_cpp_struct.M_inv_us_vec
 );




  }




 




 





// // Benchmark functions -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// 
// //[[Rcpp::export]]
// void eepy_basic_fns_double( int reps = 1000,
//                             int dim  = 1000,
//                             double lower = 0,
//                             double upper = 1,
//                             bool AVX2 = true,
//                             bool AVX512 = true) {
// 
//   Rcpp::Clock clock;
// 
//   Eigen::Array<double, -1, 1>  vec_vals_1(dim);
//   Eigen::Array<double, -1, 1>  vec_vals_2(dim);
//   Eigen::Array<double, -1, 1>  vec_vals_3(dim);
//   Eigen::Array<double, -1, 1>  vec_vals_res(dim);
// 
//   for (int i = 0; i < vec_vals_1.rows(); i++) {
//     vec_vals_1(i) =  R::runif(lower, upper);
//   }
//   vec_vals_2 =  vec_vals_1.log();
//   vec_vals_3 =  vec_vals_1.exp();
// 
// 
//   while (reps -- > 0) {
// 
// 
//     ///////////////////////////////////////////////////  exp  - using doubles
//     //////  standard library fns  (always with checks)
//     clock.tick("mult_2_arrays");
//     vec_vals_res.array() = vec_vals_1.array() * vec_vals_2.array() ;
//     clock.tock("mult_2_arrays");
// 
//     clock.tick("mult_3_arrays");
//     vec_vals_res.array() = vec_vals_1.array() * vec_vals_2.array()  * vec_vals_3.array() ;
//     clock.tock("mult_3_arrays");
// 
//     clock.tick("mult_5_arrays");
//     vec_vals_res.array() = vec_vals_1.array() * vec_vals_2.array()  * vec_vals_3.array() * vec_vals_2.array() * vec_vals_1.array() ;
//     clock.tock("mult_5_arrays");
// 
//     clock.tick("div_2_arrays");
//     vec_vals_res.array() = vec_vals_1.array() / vec_vals_2.array() ;
//     clock.tock("div_2_arrays");
// 
//     clock.tick("recip_array");
//     vec_vals_res.array() = 1.0 / vec_vals_2.array() ;
//     clock.tock("recip_array");
// 
// 
// 
//   }
// 
//   clock.stop("eepy_basic_fns_clock_double");
// 
// }
// 
// 
// 
// 
// //[[Rcpp::export]]
// void eepy_exp_fns_double( int reps,
//                           int dim,
//                           double lower,
//                           double upper,
//                           bool AVX2 = true,
//                           bool AVX512 = true) {
// 
//   Rcpp::Clock clock;
// 
//   Eigen::Matrix<double, -1, 1>  vals = Eigen::Matrix<double, -1, 1>::Zero(dim);
//   Eigen::Matrix<double, -1, 1>  log_vals = Eigen::Matrix<double, -1, 1>::Zero(dim);
// 
//   for (int i = 0; i < vals.rows(); i++) {
//     vals(i) =  R::runif(lower, upper);
//   }
//   log_vals =  vals.log();
// 
// 
//   while (reps -- > 0) {
// 
// 
//     for (int i = 0; i < vals.rows(); i++) {
//       vals(i) =  R::runif(lower, upper);
//     }
//     log_vals =  vals.log();
// 
//     Eigen::Matrix<double, -1, 1>  vec_vals = Eigen::Matrix<double, -1, 1>::Zero(dim);
// 
//     ///////////////////////////////////////////////////  exp  - using doubles
//     //////  standard library fns  (always with checks)
//     clock.tick("exp_Stan");
//     vec_vals = fn_colvec_double(log_vals, "exp", "Stan");
//     clock.tock("exp_Stan");
//     if (reps == 1)     Rcout<<"\n max_error for exp_Stan = "  << 100 *  ( (vec_vals.array() - log_vals.array().exp()).array().abs()  /  vec_vals.array() ).maxCoeff();
//     if (reps == 1)     Rcout<<"\n max_abs_error for exp_Stan = "  <<    ( (vec_vals.array() - log_vals.array().exp()).abs().array()  ).maxCoeff();
// 
// 
//     clock.tick("exp_Eigen");
//     // Rcpp::Rcout << "Before Exp (final / output call): " << vec_vals.head(1) << std::endl;
//     vec_vals = fn_colvec_double(log_vals, "exp", "Eigen");
//     //   Rcpp::Rcout << "After Exp (final / output call): " << vec_vals.head(1) << std::endl;
//     clock.tock("exp_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for exp_Eigen = "  <<  100 * ( (vec_vals.array() - log_vals.array().exp()).array().abs()  /  vec_vals.array() ).maxCoeff();
//     if (reps == 1)     Rcout<<"\n max_abs_error for exp_Eigen = "  <<    ( (vec_vals.array() - log_vals.array().exp()).abs().array()  ).maxCoeff();
// 
// 
//     ///////////////////////////////////////////////////  exp
//     //////  fast "exact" fns - with checks
//     clock.tick("fast_exp_1_Loop_Eigen");
//     vec_vals = fn_colvec_double(log_vals, "exp", "Loop", false);
//     clock.tock("fast_exp_1_Loop_Eigen");
//     if (reps == 1)     Rcout<<"\n  max_error for fast_exp_1_Loop_Eigen = "  << 100 * ( (vec_vals.array() - log_vals.array().exp()).array().abs()  /  vec_vals.array() ).maxCoeff();
//     if (reps == 1)     Rcout<<"\n max_abs_error for fast_exp_1_Loop_Eigen = "  <<    ( (vec_vals.array() - log_vals.array().exp()).abs().array()  ).maxCoeff();
// 
// 
//     //////  fast "exact" fns - without checks
//     clock.tick("fast_exp_1_wo_checks_Loop_Eigen");
//     vec_vals = fn_colvec_double(log_vals, "exp", "Loop", true);
//     clock.tock("fast_exp_1_wo_checks_Loop_Eigen");
//     if (reps == 1)     Rcout<<"\n  max_error for fast_exp_1_wo_checks_Loop_Eigen = "  << 100 * ( (vec_vals.array() - log_vals.array().exp()).array().abs()  /  vec_vals.array() ).maxCoeff();
//     if (reps == 1)     Rcout<<"\n max_abs_error for fast_exp_1_wo_checks_Loop_Eigen = "  <<    ( (vec_vals.array() - log_vals.array().exp()).abs().array()  ).maxCoeff();
// 
// 
//     if (AVX2 == true) {
//       ///////////////////////////////////////////////////  exp  - using AVX512
//       //////  fast "exact" fns - with checks
//       clock.tick("fast_exp_1_AVX2_Eigen");
//       vec_vals = fn_colvec_double(log_vals, "exp", "AVX2", false);
//       clock.tock("fast_exp_1_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n  max_error for fast_exp_1_AVX2_Eigen = "  << 100 * ( (vec_vals.array() - log_vals.array().exp()).array().abs()  /  vec_vals.array() ).maxCoeff();
//       if (reps == 1)     Rcout<<"\n max_abs_error for fast_exp_1_AVX2_Eigen = "  <<    ( (vec_vals.array() - log_vals.array().exp()).abs().array()  ).maxCoeff();
// 
// 
//       //////  fast "exact" fns - without checks
//       clock.tick("fast_exp_1_wo_checks_AVX2_Eigen");
//       vec_vals = fn_colvec_double(log_vals, "exp", "AVX2", true);
//       clock.tock("fast_exp_1_wo_checks_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n  max_error for fast_exp_1_wo_checks_AVX2_Eigen = "  << 100 * ( (vec_vals.array() - log_vals.array().exp()).array().abs()  /  vec_vals.array() ).maxCoeff();
//       if (reps == 1)     Rcout<<"\n max_abs_error for fast_exp_1_wo_checks_AVX2_Eigen = "  <<    ( (vec_vals.array() - log_vals.array().exp()).abs().array()  ).maxCoeff();
// 
//     }
// 
//     if (AVX512 == true) {
//       ///////////////////////////////////////////////////  exp  - using AVX512
//       //////  fast "exact" fns - with checks
//       clock.tick("fast_exp_1_AVX512_Eigen");
//       vec_vals = fn_colvec_double(log_vals, "exp", "AVX512", false);
//       clock.tock("fast_exp_1_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n  max_error for fast_exp_1_AVX512_Eigen = "  << 100 * ( (vec_vals.array() - log_vals.array().exp()).array().abs()  /  vec_vals.array() ).maxCoeff();
//       if (reps == 1)     Rcout<<"\n max_abs_error for fast_exp_1_AVX512_Eigen = "  <<    ( (vec_vals.array() - log_vals.array().exp()).abs().array()  ).maxCoeff();
// 
// 
//       //////  fast "exact" fns - without checks
//       clock.tick("fast_exp_1_wo_checks_AVX512_Eigen");
//       vec_vals = fn_colvec_double(log_vals, "exp", "AVX512", true);
//       clock.tock("fast_exp_1_wo_checks_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n  max_error for fast_exp_1_wo_checks_AVX512_Eigen = "  << 100 * ( (vec_vals.array() - log_vals.array().exp()).array().abs()  /  vec_vals.array() ).maxCoeff();
//       if (reps == 1)     Rcout<<"\n max_abs_error for fast_exp_1_wo_checks_AVX512_Eigen = "  <<    ( (vec_vals.array() - log_vals.array().exp()).abs().array()  ).maxCoeff();
// 
// 
//     }
// 
//   }
// 
//   clock.stop("eepy_exp_fns_clock_double");
// 
// }
// 
// 
// 
// 
// 
// 
// 
// //[[Rcpp::export]]
// void eepy_log_fns_double( int reps,
//                           int dim,
//                           double  lower,
//                           double upper,
//                           bool AVX2 = true,
//                           bool AVX512 = true) {
// 
//   Rcpp::Clock clock;
// 
//   Eigen::Matrix<double, -1, 1>  vals(dim);
//   Eigen::Matrix<double, -1, 1>  vec_vals(dim);
//   Eigen::Matrix<double, -1, 1>  log_vals(dim);
// 
//   for (int i = 0; i < dim; i++) {
//     vals(i) =  R::runif(lower, upper);
//   }
//   log_vals =  vals.log();
// 
//   const int  N = dim;
// 
//   ///////////// log
//   while (reps-- > 0) {
// 
//     for (int i = 0; i < vals.rows(); i++) {
//       vals(i) =  R::runif(lower, upper);
//     }
//     log_vals =  vals.log();
// 
//     Eigen::Matrix<double, -1, 1>  vec_vals = Eigen::Matrix<double, -1, 1>::Zero(dim);
// 
// 
//     ///////////////////////////////////////////////////  log  - using doubles
//     clock.tick("log_Stan");
//     vec_vals = fn_colvec_double(vals, "log", "Stan");
//     clock.tock("log_Stan");
//     if (reps == 1)     Rcout<<"\n max_error for log_Stan = "  <<  100 * ( (vec_vals.array() - log_vals.array()).array().abs()  /  vec_vals.array() ).array().maxCoeff();
//     if (reps == 1)     Rcout<<"\n max_abs_error for log_Stan = "  <<    ( (vec_vals.array() - log_vals.array()).array().abs()  ).array().maxCoeff();
// 
//     clock.tick("log_Eigen");
//     vec_vals = fn_colvec_double(vals, "log", "Eigen");
//     clock.tock("log_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for log_Eigen = "  <<   100 * ( (vec_vals.array() - log_vals.array()).array().abs()  /  vec_vals.array() ).array().maxCoeff();
//     if (reps == 1)     Rcout<<"\n max_abs_error for log_Eigen = "  <<     ( (vec_vals.array() - log_vals.array()).array().abs()  ).array().maxCoeff();
// 
//     ///////////////////////////////////////////////////  log  - using AVX512
//     //////  fast "exact" fns - with checks
//     clock.tick("fast_log_1_Loop_Eigen");
//     vec_vals = fn_colvec_double(vals, "log", "Loop", false);
//     clock.tock("fast_log_1_Loop_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for fast_log_1_Loop_Eigen = "  <<  100 * ( (vec_vals.array() - log_vals.array()).array().abs()  /  vec_vals.array() ).array().maxCoeff();
//     if (reps == 1)     Rcout<<"\n max_abs_error for fast_log_1_Loop_Eigen = "  <<     ( (vec_vals.array() - log_vals.array()).array().abs()  ).array().maxCoeff();
// 
//     //////  fast "exact" fns - with checks
//     clock.tick("fast_log_1_wo_checks_Loop_Eigen");
//     vec_vals = fn_colvec_double(vals, "log", "Loop", true);
//     clock.tock("fast_log_1_wo_checks_Loop_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for fast_log_1_wo_checks_Loop_Eigen = "  <<  100 * ( (vec_vals.array() - log_vals.array()).array().abs()  /  vec_vals.array() ).array().maxCoeff();
//     if (reps == 1)     Rcout<<"\n max_abs_error for fast_log_1_wo_checks_Loop_Eigen = "  <<    ( (vec_vals.array() - log_vals.array()).array().abs()  ).array().maxCoeff();
// 
// 
//     if (AVX2 == true) {
//       ///////////////////////////////////////////////////  log  - using AVX512
//       //////  fast "exact" fns - with checks
//       clock.tick("fast_log_1_AVX2_Eigen");
//       vec_vals = fn_colvec_double(vals, "log", "AVX2", false);
//       clock.tock("fast_log_1_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_log_1_AVX2_Eigen = "  <<  100 * ( (vec_vals.array() - log_vals.array()).array().abs()  /  vec_vals.array() ).array().maxCoeff();
//       if (reps == 1)     Rcout<<"\n max_abs_error for fast_log_1_AVX2_Eigen = "  <<    ( (vec_vals.array() - log_vals.array()).array().abs()  ).array().maxCoeff();
// 
//       //////  fast "exact" fns - with checks
//       clock.tick("fast_log_1_wo_checks_AVX2_Eigen");
//       vec_vals = fn_colvec_double(vals, "log", "AVX2", true);
//       clock.tock("fast_log_1_wo_checks_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_log_1_wo_checks_AVX2_Eigen = "  <<  100 * ( (vec_vals.array() - log_vals.array()).array().abs()  /  vec_vals.array() ).array().maxCoeff();
//       if (reps == 1)     Rcout<<"\n max_abs_error for fast_log_1_wo_checks_AVX2_Eigen = "  <<    ( (vec_vals.array() - log_vals.array()).array().abs()  ).array().maxCoeff();
// 
//     }
// 
//     if (AVX512 == true) {
//       ///////////////////////////////////////////////////  log  - using AVX512
//       //////  fast "exact" fns - with checks
//       clock.tick("fast_log_1_AVX512_Eigen");
//       vec_vals = fn_colvec_double(vals, "log", "AVX512", false);
//       clock.tock("fast_log_1_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_log_1_AVX512_Eigen = "  <<  100 * ( (vec_vals.array() - log_vals.array()).array().abs()  /  vec_vals.array() ).array().maxCoeff();
//       if (reps == 1)     Rcout<<"\n max_abs_error for fast_log_1_AVX512_Eigen = "  <<    ( (vec_vals.array() - log_vals.array()).array().abs()  ).array().maxCoeff();
// 
//       //////  fast "exact" fns - with checks
//       clock.tick("fast_log_1_wo_checks_AVX512_Eigen");
//       vec_vals = fn_colvec_double(vals, "log", "AVX512", true);
//       clock.tock("fast_log_1_wo_checks_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_log_1_wo_checks_AVX512_Eigen = "  <<  100 * ( (vec_vals.array() - log_vals.array()).array().abs()  /  vec_vals.array() ).array().maxCoeff();
//       if (reps == 1)     Rcout<<"\n max_abs_error for fast_log_1_wo_checks_AVX512_Eigen = "  <<    ( (vec_vals.array() - log_vals.array()).array().abs()  ).array().maxCoeff();
// 
//     }
// 
//   }
// 
//   clock.stop("eepy_log_fns_clock_double");
// 
// }
// 
// 
// 
// 
// 
// 
// 
// //[[Rcpp::export]]
// void eepy_Phi_fns_double( int reps = 1000,
//                           int dim  = 1000,
//                           bool AVX2 = true,
//                           bool AVX512 = true) {
// 
//   Rcpp::Clock clock;
// 
//   Eigen::Matrix<double, -1, 1>  probs(dim);
//   Eigen::Matrix<double, -1, 1>  log_probs(dim);
//   Eigen::Matrix<double, -1, 1>  vec_vals(dim);
//   Eigen::Matrix<double, -1, 1>  inv_Phi_vals(dim);
// 
//   for (int i = 0; i < dim; i++) {
//     probs(i) =  R::runif(0, 1);
//   }
// 
//   log_probs =  probs.log();
//   inv_Phi_vals = stan::math::inv_Phi(probs.matrix()).array();  // vals generated using inverse-CDF method
// 
//   const int  N = dim;
// 
// 
// 
// 
//   ///////////// log
//   while (reps-- > 0) {
// 
//     for (int i = 0; i < dim; i++) {
//       probs(i) =  R::runif(0, 1);
//     }
//     log_probs =  probs.log();
//     inv_Phi_vals = stan::math::inv_Phi(probs.matrix()).array();
// 
//     Eigen::Matrix<double, -1, 1>  vec_vals = Eigen::Matrix<double, -1, 1>::Zero(dim);
// 
// 
//     clock.tick("Phi_Stan");
//     vec_vals = fn_colvec_double(inv_Phi_vals, "Phi", "Stan");
//     clock.tock("Phi_Stan");
//     if (reps == 1)     Rcout<<"\n max_error for Phi_Stan = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
//     clock.tick("Phi_Eigen");
//     vec_vals = fn_colvec_double(inv_Phi_vals, "Phi", "Eigen", true);
//     clock.tock("Phi_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for Phi_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
//     if (AVX2 == true) {
//       clock.tick("fast_Phi_wo_checks_AVX2_Eigen");
//       vec_vals = fn_colvec_double(inv_Phi_vals, "Phi", "AVX2", true);
//       clock.tock("fast_Phi_wo_checks_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_Phi_wo_checks_AVX2_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
//     }
// 
//     if (AVX512 == true) {
//       clock.tick("fast_Phi_wo_checks_AVX512_Eigen");
//       vec_vals = fn_colvec_double(inv_Phi_vals, "Phi", "AVX512", true);
//       clock.tock("fast_Phi_wo_checks_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_Phi_wo_checks_AVX512_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
//     }
// 
// 
// 
// 
//   }
// 
//   clock.stop("eepy_Phi_fns_clock_double");
// 
// }
// 
// 
// 
// 
// 
// //[[Rcpp::export]]
// void eepy_Phi_approx_fns_double( int reps = 1000,
//                                  int dim  = 1000,
//                                  bool AVX2 = true,
//                                  bool AVX512 = true) {
// 
//   Rcpp::Clock clock;
// 
//   Eigen::Matrix<double, -1, 1>  probs(dim);
//   Eigen::Matrix<double, -1, 1>  log_probs(dim);
//   Eigen::Matrix<double, -1, 1>  vec_vals(dim);
//   Eigen::Matrix<double, -1, 1>  inv_Phi_vals(dim);
// 
//   for (int i = 0; i < dim; i++) {
//     probs(i) =  R::runif(0, 1);
//   }
// 
//   log_probs =  probs.log();
//   inv_Phi_vals = stan::math::inv_Phi(probs.matrix()).array();  // vals generated using inverse-CDF method
// 
//   const int  N = dim;
// 
// 
// 
// 
//   ///////////// log
//   while (reps-- > 0) {
// 
//     for (int i = 0; i < dim; i++) {
//       probs(i) =  R::runif(0, 1);
//     }
//     log_probs =  probs.log();
//     inv_Phi_vals = stan::math::inv_Phi(probs.matrix()).array();
// 
//     Eigen::Matrix<double, -1, 1>  vec_vals = Eigen::Matrix<double, -1, 1>::Zero(dim);
// 
// 
//     clock.tick("Phi_approx_Eigen");
//     vec_vals = fn_colvec_double(inv_Phi_vals, "Phi_approx", "Eigen");
//     clock.tock("Phi_approx_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for Phi_approx_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
//     clock.tick("Phi_approx_Stan");
//     vec_vals = fn_colvec_double(inv_Phi_vals, "Phi_approx", "Stan");
//     clock.tock("Phi_approx_Stan");
//     if (reps == 1)     Rcout<<"\n max_error for Phi_approx_Stan = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
// 
//     clock.tick("fast_Phi_approx_Loop_Eigen");
//     vec_vals = fn_colvec_double(inv_Phi_vals, "Phi_approx", "Loop", false);
//     clock.tock("fast_Phi_approx_Loop_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for fast_Phi_approx_Loop_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
//     clock.tick("fast_Phi_approx_wo_checks_Loop_Eigen");
//     vec_vals = fn_colvec_double(inv_Phi_vals, "Phi_approx", "Loop", true);
//     clock.tock("fast_Phi_approx_wo_checks_Loop_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for fast_Phi_approx_wo_checks_Loop_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
//     clock.tick("fast_Phi_approx_2_Loop_Eigen");
//     vec_vals = fn_colvec_double(inv_Phi_vals, "Phi_approx_2", "Loop", false);
//     clock.tock("fast_Phi_approx_2_Loop_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for fast_Phi_approx_2_Loop_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
//     clock.tick("fast_Phi_approx_wo_checks_2_Loop_Eigen");
//     vec_vals = fn_colvec_double(inv_Phi_vals, "Phi_approx_2", "Loop", true);
//     clock.tock("fast_Phi_approx_wo_checks_2_Loop_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for fast_Phi_approx_wo_checks_2_Loop_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
// 
//     if (AVX2 == true) {
// 
//       clock.tick("fast_Phi_approx_AVX2_Eigen");
//       vec_vals = fn_colvec_double(inv_Phi_vals, "Phi_approx", "AVX2", false);
//       clock.tock("fast_Phi_approx_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_Phi_approx_AVX2_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
//       clock.tick("fast_Phi_approx_wo_checks_AVX2_Eigen");
//       vec_vals = fn_colvec_double(inv_Phi_vals, "Phi_approx", "AVX2", true);
//       clock.tock("fast_Phi_approx_wo_checks_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_Phi_approx_wo_checks_AVX2_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
//       clock.tick("fast_Phi_approx_2_AVX2_Eigen");
//       vec_vals = fn_colvec_double(inv_Phi_vals, "Phi_approx_2", "AVX2", false);
//       clock.tock("fast_Phi_approx_2_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_Phi_approx_2_AVX2_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
//       clock.tick("fast_Phi_approx_wo_checks_2_AVX2_Eigen");
//       vec_vals = fn_colvec_double(inv_Phi_vals, "Phi_approx_2", "AVX2", true);
//       clock.tock("fast_Phi_approx_wo_checks_2_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_Phi_approx_wo_checks_2_AVX2_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
//     }
// 
//     if (AVX512 == true) {
// 
//       clock.tick("fast_Phi_approx_AVX512_Eigen");
//       vec_vals = fn_colvec_double(inv_Phi_vals, "Phi_approx", "AVX512", false);
//       clock.tock("fast_Phi_approx_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_Phi_approx_AVX512_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
//       clock.tick("fast_Phi_approx_wo_checks_AVX512_Eigen");
//       vec_vals = fn_colvec_double(inv_Phi_vals, "Phi_approx", "AVX512", true);
//       clock.tock("fast_Phi_approx_wo_checks_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_Phi_approx_wo_checks_AVX512_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
//       clock.tick("fast_Phi_approx_2_AVX512_Eigen");
//       vec_vals = fn_colvec_double(inv_Phi_vals, "Phi_approx_2", "AVX512", false);
//       clock.tock("fast_Phi_approx_2_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_Phi_approx_2_AVX512_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
//       clock.tick("fast_Phi_approx_wo_checks_2_AVX512_Eigen");
//       vec_vals = fn_colvec_double(inv_Phi_vals, "Phi_approx_2", "AVX512", true);
//       clock.tock("fast_Phi_approx_wo_checks_2_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_Phi_approx_wo_checks_2_AVX512_Eigen = "  << (vec_vals - (probs)).array().abs().maxCoeff();
// 
//     }
// 
// 
// 
// 
//   }
// 
//   clock.stop("eepy_Phi_approx_fns_clock_double");
// 
// }
// 
// 
// 
// 
// 
// 
// 
// //[[Rcpp::export]]
// void eepy_inv_Phi_fns_double( int reps = 1000,
//                               int dim  = 1000,
//                               bool AVX2 = true,
//                               bool AVX512 = true) {
// 
//   Rcpp::Clock clock;
// 
//   Eigen::Matrix<double, -1, 1>  vec_vals(dim);
//   Eigen::Matrix<double, -1, 1>  norm_samples(dim);
//   Eigen::Matrix<double, -1, 1>  probs_from_Phi_approx_exact(dim);
//   Eigen::Matrix<double, -1, 1>  probs_from_Phi_exact(dim);
// 
//   for (int i = 0; i < dim; i++) {
//     norm_samples(i) =  R::rnorm(0, 1);
//     probs_from_Phi_approx_exact(i) = stan::math::Phi_approx(norm_samples(i)) ;
//     probs_from_Phi_exact(i) = stan::math::Phi(norm_samples(i)) ;
//   }
// 
// 
//   ///////////// log
//   while (reps-- > 0) {
// 
//     // for (int i = 0; i < dim; i++) {
//     //   probs(i) =  R::runif(0, 1);
//     // }
//     // log_probs =  probs.log();
//     // inv_Phi_vals = stan::math::inv_Phi(probs.matrix()).array();
//     //
//     // Eigen::Matrix<double, -1, 1>  vec_vals = Eigen::Matrix<double, -1, 1>::Zero(dim);
// 
//     for (int i = 0; i < dim; i++) {
//       norm_samples(i) =  R::rnorm(0, 1);
//       probs_from_Phi_approx_exact(i) = stan::math::Phi_approx(norm_samples(i)) ;
//       probs_from_Phi_exact(i) = stan::math::Phi(norm_samples(i)) ;
//     }
// 
// 
//     clock.tick("inv_Phi_Stan");
//     vec_vals = fn_colvec_double(probs_from_Phi_exact, "inv_Phi", "Stan");
//     clock.tock("inv_Phi_Stan");
//     if (reps == 1)     Rcout<<"\n max_error for inv_Phi_Stan = "  << (vec_vals - (norm_samples)).array().abs().maxCoeff();
// 
// 
//     clock.tick("inv_Phi_Eigen");
//     vec_vals = fn_colvec_double(probs_from_Phi_exact, "inv_Phi", "Eigen");
//     clock.tock("inv_Phi_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for inv_Phi_Eigen = "  << (vec_vals - (norm_samples)).array().abs().maxCoeff();
// 
//     if (AVX2 == true) {
// 
//       clock.tick("fast_inv_Phi_wo_checks_AVX2_Eigen");
//       vec_vals = fn_colvec_double(probs_from_Phi_exact, "inv_Phi", "AVX2", true);
//       clock.tock("fast_inv_Phi_wo_checks_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_inv_Phi_wo_checks_AVX2_Eigen = "  << (vec_vals - (norm_samples)).array().abs().maxCoeff();
// 
//     }
// 
// 
//     if (AVX512 == true) {
// 
//       clock.tick("fast_inv_Phi_wo_checks_AVX512_Eigen");
//       vec_vals = fn_colvec_double(probs_from_Phi_exact, "inv_Phi", "AVX512", true);
//       clock.tock("fast_inv_Phi_wo_checks_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_inv_Phi_wo_checks_AVX512_Eigen = "  << (vec_vals - (norm_samples)).array().abs().maxCoeff();
// 
//     }
// 
// 
// 
//   }
// 
//   clock.stop("eepy_inv_Phi_fns_clock_double");
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
// //[[Rcpp::export]]
// void eepy_inv_Phi_approx_fns_double( int reps = 1000,
//                                      int dim  = 1000,
//                                      bool AVX2 = true,
//                                      bool AVX512 = true) {
// 
//   Rcpp::Clock clock;
// 
//   Eigen::Matrix<double, -1, 1>  vec_vals(dim);
//   Eigen::Matrix<double, -1, 1>  norm_samples(dim);
//   Eigen::Matrix<double, -1, 1>  probs_from_Phi_approx_exact(dim);
//   Eigen::Matrix<double, -1, 1>  probs_from_Phi_exact(dim);
// 
//   for (int i = 0; i < dim; i++) {
//     norm_samples(i) =  R::rnorm(0, 1);
//     probs_from_Phi_approx_exact(i) = stan::math::Phi_approx(norm_samples(i)) ;
//     probs_from_Phi_exact(i) = stan::math::Phi(norm_samples(i)) ;
//   }
// 
// 
//   ///////////// log
//   while (reps-- > 0) {
// 
//     // for (int i = 0; i < dim; i++) {
//     //   probs(i) =  R::runif(0, 1);
//     // }
//     // log_probs =  probs.log();
//     // inv_Phi_vals = stan::math::inv_Phi(probs.matrix()).array();
//     //
//     // Eigen::Matrix<double, -1, 1>  vec_vals = Eigen::Matrix<double, -1, 1>::Zero(dim);
// 
//     // for (int i = 0; i < dim; i++) {
//     //   norm_samples(i) =  R::rnorm(0, 1);
//     //   probs_from_Phi_approx_exact(i) = stan::math::Phi_approx(norm_samples(i)) ;
//     //   probs_from_Phi_exact(i) = stan::math::Phi(norm_samples(i)) ;
//     // }
// 
//     clock.tick("inv_Phi_approx_Stan");
//     vec_vals = fn_colvec_double(probs_from_Phi_exact, "inv_Phi_approx", "Stan");
//     clock.tock("inv_Phi_approx_Stan");
//     if (reps == 1)     Rcout<<"\n max_error for inv_Phi_approx_Stan = "  << (vec_vals - (norm_samples)).array().abs().maxCoeff();
// 
// 
//     clock.tick("inv_Phi_approx_Eigen");
//     vec_vals = fn_colvec_double(probs_from_Phi_exact, "inv_Phi_approx", "Eigen");
//     clock.tock("inv_Phi_approx_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for inv_Phi_approx_Eigen = "  << (vec_vals - (norm_samples)).array().abs().maxCoeff();
// 
// 
// 
//     // clock.tick("fast_inv_Phi_approx_Loop_Eigen");
//     // vec_vals = fn_colvec_double(probs_from_Phi_approx_exact, "inv_Phi_approx", "Loop", false);
//     // clock.tock("fast_inv_Phi_approx_Loop_Eigen");
//     // if (reps == 1)     Rcout<<"\n max_error for fast_inv_Phi_approx_Loop_Eigen = "  << (vec_vals - (norm_samples)).array().abs().maxCoeff();
// 
// 
// 
// 
//     if (AVX2 == true) {
// 
//       clock.tick("fast_inv_Phi_approx_AVX2_Eigen");
//       vec_vals = fn_colvec_double(probs_from_Phi_approx_exact, "inv_Phi_approx", "AVX2", false);
//       clock.tock("fast_inv_Phi_approx_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_inv_Phi_approx_AVX2_Eigen = "  << (vec_vals - (norm_samples)).array().abs().maxCoeff();
// 
//       clock.tick("fast_inv_Phi_approx_wo_checks_AVX2_Eigen");
//       vec_vals = fn_colvec_double(probs_from_Phi_approx_exact, "inv_Phi_approx", "AVX2", true);
//       clock.tock("fast_inv_Phi_approx_wo_checks_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_inv_Phi_approx_wo_checks_AVX2_Eigen = "  << (vec_vals - (norm_samples)).array().abs().maxCoeff();
// 
//     }
// 
// 
//     if (AVX512 == true) {
// 
//       clock.tick("fast_inv_Phi_approx_AVX512_Eigen");
//       vec_vals = fn_colvec_double(probs_from_Phi_approx_exact, "inv_Phi_approx", "AVX512", false);
//       clock.tock("fast_inv_Phi_approx_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_inv_Phi_approx_AVX512_Eigen = "  << (vec_vals - (norm_samples)).array().abs().maxCoeff();
// 
//       clock.tick("fast_inv_Phi_approx_wo_checks_AVX512_Eigen");
//       vec_vals = fn_colvec_double(probs_from_Phi_approx_exact, "inv_Phi_approx", "AVX512", true);
//       clock.tock("fast_inv_Phi_approx_wo_checks_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_inv_Phi_approx_wo_checks_AVX512_Eigen = "  << (vec_vals - (norm_samples)).array().abs().maxCoeff();
// 
//     }
// 
// 
// 
//   }
// 
//   clock.stop("eepy_inv_Phi_approx_fns_clock_double");
// 
// }
// 
// 
// 
// 
// 
// 
// 
// //[[Rcpp::export]]
// void eepy_log_Phi_approx_fns_double( int reps = 1000,
//                                      int dim  = 1000,
//                                      bool AVX2  = true,
//                                      bool AVX512 = true) {
// 
//   Rcpp::Clock clock;
// 
//   Eigen::Matrix<double, -1, 1>  p(dim);
//   Eigen::Matrix<double, -1, 1>  log_p(dim);
//   Eigen::Matrix<double, -1, 1>  vec_vals(dim);
//   Eigen::Matrix<double, -1, 1>  inv_Phi_vals(dim);
// 
//   for (int i = 0; i < dim; i++) {
//     p(i) =  R::runif(0, 1);
//   }
// 
//   log_p =  p.log();
//   inv_Phi_vals = stan::math::inv_Phi(p.matrix()).array();  // vals generated using inverse-CDF method
// 
//   const int  N = dim;
// 
// 
// 
// 
//   /////////////
//   while (reps-- > 0) {
// 
// 
//     for (int i = 0; i < dim; i++) {
//       p(i) =  R::runif(0, 1);
//     }
// 
//     log_p =  p.log();
//     inv_Phi_vals = stan::math::inv_Phi(p.matrix()).array();  // vals generated using inverse-CDF method
// 
// 
// 
//     clock.tick("log_Phi_approx_Eigen");
//     vec_vals = fn_colvec_double(inv_Phi_vals, "log_Phi_approx", "Eigen");
//     clock.tock("log_Phi_approx_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for log_Phi_approx_Eigen = "  << (vec_vals -  log_p).array().abs().maxCoeff();
// 
//     clock.tick("log_Phi_approx_Stan");
//     vec_vals = fn_colvec_double(inv_Phi_vals, "log_Phi_approx", "Stan");
//     clock.tock("log_Phi_approx_Stan");
//     if (reps == 1)     Rcout<<"\n max_error for log_Phi_approx_Stan = "  << (vec_vals -  log_p).array().abs().maxCoeff();
// 
// 
//     clock.tick("fast_log_Phi_approx_Loop_Eigen");
//     vec_vals = fn_colvec_double(inv_Phi_vals, "log_Phi_approx", "Loop", false);
//     clock.tock("fast_log_Phi_approx_Loop_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for fast_log_Phi_approx_Loop_Eigen = "  << (vec_vals -  log_p).array().abs().maxCoeff();
// 
// 
//     clock.tick("fast_log_Phi_approx_wo_checks_Loop_Eigen");
//     vec_vals = fn_colvec_double(inv_Phi_vals, "log_Phi_approx", "Loop", true);
//     clock.tock("fast_log_Phi_approx_wo_checks_Loop_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for fast_log_Phi_approx_wo_checks_Loop_Eigen = "  <<  (vec_vals -  log_p).array().abs().maxCoeff();
// 
//     if (AVX2 == true) {
// 
//       clock.tick("fast_log_Phi_approx_AVX2_Eigen");
//       vec_vals = fn_colvec_double(inv_Phi_vals, "log_Phi_approx", "AVX2", false);
//       clock.tock("fast_log_Phi_approx_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_log_Phi_approx_AVX2_Eigen = "  << (vec_vals -  log_p).array().abs().maxCoeff();
// 
// 
//       clock.tick("fast_log_Phi_approx_wo_checks_AVX2_Eigen");
//       vec_vals = fn_colvec_double(inv_Phi_vals, "log_Phi_approx", "AVX2", true);
//       clock.tock("fast_log_Phi_approx_wo_checks_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_log_Phi_approx_wo_checks_AVX2_Eigen = "  <<  (vec_vals -  log_p).array().abs().maxCoeff();
// 
//     }
// 
// 
//     if (AVX512 == true) {
// 
//       clock.tick("fast_log_Phi_approx_AVX512_Eigen");
//       vec_vals = fn_colvec_double(inv_Phi_vals, "log_Phi_approx", "AVX512", false);
//       clock.tock("fast_log_Phi_approx_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_log_Phi_approx_AVX512_Eigen = "  << (vec_vals -  log_p).array().abs().maxCoeff();
// 
// 
//       clock.tick("fast_log_Phi_approx_wo_checks_AVX512_Eigen");
//       vec_vals = fn_colvec_double(inv_Phi_vals, "log_Phi_approx", "AVX512", true);
//       clock.tock("fast_log_Phi_approx_wo_checks_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_log_Phi_approx_wo_checks_AVX512_Eigen = "  <<  (vec_vals -  log_p).array().abs().maxCoeff();
// 
//     }
// 
//   }
// 
//   clock.stop("eepy_log_Phi_approx_fns_clock_double");
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
// //[[Rcpp::export]]
// void eepy_log_sum_exp_fns_double( int reps = 1000,
//                                   int dim  = 1000,
//                                   bool AVX2 =  true,
//                                   bool AVX512 = true) {
// 
//   Rcpp::Clock clock;
// 
//   Eigen::Matrix<double, -1, 1>  probs_1(dim);
//   Eigen::Matrix<double, -1, 1>  probs_2(dim);
//   Eigen::Matrix<double, -1, 1>  log_probs_1(dim);
//   Eigen::Matrix<double, -1, 1>  log_probs_2(dim);
//   Eigen::Matrix<double, -1, 1>  vec_vals(dim);
//   Eigen::Matrix<double, -1, 2>  log_probs_2d_array(dim, 2);
// 
//   for (int i = 0; i < dim; i++) {
//     probs_1(i) =  R::runif(0.001, 0.999);
//     probs_2(i) =  R::runif(0.001, 0.999);
//   }
// 
//   log_probs_1 =  probs_1.log();
//   log_probs_2 =  probs_2.log();
// 
//   log_probs_2d_array.col(0) = log_probs_1;
//   log_probs_2d_array.col(1) = log_probs_2;
// 
//   ///////////// log
//   while (reps-- > 0) {
// 
//     // for (int i = 0; i < dim; i++) {
//     //   probs_1(i) =  R::runif(0.001, 0.999);
//     //   probs_2(i) =  R::runif(0.001, 0.999);
//     // }
//     //
//     // log_probs_1 =  probs_1.log();
//     // log_probs_2 =  probs_2.log();
//     //
//     // log_probs_2d_array.col(0) = log_probs_1;
//     // log_probs_2d_array.col(1) = log_probs_2;
// 
// 
//     clock.tick("log_sum_exp_2d_Eigen");
//     vec_vals = log_sum_exp_2d_Eigen_double(log_probs_2d_array);
//     clock.tock("log_sum_exp_2d_Eigen");
// 
// 
//     clock.tick("log_sum_exp_2d_Stan_double");
//     vec_vals = log_sum_exp_2d_Stan_double(log_probs_2d_array);
//     clock.tock("log_sum_exp_2d_Stan_double");
// 
// 
//     // clock.tick("fast_log_sum_exp_2d_double");
//     // vec_vals = fast_log_sum_exp_2d_double(log_probs_2d_array);
//     // clock.tock("fast_log_sum_exp_2d_double");
// 
//     //
//     //
//     // if (AVX2 == true) {
//     //
//     //   clock.tick("fast_log_sum_exp_2d_AVX2");
//     //   vec_vals = fast_log_sum_exp_2d_AVX2_double(log_probs_2d_array);
//     //   clock.tock("fast_log_sum_exp_2d_AVX2");
//     //
//     //   // clock.tick("fast_log_sum_exp_2d_wo_checks_AVX2");
//     //   // vec_vals = fast_log_sum_exp_2d_wo_checks_AVX2_double(log_probs_2d_array);
//     //   // clock.tock("fast_log_sum_exp_2d_wo_checks_AVX2");
//     //
//     //
//     // }
//     //
//     //
//     // if (AVX512 == true) {
//     //
//     //   clock.tick("fast_log_sum_exp_2d_AVX512");
//     //   vec_vals = fast_log_sum_exp_2d_AVX512_double(log_probs_2d_array);
//     //   clock.tock("fast_log_sum_exp_2d_AVX512");
//     //
//     //   //         clock.tick("fast_log_sum_exp_2d_wo_checks_AVX512");
//     //   //         vec_vals = fast_log_sum_exp_2d_wo_checks_AVX512_double(log_probs_2d_array);
//     //   //         clock.tock("fast_log_sum_exp_2d_wo_checks_AVX512");
//     //
//     //
//     // }
// 
// 
// 
// 
// 
//   }
// 
//   clock.stop("eepy_log_sum_exp_fns_clock_double");
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
// //[[Rcpp::export]]
// void eepy_log1m_fns_double( int reps = 1000,
//                             int dim  = 1000,
//                             bool AVX2 =  true,
//                             bool AVX512 =  true ) {
// 
//   Rcpp::Clock clock;
// 
//   Eigen::Matrix<double, -1, 1>  probs(dim);
//   Eigen::Matrix<double, -1, 1>  log1m_probs(dim);
//   Eigen::Matrix<double, -1, 1>  vec_vals(dim);
// 
//   for (int i = 0; i < dim; i++) {
//     probs(i) = R::runif(0.001, 0.999);
//     vec_vals(i) = 0.0;
//   }
// 
//   log1m_probs.array() =  (-probs.array()).array().log1p().array();
// 
//   const int  N = dim;
// 
//   ///////////// log
//   while (reps-- > 0) {
// 
//     for (int i = 0; i < dim; i++) {
//       probs(i) = R::runif(0.001, 0.999);
//       vec_vals(i) = 0.0;
//     }
// 
//     log1m_probs.array() =  (-probs.array()).array().log1p().array();
// 
// 
// 
//     ////////////////////////////////////////////////////////////////////////   log  - using doubles
//     clock.tick("log1m_Stan");
//     vec_vals = fn_colvec_double(probs, "log1m", "Stan");
//     clock.tock("log1m_Stan");
//     if (reps == 1)     Rcout<<"\n max_error for log1m_Stan = "  << (vec_vals - log1m_probs).array().abs().maxCoeff();
// 
//     clock.tick("log1m_Eigen");
//     vec_vals = fn_colvec_double(probs, "log1m", "Eigen");
//     clock.tock("log1m_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for log1m_Eigen = "  << (vec_vals - log1m_probs).array().abs().maxCoeff();
// 
// 
//     ///////////////////////////////////////////////////  log1m  - using AVX512
//     //////  fast "exact" fns - with checks
//     clock.tick("fast_log1m_1_Loop_Eigen");
//     vec_vals = fn_colvec_double(probs, "log1m", "Loop", false);
//     clock.tock("fast_log1m_1_Loop_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for fast_log1m_1_Loop_Eigen = "  << (vec_vals - log1m_probs).array().abs().maxCoeff();
// 
//     //////  fast "exact" fns - w/o checks
//     clock.tick("fast_log1m_1_wo_checks_Loop_Eigen");
//     vec_vals = fn_colvec_double(probs, "log1m", "Loop", true);
//     clock.tock("fast_log1m_1_wo_checks_Loop_Eigen");
//     if (reps == 1)     Rcout<<"\n max_error for fast_log1m_1_wo_checks_Loop_Eigen = "  << (vec_vals - log1m_probs).array().abs().maxCoeff();
// 
// 
//     if (AVX2 == true) {
//       ///////////////////////////////////////////////////  log1m  - using AVX512
//       //////  fast "exact" fns - with checks
//       clock.tick("fast_log1m_1_AVX2_Eigen");
//       vec_vals = fn_colvec_double(probs, "log1m", "AVX2", false);
//       clock.tock("fast_log1m_1_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_log1m_1_AVX2_Eigen = "  << (vec_vals - log1m_probs).array().abs().maxCoeff();
// 
//       //////  fast "exact" fns - w/o checks
//       clock.tick("fast_log1m_1_wo_checks_AVX2_Eigen");
//       vec_vals = fn_colvec_double(probs, "log1m", "AVX2", true);
//       clock.tock("fast_log1m_1_wo_checks_AVX2_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_log1m_1_wo_checks_AVX2_Eigen = "  << (vec_vals - log1m_probs).array().abs().maxCoeff();
//     }
// 
//     if (AVX512 == true) {
//       ///////////////////////////////////////////////////  log1m  - using AVX512
//       //////  fast "exact" fns - with checks
//       clock.tick("fast_log1m_1_AVX512_Eigen");
//       vec_vals = fn_colvec_double(probs, "log1m", "AVX512", false);
//       clock.tock("fast_log1m_1_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_log1m_1_AVX512_Eigen = "  << (vec_vals - log1m_probs).array().abs().maxCoeff();
// 
//       //////  fast "exact" fns - w/o checks
//       clock.tick("fast_log1m_1_wo_checks_AVX512_Eigen");
//       vec_vals = fn_colvec_double(probs, "log1m", "AVX512", true);
//       clock.tock("fast_log1m_1_wo_checks_AVX512_Eigen");
//       if (reps == 1)     Rcout<<"\n max_error for fast_log1m_1_wo_checks_AVX512_Eigen = "  << (vec_vals - log1m_probs).array().abs().maxCoeff();
//     }
// 
// 
//   }
// 
//   clock.stop("eepy_log1m_fns_clock_double");
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
// 
// 
// 
// //
