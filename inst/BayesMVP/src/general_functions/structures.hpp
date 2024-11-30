
#pragma once

 
  
 
 
 
 
#include <stan/model/model_base.hpp>  
 
 
 
#include <stan/io/array_var_context.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/dump.hpp> 
 
 
#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/rapidjson_parser.hpp>  
 
 
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
 
 
  
#include <Eigen/Dense>
  
  
  
  
#if __has_include("bridgestan.h")
    #define HAS_BRIDGESTAN_H 1
    #include "bridgestan.h" 
    #include "version.hpp"
    #include "model_rng.hpp" 
#else 
    #define HAS_BRIDGESTAN_H 0
#endif
  
  
  
  
  
 
 
 
using namespace Eigen;
  

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
 

using std_vec_of_EigenVecs_dbl = std::vector<Eigen::Matrix<double, -1, 1>>;
using std_vec_of_EigenVecs_int = std::vector<Eigen::Matrix<int, -1, 1>>;

using std_vec_of_EigenMats_dbl = std::vector<Eigen::Matrix<double, -1, -1>>;
using std_vec_of_EigenMats_int = std::vector<Eigen::Matrix<int, -1, -1>>;

using two_layer_std_vec_of_EigenVecs_dbl =  std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>;
using two_layer_std_vec_of_EigenVecs_int = std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>;

using two_layer_std_vec_of_EigenMats_dbl = std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>;
using two_layer_std_vec_of_EigenMats_int = std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>;


using three_layer_std_vec_of_EigenVecs_dbl =  std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>>;
using three_layer_std_vec_of_EigenVecs_int =  std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>>;

using three_layer_std_vec_of_EigenMats_dbl = std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>>; 
using three_layer_std_vec_of_EigenMats_int = std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>>;

 
 

 
 
#if HAS_BRIDGESTAN_H
 
 
// Struct to hold the model handle and function pointers
struct ModelHandle_struct {
    
   void* bs_handle = nullptr;
   bs_model* (*bs_model_construct)(const char*, unsigned int, char**);
   int (*bs_log_density_gradient)(bs_model*, bool, bool, const double*, double*, double*, char**);
   int (*bs_param_constrain)(bs_model*, bool, bool, const double*, double*, bs_rng*, char**) = nullptr;
   bs_rng* (*bs_rng_construct)(unsigned int, char**);
   
}; 

#else
 
/// otherwise make dummy struct
struct ModelHandle_struct {
 
 void* bs_handle = nullptr;
 
}; 
 
 
#endif
 
 
 
 
#if HAS_BRIDGESTAN_H
 
struct Stan_model_struct {
   
   void* bs_handle = nullptr;
   bs_model* bs_model_ptr = nullptr;
   bs_model* (*bs_model_construct)(const char*, unsigned int, char**) = nullptr;
   int (*bs_log_density_gradient)(bs_model*, bool, bool, const double*, double*, double*, char**) = nullptr;
   int (*bs_param_constrain)(bs_model*, bool, bool, const double*, double*, bs_rng*, char**) = nullptr;
   bs_rng* (*bs_rng_construct)(unsigned int, char**);
    
};

#else

/// otherwise make dummy struct
struct Stan_model_struct {
  
  void* bs_handle = nullptr;
  
}; 

 
#endif
 
 
 
 
 
 
 
 
 
 
// struct for other function arguments to make function signitures more general  easier to manage
// the struct name becomes a return type. So can use as a return argument to functions.
struct alignas(EIGEN_MAX_ALIGN_BYTES)   Model_fn_args_struct {
   
               int N;
               int n_nuisance;
               int n_params_main;
               
               std::string model_so_file;
               std::string json_file_path;
               
               Eigen::Matrix<bool, -1, 1>        Model_args_bools;       
               Eigen::Matrix<int, -1, 1>         Model_args_ints;       
               Eigen::Matrix<double, -1, 1>      Model_args_doubles;    
               Eigen::Matrix<std::string, -1, 1> Model_args_strings;     
               
               std_vec_of_EigenVecs_dbl  Model_args_col_vecs_double;
               std_vec_of_EigenVecs_int  Model_args_col_vecs_int;
               std_vec_of_EigenMats_dbl  Model_args_mats_double;
               std_vec_of_EigenMats_int  Model_args_mats_int;
               
               two_layer_std_vec_of_EigenVecs_dbl      Model_args_vecs_of_col_vecs_double;
               two_layer_std_vec_of_EigenVecs_int      Model_args_vecs_of_col_vecs_int;
               two_layer_std_vec_of_EigenMats_dbl      Model_args_vecs_of_mats_double; // X goes here for non-LC MVP
               two_layer_std_vec_of_EigenMats_int      Model_args_vecs_of_mats_int;
               
               three_layer_std_vec_of_EigenVecs_dbl   Model_args_2_layer_vecs_of_col_vecs_double ;
               three_layer_std_vec_of_EigenVecs_int   Model_args_2_layer_vecs_of_col_vecs_int ;
               three_layer_std_vec_of_EigenMats_dbl   Model_args_2_layer_vecs_of_mats_double ;/// X goees here  for LC-MVP (4D array)
               three_layer_std_vec_of_EigenMats_int   Model_args_2_layer_vecs_of_mats_int ;
          
         // default constructor
         Model_fn_args_struct(int N, int n_nuisance, int n_params_main, 
                              int n_bools, int n_ints, int n_doubles, int n_strings,
                              int n_col_vecs_dbl,  int n_col_vecs_int,  int n_mats_dbl,  int n_mats_int,
                              int n_vecs_of_col_vecs_dbl,  int n_vecs_of_col_vecs_int,  int n_vecs_of_mats_dbl,  int n_vecs_of_mats_int,
                              int n_2_layer_vecs_of_col_vecs_dbl,  int n_2_layer_vecs_of_col_vecs_int,  int n_2_layer_vecs_of_mats_dbl,  int n_2_layer_vecs_of_mats_int)
           :
           N(N),
           n_nuisance(n_nuisance),
           n_params_main(n_params_main),
           model_so_file("none"),
           json_file_path("none") 
         {
             
               // initialize vectors with default sizes
               Model_args_bools.resize(n_bools);     
               Model_args_ints.resize(n_ints);      
               Model_args_doubles.resize(n_doubles);   
               Model_args_strings.resize(n_strings);  
               
               Model_args_col_vecs_double.resize(n_col_vecs_dbl);
               Model_args_col_vecs_int.resize(n_col_vecs_int);
               Model_args_mats_double.resize(n_mats_dbl);
               Model_args_mats_int.resize(n_mats_int);
               
               Model_args_vecs_of_col_vecs_double.resize(n_vecs_of_col_vecs_dbl);
               Model_args_vecs_of_col_vecs_int.resize(n_vecs_of_col_vecs_int);
               Model_args_vecs_of_mats_double.resize(n_vecs_of_mats_dbl);
               Model_args_vecs_of_mats_int.resize(n_vecs_of_mats_int);
               
               Model_args_2_layer_vecs_of_col_vecs_double.resize(n_2_layer_vecs_of_col_vecs_dbl);
               Model_args_2_layer_vecs_of_col_vecs_int.resize(n_2_layer_vecs_of_col_vecs_int);
               Model_args_2_layer_vecs_of_mats_double.resize(n_2_layer_vecs_of_mats_dbl);
               Model_args_2_layer_vecs_of_mats_int.resize(n_2_layer_vecs_of_mats_int);
          
         }
         
         
         // constructor w/ default values to handle optional/empty args
         Model_fn_args_struct(
               const int &N_,
               const int &n_nuisance_,
               const int &n_params_main_, 
               const std::string  &model_so_file_ = "none",
               const std::string  &json_file_path_ = "none",
               const Eigen::Matrix<bool, -1, 1>         &Model_args_bools_ = Eigen::Matrix<bool, -1, 1>(),
               const Eigen::Matrix<int, -1, 1>          &Model_args_ints_ = Eigen::Matrix<int, -1, 1>(),
               const Eigen::Matrix<double, -1, 1>       &Model_args_doubles_ = Eigen::Matrix<double, -1, 1>(),
               const Eigen::Matrix<std::string, -1, 1>  &Model_args_strings_ = Eigen::Matrix<std::string, -1, 1>(),
               const std_vec_of_EigenVecs_dbl &Model_args_col_vecs_double_ = std_vec_of_EigenVecs_dbl(),
               const std_vec_of_EigenVecs_int &Model_args_col_vecs_int_ = std_vec_of_EigenVecs_int(),
               const std_vec_of_EigenMats_dbl &Model_args_mats_double_ = std_vec_of_EigenMats_dbl(),
               const std_vec_of_EigenMats_int &Model_args_mats_int_ = std_vec_of_EigenMats_int(),
               const two_layer_std_vec_of_EigenVecs_dbl &Model_args_vecs_of_col_vecs_double_ = two_layer_std_vec_of_EigenVecs_dbl(),
               const two_layer_std_vec_of_EigenVecs_int &Model_args_vecs_of_col_vecs_int_ = two_layer_std_vec_of_EigenVecs_int(),
               const two_layer_std_vec_of_EigenMats_dbl &Model_args_vecs_of_mats_double_ = two_layer_std_vec_of_EigenMats_dbl(),
               const two_layer_std_vec_of_EigenMats_int &Model_args_vecs_of_mats_int_ = two_layer_std_vec_of_EigenMats_int(),
               const three_layer_std_vec_of_EigenVecs_dbl &Model_args_2_layer_vecs_of_col_vecs_double_ = three_layer_std_vec_of_EigenVecs_dbl(),
               const three_layer_std_vec_of_EigenVecs_int &Model_args_2_layer_vecs_of_col_vecs_int_ = three_layer_std_vec_of_EigenVecs_int(),
               const three_layer_std_vec_of_EigenMats_dbl &Model_args_2_layer_vecs_of_mats_double_ = three_layer_std_vec_of_EigenMats_dbl(),
               const three_layer_std_vec_of_EigenMats_int &Model_args_2_layer_vecs_of_mats_int_ = three_layer_std_vec_of_EigenMats_int()
         ) : 
           N(N_),
           n_nuisance(n_nuisance_),
           n_params_main(n_params_main_),
           model_so_file(model_so_file_),
           json_file_path(json_file_path_),
           Model_args_bools(Model_args_bools_),
           Model_args_ints(Model_args_ints_),
           Model_args_doubles(Model_args_doubles_),
           Model_args_strings(Model_args_strings_),
           Model_args_col_vecs_double(Model_args_col_vecs_double_),
           Model_args_col_vecs_int(Model_args_col_vecs_int_),
           Model_args_mats_double(Model_args_mats_double_),
           Model_args_mats_int(Model_args_mats_int_),
           Model_args_vecs_of_col_vecs_double(Model_args_vecs_of_col_vecs_double_),
           Model_args_vecs_of_col_vecs_int(Model_args_vecs_of_col_vecs_int_),
           Model_args_vecs_of_mats_double(Model_args_vecs_of_mats_double_),
           Model_args_vecs_of_mats_int(Model_args_vecs_of_mats_int_),
           Model_args_2_layer_vecs_of_col_vecs_double(Model_args_2_layer_vecs_of_col_vecs_double_),
           Model_args_2_layer_vecs_of_col_vecs_int(Model_args_2_layer_vecs_of_col_vecs_int_),
           Model_args_2_layer_vecs_of_mats_double(Model_args_2_layer_vecs_of_mats_double_),
           Model_args_2_layer_vecs_of_mats_int(Model_args_2_layer_vecs_of_mats_int_)
         {}
         
 };
 
 
 
 

   
 
 
 
 
 
 
 
 // struct for other function arguments to make function signitures more general  easier to manage
 // the struct name becomes a return type. So can use as a return argument to functions.
 struct EHMC_fn_args_struct {
    
    //// for main params
    double tau_main;
    double tau_main_ii;
    double eps_main;
    //// for nuisance params
    double tau_us;
    double tau_us_ii;
    double eps_us;
    
    //// general 
    bool diffusion_HMC;
    
    /////// constructor
    EHMC_fn_args_struct(
      const double &tau_main_,
      const double &tau_main_ii_,
      const double &eps_main_,
      const double &tau_us_,
      const double &tau_us_ii_,
      const double &eps_us_,
      const bool &diffusion_HMC_
    ) : 
      tau_main(tau_main_),
      tau_main_ii(tau_main_ii_),
      eps_main(eps_main_),
      tau_us(tau_us_),
      tau_us_ii(tau_us_ii_),
      eps_us(eps_us_),
      diffusion_HMC(diffusion_HMC_)
    {} 

 }; 
 
  


 
 
 
 
 // struct for other function arguments to make function signitures more general  easier to manage
 // the struct name becomes a return type. So can use as a return argument to functions.
 struct alignas(EIGEN_MAX_ALIGN_BYTES)   EHMC_burnin_struct {  // all params in this struct are shared between all chains
   
   // for main params
   double adapt_delta_main; // shared between all chains
   double LR_main; // shared between all chains
   double eps_m_adam_main; // shared between all chains
   double eps_v_adam_main; // shared between all chains
   double tau_m_adam_main; // shared between all chains
   double tau_v_adam_main; // shared between all chains
   double eigen_max_main;   // shared between all chains
   Eigen::VectorXi  index_main; // shared between all chains
   Eigen::Matrix<double, -1, -1> M_dense_sqrt; // shared between all chains
   Eigen::Matrix<double, -1, 1>  snaper_m_vec_main;   // shared between all chains
   Eigen::Matrix<double, -1, 1>  snaper_s_vec_main_empirical;  // shared between all chains
   Eigen::Matrix<double, -1, 1>  snaper_w_vec_main;  // shared between all chains
   Eigen::Matrix<double, -1, 1>  eigen_vector_main;   // shared between all chains
   
   // for nuisance params
   double adapt_delta_us;
   double LR_us;
   double eps_m_adam_us;
   double eps_v_adam_us;
   double tau_m_adam_us;
   double tau_v_adam_us;
   double eigen_max_us;  
   Eigen::VectorXi index_us;
   Eigen::Matrix<double, -1, 1>  sqrt_M_us_vec;
   Eigen::Matrix<double, -1, 1>  snaper_m_vec_us; 
   Eigen::Matrix<double, -1, 1>  snaper_s_vec_us_empirical; 
   Eigen::Matrix<double, -1, 1>  snaper_w_vec_us;  
   Eigen::Matrix<double, -1, 1>  eigen_vector_us;   
   
   /////// constructor
   EHMC_burnin_struct(
     ///// main params
     const double &adapt_delta_main_,
     const double &LR_main_,
     const double &eps_m_adam_main_,
     const double &eps_v_adam_main_,
     const double &tau_m_adam_main_,
     const double &tau_v_adam_main_,
     const double &eigen_max_main_,  
     const Eigen::VectorXi  &index_main_,
     const Eigen::Matrix<double, -1, -1> &M_dense_sqrt_,
     const Eigen::Matrix<double, -1, 1>  &snaper_m_vec_main_,
     const Eigen::Matrix<double, -1, 1>  &snaper_s_vec_main_empirical_,
     const Eigen::Matrix<double, -1, 1>  &snaper_w_vec_main_,
     const Eigen::Matrix<double, -1, 1>  &eigen_vector_main_,   
     ///// nuisance
     const double &adapt_delta_us_,
     const double &LR_us_,
     const double &eps_m_adam_us_,
     const double &eps_v_adam_us_,
     const double &tau_m_adam_us_,
     const double &tau_v_adam_us_,
     const double &eigen_max_us_,  
     const Eigen::VectorXi  &index_us_,
     const Eigen::Matrix<double, -1, 1>  &sqrt_M_us_vec_,
     const Eigen::Matrix<double, -1, 1>  &snaper_m_vec_us_, 
     const Eigen::Matrix<double, -1, 1>  &snaper_s_vec_us_empirical_, 
     const Eigen::Matrix<double, -1, 1>  &snaper_w_vec_us_,
     const Eigen::Matrix<double, -1, 1>  &eigen_vector_us_   
   ) :  
     ///// main params
     adapt_delta_main(adapt_delta_main_),
     LR_main(LR_main_),
     eps_m_adam_main(eps_m_adam_main_),
     eps_v_adam_main(eps_v_adam_main_),
     tau_m_adam_main(tau_m_adam_main_),
     tau_v_adam_main(tau_v_adam_main_),
     eigen_max_main(eigen_max_main_),   
     index_main(index_main_),
     M_dense_sqrt(M_dense_sqrt_),
     snaper_m_vec_main(snaper_m_vec_main_),
     snaper_s_vec_main_empirical(snaper_s_vec_main_empirical_),
     snaper_w_vec_main(snaper_w_vec_main_),
     eigen_vector_main(eigen_vector_main_),  
     /////// nuisance
     adapt_delta_us(adapt_delta_us_),
     LR_us(LR_us_),
     eps_m_adam_us(eps_m_adam_us_),
     eps_v_adam_us(eps_v_adam_us_),
     tau_m_adam_us(tau_m_adam_us_),
     tau_v_adam_us(tau_v_adam_us_),
     eigen_max_us(eigen_max_us_),   
     index_us(index_us_),
     sqrt_M_us_vec(sqrt_M_us_vec_),
     snaper_m_vec_us(snaper_m_vec_us_),
     snaper_s_vec_us_empirical(snaper_s_vec_us_empirical_),
     snaper_w_vec_us(snaper_w_vec_us_),
     eigen_vector_us(eigen_vector_us_)   
     {} 
   
 };  
 
 
 
 
 
 
 
 
 // struct for other function arguments to make function signitures more general  easier to manage
 // the struct name becomes a return type. So can use as a return argument to functions.
 
 /// we dont want to modify any of these so can use const & throughout the struct. 
 struct alignas(EIGEN_MAX_ALIGN_BYTES) EHMC_Metric_struct { // all params in this struct are shared between all chains
   
   // for main params
   Eigen::Matrix<double, -1, -1> M_dense_main; // using & here AND in the constructor (bit below) is very efficient but have to be careful when calling constructor 
   Eigen::Matrix<double, -1, -1> M_inv_dense_main;
   Eigen::Matrix<double, -1, -1> M_inv_dense_main_chol;
   Eigen::Matrix<double, -1, 1>  M_inv_main_vec;
   // for nuisance params
   Eigen::Matrix<double, -1, 1>  M_inv_us_vec;
   Eigen::Matrix<double, -1, 1>  M_us_vec;
   
   std::string metric_shape_main;
   
   /////// constructor w/ eevrything (some can be empty)
   EHMC_Metric_struct(
     const Eigen::Matrix<double, -1, -1>  &M_dense_main_,
     const Eigen::Matrix<double, -1, -1>  &M_inv_dense_main_,
     const Eigen::Matrix<double, -1, -1>  &M_inv_dense_main_chol_,
     const Eigen::Matrix<double, -1, 1>   &M_inv_main_vec_,
     const Eigen::Matrix<double, -1, 1>   &M_inv_us_vec_,
     const Eigen::Matrix<double, -1, 1>   &M_us_vec_,
     const std::string  &metric_shape_main_
   ) : 
     M_dense_main(M_dense_main_),
     M_inv_dense_main(M_inv_dense_main_),
     M_inv_dense_main_chol(M_inv_dense_main_chol_),
     M_inv_main_vec(M_inv_main_vec_),
     M_inv_us_vec(M_inv_us_vec_),
     M_us_vec(M_us_vec_),
     metric_shape_main(metric_shape_main_)
   {}
   
 }; 
 
 
 
 
 
 
 
 
struct alignas(EIGEN_MAX_ALIGN_BYTES)   HMCResult {
   
         Eigen::Matrix<double, -1, 1> lp_and_grad_outs;
         
         Eigen::Matrix<double, -1, 1> main_theta_vec_0;
         Eigen::Matrix<double, -1, 1> main_theta_vec;
         Eigen::Matrix<double, -1, 1> main_theta_vec_proposed;
         Eigen::Matrix<double, -1, 1> main_velocity_0_vec;
         Eigen::Matrix<double, -1, 1> main_velocity_vec_proposed;
         Eigen::Matrix<double, -1, 1> main_velocity_vec;
         
         double main_p_jump;
         int main_div;
         
         Eigen::Matrix<double, -1, 1> us_theta_vec_0;
         Eigen::Matrix<double, -1, 1> us_theta_vec;
         Eigen::Matrix<double, -1, 1> us_theta_vec_proposed;
         Eigen::Matrix<double, -1, 1> us_velocity_0_vec;
         Eigen::Matrix<double, -1, 1> us_velocity_vec_proposed;
         Eigen::Matrix<double, -1, 1> us_velocity_vec;
         
         double us_p_jump;
         int us_div;
         
         // Constructor to initialize matrix sizes
         HMCResult(int n_params_main, int n_us,  int N)
           : lp_and_grad_outs(Eigen::Matrix<double, -1, 1>::Zero((1 + N + n_params_main + n_us))),
             main_theta_vec_0(Eigen::Matrix<double, -1, 1>::Zero(n_params_main)),
             main_theta_vec(Eigen::Matrix<double, -1, 1>::Zero(n_params_main)),
             main_theta_vec_proposed(Eigen::Matrix<double, -1, 1>::Zero(n_params_main)),
             main_velocity_0_vec(Eigen::Matrix<double, -1, 1>::Zero(n_params_main)),
             main_velocity_vec_proposed(Eigen::Matrix<double, -1, 1>::Zero(n_params_main)),
             main_velocity_vec(Eigen::Matrix<double, -1, 1>::Zero(n_params_main)),
             main_p_jump(0.0),
             main_div(0),
             us_theta_vec_0(Eigen::Matrix<double, -1, 1>::Zero(n_us)),
             us_theta_vec(Eigen::Matrix<double, -1, 1>::Zero(n_us)),
             us_theta_vec_proposed(Eigen::Matrix<double, -1, 1>::Zero(n_us)),
             us_velocity_0_vec(Eigen::Matrix<double, -1, 1>::Zero(n_us)),
             us_velocity_vec_proposed(Eigen::Matrix<double, -1, 1>::Zero(n_us)),
             us_velocity_vec(Eigen::Matrix<double, -1, 1>::Zero(n_us)),
             us_p_jump(0.0),
             us_div(0)
         {}
   
};
 
 
 
 
 
 
 
 
 
 

  
struct alignas(EIGEN_MAX_ALIGN_BYTES) HMC_output_single_chain {

           alignas(EIGEN_MAX_ALIGN_BYTES) HMCResult result_input;

           // Group trace buffers together
           struct alignas(EIGEN_MAX_ALIGN_BYTES) TraceBuffers {
             Eigen::Matrix<double, -1, -1> main;
             Eigen::Matrix<double, -1, -1> div;
             Eigen::Matrix<double, -1, -1> nuisance;
             Eigen::Matrix<double, -1, -1> log_lik;
           } traces;

           // Group diagnostic vectors together
           struct alignas(EIGEN_MAX_ALIGN_BYTES) Diagnostics {
             Eigen::Matrix<int, -1, 1> div_us;
             Eigen::Matrix<int, -1, 1> div_main;
             Eigen::Matrix<double, -1, 1> p_jump_us;
             Eigen::Matrix<double, -1, 1> p_jump_main;
           } diagnostics;

           
       // Constructor
       HMC_output_single_chain(int n_iter,
                               int n_nuisance_to_track, 
                               int n_params_main, 
                               int n_us, 
                               int N)
         : result_input(n_params_main, n_us, N) {

             // Initialize traces buffers
             traces.main.resize(n_params_main, n_iter);
             traces.div.resize(1, n_iter);
             traces.nuisance.resize(n_nuisance_to_track, n_iter);
             traces.log_lik.resize(N, n_iter);
             
             // Initialize diagnostics
             diagnostics.div_us.resize(n_iter);
             diagnostics.div_main.resize(n_iter);
             diagnostics.p_jump_us.resize(n_iter);
             diagnostics.p_jump_main.resize(n_iter);

       }

};






struct ChunkSizeInfo {
  
        int chunk_size;
        int chunk_size_orig;
        int normal_chunk_size;
        int last_chunk_size;
        int n_total_chunks;
        int n_full_chunks;
        
};




ChunkSizeInfo calculate_chunk_sizes(const int N, 
                                    const int vec_size, 
                                    const int desired_n_chunks) {
  
        ChunkSizeInfo info;
        
        if (desired_n_chunks == 1) {
          info.chunk_size = N;
          info.chunk_size_orig = N;
          info.normal_chunk_size = N;
          info.last_chunk_size = N;
          info.n_total_chunks = 1;
          info.n_full_chunks = 1;
          
          return info;
          
        }
        
        const double N_double = static_cast<double>(N);
        const double vec_size_double = static_cast<double>(vec_size);
        const double desired_n_chunks_double = static_cast<double>(desired_n_chunks);
        
        info.normal_chunk_size = vec_size_double * std::floor(N_double / (vec_size_double * desired_n_chunks_double));
        info.n_full_chunks = std::floor(N_double / static_cast<double>(info.normal_chunk_size));
        info.last_chunk_size = N_double - (static_cast<double>(info.n_full_chunks) * static_cast<double>(info.normal_chunk_size));
        
        info.n_total_chunks = (info.last_chunk_size == 0) ? info.n_full_chunks : info.n_full_chunks + 1;
        
        info.chunk_size = info.normal_chunk_size;
        info.chunk_size_orig = info.normal_chunk_size;
        
        return info;
  
}


  
 
 
 
 
 
 
 // storage struct
 struct MVP_ThreadLocalWorkspace {
   
             ////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
             //// Matrices used in MVP / LC_MVP calculations
             ////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
             std::vector<Eigen::Matrix<double, -1, -1>>   Z_std_norm;
             std::vector<Eigen::Matrix<double, -1, -1>>   Bound_Z;
             std::vector<Eigen::Matrix<double, -1, -1>>   Bound_U_Phi_Bound_Z;
             std::vector<Eigen::Matrix<double, -1, -1>>   prob;
             std::vector<Eigen::Matrix<double, -1, -1>>   Phi_Z;
             ///////////////////////////////////////////////
             Eigen::Matrix<double, -1, -1>   y1_log_prob;
             Eigen::Matrix<double, -1, -1>   phi_Z_recip;
             Eigen::Matrix<double, -1, -1>   phi_Bound_Z;
             ///////////////////////////////////////////////
             Eigen::Matrix<double, -1, -1>     u_grad_array_CM_chunk;
             ///////////////////////////////////////////////
             Eigen::Matrix<double, -1, -1>     common_grad_term_1;
             Eigen::Matrix<double, -1, -1>     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip;
             Eigen::Matrix<double, -1, -1>     y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip;
             Eigen::Matrix<double, -1, -1>     prob_rowwise_prod_temp;
             Eigen::Matrix<double, -1, -1>     prob_recip_rowwise_prod_temp;
             ///////////////////////////////////////////////
             Eigen::Matrix<double, -1, 1>      prod_container_or_inc_array;
             Eigen::Matrix<double, -1, 1>      derivs_chain_container_vec;
             Eigen::Matrix<double, -1, 1>      prob_rowwise_prod_temp_all;
             ///////////////////////////////////////////////
             Eigen::Matrix<double, -1, -1>     grad_prob;
             Eigen::Matrix<double, -1, -1>     z_grad_term;
             ///////////////////////////////////////////////
             Eigen::Matrix<double, -1, -1>     y_chunk;
             Eigen::Matrix<double, -1, -1>     u_array;
             Eigen::Matrix<double, -1, -1>     y_sign;
             Eigen::Matrix<double, -1, -1>     y_m_y_sign_x_u;
             ///////////////////////////////////////////////
             Eigen::Matrix<double, -1, -1>   u_grad_array_CM_chunk_block;
             ///////////////////////////////////////////////
             Eigen::Matrix<double, -1, 1>   u_unc_vec_chunk;
             Eigen::Matrix<double, -1, 1>   u_vec_chunk;
             Eigen::Matrix<double, -1, 1>   du_wrt_duu_chunk;
             Eigen::Matrix<double, -1, 1>   d_J_wrt_duu_chunk;
             ///////////////////////////////////////////////
             Eigen::Matrix<double, -1, -1>   lp_array;
             ///////////////////////////////////////////////
   
   // Keep track of current dimensions
   int current_chunk_size;
   
   ///// Constructor
   MVP_ThreadLocalWorkspace( int chunk_size,
                             int n_tests,
                             int n_class) : 
     current_chunk_size(chunk_size) {
     
     // Initialize vectors of matrices
     Z_std_norm.resize(n_class);
     Bound_Z.resize(n_class);
     Bound_U_Phi_Bound_Z.resize(n_class);
     prob.resize(n_class);
     Phi_Z.resize(n_class);
     
     // Initialize each matrix
     for (int c = 0; c < n_class; c++) {
       Z_std_norm[c] = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
       Bound_Z[c] = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
       Bound_U_Phi_Bound_Z[c] = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
       prob[c] = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
       Phi_Z[c] = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     }
     
     ///////////////////////////////////////////////
     y1_log_prob = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     phi_Z_recip = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     phi_Bound_Z = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     ///////////////////////////////////////////////
     u_grad_array_CM_chunk = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     ///////////////////////////////////////////////
     common_grad_term_1 = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     prob_rowwise_prod_temp = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     prob_recip_rowwise_prod_temp = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     ///////////////////////////////////////////////
     prod_container_or_inc_array = Eigen::Matrix<double, -1, 1>(chunk_size);
     derivs_chain_container_vec =  Eigen::Matrix<double, -1, 1>(chunk_size);
     prob_rowwise_prod_temp_all =  Eigen::Matrix<double, -1, 1>(chunk_size);
     ///////////////////////////////////////////////
     grad_prob = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     z_grad_term = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     ///////////////////////////////////////////////
     y_chunk = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     u_array = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     y_sign = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     y_m_y_sign_x_u = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     ///////////////////////////////////////////////
     u_grad_array_CM_chunk_block = Eigen::Matrix<double, -1, -1>(chunk_size, n_tests);
     ///////////////////////////////////////////////
     u_unc_vec_chunk =   Eigen::Matrix<double, -1, 1>(chunk_size * n_tests);
     u_vec_chunk =       Eigen::Matrix<double, -1, 1>(chunk_size * n_tests);
     du_wrt_duu_chunk =  Eigen::Matrix<double, -1, 1>(chunk_size * n_tests);
     d_J_wrt_duu_chunk = Eigen::Matrix<double, -1, 1>(chunk_size * n_tests);
     ///////////////////////////////////////////////
     lp_array = Eigen::Matrix<double, -1, -1>(chunk_size, n_class);
     ///////////////////////////////////////////////
 
               ///////////////////////////////////////////////
               y1_log_prob.setZero();
               phi_Z_recip.setZero();
               phi_Bound_Z.setZero();
               ///////////////////////////////////////////////
               u_grad_array_CM_chunk.setZero();
               ///////////////////////////////////////////////
               common_grad_term_1.setZero();
               y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.setZero();
               y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.setZero();
               prob_rowwise_prod_temp.setZero();
               prob_recip_rowwise_prod_temp.setZero();
               ///////////////////////////////////////////////
               prod_container_or_inc_array.setZero();
               derivs_chain_container_vec.setZero();
               prob_rowwise_prod_temp_all.setZero();
               ///////////////////////////////////////////////
               grad_prob.setZero();
               z_grad_term.setZero();
               ///////////////////////////////////////////////
               y_chunk.setZero();
               u_array.setZero();
               y_sign.setZero();
               y_m_y_sign_x_u.setZero();
               ///////////////////////////////////////////////
               u_grad_array_CM_chunk_block.setZero();
               ///////////////////////////////////////////////
               u_unc_vec_chunk.setZero();
               u_vec_chunk.setZero();
               du_wrt_duu_chunk.setZero();
               d_J_wrt_duu_chunk.setZero();
               ///////////////////////////////////////////////
               lp_array.setZero();
               ///////////////////////////////////////////////
   }
   
   //// Resize method for when chunk size changes
   void resize_for_last_chunk(int last_chunk_size, 
                              int n_tests,
                              int n_class) {
     
             ///////////////////////////////////////////////
             for (int c = 0; c < n_class; c++) {
               Z_std_norm[c].resize(last_chunk_size, n_tests);
               Bound_Z[c].resize(last_chunk_size, n_tests);
               Bound_U_Phi_Bound_Z[c].resize(last_chunk_size, n_tests);
               prob[c].resize(last_chunk_size, n_tests);
               Phi_Z[c].resize(last_chunk_size, n_tests);
             }
             ///////////////////////////////////////////////
             y1_log_prob.resize(last_chunk_size, n_tests);
             phi_Z_recip.resize(last_chunk_size, n_tests);
             phi_Bound_Z.resize(last_chunk_size, n_tests);
             ///////////////////////////////////////////////
             u_grad_array_CM_chunk.resize(last_chunk_size, n_tests);
             ///////////////////////////////////////////////
             common_grad_term_1.resize(last_chunk_size, n_tests);
             y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
             y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
             prob_rowwise_prod_temp.resize(last_chunk_size, n_tests);
             prob_recip_rowwise_prod_temp.resize(last_chunk_size, n_tests);
             ///////////////////////////////////////////////
             prod_container_or_inc_array.resize(last_chunk_size);
             derivs_chain_container_vec.resize(last_chunk_size);
             prob_rowwise_prod_temp_all.resize(last_chunk_size);
             ///////////////////////////////////////////////
             grad_prob.resize(last_chunk_size, n_tests);
             z_grad_term.resize(last_chunk_size, n_tests);
             ///////////////////////////////////////////////
             y_chunk.resize(last_chunk_size, n_tests);
             u_array.resize(last_chunk_size, n_tests);
             y_sign.resize(last_chunk_size, n_tests);
             y_m_y_sign_x_u.resize(last_chunk_size, n_tests);
             ///////////////////////////////////////////////
             u_grad_array_CM_chunk_block.resize(last_chunk_size, n_tests);
             ///////////////////////////////////////////////
             u_unc_vec_chunk.resize(last_chunk_size * n_tests);
             u_vec_chunk.resize(last_chunk_size * n_tests);
             du_wrt_duu_chunk.resize(last_chunk_size * n_tests);
             d_J_wrt_duu_chunk.resize(last_chunk_size * n_tests);
             ///////////////////////////////////////////////
             lp_array.resize(last_chunk_size, n_class);
             ///////////////////////////////////////////////
             
             /// Reset values (to zero)
             ///////////////////////////////////////////////
             y1_log_prob.setZero();
             phi_Z_recip.setZero();
             phi_Bound_Z.setZero();
             ///////////////////////////////////////////////
             u_grad_array_CM_chunk.setZero();
             ///////////////////////////////////////////////
             common_grad_term_1.setZero();
             y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.setZero();
             y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.setZero();
             prob_rowwise_prod_temp.setZero();
             prob_recip_rowwise_prod_temp.setZero();
             ///////////////////////////////////////////////
             prod_container_or_inc_array.setZero();
             derivs_chain_container_vec.setZero();
             prob_rowwise_prod_temp_all.setZero();
             ///////////////////////////////////////////////
             grad_prob.setZero();
             z_grad_term.setZero();
             ///////////////////////////////////////////////
             y_chunk.setZero();
             u_array.setZero();
             y_sign.setZero();
             y_m_y_sign_x_u.setZero();
             ///////////////////////////////////////////////
             u_grad_array_CM_chunk_block.setZero();
             ///////////////////////////////////////////////
             u_unc_vec_chunk.setZero();
             u_vec_chunk.setZero();
             du_wrt_duu_chunk.setZero();
             d_J_wrt_duu_chunk.setZero();
             ///////////////////////////////////////////////
             lp_array.setZero();
             ///////////////////////////////////////////////
     
   }
   
 };
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
  