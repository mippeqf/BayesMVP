
#pragma once

 

#include <sstream>
#include <stdexcept>  
#include <complex>

#include <map>
#include <vector>  
#include <string> 
#include <stdexcept>
#include <stdio.h>
#include <iostream>
 
#include <stan/model/model_base.hpp>  
 
#include <stan/io/array_var_context.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/dump.hpp> 

 
#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/rapidjson_parser.hpp>   
 
 




#include <Eigen/Dense>
 
 

 
 
 
 
#ifdef _WIN32
#include <windows.h>
#define RTLD_LAZY 0  // Windows doesn't need this flag but define for compatibility
#define dlopen(x,y) LoadLibrary(x) //#define dlopen(x,y) LoadLibraryA(x)
#define dlclose(x) FreeLibrary((HMODULE)x)
#define dlsym(x,y) GetProcAddress((HMODULE)x,y)
#else
#include <dlfcn.h>
#endif

#ifdef _WIN32
inline std::string windows_error_str() {
   
   char error_msg[256];
   DWORD error = GetLastError();
   DWORD flags = FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS;
   
   FormatMessageA(   flags,
                     NULL,
                     error,
                     MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                     error_msg,
                     sizeof(error_msg),
                     NULL);
   
   return std::string(error_msg);
   
 }
 
#define dlerror() windows_error_str()
#endif



 
using namespace Eigen;

 


 
 
 ////  Struct to hold the model handle and function pointers
 struct ModelHandle_struct {

   void* bs_handle = nullptr;
   bs_model* (*bs_model_construct)(const char*, unsigned int, char**) = nullptr;
   int (*bs_log_density_gradient)(bs_model*, bool, bool, const double*, double*, double*, char**) = nullptr;
   int (*bs_param_constrain)(bs_model*, bool, bool, const double*, double*, bs_rng*, char**) = nullptr;
   bs_rng* (*bs_rng_construct)(unsigned int, char**) = nullptr;
   void (*bs_model_destruct)(bs_model*) = nullptr;
   void (*bs_rng_destruct)(bs_rng*) = nullptr;
   
 };
 
 
 
 struct Stan_model_struct {
   
   void* bs_handle = nullptr; // has no arguments
   bs_model* bs_model_ptr = nullptr; // has no arguments
   bs_model* (*bs_model_construct)(const char*, unsigned int, char**) = nullptr;
   int (*bs_log_density_gradient)(bs_model*, bool, bool, const double*, double*, double*, char**) = nullptr;
   int (*bs_param_constrain)(bs_model*, bool, bool, const double*, double*, bs_rng*, char**) = nullptr;
   bs_rng* (*bs_rng_construct)(unsigned int, char**) = nullptr;
   void (*bs_model_destruct)(bs_model*) = nullptr;
   void (*bs_rng_destruct)(bs_rng*) = nullptr;
   
 };
 
 
 
  
 
 


//// fn to handle JSON via file input and compute the log-prob and gradient
bs_model* fn_convert_JSON_data_to_BridgeStan(ModelHandle_struct &model_handle,
                                             const std::string  &json_file, 
                                             unsigned int seed) {
 
     // Load the Stan model from the .so file using BridgeStan
     char* error_msg = nullptr;
     //  unsigned int seed = seed;
     
     // Use the user-provided JSON file path and construct the bs_model_ptr
     bs_model* bs_model_ptr = model_handle.bs_model_construct(json_file.c_str(), seed, &error_msg);
     
     if (!bs_model_ptr) {
       throw std::runtime_error("Error constructing the model: " + std::string(error_msg ? error_msg : "Unknown error"));
     } 
     
     return bs_model_ptr; 
 
}





 

 
// fn to dynamically load the user-provided .so file and resolve symbols
Stan_model_struct fn_load_Stan_model_and_data( const std::string &model_so_file, 
                                                             const std::string &json_file,
                                                             unsigned int seed) {
   
           // Load the .so file
           void* bs_handle = dlopen(model_so_file.c_str(), RTLD_LAZY);
           if (!bs_handle) {
             throw std::runtime_error("Error loading .so file: " + std::string(dlerror()));
           } 
           
           // Resolve the bs_model_construct symbol
           typedef bs_model* (*bs_model_construct_func)(const char*, unsigned int, char**);
           bs_model_construct_func bs_model_construct = (bs_model_construct_func)dlsym(bs_handle, "bs_model_construct"); 
           if (!bs_model_construct) {
             dlclose(bs_handle); 
             throw std::runtime_error("Error loading symbol 'bs_model_construct': " + std::string(dlerror()));
           }  
           
           // Resolve the bs_log_density_gradient symbol 
           typedef int (*bs_log_density_gradient_func)(bs_model*, bool, bool, const double*, double*, double*, char**);
           bs_log_density_gradient_func bs_log_density_gradient = (bs_log_density_gradient_func)dlsym(bs_handle, "bs_log_density_gradient");
           if (!bs_log_density_gradient) { 
             dlclose(bs_handle);
             throw std::runtime_error("Error loading symbol 'bs_log_density_gradient': " + std::string(dlerror()));
           }  
           
           // Resolve the bs_param_constrain symbol 
           typedef int (*bs_param_constrain_func)(bs_model*, bool, bool, const double*, double*, bs_rng*, char**);
           bs_param_constrain_func bs_param_constrain = (bs_param_constrain_func)dlsym(bs_handle, "bs_param_constrain");
           if (!bs_param_constrain) { 
             dlclose(bs_handle);
             throw std::runtime_error("Error loading symbol 'bs_param_constrain': " + std::string(dlerror()));
           } 
           
           // Resolve the bs_rng_construct symbol
           typedef bs_rng* (*bs_rng_construct_func)(unsigned int, char**);
           bs_rng_construct_func bs_rng_construct = (bs_rng_construct_func)dlsym(bs_handle, "bs_rng_construct"); 
           if (!bs_rng_construct) {
             dlclose(bs_handle); 
             throw std::runtime_error("Error loading symbol 'bs_rng_construct': " + std::string(dlerror()));
           }   
           
           // Resolve the bs_model_destruct symbol
           typedef void (*bs_model_destruct_func)(bs_model*);
           bs_model_destruct_func bs_model_destruct = (bs_model_destruct_func)dlsym(bs_handle, "bs_model_destruct"); 
           if (!bs_model_destruct) {
             dlclose(bs_handle); 
             throw std::runtime_error("Error loading symbol 'bs_model_destruct': " + std::string(dlerror()));
           }    
           
           // Resolve the bs_rng_destruct symbol
           typedef void (*bs_rng_destruct_func)(bs_rng*);
           bs_rng_destruct_func bs_rng_destruct = (bs_rng_destruct_func)dlsym(bs_handle, "bs_rng_destruct"); 
           if (!bs_rng_destruct) {
             dlclose(bs_handle); 
             throw std::runtime_error("Error loading symbol 'bs_rng_destruct': " + std::string(dlerror()));
           }     
           
           ModelHandle_struct model_handle = {bs_handle,
                                              bs_model_construct,
                                              bs_log_density_gradient, 
                                              bs_param_constrain, 
                                              bs_rng_construct,
                                              bs_model_destruct,
                                              bs_rng_destruct};
           
           bs_model* bs_model_ptr = fn_convert_JSON_data_to_BridgeStan(model_handle, json_file, seed);
   
           
           // return {bs_model_ptr, bs_handle, bs_model_construct, bs_log_density_gradient};  // Return handle and fn pointers 
           
           return {bs_handle,                 
                   bs_model_ptr,              
                   bs_model_construct,        
                   bs_log_density_gradient, 
                   bs_param_constrain, 
                   bs_rng_construct,
                   bs_model_destruct,
                   bs_rng_destruct};   
           
   
 }
 
 
 
 
 
 
 
 
Eigen::Matrix<double, -1, 1> fn_Stan_compute_log_prob_grad(    const Stan_model_struct &Stan_model_as_cpp_struct,  
                                                               const Eigen::Matrix<double, -1, 1> &params,
                                                               const int n_params_main, 
                                                               const int n_nuisance,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, 1>> lp_and_grad_outs) { 
   
             if (!Stan_model_as_cpp_struct.bs_model_ptr || !Stan_model_as_cpp_struct.bs_log_density_gradient) {
               throw std::runtime_error("Model not properly initialized");
             }
             
             const int n_params = params.size();
             if (lp_and_grad_outs.size() != (n_params + 1)) {
               throw std::runtime_error("Output vector size mismatch");
             } 
             
             double log_prob_val = 0.0;
             char* error_msg = nullptr;
             
             int result;
             
             if (n_nuisance > 10) {
               
                   result = Stan_model_as_cpp_struct.bs_log_density_gradient(  Stan_model_as_cpp_struct.bs_model_ptr,
                                                                               true,
                                                                               true,
                                                                               params.data(),
                                                                               &log_prob_val,
                                                                               lp_and_grad_outs.segment(1, n_params).data(),
                                                                               &error_msg);
               
             } else { 
               
                   result = Stan_model_as_cpp_struct.bs_log_density_gradient(  Stan_model_as_cpp_struct.bs_model_ptr,
                                                                               true,
                                                                               true,
                                                                               params.data(),
                                                                               &log_prob_val,
                                                                               lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).data(),
                                                                               &error_msg);
               
             }
             
             if (result != 0) {
               throw std::runtime_error("Gradient computation failed: " + 
                                        std::string(error_msg ? error_msg : "Unknown error"));
             } 
             
             lp_and_grad_outs(0) = log_prob_val;
             return lp_and_grad_outs;
            
 }
 
 
  
 



#ifdef _WIN32
 
void fn_bs_destroy_Stan_model(Stan_model_struct &Stan_model_as_cpp_struct) {
   
         if (Stan_model_as_cpp_struct.bs_model_ptr && Stan_model_as_cpp_struct.bs_model_destruct) {
           Stan_model_as_cpp_struct.bs_model_destruct(Stan_model_as_cpp_struct.bs_model_ptr);
           Stan_model_as_cpp_struct.bs_model_ptr = nullptr; 
         }
         
         if (Stan_model_as_cpp_struct.bs_handle) {
           
           void* handle = Stan_model_as_cpp_struct.bs_handle;  // Keep a copy
           Stan_model_as_cpp_struct.bs_handle = nullptr;  // Clear it first
           if (FreeLibrary((HMODULE)handle) == 0) {  // Use the copy
             throw std::runtime_error("Error closing library: " + std::string(dlerror()));
           }
           
         }
 
   
}

#else

void fn_bs_destroy_Stan_model(Stan_model_struct &Stan_model_as_cpp_struct) {
  
        if (Stan_model_as_cpp_struct.bs_model_ptr && Stan_model_as_cpp_struct.bs_model_destruct) {
          Stan_model_as_cpp_struct.bs_model_destruct(Stan_model_as_cpp_struct.bs_model_ptr);
          Stan_model_as_cpp_struct.bs_model_ptr = nullptr;
        }
        
        if (Stan_model_as_cpp_struct.bs_handle) {
          
            void* handle = Stan_model_as_cpp_struct.bs_handle;  // Keep a copy
            Stan_model_as_cpp_struct.bs_handle = nullptr;  // Clear it first
            if (dlclose(handle) != 0) {  // Use the copy
              throw std::runtime_error("Error closing library: " + std::string(dlerror()));
            }
          
        }
  
}
 
#endif
  
  
  
  
  
  
  
  
  
#if HAS_BRIDGESTAN_H
  
  
  struct ParamConstrainWorker : public RcppParallel::Worker {
    
    //// Inputs
    const std::vector<Eigen::Matrix<double, -1, -1>> unc_params_trace_input_main;
    const std::vector<Eigen::Matrix<double, -1, -1>> unc_params_trace_input_nuisance;
    const Eigen::VectorXi pars_indicies_to_track;
    const int n_params_full;
    const int n_nuisance;
    const int n_params_main;
    const bool include_nuisance;
    const std::string &model_so_file;
    const std::string &json_file_path;
    
    //// Output uses tbb container
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
        
        Stan_model_struct Stan_model_as_cpp_struct;
        bs_rng* bs_rng_object = nullptr;
        
        // Initialize model and RNG once per thread
        if (model_so_file != "none" && Stan_model_as_cpp_struct.bs_model_ptr == nullptr) {
          Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(model_so_file, json_file_path, 123);
          bs_rng_object = Stan_model_as_cpp_struct.bs_rng_construct(123, &error_msg);
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
           Stan_model_as_cpp_struct.bs_rng_destruct(bs_rng_object); // BOOKMARK - need to update struct to include "bs_rng_destruct"
          // bs_rng_destruct(bs_rng_object); // BOOKMARK - need to update struct to include "bs_rng_destruct"
        }
        
        // Clean up thread-local resources 
        fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
        
        
      }
      
      
      
    }
    
    
  };
  
  
#endif
  
  
  
  