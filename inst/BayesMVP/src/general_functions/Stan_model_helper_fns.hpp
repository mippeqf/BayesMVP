
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
 
 

 
 
 
 
// #ifdef _WIN32
// #include <dlfcn.h>
// #include <windows.h>
// #define dlopen(x,y) LoadLibrary(x)
// #define dlclose(x)  FreeLibrary((HMODULE)x)
// #define dlsym(x,y)  GetProcAddress((HMODULE)x,y)
// #define dlerror() "Windows error"
// #else
// #include <dlfcn.h> // For dynamic loading 
// #endif
 
#ifdef _WIN32
#include <windows.h>
#define dlopen(x,y) LoadLibraryA(x)  // Use LoadLibraryA for ANSI strings
#define dlclose(x) FreeLibrary((HMODULE)x)
#define dlsym(x,y) GetProcAddress((HMODULE)x,y)
#define dlerror() GetLastError()  // Get actual Windows error code
#else
#include <dlfcn.h>
#endif


 
using namespace Eigen;

 

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE

 

 



 
 
 


// fn to handle JSON via file input and compute the log-prob and gradient
ALWAYS_INLINE bs_model* fn_convert_JSON_data_to_BridgeStan(ModelHandle_struct &model_handle,
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





 
#ifdef _WIN32
 

 
 //// fn to dynamically load the user-provided .so file and resolve symbols
 ALWAYS_INLINE Stan_model_struct fn_load_Stan_model_and_data( const std::string &model_so_file, 
                                                              const std::string &json_file,
                                                              unsigned int seed) {
   
             void* bs_handle;
             
             // Load the DLL/SO file
 
             bs_handle = LoadLibraryA(model_so_file.c_str());
             if (!bs_handle) {
               DWORD error = GetLastError();
               char error_msg[256];
               FormatMessageA(
                 FORMAT_MESSAGE_FROM_SYSTEM,
                 NULL,
                 error,
                 0,
                 error_msg,
                 sizeof(error_msg),
                 NULL
               );
               throw std::runtime_error("Error loading DLL: " + std::string(error_msg));
             }
 
             
             // Function pointer types
             typedef bs_model* (*bs_model_construct_func)(const char*, unsigned int, char**);
             typedef int (*bs_log_density_gradient_func)(bs_model*, bool, bool, const double*, double*, double*, char**);
             typedef int (*bs_param_constrain_func)(bs_model*, bool, bool, const double*, double*, bs_rng*, char**);
             typedef bs_rng* (*bs_rng_construct_func)(unsigned int, char**);
             
             // Load bs_model_construct
             bs_model_construct_func bs_model_construct = (bs_model_construct_func)GetProcAddress((HMODULE)bs_handle, "bs_model_construct");
             if (!bs_model_construct) {
               DWORD error = GetLastError();
               char error_msg[256];
               FormatMessageA(
                 FORMAT_MESSAGE_FROM_SYSTEM,
                 NULL,
                 error,
                 0,
                 error_msg,
                 sizeof(error_msg),
                 NULL
               );
               FreeLibrary((HMODULE)bs_handle);
               throw std::runtime_error("Error loading symbol 'bs_model_construct': " + std::string(error_msg));
             }
 
             
             // Load bs_log_density_gradient
             bs_log_density_gradient_func bs_log_density_gradient = (bs_log_density_gradient_func)GetProcAddress((HMODULE)bs_handle, "bs_log_density_gradient");
             if (!bs_log_density_gradient) {
               DWORD error = GetLastError();
               char error_msg[256];
               FormatMessageA(
                 FORMAT_MESSAGE_FROM_SYSTEM,
                 NULL,
                 error,
                 0,
                 error_msg,
                 sizeof(error_msg),
                 NULL
               );
               FreeLibrary((HMODULE)bs_handle);
               throw std::runtime_error("Error loading symbol 'bs_log_density_gradient': " + std::string(error_msg));
             }
 
             
             // Load bs_param_constrain
             bs_param_constrain_func bs_param_constrain = (bs_param_constrain_func)GetProcAddress((HMODULE)bs_handle, "bs_param_constrain");
             if (!bs_param_constrain) {
               DWORD error = GetLastError();
               char error_msg[256];
               FormatMessageA(
                 FORMAT_MESSAGE_FROM_SYSTEM,
                 NULL,
                 error,
                 0,
                 error_msg,
                 sizeof(error_msg),
                 NULL
               );
               FreeLibrary((HMODULE)bs_handle);
               throw std::runtime_error("Error loading symbol 'bs_param_constrain': " + std::string(error_msg));
             }
 
             
             // Load bs_rng_construct
             bs_rng_construct_func bs_rng_construct = (bs_rng_construct_func)GetProcAddress((HMODULE)bs_handle, "bs_rng_construct");
             if (!bs_rng_construct) {
               DWORD error = GetLastError();
               char error_msg[256];
               FormatMessageA(
                 FORMAT_MESSAGE_FROM_SYSTEM,
                 NULL,
                 error, 
                 0,
                 error_msg,
                 sizeof(error_msg),
                 NULL
               );
               FreeLibrary((HMODULE)bs_handle);
               throw std::runtime_error("Error loading symbol 'bs_rng_construct': " + std::string(error_msg));
             }
 
             
             // Create model handle struct
             ModelHandle_struct model_handle = {
               bs_handle,
               bs_model_construct,
               bs_log_density_gradient,
               bs_param_constrain,
               bs_rng_construct
             };
             
             // Convert JSON data to BridgeStan model
             bs_model* bs_model_ptr = fn_convert_JSON_data_to_BridgeStan(model_handle, json_file, seed);
             
             // Return the complete struct
             return {
               bs_handle,
               bs_model_ptr,
               bs_model_construct,
               bs_log_density_gradient,
               bs_param_constrain,
               bs_rng_construct
             };
             
 }
 
 
 
 

#else 

 
// fn to dynamically load the user-provided .so file and resolve symbols
ALWAYS_INLINE Stan_model_struct fn_load_Stan_model_and_data( const std::string &model_so_file, 
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
           
           
           ModelHandle_struct model_handle = {bs_handle,
                                              bs_model_construct,
                                              bs_log_density_gradient, 
                                              bs_param_constrain, 
                                              bs_rng_construct};
           
           bs_model* bs_model_ptr = fn_convert_JSON_data_to_BridgeStan(model_handle, json_file, seed);
   
           
           // return {bs_model_ptr, bs_handle, bs_model_construct, bs_log_density_gradient};  // Return handle and fn pointers 
           
           return {bs_handle,                 
                   bs_model_ptr,              
                   bs_model_construct,        
                   bs_log_density_gradient, 
                   bs_param_constrain, 
                   bs_rng_construct};   
           
   
 }
 
 
#endif
 
 
 
 
 
 
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
 
 
  
 
 
// fn to clean up Stan model object once sampling is finished 
ALWAYS_INLINE void fn_bs_destroy_Stan_model(Stan_model_struct &Stan_model_as_cpp_struct) {
   
             if (Stan_model_as_cpp_struct.bs_model_ptr) {
               bs_model_destruct(Stan_model_as_cpp_struct.bs_model_ptr);
               Stan_model_as_cpp_struct.bs_model_ptr = nullptr;
             }
             
             if (Stan_model_as_cpp_struct.bs_handle) {
               if (dlclose(Stan_model_as_cpp_struct.bs_handle) != 0) {
                 throw std::runtime_error("Error closing .so file: " + std::string(dlerror()));
               } 
               Stan_model_as_cpp_struct.bs_handle = nullptr;
             } 
   
} 
 
 

 

 
  
  
  
  
  
  
  
  
  