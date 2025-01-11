#pragma once 


#ifndef EIGEN_CONFIG_H
#define EIGEN_CONFIG_H


#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
 

#ifdef EIGEN_MAX_ALIGN_BYTES
#undef EIGEN_MAX_ALIGN_BYTES
#endif
 

#ifdef _WIN32
 
      //  #define EIGEN_DONT_ALIGN_STATICALLY
     
    #if defined(USE_AVX_512) //// use AVX-512 if available
          #define EIGEN_VECTORIZE_AVX512
          #define EIGEN_MAX_ALIGN_BYTES 64
    #elif defined(USE_AVX2)  //// use AVX2 (if AVX-512 NOT available)
          #define EIGEN_VECTORIZE_AVX2
          #define EIGEN_MAX_ALIGN_BYTES 32
    #elif defined(USE_AVX)   //// use AVX (if both AVX-512 and AVX2 NOT available)
          #define EIGEN_VECTORIZE_AVX
          #define EIGEN_MAX_ALIGN_BYTES 16 
    #endif 
 
#else
 
    #if defined(USE_AVX_512) //// use AVX-512 if available
          #define EIGEN_VECTORIZE_AVX512
          #define EIGEN_MAX_ALIGN_BYTES 64
    #elif defined(USE_AVX2)  //// use AVX2 (if AVX-512 NOT available)
          #define EIGEN_VECTORIZE_AVX2
          #define EIGEN_MAX_ALIGN_BYTES 32
    #elif defined(USE_AVX)   //// use AVX (if both AVX-512 and AVX2 NOT available)
          #define EIGEN_VECTORIZE_AVX
          #define EIGEN_MAX_ALIGN_BYTES 16
        #endif
     
#endif





#endif
     