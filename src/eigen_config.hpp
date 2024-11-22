#ifndef EIGEN_CONFIG_H
#define EIGEN_CONFIG_H



#ifdef EIGEN_MAX_ALIGN_BYTES
#undef EIGEN_MAX_ALIGN_BYTES

#if defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__) // use AVX-512 if available 
  #define EIGEN_VECTORIZE_AVX512
  #define EIGEN_MAX_ALIGN_BYTES 64
#elif defined(__AVX2__) && ( !(defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)) ) // use AVX2
  #define EIGEN_VECTORIZE_AVX2
  #define EIGEN_MAX_ALIGN_BYTES 32 
#elif defined(__AVX__) && !(defined(__AVX2__)) &&  ( !(defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)) ) // use AVX2 // use AVX
  #define EIGEN_VECTORIZE_AVX
  #define EIGEN_MAX_ALIGN_BYTES 16
#endif




#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE






#endif   