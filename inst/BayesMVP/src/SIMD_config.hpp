#pragma once 


#ifndef SIMD_CONFIG_H
#define SIMD_CONFIG_H
 
 
//// Define global VECTORCALL (if any)
#if defined(_MSC_VER)
    #define VECTORCALL __vectorcall
#else
    #define VECTORCALL
#endif
 

//// Define AVX type to use (if any)
#if defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__) // use AVX-512 if available 
     #define USE_AVX_512// 1
     #pragma message "USE_AVX_512 has been defined!"
#elif defined(__AVX2__) && ( !(defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)) ) // use AVX2 if AVX-512 NOT available
     #define USE_AVX2// 1
     #pragma message "USE_AVX2 has been defined!"
#elif defined(__AVX__) && !(defined(__AVX2__)) &&  ( !(defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)) ) // use AVX
     #define USE_AVX// 1
     #pragma message "USE_AVX has been defined!"
#endif
 
 
//// Define global custom-AVX (i.e. intrinsics) function typedef (if any)
#if   defined(USE_AVX_512) //// use AVX-512 if available
     typedef __m512d (*FuncAVX)(const __m512d); 
     #pragma message "USE_AVX_512 is defined!"
#elif defined(USE_AVX2)  //// use AVX2 (if AVX-512 NOT available)
     typedef __m256d (*FuncAVX)(const __m256d);
     #pragma message "USE_AVX2 is defined!"
#elif defined(USE_AVX)   //// use AVX (if both AVX-512 and AVX2 NOT available)
     typedef __m128d (*FuncAVX)(const __m128d); 
     #pragma message "USE_AVX is defined!"
#endif
 
 
//// Define frd-declaration for AVX wrapper fn (or dummy fn if neither AVX-512 or AVX2 are available)
#if  (defined(USE_AVX2) || defined(USE_AVX_512))
#pragma message "defining fwd. declaration for fn_process_Ref_double_AVX"

template <typename T>
MAYBE_INLINE void fn_process_Ref_double_AVX(   Eigen::Ref<T> x_Ref,
                                               const std::string &fn,
                                               const bool &skip_checks);
#elif
#pragma message "Defining dummy fn_process_Ref_double_AVX - since neither AVX2 nor AVX-512 are available"
   
template <typename T>
MAYBE_INLINE  void       fn_process_Ref_double_AVX(         Eigen::Ref<T> x_Ref,
                                                            const std::string &fn,
                                                            const bool &skip_checks) {
 
 
}
#endif
 
 
 

#endif