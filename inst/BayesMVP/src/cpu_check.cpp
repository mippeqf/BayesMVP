#include <Rcpp.h>
#include <bitset>


#ifdef _WIN32
  // #include <intrin.h>
  #include <x86intrin.h>
#else
  #include <cpuid.h>
#endif 


// [[Rcpp::export]]
Rcpp::List checkCPUFeatures() {
    
      #ifdef _WIN32
              int info[4];
              int subinfo[4];
              
              // Get basic features
              __asm__ __volatile__ (
                  "cpuid"
                  : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3]) 
                : "a"(1), "c"(0)
              );
              std::bitset<32> f_ecx(info[2]); 
              bool has_avx = f_ecx[28];
              bool has_fma = f_ecx[12];  // FMA is bit 12 in ECX
              
              // Get extended features
              __asm__ __volatile__ (
                  "cpuid"
                  : "=a"(subinfo[0]), "=b"(subinfo[1]), "=c"(subinfo[2]), "=d"(subinfo[3])
                : "a"(7), "c"(0) 
              );
              std::bitset<32> f_ebx(subinfo[1]);
              bool has_avx2 = f_ebx[5];
              bool has_avx512 = f_ebx[16]; // only detects avx512f ??? check
      #else 
              unsigned int eax, ebx, ecx, edx;
              bool has_avx = false;
              bool has_avx2 = false;
              bool has_avx512 = false;
              bool has_fma = false;
              
              if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                has_avx = (ecx & bit_AVX) != 0;
                has_fma = (ecx & (1 << 12)) != 0;  // FMA is bit 12
              } 
              
              if (__get_cpuid_max(0, &eax) >= 7) {
                __cpuid_count(7, 0, eax, ebx, ecx, edx);
                has_avx2 = (ebx & bit_AVX2) != 0;
                has_avx512 = (ebx & bit_AVX512F) != 0;
              }
      #endif
          
          int has_AVX_int =     (has_avx == true)     ? 1 : 0;
          int has_AVX2_int =    (has_avx2 == true)    ? 1 : 0;
          int has_AVX_512_int = (has_avx512 == true)  ? 1 : 0;
          int has_FMA_int =     (has_fma == true)     ? 1 : 0;
          
          return Rcpp::List::create(
            Rcpp::_["has_avx"] = has_AVX_int,
            Rcpp::_["has_avx2"] = has_AVX2_int,
            Rcpp::_["has_avx512"] = has_AVX_512_int,
            Rcpp::_["has_fma"] = has_FMA_int
          );
    
}
