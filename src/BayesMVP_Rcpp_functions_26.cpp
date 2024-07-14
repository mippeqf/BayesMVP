
 
// #include <random>
// #include <xoshiro.h>
// #include <dqrng_distribution.h>
// #include <dqrng_generator.h>

// [[Rcpp::depends(RcppProgress)]]
// [[Rcpp::depends(RcppZiggurat)]]

// [[Rcpp::depends(StanHeaders)]]  
// [[Rcpp::depends(BH)]] 
// [[Rcpp::depends(RcppParallel)]] 
// [[Rcpp::depends(RcppEigen)]]  

///////// #include <omp.h>  ///////////// [[Rcpp::plugins(openmp)]]

#include <progress.hpp>
#include <progress_bar.hpp>
#include <R_ext/Print.h>

#include <stan/math/rev.hpp>
// #include <stan/math/fwd.hpp>
// #include <stan/math/mix.hpp> // then stuff from mix/ must come next
//////#include <stan/math.hpp> 

 
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/prim/err/check_pos_definite.hpp>
#include <stan/math/prim/err/check_square.hpp>
#include <stan/math/prim/err/check_symmetric.hpp>
 
 
#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/sqrt.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/transpose.hpp>
#include <stan/math/prim/fun/dot_product.hpp>
#include <stan/math/prim/fun/norm2.hpp>
#include <stan/math/prim/fun/diagonal.hpp>
#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/eigenvalues_sym.hpp>
#include <stan/math/prim/fun/diag_post_multiply.hpp>
 
 
 
 
#include <stan/math/prim/prob/multi_normal_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/lkj_corr_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/weibull_lpdf.hpp>
#include <stan/math/prim/prob/gamma_lpdf.hpp>
#include <stan/math/prim/prob/beta_lpdf.hpp>
 
 
 
 
 

#include <RcppEigen.h> 
#include <Rcpp.h> 
 

#include <Ziggurat.h>

 // #include <random>
//#include <pcg_random.hpp>
// #include <dqrng_sample.h>
 
 
#include <unsupported/Eigen/SpecialFunctions>
 

 
#define EIGEN_USE_MKL_ALL
#include "Eigen/src/Core/util/MKL_support.h"



// [[Rcpp::plugins(cpp17)]]      

using namespace Rcpp;    
using namespace Eigen;      


static Ziggurat::Ziggurat::Ziggurat zigg;

#define VECTORISATION_VEC_WIDTH 16;






// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>  draw_mean_zero_norm_Rcpp( Eigen::Matrix<double, -1, 1>  draws_vec,
                                                        Eigen::Matrix<double, -1, 1>  SD_vec) {
  
  for (int d = 0; d < draws_vec.rows(); d++) {
    draws_vec(d) = R::rnorm(0, 1);
  }
  
  draws_vec.array() = draws_vec.array() * SD_vec.array() ;
  
  return draws_vec;
  
}







// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>  draw_mean_zero_norm_using_Zigg_Rcpp(  Eigen::Matrix<double, -1, 1>  draws_vec,
                                                                    Eigen::Matrix<double, -1, 1>  SD_vec) {
  
  for (int d = 0; d < draws_vec.rows(); d++) {
    draws_vec(d) =  zigg.norm() ;
  }
  
  draws_vec.array() = draws_vec.array() * SD_vec.array() ;
   
  return draws_vec;
  
}









// [[Rcpp::export]]
double   Rcpp_det( Eigen::Matrix<double, -1, -1  >  mat) {
  
  return(   mat.determinant()   ) ;
  
}




// [[Rcpp::export]]
double   Rcpp_log_det( Eigen::Matrix<double, -1, -1  >  mat) {
  
  return(  log( std::abs( mat.determinant())  )  ) ;
  
}





// [[Rcpp::export]]
Eigen::Matrix<double, -1, -1  >    Rcpp_solve( Eigen::Matrix<double, -1, -1  >  mat) {
  
  return(mat.inverse());
  
}




// [[Rcpp::export]]
Eigen::Matrix<double, -1, -1  >    Rcpp_Chol( Eigen::Matrix<double, -1, -1  >  mat) {
  
  return( mat.llt().matrixL()  );
  
}








double __int_as_double (int64_t a) { double r; memcpy (&r, &a, sizeof r); return r;}
int64_t __double_as_int (double a) { int64_t r; memcpy (&r, &a, sizeof r); return r;}


float __int_as_float (int32_t a) { float r; memcpy (&r, &a, sizeof r); return r;}
int32_t __float_as_int (float a) { int32_t r; memcpy (&r, &a, sizeof r); return r;}





// see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
/* For a in [0.5, 4), compute a * 2**i, -250 < i < 250 */
inline double fast_ldexp (double a, int i)
{ 
  int64_t ia = ( (uint64_t)i << 52) + __double_as_int (a); // scale by 2**i
  a = __int_as_double (ia);
  if ((unsigned int)(i + 1021ULL) > 500) { // |i| > 125
    i = (i ^ (1021ULL  << 52)) - i; // ((i < 0) ? -125 : 125) << 52
    a = __int_as_double (ia - i); // scale by 2**(+/-125)
    a = a * __int_as_double ((1023ULL << 52) + i); // scale by 2**(+/-(i%125))
  }
  return a;
}








// see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
// Note: this function is  the same as the one in the link above, but for * double * instead of * float *. 
inline double fast_exp_double_wo_checks(double a)
{  
  
  a  =  1.442695040888963387 * a;
  const double cvt = 8106479329266893.0 ; //  ldexp(1.8, 52) ; //  12582912.0; // 0x1.8p23
  double f, r;
  int i;
  
  // exp2(a) = exp2(i + f); i = rint (a)
  r = (a + cvt) - cvt;
  f = a - r;
  i = (int)r;
  // approximate exp2(f) on interval [-0.5,+0.5]
  r =            0.000153720378875732421875;  // 0x1.426000p-13f
  r = fma (r, f, 0.00133903871756047010422); // 0x1.5f055ep-10f
  r = fma (r, f, 0.00961817800998687744141); // 0x1.3b2b20p-07f 
  r = fma (r, f, 0.0555036030709743499756); // 0x1.c6af7ep-05f
  r = fma (r, f, 0.240226522088050842285); // 0x1.ebfbe2p-03f
  r = fma (r, f, 0.693147182464599609375); // 0x1.62e430p-01f
  r = fma (r, f, 1.0); // 0x1.000000p+00f
  // exp2(a) = 2**i * exp2(f);
  
  
  //  double fast_ldexp (double a, int i)
  // { 
  int64_t ia = ( (uint64_t)i << 52) + __double_as_int (r); // scale by 2**i
  r = __int_as_double (ia);
  
  //  return __int_as_double(   ( (uint64_t)i << 52) + __double_as_int (r)    ) ; 
  if ((unsigned int)(i + 1021ULL) > 500) { // |i| > 125
    i = (i ^ (1021ULL << 52)) - i; // ((i < 0) ? -125 : 125) << 52
    r = __int_as_double (ia - i); // scale by 2**(+/-125)
    r = r * __int_as_double ((1023ULL << 52) + i); // scale by 2**(+/-(i%125))
  }
  
  // } c
  
  // r = fast_ldexp (r, i);
  return r;
  
}



// Note: Compiler needs  to auto-vectorise for the following to be fast
// [[Rcpp::export]]
Eigen::Array<double, -1, 1  > fast_exp_double_wo_checks_Eigen(  Eigen::Array<double, -1, 1  > x) {
  
     // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
  for (int i = 0; i < x.rows(); ++i) {
    x(i) = fast_exp_double_wo_checks(x(i));
  }    
  
  return x; 
  
}





// Note: Compiler needs  to auto-vectorise for the following to be fast
// [[Rcpp::export]]
Eigen::Array<double, -1, -1  > fast_exp_double_wo_checks_Eigen_mat(  Eigen::Array<double, -1, -1  > x) {
  
  for (int j = 0; j < x.cols(); ++j) {
      // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
    for (int i = 0; i < x.rows(); ++i) {
      x(i, j) = fast_exp_double_wo_checks(x(i, j));
    }
  }
  
  return x; 
  
}


// From:  https://stackoverflow.com/questions/65554112/fast-double-exp2-function-in-c/65562273#65562273s
/* compute 2**p, for p in [-1022, 1024). Maximum relative error: 4.93e-5. RMS error: 9.91e-6 */  
// Note:  modified -  uses some FMA operations 
// [[Rcpp::export]]
double fast_exp_approx_double_wo_checks (double p)
{
  
  double res;
  
  p  =  1.442695040888963387 * p;
  p = (p < -1022) ? -1022 : p; // clamp below
  
  /* 2**p = 2**(w+z), with w an integer and z in [0, 1) */
  double w = floor(p); // integral part
  double z = p - w;     // fractional part
  
  double c3_recip = 1.0 /  ( 4.84257784485816955566 - z);
  double approx;
  approx   = fma(27.7283337116241455078, c3_recip, -5.72594201564788818359);
  approx   = fma(-0.49013227410614490509, z, approx);
  
  int64_t resi = ((1LL << 52) * (w + 1023 + approx));   /* assemble the exponent and mantissa components into final result */
  
  memcpy (&res, &resi, sizeof res);
  return res;
}



// Note: Compiler needs  to auto-vectorise for the following to be fast
// [[Rcpp::export]]
Eigen::Array<double, -1, 1  > fast_exp_approx_double_wo_checks_Eigen(  Eigen::Array<double, -1, 1  > x)   {  
    // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
  for (int i = 0; i < x.rows(); ++i) {
    x(i) =  fast_exp_approx_double_wo_checks(x(i));
  }    
  return x; 
}   

// Note: Compiler needs  to auto-vectorise for the following to be fast
// [[Rcpp::export]]
Eigen::Array<double, -1, -1  >  fast_exp_approx_double_wo_checks_Eigen_mat(  Eigen::Array<double, -1, -1  > x)
{    
  
 
  for (int j = 0; j < x.cols(); ++j) {
      // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
    for (int i = 0; i < x.rows(); ++i) {
      x(i, j) =  fast_exp_approx_double_wo_checks(x(i, j));
    }  
  }
  
  return x; 
  
}    






/* natural log on [0x1.f7a5ecp-127, 0x1.fffffep127]. Maximum relative error 9.4529e-5 */
// see: https://stackoverflow.com/questions/39821367/very-fast-approximate-logarithm-natural-log-function-in-c
// Note: this function is  the same as the one in the link above, but for * double * instead of * float *.  
inline double fast_log_approx_double_wo_checks(double a)
{ 
  double m, r, s, t, i, f;
  int64_t e;
  
  e = (__double_as_int (a) - 0x3fe5555555555555 )    &   0xFFF0000000000000   ;
  m = __int_as_double (__double_as_int (a) - e);
  //  i = double(e) * (double)1.19209290e-7; // 0x1.0p-23
  i = (double)e *   0.000000000000000222044604925031308085 ; // ldexp(1.0, -52) ; 
  //  return(i); 
  
  /* m in [2/3, 4/3] */
  f = m - 1.0;
  s = f * f;
  /* Compute log1p(f) for f in [-1/3, 1/3] */
  r = fma (0.230836749076843261719, f, -0.279208570718765258789); // 0x1.d8c0f0p-3, -0x1.1de8dap-2
  t = fma (0.331826031208038330078, f, -0.498910337686538696289); // 0x1.53ca34p-2, -0x1.fee25ap-2
  r = fma (r, s, t);
  r = fma (r, s, f);
  r = fma (i, 0.693147182464599609375, r); // 0x1.62e430p-1 // log(2)
  return r; 
}  



// Note: Compiler needs  to auto-vectorise for the following to be fast
// [[Rcpp::export]]
Eigen::Array<double, -1, 1  > fast_log_approx_double_wo_checks_Eigen(  Eigen::Array<double, -1, 1  > x)
{  
  
    // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
  for (int i = 0; i < x.rows(); ++i) {
    x(i) =  fast_log_approx_double_wo_checks(x(i));
  }   
  
  return x; 
  
}   

// Note: Compiler needs  to auto-vectorise for the following to be fast
// [[Rcpp::export]]
Eigen::Array<double, -1, -1  >  fast_log_approx_double_wo_checks_Eigen_mat(  Eigen::Array<double, -1, -1  > x)
{   
  
  for (int j = 0; j < x.cols(); ++j) {
      // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH) 
    for (int i = 0; i < x.rows(); ++i) {
      x(i, j) =  fast_log_approx_double_wo_checks(x(i, j));
    }  
  }
  
  return x; 
  
}    
















// [[Rcpp::export]]
double  fn_ld_exp_bitshift_1(int b)  { 
  
  return ldexp(1.0, b); 
  
}



// [[Rcpp::export]]
double  fn_ld_exp_bitshift_2(double a, int b)  { 
  
  return ldexp(a, b); 
  
}









// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>    fast_tanh_approx_Eigen(   Eigen::Matrix<double, -1, 1 >  x  )   { 
  
  return   ( ( 2.0 / (1.0 + fast_exp_double_wo_checks_Eigen(-2.0*x.array()).array() ).array() ) - 1.0 ).matrix() ; 
  
} 









// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>    Phi_using_erfc_Eigen(   Eigen::Matrix<double, -1, 1 >  x  )   { 
  
  double minus_sqrt_2_recip =  - 1 / stan::math::sqrt(2);
  return 0.5 *  ( minus_sqrt_2_recip * x ).array().erfc().matrix() ; 
  
}






// [[Rcpp::export]]
Eigen::Array<double, -1, 1>    inv_logit_1_Eigen(   Eigen::Array<double, -1, 1 >  x  )   { 
  
  int N = x.rows(); 
  
  for (int i = 0; i < N; i++) {
    x(i) =  ( ( 1.0 / (1.0 + exp(- x(i))  ) )  ) ; 
  }
  
  return x; 
  
}









// [[Rcpp::export]]
Eigen::Array<double, -1, 1>    Phi_approx_Eigen(   Eigen::Array<double, -1, 1 >  x  )   { 
  
  
  int N = x.rows(); 
  
  for (int i = 0; i < N; i++) {
    double x_i =  x(i);
    x(i) =  (1.0/(1.0 +   exp(-(0.07056*x_i*x_i*x_i + 1.5976*x_i )) )) ; 
  }
  
  return x; 
  
}




// [[Rcpp::export]]
Eigen::Array<double, -1, 1>    fast_Phi_approx_1_Eigen(   Eigen::Array<double, -1, 1 >  x  )   { 
  
  
  int N = x.rows(); 
  
  for (int i = 0; i < N; i++) {
    double x_i =  x(i);
    x(i) =  (1.0/(1.0 +   fast_exp_double_wo_checks(-(0.07056*x_i*x_i*x_i + 1.5976*x_i )) )) ;
  }
  
  return x; 
  
}





// source: same as implemented in R and adapted from Stan code here: https://github.com/stan-dev/math/issues/2555 (Stan code written by Sean Spinkney)
inline double  qnorm_rcpp(double p) {
  
  double r; 
  double val;
  double q = p - 0.5;
  
  if (stan::math::abs(q) <= .425) {
    r = .180625 - q * q;
    val = q * (((((((r * 2509.0809287301226727 +
      33430.575583588128105) * r + 67265.770927008700853) * r +
      45921.953931549871457) * r + 13731.693765509461125) * r +
      1971.5909503065514427) * r + 133.14166789178437745) * r +
      3.387132872796366608) / (((((((r * 5226.495278852854561 +
      28729.085735721942674) * r + 39307.89580009271061) * r +
      21213.794301586595867) * r + 5394.1960214247511077) * r +
      687.1870074920579083) * r + 42.313330701600911252) * r + 1.0);
  } else { /* closer than 0.075 from {0,1} boundary */
  if (q > 0) r = 1.0 - p;
  else r = p;
  
  r = std::sqrt(-std::log(r));
  
  if (r <= 5.) { /* <==> min(p,1-p) >= exp(-25) ~= 1.3888e-11 */
    r += -1.6;
    val = (((((((r * 0.00077454501427834140764 +
                      .0227238449892691845833) * r + .24178072517745061177) *
    r + 1.27045825245236838258) * r +
    3.64784832476320460504) * r + 5.7694972214606914055) * r + 4.6303378461565452959) * r +
    1.42343711074968357734) / (((((((r *
    0.00000000105075007164441684324 + 0.0005475938084995344946) *
    r + .0151986665636164571966) * r +
        .14810397642748007459) * r + .68976733498510000455) *
    r + 1.6763848301838038494) * r +
    2.05319162663775882187) * r + 1.);
  } else { /* very close to  0 or 1 */
    r += -5.;
    val = (((((((r * 0.000000201033439929228813265 +
      0.0000271155556874348757815) * r +
       .0012426609473880784386) * r + .026532189526576123093) *
      r + .29656057182850489123) * r +
      1.7848265399172913358) * r + 5.4637849111641143699) *
      r + 6.6579046435011037772) / (((((((r *
      0.00000000000000204426310338993978564 + 0.00000014215117583164458887)*
      r + 0.000018463183175100546818) * r +
      0.0007868691311456132591) * r + .0148753612908506148525)
                                        * r + .13692988092273580531) * r +
                                              .59983220655588793769) * r + 1.);
  }
  
  if (q < 0.0) val = -val;
  }
  return val;
  
}




// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>  qnorm_rcpp_Eigen( Eigen::Matrix<double, -1, 1>  p) {
  
  int N = stan::math::num_elements(p);
  
  for (int n = 0; n < N; n++) {
    p(n) = qnorm_rcpp(p(n));
  }
  
  return p;
}





// source: same as implemented in R and adapted from Stan code here: https://github.com/stan-dev/math/issues/2555 (Stan code written by Sean Spinkney)
inline double  qnorm_w_fast_log_rcpp(double p) {
  
  double r;
  double val;
  double q = p - 0.5;
  
  if (std::abs(q) <= 0.425) {
    r = .180625 - q * q;
    val = q * (((((((r * 2509.0809287301226727 +
      33430.575583588128105) * r + 67265.770927008700853) * r +
      45921.953931549871457) * r + 13731.693765509461125) * r +
      1971.5909503065514427) * r + 133.14166789178437745) * r +
      3.387132872796366608) / (((((((r * 5226.495278852854561 +
      28729.085735721942674) * r + 39307.89580009271061) * r +
      21213.794301586595867) * r + 5394.1960214247511077) * r +
      687.1870074920579083) * r + 42.313330701600911252) * r + 1.0);
  } else { /* closer than 0.075 from {0,1} boundary */
    if (q > 0) r = 1.0 - p;
    else r = p;
    
    r = std::sqrt(-fast_log_approx_double_wo_checks(r));
    
    if (r <= 5.0) { /* <==> min(p,1-p) >= exp(-25) ~= 1.3888e-11 */
      r += -1.6;
      val = (((((((r * 0.00077454501427834140764 +
                        .0227238449892691845833) * r + .24178072517745061177) *
      r + 1.27045825245236838258) * r +
      3.64784832476320460504) * r + 5.7694972214606914055) * r + 4.6303378461565452959) * r +
      1.42343711074968357734) / (((((((r *
      0.00000000105075007164441684324 + 0.0005475938084995344946) *
      r + .0151986665636164571966) * r +
          .14810397642748007459) * r + .68976733498510000455) *
      r + 1.6763848301838038494) * r +
      2.05319162663775882187) * r + 1.);
    } else { /* very close to  0 or 1 */
      r += -5.0;
      val = (((((((r * 0.000000201033439929228813265 +
        0.0000271155556874348757815) * r +
         .0012426609473880784386) * r + .026532189526576123093) *
        r + .29656057182850489123) * r +
        1.7848265399172913358) * r + 5.4637849111641143699) *
        r + 6.6579046435011037772) / (((((((r *
        0.00000000000000204426310338993978564 + 0.00000014215117583164458887)*
        r + 0.000018463183175100546818) * r +
        0.0007868691311456132591) * r + .0148753612908506148525)
                                          * r + .13692988092273580531) * r +
                                                .59983220655588793769) * r + 1.);
    }
    
    if (q < 0.0) val = -val;
  }
  return val;
  
}




// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>  qnorm_w_fast_log_rcpp_Eigen( Eigen::Matrix<double, -1, 1>  p) {
  
  int N = p.rows() ; 
  
  for (int n = 0; n < N; n++) {
    p(n) = qnorm_w_fast_log_rcpp(p(n));
  }
  
  return p;
}






inline double  Phi_approx_fast( double x )  { 
  // fast_exp_approx_double_wo_checks
  // fast_exp_double_wo_checks
  return   1.0 / (1.0 +   fast_exp_double_wo_checks(-fma(fma(0.07056*x, x, 1.5976),  x, 0.0)) ) ;
}  


// [[Rcpp::export]]
Eigen::Array<double, -1, 1  > Phi_approx_fast_Eigen(  Eigen::Array<double, -1, 1   > x)
{    
  
    // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH) 
  for (int i = 0; i < x.rows(); ++i) { 
    x(i) =  Phi_approx_fast(x(i));
  }       
  
  return  x; 
  
}   




 


inline double  inv_Phi_approx_fast( const double x )  { 
  const double log_stuff = fast_log_approx_double_wo_checks( 1.0/x  - 1.0); // log first 
  const double x_i = -0.3418*log_stuff;
  const double asinh_stuff_div_3 =  0.33333333333333331483 * fast_log_approx_double_wo_checks( x_i  +  std::sqrt(  fma(x_i, x_i, 1.0) ) ) ;          // now do arc_sinh part
  // const double asinh_stuff_div_3 =  0.33333333333333331483 * fast_log_approx_double_wo_checks( x_i  +  std::sqrt(1.0 + (x_i*x_i))) ; 
  const double exp_x_i = fast_exp_double_wo_checks(asinh_stuff_div_3);
  return  2.74699999999999988631 * ( fma(exp_x_i, exp_x_i , -1.0) / exp_x_i ) ;  //   now do sinh parth part
}     


// [[Rcpp::export]] 
Eigen::Array<double, -1, 1  > inv_Phi_approx_fast_Eigen(  Eigen::Array<double, -1, 1   > x)  {       
    // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)  
  for (int i = 0; i < x.rows(); ++i) {  
    x(i) =  inv_Phi_approx_fast(x(i));
  }            
  return  x;  
}     


 








 


// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>    exp_stan(   Eigen::Matrix<double, -1, 1 >  x  )   { 
  
  return stan::math::exp(x);
  
}

// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>    log_stan(   Eigen::Matrix<double, -1, 1 >  x  )   { 
  
  return stan::math::log(x);
  
}



// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>    inv_logit_stan(   Eigen::Matrix<double, -1, 1 >  x  )   { 
  
  return stan::math::inv_logit(x);
  
}





// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>    tanh_stan(   Eigen::Matrix<double, -1, 1 >  x  )   { 
  
  return stan::math::tanh(x);
  
}




// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>    erfc_stan(   Eigen::Matrix<double, -1, 1 >  x  )   { 
  
  return stan::math::erfc(x);
  
}



// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>    Phi_stan(   Eigen::Matrix<double, -1, 1 >  x  )   { 
  
  return stan::math::Phi(x);
  
}


// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>    Phi_using_erfc_stan(   Eigen::Matrix<double, -1, 1 >  x  )   { 
  
  double minus_sqrt_2_recip =  - 1 / stan::math::sqrt(2);
  return 0.5 *  stan::math::erfc( minus_sqrt_2_recip * x ) ; 
  
}


 

// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>    inv_Phi_stan(   Eigen::Matrix<double, -1, 1 >  x  )   { 
  
  return stan::math::inv_Phi(x);
  
}



 

 
 // [[Rcpp::export]]
 Eigen::Matrix<double, -1, 1 >    Rcpp_mult_mat_by_col_vec( Eigen::Matrix<double, -1, -1  >  mat,
                                                            Eigen::Matrix<double, -1, 1 >  colvec) {
   
   return(  mat * colvec  );
   
 }
 
 
 // [[Rcpp::export]]
 Eigen::Matrix<float, -1, 1 >    Rcpp_mult_mat_by_col_vec_float( Eigen::Matrix<float, -1, -1  >  mat,
                                                                  Eigen::Matrix<float, -1, 1 >  colvec) {
   
   return(  mat * colvec  );
   
 }
 
 
 // [[Rcpp::export]]
 Eigen::Matrix<double, -1, -1  >    Rcpp_mult_mat_by_mat(    Eigen::Matrix<double, -1, -1  >  mat_1,
                                                             Eigen::Matrix<double, -1, -1  >  mat_2) {
   
   return(  mat_1 * mat_2  );
   
 }
 
 

 
 
 
 


 // function for use in the log-posterior function (i.e. the function to calculate gradients for)
 Eigen::Matrix<stan::math::var, -1, -1>	  fn_calculate_cutpoints_AD(
     Eigen::Matrix<stan::math::var, -1, 1> log_diffs, //this is aparameter (col vec)
     stan::math::var first_cutpoint, // this is constant
     int K) {

   Eigen::Matrix<stan::math::var, -1, -1> cutpoints_set_full(K+1, 1);

   cutpoints_set_full(0,0) = -1000;
   cutpoints_set_full(1,0) = first_cutpoint;
   cutpoints_set_full(K,0) = +1000;

   for (int k=2; k < K; ++k)
     cutpoints_set_full(k,0) =     cutpoints_set_full(k-1,0)  + (exp(log_diffs(k-2))) ;

   return cutpoints_set_full; // output is a parameter to use in the log-posterior function to be differentiated
 }





 // NEED TO MAKE INTO A TEMPLATE!!!
 // function for use in the log-posterior function (i.e. the function to calculate gradients for)
 // [[Rcpp::export]]
 Eigen::Matrix<double, -1, -1>	  fn_calculate_cutpoints(
     Eigen::Matrix<double, -1, 1> log_diffs, //this is a parameter (col vec)
     double first_cutpoint, // this is constant
     int K) {

   Eigen::Matrix<double, -1, -1> cutpoints_set_full(K+1, 1);

   cutpoints_set_full(0,0) = -1000;
   cutpoints_set_full(1,0) = first_cutpoint;
   cutpoints_set_full(K,0) = +1000;

   for (int k=2; k < K; ++k)
     cutpoints_set_full(k,0) =     cutpoints_set_full(k-1,0)  + (exp(log_diffs(k-2))) ;

   return cutpoints_set_full; // output is a parameter to use in the log-posterior function to be differentiated
 }




 
 
 
 // [[Rcpp::export]]
 std::vector<Eigen::Matrix<double, -1, -1 > > vec_of_mats_test(int n_rows,
                                                               int n_cols,
                                                               int n_mats) {
   
   
   std::vector<Eigen::Matrix<double, -1, -1 > > my_vec(n_mats);
   Eigen::Matrix<double, -1, -1 > mats  =   Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols);
   
   for (int c = 0; c < n_mats; ++c) {
     my_vec[c] = mats;
   }
   
   return(my_vec); 
   
 }
 
 
 // [[Rcpp::export]]
 std::vector<Eigen::Matrix<int, -1, -1 > > vec_of_mats_test_int(int n_rows,
                                                                int n_cols,
                                                                int n_mats) {
   
   
   std::vector<Eigen::Matrix<int, -1, -1 > > my_vec(n_mats);
   Eigen::Matrix<int, -1, -1 > mats  =   Eigen::Matrix<int, -1, -1>::Zero(n_rows, n_cols);
   
   for (int c = 0; c < n_mats; ++c) {
     my_vec[c] = mats;
   }
   
   return(my_vec); 
   
 }
 
 
 
 
 // [[Rcpp::export]]
 std::vector<Eigen::Matrix<bool, -1, -1 > > vec_of_mats_test_bool(int n_rows,
                                                                  int n_cols,
                                                                  int n_mats) {
   
   
   std::vector<Eigen::Matrix<bool, -1, -1 > > my_vec(n_mats);
   Eigen::Matrix<bool, -1, -1 > mats(n_rows, n_cols);
   
   for (int c = 0; c < n_mats; ++c) {
     my_vec[c] = mats;
   }
   
   return(my_vec); 
   
 }
 
 
 
 
 
 // [[Rcpp::export]]
 std::vector<Eigen::Matrix<int, -1, -1 > > vec_of_mats_test_int_Ones( int n_rows,
                                                                      int n_cols,
                                                                      int n_mats) {
   
   std::vector<Eigen::Matrix<int, -1, -1 > > my_vec(n_mats);
   Eigen::Matrix<int, -1, -1 > mats  =   Eigen::Matrix<int, -1, -1>::Ones(n_rows, n_cols);
   
   for (int c = 0; c < n_mats; ++c) {
     my_vec[c] = mats;
   }
   
   return(my_vec); 
   
 }
 
 
 
 
 
 // [[Rcpp::export]]
 std::vector<Eigen::Matrix<double, -1, -1  , Eigen::RowMajor > > vec_of_mats_test_RM(int n_rows,
                                                                                     int n_cols,
                                                                                     int n_mats) {
   
   std::vector<Eigen::Matrix<double, -1, -1 , Eigen::RowMajor > > my_vec(n_mats);
   Eigen::Matrix<double, -1, -1, Eigen::RowMajor> mats  =   Eigen::Matrix<double, -1, -1, Eigen::RowMajor>::Zero(n_rows, n_cols);
   
   for (int c = 0; c < n_mats; ++c) {
     my_vec[c] = mats;
   }
   
   return(my_vec);
   
 }
 
 
 
 
 
 
 
 std::vector<Eigen::Matrix<float, -1, -1 > > vec_of_mats_test_float(int n_rows,
                                                                    int n_cols,
                                                                    int n_mats) {
   
   std::vector<Eigen::Matrix<float, -1, -1 > > my_vec(n_mats);
   Eigen::Matrix<float, -1, -1 > mats  =   Eigen::Matrix<float, -1, -1>::Zero(n_rows, n_cols);
   
   for (int c = 0; c < n_mats; ++c) {
     my_vec[c] = mats;
   }
   
   return(my_vec); 
   
   
 }
 
 
 
 std::vector<Eigen::Matrix<float, -1, -1, Eigen::RowMajor > > vec_of_mats_test_float_RM(int n_rows,
                                                                                        int n_cols,
                                                                                        int n_mats) {
   
   std::vector<Eigen::Matrix<float, -1, -1 , Eigen::RowMajor > > my_vec(n_mats);
   Eigen::Matrix<float, -1, -1, Eigen::RowMajor > mats  =   Eigen::Matrix<float, -1, -1 , Eigen::RowMajor>::Zero(n_rows, n_cols);
   
   for (int c = 0; c < n_mats; ++c) {
     my_vec[c] = mats;
   }
   
   return(my_vec); 
   
   
 }
 
 
 
 
 
 
 
 
 std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > vec_of_mats_test_var(int n_rows,
                                                                            int n_cols,
                                                                            int n_mats) {
   
   std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > my_vec(n_mats);
   Eigen::Matrix<stan::math::var, -1, -1 > mats  =   Eigen::Matrix<stan::math::var, -1, -1>::Zero(n_rows, n_cols);
   
   for (int c = 0; c < n_mats; ++c) {
     my_vec[c] = mats;
   }
   
   return(my_vec); 
   
 }
 
 
 std::vector<Eigen::Matrix<stan::math::var, -1, -1 , Eigen::RowMajor > > vec_of_mats_test_var_RM(int n_rows,
                                                                                                 int n_cols,
                                                                                                 int n_mats) {
   
   
   std::vector<Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor > > my_vec(n_mats);
   Eigen::Matrix<stan::math::var, -1, -1,  Eigen::RowMajor> mats  =   Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor>::Zero(n_rows, n_cols);
   
   for (int c = 0; c < n_mats; ++c) {
     my_vec[c] = mats;
   }
   
   return(my_vec); 
   
 }
 
 
 
 
 
 
 
 
  
 
 
 
 
 // input vector, outputs upper-triangular 3d array of corrs- double
 std::vector<Eigen::Matrix<stan::math::var, -1, -1> >  fn_convert_Eigen_vec_of_corrs_to_3d_array_var(
                                                                                                                         Eigen::Matrix<stan::math::var, -1, -1  >  input_vec,
                                                                                                                         int n_rows,
                                                                                                                         int n_arrays) {
   
   std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >   output_array = vec_of_mats_test_var(n_rows, n_rows, n_arrays); // 1d vector to output
   
   int k = 0;
   for (int c = 0; c < n_arrays; ++c) {
     for (int i = 1; i < n_rows; ++i)  {
       for (int j = 0; j < i; ++j) { // equiv to 1 to K - 2 in R
         output_array[c](i,j) =  input_vec(i);
         k += 1;
       }
     }
   }
   
   return output_array; // output is a parameter to use in the log-posterior function to be differentiated
 }
 
 
 
 
 
 
 // input vector, outputs upper-triangular 3d array of corrs- double
 std::vector<Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor > >  fn_convert_Eigen_vec_of_corrs_to_3d_array_var_RM(
                                                                                                           Eigen::Matrix<stan::math::var, -1, -1  >  input_vec,
                                                                                                           int n_rows,
                                                                                                           int n_arrays) {

   std::vector<Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor > >   output_array = vec_of_mats_test_var_RM(n_rows, n_rows, n_arrays); // 1d vector to output

   int k = 0;
   for (int c = 0; c < n_arrays; ++c) {
     for (int i = 1; i < n_rows; ++i)  {
       for (int j = 0; j < i; ++j) { // equiv to 1 to K - 2 in R
         output_array[c](i,j) =  input_vec(i);
         k += 1;
       }
     }
   }

   return output_array; // output is a parameter to use in the log-posterior function to be differentiated
 }





 


 // outputs vector, input upper-triangular 3d array of corrs- double
 // [[Rcpp::export]]
 Eigen::Matrix<double, -1, 1 >     fn_convert_3d_array_of_corrs_to_Eigen_vec_RM(
                                                                     std::vector<Eigen::Matrix<double, -1, -1  > >  input_array,
                                                                     int n_rows,
                                                                     int n_arrays) {

   
    int dim = n_arrays * 0.5 * n_rows * (n_rows - 1);
    Eigen::Matrix<double, -1, 1>   output_vec(dim); // 1d vector to output
    
    std::vector<Eigen::Matrix<double, -1, -1, Eigen::RowMajor > >  input_array_RM =  vec_of_mats_test_RM(n_rows, n_rows, n_arrays)  ; // input_array;
    
    for (int c = 0; c < n_arrays; ++c) {
        input_array_RM[c] = input_array[c]; 
    }

   int k = 0;
   for (int c = 0; c < n_arrays; ++c) {
     for (int i = 1; i < n_rows; ++i)  {
       for (int j = 0; j < i; ++j) { // equiv to 1 to K - 2 in R
         output_vec(k) = input_array_RM[c](i,j);
         k += 1;
       }
     }
   }

   return output_vec; // output is a parameter to use in the log-posterior function to be differentiated
 }


 
 
 
 
 
 // outputs vector, input upper-triangular 3d array of corrs- var
 Eigen::Matrix<stan::math::var, -1, 1  >     fn_convert_3d_array_of_corrs_to_Eigen_vec_var_RM(
                                                                                             std::vector<Eigen::Matrix<stan::math::var, -1, -1  > >  input_array,
                                                                                             int n_rows,
                                                                                             int n_arrays) {

     int dim = n_arrays * 0.5 * n_rows * (n_rows - 1);
     Eigen::Matrix<stan::math::var, -1, 1  >    output_vec(dim); // 1d vector to output

     
     std::vector<Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor > >  input_array_RM =  vec_of_mats_test_var_RM(n_rows, n_rows, n_arrays)  ; // input_array;
     
     
     
     for (int c = 0; c < n_arrays; ++c) {
       input_array_RM[c] = input_array[c]; 
     }
     
     
   int k = 0;
   for (int c = 0; c < n_arrays; ++c) {
     for (int i = 1; i < n_rows; ++i)  {
       for (int j = 0; j < i; ++j) { // equiv to 1 to K - 2 in R
         output_vec(k) = input_array_RM[c](i,j);
         k += 1;
       }
     }
   }

   return output_vec; // output is a parameter to use in the log-posterior function to be differentiated
 }


 
 
 
 
 
 
 
 // convert std vec to eigen vec - var
 Eigen::Matrix<stan::math::var, -1, 1> std_vec_to_Eigen_vec_var(std::vector<stan::math::var> std_vec) {

   Eigen::Matrix<stan::math::var, -1, 1>  Eigen_vec(std_vec.size());

   for (int i = 0; i < std_vec.size(); ++i) {
     Eigen_vec(i) = std_vec[i];
   }

   return(Eigen_vec);
 }




 // convert std vec to eigen vec - double
 // [[Rcpp::export]]
 Eigen::Matrix<double, -1, 1> std_vec_to_Eigen_vec(std::vector<double> std_vec) {

   Eigen::Matrix<double, -1, 1>  Eigen_vec(std_vec.size());

   for (int i = 0; i < std_vec.size(); ++i) {
     Eigen_vec(i) = std_vec[i];
   }

   return(Eigen_vec);
 }

 // [[Rcpp::export]]
 std::vector<double> Eigen_vec_to_std_vec(Eigen::Matrix<double, -1, 1> Eigen_vec) {

   std::vector<double>  std_vec(Eigen_vec.rows());

   for (int i = 0; i < Eigen_vec.rows(); ++i) {
     std_vec[i] = Eigen_vec(i);
   }

   return(std_vec);
 }


 std::vector<stan::math::var> Eigen_vec_to_std_vec_var(Eigen::Matrix<stan::math::var, -1, 1> Eigen_vec) {

   std::vector<stan::math::var>  std_vec(Eigen_vec.rows());

   for (int i = 0; i < Eigen_vec.rows(); ++i) {
     std_vec[i] = Eigen_vec(i);
   }

   return(std_vec);
 }




 

 
 
 
 std::vector<std::vector<Eigen::Matrix<double, -1, -1 > > > vec_of_vec_of_mats_test(int n_rows,
                                                                                    int n_cols,
                                                                                    int n_mats_inner, 
                                                                                    int n_mats_outer) {
   
   /// need to figure out more efficient way to do this + make work for all types easily (not just double)
   std::vector<std::vector<Eigen::Matrix<double, -1, -1 > > > my_vec_of_vecs(n_mats_outer);
   Eigen::Matrix<double, -1, -1 > mat_sizes(n_rows, n_cols);
   
   
   
   for (int c1 = 0; c1 < n_mats_outer; ++c1) {
     std::vector<Eigen::Matrix<double, -1, -1 > > my_vec(n_mats_inner);
     my_vec_of_vecs[c1] = my_vec; 
     for (int c2 = 0; c2 < n_mats_inner; ++c2) {
       my_vec_of_vecs[c1][c2] = mat_sizes;
       for (int i = 0; i < n_rows; ++i) {
         for (int j = 0; j < n_cols; ++j) {
           my_vec_of_vecs[c1][c2](i, j) = 0;
         }
       }
     } 
   }
   
   
   return(my_vec_of_vecs);
   
 } 
 
 
  



 
 
 
 
 std::vector<std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > > vec_of_vec_of_mats_test_var(int n_rows,
                                                                                                 int n_cols,
                                                                                                 int n_mats_inner, 
                                                                                                 int n_mats_outer) {
   
   /// need to figure out more efficient way to do this + make work for all types easily (not just double)
   std::vector<std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > > my_vec_of_vecs(n_mats_outer);
   Eigen::Matrix<stan::math::var, -1, -1 > mat_sizes(n_rows, n_cols);
   
   
   
   for (int c1 = 0; c1 < n_mats_outer; ++c1) {
     std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > my_vec(n_mats_inner);
     my_vec_of_vecs[c1] = my_vec; 
     for (int c2 = 0; c2 < n_mats_inner; ++c2) {
       my_vec_of_vecs[c1][c2] = mat_sizes;
       for (int i = 0; i < n_rows; ++i) {
         for (int j = 0; j < n_cols; ++j) {
           my_vec_of_vecs[c1][c2](i,j) = 0;
         }
       }
     }
   }
   
   
   return(my_vec_of_vecs);
   
 } 
 
 
 
 
 
 
 
 
 // input vector, outputs upper-triangular 3d array of corrs- double
 std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >   fn_convert_std_vec_of_corrs_to_3d_array_var( 
     std::vector<stan::math::var>   input_vec, 
     int n_rows,
     int n_arrays) {
   
   std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >   output_array = vec_of_mats_test_var(n_rows, n_rows, n_arrays); // 1d vector to output
   
   int k = 0;
   for (int c = 0; c < n_arrays; ++c) {
     for (int i = 1; i < n_rows; ++i)  {
       for (int j = 0; j < i; ++j) { // equiv to 1 to K - 2 in R
         output_array[c](i,j) =  input_vec[k];
         k = k + 1; 
       }
     }
   }
   
   return output_array; // output is a parameter to use in the log-posterior function to be differentiated 
 }
 
 
 
 
 
 
 
 
 
  
 
 
 
 // [[Rcpp::export]]
 std::vector<Eigen::Matrix<double, -1, -1, Eigen::RowMajor > >    fn_convert_unconstrained_to_corrs_double_RM(Eigen::Matrix<double, -1, -1  > Omega_unconstrained
 ) {


   int dim = Omega_unconstrained.rows();

   ///////////  (1) define lower-triangular matrix Z (tanh transform of Omega_unconstrained)
   Eigen::Matrix<double, -1, -1, Eigen::RowMajor > Omega_Z = Eigen::Matrix<double, -1, -1, Eigen::RowMajor >::Zero(dim,  dim);


   // lower triangle of tanh transforms (j < i)
   for (int i = 1; i < dim; ++i)  {
     for (int j = 0; j < i; ++j) { // equiv to 1 to K - 2 in R
       Omega_Z(i,j) = stan::math::tanh(Omega_unconstrained(i,j)) ; // (exp(2 * Omega_unconstrained(i,j)) - 1 ) /   (exp(2 * Omega_unconstrained(i,j)) + 1 );
     }
   }

 
   
   // convert to a vector
   std::vector<Eigen::Matrix<double, -1, -1  > > Omega_Z_array_CM = vec_of_mats_test(dim, dim, 1);
   Omega_Z_array_CM[0] = Omega_Z;

   Eigen::Matrix<double, -1, 1> Omega_Z_vec =  fn_convert_3d_array_of_corrs_to_Eigen_vec_RM(Omega_Z_array_CM,
                                                                                              dim,
                                                                                              1);

   ///////////// (2) then map Omega_Z -> LOWER - triangular cholesky factor corr's
   Eigen::Matrix<double, -1, -1  > L_Omega_lower = Eigen::Matrix<double, -1, -1, Eigen::RowMajor >::Zero(dim,  dim);


   int counter = 0;

   L_Omega_lower(0,0) = 1;  // 1 if i = j = 1

   double term_2 = 0.0;


   Eigen::Matrix<double, -1, -1, Eigen::RowMajor  > sqrt_term = Eigen::Matrix<double, -1, -1, Eigen::RowMajor >::Zero(dim,  dim);
   Eigen::Matrix<double, -1, -1, Eigen::RowMajor  > term_inv = Eigen::Matrix<double, -1, -1, Eigen::RowMajor >::Zero(dim,  dim);


   for (int i = 1; i < dim; ++i) {

     //  L_Omega_lower(i, 0) = Omega_Z_vec(counter);
     L_Omega_lower(i, 0) = Omega_Z(i, 0);
     counter = counter + 1;

     double temp_sum_square = L_Omega_lower(i, 0) * L_Omega_lower(i, 0);

     for (int j = 1; j < i + 1; ++j) {

       if (j == i) { continue; }


       //
       if (j != i) {

         term_2 += 0.5 * stan::math::log1m(temp_sum_square);

         L_Omega_lower(i, j) =  Omega_Z(i, j) * stan::math::sqrt(1.0 - temp_sum_square);

         sqrt_term(i, j) = stan::math::sqrt(1.0 - temp_sum_square);
         term_inv(i, j) = (1.0 /  sqrt_term(i, j)) * (1.0 /  sqrt_term(i, j));

         temp_sum_square += L_Omega_lower(i, j) * L_Omega_lower(i, j);


       }

     }
     L_Omega_lower(i, i) =  stan::math::sqrt(1.0 - temp_sum_square);

   }


   //////////// (3) calculate correlation matrix
   Eigen::Matrix<double, -1, -1, Eigen::RowMajor  > Omega = L_Omega_lower * L_Omega_lower.transpose();

   for (int i = 0; i < dim; ++i) {
     Omega(i,i) = 1.0;
   }

   //////////////////// calculate Jacobian adjustment
   // make vector of unconstrained params
   Eigen::Matrix<double, -1, 1  > Omega_y_vec =   Omega_Z_vec;
   Eigen::Matrix<double, -1, 1  > jac_vec =   Omega_Z_vec;

   for (int i = 0; i < Omega_Z_vec.rows(); ++i) {
     Omega_y_vec(i) = 0.5 * stan::math::log( (1.0 + Omega_Z_vec(i)) / (1.0 - Omega_Z_vec(i)) );  // undo tanh transform
     jac_vec(i) = stan::math::log( ( stan::math::exp( Omega_y_vec(i) ) +  stan::math::exp( - Omega_y_vec(i) ) ) / 2.0 ); // cosh(y)
   }

   double term_1 = -2.0 * jac_vec.sum();

   double log_det_jacobian = term_1 + term_2;


   //////// outputs
   std::vector<Eigen::Matrix<double, -1, -1, Eigen::RowMajor  > > out = vec_of_mats_test_RM(dim, dim, 6) ;

   out[0] = L_Omega_lower;
   out[1] = Omega;
   out[2] = Omega_Z;
   out[3](0,0) = log_det_jacobian;
   out[4] = sqrt_term;
   out[5] =  term_inv;


   return(out);

 }


 
 
 
 
 // 
 /////////////// for var corr's
 std::vector<Eigen::Matrix<stan::math::var, -1, -1 , Eigen::RowMajor  > >    fn_convert_unconstrained_to_corrs_var_RM(Eigen::Matrix<stan::math::var, -1, -1  > Omega_unconstrained
 ) {


   int dim = Omega_unconstrained.rows();

   ///////////  (1) define lower-triangular matrix Z (tanh transform of Omega_unconstrained)
   Eigen::Matrix<stan::math::var, -1, -1 , Eigen::RowMajor > Omega_Z =    Eigen::Matrix<stan::math::var, -1, -1 , Eigen::RowMajor >::Zero(dim, dim);


   // lower triangle of tanh transforms (j < i)
   for (int i = 1; i < dim; ++i)  {
     for (int j = 0; j < i; ++j) {
       Omega_Z(i,j) = stan::math::tanh(Omega_unconstrained(i,j))  ; // (exp(2 * Omega_unconstrained(i,j)) - 1 ) /   (exp(2 * Omega_unconstrained(i,j)) + 1 );
     }
   }

   
   
   // convert to a vector
   std::vector<Eigen::Matrix<stan::math::var, -1, -1  > > Omega_Z_array_CM = vec_of_mats_test_var(dim, dim, 1);
   Omega_Z_array_CM[0] = Omega_Z;

   Eigen::Matrix<stan::math::var, -1, 1  > Omega_Z_vec = fn_convert_3d_array_of_corrs_to_Eigen_vec_var_RM(Omega_Z_array_CM,
                                                                                                          dim,
                                                                                                          1);

   ///////////// (2) then map Omega_Z -> LOWER - triangular cholesky factor corr's
   Eigen::Matrix<stan::math::var, -1, -1 , Eigen::RowMajor > L_Omega_lower = Eigen::Matrix< stan::math::var, -1, -1 , Eigen::RowMajor >::Zero(dim, dim);

   Eigen::Matrix<stan::math::var, -1, -1 , Eigen::RowMajor > sqrt_term =  Eigen::Matrix< stan::math::var, -1, -1 , Eigen::RowMajor >::Zero(dim, dim);
   Eigen::Matrix<stan::math::var, -1, -1 , Eigen::RowMajor > term_inv = Eigen::Matrix< stan::math::var, -1, -1 , Eigen::RowMajor >::Zero(dim, dim);

   L_Omega_lower(0,0) = 1.0;  // 1 if i = j = 1
   stan::math::var term_2 = 0.0;

   for (int i = 1; i < dim; ++i) {

     L_Omega_lower(i, 0) = Omega_Z(i, 0);

     stan::math::var temp_sum_square = L_Omega_lower(i, 0) * L_Omega_lower(i, 0);

     for (int j = 1; j < i + 1; ++j) {

       if (j == i) { continue; }


       if (j != i) {

         term_2 += 0.5 * stan::math::log1m(temp_sum_square);

         sqrt_term(i, j) = stan::math::sqrt( 1.0 - temp_sum_square);
         term_inv(i, j) = (1.0 /  sqrt_term(i, j)) * (1.0 /  sqrt_term(i, j));

         L_Omega_lower(i, j) =  Omega_Z(i, j) * stan::math::sqrt( 1.0 - temp_sum_square);

         temp_sum_square += L_Omega_lower(i, j) * L_Omega_lower(i, j);


       }

     }

     L_Omega_lower(i, i) =  stan::math::sqrt(1.0 - temp_sum_square);

   }


   //////////// (3) calculate correlation matrix
   Eigen::Matrix<stan::math::var, -1, -1 , Eigen::RowMajor  > Omega = L_Omega_lower * L_Omega_lower.transpose();

   for (int i = 0; i < dim; ++i) {
     Omega(i,i) = 1.0;
   }

   //////////////////// calculate Jacobian adjustment
   // make vector of unconstrained params
   Eigen::Matrix< stan::math::var, -1, 1  > Omega_y_vec(dim, dim) ; // =   Omega_Z_vec;
   Eigen::Matrix< stan::math::var, -1, 1  > jac_vec(dim, dim) ; //  =   Omega_Z_vec;

   for (int i = 0; i < Omega_Z_vec.rows(); ++i) {
     Omega_y_vec(i) = 0.5 * stan::math::log( (1.0 + Omega_Z_vec(i)) / (1.0 - Omega_Z_vec(i)) );  // undo tanh transform
     jac_vec(i) = stan::math::log( ( stan::math::exp( Omega_y_vec(i) ) +  stan::math::exp( - Omega_y_vec(i) ) ) / 2 ); // cosh(y)
   }

   stan::math::var term_1 =   -2.0 * jac_vec.sum();

   Eigen::Matrix< stan::math::var, -1, -1 , Eigen::RowMajor > jac_mtx = Eigen::Matrix< stan::math::var, -1, -1 , Eigen::RowMajor >::Zero(dim, dim);

   stan::math::var log_det_jacobian = term_1 + term_2;

   ////////// outputs

   std::vector<Eigen::Matrix<stan::math::var, -1, -1 , Eigen::RowMajor > > out = vec_of_mats_test_var_RM(dim, dim, 6) ;

   out[0] = L_Omega_lower;
   out[1] = Omega;
   out[2] = Omega_Z;
   out[3](0,0) = log_det_jacobian;
   out[4] = sqrt_term;
   out[5] =  term_inv;

   return(out);

 }

 
  
 
 
 Eigen::Matrix<double, -1, 1>     fn_first_element_neg_rest_pos_colvec(      Eigen::Matrix<double, -1, 1>  col_vec    ) {
   
   col_vec(0) = - col_vec(0);
   
   return(col_vec);
   
 }
 
 
 Eigen::Matrix<double, 1, -1>     fn_first_element_neg_rest_pos(      Eigen::Matrix<double, 1, -1>  row_vec    ) {
   
   row_vec(0) = - row_vec(0);
   
   return(row_vec);
   
 }
 
 
 Eigen::Matrix<float, 1, -1>     fn_first_element_neg_rest_pos_float(   
     Eigen::Matrix<float, 1, -1>  row_vec 
 ) {
   
   row_vec(0) = - row_vec(0);
   
   return(row_vec);
   
 }
 
 
 
 
 Eigen::Matrix<stan::math::var, 1, -1>     fn_first_element_neg_rest_pos_var(   
     Eigen::Matrix<stan::math::var, 1, -1>  row_vec 
 ) {
   
   row_vec(0) = - row_vec(0);
   
   return(row_vec);
   
 }
 
 
 
 
 
 
 std::unique_ptr<size_t[]> get_commutation_unequal_vec
  (unsigned const n, unsigned const m, bool const transpose){
   unsigned const nm = n * m, 
     nnm_p1 = n * nm + 1L, 
     nm_pm = nm + m;
   std::unique_ptr<size_t[]> out(new size_t[nm]);
   size_t * const o_begin = out.get();
   size_t idx = 0L;
   for(unsigned i = 0; i < n; ++i, idx += nm_pm){
     size_t idx1 = idx;
     for(unsigned j = 0; j < m; ++j, idx1 += nnm_p1)
       if(transpose)
         *(o_begin + idx1 / nm) = (idx1 % nm);
       else
         *(o_begin + idx1 % nm) = (idx1 / nm);
   }
   
   return out;
 }

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector commutation_dot
  (unsigned const n, unsigned const m, Rcpp::NumericVector x, 
   bool const transpose){
  size_t const nm = n * m;
  Rcpp::NumericVector out(nm);
  auto const indices = get_commutation_unequal_vec(n, m, transpose);
  
  for(size_t i = 0; i < nm; ++i)
    out[i] = x[*(indices.get() +i )];
  
  return out;
}

Rcpp::NumericMatrix get_commutation_unequal
  (unsigned const n, unsigned const m){
  
  unsigned const nm = n * m, 
    nnm_p1 = n * nm + 1L, 
    nm_pm = nm + m;
  Rcpp::NumericMatrix out(nm, nm);
  double * o = &out[0];
  for(unsigned i = 0; i < n; ++i, o += nm_pm){
    double *o1 = o;
    for(unsigned j = 0; j < m; ++j, o1 += nnm_p1)
      *o1 = 1.;
  }
  
  return out;
}

Rcpp::NumericMatrix get_commutation_equal(unsigned const m){
  unsigned const mm = m * m, 
    mmm = mm * m, 
    mmm_p1 = mmm + 1L, 
    mm_pm = mm + m;
  Rcpp::NumericMatrix out(mm, mm);
  double * const o = &out[0];
  unsigned inc_i(0L);
  for(unsigned i = 0; i < m; ++i, inc_i += m){
    double *o1 = o + inc_i + i * mm, 
      *o2 = o + i     + inc_i * mm;
    for(unsigned j = 0; j < i; ++j, o1 += mmm_p1, o2 += mm_pm){
      *o1 = 1.;
      *o2 = 1.;
    }
    *o1 += 1.;
  }
  return out;
}

// [[Rcpp::export(rng = false)]]
Eigen::Matrix<double, -1, -1  >  get_commutation(unsigned const n, unsigned const m) {
  
  if (n == m)  {
    
    Rcpp::NumericMatrix commutation_mtx_Nuemric_Matrix =  get_commutation_equal(n);
    
    double n_rows = commutation_mtx_Nuemric_Matrix.nrow(); 
    double n_cols = commutation_mtx_Nuemric_Matrix.ncol(); 
    
    Eigen::Matrix<double, -1, -1>  commutation_mtx_Eigen   =  Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols); 
    
    
    for (int i = 0; i < n_rows; ++i) {
      for (int j = 0; j < n_cols; ++j) {
        commutation_mtx_Eigen(i, j) = commutation_mtx_Nuemric_Matrix(i, j) ; 
      }
    }
    
    return commutation_mtx_Eigen;
    
    
  } else { 
    
    Rcpp::NumericMatrix commutation_mtx_Nuemric_Matrix =  get_commutation_unequal(n, m);
    
    double n_rows = commutation_mtx_Nuemric_Matrix.nrow(); 
    double n_cols = commutation_mtx_Nuemric_Matrix.ncol(); 
    
    Eigen::Matrix<double, -1, -1>  commutation_mtx_Eigen   =  Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols); 
    
    
    for (int i = 0; i < n_rows; ++i) {
      for (int j = 0; j < n_cols; ++j) {
        commutation_mtx_Eigen(i, j) = commutation_mtx_Nuemric_Matrix(i, j) ; 
      }
    }
    
    return commutation_mtx_Eigen;
    
    
  }
 
  
}







// [[Rcpp::export(rng = false)]]
Eigen::Matrix<double, -1, -1  > elimination_matrix(const int &n) {
  
  Eigen::Matrix<double, -1, -1> out   =  Eigen::Matrix<double, -1, -1>::Zero((n*(n+1))/2,  n*n); 
  
  for (int j = 0; j < n; ++j) {
    Eigen::Matrix<double, 1, -1> e_j   =  Eigen::Matrix<double, 1, -1>::Zero(n); 
    
    e_j(j) = 1.0;
    
    for (int i = j; i < n; ++i) {
      Eigen::Matrix<double, -1, 1> u   =  Eigen::Matrix<double, -1, 1>::Zero((n*(n+1))/2); 
      u(j*n+i-((j+1)*j)/2) = 1.0;
      Eigen::Matrix<double, 1, -1> e_i   =  Eigen::Matrix<double, 1, -1>::Zero(n); 
      e_i(i) = 1.0;
      
      out += Eigen::kroneckerProduct(u, Eigen::kroneckerProduct(e_j, e_i)); 
    }
  }
  
  return out;
}




// [[Rcpp::export(rng = false)]]
Eigen::Matrix<double, -1, -1  > duplication_matrix(const int &n) {
  
  //arma::mat out((n*(n+1))/2, n*n, arma::fill::zeros);
  Eigen::Matrix<double, -1, -1> out   =  Eigen::Matrix<double, -1, -1>::Zero((n*(n+1))/2,  n*n); 
  
  for (int j = 0; j < n; ++j) {
    for (int i = j; i < n; ++i) {
      // arma::vec u((n*(n+1))/2, arma::fill::zeros);
      Eigen::Matrix<double, -1, 1> u   =  Eigen::Matrix<double, -1, 1>::Zero((n*(n+1))/2);
      u(j*n+i-((j+1)*j)/2) = 1.0;
      
      //       arma::mat T(n,n, arma::fill::zeros);
      Eigen::Matrix<double, -1, -1> T   =  Eigen::Matrix<double, -1, -1>::Zero(n, n);
      T(i,j) = 1.0;
      T(j,i) = 1.0;
      
      Eigen::Map<Eigen::Matrix<double, -1, 1> > T_vec(T.data(), n*n);
      
      out += u * T_vec.transpose();
    }
  }
  
  return out.transpose(); 
  
}








 
 
 
 
 Eigen::Matrix<stan::math::var, -1, 1 >                        lb_ub_lp (stan::math::var  y,
                                                                         stan::math::var lb,
                                                                         stan::math::var ub) {
   
   stan::math::var target = 0 ;
   
   // stan::math::var val   = (lb  + (ub  - lb) * stan::math::inv_logit(y)) ;
   stan::math::var val   =  lb +  (ub - lb) *  0.5 * (1 +  stan::math::tanh(y));
   
   // target += stan::math::log(ub - lb) + stan::math::log_inv_logit(y) + stan::math::log1m_inv_logit(y);
   target +=  stan::math::log(ub - lb) - log(2)  + stan::math::log1m(stan::math::square(stan::math::tanh(y)));
   
   Eigen::Matrix<stan::math::var, -1, 1 > out_mat  = Eigen::Matrix<stan::math::var, -1, 1 >::Zero(2);
   out_mat(0) = target;
   out_mat(1) = val;
   
   return(out_mat) ;
   
 }
 
 
 
 
 
 
 Eigen::Matrix<stan::math::var, -1, 1 >   lb_ub_lp_vec_y (Eigen::Matrix<stan::math::var, -1, 1 > y,
                                                          Eigen::Matrix<stan::math::var, -1, 1 > lb,
                                                          Eigen::Matrix<stan::math::var, -1, 1 > ub) {
   
   stan::math::var target = 0 ;
   
   
   //   stan::math::var val   =  lb +  (ub - lb) *  0.5 * (1 +  stan::math::tanh(y));
   Eigen::Matrix<stan::math::var, -1, 1 >  vec =   (lb.array() +  (ub.array()  - lb.array() ) *  0.5 * (1 +  stan::math::tanh(y).array() )).matrix();
   
   //  target += (stan::math::log( (ub.array() - lb.array()).matrix()).array() + stan::math::log_inv_logit(y).array() + stan::math::log1m_inv_logit(y).array()).matrix().sum() ;
   target +=  (stan::math::log((ub.array() - lb.array()).matrix()).array() - log(2)  +  stan::math::log1m(stan::math::square(stan::math::tanh(y))).array()).matrix().sum();
   
   Eigen::Matrix<stan::math::var, -1, 1 > out_mat  = Eigen::Matrix<stan::math::var, -1, 1 >::Zero(vec.rows() + 1);
   out_mat(0) = target;
   out_mat.segment(1, vec.rows()) = vec;
   
   return(out_mat);
   
 }
 
 
 
 
 // 
 Eigen::Matrix<stan::math::var, -1, -1 >    Spinkney_cholesky_corr_transform_opt( int n,
                                                                                  Eigen::Matrix<stan::math::var, -1, -1 >  lb,
                                                                                  Eigen::Matrix<stan::math::var, -1, -1 >  ub,
                                                                                  Eigen::Matrix<stan::math::var, -1, -1 >  Omega_theta_unconstrained_array,
                                                                                  Eigen::Matrix<int, -1, -1 >  known_values_indicator,
                                                                                  Eigen::Matrix<double, -1, -1 >  known_values) {


   stan::math::var target = 0 ;


   Eigen::Matrix<stan::math::var, -1, -1 > L = Eigen::Matrix<stan::math::var, -1, -1 >::Zero(n, n);
   Eigen::Matrix<stan::math::var, -1, 1 > first_col = Omega_theta_unconstrained_array.col(0).segment(1, n - 1);

   Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_vec_y_outs = lb_ub_lp_vec_y(first_col, lb.col(0), ub.col(0)) ;  // logit bounds
   target += lb_ub_lp_vec_y_outs.eval()(0);

   Eigen::Matrix<stan::math::var, -1, 1 >  z = lb_ub_lp_vec_y_outs.segment(1, n - 1);
   L.col(0).segment(1, n - 1) = z;

   for (int i = 2; i < n + 1; ++i) {
     if (known_values_indicator(i-1, 0) == 1) {
       L(i-1, 0) = stan::math::to_var(known_values(i-1, 0));
       Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_vec_y_out = lb_ub_lp(first_col(i-2), lb(i-1, 0), ub(i-1, 0)) ;  // logit bounds
       target += - lb_ub_lp_vec_y_out.eval()(0); // undo jac adjustment
     }
   }
   L(1, 1) = stan::math::sqrt(1 - stan::math::square(L(1, 0))) ;

   for (int i = 3; i < n + 1; ++i) {

     Eigen::Matrix<stan::math::var, 1, -1 >  row_vec_rep = stan::math::rep_row_vector(stan::math::sqrt(1 - L(i - 1, 0)* L(i - 1, 0)), i - 1) ;
     L.row(i - 1).segment(1, i - 1) = row_vec_rep;

     for (int j = 2; j < i; ++j) {

       stan::math::var   l_ij_old = L(i-1, j-1);
       stan::math::var   l_ij_old_x_l_jj = l_ij_old * L(j-1, j-1); // new
       stan::math::var b1 = stan::math::dot_product(L.row(j - 1).segment(0, j - 1), L.row(i - 1).segment(0, j - 1)) ;
       // stan::math::var b2 = L(j - 1, j - 1) * L(i - 1, j - 1) ; // old

       // stan::math::var  low = std::min(   std::max( b1 - b2, lb(i-1, j-1) / stan::math::abs(L(i-1, j-1)) ), b1 + b2 ); // old
       // stan::math::var   up = std::max(   std::min( b1 + b2, ub(i-1, j-1) / stan::math::abs(L(i-1, j-1)) ), b1 - b2 ); // old

       stan::math::var  low =   std::max( -l_ij_old_x_l_jj, (lb(i-1, j-1) - b1)    );   // new
       stan::math::var   up =   std::min( +l_ij_old_x_l_jj, (ub(i-1, j-1) - b1)    ); // new

       if (known_values_indicator(i-1, j-1) == 1) {
         // L(i-1, j-1) *= ( stan::math::to_var(known_values(i-1, j-1))  - b1) / b2; // old
         L(i-1, j-1)  = stan::math::to_var(known_values(i-1, j-1)) / L(j-1, j-1);  // new
       } else {
         Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_outs = lb_ub_lp(Omega_theta_unconstrained_array(i-1, j-1), low,  up) ;
         target += lb_ub_lp_outs.eval()(0); // old

         stan::math::var x = lb_ub_lp_outs.eval()(1);    // logit bounds
         target +=  - stan::math::log(L(j-1, j-1)) ;  //  Jacobian for transformation  z -> L_Omega

         //   L(i-1, j-1) *= (x - b1) / b2; // old
         L(i-1, j-1)  = x / L(j-1, j-1); //  low + (up - low) * x; // new
       }

       //    target += - stan::math::log(L(j-1, j-1)); // old

       stan::math::var   l_ij_new = L(i-1, j-1);
       L.row(i - 1).segment(j, i - j).array() *= stan::math::sqrt(  1 -  ( (l_ij_new / l_ij_old) * (l_ij_new / l_ij_old)  )  );

     }

   }
   L(0, 0) = 1;

   //////////// output
   Eigen::Matrix<stan::math::var, -1, -1 > out_mat = Eigen::Matrix<stan::math::var, -1, -1 >::Zero(1 + n , n);

   out_mat(0, 0) = target;
   out_mat.block(1, 0, n, n) = L;

   return(out_mat);

 }


 
 

 
 Eigen::Matrix<stan::math::var, -1, -1 >    Spinkney_LDL_bounds_opt( int K,
                                                                     Eigen::Matrix<stan::math::var, -1, -1 >  lb,
                                                                     Eigen::Matrix<stan::math::var, -1, -1 >  ub,
                                                                     Eigen::Matrix<stan::math::var, -1, -1 >  Omega_theta_unconstrained_array,
                                                                     Eigen::Matrix<int, -1, -1 >  known_values_indicator,
                                                                     Eigen::Matrix<double, -1, -1 >  known_values) {
   
   
   stan::math::var target = 0 ;
   
   Eigen::Matrix<stan::math::var, -1, 1 > first_col = Omega_theta_unconstrained_array.col(0).segment(1, K - 1);
   Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_vec_y_outs = lb_ub_lp_vec_y(first_col, lb.col(0), ub.col(0)) ;  // logit bounds
   target += lb_ub_lp_vec_y_outs.eval()(0);
   Eigen::Matrix<stan::math::var, -1, 1 >  z = lb_ub_lp_vec_y_outs.segment(1, K - 1);
   
   Eigen::Matrix<stan::math::var, -1, -1 > L = Eigen::Matrix<stan::math::var, -1, -1 >::Zero(K, K);
   
   for (int i = 0; i < K; ++i) {
     L(i, i) = 1;
   }
   
   Eigen::Matrix<stan::math::var, -1, 1 >  D = Eigen::Matrix<stan::math::var, -1, 1 >::Zero(K);
   
   D(0) = 1;
   L.col(0).segment(1, K - 1) = z;
   D(1) = 1 -  stan::math::square(L(1, 0)) ;
   
   for (int i = 2; i < K + 1; ++i) {
     if (known_values_indicator(i-1, 0) == 1) {
       L(i-1, 0) = stan::math::to_var(known_values(i-1, 0));
       Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_vec_y_out = lb_ub_lp(first_col(i-2), lb(i-1, 0), ub(i-1, 0)) ;  // logit bounds
       target += - lb_ub_lp_vec_y_out.eval()(0); // undo jac adjustment
     }
   }
   
   for (int i = 3; i < K + 1; ++i) {
     
     D(i-1) = 1 - stan::math::square(L(i-1, 0)) ;
     Eigen::Matrix<stan::math::var, 1, -1 >  row_vec_rep = stan::math::rep_row_vector(1 - stan::math::square(L(i-1, 0)), i - 2) ;
     L.row(i - 1).segment(1, i - 2) = row_vec_rep;
     stan::math::var   l_ij_old = L(i-1, 1);
     
     for (int j = 2; j < i; ++j) {
       
       stan::math::var b1 = stan::math::dot_product(L.row(j - 1).head(j - 1), (D.head(j - 1).transpose().array() *  L.row(i - 1).head(j - 1).array() ).matrix()  ) ;
       
       Eigen::Matrix<stan::math::var, -1, 1 > low_vec_to_max(2);
       Eigen::Matrix<stan::math::var, -1, 1 > up_vec_to_min(2);
       low_vec_to_max(0) = - stan::math::sqrt(l_ij_old) * D(j-1) ;
       low_vec_to_max(1) =   (lb(i-1, j-1) - b1) ;
       up_vec_to_min(0) =    stan::math::sqrt(l_ij_old) * D(j-1) ;
       up_vec_to_min(1) =    (ub(i-1, j-1) - b1)  ;
       
       stan::math::var  low =    stan::math::max( low_vec_to_max   );   // new
       stan::math::var  up  =    stan::math::min( up_vec_to_min    );   // new
       
       if (known_values_indicator(i-1, j-1) == 1) {
         L(i-1, j-1) =  stan::math::to_var(known_values(i-1, j-1)) /  D(j-1)  ; // new
       } else {
         Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_outs = lb_ub_lp(Omega_theta_unconstrained_array(i-1, j-1), low,  up) ;
         target += lb_ub_lp_outs.eval()(0);
         stan::math::var x = lb_ub_lp_outs.eval()(1);    // logit bounds
         L(i-1, j-1)  = x / D(j-1) ;
         target += -0.5 * stan::math::log(D(j-1)) ;
         // target += -  stan::math::log(D(j-1)) ;
       }
       
       l_ij_old *= 1 - (D(j-1) *  stan::math::square(L(i-1, j-1) )) / l_ij_old;
     }
     
     D(i-1) = l_ij_old;
   }
   //L(0, 0) = 1;
   
   //////////// output
   Eigen::Matrix<stan::math::var, -1, -1 > out_mat = Eigen::Matrix<stan::math::var, -1, -1 >::Zero(1 + K , K);
   
   out_mat(0, 0) = target;
   // out_mat.block(1, 0, n, n) = L;
   out_mat.block(1, 0, K, K) = stan::math::diag_post_multiply(L, stan::math::sqrt(stan::math::abs(D)));
   
   return(out_mat);
   
 }
 
 
 
 
 
 
 
 Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor >    Spinkney_LDL_bounds_opt_RM( int K,
                                                                     Eigen::Matrix<stan::math::var, -1, -1 >  lb,
                                                                     Eigen::Matrix<stan::math::var, -1, -1 >  ub,
                                                                     Eigen::Matrix<stan::math::var, -1, -1 >  Omega_theta_unconstrained_array,
                                                                     Eigen::Matrix<int, -1, -1 >  known_values_indicator,
                                                                     Eigen::Matrix<double, -1, -1 >  known_values) {


   stan::math::var target = 0.0;

   Eigen::Matrix<stan::math::var, -1, 1 > first_col = Omega_theta_unconstrained_array.col(0).segment(1, K - 1);
   Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_vec_y_outs = lb_ub_lp_vec_y(first_col, lb.col(0), ub.col(0)) ;  // logit bounds
   target += lb_ub_lp_vec_y_outs.eval()(0);
   Eigen::Matrix<stan::math::var, -1, 1 >  z = lb_ub_lp_vec_y_outs.segment(1, K - 1);

   Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor  > L = Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor  >::Zero(K, K);

   for (int i = 0; i < K; ++i) {
     L(i, i) = 1.0;
   }

   Eigen::Matrix<stan::math::var, -1, 1 >  D = Eigen::Matrix<stan::math::var, -1, 1 >::Zero(K);

   D(0) = 1.0;
   L.col(0).segment(1, K - 1) = z;
   D(1) = 1.0 -  stan::math::square(L(1, 0)) ;

   for (int i = 2; i < K + 1; ++i) {
     if (known_values_indicator(i-1, 0) == 1) {
       L(i-1, 0) = stan::math::to_var(known_values(i-1, 0));
       Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_vec_y_out = lb_ub_lp(first_col(i-2), lb(i-1, 0), ub(i-1, 0)) ;  // logit bounds
       target += - lb_ub_lp_vec_y_out.eval()(0); // undo jac adjustment
     }
   }

   for (int i = 3; i < K + 1; ++i) {

     D(i-1) = 1 - stan::math::square(L(i-1, 0)) ;
     Eigen::Matrix<stan::math::var, 1, -1 >  row_vec_rep = stan::math::rep_row_vector(1 - stan::math::square(L(i-1, 0)), i - 2) ;
     L.row(i - 1).segment(1, i - 2) = row_vec_rep;
     stan::math::var   l_ij_old = L(i-1, 1);

     for (int j = 2; j < i; ++j) {

       stan::math::var b1 = stan::math::dot_product(L.row(j - 1).head(j - 1), (D.head(j - 1).transpose().array() *  L.row(i - 1).head(j - 1).array() ).matrix()  ) ;

       Eigen::Matrix<stan::math::var, -1, 1 > low_vec_to_max(2);
       Eigen::Matrix<stan::math::var, -1, 1 > up_vec_to_min(2);
       low_vec_to_max(0) = - stan::math::sqrt(l_ij_old) * D(j-1) ;
       low_vec_to_max(1) =   (lb(i-1, j-1) - b1) ;
       up_vec_to_min(0) =    stan::math::sqrt(l_ij_old) * D(j-1) ;
       up_vec_to_min(1) =    (ub(i-1, j-1) - b1)  ;

       stan::math::var  low =    stan::math::max( low_vec_to_max   );   // new
       stan::math::var  up  =    stan::math::min( up_vec_to_min    );   // new

       if (known_values_indicator(i-1, j-1) == 1) {
         L(i-1, j-1) =  stan::math::to_var(known_values(i-1, j-1)) /  D(j-1)  ; // new
       } else {
         Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_outs = lb_ub_lp(Omega_theta_unconstrained_array(i-1, j-1), low,  up) ;
         target += lb_ub_lp_outs.eval()(0);
         stan::math::var x = lb_ub_lp_outs.eval()(1);    // logit bounds
         L(i-1, j-1)  = x / D(j-1) ;
         target += -0.5 * stan::math::log(D(j-1)) ;
         // target += -  stan::math::log(D(j-1)) ;
       }

       l_ij_old *= 1 - (D(j-1) *  stan::math::square(L(i-1, j-1) )) / l_ij_old;
     }
     D(i-1) = l_ij_old;
   }
   //L(0, 0) = 1;

   //////////// output
   Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor  > out_mat = Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor  >::Zero(1 + K , K);

   out_mat(0, 0) = target;
   // out_mat.block(1, 0, n, n) = L;
   out_mat.block(1, 0, K, K) = stan::math::diag_post_multiply(L, stan::math::sqrt(stan::math::abs(D)));

   return(out_mat);

 }



 
 
 
 
 
 


 
 
 
 
 
 
 
 



 // [[Rcpp::export]]
 Eigen::Matrix<double, -1, 1 >    fn_lp_and_grad_MVP_LC_using_Chol_Spinkney_MD_and_AD(      Eigen::Matrix<double, -1, 1  > theta_main,
                                                                                         Eigen::Matrix<double, -1, 1  > theta_us,
                                                                                         Eigen::Matrix<int, -1, -1>	 y,
                                                                                         std::vector<Eigen::Matrix<double, -1, -1 > >  X,
                                                                                         Rcpp::List other_args
                                                                                  ) {


   
   
   
   const int n_cores = other_args(0); 
   const bool exclude_priors = other_args(1); 
   const bool CI = other_args(2); 
   Eigen::Matrix<double, -1, 1>  lkj_cholesky_eta = other_args(3); 
   Eigen::Matrix<double, -1, -1> prior_coeffs_mean  = other_args(4); 
   Eigen::Matrix<double, -1, -1> prior_coeffs_sd = other_args(5); 
   const int n_class = other_args(6); 
   const int ub_threshold_phi_approx = other_args(7); 
   const int n_chunks = other_args(8); 
   const bool corr_force_positive = other_args(9); 
   std::vector<Eigen::Matrix<double, -1, -1 > >  prior_for_corr_a = other_args(10); 
   std::vector<Eigen::Matrix<double, -1, -1 > >  prior_for_corr_b = other_args(11); 
   const bool corr_prior_beta  = other_args(12); 
   const bool corr_prior_norm  = other_args(13); 
   std::vector<Eigen::Matrix<double, -1, -1 > >  lb_corr = other_args(14); 
   std::vector<Eigen::Matrix<double, -1, -1 > >  ub_corr = other_args(15); 
   std::vector<Eigen::Matrix<int, -1, -1 > >      known_values_indicator = other_args(16); 
   std::vector<Eigen::Matrix<double, -1, -1 > >   known_values = other_args(17); 
   const double prev_prior_a = other_args(18); 
   const double prev_prior_b = other_args(19); 
   const bool exp_fast = other_args(20); 
   const bool log_fast = other_args(21); 
   std::string Phi_type = other_args(22); 
   Eigen::Matrix<double, -1, -1> LT_b_priors_shape  = other_args(23); 
   Eigen::Matrix<double, -1, -1> LT_b_priors_scale  = other_args(24); 
   Eigen::Matrix<double, -1, -1> LT_known_bs_indicator = other_args(25); 
   Eigen::Matrix<double, -1, -1> LT_known_bs_values  = other_args(26); 

   const int n_tests = y.cols();
   const int N = y.rows();
   const int n_corrs =  n_class * n_tests * (n_tests - 1) * 0.5;
   const int n_coeffs = n_class * n_tests * 1;
   const int n_us =  1 *  N * n_tests;

   const int n_params = theta_us.rows() +  theta_main.rows()   ; // n_corrs + n_coeffs + n_us + n_class;
   const int n_params_main = n_params - n_us;
   
   const double sqrt_2_pi_recip =   1.0 / sqrt(2.0 * M_PI) ; //  0.3989422804;
   const double sqrt_2_recip = 1.0 / stan::math::sqrt(2.0);
   const double minus_sqrt_2_recip =  - sqrt_2_recip;
   const double a = 0.07056;
   const double b = 1.5976;
   const double a_times_3 = 3.0 * 0.07056;
   const double s = 1.0/1.702;
   

   // corrs
   Eigen::Matrix<double, -1, 1  >  Omega_raw_vec_double = theta_main.head(n_corrs); // .cast<double>();

   Eigen::Matrix<stan::math::var, -1, 1  >  Omega_raw_vec_var =  stan::math::to_var(Omega_raw_vec_double) ;
   Eigen::Matrix<stan::math::var, -1, 1  >  Omega_constrained_raw_vec_var =  Eigen::Matrix<stan::math::var, -1, 1  >::Zero(n_corrs) ;
   Omega_constrained_raw_vec_var = Omega_raw_vec_var ; // no transformation for Nump needed! done later on


   // coeffs
   Eigen::Matrix<double, -1, -1> beta_double_array(n_class, n_tests);

   {
     int i = n_corrs;
     for (int c = 0; c < n_class; ++c) {
       for (int t = 0; t < n_tests; ++t) {
         beta_double_array(c, t) = theta_main(i);
         i += 1;
       }
     }
   }


   // prev
   double u_prev_diseased = theta_main(n_params_main - 1);



   Eigen::Matrix<double, -1, 1 >  target_AD_grad(n_corrs);
   stan::math::var target_AD = 0.0;
   double grad_prev_AD = 0.0;

   int dim_choose_2 = n_tests * (n_tests - 1) * 0.5 ;
   std::vector<Eigen::Matrix<double, -1, -1 > > deriv_L_wrt_unc_full = vec_of_mats_test(dim_choose_2 + n_tests, dim_choose_2, n_class);
   std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_double = vec_of_mats_test(n_tests, n_tests, n_class);
   std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_recip_double = L_Omega_double ; 

 
   
   {
     
     
     std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > Omega_unconstrained_var = fn_convert_std_vec_of_corrs_to_3d_array_var( Eigen_vec_to_std_vec_var(Omega_constrained_raw_vec_var),
                                                                                                                                  n_tests,
                                                                                                                                  n_class);
     
     
     std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > L_Omega_var = vec_of_mats_test_var(n_tests, n_tests, n_class);
     std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_var   = vec_of_mats_test_var(n_tests, n_tests, n_class);
     
     for (int c = 0; c < n_class; ++c) {
       Eigen::Matrix<stan::math::var, -1, -1 >  ub = stan::math::to_var(ub_corr[c]);
       Eigen::Matrix<stan::math::var, -1, -1 >  lb = stan::math::to_var(lb_corr[c]);
       
       Eigen::Matrix<stan::math::var, -1, -1  >  Chol_Schur_outs =  Spinkney_LDL_bounds_opt(n_tests, lb, ub, Omega_unconstrained_var[c], known_values_indicator[c], known_values[c]) ; //   Omega_unconstrained_var[c], n_tests, tol )  ;
       
       L_Omega_var[c]   =  Chol_Schur_outs.block(1, 0, n_tests, n_tests);
       Omega_var[c] =   L_Omega_var[c] * L_Omega_var[c].transpose() ;
       
       
       target_AD +=   Chol_Schur_outs(0, 0); // now can set prior directly on Omega
     }
     
     
     
     for (int c = 0; c < n_class; ++c) {
       if ( (corr_prior_beta == false)   &&  (corr_prior_norm == false) ) {
         target_AD +=  stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c], lkj_cholesky_eta(c)) ;
       } else if ( (corr_prior_beta == true)   &&  (corr_prior_norm == false) ) {
         for (int i = 1; i < n_tests; i++) {
           for (int j = 0; j < i; j++) {
             target_AD +=  stan::math::beta_lpdf(  (Omega_var[c](i, j) + 1)/2, prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
           }
         }
         //  Jacobian for  Omega -> L_Omega transformation for prior log-densities (since both LKJ and truncated normal prior densities are in terms of Omega, not L_Omega)
         Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
         for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i+1) ) * log(L_Omega_var[c](i, i));
         target_AD  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
       } else if  ( (corr_prior_beta == false)   &&  (corr_prior_norm == true) ) {
         for (int i = 1; i < n_tests; i++) {
           for (int j = 0; j < i; j++) {
             target_AD +=  stan::math::normal_lpdf(  Omega_var[c](i, j), prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
           }
         }
         Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
         for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i+1) ) * log(L_Omega_var[c](i, i));
         target_AD  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
       }
     }
     
     
     ///////////////////////
     stan::math::set_zero_all_adjoints();
     target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
     target_AD_grad =  Omega_raw_vec_var.adj();    // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
     stan::math::set_zero_all_adjoints();
     //////////////////////////////////////////////////////////// end of AD part
     
     
     
     /////////////  prev stuff  ---- vars
     std::vector<stan::math::var> 	 u_prev_var_vec_var(n_class, 0.0);
     std::vector<stan::math::var> 	 prev_var_vec_var(n_class, 0.0);
     std::vector<stan::math::var> 	 tanh_u_prev_var(n_class, 0.0);
     Eigen::Matrix<stan::math::var, -1, -1>	 prev_var(1, n_class);
     
     u_prev_var_vec_var[1] =  stan::math::to_var(u_prev_diseased);
     tanh_u_prev_var[1] = ( exp(2*u_prev_var_vec_var[1] ) - 1) / ( exp(2*u_prev_var_vec_var[1] ) + 1) ;
     u_prev_var_vec_var[0] =   0.5 *  log( (1 + ( (1 - 0.5 * ( tanh_u_prev_var[1] + 1))*2 - 1) ) / (1 - ( (1 - 0.5 * ( tanh_u_prev_var[1] + 1))*2 - 1) ) )  ;
     tanh_u_prev_var[0] = (exp(2*u_prev_var_vec_var[0] ) - 1) / ( exp(2*u_prev_var_vec_var[0] ) + 1) ;
     
     prev_var_vec_var[1] = 0.5 * ( tanh_u_prev_var[1] + 1);
     prev_var_vec_var[0] =  0.5 * ( tanh_u_prev_var[0] + 1);
     prev_var(0,1) =  prev_var_vec_var[1];
     prev_var(0,0) =  prev_var_vec_var[0];
     
     stan::math::var tanh_pu_deriv_var = ( 1 - tanh_u_prev_var[1] * tanh_u_prev_var[1]  );
     stan::math::var deriv_p_wrt_pu_var = 0.5 *  tanh_pu_deriv_var;
     stan::math::var tanh_pu_second_deriv_var  = -2 * tanh_u_prev_var[1]  * tanh_pu_deriv_var;
     stan::math::var log_jac_p_deriv_wrt_pu_var  = ( 1 / deriv_p_wrt_pu_var) * 0.5 * tanh_pu_second_deriv_var; // for gradient of u's
     stan::math::var  log_jac_p_var =    log( deriv_p_wrt_pu_var );
     
     
     stan::math::var  target_AD_prev = beta_lpdf(  prev_var(0,1), prev_prior_a, prev_prior_b  ); // weakly informative prior - helps avoid boundaries with slight negative skew (for lower N)
     target_AD_prev += log_jac_p_var;
     
     target_AD  +=  target_AD_prev;
     
     ///////////////////////
     target_AD_prev.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
     grad_prev_AD  =  u_prev_var_vec_var[1].adj() - u_prev_var_vec_var[0].adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
     stan::math::set_zero_all_adjoints();
     //////////////////////////////////////////////////////////// end of AD part
     
     
     
     
     for (int c = 0; c < n_class; ++c) {
       int cnt_1 = 0;
       for (int k = 0; k < n_tests; k++) {
         for (int l = 0; l < k + 1; l++) {
           (  L_Omega_var[c](k, l)).grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
           int cnt_2 = 0;
           for (int i = 1; i < n_tests; i++) {
             for (int j = 0; j < i; j++) {
               deriv_L_wrt_unc_full[c](cnt_1, cnt_2)  =   Omega_unconstrained_var[c](i, j).adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
               cnt_2 += 1;
             }
           }
           stan::math::set_zero_all_adjoints();
           cnt_1 += 1;
         }
       }
     }
     
     
     ///////////////// get cholesky factor's (lower-triangular) of corr matrices
     // convert to 3d var array
     for (int c = 0; c < n_class; ++c) {
       for (int t1 = 0; t1 < n_tests; ++t1) {
         for (int t2 = 0; t2 < n_tests; ++t2) {
           L_Omega_double[c](t1, t2) =   L_Omega_var[c](t1, t2).val()  ;
           L_Omega_recip_double[c](t1, t2) =   1.0 / L_Omega_double[c](t1, t2) ;
         }
       }
     }
     
     stan::math::recover_memory();
   }
   



   /////////////  prev stuff
   std::vector<double> 	 u_prev_var_vec(n_class, 0.0);
   std::vector<double> 	 prev_var_vec(n_class, 0.0);
   std::vector<double> 	 tanh_u_prev(n_class, 0.0);
   Eigen::Matrix<double, -1, -1>	 prev(1, n_class);

   u_prev_var_vec[1] =  (double) u_prev_diseased ;
   tanh_u_prev[1] = ( exp(2.0*u_prev_var_vec[1] ) - 1.0) / ( exp(2.0*u_prev_var_vec[1] ) + 1.0) ;
   u_prev_var_vec[0] =   0.5 *  log( (1 + ( (1 - 0.5 * ( tanh_u_prev[1] + 1))*2.0 - 1.0) ) / (1.0 - ( (1.0 - 0.5 * ( tanh_u_prev[1] + 1.0))*2.0 - 1.0) ) )  ;
   tanh_u_prev[0] = (exp(2.0*u_prev_var_vec[0] ) - 1.0) / ( exp(2.0*u_prev_var_vec[0] ) + 1.0) ;

   prev_var_vec[1] =  0.5 * ( tanh_u_prev[1] + 1.0);
   prev_var_vec[0] =  0.5 * ( tanh_u_prev[0] + 1.0);
   prev(0,1) =  prev_var_vec[1];
   prev(0,0) =  prev_var_vec[0];


   double tanh_pu_deriv = ( 1.0 - tanh_u_prev[1] * tanh_u_prev[1]  );
   double deriv_p_wrt_pu_double = 0.5 *  tanh_pu_deriv;
   double tanh_pu_second_deriv  = -2.0 * tanh_u_prev[1]  * tanh_pu_deriv;
   double log_jac_p_deriv_wrt_pu  = ( 1.0 / deriv_p_wrt_pu_double) * 0.5 * tanh_pu_second_deriv; // for gradient of u's
   double  log_jac_p =    log( deriv_p_wrt_pu_double );



   ///////////////////////////////////////////////////////////////////////// prior densities
   double prior_densities = 0.0;


   if (exclude_priors == false) {
     ///////////////////// priors for coeffs
     double prior_densities_coeffs = 0.0;
     for (int c = 0; c < n_class; c++) {
       for (int t = 0; t < n_tests; t++) {
            prior_densities_coeffs  += stan::math::normal_lpdf(beta_double_array(c, t), prior_coeffs_mean(c, t), prior_coeffs_sd(c, t));
       }
     }
     double prior_densities_corrs = target_AD.val();
     prior_densities = prior_densities_coeffs  +      prior_densities_corrs ;     // total prior densities and Jacobian adjustments
   }


   /////////////////////////////////////////////////////////////////////////////////////////////////////
   ///////// likelihood
   int chunk_counter = 0;
   int chunk_size  = std::round( N / n_chunks  / 2) * 2;  ; // N / n_chunks;


   double log_prob_out = 0.0;


   Eigen::Matrix<double, -1, -1 >  log_prev = prev;

   for (int c = 0; c < n_class; c++) {
     log_prev(0,c) =  log(prev(0,c));
     for (int t = 0; t < n_tests; t++) {
       if (CI == true)      L_Omega_double[c](t,t) = 1.0;
     }
   }

 
   
   if (exp_fast == true)   theta_us.array() =      fast_tanh_approx_Eigen( theta_us ).array(); 
   else                    theta_us.array() =      ( theta_us ).array().tanh(); 
   //tanh_approx_2_Eigen
   
   

   double log_jac_u  =  0.0;
   if    (log_fast == true)    {  // most stable
       log_jac_u  =    (  fast_log_approx_double_wo_checks_Eigen( 0.5 *   (  1.0 -  ( theta_us.array()   *  theta_us.array()   ).array()  ).array()  ).array()   ).matrix().sum();  // log
   } else {
       log_jac_u  =    (  ( 0.5 *   (  1.0 -  ( theta_us.array()   *  theta_us.array()   ).array()  ).array()  ).array().log()   ).matrix().sum();  // log
   }


   ///////////////////////////////////////////////
   Eigen::Matrix<double , -1, 1>   beta_grad_vec   =  Eigen::Matrix<double, -1, 1>::Zero(n_coeffs);  //
   Eigen::Matrix<double, -1, -1>   beta_grad_array  =  Eigen::Matrix<double, -1, -1>::Zero(2, n_tests); //
   std::vector<Eigen::Matrix<double, -1, -1 > > U_Omega_grad_array =  vec_of_mats_test(n_tests, n_tests, 2); //
   Eigen::Matrix<double, -1, 1 > L_Omega_grad_vec(n_corrs + (2 * n_tests)); //
   Eigen::Matrix<double, -1, 1 > U_Omega_grad_vec(n_corrs); //
   Eigen::Matrix<double, -1, 1>  prev_unconstrained_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(2); //
   Eigen::Matrix<double, -1, 1>  prev_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(2); //
   Eigen::Matrix<double, -1, 1>  prev_unconstrained_grad_vec_out =   Eigen::Matrix<double, -1, 1>::Zero(2 - 1); //
   // ///////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////// output vec
   Eigen::Matrix<double, -1, 1> out_mat    =  Eigen::Matrix<double, -1, 1>::Zero(n_params + 1 + N);  ///////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////////////

   double log_prob = 0.0;

   {


     //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
     ///////////////////////////////////////////////
     std::vector<Eigen::Matrix<double, -1, -1 > >  Z_std_norm =  vec_of_mats_test(chunk_size, n_tests, 2); //////////////////////////////
     std::vector<Eigen::Matrix<double, -1, -1 > >  Bound_Z =  Z_std_norm ;  
     std::vector<Eigen::Matrix<double, -1, -1 > >  abs_Bound_Z =  Z_std_norm ;  
     std::vector<Eigen::Matrix<double, -1, -1 > >  Bound_U_Phi_Bound_Z =  Z_std_norm ;  //  vec_of_mats_test(chunk_size, n_tests, 2); //////////////////////////////
     std::vector<Eigen::Matrix<double, -1, -1 > >  Phi_Z =  Z_std_norm ;  //  vec_of_mats_test(chunk_size, n_tests, 2); //////////////////////////////
     std::vector<Eigen::Matrix<double, -1, -1 > >  y1_or_phi_Bound_Z =  Z_std_norm ;  //  vec_of_mats_test(chunk_size, n_tests, 2); //////////////////////////////
     std::vector<Eigen::Matrix<double, -1, -1 > >  prob =  Z_std_norm ;  //  vec_of_mats_test(chunk_size, n_tests, 2); //////////////////////////////
     ///////////////////////////////////////////////);
     Eigen::Array<double, -1, -1> y_chunk = Eigen::Array<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Array<double, -1, -1> u_array =  y_chunk ;  
     ///////////////////////////////////////////////
     Eigen::Matrix<double, -1, 1> prod_container_or_inc_array  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
     Eigen::Matrix<double, -1, 1>      prob_n  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
     ///////////////////////////////////////////////
     Eigen::Matrix<double, -1, -1>     common_grad_term_1   =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Matrix<double, -1, -1 >    L_Omega_diag_recip_array   = common_grad_term_1 ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Matrix<double, -1, -1 >    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip   =  common_grad_term_1 ; //   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Matrix<double, -1, -1 >    y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip   =   common_grad_term_1 ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Matrix<double, -1, -1 >    prop_rowwise_prod_temp   =   common_grad_term_1 ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Matrix<double, -1, -1>     grad_prob =    common_grad_term_1 ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Matrix<double, -1, -1>     z_grad_term =  common_grad_term_1 ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Matrix<double, -1, -1>     u_grad_array_CM_chunk   =   common_grad_term_1 ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);    ////////
     Eigen::Matrix<double, -1, -1>     phi_Z_recip  =   common_grad_term_1 ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     ///////////////////////////////////////////////
     Eigen::Matrix<double, -1, 1>      derivs_chain_container_vec  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
     Eigen::Matrix<double, -1, 1>      prop_rowwise_prod_temp_all  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
     Eigen::Array<int, -1, -1>         abs_indicator = Eigen::Array<int, -1, -1>::Zero(n_tests, n_class);
     ////////////////////////////////////////////////


     {

       // /////////////////////////////////////////////////////////////////////////////////////////////////////
       Eigen::Matrix<double, -1, -1>    lp_array  =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, 2);///////
       ////////////////////////////////////////////////////////////////////////////////////////////////////////


     for (int nc = 0; nc < n_chunks; nc++) {
       
       abs_indicator.array() = 0;

         u_grad_array_CM_chunk.array() = 0.0;

              int chunk_counter = nc;

              y_chunk = y.middleRows(chunk_size * chunk_counter , chunk_size).array().cast<double>() ;
              
              u_array  = 0.5 * (  theta_us.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests).reshaped(chunk_size, n_tests).array() + 1.0 ).array() ;


                 for (int c = 0; c < n_class; c++) {

                                   prod_container_or_inc_array.array()  = 0.0; // needs to be reset to 0

                                   for (int t = 0; t < n_tests; t++) { 
                                     
                                                    Bound_Z[c].col(t).array() =    L_Omega_recip_double[c](t, t) * (  - ( beta_double_array(c, t) +      prod_container_or_inc_array.array()   )  ) ; // / L_Omega_double[c](t, t)     ;
                                                    abs_Bound_Z[c].col(t).array() =    Bound_Z[c].col(t).array().abs(); 
                                                    if ((abs_Bound_Z[c].col(t).array() > 5.0).matrix().sum() > 0)     abs_indicator(t, c) = 1;
                                                    
                                                    if (Phi_type == "Phi")   Bound_U_Phi_Bound_Z[c].col(t).array() =  0.5 *   ( minus_sqrt_2_recip * Bound_Z[c].col(t).array() ).erfc() ;
                                                    else   Bound_U_Phi_Bound_Z[c].col(t).array() = Phi_approx_fast_Eigen(Bound_Z[c].col(t).array() );
                                    
                                                    // for all configs 
                                                   if (  abs_indicator(t, c) == 1) {
                                                        // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
                                                      for (int n_index = 0; n_index < chunk_size; n_index++) { // can't vectorise ?!
                                                         if ( abs_Bound_Z[c](n_index, t) > 5.0   ) {
                                                            Bound_U_Phi_Bound_Z[c](n_index, t) =    stan::math::inv_logit( 1.702 *  Bound_Z[c](n_index, t)  );  //  stan::math::inv_logit( 1.702 * Bound_Z[c].col(t)  )
                                                         }
                                                       }
                                                    }
                                                    
                                                    Phi_Z[c].col(t).array()    =      (   y_chunk.col(t)   *  Bound_U_Phi_Bound_Z[c].col(t).array() +
                                                                                      (   y_chunk.col(t)   -  Bound_U_Phi_Bound_Z[c].col(t).array() ) *   ( y_chunk.col(t)  + (  y_chunk.col(t)  - 1.0)  )  * u_array.col(t)   )   ;  
                                                
                                                if (Phi_type == "Phi") {    
                                                        if (log_fast == false) Z_std_norm[c].col(t).array() = stan::math::inv_Phi(Phi_Z[c].col(t)).array() ; //    qnorm_rcpp_Eigen( Phi_Z[c].col(t).array());   
                                                        else                   Z_std_norm[c].col(t).array() = qnorm_w_fast_log_rcpp_Eigen( Phi_Z[c].col(t).array());  // with approximations
                                                } else if (Phi_type == "Phi_approx") {
                                                      Z_std_norm[c].col(t).array() =  inv_Phi_approx_fast_Eigen(Phi_Z[c].col(t).array());
                                                }
      
                                                 // for all configs 
                                                 if (abs_indicator(t, c) == 1) {
                                                     // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
                                                   for (int n_index = 0; n_index < chunk_size; n_index++) { // can't vectoriose ?!
                                                         if ( abs_Bound_Z[c](n_index, t) > 5.0   ) {
                                                            Z_std_norm[c](n_index, t) =    s  *  stan::math::logit(Phi_Z[c](n_index, t));  //   s *  fast_logit_1_Eigen(    Phi_Z[c].col(t) ).array()
                                                          }
                                                   }
                                                 } 
                                                 
                                                 if (t < n_tests - 1)       prod_container_or_inc_array.array()  =   ( Z_std_norm[c].topLeftCorner(chunk_size, t + 1)  *   ( L_Omega_double[c].row(t+1).head(t+1).transpose()  ) ) ;      // for all configs 

                                   } // end of t loop


                                   prob[c].array() =  y_chunk.array() * ( 1.0 -    Bound_U_Phi_Bound_Z[c].array() ) +  ( y_chunk - 1.0  )   *    Bound_U_Phi_Bound_Z[c].array() *   ( y_chunk +  (  y_chunk - 1.0)  )  ;   // for all configs 
                                     
                                 if (log_fast == true)  y1_or_phi_Bound_Z[c].array()  =   fast_log_approx_double_wo_checks_Eigen_mat(prob[c].array() ) ;
                                 else                   y1_or_phi_Bound_Z[c].array()  =   prob[c].array().log();

                                lp_array.col(c).array() =     y1_or_phi_Bound_Z[c].rowwise().sum().array() +  log_prev(0,c) ;

                        } // end of c loop

                                                                                           
                         if    ( (log_fast == false) && (exp_fast == false)  )  {  
                               out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()   = (  lp_array.array().maxCoeff() +  (   (  (lp_array.array() - lp_array.array().maxCoeff() ).array()).exp().matrix().rowwise().sum().array()   ).log()  )  ;
                               prob_n  =  (out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()).exp() ;
                         } else if  ( (log_fast == true) && (exp_fast == false)  )  {  
                               out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()   = (  lp_array.array().maxCoeff() +  fast_log_approx_double_wo_checks_Eigen(   (  (lp_array.array() - lp_array.array().maxCoeff() ).array()).exp().matrix().rowwise().sum().array()   )  )  ;
                               prob_n  =  (out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()).exp() ;
                         } else if  ( (log_fast == false) && (exp_fast == true)  )  { 
                               out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()   = (  lp_array.array().maxCoeff() +  (   fast_exp_double_wo_checks_Eigen_mat(  (lp_array.array() - lp_array.array().maxCoeff() ).array()).matrix().rowwise().sum().array()   ).log()  )  ;
                               prob_n  =  fast_exp_double_wo_checks_Eigen(out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()) ;
                         } else if  ( (log_fast == true) && (exp_fast == true)  )   { 
                               out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()   = (  lp_array.array().maxCoeff() +  fast_log_approx_double_wo_checks_Eigen(   fast_exp_double_wo_checks_Eigen_mat(  (lp_array.array() - lp_array.array().maxCoeff() ).array()).matrix().rowwise().sum().array()   )  )  ;
                               prob_n  =  fast_exp_double_wo_checks_Eigen(out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()) ;
                         }

                         
                         // fast_exp_approx_double_wo_checks_Eigen_mat 
                         // fast_exp_approx_double_wo_checks_Eigen 
                         // fast_exp_double_wo_checks_Eigen
                         // fast_exp_double_wo_checks_Eigen_mat

    for (int c = 0; c < n_class; c++) {

                 for (int i = 0; i < n_tests; i++) { // i goes from 1 to 3
                   int t = n_tests - (i+1) ;
                   prop_rowwise_prod_temp.col(t).array()   =   prob[c].block(0, t + 0, chunk_size, i + 1).rowwise().prod().array() ;
                 }

                  prop_rowwise_prod_temp_all.array() =  prob[c].rowwise().prod().array()  ;

                 for (int i = 0; i < n_tests; i++) { // i goes from 1 to 3
                   int t = n_tests - (i + 1) ;
                   common_grad_term_1.col(t) =   (  ( prev(0,c) / prob_n.array() ) * (    prop_rowwise_prod_temp_all.array() /  prop_rowwise_prod_temp.col(t).array()  ).array() )  ;
                 }
                 for (int t = 0; t < n_tests; t++) {
                   L_Omega_diag_recip_array.col(t).array() =  L_Omega_recip_double[c](t, t) ;
                 }

                 
                 for (int t = 0; t < n_tests; t++) {
                       
                       if (Phi_type == "Phi_approx")  {
                             y1_or_phi_Bound_Z[c].array() =                (  (    (  a_times_3*Bound_Z[c].array()*Bound_Z[c].array()   + b  ).array()  ).array() )  *  Bound_U_Phi_Bound_Z[c].array() * (1.0 -  Bound_U_Phi_Bound_Z[c].array() )   ;
                       } else if (Phi_type == "Phi") { 
                          if (exp_fast == true)       y1_or_phi_Bound_Z[c].array() =                   sqrt_2_pi_recip *  fast_exp_double_wo_checks_Eigen_mat( - 0.5 * Bound_Z[c].array()    *  Bound_Z[c].array()    ).array()  ;  
                          else                        y1_or_phi_Bound_Z[c].array() =                   sqrt_2_pi_recip *  ( - 0.5 * Bound_Z[c].array()    *  Bound_Z[c].array()    ).array().exp()  ;
                       }
                       
                                 // for all configs
                                 if (abs_indicator(t, c) == 1) {
                                    // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
                                  for (int n_index = 0; n_index < chunk_size; n_index++) { // can't vectorise ?!
                                    if ( abs_Bound_Z[c](n_index, t) > 5.0   ) {
                                      double Phi_x_i =  Bound_U_Phi_Bound_Z[c](n_index, t);
                                      double Phi_x_1m_Phi  =    (  Phi_x_i * (1.0 - Phi_x_i)  ) ;
                                      y1_or_phi_Bound_Z[c](n_index, t) =  1.702 * Phi_x_1m_Phi;
                                    }
                                  }
                                 }
                    }
                
                     
 
            y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.array()  =                  ( y_chunk.array()  + (  y_chunk.array() - 1.0).array() ).array() *    y1_or_phi_Bound_Z[c].array()  *   L_Omega_diag_recip_array.array() ;

                          if (Phi_type == "Phi_approx")  {
                                         phi_Z_recip.array()  =    1.0 / (    (   (a_times_3*Z_std_norm[c].array()*Z_std_norm[c].array()   + b  ).array()  ).array() *  Phi_Z[c].array() * (1.0 -  Phi_Z[c].array() )  ).array()  ;  // Phi_type == 2
                          } else  if (Phi_type == "Phi")  {
                                if (exp_fast == true)      phi_Z_recip.array() =                 1.0  / (   sqrt_2_pi_recip *  fast_exp_double_wo_checks_Eigen_mat( - 0.5 * Z_std_norm[c].array()    *  Z_std_norm[c].array()    ).array() )  ; 
                                else                       phi_Z_recip.array() =                 1.0  / (   sqrt_2_pi_recip *  ( - 0.5 * Z_std_norm[c].array()    *  Z_std_norm[c].array()    ).array().exp() ).array()  ;
                          }
                          // for all configs
                                for (int t = 0; t < n_tests; t++) {
                                  if (abs_indicator(t, c) == 1) {
                                       // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
                                        for (int n_index = 0; n_index < chunk_size; n_index++) { // can't vectorise ?!
                                          if ( abs_Bound_Z[c](n_index, t) > 5.0   ) {
                                            double Phi_x_i =  Phi_Z[c](n_index, t);
                                            double Phi_x_1m_Phi_x_recip =  1.0 / (  Phi_x_i * (1.0 - Phi_x_i)  ) ;
                                            Phi_x_1m_Phi_x_recip =  1.0 / (  Phi_x_i * (1.0 - Phi_x_i)  ) ;
                                            phi_Z_recip(n_index, t) =      s * Phi_x_1m_Phi_x_recip;
                                          }
                                        }
                                  }
                                }
                       


             y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.array()  =    ( (  (   y_chunk.array()   -  ( y_chunk.array()  + (  y_chunk.array()  - 1.0).array() ).array()    * u_array.array()    ).array() ) ).array() *
                                                                                                        phi_Z_recip.array()  *   y1_or_phi_Bound_Z[c].array()   *    L_Omega_diag_recip_array.array() ;




           ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of nuisance parameters / u's (manual)
           {

                 ///// then second-to-last term (test T - 1)
                 int t = n_tests - 1;

                 u_grad_array_CM_chunk.col(n_tests - 2).array()  +=  (  common_grad_term_1.col(t).array()  * (y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()  *  L_Omega_double[c](t,t - 1) * ( phi_Z_recip.col(t-1).array() )  *  prob[c].col(t-1).array()) ).array()  ;

                   { ///// then third-to-last term (test T - 2)
                     t = n_tests - 2;

                     z_grad_term.col(0) = ( phi_Z_recip.col(t-1).array())  *  prob[c].col(t-1).array() ;
                     grad_prob.col(0) =        y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   *     L_Omega_double[c](t, t - 1) *   z_grad_term.col(0).array() ; // lp(T-1) - part 2;
                     z_grad_term.col(1).array()  =      L_Omega_double[c](t,t-1) *   z_grad_term.col(0).array() *       y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array() ;
                     grad_prob.col(1)  =         (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t+1).array()   ) *  (  z_grad_term.col(0).array() *  L_Omega_double[c](t + 1,t - 1)  -   z_grad_term.col(1).array()  * L_Omega_double[c](t+1,t)) ;

                    u_grad_array_CM_chunk.col(n_tests - 3).array()  +=  ( common_grad_term_1.col(t).array()   *  (  grad_prob.col(1).array() *  prob[c].col(t).array()  +      grad_prob.col(0).array() *   prob[c].col(t+1).array()  )  )   ;
                   }

                   // then rest of terms
                   for (int i = 1; i < n_tests - 2; i++) { // i goes from 1 to 3

                             grad_prob.array()   = 0.0;
                             z_grad_term.array() = 0.0;

                             int t = n_tests - (i+2) ; // starts at t = 6 - (1+2) = 3, ends at t = 6 - (3+2) = 6 - 5 = 1 (when i = 3)

                             z_grad_term.col(0) = (  phi_Z_recip.col(t-1).array())  *  prob[c].col(t-1).array() ;
                             grad_prob.col(0) =         y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   *   L_Omega_double[c](t, t - 1) *   z_grad_term.col(0).array() ; // lp(T-1) - part 2;

                             for (int ii = 0; ii < i+1; ii++) { // if i = 1, ii goes from 0 to 1 u_grad_z u_grad_term
                            //   if (ii == 0)    prod_container_or_inc_array  = (   (z_grad_term.block(0, 0, chunk_size, ii + 1) ) *  (fn_first_element_neg_rest_pos_colvec(L_Omega_double[c].col( t + (ii-1) + 1).segment(t - 1, ii + 1)))  )      ;
                               if (ii == 0)    prod_container_or_inc_array  = (   (z_grad_term.topLeftCorner(chunk_size, ii + 1) ) *  (fn_first_element_neg_rest_pos(L_Omega_double[c].row( t + (ii-1) + 1).segment(t - 1, ii + 1))).transpose()  )      ;
                               z_grad_term.col(ii+1)  =           y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii).array() *  -   prod_container_or_inc_array.array() ;
                             //  prod_container_or_inc_array  = (   (z_grad_term.block(0, 0, chunk_size, ii + 2) ) *  (fn_first_element_neg_rest_pos_colvec(L_Omega_double[c].col( t + (ii) + 1).segment(t - 1, ii + 2)))   )      ;
                               prod_container_or_inc_array  = (   (z_grad_term.topLeftCorner(chunk_size, ii + 2) ) *  (fn_first_element_neg_rest_pos(L_Omega_double[c].row( t + (ii) + 1).segment(t - 1, ii + 2))).transpose()  )      ;
                               grad_prob.col(ii+1)  =       (    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t+ii+1).array()  ) *     -    prod_container_or_inc_array.array()  ;
                             } // end of ii loop

                             {
                               derivs_chain_container_vec.array() = 0.0;

                               for (int ii = 0; ii < i + 2; ii++) {
                                 derivs_chain_container_vec.array()  +=  ( grad_prob.col(ii).array()    * (       prop_rowwise_prod_temp.col(t).array() /   prob[c].col(t + ii).array()  ).array() ).array()  ;
                               }
                             u_grad_array_CM_chunk.col(n_tests - (i+3)).array()    +=   (  ( (   common_grad_term_1.col(t).array()   *  derivs_chain_container_vec.array() ) ).array()  ).array() ;
                             }

                   }


                    out_mat.segment(1, n_us).segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests)  =  u_grad_array_CM_chunk.reshaped() ; //   u_grad_array_CM.block(chunk_size * chunk_counter, 0, chunk_size, n_tests).reshaped() ; // .cast<float>()     ;         }

             }

             // /////////////////////////////////////////////////////////////////////////// Grad of intercepts / coefficients (beta's)
             // ///// last term first (test T)

             {

               int t = n_tests - 1;

               beta_grad_array(c, t) +=     (common_grad_term_1.col(t).array()  *   (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()    )).sum();

               ///// then second-to-last term (test T - 1)
               {
                 t = n_tests - 2;
                 grad_prob.col(0) =       (     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   ) ;
                 z_grad_term.col(0)   =     - y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array()         ;
                 grad_prob.col(1)  =       (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t+1).array()    )   * (   L_Omega_double[c](t + 1,t) *      z_grad_term.col(0).array() ) ;
                 beta_grad_array(c, t) +=  (common_grad_term_1.col(t).array()   * ( grad_prob.col(1).array() *  prob[c].col(t).array() +         grad_prob.col(0).array() *   prob[c].col(t+1).array() ) ).sum() ;
               }

               // then rest of terms
               for (int i = 1; i < n_tests - 1; i++) { // i goes from 1 to 3

                 t = n_tests - (i+2) ; // starts at t = 6 - (1+2) = 3, ends at t = 6 - (3+2) = 6 - 5 = 1 (when i = 3)

                 grad_prob.col(0)  =     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   ;

                 // component 2 (second-to-earliest test)
                 z_grad_term.col(0)  =        -y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array()       ;
                 grad_prob.col(1) =        y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 1).array()    *    (   L_Omega_double[c](t + 1,t) *   z_grad_term.col(0).array() ) ;

                 // rest of components
                 for (int ii = 1; ii < i+1; ii++) { // if i = 1, ii goes from 0 to 1
                           // if (ii == 1)  prod_container_or_inc_array  = (    z_grad_term.block(0, 1, chunk_size, ii)  *   L_Omega_double[c].col( t + (ii - 1) + 1).segment(t + 0, ii + 0)  );
                           //  if (ii == 1)  prod_container_or_inc_array  = (    z_grad_term.block(0, 1, chunk_size, ii)  *   L_Omega_double[c].row( t + (ii - 1) + 1).segment(t + 0, ii + 0).transpose()  );
                            if (ii == 1)  prod_container_or_inc_array  = (    z_grad_term.topLeftCorner(chunk_size, ii)  *   L_Omega_double[c].row( t + (ii - 1) + 1).segment(t + 0, ii + 0).transpose()  );
                           z_grad_term.col(ii)  =        -y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii).array()    *   prod_container_or_inc_array.array();
                        //    prod_container_or_inc_array  = (    z_grad_term.block(0, 1, chunk_size, ii + 1)  *   L_Omega_double[c].col( t + (ii) + 1).segment(t + 0, ii + 1)   );
                            prod_container_or_inc_array  = (    z_grad_term.topLeftCorner(chunk_size, ii + 1)  *   L_Omega_double[c].row( t + (ii) + 1).segment(t + 0, ii + 1).transpose()  );
                         //  prod_container_or_inc_array  = (    z_grad_term.block(0, 1, chunk_size, ii + 1)  *   L_Omega_double[c].row( t + (ii) + 1).segment(t + 0, ii + 1).transpose()  );
                         grad_prob.col(ii + 1) =      (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + ii + 1).array()  ).array()     *  prod_container_or_inc_array.array();
                 }

                 {
                   derivs_chain_container_vec.array() = 0.0;

                   ///// attempt at vectorising  // bookmark
                   for (int ii = 0; ii < i + 2; ii++) {
                     derivs_chain_container_vec.array() +=  ( grad_prob.col(ii).array()  * (      prop_rowwise_prod_temp.col(t).array() /   prob[c].col(t + ii).array()  ).array() ).array() ;
                   }
                   beta_grad_array(c, t) +=        ( common_grad_term_1.col(t).array()   *  derivs_chain_container_vec.array() ).sum();
                 }

               }

             }



             ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of L_Omega ('s)
             {
               {
                 ///////////////////////// deriv of diagonal elements (not needed if using the "standard" or "Stan" Cholesky parameterisation of Omega)

                 //////// w.r.t last diagonal first
                 {
                   int  t1 = n_tests - 1;

                   U_Omega_grad_array[c](t1, t1) +=   ( common_grad_term_1.col(t1).array()   *   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()  *   Bound_Z[c].col(t1).array()       ).sum() ;
                 }


                 //////// then w.r.t the second-to-last diagonal
                 int  t1 = n_tests - 2;
                 grad_prob.col(0).array()  =         y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array() *     Bound_Z[c].col(t1).array()     ;     // correct  (standard form)
                 z_grad_term.col(0).array()  =      (   - y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1).array()  )    *  Bound_Z[c].col(t1).array()    ;  // correct

                 prod_container_or_inc_array.array()  =   (  L_Omega_double[c](t1 + 1, t1)    *   z_grad_term.col(0).array()   ) ; // sequence
                 grad_prob.col(1).array()  =   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + 1).array()   *          prod_container_or_inc_array.array()      ;    // correct   (standard form)

                 U_Omega_grad_array[c](t1, t1) +=   ( (   common_grad_term_1.col(t1).array() )     *    (  prob[c].col(t1 + 1).array()  *   grad_prob.col(0).array()  +    prob[c].col(t1).array()  *      grad_prob.col(1).array()   )  ).sum()   ;

               }

               // //////// then w.r.t the third-to-last diagonal .... etc
               {

                 for (int i = 3; i < n_tests + 1; i++) {

                   int  t1 = n_tests - i;

                   //////// 1st component
                   // 1st grad_Z term and 1st grad_prob term (simplest terms)
                   grad_prob.col(0).array()  =   (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array() )  *  ( Bound_Z[c].col(t1).array()   ).array()  ; // correct  (standard form)
                   z_grad_term.col(0).array()  =    ( -y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1).array()  )  *  Bound_Z[c].col(t1).array()       ;   // correct  (standard form)

                   // 2nd   grad_Z term and 2nd grad_prob  (more complicated than 1st term)
                   prod_container_or_inc_array.array() =    L_Omega_double[c](t1 + 1, t1)   * z_grad_term.col(0).array()  ; // correct  (standard form)
                   grad_prob.col(1).array()  =   (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + 1).array()  )   *  (    prod_container_or_inc_array.array() ).array()  ; // correct  (standard form)


                   for (int ii = 1; ii < i - 1; ii++) {
                     z_grad_term.col(ii).array()  =     (- y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1 + ii).array()  )    *   prod_container_or_inc_array.array()   ;   // correct  (standard form)       // grad_z term
                     ////////// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    //  prod_container_or_inc_array.matrix() =   (  L_Omega_double[c].row(t1 + ii + 1).segment(t1, ii + 1) *   z_grad_term.block(0, 0, chunk_size, ii + 1).transpose() ).transpose().matrix(); ////////// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                     prod_container_or_inc_array.matrix() =   (  L_Omega_double[c].row(t1 + ii + 1).segment(t1, ii + 1) *   z_grad_term.topLeftCorner(chunk_size, ii + 1).transpose() ).transpose().matrix(); // correct  (standard form)
                     ////////// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                     grad_prob.col(ii + 1).array()  =    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + ii + 1).array()    *    prod_container_or_inc_array.array()  ;  // correct  (standard form)     //    grad_prob term
                   }

                   {

                     derivs_chain_container_vec.array() = 0.0;

                     ///// attempt at vectorising  // bookmark
                     for (int iii = 0; iii <  i; iii++) {
                       derivs_chain_container_vec.array()  +=    grad_prob.col(iii).array()  * (       prop_rowwise_prod_temp.col(t1).array()    /   prob[c].col(t1 + iii).array()  ).array()  ;  // correct  (standard form)
                     }

                     U_Omega_grad_array[c](t1, t1)   +=       ( common_grad_term_1.col(t1).array()   * derivs_chain_container_vec.array() ).sum()  ; // correct  (standard form)
                   }
                 }
               }
             }

             {
               z_grad_term.array() = 0.0 ;
               grad_prob.array() = 0.0;

               { ///////////////////// last row first
                 int t1_dash = 0;  // t1 = n_tests - 1

                 int t1 = n_tests - (t1_dash + 1); //  starts at n_tests - 1;  // if t1_dash = 0 -> t1 = T - 1
                 int t2 = n_tests - (t1_dash + 2); //  starts at n_tests - 2;

                 U_Omega_grad_array[c](t1, t2) +=        (  common_grad_term_1.col(t1).array()      *  (  - y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()   )     *  ( -  Z_std_norm[c].col(t2).array()  )  ).sum() ;

                 if (t1 > 1) { // starts at  L_{T, T-2}
                   {
                     t2 =   n_tests - (t1_dash + 3); // starts at n_tests - 3;
                     U_Omega_grad_array[c](t1, t2) +=     (  common_grad_term_1.col(t1).array()  *   (  - y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()    )   *     (  - Z_std_norm[c].col(t2).array()   ) ).sum()  ;
                   }
                 }

                 if (t1 > 2) {// starts at  L_{T, T-3}
                   for (int t2_dash = 3; t2_dash < n_tests; t2_dash++ ) { // t2 < t1
                     t2 = n_tests - (t1_dash + t2_dash + 1); // starts at T - 4
                     if (t2 < n_tests - 1) {
                       U_Omega_grad_array[c](t1, t2)  +=   (  common_grad_term_1.col(t1).array() *   (  - y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()    )  *   (      - Z_std_norm[c].col(t2).array()  )  ).sum() ;
                     }
                   }
                 }
               }
             }



             {
               /////////////////// then rest of rows (second-to-last row, then third-to-last row, .... , then first row)
               for (int t1_dash = 1; t1_dash <  n_tests - 1;  t1_dash++) {
                 int  t1 = n_tests - (t1_dash + 1);

                 for (int t2_dash = t1_dash + 1; t2_dash <  n_tests;  t2_dash++) {
                   int t2 = n_tests - (t2_dash + 1); // starts at t1 - 1, then t1 - 2, up to 0


                   {


                     //prod_container_or_inc_array.array()  =  Z_std_norm[c].block(0, t2, chunk_size, t1 - t2) * deriv_L_t1.head(t1 - t2) ;
                     prod_container_or_inc_array.array()  =  Z_std_norm[c].col(t2) ; // block(0, t2, chunk_size, t1 - t2) * deriv_L_t1.head(t1 - t2) ;
                     grad_prob.col(0) =       y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()   *           prod_container_or_inc_array.array()   ;
                     z_grad_term.col(0).array()  =                 y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1).array()     *   (   -    prod_container_or_inc_array.array()     )      ;

                     if (t1_dash > 0) {
                       for (int t1_dash_dash = 1; t1_dash_dash <  t1_dash + 1;  t1_dash_dash++) {
                         if (t1_dash_dash > 1) {
                           z_grad_term.col(t1_dash_dash - 1)   =           y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1 + t1_dash_dash - 1).array()   *  ( - prod_container_or_inc_array.array() )    ;
                         }
                       //   prod_container_or_inc_array.array()  =            (   z_grad_term.block(0, 0, chunk_size, t1_dash_dash) *   L_Omega_double[c].col(t1 + t1_dash_dash).segment(t1, t1_dash_dash)    ) ;
                         prod_container_or_inc_array.array()  =            (   z_grad_term.topLeftCorner(chunk_size, t1_dash_dash) *   L_Omega_double[c].row(t1 + t1_dash_dash).segment(t1, t1_dash_dash).transpose()   ) ;
                         // prod_container_or_inc_array.array()  =            (   z_grad_term.block(0, 0, chunk_size, t1_dash_dash) *   L_Omega_double[c].row(t1 + t1_dash_dash).segment(t1, t1_dash_dash).transpose()   ) ;
                         grad_prob.col(t1_dash_dash)  =             y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + t1_dash_dash).array()    *      prod_container_or_inc_array.array()  ;
                       }
                     }

                     {

                       derivs_chain_container_vec.array() = 0.0;

                       ///// attempt at vectorising  // bookmark
                       for (int ii = 0; ii <  t1_dash + 1; ii++) {
                         derivs_chain_container_vec.array() += ( grad_prob.col(ii).array()  * ( prop_rowwise_prod_temp.col(t1).array()     /   prob[c].col(t1 + ii).array()  ).array() ).array() ; // correct i think
                       }
                       U_Omega_grad_array[c](t1, t2)   +=       ( common_grad_term_1.col(t1).array()   * derivs_chain_container_vec.array() ).sum()  ; // correct  (standard form)

                     }



                   }
                 }
               }

             }

             prev_grad_vec(c)  +=  ( ( 1.0 / prob_n.array() ) *  prob[c].rowwise().prod().array() ).matrix().sum()  ;

          }

        }

     }





     //////////////////////// gradients for latent class membership probabilitie(s) (i.e. disease prevalence)
     for (int c = 0; c < n_class; c++) {
       prev_unconstrained_grad_vec(c)  =   prev_grad_vec(c)   * deriv_p_wrt_pu_double ;
     }
     prev_unconstrained_grad_vec(0) = prev_unconstrained_grad_vec(1) - prev_unconstrained_grad_vec(0) - 2 * tanh_u_prev[1];
     prev_unconstrained_grad_vec_out(0) = prev_unconstrained_grad_vec(0);


      // log_prob_out += log_lik.sum();
       log_prob_out += out_mat.segment(1 + n_params, N).sum();

     if (exclude_priors == false)  log_prob_out += prior_densities;

     log_prob_out +=  log_jac_u;

     log_prob = (double) log_prob_out;

     int i = 0; // probs_all_range.prod() cancels out
     for (int c = 0; c < n_class; c++) {
       for (int t = 0; t < n_tests; t++) {
         if (exclude_priors == false) {
           beta_grad_array(c, t) +=  - ((beta_double_array(c,t) - prior_coeffs_mean(c, t)) / prior_coeffs_sd(c, t) ) * (1.0/ prior_coeffs_sd(c, t) ) ;     // add normal prior density derivative to gradient
         }
         beta_grad_vec(i) = beta_grad_array(c, t);
         i += 1;
       }
     }


     {
       int i = 0;
       for (int c = 0; c < n_class; c++ ) {
         for (int t1 = 0; t1 < n_tests  ; t1++ ) {
          for (int t2 = 0; t2 <  t1 + 1; t2++ ) {
             L_Omega_grad_vec(i) = U_Omega_grad_array[c](t1,t2);
             i += 1;
           }
         }
       }
     }



     Eigen::Matrix<double, -1, 1>  grad_wrt_L_Omega_nd =   L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
     Eigen::Matrix<double, -1, 1>  grad_wrt_L_Omega_d =   L_Omega_grad_vec.segment(dim_choose_2 + n_tests, dim_choose_2 + n_tests);

     U_Omega_grad_vec.segment(0, dim_choose_2) =  ( grad_wrt_L_Omega_nd.transpose()  *  deriv_L_wrt_unc_full[0].cast<double>() ).transpose() ;
     U_Omega_grad_vec.segment(dim_choose_2, dim_choose_2) =   ( grad_wrt_L_Omega_d.transpose()  *  deriv_L_wrt_unc_full[1].cast<double>() ).transpose()  ;


   }




   {

   ////////////////////////////  outputs // add log grad and sign stuff';///////////////
   out_mat(0) =  log_prob;
   out_mat.segment(1 + n_us, n_corrs) = target_AD_grad ;          // .cast<float>();
   out_mat.segment(1 + n_us, n_corrs) += U_Omega_grad_vec ;        //.cast<float>()  ;
   out_mat.segment(1 + n_us + n_corrs, n_coeffs) = beta_grad_vec ; //.cast<float>() ;
   out_mat(n_params) = ((grad_prev_AD +  prev_unconstrained_grad_vec_out(0)));
   out_mat.segment(1, n_us).array() =     (  out_mat.segment(1, n_us).array() *  ( 0.5 * (1.0 - theta_us.array() * theta_us.array()  )  )   ).array()    - 2.0 * theta_us.array()   ;

   }

   return(out_mat);


 }



 

  


 
 
 
 
 
 
 
 
 
 // [[Rcpp::export]]
 Eigen::Matrix<double, -1, 1 >    fn_lp_and_grad_MVP_using_Chol_Spinkney_MD_and_AD(      Eigen::Matrix<double, -1, 1  > theta_main,
                                                                                         Eigen::Matrix<double, -1, 1  > theta_us,
                                                                                         Eigen::Matrix<int, -1, -1>	 y,
                                                                                         std::vector<Eigen::Matrix<double, -1, -1 > >  X,
                                                                                         Rcpp::List other_args
 ) { 
   
   
   
   
   
   const int n_cores = other_args(0); 
   const bool exclude_priors = other_args(1); 
   // const bool CI = other_args(2); // CI makes no sense for non-LCM MVP model??!!
   Eigen::Matrix<double, -1, 1>  lkj_cholesky_eta = other_args(3); 
   Eigen::Matrix<double, -1, -1> prior_coeffs_mean  = other_args(4); 
   Eigen::Matrix<double, -1, -1> prior_coeffs_sd = other_args(5); 
   const int n_class = 1;
   const int ub_threshold_phi_approx = other_args(7); 
   const int n_chunks = other_args(8); 
   const bool corr_force_positive = other_args(9); 
   std::vector<Eigen::Matrix<double, -1, -1 > >  prior_for_corr_a = other_args(10); 
   std::vector<Eigen::Matrix<double, -1, -1 > >  prior_for_corr_b = other_args(11); 
   const bool corr_prior_beta  = other_args(12); 
   const bool corr_prior_norm  = other_args(13); 
   std::vector<Eigen::Matrix<double, -1, -1 > >  lb_corr = other_args(14); 
   std::vector<Eigen::Matrix<double, -1, -1 > >  ub_corr = other_args(15); 
   std::vector<Eigen::Matrix<int, -1, -1 > >      known_values_indicator = other_args(16); 
   std::vector<Eigen::Matrix<double, -1, -1 > >   known_values = other_args(17); 
   // const double prev_prior_a = other_args(18); 
   // const double prev_prior_b = other_args(19); 
   const bool exp_fast = other_args(20); 
   const bool log_fast = other_args(21); 
   std::string Phi_type = other_args(22); 
    
   const int n_tests = y.cols();
   const int N = y.rows();
   const int n_corrs =   n_tests * (n_tests - 1) * 0.5;
   const int n_covariates = X.size() ;   ////////////////////////////  
   const int n_coeffs =  n_tests * n_covariates;
   const int n_us =    N * n_tests; 
   
   const int n_params = theta_us.rows() +  theta_main.rows()   ; // n_corrs + n_coeffs + n_us + 1;
   const int n_params_main = n_params - n_us;
   
   const double sqrt_2_pi_recip =   1.0 / sqrt(2.0 * M_PI) ; //  0.3989422804;
   const double sqrt_2_recip = 1.0 / stan::math::sqrt(2.0);
   const double minus_sqrt_2_recip =  - sqrt_2_recip;
   const double a = 0.07056;
   const double b = 1.5976;
   const double a_times_3 = 3.0 * 0.07056;
   const double s = 1.0/1.702;
   
    
   // corrs
   Eigen::Matrix<double, -1, 1  >  Omega_raw_vec_double = theta_main.head(n_corrs); // .cast<double>();
   Eigen::Matrix<stan::math::var, -1, 1  >  Omega_raw_vec_var =  stan::math::to_var(Omega_raw_vec_double) ;
   Eigen::Matrix<stan::math::var, -1, 1  >  Omega_constrained_raw_vec_var =  Eigen::Matrix<stan::math::var, -1, 1  >::Zero(n_corrs) ;
   Omega_constrained_raw_vec_var = Omega_raw_vec_var ; // no transformation for Nump needed! done later on
   
    
   // coeffs
   Eigen::Matrix<double, -1, -1> beta_double_array(1, n_coeffs);
    
   {
     int i = n_corrs;
       for (int k = 0; k < n_coeffs; ++k) {
         beta_double_array(0, k) = theta_main(i);
         i += 1;
       }
   }
     
   
    
   Eigen::Matrix<double, -1, 1 >  target_AD_grad(n_corrs);
   stan::math::var target_AD = 0.0;
    
   int dim_choose_2 = n_tests * (n_tests - 1) * 0.5 ;
   std::vector<Eigen::Matrix<double, -1, -1 > > deriv_L_wrt_unc_full = vec_of_mats_test(dim_choose_2 + n_tests, dim_choose_2, 1);
   std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_double = vec_of_mats_test(n_tests, n_tests, 1);
   std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_recip_double = L_Omega_double ; 
    
   
   
   {
     
     
     std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > Omega_unconstrained_var = fn_convert_std_vec_of_corrs_to_3d_array_var( Eigen_vec_to_std_vec_var(Omega_constrained_raw_vec_var),
                                                                                                                                  n_tests,
                                                                                                                                  1);
     
     
     std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  L_Omega_var = vec_of_mats_test_var(n_tests, n_tests, 1);
     std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_var   = vec_of_mats_test_var(n_tests, n_tests, 1);
     
     {
       Eigen::Matrix<stan::math::var, -1, -1 >  ub = stan::math::to_var(ub_corr[0]);
       Eigen::Matrix<stan::math::var, -1, -1 >  lb = stan::math::to_var(lb_corr[0]);
       
       Eigen::Matrix<stan::math::var, -1, -1  >  Chol_Schur_outs =  Spinkney_LDL_bounds_opt(n_tests, lb, ub, Omega_unconstrained_var[0], known_values_indicator[0], known_values[0]) ; //   Omega_unconstrained_var[0], n_tests, tol )  ;
       
       L_Omega_var[0]   =  Chol_Schur_outs.block(1, 0, n_tests, n_tests);
       Omega_var[0] =   L_Omega_var[0] * L_Omega_var[0].transpose() ;
       
       
       target_AD +=   Chol_Schur_outs(0, 0); // now can set prior directly on Omega
     } 
     
     
     
      {
       if ( (corr_prior_beta == false)   &&  (corr_prior_norm == false) ) {
         target_AD +=  stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[0], lkj_cholesky_eta(0)) ; 
       } else if ( (corr_prior_beta == true)   &&  (corr_prior_norm == false) ) {
         for (int i = 1; i < n_tests; i++) {
           for (int j = 0; j < i; j++) {
             target_AD +=  stan::math::beta_lpdf(  (Omega_var[0](i, j) + 1)/2, prior_for_corr_a[0](i, j), prior_for_corr_b[0](i, j));
           } 
         }
         //  Jacobian for  Omega -> L_Omega transformation for prior log-densities (since both LKJ and truncated normal prior densities are in terms of Omega, not L_Omega)
         Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
         for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i+1) ) * log(L_Omega_var[0](i, i));
         target_AD  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
       } else if  ( (corr_prior_beta == false)   &&  (corr_prior_norm == true) ) { 
         for (int i = 1; i < n_tests; i++) {
           for (int j = 0; j < i; j++) {
             target_AD +=  stan::math::normal_lpdf(  Omega_var[0](i, j), prior_for_corr_a[0](i, j), prior_for_corr_b[0](i, j));
           } 
         }
         Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
         for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i+1) ) * log(L_Omega_var[0](i, i));
         target_AD  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
       } 
     }
     
     
     ///////////////////////
     stan::math::set_zero_all_adjoints();
     target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
     target_AD_grad =  Omega_raw_vec_var.adj();    // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
     stan::math::set_zero_all_adjoints();
     //////////////////////////////////////////////////////////// end of AD part
     
      {
       int cnt_1 = 0; 
       for (int k = 0; k < n_tests; k++) {
         for (int l = 0; l < k + 1; l++) {
           (  L_Omega_var[0](k, l)).grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
           int cnt_2 = 0;
           for (int i = 1; i < n_tests; i++) {
             for (int j = 0; j < i; j++) {
               deriv_L_wrt_unc_full[0](cnt_1, cnt_2)  =   Omega_unconstrained_var[0](i, j).adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
               cnt_2 += 1;
             }
           }
           stan::math::set_zero_all_adjoints();
           cnt_1 += 1;
         }
       }
     }
     
     ///////////////// get cholesky factor's (lower-triangular) of corr matrices
     // convert to 3d var array
      {
       for (int t1 = 0; t1 < n_tests; ++t1) {
         for (int t2 = 0; t2 < n_tests; ++t2) {
           L_Omega_double[0](t1, t2) =   L_Omega_var[0](t1, t2).val()  ;
           L_Omega_recip_double[0](t1, t2) =   1.0 / L_Omega_double[0](t1, t2) ;
         }
       }
     }
     
     stan::math::recover_memory();
   }
   
   
   ///////////////////////////////////////////////////////////////////////// prior densities
   double prior_densities = 0.0;
   
   
   if (exclude_priors == false) {
     ///////////////////// priors for coeffs
     double prior_densities_coeffs = 0.0;
     {
       for (int t = 0; t < n_coeffs; t++) {
         prior_densities_coeffs  += stan::math::normal_lpdf(beta_double_array(0, t), prior_coeffs_mean(0, t), prior_coeffs_sd(0, t));
       }
     }
     double prior_densities_corrs = target_AD.val();
     prior_densities = prior_densities_coeffs  +      prior_densities_corrs ;     // total prior densities and Jacobian adjustments
   }
   
   
   /////////////////////////////////////////////////////////////////////////////////////////////////////
   ///////// likelihood
   int chunk_counter = 0;
   int chunk_size  = std::round( N / n_chunks  / 2) * 2;  ; // N / n_chunks;
   
   
   double log_prob_out = 0.0;
   
   
   if (exp_fast == true)   theta_us.array() =      fast_tanh_approx_Eigen( theta_us ).array(); 
   else                    theta_us.array() =      ( theta_us ).array().tanh(); 
   
   double log_jac_u  =  0.0;
   if    (log_fast == true)    {  // most stable
     log_jac_u  =    (  fast_log_approx_double_wo_checks_Eigen( 0.5 *   (  1.0 -  ( theta_us.array()   *  theta_us.array()   ).array()  ).array()  ).array()   ).matrix().sum();  // log
   } else {
     log_jac_u  =    (  ( 0.5 *   (  1.0 -  ( theta_us.array()   *  theta_us.array()   ).array()  ).array()  ).array().log()   ).matrix().sum();  // log
   }
   
   
   ///////////////////////////////////////////////
   Eigen::Matrix<double , -1, 1>   beta_grad_vec   =  Eigen::Matrix<double, -1, 1>::Zero(n_coeffs);  //
   Eigen::Matrix<double, -1, -1>   beta_grad_array  =  Eigen::Matrix<double, -1, -1>::Zero(1, n_coeffs); //
   std::vector<Eigen::Matrix<double, -1, -1 > > U_Omega_grad_array =  vec_of_mats_test(n_tests, n_tests, 1); //
   Eigen::Matrix<double, -1, 1 > L_Omega_grad_vec(n_corrs + (1 * n_tests)); //
   Eigen::Matrix<double, -1, 1 > U_Omega_grad_vec(n_corrs); //
   // ///////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////// output vec
   Eigen::Matrix<double, -1, 1> out_mat    =  Eigen::Matrix<double, -1, 1>::Zero(n_params + 1 + N);  ///////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////////////
   
   double log_prob = 0.0;
   
   {
     
     
     //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
     ///////////////////////////////////////////////
     std::vector<Eigen::Matrix<double, -1, -1 > >  Z_std_norm =  vec_of_mats_test(chunk_size, n_tests, 1); //////////////////////////////
     std::vector<Eigen::Matrix<double, -1, -1 > >  Bound_Z =  Z_std_norm ;  
     std::vector<Eigen::Matrix<double, -1, -1 > >  abs_Bound_Z =  Z_std_norm ;  
     std::vector<Eigen::Matrix<double, -1, -1 > >  Bound_U_Phi_Bound_Z =  Z_std_norm ;   
     std::vector<Eigen::Matrix<double, -1, -1 > >  Phi_Z =  Z_std_norm ;  
     std::vector<Eigen::Matrix<double, -1, -1 > >  y1_or_phi_Bound_Z =  Z_std_norm ;  
     std::vector<Eigen::Matrix<double, -1, -1 > >  prob =  Z_std_norm ;  
     ///////////////////////////////////////////////);
     Eigen::Array<double, -1, -1> y_chunk = Eigen::Array<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Array<double, -1, -1> u_array =  y_chunk ;  
     ///////////////////////////////////////////////
     Eigen::Matrix<double, -1, 1>      prod_container_or_inc_array  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
     Eigen::Matrix<double, -1, 1>      prob_n  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
     ///////////////////////////////////////////////
     Eigen::Matrix<double, -1, -1 >    L_Omega_diag_recip_array   = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Matrix<double, -1, -1 >    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip   =  L_Omega_diag_recip_array ; //   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Matrix<double, -1, -1 >    y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip   =   L_Omega_diag_recip_array ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Matrix<double, -1, -1 >    prop_rowwise_prod_temp   =   L_Omega_diag_recip_array ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Matrix<double, -1, -1>     grad_prob =    L_Omega_diag_recip_array ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Matrix<double, -1, -1>     z_grad_term =  L_Omega_diag_recip_array ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     Eigen::Matrix<double, -1, -1>     u_grad_array_CM_chunk   =   L_Omega_diag_recip_array ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);    ////////
     Eigen::Matrix<double, -1, -1>     phi_Z_recip  =   L_Omega_diag_recip_array ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
     ///////////////////////////////////////////////
     Eigen::Matrix<double, -1, 1>      derivs_chain_container_vec  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
     Eigen::Matrix<double, -1, 1>      prop_rowwise_prod_temp_all  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
     Eigen::Array<int, -1, -1>         abs_indicator = Eigen::Array<int, -1, -1>::Zero(n_tests, n_class);
     ////////////////////////////////////////////////
     
     
     {
       
       // /////////////////////////////////////////////////////////////////////////////////////////////////////
       Eigen::Matrix<double, -1, -1>    lp_array  =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, 1);///////
       ////////////////////////////////////////////////////////////////////////////////////////////////////////
       
       
       for (int nc = 0; nc < n_chunks; nc++) {
         
         abs_indicator.array() = 0;
         
         u_grad_array_CM_chunk.array() = 0.0;
         
         int chunk_counter = nc;
         
         y_chunk = y.middleRows(chunk_size * chunk_counter , chunk_size).array().cast<double>() ;
         
         u_array  = 0.5 * (  theta_us.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests).reshaped(chunk_size, n_tests).array() + 1.0 ).array() ;
         
         
        {
                       
                       prod_container_or_inc_array.array()  = 0.0; // needs to be reset to 0
                       
                       for (int t = 0; t < n_tests; t++) { 
                         
                                     Bound_Z[0].col(t).array() =    L_Omega_recip_double[0](t, t) * (  - ( beta_double_array(0, t) +      prod_container_or_inc_array.array()   )  ) ; // / L_Omega_double[0](t, t)     ;
                                     abs_Bound_Z[0].col(t).array() =    Bound_Z[0].col(t).array().abs(); 
                                     if ((abs_Bound_Z[0].col(t).array() > 5.0).matrix().sum() > 0)     abs_indicator(t, 0) = 1;
                                     
                                     if (Phi_type == "Phi")   Bound_U_Phi_Bound_Z[0].col(t).array() =  0.5 *   ( minus_sqrt_2_recip * Bound_Z[0].col(t).array() ).erfc() ;
                                     else   Bound_U_Phi_Bound_Z[0].col(t).array() = Phi_approx_fast_Eigen(Bound_Z[0].col(t).array() );
                                     
                                     // for all configs 
                                     if (  abs_indicator(t, 0) == 1) {
                                       // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
                                       for (int n_index = 0; n_index < chunk_size; n_index++) { // can't vectorise ?!
                                         if ( abs_Bound_Z[0](n_index, t) > 5.0   ) {
                                           Bound_U_Phi_Bound_Z[0](n_index, t) =    stan::math::inv_logit( 1.702 *  Bound_Z[0](n_index, t)  );  //  stan::math::inv_logit( 1.702 * Bound_Z[0].col(t)  )
                                         }
                                       }
                                     }
                                     
                                     Phi_Z[0].col(t).array()    =      (   y_chunk.col(t)   *  Bound_U_Phi_Bound_Z[0].col(t).array() +
                                       (   y_chunk.col(t)   -  Bound_U_Phi_Bound_Z[0].col(t).array() ) *   ( y_chunk.col(t)  + (  y_chunk.col(t)  - 1.0)  )  * u_array.col(t)   )   ;  
                                     
                                     if (Phi_type == "Phi") {    
                                       if (log_fast == false) Z_std_norm[0].col(t).array() = stan::math::inv_Phi(Phi_Z[0].col(t)).array() ; //    qnorm_rcpp_Eigen( Phi_Z[0].col(t).array());   
                                       else                   Z_std_norm[0].col(t).array() = qnorm_w_fast_log_rcpp_Eigen( Phi_Z[0].col(t).array());  // with approximations
                                     } else if (Phi_type == "Phi_approx") {
                                       Z_std_norm[0].col(t).array() =  inv_Phi_approx_fast_Eigen(Phi_Z[0].col(t).array());
                                     }
                                     
                                     // for all configs 
                                     if (abs_indicator(t,0) == 1) {
                                       // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
                                       for (int n_index = 0; n_index < chunk_size; n_index++) { // can't vectoriose ?!
                                         if ( abs_Bound_Z[0](n_index, t) > 5.0   ) {
                                           Z_std_norm[0](n_index, t) =    s  *  stan::math::logit(Phi_Z[0](n_index, t));  //   s *  fast_logit_1_Eigen(    Phi_Z[0].col(t) ).array()
                                         }
                                       }
                                     } 
                                     
                                     if (t < n_tests - 1)       prod_container_or_inc_array.array()  =   ( Z_std_norm[0].topLeftCorner(chunk_size, t + 1)  *   ( L_Omega_double[0].row(t+1).head(t+1).transpose()  ) ) ;      // for all configs 
                                     
                       } // end of t loop
                       
                       prob[0].array() =  y_chunk.array() * ( 1.0 -    Bound_U_Phi_Bound_Z[0].array() ) +  ( y_chunk - 1.0  )   *    Bound_U_Phi_Bound_Z[0].array() *   ( y_chunk +  (  y_chunk - 1.0)  )  ;   // for all configs 
                       
                       if (log_fast == true)  y1_or_phi_Bound_Z[0].array()  =   fast_log_approx_double_wo_checks_Eigen_mat(prob[0].array() ) ;
                       else                   y1_or_phi_Bound_Z[0].array()  =   prob[0].array().log();
                       
                       lp_array.col(0).array() =     y1_or_phi_Bound_Z[0].rowwise().sum().array() ;
                       
         }  
         
         
         out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()   =    lp_array.col(0).sum(); 
        
         if    ( (log_fast == false) && (exp_fast == false)  )  {  
           prob_n  =  (out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()).exp() ;
         } else if  ( (log_fast == true) && (exp_fast == false)  )  {  
           prob_n  =  (out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()).exp() ;
         } else if  ( (log_fast == false) && (exp_fast == true)  )  { 
           prob_n  =  fast_exp_double_wo_checks_Eigen(out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()) ;
         } else if  ( (log_fast == true) && (exp_fast == true)  )   { 
           prob_n  =  fast_exp_double_wo_checks_Eigen(out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()) ;
         }
         
         
         // fast_exp_approx_double_wo_checks_Eigen_mat 
         // fast_exp_approx_double_wo_checks_Eigen 
         // fast_exp_double_wo_checks_Eigen
         // fast_exp_double_wo_checks_Eigen_mat
         
         {
           
           
           for (int t = 0; t < n_tests; t++) {
             L_Omega_diag_recip_array.col(t).array() =  L_Omega_recip_double[0](t, t) ;
           }
           
           
           for (int t = 0; t < n_tests; t++) {
             
             if (Phi_type == "Phi_approx")  {
               y1_or_phi_Bound_Z[0].array() =                (  (    (  a_times_3*Bound_Z[0].array()*Bound_Z[0].array()   + b  ).array()  ).array() )  *  Bound_U_Phi_Bound_Z[0].array() * (1.0 -  Bound_U_Phi_Bound_Z[0].array() )   ;
             } else if (Phi_type == "Phi") { 
               if (exp_fast == true)       y1_or_phi_Bound_Z[0].array() =                   sqrt_2_pi_recip *  fast_exp_double_wo_checks_Eigen_mat( - 0.5 * Bound_Z[0].array()    *  Bound_Z[0].array()    ).array()  ;  
               else                        y1_or_phi_Bound_Z[0].array() =                   sqrt_2_pi_recip *  ( - 0.5 * Bound_Z[0].array()    *  Bound_Z[0].array()    ).array().exp()  ;
             }
             
             // for all configs
             if (abs_indicator(t, 0) == 1) {
               // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
               for (int n_index = 0; n_index < chunk_size; n_index++) { // can't vectorise ?!
                 if ( abs_Bound_Z[0](n_index, t) > 5.0   ) {
                   double Phi_x_i =  Bound_U_Phi_Bound_Z[0](n_index, t);
                   double Phi_x_1m_Phi  =    (  Phi_x_i * (1.0 - Phi_x_i)  ) ;
                   y1_or_phi_Bound_Z[0](n_index, t) =  1.702 * Phi_x_1m_Phi;
                 }
               }
             }
           }
           
           y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.array()  =                  ( y_chunk.array()  + (  y_chunk.array() - 1.0).array() ).array() *    y1_or_phi_Bound_Z[0].array()  *   L_Omega_diag_recip_array.array() ;
           
           if (Phi_type == "Phi_approx")  {
             phi_Z_recip.array()  =    1.0 / (    (   (a_times_3*Z_std_norm[0].array()*Z_std_norm[0].array()   + b  ).array()  ).array() *  Phi_Z[0].array() * (1.0 -  Phi_Z[0].array() )  ).array()  ;  // Phi_type == 2
           } else  if (Phi_type == "Phi")  {
             if (exp_fast == true)      phi_Z_recip.array() =                 1.0  / (   sqrt_2_pi_recip *  fast_exp_double_wo_checks_Eigen_mat( - 0.5 * Z_std_norm[0].array()    *  Z_std_norm[0].array()    ).array() )  ; 
             else                       phi_Z_recip.array() =                 1.0  / (   sqrt_2_pi_recip *  ( - 0.5 * Z_std_norm[0].array()    *  Z_std_norm[0].array()    ).array().exp() ).array()  ;
           }
           // for all configs
           for (int t = 0; t < n_tests; t++) {
             if (abs_indicator(t, 0) == 1) {
               // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
               for (int n_index = 0; n_index < chunk_size; n_index++) { // can't vectorise ?!
                 if ( abs_Bound_Z[0](n_index, t) > 5.0   ) {
                   double Phi_x_i =  Phi_Z[0](n_index, t);
                   double Phi_x_1m_Phi_x_recip =  1.0 / (  Phi_x_i * (1.0 - Phi_x_i)  ) ;
                   Phi_x_1m_Phi_x_recip =  1.0 / (  Phi_x_i * (1.0 - Phi_x_i)  ) ;
                   phi_Z_recip(n_index, t) =      s * Phi_x_1m_Phi_x_recip;
                 }
               }
             }
           }
           
           y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.array()  =    ( (  (   y_chunk.array()   -  ( y_chunk.array()  + (  y_chunk.array()  - 1.0).array() ).array()    * u_array.array()    ).array() ) ).array() *
             phi_Z_recip.array()  *   y1_or_phi_Bound_Z[0].array()   *    L_Omega_diag_recip_array.array() ;
           
           
           ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of nuisance parameters / u's (manual)
             {
               
               ///// then second-to-last term (test T - 1)
               int t = n_tests - 1;
               
               u_grad_array_CM_chunk.col(n_tests - 2).array()  +=  (   (y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()  *  L_Omega_double[0](t,t - 1) * ( phi_Z_recip.col(t-1).array() )  *  prob[0].col(t-1).array()) ).array()  ;
               
               { ///// then third-to-last term (test T - 2)
                 t = n_tests - 2;
                 
                 z_grad_term.col(0) = ( phi_Z_recip.col(t-1).array())  *  prob[0].col(t-1).array() ;
                 grad_prob.col(0) =        y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   *     L_Omega_double[0](t, t - 1) *   z_grad_term.col(0).array() ; // lp(T-1) - part 2;
                 z_grad_term.col(1).array()  =      L_Omega_double[0](t,t-1) *   z_grad_term.col(0).array() *       y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array() ;
                 grad_prob.col(1)  =         (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t+1).array()   ) *  (  z_grad_term.col(0).array() *  L_Omega_double[0](t + 1,t - 1)  -   z_grad_term.col(1).array()  * L_Omega_double[0](t+1,t)) ;
                 
                 u_grad_array_CM_chunk.col(n_tests - 3).array()  +=  (  (  grad_prob.col(1).array() *  prob[0].col(t).array()  +      grad_prob.col(0).array() *   prob[0].col(t+1).array()  )  )   ;
               }
               
               // then rest of terms
               for (int i = 1; i < n_tests - 2; i++) { // i goes from 1 to 3
                 
                 grad_prob.array()   = 0.0;
                 z_grad_term.array() = 0.0;
                 
                 int t = n_tests - (i+2) ; // starts at t = 6 - (1+2) = 3, ends at t = 6 - (3+2) = 6 - 5 = 1 (when i = 3)
                 
                 z_grad_term.col(0) = (  phi_Z_recip.col(t-1).array())  *  prob[0].col(t-1).array() ;
                 grad_prob.col(0) =         y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   *   L_Omega_double[0](t, t - 1) *   z_grad_term.col(0).array() ; // lp(T-1) - part 2;
                 
                 for (int ii = 0; ii < i+1; ii++) { // if i = 1, ii goes from 0 to 1 u_grad_z u_grad_term     ;
                   if (ii == 0)    prod_container_or_inc_array  = (   (z_grad_term.topLeftCorner(chunk_size, ii + 1) ) *  (fn_first_element_neg_rest_pos(L_Omega_double[0].row( t + (ii-1) + 1).segment(t - 1, ii + 1))).transpose()  )      ;
                   z_grad_term.col(ii+1)  =           y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii).array() *  -   prod_container_or_inc_array.array() ;
                   prod_container_or_inc_array  = (   (z_grad_term.topLeftCorner(chunk_size, ii + 2) ) *  (fn_first_element_neg_rest_pos(L_Omega_double[0].row( t + (ii) + 1).segment(t - 1, ii + 2))).transpose()  )      ;
                   grad_prob.col(ii+1)  =       (    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t+ii+1).array()  ) *     -    prod_container_or_inc_array.array()  ;
                 } // end of ii loop
                 
                 {
                   derivs_chain_container_vec.array() = 0.0;
                   
                   for (int ii = 0; ii < i + 2; ii++) {
                     derivs_chain_container_vec.array()  +=  ( grad_prob.col(ii).array()    * (       prop_rowwise_prod_temp.col(t).array() /   prob[0].col(t + ii).array()  ).array() ).array()  ;
                   }
                   u_grad_array_CM_chunk.col(n_tests - (i+3)).array()    +=   (  ( (    derivs_chain_container_vec.array() ) ).array()  ).array() ;
                 }
                 
               }
               
               
               out_mat.segment(1, n_us).segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests)  =  u_grad_array_CM_chunk.reshaped() ; //   u_grad_array_CM.block(chunk_size * chunk_counter, 0, chunk_size, n_tests).reshaped() ; // .cast<float>()     ;         }
               
             }
             
             // /////////////////////////////////////////////////////////////////////////// Grad of intercepts / coefficients (beta's)
             // ///// last term first (test T)
             
             {
               
               int t = n_tests - 1;
               
               beta_grad_array(0, t) +=     (     (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()    )).sum();
               
               ///// then second-to-last term (test T - 1)
               {
                 t = n_tests - 2;
                 grad_prob.col(0) =       (     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   ) ;
                 z_grad_term.col(0)   =     - y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array()         ;
                 grad_prob.col(1)  =       (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t+1).array()    )   * (   L_Omega_double[0](t + 1,t) *      z_grad_term.col(0).array() ) ;
                 beta_grad_array(0, t) +=  ( ( grad_prob.col(1).array() *  prob[0].col(t).array() +         grad_prob.col(0).array() *   prob[0].col(t+1).array() ) ).sum() ;
               }
               
               // then rest of terms
               for (int i = 1; i < n_tests - 1; i++) { // i goes from 1 to 3
                 
                 t = n_tests - (i+2) ; // starts at t = 6 - (1+2) = 3, ends at t = 6 - (3+2) = 6 - 5 = 1 (when i = 3)
                 
                 grad_prob.col(0)  =     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   ;
                 
                 // component 2 (second-to-earliest test)
                 z_grad_term.col(0)  =        -y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array()       ;
                 grad_prob.col(1) =        y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 1).array()    *    (   L_Omega_double[0](t + 1,t) *   z_grad_term.col(0).array() ) ;
                 
                 // rest of components
                 for (int ii = 1; ii < i+1; ii++) { // if i = 1, ii goes from 0 to 1
                   if (ii == 1)  prod_container_or_inc_array  = (    z_grad_term.topLeftCorner(chunk_size, ii)  *   L_Omega_double[0].row( t + (ii - 1) + 1).segment(t + 0, ii + 0).transpose()  );
                   z_grad_term.col(ii)  =        -y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii).array()    *   prod_container_or_inc_array.array();
                   prod_container_or_inc_array  = (    z_grad_term.topLeftCorner(chunk_size, ii + 1)  *   L_Omega_double[0].row( t + (ii) + 1).segment(t + 0, ii + 1).transpose()  );
                   grad_prob.col(ii + 1) =      (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + ii + 1).array()  ).array()     *  prod_container_or_inc_array.array();
                 }
                 
                 {
                   derivs_chain_container_vec.array() = 0.0;
                   
                   ///// attempt at vectorising  // bookmark
                   for (int ii = 0; ii < i + 2; ii++) {
                     derivs_chain_container_vec.array() +=  ( grad_prob.col(ii).array()  * (      prop_rowwise_prod_temp.col(t).array() /   prob[0].col(t + ii).array()  ).array() ).array() ;
                   }
                   beta_grad_array(0, t) +=        (    derivs_chain_container_vec.array() ).sum();
                 }
                 
               }
               
             }
             
             ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of L_Omega ('s)
             {
               {
                 ///////////////////////// deriv of diagonal elements (not needed if using the "standard" or "Stan" Cholesky parameterisation of Omega)
                 
                 //////// w.r.t last diagonal first
                 {
                   int  t1 = n_tests - 1;
                   
                   U_Omega_grad_array[0](t1, t1) +=   (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()  *   Bound_Z[0].col(t1).array()       ).sum() ;
                 }
                 
                 
                 //////// then w.r.t the second-to-last diagonal
                 int  t1 = n_tests - 2;
                 grad_prob.col(0).array()  =         y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array() *     Bound_Z[0].col(t1).array()     ;     // correct  (standard form)
                 z_grad_term.col(0).array()  =      (   - y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1).array()  )    *  Bound_Z[0].col(t1).array()    ;  // correct
                 
                 prod_container_or_inc_array.array()  =   (  L_Omega_double[0](t1 + 1, t1)    *   z_grad_term.col(0).array()   ) ; // sequence
                 grad_prob.col(1).array()  =   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + 1).array()   *          prod_container_or_inc_array.array()      ;    // correct   (standard form)
                 
                 U_Omega_grad_array[0](t1, t1) +=   (    (  prob[0].col(t1 + 1).array()  *   grad_prob.col(0).array()  +    prob[0].col(t1).array()  *      grad_prob.col(1).array()   )  ).sum()   ;
                 
               }
               
               // //////// then w.r.t the third-to-last diagonal .... etc
               {
                 
                 for (int i = 3; i < n_tests + 1; i++) {
                   
                   int  t1 = n_tests - i;
                   
                   //////// 1st component
                   // 1st grad_Z term and 1st grad_prob term (simplest terms)
                   grad_prob.col(0).array()  =   (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array() )  *  ( Bound_Z[0].col(t1).array()   ).array()  ; // correct  (standard form)
                   z_grad_term.col(0).array()  =    ( -y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1).array()  )  *  Bound_Z[0].col(t1).array()       ;   // correct  (standard form)
                   
                   // 2nd   grad_Z term and 2nd grad_prob  (more complicated than 1st term)
                   prod_container_or_inc_array.array() =    L_Omega_double[0](t1 + 1, t1)   * z_grad_term.col(0).array()  ; // correct  (standard form)
                   grad_prob.col(1).array()  =   (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + 1).array()  )   *  (    prod_container_or_inc_array.array() ).array()  ; // correct  (standard form)
                   
                   
                   for (int ii = 1; ii < i - 1; ii++) {
                     z_grad_term.col(ii).array()  =     (- y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1 + ii).array()  )    *   prod_container_or_inc_array.array()   ;   // correct  (standard form)       // grad_z term
                     prod_container_or_inc_array.matrix() =   (  L_Omega_double[0].row(t1 + ii + 1).segment(t1, ii + 1) *   z_grad_term.topLeftCorner(chunk_size, ii + 1).transpose() ).transpose().matrix(); // correct  (standard form)
                     grad_prob.col(ii + 1).array()  =    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + ii + 1).array()    *    prod_container_or_inc_array.array()  ;  // correct  (standard form)     //    grad_prob term
                   }
                   
                   {
                     
                     derivs_chain_container_vec.array() = 0.0;
                     
                     ///// attempt at vectorising  // bookmark
                     for (int iii = 0; iii <  i; iii++) {
                       derivs_chain_container_vec.array()  +=    grad_prob.col(iii).array()  * (       prop_rowwise_prod_temp.col(t1).array()    /   prob[0].col(t1 + iii).array()  ).array()  ;  // correct  (standard form)
                     }
                     
                     U_Omega_grad_array[0](t1, t1)   +=       (   derivs_chain_container_vec.array() ).sum()  ; // correct  (standard form)
                   }
                 }
               }
             }
             
             {
               z_grad_term.array() = 0.0 ;
               grad_prob.array() = 0.0;
               
               { ///////////////////// last row first
                 int t1_dash = 0;  // t1 = n_tests - 1
                 
                 int t1 = n_tests - (t1_dash + 1); //  starts at n_tests - 1;  // if t1_dash = 0 -> t1 = T - 1
                 int t2 = n_tests - (t1_dash + 2); //  starts at n_tests - 2;
                 
                 U_Omega_grad_array[0](t1, t2) +=        (     (  - y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()   )     *  ( -  Z_std_norm[0].col(t2).array()  )  ).sum() ;
                 
                 if (t1 > 1) { // starts at  L_{T, T-2}
                   {
                     t2 =   n_tests - (t1_dash + 3); // starts at n_tests - 3;
                     U_Omega_grad_array[0](t1, t2) +=     (      (  - y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()    )   *     (  - Z_std_norm[0].col(t2).array()   ) ).sum()  ;
                   }
                 }
                 
                 if (t1 > 2) {// starts at  L_{T, T-3}
                   for (int t2_dash = 3; t2_dash < n_tests; t2_dash++ ) { // t2 < t1
                     t2 = n_tests - (t1_dash + t2_dash + 1); // starts at T - 4
                     if (t2 < n_tests - 1) {
                       U_Omega_grad_array[0](t1, t2)  +=   (      (  - y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()    )  *   (      - Z_std_norm[0].col(t2).array()  )  ).sum() ;
                     }
                   }
                 }
               }
             }
             
             {
               /////////////////// then rest of rows (second-to-last row, then third-to-last row, .... , then first row)
               for (int t1_dash = 1; t1_dash <  n_tests - 1;  t1_dash++) {
                 int  t1 = n_tests - (t1_dash + 1);
                 
                 for (int t2_dash = t1_dash + 1; t2_dash <  n_tests;  t2_dash++) {
                   int t2 = n_tests - (t2_dash + 1); // starts at t1 - 1, then t1 - 2, up to 0
                   
                   {
                     //prod_container_or_inc_array.array()  =  Z_std_norm[0].block(0, t2, chunk_size, t1 - t2) * deriv_L_t1.head(t1 - t2) ;
                     prod_container_or_inc_array.array()  =  Z_std_norm[0].col(t2) ; // block(0, t2, chunk_size, t1 - t2) * deriv_L_t1.head(t1 - t2) ;
                     grad_prob.col(0) =       y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()   *           prod_container_or_inc_array.array()   ;
                     z_grad_term.col(0).array()  =                 y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1).array()     *   (   -    prod_container_or_inc_array.array()     )      ;
                     
                     if (t1_dash > 0) {
                       for (int t1_dash_dash = 1; t1_dash_dash <  t1_dash + 1;  t1_dash_dash++) {
                         if (t1_dash_dash > 1) {
                           z_grad_term.col(t1_dash_dash - 1)   =           y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1 + t1_dash_dash - 1).array()   *  ( - prod_container_or_inc_array.array() )    ;
                         }
                         prod_container_or_inc_array.array()  =            (   z_grad_term.topLeftCorner(chunk_size, t1_dash_dash) *   L_Omega_double[0].row(t1 + t1_dash_dash).segment(t1, t1_dash_dash).transpose()   ) ;
                         grad_prob.col(t1_dash_dash)  =             y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + t1_dash_dash).array()    *      prod_container_or_inc_array.array()  ;
                       }
                     }
                     
                     {
                       
                       derivs_chain_container_vec.array() = 0.0;
                       
                       ///// attempt at vectorising  // bookmark
                       for (int ii = 0; ii <  t1_dash + 1; ii++) {
                         derivs_chain_container_vec.array() += ( grad_prob.col(ii).array()  * ( prop_rowwise_prod_temp.col(t1).array()     /   prob[0].col(t1 + ii).array()  ).array() ).array() ; // correct i think
                       }
                       U_Omega_grad_array[0](t1, t2)   +=       (   derivs_chain_container_vec.array() ).sum()  ; // correct  (standard form)
                       
                     }
                     
                   }
                 }
               }
               
             }
 
             
         }
         
       }
       
     }
     
     
     
     
     
     //////////////////////// gradients for latent class membership probabilitie(s) (i.e. disease prevalence)
     log_prob_out += out_mat.segment(1 + n_params, N).sum();
     if (exclude_priors == false)  log_prob_out += prior_densities;
     log_prob_out +=  log_jac_u;
     log_prob = (double) log_prob_out;
     
     int i = 0; // probs_all_range.prod() cancels out
      {
       for (int t = 0; t < n_tests; t++) {
         if (exclude_priors == false) {
           beta_grad_array(0, t) +=  - ((beta_double_array(0,t) - prior_coeffs_mean(0, t)) / prior_coeffs_sd(0, t) ) * (1.0/ prior_coeffs_sd(0, t) ) ;     // add normal prior density derivative to gradient
         }
         beta_grad_vec(i) = beta_grad_array(0, t);
         i += 1;
       }
     }
     
     
     {
       int i = 0;
       {
         for (int t1 = 0; t1 < n_tests  ; t1++ ) {
           for (int t2 = 0; t2 <  t1 + 1; t2++ ) {
             L_Omega_grad_vec(i) = U_Omega_grad_array[0](t1,t2);
             i += 1;
           }
         }
       }
     }
     

     Eigen::Matrix<double, -1, 1>  grad_wrt_L_Omega  =   L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
     U_Omega_grad_vec.segment(0, dim_choose_2) =  ( grad_wrt_L_Omega.transpose()  *  deriv_L_wrt_unc_full[0].cast<double>() ).transpose() ;

     
   }
   
 
   {
     ////////////////////////////  outputs // add log grad and sign stuff';///////////////
     out_mat(0) =  log_prob;
     out_mat.segment(1, n_us).array() =     (  out_mat.segment(1, n_us).array() *  ( 0.5 * (1.0 - theta_us.array() * theta_us.array()  )  )   ).array()    - 2.0 * theta_us.array()   ;
     out_mat.segment(1 + n_us, n_corrs) = target_AD_grad ;          // .cast<float>();
     out_mat.segment(1 + n_us, n_corrs) += U_Omega_grad_vec ;        //.cast<float>()  ;
     out_mat.segment(1 + n_us + n_corrs, n_coeffs) = beta_grad_vec ; //.cast<float>() ;
   }
   
   return(out_mat);
   
   
 }







 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 // [[Rcpp::export]]
 Eigen::Matrix<double, -1, 1 >        fn_lp_and_grad_latent_trait_MD_and_AD(  Eigen::Matrix<double, -1, 1  > theta_main,
                                                                              Eigen::Matrix<double, -1, 1  > theta_us,
                                                                              Eigen::Matrix< int, -1, -1>	 y,
                                                                              std::vector<Eigen::Matrix<double, -1, -1 > >  X,
                                                                              Rcpp::List other_args
 ) { 
   
   
   
   
   
   const int n_cores = other_args(0); 
   const bool exclude_priors = other_args(1); 
   const bool CI = other_args(2); 
   Eigen::Matrix<double, -1, 1>  lkj_cholesky_eta = other_args(3); 
   Eigen::Matrix<double, -1, -1> prior_coeffs_mean  = other_args(4); 
   Eigen::Matrix<double, -1, -1> prior_coeffs_sd = other_args(5); 
   const int n_class = other_args(6); 
   const int ub_threshold_phi_approx = other_args(7); 
   const int n_chunks = other_args(8); 
   const bool corr_force_positive = other_args(9); 
   std::vector<Eigen::Matrix<double, -1, -1 > >  prior_for_corr_a = other_args(10); 
   std::vector<Eigen::Matrix<double, -1, -1 > >  prior_for_corr_b = other_args(11); 
   const bool corr_prior_beta  = other_args(12); 
   const bool corr_prior_norm  = other_args(13); 
   std::vector<Eigen::Matrix<double, -1, -1 > >  lb_corr = other_args(14); 
   std::vector<Eigen::Matrix<double, -1, -1 > >  ub_corr = other_args(15); 
   std::vector<Eigen::Matrix<int, -1, -1 > >      known_values_indicator = other_args(16); 
   std::vector<Eigen::Matrix<double, -1, -1 > >   known_values = other_args(17); 
   const double prev_prior_a = other_args(18); 
   const double prev_prior_b = other_args(19); 
   const bool exp_fast = other_args(20); 
   const bool log_fast = other_args(21); 
   std::string Phi_type = other_args(22); 
   Eigen::Matrix<double, -1, -1> LT_b_priors_shape  = other_args(23); 
   Eigen::Matrix<double, -1, -1> LT_b_priors_scale  = other_args(24); 
   Eigen::Matrix<double, -1, -1> LT_known_bs_indicator = other_args(25); 
   Eigen::Matrix<double, -1, -1> LT_known_bs_values  = other_args(26); 
    
   
   const int n_tests = y.cols();
   const int N = y.rows();
   const int n_corrs =  n_class * n_tests * (n_tests - 1) * 0.5;
   const int n_coeffs = n_class * n_tests * 1;
   const int n_us =  1 *  N * n_tests;
   const int n_params = theta_us.rows() +  theta_main.rows()   ; // n_corrs + n_coeffs + n_us + n_class;
   const int n_params_main = n_params - n_us;
   
   const int n_bs_LT = n_class * n_tests;
   
   const double sqrt_2_pi_recip =   1.0 / sqrt(2.0 * M_PI) ; //  0.3989422804;
   const double sqrt_2_recip = 1.0 / stan::math::sqrt(2.0);
   const double minus_sqrt_2_recip =  - sqrt_2_recip;
   const double a = 0.07056;
   const double b = 1.5976;
   const double a_times_3 = 3.0 * 0.07056;
   const double s = 1.0/1.702;
   
   
   double prior_densities = 0.0;
   
    
   // corrs / b's
   Eigen::Matrix<double, -1, 1  >  bs_raw_vec_double = theta_main.segment(0, n_bs_LT) ;
   Eigen::Matrix<stan::math::var, -1, 1  >  bs_raw_vec_var =  stan::math::to_var(bs_raw_vec_double) ;
   Eigen::Matrix<stan::math::var, -1, -1 > bs_mat =    Eigen::Matrix<stan::math::var, -1, -1 >::Zero(n_class, n_tests);
   Eigen::Matrix<double, -1, -1 > bs_mat_double =    Eigen::Matrix<double, -1, -1 >::Zero(n_class, n_tests);
   Eigen::Matrix<stan::math::var, -1, -1 > bs_raw_mat =  Eigen::Matrix<stan::math::var, -1, -1 >::Zero(n_class, n_tests);
   
    
   
   bs_raw_mat.row(0) =  bs_raw_vec_var.segment(0, n_tests).transpose();
   bs_raw_mat.row(1) =  bs_raw_vec_var.segment(n_tests, n_tests).transpose();
    
   bs_mat.row(0) = stan::math::exp( bs_raw_mat.row(0)) ;
   bs_mat.row(1) = stan::math::exp( bs_raw_mat.row(1)) ;
    
   stan::math::var known_bs_raw_sum = 0.0;
   
    
   Eigen::Matrix<stan::math::var, -1, 1 > bs_nd  =   bs_mat.row(0).transpose() ; //  bs_constrained_raw_vec_var.head(n_tests);
   Eigen::Matrix<stan::math::var, -1, 1 > bs_d   =   bs_mat.row(1).transpose() ; //  bs_constrained_raw_vec_var.segment(n_tests, n_tests);
    
   
   // coeffs
   Eigen::Matrix<stan::math::var, -1, -1  > LT_theta(n_class, n_tests);
   Eigen::Matrix<stan::math::var, -1, -1  > LT_a(n_class, n_tests);
    
   Eigen::Matrix<double, -1, 1  > coeffs_vec_double(n_coeffs);
   Eigen::Matrix<stan::math::var, -1, 1  > coeffs_vec_var(n_coeffs);
    
   coeffs_vec_double = theta_main.segment(0 + n_corrs, n_coeffs);
   coeffs_vec_var = stan::math::to_var(coeffs_vec_double); 
   
   {
     int i = 0 ; // 0 + n_corrs;
     for (int c = 0; c < n_class; ++c) {
       for (int t = 0; t < n_tests; ++t) {
         LT_a(c, t) = coeffs_vec_var(i);
         bs_mat_double(c, t) = bs_mat(c, t).val();
         i = i + 1;
       }
     }
   } 
   
   //// LT_theta as TRANSFORMED parameter (need Jacobian adj. if wish to put prior on theta!!!)
   for (int t = 0; t < n_tests; ++t) {
     LT_theta(1, t)   =    LT_a(1, t) /  stan::math::sqrt(1 + ( bs_d(t) * bs_d(t)));
     LT_theta(0, t)   =    LT_a(0, t) /  stan::math::sqrt(1 + ( bs_nd(t) * bs_nd(t)));
   }
    
   
   // prev
   double u_prev_diseased = theta_main(n_params_main - 1);
   
    
   
   ///////////////////////////////////////////////////////////////////////////////////////// output vec
   Eigen::Matrix<double, -1, 1> out_mat    =  Eigen::Matrix<double, -1, 1>::Zero(n_params + 1 + N);  ///////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////////////
   
   
    
   ////////////////////////////////////// AD part  -   for non-LKJ corr priors
   int n_choose_2 = n_tests * (n_tests - 1) * 0.5 ;
   std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  L_Omega_var = vec_of_mats_test_var(n_tests, n_tests, n_class);
   std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_var   = vec_of_mats_test_var(n_tests, n_tests, n_class);
   
   
   Eigen::Matrix<stan::math::var, -1, -1 > identity_dim_T =     Eigen::Matrix<stan::math::var, -1, -1 > ::Zero(n_tests, n_tests) ; //  stan::math::diag_matrix(  stan::math::rep_vector(1, n_tests)  ) ;
   
    
   Eigen::Matrix<double, -1, 1 >   bs_d_double(n_tests);
   Eigen::Matrix<double, -1, 1 >   bs_nd_double(n_tests);
    
   for (int i = 0; i < n_tests; ++i) {
     identity_dim_T(i, i) = 1.0;
     bs_d_double(i) = bs_d(i).val() ;
     bs_nd_double(i) = bs_nd(i).val() ;
   }
    
   
   Omega_var[0] = identity_dim_T +  bs_nd * bs_nd.transpose();
   Omega_var[1] = identity_dim_T +  bs_d * bs_d.transpose();
    
   
   double grad_prev_AD = 0.0;
   Eigen::Matrix<double, -1, 1 >  target_AD_grad(n_corrs);
   stan::math::var target_AD = 0.0;
   
    
   for (int c = 0; c < n_class; ++c) {
     L_Omega_var[c]   = stan::math::cholesky_decompose( Omega_var[c]) ;
   }
    
   
   //////////////// Jacobian L_Sigma -> b's
   std::vector< std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > > Jacobian_d_L_Sigma_wrt_b_3d_arrays_var = vec_of_vec_of_mats_test_var(n_tests, n_tests, n_tests, n_class);
   std::vector< std::vector<Eigen::Matrix<double, -1, -1 > > > Jacobian_d_L_Sigma_wrt_b_3d_arrays_double = vec_of_vec_of_mats_test(n_tests, n_tests, n_tests, n_class);
   std::vector<Eigen::Matrix<double, -1, -1 > >  Jacobian_d_L_Sigma_wrt_b_matrix = vec_of_mats_test(n_choose_2 + n_tests, n_tests, n_class);
    
   for (int c = 0; c < n_class; ++c) {
     
     //  # -----------  wrt last b first
     int t = n_tests;
     stan::math::var sum_sq_1 = 0.0;
     for (int j = 1; j < t; ++j) {
       Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, j-1) = ( L_Omega_var[c](n_tests-1, j-1) / bs_mat(c, n_tests-1) ) ;//* bs_nd(n_tests-1) ;
       sum_sq_1 +=   bs_mat(c, j-1) * bs_mat(c, j-1) ;
     }
     stan::math::var big_denom_p1 =  1 + sum_sq_1;
     Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, n_tests-1) =   (1 / L_Omega_var[c](n_tests-1, n_tests-1) ) * ( bs_mat(c, n_tests-1) / big_denom_p1 ) ;//* bs_nd(n_tests-1) ;
     
     //  # -----------  wrt 2nd-to-last b  
     t = n_tests - 1;
     sum_sq_1 = 0;
     stan::math::var  sum_sq_2 = 0.0;
     for (int j = 1; j < t + 1; ++j) {
       Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, j-1) = ( L_Omega_var[c](t-1, j-1) / bs_mat(c, t-1) );// * bs_nd(t-1) ;
       sum_sq_1 +=   bs_mat(c, j-1) * bs_mat(c, j-1) ;
       if (j < (t))   sum_sq_2 +=  bs_mat(c, j-1) * bs_mat(c, j-1) ;
     }
     big_denom_p1 =  1 + sum_sq_1;
     stan::math::var big_denom_p2 =  1 + sum_sq_2;
     stan::math::var  big_denom_part =  big_denom_p1 * big_denom_p2;
     Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, t-1) =   (1 / L_Omega_var[c](t-1, t-1)) * ( bs_mat(c, t-1) / big_denom_p2 );// * bs_nd(t-1) ;
     
     for (int j = t+1; j < n_tests + 1; ++j) {
       Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](j-1, t-1) =   ( 1/L_Omega_var[c](j-1, t-1) ) * (bs_mat(c, j-1) *  bs_mat(c, j-1)  ) * (   bs_mat(c, t-1)  / big_denom_part) * (1 - ( bs_mat(c, t-1) * bs_mat(c, t-1)  / big_denom_p1 ) );// * bs_nd(t-1)   ;
     }
     
     Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t, t)   =  - ( 1/L_Omega_var[c](t, t) ) * (bs_mat(c, t) * bs_mat(c, t)) * ( bs_mat(c, t-1)  / (big_denom_p1*big_denom_p1));//*  bs_nd(t-1) ;
     
     // # -----------  wrt rest of b's
     for (int t = 1; t < (n_tests - 2) + 1; ++t) {
       
       sum_sq_1  = 0;
       sum_sq_2  = 0;
       
       for (int j = 1; j < t + 1; ++j) {
         if (j < (t)) Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, j-1) = ( L_Omega_var[c](t-1, j-1) /  bs_mat(c, t-1) ) ;//* ;// bs_nd(t-1) ;
         sum_sq_1 +=   bs_mat(c, j-1) *   bs_mat(c, j-1) ;
         if (j < (t))   sum_sq_2 +=    bs_mat(c, j-1) *   bs_mat(c, j-1) ;
       }
       big_denom_p1 = 1 + sum_sq_1;
       big_denom_p2 = 1 + sum_sq_2;
       big_denom_part =  big_denom_p1 * big_denom_p2;
       
       Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, t-1) =   (1 / L_Omega_var[c](t-1, t-1) ) * (  bs_mat(c, t-1) / big_denom_p2 ) ;//*  bs_nd(t-1) ;
       
       for (int j = t + 1; j < n_tests + 1; ++j) {
         Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](j-1, t-1)  =   (1/L_Omega_var[c](j-1, t-1)) * (  bs_mat(c, j-1) *   bs_mat(c, j-1) ) * (   bs_mat(c, t-1) / big_denom_part) * (1 - ( ( bs_mat(c, t-1) *  bs_mat(c, t-1) ) / big_denom_p1 ) ) ;//*  bs_nd(t-1) ;
       }
       
       for (int j = t + 1; j < n_tests ; ++j) {
         Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](j-1, j-1) =  - (1/L_Omega_var[c](j-1, j-1)) * (  bs_mat(c, j-1) *   bs_mat(c, j-1) ) * ( bs_mat(c, t-1) / (big_denom_p1*big_denom_p1)) ;//*  bs_nd(t-1) ;
         big_denom_p1 = big_denom_p1 +   bs_mat(c, j-1) *   bs_mat(c, j-1) ;
         big_denom_p2 = big_denom_p2 + bs_mat(c, j-2) * bs_mat(c, j-2) ;
         big_denom_part =  big_denom_p1 * big_denom_p2 ;
         if (t < n_tests - 1) {
           for (int k = j + 1; k < n_tests + 1; ++k) {
             Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](k-1, j-1) =   (-1 / L_Omega_var[c](k-1, j-1)) * (  bs_mat(c, k-1) *   bs_mat(c, k-1) ) * (  bs_mat(c, j-1) *   bs_mat(c, j-1) ) * (  bs_mat(c, t-1) / big_denom_part ) * ( ( 1 / big_denom_p2 )  +  ( 1 / big_denom_p1 ) ) ;//*  bs_nd(t-1) ;
           }
         }
       }
       
       Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](n_tests-1, n_tests-1) =  - (1/L_Omega_var[c](n_tests-1, n_tests-1)) * (bs_mat(c, n_tests-1) * bs_mat(c, n_tests-1)) * ( bs_mat(c, t-1) / (big_denom_p1*big_denom_p1)) ;//*  bs_nd(t-1) ;
       
     }
     
     
     
     for (int t1 = 0; t1 < n_tests; ++t1) {
       for (int t2 = 0; t2 < n_tests; ++t2) {
         for (int t3 = 0; t3 < n_tests; ++t3) {
           Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t2, t3)    =      Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t1](t2, t3).val();
         }
       }
     }
   }
   
   
   ////////////////////// priors for corr
   for (int t = 0; t < n_tests; ++t) {
     target_AD += stan::math::weibull_lpdf(  bs_nd(t) ,   LT_b_priors_shape(0, t), LT_b_priors_scale(0, t)  );
     target_AD += stan::math::weibull_lpdf(  bs_d(t)  ,   LT_b_priors_shape(1, t), LT_b_priors_scale(1, t)  );
   }
   
   target_AD +=  (bs_raw_mat).sum()  - known_bs_raw_sum ; // Jacobian b -> raw_b
   
   /// priors and Jacobians for coeffs
   for (int c = 0; c < n_class; ++c) {
     for (int t = 0; t < n_tests; ++t) {
       target_AD += stan::math::normal_lpdf(LT_theta(c, t), prior_coeffs_mean(c, t), prior_coeffs_sd(c, t));
       target_AD +=  - 0.5 * stan::math::log(1 + stan::math::square(stan::math::abs(bs_mat(c, t) ))); // Jacobian for LT_theta -> LT_a
     }
   }
   
   
   /////////////  prev stuff  ---- vars
   std::vector<stan::math::var> 	 u_prev_var_vec_var(n_class, 0.0);
   std::vector<stan::math::var> 	 prev_var_vec_var(n_class, 0.0);
   std::vector<stan::math::var> 	 tanh_u_prev_var(n_class, 0.0);
   Eigen::Matrix<stan::math::var, -1, -1>	 prev_var(1, n_class);
   
   u_prev_var_vec_var[1] =  stan::math::to_var(u_prev_diseased);
   tanh_u_prev_var[1] = ( exp(2*u_prev_var_vec_var[1] ) - 1) / ( exp(2*u_prev_var_vec_var[1] ) + 1) ;
   u_prev_var_vec_var[0] =   0.5 *  log( (1 + ( (1 - 0.5 * ( tanh_u_prev_var[1] + 1))*2 - 1) ) / (1 - ( (1 - 0.5 * ( tanh_u_prev_var[1] + 1))*2 - 1) ) )  ;
   tanh_u_prev_var[0] = (exp(2*u_prev_var_vec_var[0] ) - 1) / ( exp(2*u_prev_var_vec_var[0] ) + 1) ;
   
   prev_var_vec_var[1] = 0.5 * ( tanh_u_prev_var[1] + 1);
   prev_var_vec_var[0] =  0.5 * ( tanh_u_prev_var[0] + 1);
   prev_var(0,1) =  prev_var_vec_var[1];
   prev_var(0,0) =  prev_var_vec_var[0];
   
   stan::math::var tanh_pu_deriv_var = ( 1 - tanh_u_prev_var[1] * tanh_u_prev_var[1]  );
   stan::math::var deriv_p_wrt_pu_var = 0.5 *  tanh_pu_deriv_var;
   stan::math::var tanh_pu_second_deriv_var  = -2 * tanh_u_prev_var[1]  * tanh_pu_deriv_var;
   stan::math::var log_jac_p_deriv_wrt_pu_var  = ( 1 / deriv_p_wrt_pu_var) * 0.5 * tanh_pu_second_deriv_var; // for gradient of u's
   stan::math::var  log_jac_p_var =    log( deriv_p_wrt_pu_var );
   
   
   // stan::math::var  target_AD_prev = beta_lpdf(  prev_var(0,1), prev_prior_a, prev_prior_b  ); // weakly informative prior - helps avoid boundaries with slight negative skew (for lower N)
   // target_AD_prev += log_jac_p_var;
   // target_AD  +=  target_AD_prev;
   
   
   target_AD += beta_lpdf(  prev_var(0,1), prev_prior_a, prev_prior_b  ); // weakly informative prior - helps avoid boundaries with slight negative skew (for lower N)
   target_AD += log_jac_p_var;
   
   
   
   prior_densities += target_AD.val() ; // target_AD_coeffs.val() + target_AD_corrs.val();
   
   //  ///////////////////////
   stan::math::set_zero_all_adjoints();
   target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
   out_mat.segment(1 + n_us, n_bs_LT) = bs_raw_vec_var.adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
   stan::math::set_zero_all_adjoints();
   //////////////////////////////////////////////////////////// end of AD part
   
   
   //  ///////////////////////
   stan::math::set_zero_all_adjoints();
   target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
   out_mat.segment(1 + n_us + n_corrs, n_coeffs)  = coeffs_vec_var.adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
   stan::math::set_zero_all_adjoints();
   //////////////////////////////////////////////////////////// end of AD part
   
   
   
   ///////////////////////
   stan::math::set_zero_all_adjoints();
   target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
   grad_prev_AD  =  u_prev_var_vec_var[1].adj() - u_prev_var_vec_var[0].adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
   stan::math::set_zero_all_adjoints();
   //////////////////////////////////////////////////////////// end of AD part
   
   
   
   
   ///////////////// get cholesky factor's (lower-triangular) of corr matrices
   // convert to 3d var array
   std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_double = vec_of_mats_test(n_tests, n_tests, n_class);
   std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_recip_double = vec_of_mats_test(n_tests, n_tests, n_class);
   
   for (int c = 0; c < n_class; ++c) {
     for (int t1 = 0; t1 < n_tests; ++t1) {
       for (int t2 = 0; t2 < n_tests; ++t2) {
         L_Omega_double[c](t1, t2) =   L_Omega_var[c](t1, t2).val();
         L_Omega_recip_double[c](t1, t2) =   1.0 / L_Omega_double[c](t1, t2) ;
       }
     }
   }
   
   
   stan::math::recover_memory();
   
   
   /////////////  prev stuff
   std::vector<double> 	 u_prev_var_vec(n_class, 0.0);
   std::vector<double> 	 prev_var_vec(n_class, 0.0);
   std::vector<double> 	 tanh_u_prev(n_class, 0.0);
   Eigen::Matrix<double, -1, -1>	 prev(1, n_class);
   
   u_prev_var_vec[1] =  (double) u_prev_diseased ;
   tanh_u_prev[1] = ( exp(2.0*u_prev_var_vec[1] ) - 1.0) / ( exp(2.0*u_prev_var_vec[1] ) + 1.0) ;
   u_prev_var_vec[0] =   0.5 *  log( (1 + ( (1 - 0.5 * ( tanh_u_prev[1] + 1))*2.0 - 1.0) ) / (1.0 - ( (1.0 - 0.5 * ( tanh_u_prev[1] + 1.0))*2.0 - 1.0) ) )  ;
   tanh_u_prev[0] = (exp(2.0*u_prev_var_vec[0] ) - 1.0) / ( exp(2.0*u_prev_var_vec[0] ) + 1.0) ;
   
   prev_var_vec[1] =  0.5 * ( tanh_u_prev[1] + 1.0);
   prev_var_vec[0] =  0.5 * ( tanh_u_prev[0] + 1.0);
   prev(0,1) =  prev_var_vec[1];
   prev(0,0) =  prev_var_vec[0];
   
   
   double tanh_pu_deriv = ( 1.0 - tanh_u_prev[1] * tanh_u_prev[1]  );
   double deriv_p_wrt_pu_double = 0.5 *  tanh_pu_deriv;
   double tanh_pu_second_deriv  = -2.0 * tanh_u_prev[1]  * tanh_pu_deriv;
   double log_jac_p_deriv_wrt_pu  = ( 1.0 / deriv_p_wrt_pu_double) * 0.5 * tanh_pu_second_deriv; // for gradient of u's
   
   
   
   
   
   /////////////////////////////////////////////////////////////////////////////////////////////////////
   ///////// likelihood
   int chunk_counter = 0;
   int chunk_size  = std::round( N / n_chunks  / 2) * 2;  ; // N / n_chunks;
   
   
   double log_prob_out = 0.0;
   
   
   if (exp_fast == true)   theta_us.array() =      fast_tanh_approx_Eigen( theta_us ).array(); 
   else                    theta_us.array() =      ( theta_us ).array().tanh(); 
   
   
   double log_jac_u  =  0.0;
   if    (log_fast == true)   log_jac_u  =    (  fast_log_approx_double_wo_checks_Eigen( 0.5 *   (  1.0 -  ( theta_us.array()   *  theta_us.array()   ).array()  ).array()  ).array()   ).matrix().sum(); 
   else  log_jac_u  =    (  ( 0.5 *   (  1.0 -  ( theta_us.array()   *  theta_us.array()   ).array()  ).array()  ).array().log()   ).matrix().sum();   
   
   
   
   ///////////////////////////////////////////////
   Eigen::Matrix<double, -1, 1>   beta_grad_vec   =  Eigen::Matrix<double, -1, 1>::Zero(n_coeffs);  //
   Eigen::Matrix<double, -1, -1>  beta_grad_array  =  Eigen::Matrix<double, -1, -1>::Zero(2, n_tests); //
   Eigen::Matrix<double, -1, 1>  prev_unconstrained_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(2); //
   Eigen::Matrix<double, -1, 1>  prev_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(2); //
   Eigen::Matrix<double, -1, 1>  prev_unconstrained_grad_vec_out =   Eigen::Matrix<double, -1, 1>::Zero(2 - 1); //
   // ///////////////////////////////////////////////
   
   
   double log_prob = 0.0;
   
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   ///////////////////////////////////////////////
   std::vector<Eigen::Matrix<double, -1, -1 > >  Z_std_norm =  vec_of_mats_test(chunk_size, n_tests, 2); //////////////////////////////
   std::vector<Eigen::Matrix<double, -1, -1 > >  Bound_Z =  Z_std_norm ;
   std::vector<Eigen::Matrix<double, -1, -1 > >  abs_Bound_Z =  Z_std_norm ;
   std::vector<Eigen::Matrix<double, -1, -1 > >  Bound_U_Phi_Bound_Z =  Z_std_norm ;  //  vec_of_mats_test(chunk_size, n_tests, 2); //////////////////////////////
   std::vector<Eigen::Matrix<double, -1, -1 > >  Phi_Z =  Z_std_norm ;  //  vec_of_mats_test(chunk_size, n_tests, 2); //////////////////////////////
   std::vector<Eigen::Matrix<double, -1, -1 > >  y1_or_phi_Bound_Z =  Z_std_norm ;  //  vec_of_mats_test(chunk_size, n_tests, 2); //////////////////////////////
   std::vector<Eigen::Matrix<double, -1, -1 > >  prob =  Z_std_norm ;  //  vec_of_mats_test(chunk_size, n_tests, 2); //////////////////////////////
   ///////////////////////////////////////////////);
   Eigen::Array<double, -1, -1> y_chunk = Eigen::Array<double, -1, -1>::Zero(chunk_size, n_tests);
   Eigen::Array<double, -1, -1> u_array =  y_chunk ;
   ///////////////////////////////////////////////
   Eigen::Matrix<double, -1, 1> prod_container_or_inc_array  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
   Eigen::Matrix<double, -1, 1>      prob_n  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
   ///////////////////////////////////////////////
   Eigen::Matrix<double, -1, -1>     common_grad_term_1   =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
   Eigen::Matrix<double, -1, -1 >    L_Omega_diag_recip_array   = common_grad_term_1 ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
   Eigen::Matrix<double, -1, -1 >    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip   =  common_grad_term_1 ; //   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
   Eigen::Matrix<double, -1, -1 >    y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip   =   common_grad_term_1 ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
   Eigen::Matrix<double, -1, -1 >    prop_rowwise_prod_temp   =   common_grad_term_1 ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
   Eigen::Matrix<double, -1, -1>     u_grad_array_CM_chunk   =   common_grad_term_1 ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);    ////////
   Eigen::Matrix<double, -1, -1>     phi_Z_recip  =   common_grad_term_1 ; //  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
   ///////////////////////////////////////////////
   Eigen::Matrix<double, -1, 1>      derivs_chain_container_vec  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
   Eigen::Matrix<double, -1, 1>      prop_rowwise_prod_temp_all  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
   Eigen::Array<int, -1, -1>         abs_indicator = Eigen::Array<int, -1, -1>::Zero(n_tests, 2);
   ////////////////////////////////////////////////
   Eigen::Matrix<double, -1, -1 >  log_prev = stan::math::log(prev);
   ////////////////////////////////////////////////
   
   
   Eigen::Matrix<double, -1, 1>  deriv_L_t1 =     Eigen::Matrix<double, -1, 1>::Zero( n_tests);
   Eigen::Matrix<double, -1, 1>  deriv_L_t1_output_vec =     Eigen::Matrix<double, -1, 1>::Zero( n_tests);
   Eigen::Matrix<double, -1, -1 >   deriv_inc  =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
   Eigen::Matrix<double, -1, 1> deriv_comp_2  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
   
   
   // std::vector<Eigen::Matrix<double, -1, -1 > >  z_grad_term = vec_of_mats_test(chunk_size, n_tests, 2) ;
   //  std::vector<Eigen::Matrix<double, -1, -1 > >  grad_bound_z = vec_of_mats_test(chunk_size, n_tests, 2) ;
   Eigen::Matrix<double, -1, -1>     grad_bound_z =     Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests*2);
   std::vector<Eigen::Matrix<double, -1, -1 > >  grad_Phi_bound_z = vec_of_mats_test(chunk_size, n_tests*2, 2) ;
   std::vector<Eigen::Matrix<double, -1, -1 > >  deriv_Bound_Z_x_L = vec_of_mats_test(chunk_size, n_tests*2, 2) ;
   Eigen::Matrix<double, -1, -1>     grad_prob =      Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests*2);
   Eigen::Matrix<double, -1, -1>     z_grad_term =      Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests*2);
   
   Eigen::Matrix< double , -1, 1>  temp_L_Omega_x_grad_z_sum_1(chunk_size);
   Eigen::Matrix<double, -1, -1>  grad_pi_wrt_b_raw =  Eigen::Matrix<double, -1, -1>::Zero(2, n_tests) ;
   
   
   Eigen::Matrix<double, -1, -1> y_sign   =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
   Eigen::Matrix<double, -1, -1 >  y_m_ysign_x_u_array =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests) ;  ;
   std::vector<Eigen::Matrix<double, -1, -1 > >  phi_Z  =  vec_of_mats_test(chunk_size, n_tests, 2) ;
   
   
   
   {
     
     // /////////////////////////////////////////////////////////////////////////////////////////////////////
     Eigen::Matrix<double, -1, -1>    lp_array  =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, 2);///////
     ////////////////////////////////////////////////////////////////////////////////////////////////////////
     
     
     
     for (int nc = 0; nc < n_chunks; nc++) {
       
       int chunk_counter = nc;
       
       abs_indicator.array() = 0;
       u_grad_array_CM_chunk.array() = 0.0;
       
       y_chunk = y.middleRows(chunk_size * chunk_counter , chunk_size).array().cast<double>() ;
       
       u_array  = 0.5 * (  theta_us.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests).reshaped(chunk_size, n_tests).array() + 1.0 ).array() ;
       
       y_sign.array() = Eigen::Select(y_chunk.array() == 1.0,  Eigen::Array<double, -1, -1>::Ones(chunk_size, n_tests), -Eigen::Array<double, -1, -1>::Ones(chunk_size,  n_tests));
       y_m_ysign_x_u_array.array()  =   ( (  (y_chunk.array()  - y_sign.array()  *  u_array.array()  ).array() ) ) ;   // temp - delete this!!
       
       for (int c = 0; c < n_class; c++) {
         
         prod_container_or_inc_array.array()  = 0.0; // needs to be reset to 0
         
         for (int t = 0; t < n_tests; t++) {
           
           //   Bound_Z[c].col(t).array() =    L_Omega_recip_double[c](t, t) * (  - ( beta_double_array(c, t) +      prod_container_or_inc_array.array()   )  ) ; // / L_Omega_double[c](t, t)     ;
           Bound_Z[c].col(t).array() =     L_Omega_recip_double[c](t, t) *    ( (  0.0 - ( LT_a(c, t).val() +    prod_container_or_inc_array.array()    )     )  ).array() ;
           abs_Bound_Z[c].col(t).array() =    Bound_Z[c].col(t).array().abs();
           if ((abs_Bound_Z[c].col(t).array() > 5.0).matrix().sum() > 0)     abs_indicator(t, c) = 1;
           
           if (Phi_type == "Phi")   Bound_U_Phi_Bound_Z[c].col(t).array() =  0.5 *   ( minus_sqrt_2_recip * Bound_Z[c].col(t).array() ).erfc() ;
           else   Bound_U_Phi_Bound_Z[c].col(t).array() = Phi_approx_fast_Eigen(Bound_Z[c].col(t).array() );
           
           // for all configs
           if (  abs_indicator(t, c) == 1) {
               // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
             for (int n_index = 0; n_index < chunk_size; n_index++) { // can't vectorise ?!
               if ( abs_Bound_Z[c](n_index, t) > 5.0   ) {
                 Bound_U_Phi_Bound_Z[c](n_index, t) =    stan::math::inv_logit( 1.702 *  Bound_Z[c](n_index, t)  );  //  stan::math::inv_logit( 1.702 * Bound_Z[c].col(t)  )
               }
             }
           }
           
           Phi_Z[c].col(t).array()    =      (   y_chunk.col(t)   *  Bound_U_Phi_Bound_Z[c].col(t).array() +
             (   y_chunk.col(t)   -  Bound_U_Phi_Bound_Z[c].col(t).array() ) *   ( y_chunk.col(t)  + (  y_chunk.col(t)  - 1.0)  )  * u_array.col(t)   )   ;
           
           if (Phi_type == "Phi") {
             if (log_fast == false) Z_std_norm[c].col(t).array() = stan::math::inv_Phi(Phi_Z[c].col(t)).array() ; //    qnorm_rcpp_Eigen( Phi_Z[c].col(t).array());
             else                   Z_std_norm[c].col(t).array() = qnorm_w_fast_log_rcpp_Eigen( Phi_Z[c].col(t).array());  // with approximations
           } else if (Phi_type == "Phi_approx") {
             Z_std_norm[c].col(t).array() =  inv_Phi_approx_fast_Eigen(Phi_Z[c].col(t).array());
           }
           
           // for all configs
           if (abs_indicator(t, c) == 1) {
               // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
             for (int n_index = 0; n_index < chunk_size; n_index++) { // can't vectoriose ?!
               if ( abs_Bound_Z[c](n_index, t) > 5.0   ) {
                 Z_std_norm[c](n_index, t) =    s  *  stan::math::logit(Phi_Z[c](n_index, t));  //   s *  fast_logit_1_Eigen(    Phi_Z[c].col(t) ).array()
               }
             }
           }
           
           if (t < n_tests - 1)       prod_container_or_inc_array.array()  =   ( Z_std_norm[c].topLeftCorner(chunk_size, t + 1)  *   ( L_Omega_double[c].row(t+1).head(t+1).transpose()  ) ) ;      // for all configs
           
         } // end of t loop
         
         
         prob[c].array() =  y_chunk.array() * ( 1.0 -    Bound_U_Phi_Bound_Z[c].array() ) +  ( y_chunk - 1.0  )   *    Bound_U_Phi_Bound_Z[c].array() *   ( y_chunk +  (  y_chunk - 1.0)  )  ;   // for all configs
         
         if (log_fast == true)  y1_or_phi_Bound_Z[c].array()  =   fast_log_approx_double_wo_checks_Eigen_mat(prob[c].array() ) ;
         else                   y1_or_phi_Bound_Z[c].array()  =   prob[c].array().log();
         
         lp_array.col(c).array() =     y1_or_phi_Bound_Z[c].rowwise().sum().array() +  log_prev(0,c) ;
         
       } // end of c loop
       
       
       
       if    ( (log_fast == false) && (exp_fast == false)  )  {
         out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()   = (  lp_array.array().maxCoeff() +  (   (  (lp_array.array() - lp_array.array().maxCoeff() ).array()).exp().matrix().rowwise().sum().array()   ).log()  )  ;
         prob_n  =  (out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()).exp() ;
       } else if  ( (log_fast == true) && (exp_fast == false)  )  {
         out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()   = (  lp_array.array().maxCoeff() +  fast_log_approx_double_wo_checks_Eigen(   (  (lp_array.array() - lp_array.array().maxCoeff() ).array()).exp().matrix().rowwise().sum().array()   )  )  ;
         prob_n  =  (out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()).exp() ;
       } else if  ( (log_fast == false) && (exp_fast == true)  )  {
         out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()   = (  lp_array.array().maxCoeff() +  (   fast_exp_double_wo_checks_Eigen_mat(  (lp_array.array() - lp_array.array().maxCoeff() ).array()).matrix().rowwise().sum().array()   ).log()  )  ;
         prob_n  =  fast_exp_double_wo_checks_Eigen(out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()) ;
       } else if  ( (log_fast == true) && (exp_fast == true)  )   {
         out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()   = (  lp_array.array().maxCoeff() +  fast_log_approx_double_wo_checks_Eigen(   fast_exp_double_wo_checks_Eigen_mat(  (lp_array.array() - lp_array.array().maxCoeff() ).array()).matrix().rowwise().sum().array()   )  )  ;
         prob_n  =  fast_exp_double_wo_checks_Eigen(out_mat.segment(n_params + 1, N).segment(chunk_size * chunk_counter, chunk_size).array()) ;
       }
       
       
       
       for (int c = 0; c < n_class; c++) {
         
         for (int i = 0; i < n_tests; i++) { // i goes from 1 to 3
           int t = n_tests - (i+1) ;
           prop_rowwise_prod_temp.col(t).array()   =   prob[c].block(0, t + 0, chunk_size, i + 1).rowwise().prod().array() ;
         }
         
         prop_rowwise_prod_temp_all.array() =  prob[c].rowwise().prod().array()  ;
         
         for (int i = 0; i < n_tests; i++) { // i goes from 1 to 3
           int t = n_tests - (i + 1) ;
           common_grad_term_1.col(t) =   (  ( prev(0,c) / prob_n.array() ) * (    prop_rowwise_prod_temp_all.array() /  prop_rowwise_prod_temp.col(t).array()  ).array() )  ;
         }
         for (int t = 0; t < n_tests; t++) {
           L_Omega_diag_recip_array.col(t).array() =  L_Omega_recip_double[c](t, t) ;
         }
         
         
         for (int t = 0; t < n_tests; t++) {
           
           if (Phi_type == "Phi_approx")  {
             y1_or_phi_Bound_Z[c].array() =                (  (    (  a_times_3*Bound_Z[c].array()*Bound_Z[c].array()   + b  ).array()  ).array() )  *  Bound_U_Phi_Bound_Z[c].array() * (1.0 -  Bound_U_Phi_Bound_Z[c].array() )   ;
           } else if (Phi_type == "Phi") {
             if (exp_fast == true)       y1_or_phi_Bound_Z[c].array() =                   sqrt_2_pi_recip *  fast_exp_double_wo_checks_Eigen_mat( - 0.5 * Bound_Z[c].array()    *  Bound_Z[c].array()    ).array()  ;
             else                        y1_or_phi_Bound_Z[c].array() =                   sqrt_2_pi_recip *  ( - 0.5 * Bound_Z[c].array()    *  Bound_Z[c].array()    ).array().exp()  ;
           }
           
           // for all configs
           if (abs_indicator(t, c) == 1) {
               // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
             for (int n_index = 0; n_index < chunk_size; n_index++) { // can't vectorise ?!
               if ( abs_Bound_Z[c](n_index, t) > 5.0   ) {
                 double Phi_x_i =  Bound_U_Phi_Bound_Z[c](n_index, t);
                 double Phi_x_1m_Phi  =    (  Phi_x_i * (1.0 - Phi_x_i)  ) ;
                 y1_or_phi_Bound_Z[c](n_index, t) =  1.702 * Phi_x_1m_Phi;
               }
             }
           }
         }
         
         
         
         y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.array()  =                  ( y_chunk.array()  + (  y_chunk.array() - 1.0).array() ).array() *    y1_or_phi_Bound_Z[c].array()  *   L_Omega_diag_recip_array.array() ;
         
         if (Phi_type == "Phi_approx")  {
           phi_Z_recip.array()  =    1.0 / (    (   (a_times_3*Z_std_norm[c].array()*Z_std_norm[c].array()   + b  ).array()  ).array() *  Phi_Z[c].array() * (1.0 -  Phi_Z[c].array() )  ).array()  ;  // Phi_type == 2
         } else  if (Phi_type == "Phi")  {
           if (exp_fast == true)      phi_Z_recip.array() =                 1.0  / (   sqrt_2_pi_recip *  fast_exp_double_wo_checks_Eigen_mat( - 0.5 * Z_std_norm[c].array()    *  Z_std_norm[c].array()    ).array() )  ;
           else                       phi_Z_recip.array() =                 1.0  / (   sqrt_2_pi_recip *  ( - 0.5 * Z_std_norm[c].array()    *  Z_std_norm[c].array()    ).array().exp() ).array()  ;
         }
         // for all configs
         for (int t = 0; t < n_tests; t++) {
           if (abs_indicator(t, c) == 1) {
            // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH) 
             for (int n_index = 0; n_index < chunk_size; n_index++) { // can't vectorise ?!
               if ( abs_Bound_Z[c](n_index, t) > 5.0   ) {
                 double Phi_x_i =  Phi_Z[c](n_index, t);
                 double Phi_x_1m_Phi_x_recip =  1.0 / (  Phi_x_i * (1.0 - Phi_x_i)  ) ;
                 Phi_x_1m_Phi_x_recip =  1.0 / (  Phi_x_i * (1.0 - Phi_x_i)  ) ;
                 phi_Z_recip(n_index, t) =      s * Phi_x_1m_Phi_x_recip;
               }
             }
           }
         }
         
         phi_Z[c].array() =  1.0 /     phi_Z_recip.array();
         
         
         y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.array()  =    ( (  (   y_chunk.array()   -  ( y_chunk.array()  + (  y_chunk.array()  - 1.0).array() ).array()    * u_array.array()    ).array() ) ).array() *
           phi_Z_recip.array()  *   y1_or_phi_Bound_Z[c].array()   *    L_Omega_diag_recip_array.array() ;
         
         
         
         
         /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of nuisance parameters / u's (manual)
           {
             
             ///// then second-to-last term (test T - 1)
             int t = n_tests - 1;
             
             u_grad_array_CM_chunk.col(n_tests - 2).array()  +=  (  common_grad_term_1.col(t).array()  * (y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()  *  L_Omega_double[c](t,t - 1) * ( phi_Z_recip.col(t-1).array() )  *  prob[c].col(t-1).array()) ).array()  ;
             
             { ///// then third-to-last term (test T - 2)
               t = n_tests - 2;
               
               z_grad_term.col(0) = ( phi_Z_recip.col(t-1).array())  *  prob[c].col(t-1).array() ;
               grad_prob.col(0) =        y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   *     L_Omega_double[c](t, t - 1) *   z_grad_term.col(0).array() ; // lp(T-1) - part 2;
               z_grad_term.col(1).array()  =      L_Omega_double[c](t,t-1) *   z_grad_term.col(0).array() *       y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array() ;
               grad_prob.col(1)  =         (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t+1).array()   ) *  (  z_grad_term.col(0).array() *  L_Omega_double[c](t + 1,t - 1)  -   z_grad_term.col(1).array()  * L_Omega_double[c](t+1,t)) ;
               
               u_grad_array_CM_chunk.col(n_tests - 3).array()  +=  ( common_grad_term_1.col(t).array()   *  (  grad_prob.col(1).array() *  prob[c].col(t).array()  +      grad_prob.col(0).array() *   prob[c].col(t+1).array()  )  )   ;
             }
             
             // then rest of terms
             for (int i = 1; i < n_tests - 2; i++) { // i goes from 1 to 3
               
               grad_prob.array()   = 0.0;
               z_grad_term.array() = 0.0;
               
               int t = n_tests - (i+2) ; // starts at t = 6 - (1+2) = 3, ends at t = 6 - (3+2) = 6 - 5 = 1 (when i = 3)
               
               z_grad_term.col(0) = (  phi_Z_recip.col(t-1).array())  *  prob[c].col(t-1).array() ;
               grad_prob.col(0) =         y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   *   L_Omega_double[c](t, t - 1) *   z_grad_term.col(0).array() ; // lp(T-1) - part 2;
               
               for (int ii = 0; ii < i+1; ii++) { // if i = 1, ii goes from 0 to 1 u_grad_z u_grad_term
                 if (ii == 0)    prod_container_or_inc_array  = (   (z_grad_term.topLeftCorner(chunk_size, ii + 1) ) *  (fn_first_element_neg_rest_pos(L_Omega_double[c].row( t + (ii-1) + 1).segment(t - 1, ii + 1))).transpose()  )      ;
                 z_grad_term.col(ii+1)  =           y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii).array() *  -   prod_container_or_inc_array.array() ;
                 prod_container_or_inc_array  = (   (z_grad_term.topLeftCorner(chunk_size, ii + 2) ) *  (fn_first_element_neg_rest_pos(L_Omega_double[c].row( t + (ii) + 1).segment(t - 1, ii + 2))).transpose()  )      ;
                 grad_prob.col(ii+1)  =       (    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t+ii+1).array()  ) *     -    prod_container_or_inc_array.array()  ;
               } // end of ii loop
               
               {
                 derivs_chain_container_vec.array() = 0.0;
                 
                 for (int ii = 0; ii < i + 2; ii++) {
                   derivs_chain_container_vec.array()  +=  ( grad_prob.col(ii).array()    * (       prop_rowwise_prod_temp.col(t).array() /   prob[c].col(t + ii).array()  ).array() ).array()  ;
                 }
                 u_grad_array_CM_chunk.col(n_tests - (i+3)).array()    +=   (  ( (   common_grad_term_1.col(t).array()   *  derivs_chain_container_vec.array() ) ).array()  ).array() ;
               }
               
             }
             
             
             out_mat.segment(1, n_us).segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests)  =  u_grad_array_CM_chunk.reshaped() ; //   u_grad_array_CM.block(chunk_size * chunk_counter, 0, chunk_size, n_tests).reshaped() ; // .cast<float>()     ;         }
             
           }
           
           
           // /////////////////////////////////////////////////////////////////////////// Grad of intercepts / coefficients (beta's)
           // ///// last term first (test T)
           {
             
             int t = n_tests - 1;
             
             beta_grad_array(c, t) +=     (common_grad_term_1.col(t).array()  *   (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()    )).sum();
             
             ///// then second-to-last term (test T - 1)
             {
               t = n_tests - 2;
               grad_prob.col(0) =       (     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   ) ;
               z_grad_term.col(0)   =     - y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array()         ;
               grad_prob.col(1)  =       (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t+1).array()    )   * (   L_Omega_double[c](t + 1,t) *      z_grad_term.col(0).array() ) ;
               beta_grad_array(c, t) +=  (common_grad_term_1.col(t).array()   * ( grad_prob.col(1).array() *  prob[c].col(t).array() +         grad_prob.col(0).array() *   prob[c].col(t+1).array() ) ).sum() ;
             }
             
             // then rest of terms
             for (int i = 1; i < n_tests - 1; i++) { // i goes from 1 to 3
               
               t = n_tests - (i+2) ; // starts at t = 6 - (1+2) = 3, ends at t = 6 - (3+2) = 6 - 5 = 1 (when i = 3)
               
               grad_prob.col(0)  =     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   ;
               
               // component 2 (second-to-earliest test)
               z_grad_term.col(0)  =        -y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array()       ;
               grad_prob.col(1) =        y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 1).array()    *    (   L_Omega_double[c](t + 1,t) *   z_grad_term.col(0).array() ) ;
               
               // rest of components
               for (int ii = 1; ii < i+1; ii++) { // if i = 1, ii goes from 0 to 1
                 if (ii == 1)  prod_container_or_inc_array  = (    z_grad_term.topLeftCorner(chunk_size, ii)  *   L_Omega_double[c].row( t + (ii - 1) + 1).segment(t + 0, ii + 0).transpose()  );
                 z_grad_term.col(ii)  =        -y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii).array()    *   prod_container_or_inc_array.array();
                 prod_container_or_inc_array  = (    z_grad_term.topLeftCorner(chunk_size, ii + 1)  *   L_Omega_double[c].row( t + (ii) + 1).segment(t + 0, ii + 1).transpose()  );
                 grad_prob.col(ii + 1) =      (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + ii + 1).array()  ).array()     *  prod_container_or_inc_array.array();
               }
               
               {
                 derivs_chain_container_vec.array() = 0.0;
                 
                 ///// attempt at vectorising  // bookmark
                 for (int ii = 0; ii < i + 2; ii++) {
                   derivs_chain_container_vec.array() +=  ( grad_prob.col(ii).array()  * (      prop_rowwise_prod_temp.col(t).array() /   prob[c].col(t + ii).array()  ).array() ).array() ;
                 }
                 beta_grad_array(c, t) +=        ( common_grad_term_1.col(t).array()   *  derivs_chain_container_vec.array() ).sum();
               }
               
             }
             
           }
           
           
           
           ////////////////////////////////////////////////////////////////////////////////////////////////// Grad of L_Omega ('s)
           {
             
             
             ///////////////////////// deriv of diagonal elements (not needed if using the "standard" or "Stan" Cholesky parameterisation of Omega)
             
             //////// w.r.t last diagonal first
             {
               int  t1 = n_tests - 1;
               
               
               
               double deriv_L_T_T_inv =  ( - 1 /  ( L_Omega_double[c](t1,t1)  * L_Omega_double[c](t1,t1) ) )   *   Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t1, t1)  ;
               
               deriv_Bound_Z_x_L[c].col(0).array() = 0;
               for (int t = 0; t < t1; t++) {
                 deriv_Bound_Z_x_L[c].col(0).array() +=   Z_std_norm[c].col(t).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t1, t);
               }
               
               double deriv_a = 0 ; //  stan::math::pow( (1 + (bs_mat_double(c, t1)*bs_mat_double(c, t1)) ), -0.5) * bs_mat_double(c, t1)  * LT_theta(c, t1)  ;
               deriv_Bound_Z_x_L[c].col(0).array()   = deriv_a -     deriv_Bound_Z_x_L[c].col(0).array();
               
               grad_bound_z.col(0).array() =   deriv_L_T_T_inv * (Bound_Z[c].col(t1).array() * L_Omega_double[c](t1,t1)  ) +  (1 / L_Omega_double[c](t1, t1)) *   deriv_Bound_Z_x_L[c].col(0).array()  ;
               grad_Phi_bound_z[c].col(0)  =  ( y1_or_phi_Bound_Z[c].col(t1).array() *  (  grad_bound_z.col(0).array() )  ) .matrix();   // correct  (standard form)
               grad_prob.col(0)   =  (   - y_sign.col(t1).array()  *   grad_Phi_bound_z[c].col(0).array() ).matrix() ;     // correct  (standard form)
               
               
               grad_pi_wrt_b_raw(c, t1)  +=   (   common_grad_term_1.col(t1).array()    *            grad_prob.col(0).array()    ).matrix().sum()   ; // correct  (standard form)
               
               
             }
             
             
             //////// then w.r.t the second-to-last diagonal
             {
               int  t1 = n_tests - 2;
               
               double deriv_L_T_T_inv =  ( - 1 /   ( L_Omega_double[c](t1,t1)  * L_Omega_double[c](t1,t1) ) )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t1, t1)  ;
               
               deriv_Bound_Z_x_L[c].col(0).array() = 0.0;
               for (int t = 0; t < t1; t++) {
                 deriv_Bound_Z_x_L[c].col(0).array() += Z_std_norm[c].col(t).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t1, t);
               }
               
               double deriv_a = 0 ; //   stan::math::pow( (1 + (bs_mat_double(c, t1)*bs_mat_double(c, t1)) ), -0.5) * bs_mat_double(c, t1)   * LT_theta(c, t1)   ;
               deriv_Bound_Z_x_L[c].col(0).array()   = deriv_a -     deriv_Bound_Z_x_L[c].col(0).array();
               
               
               grad_bound_z.col(0).array() =  deriv_L_T_T_inv * (Bound_Z[c].col(t1).array() * L_Omega_double[c](t1,t1)  ) +  (1 / L_Omega_double[c](t1, t1)) *     deriv_Bound_Z_x_L[c].col(0).array()  ;
               grad_Phi_bound_z[c].col(0)  =  ( y1_or_phi_Bound_Z[c].col(t1).array() *  (  grad_bound_z.col(0).array() )  ) .matrix();   // correct  (standard form)
               grad_prob.col(0)   =  (   - y_sign.col(t1).array()  *   grad_Phi_bound_z[c].col(0).array() ).matrix() ;     // correct  (standard form)
               
               
               z_grad_term.col(0).array()  =      (  ( (  y_m_ysign_x_u_array.col(t1).array()   / phi_Z[c].col(t1).array()  ).array()    * y1_or_phi_Bound_Z[c].col(t1).array() *   grad_bound_z.col(0).array()  ).array() ).matrix()  ;  // correct  (standard form)
               
               deriv_L_T_T_inv =  ( - 1 /  ( L_Omega_double[c](t1+1,t1+1)  * L_Omega_double[c](t1+1,t1+1)  )  )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t1+1, t1+1)  ;
               deriv_Bound_Z_x_L[c].col(1).array()  =    L_Omega_double[c](t1+1,t1) *   z_grad_term.col(0).array()     +   Z_std_norm[c].col(t1).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t1+1, t1);
               grad_bound_z.col(1).array() =  deriv_L_T_T_inv * (Bound_Z[c].col(t1+1).array() * L_Omega_double[c](t1+1,t1+1)  ) +  (1 / L_Omega_double[c](t1+1, t1+1)) * -   deriv_Bound_Z_x_L[c].col(1).array()  ;
               
               grad_Phi_bound_z[c].col(1) =   ( y1_or_phi_Bound_Z[c].col(t1 + 1).array() *  (    grad_bound_z.col(1).array()   ) ).matrix();   // correct  (standard form)
               grad_prob.col(1)   =   (  - y_sign.col(t1 + 1).array()  *     grad_Phi_bound_z[c].col(1).array()  ).array().matrix() ;    // correct   (standard form)
               
               grad_pi_wrt_b_raw(c, t1) +=   ( ( common_grad_term_1.col(t1).array() )    *
                 ( prob[c].col(t1 + 1).array()  *      grad_prob.col(0).array()  +   prob[c].col(t1).array()  *         grad_prob.col(1).array()   ) ).sum() ;
             }
             
             
             //////// then w.r.t the third-to-last diagonal .... etc
             {
               
               //   int i = 4;
               for (int i = 3; i < n_tests + 1; i++) {
                 
                 grad_prob.array()   = 0.0;
                 z_grad_term.array() = 0.0;
                 
                 
                 int  t1 = n_tests - i;
                 
                 double deriv_L_T_T_inv =  ( - 1 /   ( L_Omega_double[c](t1,t1)  * L_Omega_double[c](t1,t1) ) )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t1, t1)  ;
                 
                 deriv_Bound_Z_x_L[c].col(0).array() = 0;
                 for (int t = 0; t < t1; t++) {
                   deriv_Bound_Z_x_L[c].col(0).array() += Z_std_norm[c].col(t).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t1, t);
                 }
                 
                 double deriv_a = 0 ; //  stan::math::pow( (1 + (bs_mat_double(c, t1)*bs_mat_double(c, t1)) ), -0.5) * bs_mat_double(c, t1) *   LT_theta(c, t1)   ;
                 deriv_Bound_Z_x_L[c].col(0).array()   = deriv_a -     deriv_Bound_Z_x_L[c].col(0).array();
                 
                 
                 grad_bound_z.col(0).array() =  deriv_L_T_T_inv * (Bound_Z[c].col(t1).array() * L_Omega_double[c](t1,t1)  ) +  (1 / L_Omega_double[c](t1, t1)) *    deriv_Bound_Z_x_L[c].col(0).array()  ;
                 grad_Phi_bound_z[c].col(0)  =  ( y1_or_phi_Bound_Z[c].col(t1).array() *  (  grad_bound_z.col(0).array() )  ) .matrix();   // correct  (standard form)
                 grad_prob.col(0)   =  (   - y_sign.col(t1).array()  *   grad_Phi_bound_z[c].col(0).array() ).matrix() ;     // correct  (standard form)
                 
                 
                 z_grad_term.col(0).array()  =      (  ( (  y_m_ysign_x_u_array.col(t1).array()   / phi_Z[c].col(t1).array()  ).array()    * y1_or_phi_Bound_Z[c].col(t1).array() *   grad_bound_z.col(0).array()  ).array() ).matrix()  ;  // correct  (standard form)
                 
                 deriv_L_T_T_inv =  ( - 1 /  ( L_Omega_double[c](t1+1,t1+1)  * L_Omega_double[c](t1+1,t1+1)  )  )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t1+1, t1+1)  ;
                 deriv_Bound_Z_x_L[c].col(1).array()  =    L_Omega_double[c](t1+1,t1) *   z_grad_term.col(0).array()     +   Z_std_norm[c].col(t1).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t1+1, t1);
                 grad_bound_z.col(1).array() =  deriv_L_T_T_inv * (Bound_Z[c].col(t1+1).array() * L_Omega_double[c](t1+1,t1+1)  ) +  (1 / L_Omega_double[c](t1+1, t1+1)) * -  deriv_Bound_Z_x_L[c].col(1).array()  ;
                 
                 grad_Phi_bound_z[c].col(1) =   ( y1_or_phi_Bound_Z[c].col(t1 + 1).array() *  (    grad_bound_z.col(1).array()   ) ).matrix();   // correct  (standard form)
                 grad_prob.col(1)  =   (  - y_sign.col(t1 + 1).array()  *     grad_Phi_bound_z[c].col(1).array()  ).array().matrix() ;    // correct   (standard form)
                 
                 
                 for (int ii = 1; ii < i - 1; ii++) {
                   z_grad_term.col(ii).array()  =    (  ( (  y_m_ysign_x_u_array.col(t1 + ii).array()   / phi_Z[c].col(t1 + ii).array()  ).array()    * y1_or_phi_Bound_Z[c].col(t1 + ii).array() *   grad_bound_z.col(ii).array()  ).array() ).matrix() ;     // correct  (standard form)
                   
                   deriv_L_T_T_inv =  ( - 1 /  (  L_Omega_double[c](t1 + ii + 1,t1 + ii + 1)  * L_Omega_double[c](t1 + ii + 1,t1 + ii + 1) )  )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t1 + ii + 1, t1 + ii + 1)  ;
                   
                   deriv_Bound_Z_x_L[c].col(ii + 1).array()   =  0.0;
                   for (int jj = 0; jj < ii + 1; jj++) {
                     deriv_Bound_Z_x_L[c].col(ii + 1).array()  +=    L_Omega_double[c](t1 + ii + 1,t1 + jj)     *   z_grad_term.col(jj).array()     +   Z_std_norm[c].col(t1 + jj).array()     *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t1 + ii + 1, t1 + jj) ;// +
                   }
                   grad_bound_z.col(ii + 1).array() =  deriv_L_T_T_inv * (Bound_Z[c].col(t1 + ii + 1).array() * L_Omega_double[c](t1 + ii + 1,t1 + ii + 1)  ) +  (1 / L_Omega_double[c](t1 + ii + 1, t1 + ii + 1)) * -   deriv_Bound_Z_x_L[c].col(ii + 1).array()  ;
                   grad_Phi_bound_z[c].col(ii + 1).array()  =     y1_or_phi_Bound_Z[c].col(t1 + ii + 1).array()  *   grad_bound_z.col(ii + 1).array() ;   // correct  (standard form)
                   grad_prob.col(ii + 1).array()  =   ( - y_sign.col(t1 + ii + 1).array()  ) *    grad_Phi_bound_z[c].col(ii + 1).array() ;  // correct  (standard form)
                   
                 }
                 
                 
                 derivs_chain_container_vec.array() = 0.0;
                 
                 ///// attempt at vectorising  // bookmark
                 for (int iii = 0; iii <  i; iii++) {
                   derivs_chain_container_vec.array() +=  (    grad_prob.col(iii).array()  * (   prob[c].block(0, t1 + 0, chunk_size, i).rowwise().prod().array()  /  prob[c].col(t1 + iii).array()  ).array() ).array() ; // correct  (standard form)
                 }
                 
                 
                 grad_pi_wrt_b_raw(c, t1) +=        ( common_grad_term_1.col(t1).array()   *  derivs_chain_container_vec.array() ).sum();
                 
                 
               }
               
             }
             
             prev_grad_vec(c)  +=  ( ( 1.0 / prob_n.array() ) * prob[c].rowwise().prod().array() ).matrix().sum() ;
             
             
           }
           
       }
       
     }
     
     
     //////////////////////// gradients for latent class membership probabilitie(s) (i.e. disease prevalence)
     for (int c = 0; c < n_class; c++) {
       prev_unconstrained_grad_vec(c)  =   prev_grad_vec(c)   * deriv_p_wrt_pu_double ;
     }
     prev_unconstrained_grad_vec(0) = prev_unconstrained_grad_vec(1) - prev_unconstrained_grad_vec(0) - 2 * tanh_u_prev[1];
     prev_unconstrained_grad_vec_out(0) = prev_unconstrained_grad_vec(0);
     
     
     // log_prob_out += log_lik.sum();
     log_prob_out += out_mat.segment(1 + n_params, N).sum();
     
     if (exclude_priors == false)  log_prob_out += prior_densities;
     
     log_prob_out +=  log_jac_u;
     //  log_prob_out +=  log_jac_p;
     
     log_prob = (double) log_prob_out;
     
     int i = 0; // probs_all_range.prod() cancels out
     for (int c = 0; c < n_class; c++) {
       for (int t = 0; t < n_tests; t++) {
         beta_grad_vec(i) = beta_grad_array(c, t);
         i += 1;
       }
     }
     
     
     Eigen::Matrix<double, -1, 1 >  bs_grad_vec_nd =  (grad_pi_wrt_b_raw.row(0).transpose().array() * bs_nd_double.array()).matrix() ; //     ( deriv_log_pi_wrt_L_Omega[0].asDiagonal().diagonal().array() * bs_nd_double.array()  ).matrix()  ; //  Jacobian_d_L_Sigma_wrt_b_matrix[0].transpose() * deriv_log_pi_wrt_L_Omega_vec_nd;
     Eigen::Matrix<double, -1, 1 >  bs_grad_vec_d =   (grad_pi_wrt_b_raw.row(1).transpose().array() * bs_d_double.array()).matrix() ; //    ( deriv_log_pi_wrt_L_Omega[1].asDiagonal().diagonal().array() * bs_d_double.array()  ).matrix()  ; //   Jacobian_d_L_Sigma_wrt_b_matrix[1].transpose()  * deriv_log_pi_wrt_L_Omega_vec_d;
     
     Eigen::Matrix<double, -1, 1 >   bs_grad_vec(n_bs_LT);
     bs_grad_vec.head(n_tests)              = bs_grad_vec_nd ;
     bs_grad_vec.segment(n_tests, n_tests)  = bs_grad_vec_d;
     
     //   stan::math::recover_memory();
     
     
     
     ////////////////////////////  outputs // add log grad and sign stuff';///////////////
     out_mat(0) =  log_prob;
     out_mat.segment(1 + n_us, n_bs_LT)  += bs_grad_vec ;
     out_mat.segment(1 + n_us + n_corrs, n_coeffs) += beta_grad_vec;//.cast<float>();
     out_mat(n_params) = ((grad_prev_AD +  prev_unconstrained_grad_vec_out(0)));
     out_mat.segment(1, n_us).array() =     (  out_mat.segment(1, n_us).array() *  ( 0.5 * (1.0 - theta_us.array() * theta_us.array()  )  )   ).array()    - 2.0 * theta_us.array()   ;
     
     
     
   }
   
   
   
   int LT_cnt_2 = 0;
   for (int c = 0; c < n_class; ++c) {
     for (int t = 0; t < n_tests; ++t) {
       if (LT_known_bs_indicator(c, t) == 1) {
         out_mat(1 + n_us + LT_cnt_2) = 0;
       }
       LT_cnt_2 += 1;
     }
   }
   
   
   return(out_mat);
   
   
   
   
 }







 
 

 
 

 

 // [[Rcpp::export]]
 Eigen::Matrix<double, -1, -1>     Rcpp_fn_sampling_single_iter_burnin( std::string Model_type,
                                                                        Eigen::Array<double, -1, 1 >  theta_main_array,
                                                                        Eigen::Array<double, -1, 1 >  theta_us_array,   //////////////////////////////////////////////
                                                                        Eigen::Matrix<int, -1, -1>	 y,    /////////////////////////////////////////////
                                                                        std::vector<Eigen::Matrix<double, -1, -1 > >  X, /////////////////////////////////////////////
                                                                        Rcpp::List other_args,
                                                                        int L, 
                                                                        double eps,
                                                                        double log_posterior,
                                                                        Eigen::Array<double, -1, 1  > M_inv_us_array, //////////////////////////////////////////////
                                                                        Eigen::Matrix<double, -1, -1  > M_dense_main,
                                                                        Eigen::Matrix<double, -1, -1  > M_inv_dense_main,
                                                                        Eigen::Matrix<double, -1, -1  > M_inv_dense_main_chol
                                                                        )  {

   
   const int n_class = other_args(6); 
   const int n_chunks = other_args(8); 
   
   
   const int n_tests = y.cols();
   const int N = y.rows();
   const int n_corrs =  n_class * n_tests * (n_tests - 1) * 0.5;
   const int n_coeffs = n_class * n_tests * 1;
   const int n_us =  1 *  N * n_tests;
   
   const int n_bs_LT =  n_tests * n_class;
   
   const int n_params = theta_us_array.rows() +  theta_main_array.rows()   ; // n_corrs + n_coeffs + n_us + n_class;
   const int n_params_main = n_params - n_us;

   Eigen::Array<double, -1, 1 >       theta_us_previous_iter =   theta_us_array.cast<double>();
   Eigen::Array<double, -1, 1 >       theta_main_previous_iter = theta_main_array.cast<double>();
   Eigen::Matrix<int,   -1, 1 >       div_vec  = Eigen::Matrix<int, -1, 1  >::Zero(1);

   Eigen::Array<double, -1, 1 >   theta_us_array_initial =    theta_us_array.cast<double>();
   Eigen::Array<double, -1, 1 >   theta_main_array_initial =  theta_main_array.cast<double>();
   
   Eigen::Array<double, -1, 1 >   theta_us_array_proposed =    theta_us_array.cast<double>();
   Eigen::Array<double, -1, 1 >   theta_main_array_proposed =  theta_main_array.cast<double>();

   int L_main_ii = 0;
   
   int chunk_counter = 0;
   int chunk_size  = std::round( N / n_chunks  / 2) * 2;  ; // N / n_chunks;

   double p_jump = 0.0;
   int accept  = 0;
   
   Eigen::Array<double,  -1, 1  > velocity_0_us_array(n_us) ; /////////////////////////////////////////////////////////////////////////////////////////
   Eigen::Array<double,  -1, 1  > velocity_0_main_array(n_params_main) ;
   
   Eigen::Array<double,  -1, 1  > velocity_us_array_proposed(n_us) ; /////////////////////////////////////////////////////////////////////////////////////////
   Eigen::Array<double,  -1, 1  > velocity_main_array_proposed(n_params_main) ;
   
   Eigen::Array<double,  -1, 1  > velocity_us_array(n_us) ; //. =  velocity_us_array_pre_attempts; 
   Eigen::Array<double,  -1, 1  > velocity_main_array(n_params_main) ;
   
   double U_x_initial =  0.0 ; //  - log_posterior_initial;
   double U_x =  0.0 ; // - log_posterior_initial;
   double log_posterior_prop =  0.0 ;// log_posterior;
   double log_posterior_0  =   0.0 ; // log_posterior_initial;
   
   int  ii = 0;

   {
                                 U_x_initial =  0.0 ; //  - log_posterior_initial;
                                 U_x =  0.0 ; // - log_posterior_initial;
                                 log_posterior  =    0.0 ; // log_posterior_initial;
                                 log_posterior_prop =  0.0 ;// log_posterior;
                                 
                           ////////////////////// make complete parameter vector (log_diffs then coeffs)
                           double minus_half_eps = -  0.5 * eps;

                           double energy_old_wo_U =  0.0;
                           log_posterior_0 = 0.0;

                           {
                                     {
                                       
                                         Eigen::Matrix<double, -1, 1>  std_norm_vec_main(n_params_main);
                                         for (int d = 0; d < n_params_main; d++) { 
                                           std_norm_vec_main(d) =      zigg.norm();
                                         }
                                         velocity_0_main_array.matrix()  = M_inv_dense_main_chol * std_norm_vec_main;
  
                                         for (int d = 0; d < n_us; d++) {
                                           velocity_0_us_array(d) =  zigg.norm() ;
                                         }
                                         velocity_0_us_array.array() = velocity_0_us_array.array() * M_inv_us_array.sqrt().array() ;
                                         
                                     }

                                     energy_old_wo_U +=  0.5 * ( velocity_0_main_array * Rcpp_mult_mat_by_col_vec(M_dense_main, velocity_0_main_array.matrix()).array() ).matrix().sum() ;
                                     energy_old_wo_U +=  0.5 * ( stan::math::square( velocity_0_us_array.matrix() ).array() * ( 1.0 / M_inv_us_array.array()    ) ).matrix().sum() ; /////////////////////////////////////////////////////////////////////////////////////////
                           }



                           theta_main_previous_iter = theta_main_array;
                           theta_us_previous_iter = theta_us_array;


                        {
                             
                                   velocity_us_array =   velocity_0_us_array; // reset velocity
                                   velocity_main_array = velocity_0_main_array;  // reset velocity
                                   
                                   velocity_us_array_proposed =   velocity_0_us_array; // reset velocity
                                   velocity_main_array_proposed = velocity_0_main_array;  // reset velocity
                                   
                                   theta_main_array = theta_main_previous_iter; // reset theta
                                   theta_us_array = theta_us_previous_iter;  // reset theta
                                   
                                   theta_main_array_proposed = theta_main_array;
                                   theta_us_array_proposed = theta_us_array;
 
                                   div_vec(ii) = 0;  // reset div


                              try {



                               // ---------------------------------------------------------------------------------------------------------------///    Perform L leapfrogs   ///-----------------------------------------------------------------------------------------------------------------------------------------


                             {
                               Eigen::Matrix<double, -1, 1>  neg_lp_and_grad_outs  =  Eigen::Matrix<double, -1, 1>::Zero(n_params + 1 + N) ;   /////////////////////////////////////////////////////////////////////////////////////////////



                               for (int l = -1; l < L; l++) {

                                                       if (l > -1) {

                                                                   for (int nc = 0; nc < n_chunks; nc++) {

                                                                     int chunk_counter = nc ;

                                                                     velocity_us_array_proposed.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests)    +=   minus_half_eps *
                                                                                                                                                                    neg_lp_and_grad_outs.segment(1 + chunk_size * n_tests * chunk_counter , chunk_size * n_tests).array() *
                                                                                                                                                                    M_inv_us_array.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests).array()  ;

                                                                     theta_us_array_proposed.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests)    +=   eps   * velocity_us_array_proposed.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests) ;

                                                                   }

                                                                   velocity_main_array_proposed +=   minus_half_eps *  ( M_inv_dense_main  *    neg_lp_and_grad_outs.segment(n_us + 1, n_params_main).matrix()  ).array() ;
                                                                   theta_main_array_proposed  +=   eps  *  velocity_main_array_proposed; //// update params by full step
                                                                   
                                                                   if (Model_type == "LT_LC")  { // for latent trait  
                                                                     for (int i = 0 + n_bs_LT; i < 0 + n_corrs; i++) {
                                                                       velocity_main_array_proposed(i) = 0.0 ;
                                                                       theta_main_array_proposed(i) =   R::rnorm(0, 1);
                                                                     }
                                                                   }

                                                         }


                                                       
                                                       if (Model_type == "MVP_LC")  {
                                                           neg_lp_and_grad_outs.array()    =    - fn_lp_and_grad_MVP_LC_using_Chol_Spinkney_MD_and_AD(         theta_main_array_proposed , // .cast<double>().matrix(),
                                                                                                                                                                        theta_us_array_proposed , //  .cast<double>() , // .matrix(), // needs tto be a double? check
                                                                                                                                                                        y,
                                                                                                                                                                        X,
                                                                                                                                                                        other_args).array() ; 
                                                       } else if (Model_type == "LT_LC") {   
                                                         neg_lp_and_grad_outs.array()    =    - fn_lp_and_grad_latent_trait_MD_and_AD(                 theta_main_array_proposed ,    
                                                                                                                                                                      theta_us_array_proposed ,  
                                                                                                                                                                      y,
                                                                                                                                                                      X,
                                                                                                                                                                      other_args).array() ; 
                                                       }   else if (Model_type == "MVP_standard") { 
                                                         neg_lp_and_grad_outs.array()    =    - fn_lp_and_grad_MVP_using_Chol_Spinkney_MD_and_AD(     theta_main_array_proposed ,  
                                                                                                                                                      theta_us_array_proposed , 
                                                                                                                                                      y,
                                                                                                                                                      X,
                                                                                                                                                      other_args).array() ; 
                                                       }
                                                       


                                                         if (l == -1)      log_posterior_0 = - neg_lp_and_grad_outs(0)  ;


                                                         if (l > -1) {

                                                           for (int nc = 0; nc < n_chunks; nc++) {

                                                             int chunk_counter = nc ;

                                                             velocity_us_array_proposed.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests)    +=   minus_half_eps *
                                                                                                                                                            neg_lp_and_grad_outs.segment(1 + chunk_size * n_tests * chunk_counter , chunk_size * n_tests).array() *
                                                                                                                                                            M_inv_us_array.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests).array()  ;

                                                           }

                                                           velocity_main_array_proposed +=   minus_half_eps *  ( M_inv_dense_main  *    neg_lp_and_grad_outs.segment(n_us + 1, n_params_main).matrix()  ).array() ;
                                                           
                                                           if (Model_type == "LT_LC")  { // for latent trait  
                                                             for (int i = 0 + n_bs_LT; i < 0 + n_corrs; i++) {
                                                               velocity_main_array_proposed(i) = 0.0 ;
                                                               theta_main_array_proposed(i) =   R::rnorm(0, 1);
                                                             }
                                                           }

                                                         }




                                     }

                                     // individual_log_lik.array() = - neg_lp_and_grad_outs.segment(1 + n_params, N).array().cast<double>()  ;   /////////////////////////////////////////////////////////////////////////////////////////
                                     log_posterior =  - neg_lp_and_grad_outs(0) ;

                             }



                                       

                                       


                                         //////////////////////////////////////////////////////////////////    M-H acceptance step  (i.e, Accept/Reject step)
                                         
                                      //   velocity_us_array_proposed = velocity_us_array;
                                        // velocity_main_array_proposed = velocity_main_array;
                                         
                                         log_posterior_prop = log_posterior ;
                                         U_x = - log_posterior_prop;
                                         U_x_initial =  - log_posterior_0;

                                         double energy_old = U_x_initial  + energy_old_wo_U ;
      
                                         double energy_new = U_x ;
                                         energy_new +=   0.5 * ( velocity_main_array_proposed.cast<double>() * Rcpp_mult_mat_by_col_vec(M_dense_main.cast<double>(), velocity_main_array_proposed.cast<double>().matrix()).array() ).matrix().sum() ;
                                         energy_new +=   0.5 * ( stan::math::square( velocity_us_array_proposed.matrix().cast<double>() ).array() * ( 1.0  / M_inv_us_array.cast<double>()   ) ).cast<double>().matrix().sum() ;  /////////////////////////////////////////////////////////////////////////////////////////



                                         double log_ratio = - energy_new + energy_old;

                                         Eigen::Matrix<double, -1, 1 >  p_jump_vec(2);
                                         p_jump_vec(0) = 1.0;
                                         p_jump_vec(1) = std::exp(log_ratio);

                                         p_jump = stan::math::min(p_jump_vec);

                                         accept = 0;

                                         if  ( (R::runif(0, 1) > p_jump) ) {  // # reject proposal
                                                 accept = 0;
                                                 theta_us_array = theta_us_previous_iter;// .cast<double>()  ;   /////////////////////////////////////////////////////////////////////////////////////////
                                                 theta_main_array = theta_main_previous_iter; // .cast<double>()  ;
                                                 velocity_us_array = velocity_0_us_array;
                                                 velocity_main_array = velocity_0_main_array;
                                         } else {   // # accept proposal
                                                 accept = 1;
                                                 theta_us_array = theta_us_array_proposed ;//.cast<double>()  ;   /////////////////////////////////////////////////////////////////////////////////////////
                                                 theta_main_array = theta_main_array_proposed ; // .cast<double>()  ;
                                                 velocity_us_array = velocity_us_array_proposed;
                                                 velocity_main_array = velocity_main_array_proposed;
                                         }

                                         div_vec(ii) = 0;

                                     } catch (...) {
                                               div_vec(ii) = 1;
                                     }

                                 }

   }


   
   Eigen::Matrix<double, -1, 1> theta(n_params);
   Eigen::Matrix<double, -1, 1> theta_initial(n_params);
   Eigen::Matrix<double, -1, 1> theta_prop(n_params);
   
   Eigen::Matrix<double, -1, 1> velocity(n_params);
   Eigen::Matrix<double, -1, 1> velocity_0(n_params);
   Eigen::Matrix<double, -1, 1> velocity_prop(n_params);
   
   theta.head(n_us) = theta_us_array.matrix();   
   theta.segment(n_us, n_params_main) = theta_main_array.matrix(); 
   theta_initial.head(n_us) = theta_us_array_initial.matrix();  
   theta_initial.segment(n_us, n_params_main)  = theta_main_array_initial.matrix(); 
   theta_prop.head(n_us) = theta_us_array_proposed.matrix();  
   theta_prop.segment(n_us, n_params_main)  = theta_main_array_proposed.matrix(); 
   
   velocity.head(n_us) = velocity_us_array.matrix();  
   velocity.segment(n_us, n_params_main)  = velocity_main_array.matrix(); 
   velocity_0.head(n_us) = velocity_0_us_array.matrix(); 
   velocity_0.segment(n_us, n_params_main)  = velocity_0_main_array.matrix(); 
   velocity_prop.head(n_us) = velocity_us_array_proposed.matrix();  
   velocity_prop.segment(n_us, n_params_main)  = velocity_main_array_proposed.matrix(); 
     
   Eigen::Matrix<double, -1, -1>  out_mat =   Eigen::Matrix<double, -1, -1>::Zero(n_params, 7);  //////////////////////////////////////////////
   
   out_mat(0, 0) = log_posterior_0;
   out_mat(1, 0) = log_posterior;
   out_mat(2, 0) = log_posterior_prop;
   
   out_mat(3, 0) = div_vec(0);
   out_mat(4, 0) = p_jump;
   out_mat(5, 0) = accept;
   
   out_mat.col(1) = theta;
   out_mat.col(2) = theta_initial;
   out_mat.col(3) = theta_prop;
   
   out_mat.col(4) = velocity;
   out_mat.col(5) = velocity_0;
   out_mat.col(6) = velocity_prop;
   
   return(out_mat);
   



 }
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
  
 
 
 // [[Rcpp::export]]
 Rcpp::List   Rcpp_fn_post_burnin_HMC_post_adaptation_phase_float_big_version(      std::string Model_type,
                                                                                    Eigen::Array<double, -1, 1 >  theta_main_array,
                                                                                    Eigen::Array<double, -1, 1 >  theta_us_array,   //////////////////////////////////////////////
                                                                                    Eigen::Matrix<int, -1, -1>	 y, /////////////////////////////////////////////
                                                                                    std::vector<Eigen::Matrix<double, -1, -1 > >  X, /////////////////////////////////////////////
                                                                                    Rcpp::List other_args,
                                                                                    const bool tau_jittered,
                                                                                    const int n_iter,
                                                                                    const int n_chain_for_loading_bar,
                                                                                    double tau,
                                                                                    const double eps,
                                                                                    double log_posterior,
                                                                                    Eigen::Array<double, -1, 1  > M_inv_us_array, //////////////////////////////////////////////
                                                                                    Eigen::Matrix<double, -1, -1  > M_dense_main,
                                                                                    Eigen::Matrix<double, -1, -1  > M_inv_dense_main,
                                                                                    Eigen::Matrix<double, -1, -1  > M_inv_dense_main_chol
                                                                                   
 )  { 
   
   const int n_class = other_args(6); 
   const int n_chunks = other_args(8);

   const int n_tests = y.cols();
   const int N = y.rows();
   const int n_corrs =  n_class * n_tests * (n_tests - 1) * 0.5;
   const int n_coeffs = n_class * n_tests * 1;
   const int n_us =   N * n_tests;
   
   const int n_bs_LT =  n_tests * n_class;
   
   const int n_params = theta_us_array.rows() +  theta_main_array.rows()   ; // n_corrs + n_coeffs + n_us + n_class;
   const int n_params_main = n_params - n_us;
     
   Eigen::Array<double, -1, -1 >   theta_main_trace  = Eigen::Matrix<double, -1, -1  >::Zero(n_params_main, n_iter);
   Eigen::Array<double, -1, 1 >       theta_main_previous_iter = theta_main_array.cast<double>();
   Eigen::Array<double, -1, 1 >       theta_us_previous_iter = theta_us_array.cast<double>();
   //  Eigen::Matrix<double, -1, -1 >   log_lik_trace  = Eigen::Matrix<double, -1, -1  >::Zero(N, n_iter);
   Eigen::Matrix<int, -1, 1 >   div_vec  = Eigen::Matrix<int, -1, 1  >::Zero(n_iter);
    
   bool display_progress = false;
   if (n_chain_for_loading_bar == 1)    display_progress = true;
    
   Progress p(n_iter, display_progress);
   
   int L_main_ii = 0; 
   
   Eigen::Matrix<double, -1, 1>   individual_log_lik  =  Eigen::Matrix<double, -1, 1>::Zero(N)  ; //////////////////////////////////////////////////////////////////////////////////////////// 
   
   int chunk_counter = 0;
   int chunk_size  = std::round( N / n_chunks  / 2) * 2;  ; // N / n_chunks;
   
   for (int ii = 0; ii < n_iter; ++ii)  {  
     
     theta_main_previous_iter = theta_main_array; 
     theta_us_previous_iter = theta_us_array;
     
     
     if (n_chain_for_loading_bar == 1)      p.increment(); // update progress
     if (Progress::check_abort() )    return -1;
     
     if (tau_jittered == true) {
       double tau_main_ii = R::runif(0.0,  2.0*tau);
       L_main_ii = std::ceil(tau_main_ii / eps);
     } else {
       L_main_ii = std::ceil(tau / eps); 
     } 
     
     double U_x_initial = 0.0 ; //  - log_posterior_initial;
     double U_x = 0.0 ; // - log_posterior_initial;
     double log_posterior  =   0.0 ; // log_posterior_initial;
     double log_posterior_prop = 0.0 ;// log_posterior;
     
     ////////////////////// make complete parameter vector (log_diffs then coeffs)
     double minus_half_eps = -  0.5 * eps;
     
     double energy_old_wo_U = 0.0;
     double log_posterior_0 = 0.0;
     
     Eigen::Array<double,  -1, 1  > velocity_us_array_pre_attempts(n_us); //  = velocity_0_us_array;//.cast<double>()  ; ////////////////////////////////////////////////////////////////////////////////////////////
     Eigen::Array<double,  -1, 1  > velocity_main_array_pre_attempts(n_params_main) ;
     
     Eigen::Array<double,  -1, 1  > velocity_0_us_array(n_us) ; /////////////////////////////////////////////////////////////////////////////////////////
     Eigen::Array<double,  -1, 1  > velocity_0_main_array(n_params_main) ;
     
     
     
     {
        
       
       {
         Eigen::Matrix<double, -1, 1>  std_norm_vec_main(n_params_main);
           // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
         for (int d = 0; d < n_params_main; d++) {
           std_norm_vec_main(d) =      zigg.norm();
         } 
         velocity_0_main_array.matrix()  = M_inv_dense_main_chol * std_norm_vec_main;
         
         // velocity_0_us_array =  draw_mean_zero_norm_using_Zigg_Rcpp(velocity_0_us_array,  M_inv_us_array.sqrt().matrix() ) ;
           // #pragma clang loop vectorize_width(VECTORISATION_VEC_WIDTH)
         for (int d = 0; d < n_us; d++) {
           velocity_0_us_array(d) =  zigg.norm() ;
         } 
         velocity_0_us_array.array() = velocity_0_us_array.array() * M_inv_us_array.sqrt().array() ;
         
         
       }
        
       velocity_us_array_pre_attempts = velocity_0_us_array;
       velocity_main_array_pre_attempts = velocity_0_main_array;
       
       energy_old_wo_U +=  0.5 * ( velocity_0_main_array * Rcpp_mult_mat_by_col_vec(M_dense_main, velocity_0_main_array.matrix()).array() ).matrix().sum() ;
       energy_old_wo_U +=  0.5 * ( stan::math::square( velocity_0_us_array.matrix() ).array() * ( 1.0 / M_inv_us_array.array()    ) ).matrix().sum() ; /////////////////////////////////////////////////////////////////////////////////////////
     } 
     
     
     
     theta_main_previous_iter = theta_main_array;
     theta_us_previous_iter = theta_us_array;
     
     
     {
       
       
       Eigen::Array<double,  -1, 1  > velocity_us_array =  velocity_us_array_pre_attempts; // reset velocity
       Eigen::Array<double,  -1, 1  > velocity_main_array = velocity_main_array_pre_attempts;  // reset velocity
       
       theta_main_array = theta_main_previous_iter; // reset theta
       theta_us_array = theta_us_previous_iter;  // reset theta
       
       
       div_vec(ii) = 0;  // reset div
       
       
       try {
         
         
         
         // ---------------------------------------------------------------------------------------------------------------///    Perform L leapfrogs   ///-----------------------------------------------------------------------------------------------------------------------------------------
         Eigen::Array<double,  -1, 1  > theta_main_array_proposed = theta_main_array;
         Eigen::Array<double,  -1, 1  > theta_us_array_proposed = theta_us_array;
         
         {
           Eigen::Matrix<double, -1, 1>  neg_lp_and_grad_outs  =  Eigen::Matrix<double, -1, 1>::Zero(n_params + 1 + N) ;   /////////////////////////////////////////////////////////////////////////////////////////////
           
           
            
           for (int l = -1; l < L_main_ii; l++) {
             
             
             if (l > -1) {
               
               for (int nc = 0; nc < n_chunks; nc++) {
                 
                 int chunk_counter = nc ;
                 
                 velocity_us_array.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests)    +=   minus_half_eps *
                   neg_lp_and_grad_outs.segment(1 + chunk_size * n_tests * chunk_counter , chunk_size * n_tests).array() *
                   M_inv_us_array.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests).array()  ;
                 
                 theta_us_array_proposed.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests)    +=   eps   * velocity_us_array.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests) ;
                 
               }
               
               velocity_main_array +=   minus_half_eps *  ( M_inv_dense_main  *    neg_lp_and_grad_outs.segment(n_us + 1, n_params_main).matrix()  ).array() ;
               theta_main_array_proposed  +=   eps  *  velocity_main_array; //// update params by full step
               
             }
             
             
             
             if (Model_type == "MVP_LC")  {
                         neg_lp_and_grad_outs.array()    =    - fn_lp_and_grad_MVP_LC_using_Chol_Spinkney_MD_and_AD(         theta_main_array_proposed ,  
                                                                                                                             theta_us_array_proposed , 
                                                                                                                             y,
                                                                                                                             X,
                                                                                                                             other_args).array() ; 
             } else if (Model_type == "LT_LC") {   
                         neg_lp_and_grad_outs.array()    =    - fn_lp_and_grad_latent_trait_MD_and_AD(                 theta_main_array_proposed ,    
                                                                                                                       theta_us_array_proposed ,  
                                                                                                                       y,
                                                                                                                       X,
                                                                                                                       other_args).array() ; 
             }   else if (Model_type == "MVP_standard") { 
                          neg_lp_and_grad_outs.array()    =    - fn_lp_and_grad_MVP_using_Chol_Spinkney_MD_and_AD(     theta_main_array_proposed ,  
                                                                                                                       theta_us_array_proposed , 
                                                                                                                       y,
                                                                                                                       X,
                                                                                                                       other_args).array() ; 
             }
             
             
             if (l == -1)      log_posterior_0 = - neg_lp_and_grad_outs(0)  ;
             
             
             if (l > -1) {
               
               for (int nc = 0; nc < n_chunks; nc++) {
                 
                 int chunk_counter = nc ;
                 
                 velocity_us_array.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests)    +=   minus_half_eps *
                   neg_lp_and_grad_outs.segment(1 + chunk_size * n_tests * chunk_counter , chunk_size * n_tests).array() *
                   M_inv_us_array.segment(chunk_size * n_tests * chunk_counter , chunk_size * n_tests).array()  ;
                 
               }
               
               velocity_main_array +=   minus_half_eps *  ( M_inv_dense_main  *    neg_lp_and_grad_outs.segment(n_us + 1, n_params_main).matrix()  ).array() ;
               
             }
             
             
             
             
           }
           
           //  individual_log_lik.array() = - neg_lp_and_grad_outs.segment(1 + n_params, N).array().cast<double>()  ;   /////////////////////////////////////////////////////////////////////////////////////////
           log_posterior =  - neg_lp_and_grad_outs(0) ;
           
         }
         
         
         if (Model_type == "LT_LC")  { // for latent trait
           for (int i = n_bs_LT; i < n_corrs; i++) {
             velocity_main_array(i) = 0.0 ;
             velocity_0_main_array(i) = 0.0 ;
             theta_main_array(i) =   R::rnorm(0.0, 1.0);
           }
         }
         // } else if (Model_type == "MVP_standard") { 
         //   for (int i = end_index_for_MVP_standard; i < n_params_main; i++) {
         //     velocity_main_array(i) = 0.0 ;
         //     velocity_0_main_array(i) = 0.0 ;
         //     theta_main_array(i) =   R::rnorm(0.0, 1.0);
         //   } 
         // }
         
         
         //////////////////////////////////////////////////////////////////    M-H acceptance step  (i.e, Accept/Reject step)
         log_posterior_prop = log_posterior ;
         U_x = - log_posterior_prop;
         U_x_initial =  - log_posterior_0;
         
         double energy_old = U_x_initial  + energy_old_wo_U ;
         // double energy_old = U_x_initial ;
         // energy_old +=  0.5 * ( velocity_0_main_array.cast<double>() * Rcpp_mult_mat_by_col_vec(M_dense_main.cast<double>(), velocity_0_main_array.cast<double>().matrix()).array() ).matrix().sum() ;
         // energy_old +=  0.5 * ( stan::math::square( velocity_0_us_array.matrix().cast<double>() ).array() * ( 1 / M_inv_us_array.cast<double>()   ) ).cast<double>().matrix().sum() ; /////////////////////////////////////////////////////////////////////////////////////////
         
         double energy_new = U_x ;
         energy_new +=   0.5 * ( velocity_main_array.cast<double>() * Rcpp_mult_mat_by_col_vec(M_dense_main.cast<double>(), velocity_main_array.cast<double>().matrix()).array() ).matrix().sum() ;
         energy_new +=   0.5 * ( stan::math::square( velocity_us_array.matrix().cast<double>() ).array() * ( 1.0 / M_inv_us_array.cast<double>()   ) ).cast<double>().matrix().sum() ;  /////////////////////////////////////////////////////////////////////////////////////////
         
         
         
         double log_ratio = - energy_new + energy_old;
         
         Eigen::Matrix<double, -1, 1 >  p_jump_vec(2);
         p_jump_vec(0) = 1.0;
         p_jump_vec(1) = std::exp(log_ratio);
         
         double p_jump = stan::math::min(p_jump_vec);
         
         int accept = 0;
         
         
         //Eigen::Matrix<double, -1, 1>  out_mat =   Eigen::Matrix<double, -1, 1>::Zero(n_params + N);   /////////////////////////////////////////////////////////////////////////////////////////
         
         if  ((R::runif(0, 1) > p_jump) ) {  // # reject proposal
           accept = 0;
           theta_us_array = theta_us_previous_iter;// .cast<double>()  ;   /////////////////////////////////////////////////////////////////////////////////////////
           theta_main_array = theta_main_previous_iter; // .cast<double>()  ;
         } else {   // # accept proposal
           accept = 1;
           theta_us_array = theta_us_array_proposed ;//.cast<double>()  ;   /////////////////////////////////////////////////////////////////////////////////////////
           theta_main_array = theta_main_array_proposed ; // .cast<double>()  ;
         }
         
         
         
         ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
         
         //  theta_main_previous_iter =  theta_main_array
         theta_main_trace.col(ii) =  theta_main_array ;
         //  theta_us_previous_iter   =  theta_us_array
         
         div_vec(ii) = 0;
         
         
       } catch (...) {
         
         div_vec(ii) = 1;
         if (ii > 0) theta_main_trace.col(ii) =   theta_main_trace.col(ii - 1)  ;
         //  log_lik_trace.col(ii) =   log_lik_trace.col(ii - 1);
         
         
       }
     }
   }
   
   
   Rcpp::List out_list(3);
   
   out_list(0) = theta_main_trace.cast<double>();
   //out_list(1) = log_lik_trace;
   
   out_list(2) = div_vec;
   
   return(out_list);
   
 }
 
 
 
 
 
 
 
 
 


 
  