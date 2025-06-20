

#pragma once

 
 

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <limits>
#include <string>


 


 
 
 
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











struct ModelArgs {
  std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> Model_args_2_layer_vecs_of_mats_double;
  std::vector<bool> Model_args_bools;
  std::vector<int> Model_args_ints;
  std::vector<double> Model_args_doubles;
  std::vector<std::string> Model_args_strings;
  std::vector<Eigen::Matrix<double, -1, 1>> Model_args_col_vecs_double;
  std::vector<Eigen::Matrix<int, -1, -1>> Model_args_mats_int;
  std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> Model_args_vecs_of_mats_double;
  std::vector<Eigen::Matrix<int, -1, -1>> Model_args_vecs_of_mats_int;
};



//  C++ struct to hold the unpacked values
struct unpacked_arg {
  
  int N, n_tests, n_us, n_params_main, n_params;
  std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> X;
  bool exclude_priors, CI, corr_force_positive, corr_prior_beta, corr_prior_norm;
  bool handle_numerical_issues, skip_checks_exp, skip_checks_log, skip_checks_lse, skip_checks_tanh;
  bool skip_checks_Phi, skip_checks_log_Phi, skip_checks_inv_Phi, skip_checks_inv_Phi_approx_from_logit_prob, debug;
  int n_cores, n_class, ub_threshold_phi_approx, n_chunks;
  double prev_prior_a, prev_prior_b, overflow_threshold, underflow_threshold;
  std::string vect_type, Phi_type, inv_Phi_type, vect_type_exp, vect_type_log;
  std::string vect_type_lse, vect_type_tanh, vect_type_Phi, vect_type_log_Phi, vect_type_inv_Phi, vect_type_inv_Phi_approx_from_logit_prob, nuisance_transformation;
  Eigen::Matrix<double, -1, 1> lkj_cholesky_eta; 
  Eigen::Matrix<int, -1, -1> n_covariates_per_outcome_vec;
  std::vector<Eigen::Matrix<double, -1, -1>> prior_coeffs_mean, prior_coeffs_sd, prior_for_corr_a, prior_for_corr_b, lb_corr, ub_corr, known_values;
  std::vector<Eigen::Matrix<int, -1, -1>> known_values_indicator;
  int n_corrs;
  int n_covariates_total_nd, n_covariates_total_d, n_covariates_total;
  int n_covariates_max_nd, n_covariates_max_d, n_covariates_max;
  double sqrt_2_pi_recip, sqrt_2_recip, minus_sqrt_2_recip, a, b, a_times_3, s, Inf;
  int chunk_size, chunk_size_orig, n_total_chunks, n_full_chunks, last_chunk_size;
  
};




// Function to initialize and unpack arguments
void initializeAndUnpackArgs(const ModelArgs& model_args, 
                             const Eigen::Matrix<int, -1, -1>& y_ref, 
                             const Eigen::Matrix<double, -1, 1>& theta_us_vec_ref,
                             const Eigen::Matrix<double, -1, 1>& theta_main_vec_ref, 
                             unpacked_arg& unpacked) {
  
  // Unpack basic parameters
  unpacked.N = y_ref.rows();
  unpacked.n_tests = y_ref.cols();
  unpacked.n_us = theta_us_vec_ref.rows();
  unpacked.n_params_main = theta_main_vec_ref.rows();
  unpacked.n_params = unpacked.n_params_main + unpacked.n_us;
  
  // Unpack struct values
  unpacked.X = model_args.Model_args_2_layer_vecs_of_mats_double[0];
  unpacked.exclude_priors = model_args.Model_args_bools[0];
  unpacked.CI = model_args.Model_args_bools[1];
  unpacked.corr_force_positive = model_args.Model_args_bools[2];
  unpacked.corr_prior_beta = model_args.Model_args_bools[3];
  unpacked.corr_prior_norm = model_args.Model_args_bools[4];
  unpacked.handle_numerical_issues = model_args.Model_args_bools[5];
  unpacked.skip_checks_exp = model_args.Model_args_bools[6];
  unpacked.skip_checks_log = model_args.Model_args_bools[7];
  unpacked.skip_checks_lse = model_args.Model_args_bools[8];
  unpacked.skip_checks_tanh = model_args.Model_args_bools[9];
  unpacked.skip_checks_Phi = model_args.Model_args_bools[10];
  unpacked.skip_checks_log_Phi = model_args.Model_args_bools[11];
  unpacked.skip_checks_inv_Phi = model_args.Model_args_bools[12];
  unpacked.skip_checks_inv_Phi_approx_from_logit_prob = model_args.Model_args_bools[13];
  unpacked.debug = model_args.Model_args_bools[14];
  
  // Unpack integer and double parameters
  unpacked.n_cores = model_args.Model_args_ints[0];
  unpacked.n_class = model_args.Model_args_ints[1];
  unpacked.ub_threshold_phi_approx = model_args.Model_args_ints[2];
  unpacked.n_chunks = model_args.Model_args_ints[3];
  unpacked.prev_prior_a = model_args.Model_args_doubles[0];
  unpacked.prev_prior_b = model_args.Model_args_doubles[1];
  unpacked.overflow_threshold = model_args.Model_args_doubles[2];
  unpacked.underflow_threshold = model_args.Model_args_doubles[3];
  
  // Unpack strings
  unpacked.vect_type = model_args.Model_args_strings[0];
  unpacked.Phi_type = model_args.Model_args_strings[1];
  unpacked.inv_Phi_type = model_args.Model_args_strings[2];
  unpacked.vect_type_exp = model_args.Model_args_strings[3];
  unpacked.vect_type_log = model_args.Model_args_strings[4];
  unpacked.vect_type_lse = model_args.Model_args_strings[5];
  unpacked.vect_type_tanh = model_args.Model_args_strings[6];
  unpacked.vect_type_Phi = model_args.Model_args_strings[7];
  unpacked.vect_type_log_Phi = model_args.Model_args_strings[8];
  unpacked.vect_type_inv_Phi = model_args.Model_args_strings[9];
  unpacked.vect_type_inv_Phi_approx_from_logit_prob = model_args.Model_args_strings[10];
  unpacked.nuisance_transformation = model_args.Model_args_strings[12];
  
  // Unpack other fields
  unpacked.lkj_cholesky_eta = model_args.Model_args_col_vecs_double[0];
  unpacked.n_covariates_per_outcome_vec = model_args.Model_args_mats_int[0];
  
  // Constants and calculations
  unpacked.sqrt_2_pi_recip = 1.0 / std::sqrt(2.0 * M_PI);
  unpacked.sqrt_2_recip = 1.0 / std::sqrt(2.0);
  unpacked.minus_sqrt_2_recip = -unpacked.sqrt_2_recip;
  unpacked.a = 0.07056;
  unpacked.b = 1.5976;
  unpacked.a_times_3 = 3.0 * 0.07056;
  unpacked.s = 1.0 / 1.702;
  unpacked.Inf = std::numeric_limits<double>::infinity();
  
  // Chunk calculations based on vectorization type
  int vec_size = 1;  // Default
  if (unpacked.vect_type == "AVX512") vec_size = 8;
  else if (unpacked.vect_type == "AVX2") vec_size = 4;
  else if (unpacked.vect_type == "AVX") vec_size = 2;
  
  double N_double = static_cast<double>(unpacked.N);
  double vec_size_double = static_cast<double>(vec_size);
  double desired_n_chunks_double = static_cast<double>(unpacked.n_chunks);
  
  unpacked.chunk_size = vec_size * static_cast<int>(std::floor(N_double / (vec_size_double * desired_n_chunks_double)));
  unpacked.n_full_chunks = static_cast<int>(std::floor(N_double / unpacked.chunk_size));
  unpacked.last_chunk_size = unpacked.N - (unpacked.n_full_chunks * unpacked.chunk_size);
  unpacked.n_total_chunks = unpacked.last_chunk_size == 0 ? unpacked.n_full_chunks : unpacked.n_full_chunks + 1;
  unpacked.chunk_size_orig = unpacked.chunk_size;
  
}


