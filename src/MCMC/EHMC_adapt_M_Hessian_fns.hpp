
#pragma once

 

#include <Eigen/Dense>
 
 
 
 
using namespace Eigen;

 

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
 


 
 
// Helper to check positive-definiteness using Eigen's LDLT decomposition
bool is_positive_definite(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  mat) {
  
        Eigen::LLT<Eigen::Matrix<double, -1, -1> > llt((mat + mat.transpose()) * 0.50);  /// make symmetric first  and then attempt Cholesky factorisation
        
        return llt.info() == Eigen::Success;
  
}

 
// Helper function to make a matrix positive-definite
Eigen::Matrix<double, -1, -1> near_PD(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  mat) {
  
         // Eigen::Matrix<double, -1, -1>  symMat = (mat + mat.transpose()) / 2.0; 
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, -1, -1> > es((mat + mat.transpose()) * 0.50); // Make symmetric and apply es
        Eigen::Matrix<double, -1, 1>   eigenValues = es.eigenvalues();
        Eigen::Matrix<double, -1, -1>  eigenVectors = es.eigenvectors();
        
        
        double max_eigenvalue = eigenValues.maxCoeff();
        double epsilon = std::max(1e-8, 1e-6 * max_eigenvalue);
        for (int i = 0; i < eigenValues.size(); ++i) {
          eigenValues(i) = std::max(eigenValues(i), epsilon);
        }
        
        // // Shift eigenvalues to ensure all are positive
        // for (int i = 0; i < eigenValues.size(); ++i) {
        //   if (eigenValues(i) < 0.0) {
        //     eigenValues(i) = 1e-6; // Set negative eigenvalues to a small positive value
        //   }
        // } 
        
        // Recompose the matrix with positive eigenvalues
        return eigenVectors * eigenValues.asDiagonal() * eigenVectors.transpose();
  
}


 





Eigen::Matrix<double, -1, -1> shrink_hessian( const Eigen::Matrix<double, -1, -1> &hessian, 
                                              double shrinkage_factor) {
  
        Eigen::Matrix<double, -1, -1> diagonal = hessian.diagonal().asDiagonal();
        return (1 - shrinkage_factor) * hessian + shrinkage_factor * diagonal;
  
}







Eigen::Matrix<double, -1, -1> num_diff_Hessian_main_given_nuisance(   const double num_diff_e,
                                                                      const double shrinkage_factor,
                                                                      const std::string &Model_type,
                                                                      const bool force_autodiff,
                                                                      const bool force_PartialLog,
                                                                      const bool multi_attemps, 
                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                      const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                     // MVP_ThreadLocalWorkspace &MVP_workspace,
                                                                      const Stan_model_struct &Stan_model_as_cpp_struct) {
  
  const int n_params_main = theta_main_vec_ref.rows();
  const int n_us = theta_us_vec_ref.rows();
  const int n_params = n_params_main + n_us;
  const int N = y_ref.rows();
  const std::string grad_option = "main_only";
  
  Eigen::Matrix<double, -1, 1> lp_and_grad_outs = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
  Eigen::Matrix<double, -1, -1> Hessian(n_params_main, n_params_main);
  Eigen::Matrix<double, -1, 1> theta_main_vec_perturbed = theta_main_vec_ref;
  
  double num_diff_e_inv = 1.0 / (2 * num_diff_e);
  
  Eigen::Matrix<double, -1, 1> grad_plus(n_params_main);
  Eigen::Matrix<double, -1, 1> grad_minus(n_params_main);
  
  for (int j = 0; j < n_params_main; ++j) {
    // Compute g(x + hⱼeⱼ)
    theta_main_vec_perturbed(j) = theta_main_vec_ref(j) + num_diff_e;
    fn_lp_grad_InPlace(lp_and_grad_outs, 
                       Model_type,
                       force_autodiff, force_PartialLog, multi_attemps,
                       theta_main_vec_perturbed, theta_us_vec_ref, y_ref, grad_option, 
                       Model_args_as_cpp_struct, //MVP_workspace, 
                       Stan_model_as_cpp_struct);
    grad_plus.array() = - lp_and_grad_outs.segment(1 + n_us, n_params_main).array();
    
    // Compute g(x - hⱼeⱼ)
    theta_main_vec_perturbed(j) = theta_main_vec_ref(j) - num_diff_e;
    fn_lp_grad_InPlace(lp_and_grad_outs, 
                       Model_type, 
                       force_autodiff, force_PartialLog, multi_attemps,
                       theta_main_vec_perturbed, theta_us_vec_ref, y_ref, grad_option, 
                       Model_args_as_cpp_struct, //MVP_workspace,
                       Stan_model_as_cpp_struct);
    grad_minus.array() = - lp_and_grad_outs.segment(1 + n_us, n_params_main).array();
    
    // Compute j-th column of Hessian
    Hessian.col(j) = (grad_plus - grad_minus) * num_diff_e_inv;
    
    // Reset perturbed vector
    theta_main_vec_perturbed(j) = theta_main_vec_ref(j);
  }
  
  // Symmetrize the Hessian
  Hessian = (Hessian + Hessian.transpose()) * 0.5;
  
  // Apply shrinkage
  Hessian = shrink_hessian(Hessian, shrinkage_factor);
  
  // Symmetrize the Hessian
  Hessian = (Hessian + Hessian.transpose()) * 0.5;
  
  return Hessian;
  
}












Eigen::Matrix<double, -1, -1>  compute_PD_Hessian_main(     const double shrinkage_factor,
                                                            const double num_diff_e,
                                                            const std::string  &Model_type,
                                                            const bool force_autodiff,
                                                            const bool force_PartialLog,
                                                            const bool multi_attemps, 
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                            const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                            const Model_fn_args_struct  &Model_args_as_cpp_struct
) {
  
  
          const int n_params_main = theta_main_vec_ref.rows();
          Eigen::Matrix<double, -1, -1>    Hessian(n_params_main, n_params_main);
          
          
          if (Model_type == "Stan") {
                  
                  Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct.model_so_file, 
                                                                                             Model_args_as_cpp_struct.json_file_path, 
                                                                                             123);
                  
                  
                  //// compute proposed Hessian
                  Hessian  = num_diff_Hessian_main_given_nuisance(    num_diff_e,
                                                                      shrinkage_factor,
                                                                      Model_type,
                                                                      force_autodiff,
                                                                      force_PartialLog,
                                                                      multi_attemps,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref, 
                                                                      y_ref,
                                                                      Model_args_as_cpp_struct, 
                                                                      Stan_model_as_cpp_struct);
                  
                  
                  
                  //// destroy Stan model object
                  fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
            
          } else { 
            
                  Stan_model_struct Stan_model_as_cpp_struct; /// dummy struct 
                  
                  //// compute proposed Hessian
                  Hessian  = num_diff_Hessian_main_given_nuisance(    num_diff_e,
                                                                      shrinkage_factor,
                                                                      Model_type,
                                                                      force_autodiff,
                                                                      force_PartialLog,
                                                                      multi_attemps,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref, 
                                                                      y_ref,
                                                                      Model_args_as_cpp_struct, 
                                                                      Stan_model_as_cpp_struct);
            
          }
  
  
          // force-symmetric positive-definiteness check
          if (!is_positive_definite(Hessian)) {
            Hessian = near_PD(Hessian); // Make the Hessian positive-definite
          }
          
          return Hessian;
          
  
} 





















void update_M_dense_main_Hessian_InPlace( Eigen::Ref<Eigen::Matrix<double, -1, -1>> M_dense_main,  /// to be updated 
                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> M_inv_dense_main, /// to be updated 
                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> M_inv_dense_main_chol, /// to be updated 
                                          const double shrinkage_factor,
                                          const double ratio,
                                          const int interval_width,
                                          const double num_diff_e,
                                          const std::string  &Model_type,
                                          const bool force_autodiff,
                                          const bool force_PartialLog,
                                          const bool multi_attemps, 
                                          const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                          const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                          const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                          const Model_fn_args_struct  &Model_args_as_cpp_struct,
                                          const double   &ii, 
                                          const double   &n_burnin, 
                                          const std::string &metric_type) {
  
  
                      const int n_params_main = theta_main_vec_ref.rows();
                      Eigen::Matrix<double, -1, -1>    Hessian(n_params_main, n_params_main);
                      
                      
                      if (Model_type == "Stan") {
                      
                                    Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct.model_so_file, 
                                                                                                               Model_args_as_cpp_struct.json_file_path, 
                                                                                                               123);
                   
                                    
                                    //// compute proposed Hessian
                                    Hessian  = num_diff_Hessian_main_given_nuisance(    num_diff_e,
                                                                                        shrinkage_factor,
                                                                                        Model_type,
                                                                                        force_autodiff,
                                                                                        force_PartialLog,
                                                                                        multi_attemps,
                                                                                        theta_main_vec_ref,
                                                                                        theta_us_vec_ref, 
                                                                                        y_ref,
                                                                                        Model_args_as_cpp_struct, 
                                                                                        Stan_model_as_cpp_struct);
                    
                    
                  
                                    //// destroy Stan model object
                                    fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
                                      
                      } else { 
                        
                                    Stan_model_struct Stan_model_as_cpp_struct; /// dummy struct 
                        
                                    //// compute proposed Hessian
                                    Hessian  = num_diff_Hessian_main_given_nuisance(    num_diff_e,
                                                                                        shrinkage_factor,
                                                                                        Model_type,
                                                                                        force_autodiff,
                                                                                        force_PartialLog,
                                                                                        multi_attemps,
                                                                                        theta_main_vec_ref,
                                                                                        theta_us_vec_ref, 
                                                                                        y_ref,
                                                                                        Model_args_as_cpp_struct, 
                                                                                        Stan_model_as_cpp_struct);
                        
                      }
  
                            
                      // force-symmetric positive-definiteness check
                      if (!is_positive_definite(Hessian)) {
                        Hessian = near_PD(Hessian); // Make the Hessian positive-definite
                      }
                      
                      // update M_dense_main
                      M_dense_main = (1.0 - ratio) * M_dense_main + ratio * Hessian;  
                      
                     
                      if (!is_positive_definite(M_dense_main)) {
                        M_dense_main = near_PD(M_dense_main);
                      }
                      
                      // update M_inv_dense_main
                      M_inv_dense_main = M_dense_main.inverse();
                      
                      // update M_inv_dense_main_chol
                      Eigen::LLT<Eigen::MatrixXd> llt(M_inv_dense_main);
                      if (llt.info() == Eigen::Success) {
                        
                        M_inv_dense_main_chol = llt.matrixL();
                        
                      } else {
                        
                         throw std::runtime_error("Cholesky decomposition failed");
                        
                      }
 
  
}
















 





