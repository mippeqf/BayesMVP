 


class HMCResult {
   
   
           /////////// --------------  HELPER FNS - local to this class 
           // HMC helper fns
           void store_current_state() {
             main_theta_vec_0_ = main_theta_vec_;
             us_theta_vec_0_ = us_theta_vec_;
             main_velocity_0_vec_ = main_velocity_vec_;
             us_velocity_0_vec_ = us_velocity_vec_;
           } 
           
           void accept_proposal_main() {
             main_theta_vec_ = main_theta_vec_proposed_;
             main_velocity_vec_ = main_velocity_vec_proposed_;
           } 
  
           void accept_proposal_us() {
             us_theta_vec_ = us_theta_vec_proposed_;
             us_velocity_vec_ = us_velocity_vec_proposed_;
           } 
           
           void reject_proposal_main() {
             main_theta_vec_ = main_theta_vec_0_;
             main_velocity_vec_ = main_velocity_0_vec_;
           } 
  
           void reject_proposal_us() {
             us_theta_vec_ = us_theta_vec_0_;
             us_velocity_vec_ = us_velocity_0_vec_;
           } 
           
           // fn to check state validity
           bool check_state_valid() const {
             return main_theta_vec_.allFinite() && 
               main_velocity_vec_.allFinite() && 
               us_theta_vec_.allFinite() && 
               us_velocity_vec_.allFinite();
           }
           
           
         /////////// -------------- PRIVATE members (ONLY accessible WITHIN this class)
         private:
           // Making states private prevents accidental misuse
           Eigen::Matrix<double, -1, 1> lp_and_grad_outs_;
           
           Eigen::Matrix<double, -1, 1> main_theta_vec_0_;
           Eigen::Matrix<double, -1, 1> main_theta_vec_;
           Eigen::Matrix<double, -1, 1> main_theta_vec_proposed_;
           Eigen::Matrix<double, -1, 1> main_velocity_0_vec_;
           Eigen::Matrix<double, -1, 1> main_velocity_vec_proposed_;
           Eigen::Matrix<double, -1, 1> main_velocity_vec_;
           
           double main_p_jump_;
           int main_div_;
           
           Eigen::Matrix<double, -1, 1> us_theta_vec_0_;
           Eigen::Matrix<double, -1, 1> us_theta_vec_;
           Eigen::Matrix<double, -1, 1> us_theta_vec_proposed_;
           Eigen::Matrix<double, -1, 1> us_velocity_0_vec_;
           Eigen::Matrix<double, -1, 1> us_velocity_vec_proposed_;
           Eigen::Matrix<double, -1, 1> us_velocity_vec_;
           
           double us_p_jump_;
           int us_div_;
           
         /////////// --------------- PUBLIC members (accessible from OUTSIDE this class [e.g. can do: "class.public_member"])
         public:
           //// Constructor 
           HMCResult(int n_params_main, 
                     int n_us, 
                     int N)
           : lp_and_grad_outs_(Eigen::Matrix<double, -1, 1>::Zero((1 + N + n_params_main + n_us)))
           //// main
           , main_theta_vec_0_(Eigen::Matrix<double, -1, 1>::Zero(n_params_main))
           , main_theta_vec_(Eigen::Matrix<double, -1, 1>::Zero(n_params_main))
           , main_theta_vec_proposed_(Eigen::Matrix<double, -1, 1>::Zero(n_params_main))
           , main_velocity_0_vec_(Eigen::Matrix<double, -1, 1>::Zero(n_params_main))
           , main_velocity_vec_proposed_(Eigen::Matrix<double, -1, 1>::Zero(n_params_main))
           , main_velocity_vec_(Eigen::Matrix<double, -1, 1>::Zero(n_params_main))
           , main_p_jump_(0.0)
           , main_div_(0)
           //// nuisance
           , us_theta_vec_0_(Eigen::Matrix<double, -1, 1>::Zero(n_us))
           , us_theta_vec_(Eigen::Matrix<double, -1, 1>::Zero(n_us))
           , us_theta_vec_proposed_(Eigen::Matrix<double, -1, 1>::Zero(n_us))
           , us_velocity_0_vec_(Eigen::Matrix<double, -1, 1>::Zero(n_us))
           , us_velocity_vec_proposed_(Eigen::Matrix<double, -1, 1>::Zero(n_us))
           , us_velocity_vec_(Eigen::Matrix<double, -1, 1>::Zero(n_us))
           , us_p_jump_(0.0)
           , us_div_(0) {}
           
           //// "getters"/"setters" to access the private members
           Eigen::Matrix<double, -1, 1> &lp_and_grad_outs() { return lp_and_grad_outs_; }
           const Eigen::Matrix<double, -1, 1> &lp_and_grad_outs() const { return lp_and_grad_outs_; }
           //// main
           Eigen::Matrix<double, -1, 1> &main_theta_vec_0() { return main_theta_vec_0_; }
           const Eigen::Matrix<double, -1, 1> &main_theta_vec_0() const { return main_theta_vec_0_; }
           Eigen::Matrix<double, -1, 1> &main_theta_vec() { return main_theta_vec_; }
           const Eigen::Matrix<double, -1, 1> &main_theta_vec() const { return main_theta_vec_; }
           Eigen::Matrix<double, -1, 1> &main_theta_vec_proposed() { return main_theta_vec_proposed_; }
           const Eigen::Matrix<double, -1, 1> &main_theta_vec_proposed() const { return main_theta_vec_proposed_; }
           Eigen::Matrix<double, -1, 1> &main_velocity_0_vec() { return main_velocity_0_vec_; }
           const Eigen::Matrix<double, -1, 1> &main_velocity_0_vec() const { return main_velocity_0_vec_; }
           Eigen::Matrix<double, -1, 1> &main_velocity_vec_proposed() { return main_velocity_vec_proposed_; }
           const Eigen::Matrix<double, -1, 1> &main_velocity_vec_proposed() const { return main_velocity_vec_proposed_; }
           Eigen::Matrix<double, -1, 1> &main_velocity_vec() { return main_velocity_vec_; }
           const Eigen::Matrix<double, -1, 1> &main_velocity_vec() const { return main_velocity_vec_; }
           double &main_p_jump() { return main_p_jump_; }
           const double &main_p_jump() const { return main_p_jump_; }
           int &main_div() { return main_div_; }
           const int &main_div() const { return main_div_; }
           //// nuisance
           Eigen::Matrix<double, -1, 1> &us_theta_vec_0() { return us_theta_vec_0_; }
           const Eigen::Matrix<double, -1, 1> &us_theta_vec_0() const { return us_theta_vec_0_; }
           Eigen::Matrix<double, -1, 1> &us_theta_vec() { return us_theta_vec_; }
           const Eigen::Matrix<double, -1, 1> &us_theta_vec() const { return us_theta_vec_; }
           Eigen::Matrix<double, -1, 1> &us_theta_vec_proposed() { return us_theta_vec_proposed_; }
           const Eigen::Matrix<double, -1, 1> &us_theta_vec_proposed() const { return us_theta_vec_proposed_; }
           Eigen::Matrix<double, -1, 1> &us_velocity_0_vec() { return us_velocity_0_vec_; }
           const Eigen::Matrix<double, -1, 1> &us_velocity_0_vec() const { return us_velocity_0_vec_; }
           Eigen::Matrix<double, -1, 1> &us_velocity_vec_proposed() { return us_velocity_vec_proposed_; }
           const Eigen::Matrix<double, -1, 1> &us_velocity_vec_proposed() const { return us_velocity_vec_proposed_; }
           Eigen::Matrix<double, -1, 1> &us_velocity_vec() { return us_velocity_vec_; }
           const Eigen::Matrix<double, -1, 1> &us_velocity_vec() const { return us_velocity_vec_; }
           double &us_p_jump() { return us_p_jump_; }
           const double &us_p_jump() const { return us_p_jump_; }
           int &us_div() { return us_div_; }
           const int &us_div() const { return us_div_; }

   
};























