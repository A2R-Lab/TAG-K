#include "kf.h"

namespace estim {

// --- KF Implementation ---

KF::KF(int n, const RowMatrixXd& process_noise, const RowMatrixXd& measurement_noise, double p_coeff, const std::optional<RowMatrixXd>& x0)
    : Estimator(n, x0),
      P_(RowMatrixXd::Identity(n, n) * p_coeff),
      Q_(process_noise),
      R_(measurement_noise) {

    if (p_coeff < 0)
        throw std::invalid_argument("p_coeff cannot be negative.");

    // ensure square for noise matrices and that Q is (n, n) specifically
    validate_shape(Q_, n, n, "Process noise");
    validate_shape(R_, R_.rows(), R_.rows(), "Measurement noise");
}

RowMatrixXd KF::iterate(
    const Eigen::Ref<const RowMatrixXd>& A,
    const Eigen::Ref<const RowMatrixXd>& b,
    const std::optional<RowMatrixXd>& x0) {
    
    if (x0.has_value())
        validate_shape(*x0, n_, 1, "x0");

    RowMatrixXd x = x0.has_value() ? *x0 : this->x_;

    // 1. Predict
    this->P_ += this->Q_; // P_k|k-1 = P_k-1|k-1 + Q (since F=I)

    // 2. Update
    RowMatrixXd S = A * this->P_ * A.transpose() + this->R_; // Innovation covariance
    RowMatrixXd K = this->P_ * A.transpose() * S.inverse(); // Kalman Gain
    x += K * (b - A * x);
    this->P_ = this->P_ - K * A * this->P_; // Simplified covariance update

    // Update persistent state and return result
    this->x_ = x;
    return this->x_;
}

// --- KF_Robust Implementation ---

KF_Robust::KF_Robust(int n, const RowMatrixXd& process_noise, const RowMatrixXd& measurement_noise, double p_coeff, const std::optional<RowMatrixXd>& x0)
    : Estimator(n, x0),
      P_(RowMatrixXd::Identity(n, n) * p_coeff),
      Q_(process_noise),
      R_(measurement_noise) {
    if (p_coeff < 0)
        throw std::invalid_argument("p_coeff cannot be negative.");

    // ensure square for noise matrices and Q is (n, n) specifically
    validate_shape(Q_, n, n, "Process noise");
    validate_shape(R_, R_.rows(), R_.rows(), "Measurement noise");
}

RowMatrixXd KF_Robust::iterate(
    const Eigen::Ref<const RowMatrixXd>& A,
    const Eigen::Ref<const RowMatrixXd>& b,
    const std::optional<RowMatrixXd>& x0) {

    if (x0.has_value())
        validate_shape(*x0, n_, 1, "x0");
        
    RowMatrixXd x = x0.has_value() ? *x0 : this->x_;

    // 1. Predict
    this->P_ += this->Q_;

    // 2. Update
    RowMatrixXd S = A * this->P_ * A.transpose() + this->R_;
    
    // Solve for Gain K without explicit inversion
    Eigen::LLT<RowMatrixXd> lltOfS(S);
    RowMatrixXd K = lltOfS.solve(A * this->P_).transpose();

    x += K * (b - A * x);

    // Use the numerically stable Joseph form for covariance update
    RowMatrixXd I = RowMatrixXd::Identity(n_, n_);
    RowMatrixXd I_KH = I - K * A;
    this->P_ = I_KH * this->P_ * I_KH.transpose() + K * this->R_ * K.transpose();

    this->x_ = x;
    return this->x_;
}

} // namespace estim
