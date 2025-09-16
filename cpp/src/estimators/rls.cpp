#include "rls.h"

namespace estim {

// --- RLS Implementation ---

RLS::RLS(int n, double lambda, double p_coeff, const std::optional<RowMatrixXd>& x0)
    : Estimator(n, x0),
      lambda_(lambda),
      P_(RowMatrixXd::Identity(n, n) * p_coeff) {

    if (lambda < 0)
        throw std::invalid_argument("Lambda cannot be negative.");

    if (p_coeff < 0)
        throw std::invalid_argument("p_coeff cannot be negative.");
}

RowMatrixXd RLS::iterate(
    const Eigen::Ref<const RowMatrixXd>& A,
    const Eigen::Ref<const RowMatrixXd>& b,
    const std::optional<RowMatrixXd>& x0) {
    
    if (x0.has_value())
        validate_shape(*x0, n_, 1, "x0");

    RowMatrixXd x = x0.has_value() ? *x0 : this->x_;
    
    const long m = A.rows();

    // S = A * P * A' + lambda * I
    RowMatrixXd S = A * this->P_ * A.transpose() +
                    lambda_ * RowMatrixXd::Identity(m, m);

    // K = P * A' * S^-1 (Kalman Gain)
    RowMatrixXd K = this->P_ * A.transpose() * S.inverse();

    // x = x + K * (b - A * x)
    x += K * (b - A * x);

    // P = (P - K * A * P) / lambda
    this->P_ = (this->P_ - K * A * this->P_) / lambda_;

    // Update persistent state and return result
    this->x_ = x;
    return this->x_;
}


// --- RLS_Robust Implementation ---

RLS_Robust::RLS_Robust(int n, double lambda, double p_coeff, const std::optional<RowMatrixXd>& x0)
    : Estimator(n, x0),
      lambda_(lambda),
      P_(RowMatrixXd::Identity(n, n) * p_coeff) {

    if (lambda < 0)
        throw std::invalid_argument("Lambda cannot be negative.");

    if (p_coeff < 0)
        throw std::invalid_argument("p_coeff cannot be negative.");
}

RowMatrixXd RLS_Robust::iterate(
    const Eigen::Ref<const RowMatrixXd>& A,
    const Eigen::Ref<const RowMatrixXd>& b,
    const std::optional<RowMatrixXd>& x0) {
    
    if (x0.has_value())
        validate_shape(*x0, n_, 1, "x0");

    RowMatrixXd x = x0.has_value() ? *x0 : this->x_;

    const long m = A.rows();

    // S = A * P * A' + lambda * I
    RowMatrixXd S = A * this->P_ * A.transpose() +
                    lambda_ * RowMatrixXd::Identity(m, m);

    // Solve for Gain K without explicit inversion: S * K' = A * P
    Eigen::LLT<RowMatrixXd> lltOfS(S);
    RowMatrixXd K = lltOfS.solve(A * this->P_).transpose();

    // x = x + K * (b - A * x)
    x += K * (b - A * x);

    // P = (P - K * A * P) / lambda
    this->P_ = (this->P_ - K * A * this->P_) / lambda_;

    // Update persistent state and return result
    this->x_ = x;
    return this->x_;
}

} // namespace estim
