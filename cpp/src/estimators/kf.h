#pragma once

#include "estimator.h"

namespace estim {

class KF final : public Estimator {
public:
    // Constructs KF estimator.
    // Params:
    //     n: 
    //     process_noise:     (n, n) matrix
    //     measurement_noise: (m, m) matrix
    //     p_coeff:
    //     x0:
    KF(int n,
       const RowMatrixXd& process_noise,
       const RowMatrixXd& measurement_noise,
       double p_coeff = 100.0,
       const std::optional<RowMatrixXd>& x0 = std::nullopt);

    /// Iterate method implementation for KF.
    RowMatrixXd iterate(
        const Eigen::Ref<const RowMatrixXd>& A, // Measurement matrix H
        const Eigen::Ref<const RowMatrixXd>& b, // Measurement z
        const std::optional<RowMatrixXd>& x0) override;

private:
    RowMatrixXd P_; // State covariance
    RowMatrixXd Q_; // Process noise covariance
    RowMatrixXd R_; // Measurement noise covariance
};

class KF_Robust final : public Estimator {
public:
    // Constructs Robust KF estimator.
    // Params:
    //     n: 
    //     process_noise:     (n, n) matrix
    //     measurement_noise: (m, m) matrix
    //     p_coeff:
    //     x0:
    KF_Robust(int n,
              const RowMatrixXd& process_noise,
              const RowMatrixXd& measurement_noise,
              double p_coeff = 100.0,
              const std::optional<RowMatrixXd>& x0 = std::nullopt);

    /// Iterate method for implementation Robust KF.
    RowMatrixXd iterate(
        const Eigen::Ref<const RowMatrixXd>& A,
        const Eigen::Ref<const RowMatrixXd>& b,
        const std::optional<RowMatrixXd>& x0) override;

private:
    RowMatrixXd P_;
    RowMatrixXd Q_;
    RowMatrixXd R_;
};

} // namespace estim
