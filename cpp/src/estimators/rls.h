#pragma once

#include "estimator.h"

namespace estim {

class RLS : public Estimator {
public:
    // Constructs RLS estimator.
    // Params:
    //     n:
    //     lambda:
    //     p_coeff:
    //     x0:
    RLS(int n, double lambda, double p_coeff,
        const std::optional<RowMatrixXd>& x0 = std::nullopt);

    // Iterate method implementation for RLS.
    RowMatrixXd iterate(
        const Eigen::Ref<const RowMatrixXd>& A,
        const Eigen::Ref<const RowMatrixXd>& b,
        const std::optional<RowMatrixXd>& x0) override;

private:
    double lambda_;
    RowMatrixXd P_;
};

class RLS_Robust final : public Estimator {
public:
    // Constructs Robust RLS estimator.
    // Params:
    //     n:
    //     lambda:
    //     p_coeff:
    //     x0:
    RLS_Robust(int n,
               double lambda = 0.99,
               double p_coeff = 1000.0,
               const std::optional<RowMatrixXd>& x0 = std::nullopt);

    // Iterate method implementation for Robust RLS.
    RowMatrixXd iterate(
        const Eigen::Ref<const RowMatrixXd>& A,
        const Eigen::Ref<const RowMatrixXd>& b,
        const std::optional<RowMatrixXd>& x0) override;

private:
    double lambda_;
    RowMatrixXd P_;
};

} // namespace estim
