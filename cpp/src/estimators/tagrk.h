#pragma once

#include <random>

#include "estimator.h"

namespace estim {

struct TAGRK : public Estimator {
public:
    // Constructs TAGRK estimator.
    // Params:
    //     n:
    //     burnin_steps:
    //     tolerance:
    //     x0:
    TAGRK(int n, int burnin_steps, double tolerance, const std::optional<RowMatrixXd>& x0 = std::nullopt);

    // Iterate method implementation for TAGRK.
    RowMatrixXd iterate(
        const Eigen::Ref<const RowMatrixXd>& A,
        const Eigen::Ref<const RowMatrixXd>& b,
        const std::optional<RowMatrixXd>& x0) override;

private:
    int burnin_steps_;
    double tolerance_;
    std::mt19937 rng_eng_;
};

} // namespace estim
