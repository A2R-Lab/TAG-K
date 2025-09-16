#pragma once

#include <random>

#include "estimator.h"

namespace estim {

class TARK : public Estimator {
public:
    // Constructs TARK estimator.
    // Params:
    //     n:
    //     burnin_steps:
    //     x0:
    TARK(int n, int burnin_steps, const std::optional<RowMatrixXd>& x0 = std::nullopt);

    // Iterate method implementation for TARK.
    RowMatrixXd iterate(
        const Eigen::Ref<const RowMatrixXd>& A,
        const Eigen::Ref<const RowMatrixXd>& b,
        const std::optional<RowMatrixXd>& x0) override;

private:
    int burnin_steps_;
    std::mt19937 rng_eng_;
};

} // namespace estim
