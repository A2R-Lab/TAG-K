#pragma once

#include <random>

#include "estimator.h"

namespace estim {

class RK : public Estimator {
public:
    // Constructs RK estimator.
    // Params:
    //     n:
    //     x0:
    RK(int n, const std::optional<RowMatrixXd>& x0 = std::nullopt);

    // Iterate method implementation for RK.
    RowMatrixXd iterate(
        const Eigen::Ref<const RowMatrixXd>& A,
        const Eigen::Ref<const RowMatrixXd>& b,
        const std::optional<RowMatrixXd>& x0) override;
    
private:
    std::mt19937 rng_eng_;
};

} // namespace estim
