#pragma once

#include <random>

#include "estimator.h"

namespace estim {

class GRK : public Estimator {
public:
    // Constructs GRK estimator.
    // Params:
    //     n: 
    //     tolerance:
    //     x0: 
    GRK(int n, double tolerance, const std::optional<RowMatrixXd>& x0 = std::nullopt);

    /// Iterate method implementation for GRK.
    RowMatrixXd iterate(
        const Eigen::Ref<const RowMatrixXd>& A,
        const Eigen::Ref<const RowMatrixXd>& b,
        const std::optional<RowMatrixXd>& x0) override;

    /// Seed the internal RNG for reproducibility.
    void seed_rng(unsigned int seed) { rng_eng_.seed(seed); }

private:
    double tolerance_;
    std::mt19937 rng_eng_;
};

} // namespace estim
