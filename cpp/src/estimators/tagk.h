#pragma once

#include <random>

#include "estimator.h"

namespace estim {

struct TAGK : public Estimator {
public:
    // Constructs TAGK estimator.
    // Params:
    //     n:
    //     burnin_steps:
    //     tolerance:
    //     x0:
    TAGK(int n, int burnin_steps, double tolerance, const std::optional<RowMatrixXd>& x0 = std::nullopt);

    // Iterate method implementation for TAGK.
    RowMatrixXd iterate(
        const Eigen::Ref<const RowMatrixXd>& A,
        const Eigen::Ref<const RowMatrixXd>& b,
        const std::optional<RowMatrixXd>& x0) override;

    /// Seed the internal RNG for reproducibility.
    void seed_rng(unsigned int seed) { rng_eng_.seed(seed); }

private:
    int burnin_steps_;
    double tolerance_;
    std::mt19937 rng_eng_;
};

} // namespace estim
