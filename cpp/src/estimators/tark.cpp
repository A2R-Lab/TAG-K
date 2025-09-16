#include <vector>

#include "tark.h"

namespace estim {

TARK::TARK(int n_params, int burnin_steps, const std::optional<RowMatrixXd>& x0)
    : Estimator(n_params, x0),
      burnin_steps_(burnin_steps) {

    if (burnin_steps < 0)
        throw std::invalid_argument("Burnin steps cannot be negative.");

    std::random_device rd;
    rng_eng_.seed(rd());
}

RowMatrixXd TARK::iterate(
    const Eigen::Ref<const RowMatrixXd>& A,
    const Eigen::Ref<const RowMatrixXd>& b,
    const std::optional<RowMatrixXd>& x0) {
    
    if (x0.has_value())
        validate_shape(*x0, n_, 1, "x0");

    RowMatrixXd x = x0.has_value() ? *x0 : this->x_;
    
    // Initialize averaging variables.
    RowMatrixXd x_sum = RowMatrixXd::Zero(n_, 1);
    int count = 0;

    const long m = A.rows();
    const double eps = 1e-12;

    // Set up random sampling.
    Eigen::VectorXd row_norms_sq = A.rowwise().squaredNorm();
    double total_norm_sum = row_norms_sq.sum();
    Eigen::VectorXd probs_vec = (row_norms_sq.array() + eps) / (total_norm_sum + m * eps);
    std::vector<double> probs(probs_vec.data(), probs_vec.data() + probs_vec.size());
    std::discrete_distribution<> dist(probs.begin(), probs.end());

    for (int s = 0; s < m; ++s) {
        int i = dist(rng_eng_);
        Eigen::RowVectorXd ai = A.row(i);
        double bi = b(i, 0);
        double residual = bi - ai.dot(x.col(0));
        x += (residual / row_norms_sq(i)) * ai.transpose();

        // If past the burn-in period, add the current iterate to the sum.
        if (s >= burnin_steps_) {
            x_sum += x;
            count++;
        }
    }

    // Update the object's persistent state with the *last* iterate.
    this->x_ = x;

    // Calculate and return the *tail-averaged* result.
    if (count > 0) {
        return x_sum / count;
    }

    // If nothing was averaged, return the last iterate.
    return this->x_;
}

} // namespace estim
