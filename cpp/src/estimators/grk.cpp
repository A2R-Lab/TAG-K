#include <vector>

#include "grk.h"

namespace estim {

GRK::GRK(int n, double tolerance, const std::optional<RowMatrixXd>& x0)
    : Estimator(n, x0),
      tolerance_(tolerance) {

    if (tolerance <= 0)
        throw std::invalid_argument("Tolerance cannot be negative or zero.");

    std::random_device rd;
    rng_eng_.seed(rd());
}

RowMatrixXd GRK::iterate(
    const Eigen::Ref<const RowMatrixXd>& A,
    const Eigen::Ref<const RowMatrixXd>& b,
    const std::optional<RowMatrixXd>& x0) {

    if (x0.has_value())
        validate_shape(*x0, n_, 1, "x0");

    RowMatrixXd x = x0.has_value() ? *x0 : this->x_;

    const long m = A.rows();

    const Eigen::VectorXd row_norms_sq = A.rowwise().squaredNorm();
    const double fro_sq = A.squaredNorm();
    const double fro_sq_safe = std::max(fro_sq, 1e-300);

    for (int k = 0; k < m; ++k) {
        Eigen::VectorXd r = b - A * x;
        double rnorm_sq = r.squaredNorm();

        if (rnorm_sq <= tolerance_ * tolerance_) {
            break;
        }

        Eigen::ArrayXd r_sq = r.array().square();
        Eigen::ArrayXd crit = (row_norms_sq.array() > 0).select(r_sq / row_norms_sq.array(), 0.0);
        double rnorm_sq_safe = std::max(rnorm_sq, 1e-300);
        double eps_k = 0.5 * (crit.maxCoeff() / rnorm_sq_safe + 1.0 / fro_sq_safe);
        
        Eigen::Array<bool, Eigen::Dynamic, 1> tau_mask = (r_sq >= eps_k * rnorm_sq * row_norms_sq.array());

        if (!tau_mask.any()) {
            Eigen::Index max_idx;
            r.array().abs().maxCoeff(&max_idx);
            tau_mask.setConstant(false);
            tau_mask(max_idx) = true;
        }

        Eigen::VectorXd r_tilde = tau_mask.select(r, Eigen::VectorXd::Zero(m));
        double rt_sq = r_tilde.squaredNorm();
        int i_k;

        if (rt_sq <= 1e-20) {
            double max_abs_r = -1.0;
            i_k = -1;
            for(int i = 0; i < m; ++i) {
                if (tau_mask(i) && std::abs(r(i)) > max_abs_r) {
                    max_abs_r = std::abs(r(i));
                    i_k = i;
                }
            }
        } else {
            Eigen::VectorXd probs_vec = r_tilde.array().square() / rt_sq;
            std::vector<double> probs(probs_vec.data(), probs_vec.data() + probs_vec.size());
            std::discrete_distribution<> dist(probs.begin(), probs.end());
            i_k = dist(rng_eng_);
        }

        double den_safe = std::max(row_norms_sq(i_k), 1e-300);
        x += (r(i_k) / den_safe) * A.row(i_k).transpose();
    }

    this->x_ = x;

    return this->x_;
}

} // namespace estim
