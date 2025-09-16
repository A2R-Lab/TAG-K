#include <vector>

#include "rk.h"

namespace estim {

RK::RK(int n, const std::optional<RowMatrixXd>& x0)
    : Estimator(n, x0) {

    std::random_device rd;
    rng_eng_.seed(rd());
}

RowMatrixXd RK::iterate(
    const Eigen::Ref<const RowMatrixXd>& A,
    const Eigen::Ref<const RowMatrixXd>& b,
    const std::optional<RowMatrixXd>& x0) {
    
    if (x0.has_value())
        validate_shape(*x0, n_, 1, "x0");

    RowMatrixXd x = x0.has_value() ? *x0 : this->x_;

    const long m = A.rows();
    const double eps = 1e-12;

    Eigen::VectorXd row_norms_sq = A.rowwise().squaredNorm();
    double total_norm_sum = row_norms_sq.sum();
    Eigen::VectorXd probs_vec = (row_norms_sq.array() + eps) / (total_norm_sum + m * eps);
    std::vector<double> probs(probs_vec.data(), probs_vec.data() + probs_vec.size());
    std::discrete_distribution<> dist(probs.begin(), probs.end());

    for (int k = 0; k < m; k++) {
        int i = dist(rng_eng_); // random choice
        Eigen::RowVectorXd ai = A.row(i);
        double bi = b(i, 0);
        double residual = bi - (ai * x)(0);
        x += (residual / row_norms_sq(i)) * ai.transpose();
    }

    this->x_ = x;

    return this->x_;
}

} // namespace estim
