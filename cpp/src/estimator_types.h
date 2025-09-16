#pragma once

#include <Eigen/Dense>

namespace estim {

// RowMatrixXd is the default type as it is more easily mapped into by by numpy.
// The various Kaczmarz methods are also generally row-based, so this assists with locality.
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

} // namespace estim
