#pragma once

#include <optional>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "../estimator_types.h"

namespace estim {

// Estimator is a generic interface for online parameter estimation methods.
// The only function that needs to be implemented is iterate, which allows for
// the user to reset the value of x0.
class Estimator {
public:
    virtual ~Estimator() = default;

    // Getter/setter for state parameter.
    const RowMatrixXd& get_state() const { return x_; }

    void set_state(const Eigen::Ref<const RowMatrixXd>& x) {
        validate_shape(x, n_, 1, "New state");
        x_ = x;
    }

    // Iterate method solving for x_approx = A^(-1) b.
    // Params:
    //     A:  (m, n) matrix
    //     b:  (m, 1) matrix
    //     x0: (n, 1) matrix
    // Returns: estimate of solution to A x = b
    virtual RowMatrixXd iterate(
        const Eigen::Ref<const RowMatrixXd>& A,
        const Eigen::Ref<const RowMatrixXd>& b,
        const std::optional<RowMatrixXd>& x0) = 0;

protected:
    // Validate shape of provided RowMatrix
    // Params:
    //     mat:
    //     rows:
    //     cols:
    void validate_shape(const Eigen::Ref<const RowMatrixXd>& mat, int rows, int cols, std::string identifier) {
        if (mat.rows() != rows || mat.cols() != cols) {
            throw std::invalid_argument(
                identifier + " has incorrect dimensions. Expected (" +
                std::to_string(rows) + ", " + std::to_string(cols) + "), but got (" +
                std::to_string(mat.rows()) + ", " + std::to_string(mat.cols()) + ")."
            );
        }
    }

    // Default constructor for Estimator class.
    // Params:
    //     n:
    //     x0:
    Estimator(int n, const std::optional<RowMatrixXd>& x0)
        : n_(n),
          x_(x0.has_value() ? *x0 : RowMatrixXd::Zero(n_, 1)) {

            if (n <= 0)
                throw std::invalid_argument("n cannot be negative or zero.");

            validate_shape(x_, n_, 1, "x0");
        }

    int n_;
    RowMatrixXd x_;
};

} // namespace estim
