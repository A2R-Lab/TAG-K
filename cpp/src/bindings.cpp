#include <pybind11/pybind11.h>
#include <pybind11/eigen/matrix.h>
#include <pybind11/stl.h>

#include "estimator_types.h"
#include "estimators/estimator.h"
#include "estimators/grk.h"
#include "estimators/kf.h"
#include "estimators/rk.h"
#include "estimators/rls.h"
#include "estimators/tagk.h"
#include "estimators/tark.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {    
    m.doc() = R"pbdoc(
        Online estimation methods.
        -----------------------

        .. currentmodule:: online_estimation

        .. autosummary::
         :toctree: _generate
    )pbdoc";

    /// Estimator interface
    py::class_<estim::Estimator>(m, "Estimator")
        .def_property("state", &estim::Estimator::get_state, &estim::Estimator::set_state);

    /// Estimator implementations
    py::class_<estim::RLS, estim::Estimator>(m, "RLS")
        .def(py::init<int, double, double, const std::optional<estim::RowMatrixXd>&>(),
             py::arg("n"), py::arg("lambda"), py::arg("p_coeff"), py::arg("x0"))
        .def("iterate", &estim::RLS::iterate, py::arg("A"), py::arg("b"), py::arg("x0") = std::nullopt,
             py::return_value_policy::reference_internal);

    py::class_<estim::RLS_Robust, estim::Estimator>(m, "RLS_Robust")
        .def(py::init<int, double, double, const std::optional<estim::RowMatrixXd>&>(),
             py::arg("n"), py::arg("lambda"), py::arg("p_coeff"), py::arg("x0"))
        .def("iterate", &estim::RLS_Robust::iterate, py::arg("A"), py::arg("b"), py::arg("x0") = std::nullopt,
             py::return_value_policy::reference_internal);

    py::class_<estim::KF, estim::Estimator>(m, "KF")
        .def(py::init<int, const estim::RowMatrixXd&, const estim::RowMatrixXd&, double, const std::optional<estim::RowMatrixXd>&>(),
             py::arg("n"), py::arg("process_noise"), py::arg("measurement_noise"), py::arg("p_coeff"), py::arg("x0"))
        .def("iterate", &estim::KF::iterate, py::arg("A"), py::arg("b"), py::arg("x0") = std::nullopt,
             py::return_value_policy::reference_internal);

    py::class_<estim::KF_Robust, estim::Estimator>(m, "KF_Robust")
        .def(py::init<int, const estim::RowMatrixXd&, const estim::RowMatrixXd&, double, const std::optional<estim::RowMatrixXd>&>(),
             py::arg("n"), py::arg("process_noise"), py::arg("measurement_noise"), py::arg("p_coeff"), py::arg("x0"))
        .def("iterate", &estim::KF_Robust::iterate, py::arg("A"), py::arg("b"), py::arg("x0") = std::nullopt,
             py::return_value_policy::reference_internal);

    py::class_<estim::RK, estim::Estimator>(m, "RK")
        .def(py::init<int, const std::optional<estim::RowMatrixXd>&>(),
             py::arg("n"), py::arg("x0"))
        .def("iterate", &estim::RK::iterate, py::arg("A"), py::arg("b"), py::arg("x0") = std::nullopt,
             py::return_value_policy::reference_internal)
        .def("seed_rng", &estim::RK::seed_rng, py::arg("seed"));

    py::class_<estim::GRK, estim::Estimator>(m, "GRK")
        .def(py::init<int, double, const std::optional<estim::RowMatrixXd>&>(),
             py::arg("n"), py::arg("tolerance"), py::arg("x0"))
        .def("iterate", &estim::GRK::iterate, py::arg("A"), py::arg("b"), py::arg("x0") = std::nullopt,
             py::return_value_policy::reference_internal)
        .def("seed_rng", &estim::GRK::seed_rng, py::arg("seed"));

    py::class_<estim::TARK, estim::Estimator>(m, "TARK")
        .def(py::init<int, int, const std::optional<estim::RowMatrixXd>&>(),
             py::arg("n"), py::arg("burnin_steps"), py::arg("x0"))
        .def("iterate", &estim::TARK::iterate, py::arg("A"), py::arg("b"), py::arg("x0") = std::nullopt,
             py::return_value_policy::reference_internal)
        .def("seed_rng", &estim::TARK::seed_rng, py::arg("seed"));

    py::class_<estim::TAGK, estim::Estimator>(m, "TAGK")
        .def(py::init<int, int, double, const std::optional<estim::RowMatrixXd>&>(),
             py::arg("n"), py::arg("burnin_steps"), py::arg("tolerance"), py::arg("x0"))
        .def("iterate", &estim::TAGK::iterate, py::arg("A"), py::arg("b"), py::arg("x0") = std::nullopt,
             py::return_value_policy::reference_internal)
        .def("seed_rng", &estim::TAGK::seed_rng, py::arg("seed"));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
