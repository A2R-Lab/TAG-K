cmake_minimum_required(VERSION 3.18...3.31)
project(
  ${SKBUILD_PROJECT_NAME}
  DESCRIPTION "Online estimation methods."
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
)

set(
  SOURCES
  src/bindings.cpp
  src/estimators/grk.cpp
  src/estimators/kf.cpp
  src/estimators/rk.cpp
  src/estimators/rls.cpp
  src/estimators/tagrk.cpp
  src/estimators/tark.cpp
)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(Eigen3 REQUIRED NO_MODULE)

pybind11_add_module(_core ${SOURCES})

target_link_libraries(_core PRIVATE pybind11::headers Eigen3::Eigen Python::NumPy)
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
