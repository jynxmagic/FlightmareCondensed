#pragma once
#include <Eigen/Eigen>

// --- Setup some Eigen quick shorthands ---

// Define the scalar type used.
using Scalar = double; // numpy float64
static constexpr int Dynamic = Eigen::Dynamic;

// Using shorthand for `Matrix<rows, cols>` with scalar type.
template <int rows = Dynamic, int cols = Dynamic>
using Matrix = Eigen::Matrix<Scalar, rows, cols>; // lets you do Matrix<3,3> or Matrix<3,1> etc;

template <int rows = Dynamic>
using Vector = Matrix<rows, 1>; // lets you do Vector<3> or Vector<4> etc;

using Quaternion = Eigen::Quaternion<Scalar>; // type Quaternion instead of Eigen::Quaternion<Scalar>