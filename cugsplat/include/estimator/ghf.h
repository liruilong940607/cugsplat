// Gauss-Hermite Filter precomputed matrices
//
// Precomputes:
// 1. Standardized sigma points
// 2. Quadratic features matrix
// 3. Weighted least squares matrices
//
// Reference:
//   numpy.polynomial.hermite.hermgauss

#pragma once

#include "estimator/hermgauss.h"
#include <array>
#include <functional>
#include <glm/glm.hpp>

#include "core/macros.h" // for GSPLAT_HOST_DEVICE

namespace cugsplat::ghf {

// Helper function to compute number of quadratic features
template <int N> constexpr int num_quadratic_features() {
    return 1 + N + N * (N + 1) / 2;
}

// Helper function to compute quadratic features for a single point
template <int N>
constexpr auto build_quadratic_features_point(glm::vec<N, float> const &point
) -> std::array<float, num_quadratic_features<N>()> {
    std::array<float, num_quadratic_features<N>()> features{};

    // Constant term
    features[0] = 1.0f;

    // Linear terms
    for (int i = 0; i < N; ++i) {
        features[1 + i] = point[i];
    }

    // Quadratic terms
    int idx = 1 + N;
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            features[idx++] = point[i] * point[j];
        }
    }

    return features;
}

// Build quadratic features matrix for all points
template <int N, int M>
constexpr auto build_quadratic_features(std::array<glm::vec<N, float>, M> const &points
) -> std::array<std::array<float, num_quadratic_features<N>()>, M> {
    std::array<std::array<float, num_quadratic_features<N>()>, M> features_matrix{};
    for (int i = 0; i < M; ++i) {
        features_matrix[i] = build_quadratic_features_point<N>(points[i]);
    }
    return features_matrix;
}

// Compute weighted features transpose
template <int N, int M>
constexpr auto compute_weighted_features_transpose(
    std::array<std::array<float, num_quadratic_features<N>()>, M> const
        &features_matrix,
    std::array<float, M> const &weights
) -> std::array<std::array<float, M>, num_quadratic_features<N>()> {
    std::array<std::array<float, M>, num_quadratic_features<N>()>
        weighted_features_transpose{};
    for (int i = 0; i < num_quadratic_features<N>(); ++i) {
        for (int j = 0; j < M; ++j) {
            weighted_features_transpose[i][j] = features_matrix[j][i] * weights[j];
        }
    }
    return weighted_features_transpose;
}

// Compute weighted features covariance
template <int N, int M>
constexpr auto compute_weighted_features_covariance(
    std::array<std::array<float, num_quadratic_features<N>()>, M> const
        &features_matrix,
    std::array<std::array<float, M>, num_quadratic_features<N>()> const
        &weighted_features_transpose
)
    -> std::array<
        std::array<float, num_quadratic_features<N>()>,
        num_quadratic_features<N>()> {
    std::array<
        std::array<float, num_quadratic_features<N>()>,
        num_quadratic_features<N>()>
        weighted_covariance{};
    for (int i = 0; i < num_quadratic_features<N>(); ++i) {
        for (int j = 0; j < num_quadratic_features<N>(); ++j) {
            float sum = 0.0f;
            for (int k = 0; k < M; ++k) {
                sum += weighted_features_transpose[i][k] * features_matrix[k][j];
            }
            weighted_covariance[i][j] = sum;
        }
    }
    return weighted_covariance;
}

// Compute inverse of weighted features covariance using Gauss-Jordan elimination
template <int N>
constexpr auto compute_inverse(std::array<
                               std::array<float, num_quadratic_features<N>()>,
                               num_quadratic_features<N>()> const &matrix)
    -> std::array<
        std::array<float, num_quadratic_features<N>()>,
        num_quadratic_features<N>()> {
    constexpr int size = num_quadratic_features<N>();
    std::array<std::array<float, size>, size> inverse{};

    // Initialize inverse as identity matrix
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            inverse[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Create a copy of matrix to work with
    std::array<std::array<float, size>, size> mat = matrix;

    // Perform Gauss-Jordan elimination
    for (int i = 0; i < size; ++i) {
        // Find pivot
        float pivot = mat[i][i];
        if (pivot == 0.0f) {
            // Find a row to swap with
            for (int j = i + 1; j < size; ++j) {
                if (mat[j][i] != 0.0f) {
                    std::swap(mat[i], mat[j]);
                    std::swap(inverse[i], inverse[j]);
                    pivot = mat[i][i];
                    break;
                }
            }
            if (pivot == 0.0f) {
                // Matrix is singular
                return inverse;
            }
        }

        // Normalize pivot row
        for (int j = 0; j < size; ++j) {
            mat[i][j] /= pivot;
            inverse[i][j] /= pivot;
        }

        // Eliminate other rows
        for (int k = 0; k < size; ++k) {
            if (k != i && mat[k][i] != 0.0f) {
                float factor = mat[k][i];
                for (int j = 0; j < size; ++j) {
                    mat[k][j] -= factor * mat[i][j];
                    inverse[k][j] -= factor * inverse[i][j];
                }
            }
        }
    }

    return inverse;
}

// Compute weighted least squares coefficients
template <int N, int M>
constexpr auto compute_weighted_least_squares_coefficients(
    std::array<
        std::array<float, num_quadratic_features<N>()>,
        num_quadratic_features<N>()> const &weighted_covariance_inverse,
    std::array<std::array<float, M>, num_quadratic_features<N>()> const
        &weighted_features_transpose
) -> std::array<std::array<float, M>, num_quadratic_features<N>()> {
    std::array<std::array<float, M>, num_quadratic_features<N>()> coefficients{};
    for (int i = 0; i < num_quadratic_features<N>(); ++i) {
        for (int j = 0; j < M; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < num_quadratic_features<N>(); ++k) {
                sum += weighted_covariance_inverse[i][k] *
                       weighted_features_transpose[k][j];
            }
            coefficients[i][j] = sum;
        }
    }
    return coefficients;
}

// Precompute all matrices for order 3
constexpr auto features_matrix_order3 =
    build_quadratic_features<3, HERMGAUSS_POINTS_3D_ORDER3.size()>(
        HERMGAUSS_POINTS_3D_ORDER3
    );
constexpr auto weighted_features_transpose_order3 =
    compute_weighted_features_transpose<3, HERMGAUSS_POINTS_3D_ORDER3.size()>(
        features_matrix_order3, HERMGAUSS_WEIGHTS_3D_ORDER3
    );
constexpr auto weighted_covariance_order3 =
    compute_weighted_features_covariance<3, HERMGAUSS_POINTS_3D_ORDER3.size()>(
        features_matrix_order3, weighted_features_transpose_order3
    );
constexpr auto weighted_covariance_inverse_order3 =
    compute_inverse<3>(weighted_covariance_order3);
constexpr auto weighted_least_squares_coefficients_order3 =
    compute_weighted_least_squares_coefficients<3, constexpr_pow<ORDER3, 3>::value>(
        weighted_covariance_inverse_order3, weighted_features_transpose_order3
    );

// Precompute all matrices for order 5
constexpr auto features_matrix_order5 =
    build_quadratic_features<3, constexpr_pow<ORDER5, 3>::value>(
        HERMGAUSS_POINTS_3D_ORDER5
    );
constexpr auto weighted_features_transpose_order5 =
    compute_weighted_features_transpose<3, constexpr_pow<ORDER5, 3>::value>(
        features_matrix_order5, HERMGAUSS_WEIGHTS_3D_ORDER5
    );
constexpr auto weighted_covariance_order5 =
    compute_weighted_features_covariance<3, constexpr_pow<ORDER5, 3>::value>(
        features_matrix_order5, weighted_features_transpose_order5
    );
constexpr auto weighted_covariance_inverse_order5 =
    compute_inverse<3>(weighted_covariance_order5);
constexpr auto weighted_least_squares_coefficients_order5 =
    compute_weighted_least_squares_coefficients<3, constexpr_pow<ORDER5, 3>::value>(
        weighted_covariance_inverse_order5, weighted_features_transpose_order5
    );

// Struct to hold precomputed matrices
template <int N, int Size> struct PrecomputedMatrices {
    std::array<glm::vec<N, float>, Size> points_std;
    std::array<std::array<float, Size>, num_quadratic_features<N>()> coefficients;
};

// Helper function to get precomputed matrices based on order
template <int N, int order> auto get_precomputed_matrices() {
    if constexpr (N == 3) {
        if constexpr (order == 3) {
            return PrecomputedMatrices<3, 27>{
                HERMGAUSS_POINTS_3D_ORDER3, weighted_least_squares_coefficients_order3
            };
        } else {
            return PrecomputedMatrices<3, 125>{
                HERMGAUSS_POINTS_3D_ORDER5, weighted_least_squares_coefficients_order5
            };
        }
    } else {
        static_assert(N == 3, "Only 3D is supported");
    }
}

// Runtime function to estimate Jacobian and Hessian
template <int N, int M, typename Func, int order = 3>
GSPLAT_HOST_DEVICE inline auto estimate_jacobian_and_hessian(
    Func const &f, glm::vec<N, float> const &mu, glm::vec<N, float> const &std_dev
) -> std::pair<glm::mat<M, N, float>, std::array<glm::mat<N, N, float>, M>> {
    // Get precomputed matrices based on order
    auto matrices = get_precomputed_matrices<N, order>();

    // Evaluate function at all points
    std::array<glm::vec<M, float>, matrices.points_std.size()> outputs;
#pragma unroll
    for (size_t i = 0; i < matrices.points_std.size(); ++i) {
        outputs[i] = f(mu + matrices.points_std[i] * std_dev);
    }

    // Initialize results
    glm::mat<M, N, float> J{};
    std::array<glm::mat<N, N, float>, M> H{};

// For each output dimension
#pragma unroll
    for (int output_dim = 0; output_dim < M; ++output_dim) {
        // Extract output values for this dimension
        std::array<float, matrices.points_std.size()> y;
#pragma unroll
        for (size_t i = 0; i < matrices.points_std.size(); ++i) {
            y[i] = outputs[i][output_dim];
        }

        // Compute regression coefficients
        std::array<float, num_quadratic_features<N>()> theta{};
#pragma unroll
        for (int i = 0; i < num_quadratic_features<N>(); ++i) {
            float sum = 0.0f;
#pragma unroll
            for (size_t j = 0; j < matrices.points_std.size(); ++j) {
                sum += matrices.coefficients[i][j] * y[j];
            }
            theta[i] = sum;
        }

// Extract Jacobian
#pragma unroll
        for (int i = 0; i < N; ++i) {
            J[output_dim][i] = theta[1 + i] / std_dev[i];
        }

        // Extract Hessian
        int k = 0;
#pragma unroll
        for (int i = 0; i < N; ++i) {
#pragma unroll
            for (int j = i; j < N; ++j) {
                float coeff = theta[1 + N + k] / (std_dev[i] * std_dev[j]);
                if (i == j) {
                    coeff *= 2.0f;
                }
                H[output_dim][i][j] = coeff;
                H[output_dim][j][i] = coeff; // symmetric
                ++k;
            }
        }
    }

    return {J, H};
}

} // namespace cugsplat::ghf