// Unscented Transform

#pragma once

#include "cugsplat/core/macros.h" // for GSPLAT_HOST_DEVICE
#include <array>
#include <functional>
#include <glm/glm.hpp>
#include <tuple>

namespace cugsplat::ut {

/**
 * @brief Performs the Unscented Transform (UT) on a nonlinear function.
 *
 * The Unscented Transform is a method for calculating the statistics of a random
 * variable which undergoes a nonlinear transformation. It uses a set of carefully
 * chosen sample points (sigma points) to capture the mean and covariance of the
 * transformed distribution.
 *
 * @tparam N Input dimension of the function
 * @tparam M Output dimension of the function
 * @tparam Func Function type that takes a glm::vec<N, float> and returns a tuple of
 * (glm::vec<M, float>, bool)
 *
 * @param f The nonlinear function to transform
 * @param mu Mean of the input distribution
 * @param sqrt_covar Square root of the input covariance matrix
 * @param alpha Spread of the sigma points (default: 0.1)
 * @param beta Prior knowledge parameter (default: 2.0)
 * @param kappa Secondary scaling parameter (default: 0.0)
 *
 * @return A tuple containing:
 *         - Transformed mean (glm::vec<M, float>)
 *         - Transformed covariance matrix (glm::mat<M, M, float>)
 *         - Success flag (bool)
 *
 * @note The function f must return a tuple of (transformed_point, valid_flag) where
 * valid_flag indicates if the transformation was successful. If any sigma point
 * transformation fails, the entire transform will return false.
 */
template <int N, int M, typename Func>
GSPLAT_HOST_DEVICE inline auto transform(
    Func const &f,
    glm::vec<N, float> const &mu,
    glm::mat<N, N, float> const &sqrt_covar,
    const float &alpha = 0.1f,
    const float &beta = 2.0f,
    const float &kappa = 0.0f
) -> std::tuple<glm::vec<M, float>, glm::mat<M, M, float>, bool> {
    auto mu_ut = glm::vec<M, float>{};
    auto covar_ut = glm::mat<M, M, float>{};

    constexpr int num_sigma = 2 * N + 1;

    auto const lambda = alpha * alpha * (N + kappa) - N;
    auto const std_dev = std::sqrt(N + lambda);

    // Calculate the sigma points
    glm::vec<N, float> sigma_points[num_sigma];
    sigma_points[0] = mu;
#pragma unroll
    for (int i = 0; i < N; i++) {
        auto const delta = std_dev * sqrt_covar[i];
        sigma_points[i + 1] = mu + delta;
        sigma_points[i + 1 + N] = mu - delta;
    }

    // Calculate the weights
    float weights_mean[num_sigma];
    float weights_covar[num_sigma];
    weights_mean[0] = lambda / (N + lambda);
    weights_covar[0] = weights_mean[0] + 1 - alpha * alpha + beta;
#pragma unroll
    for (int i = 1; i < num_sigma; i++) {
        weights_mean[i] = 1.0f / (2.0f * (N + lambda));
        weights_covar[i] = weights_mean[i];
    }

    // Calculate the transformed sigma points
    glm::vec<M, float> transformed_points[num_sigma];
#pragma unroll
    for (int i = 0; i < num_sigma; i++) {
        auto const &[point, valid_flag] = f(sigma_points[i]);
        if (!valid_flag) {
            return {mu_ut, covar_ut, false};
        }
        transformed_points[i] = point;
    }

    // Calculate the unscented transform mean
#pragma unroll
    for (int i = 0; i < num_sigma; i++) {
        mu_ut += weights_mean[i] * transformed_points[i];
    }

    // Calculate the unscented transform covariance
#pragma unroll
    for (int i = 0; i < num_sigma; i++) {
        auto const diff = transformed_points[i] - mu_ut;
        covar_ut += weights_covar[i] * glm::outerProduct(diff, diff);
    }

    return {mu_ut, covar_ut, true};
}

} // namespace cugsplat::ut
