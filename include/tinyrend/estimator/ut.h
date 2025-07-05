// Unscented Transform

#pragma once

#include <array>
#include <functional>
#include <tuple>

#include "tinyrend/common/macros.h" // for TREND_HOST_DEVICE
#include "tinyrend/common/mat.h"
#include "tinyrend/common/vec.h"

namespace tinyrend::ut {

/// \brief Structure that holds the result of the Unscented Transform
/// \tparam M Output dimension of the function
template <int M, typename Aux> struct UnscentedTransformResult {
    /// \brief Mean of the transformed distribution
    fvec<M> mu;
    /// \brief Covariance of the transformed distribution
    fmat<M, M> covar;
    /// \brief Success flag
    bool valid_flag;
    /// \brief Auxiliary data
    Aux aux;
};

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
 * @tparam Func Function type that takes a fvec<N> and returns a tuple of
 * (fvec<M>, bool, Aux)
 *
 * @param fn The nonlinear function to transform. Must return a tuple of
 *     (transformed_point, valid_flag, aux_data)
 * @param mu Mean of the input distribution
 * @param sqrt_covar Square root of the input covariance matrix
 * @param alpha Spread of the sigma points (default: 0.1)
 * @param beta Prior knowledge parameter (default: 2.0)
 * @param kappa Secondary scaling parameter (default: 0.0)
 *
 * @return a UnscentedTransformResult structure
 *
 * @note The function fn must return a tuple of (transformed_point, valid_flag) where
 * valid_flag indicates if the transformation was successful. If any sigma point
 * transformation fails, the entire transform will return false.
 */
template <int N, int M, typename Aux, typename Func>
TREND_HOST_DEVICE inline auto transform(
    Func const &fn,
    fvec<N> const &mu,
    fmat<N, N> const &sqrt_covar,
    const float &alpha = 0.1f,
    const float &beta = 2.0f,
    const float &kappa = 0.0f
) -> UnscentedTransformResult<M, Aux> {
    // Compile-time constraint: ensure Func has the correct signature
    static_assert(
        std::is_invocable_v<Func, fvec<N>>,
        "Func must be callable with a fvec<N> argument"
    );
    static_assert(
        std::is_same_v<
            std::invoke_result_t<Func, fvec<N>>,
            std::tuple<fvec<M>, bool, Aux>>,
        "Func must return std::tuple<fvec<M>, bool, Aux> representing "
        "{transformed_point, valid_flag, aux}"
    );

    auto mu_ut = fvec<M>{};
    auto covar_ut = fmat<M, M>{};

    constexpr int num_sigma = 2 * N + 1;

    auto const lambda = alpha * alpha * (N + kappa) - N;
    auto const std_dev = std::sqrt(N + lambda);

    // Calculate the sigma points
    fvec<N> sigma_points[num_sigma];
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
    fvec<M> transformed_points[num_sigma];
    Aux center_aux;
#pragma unroll
    for (int i = 0; i < num_sigma; i++) {
        auto const &[point, valid_flag, aux] = fn(sigma_points[i]);
        if (!valid_flag) {
            return {mu_ut, covar_ut, false, center_aux};
        }
        transformed_points[i] = point;
        if (i == 0) {
            // store the auxiliary data for the center sigma point
            center_aux = aux;
        }
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
        covar_ut += weights_covar[i] * outer(diff, diff);
    }

    return {mu_ut, covar_ut, true, center_aux};
}

} // namespace tinyrend::ut
