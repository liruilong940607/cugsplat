#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <tuple>

#include "core/macros.h" // for GSPLAT_HOST_DEVICE
#include "core/math.h"
#include "core/tensor.h"
#include "utils/solver.h" // for solver_newton

namespace gsplat::fisheye {

// Compute the radial distortion: theta -> theta_d
GSPLAT_HOST_DEVICE auto distortion(
    float const &theta, std::array<float, 4> const &radial_coeffs
) -> float {
    auto const theta2 = theta * theta;
    auto const &[k1, k2, k3, k4] = radial_coeffs;
    return theta * eval_poly_horner<5>({1.f, k1, k2, k3, k4}, theta2);
}

// Compute the Jacobian of the distortion: J = d(theta_d) / d(theta)
GSPLAT_HOST_DEVICE auto distortion_jac(
    float const &theta, std::array<float, 4> const &radial_coeffs
) -> float {
    auto const theta2 = theta * theta;
    auto const &[k1, k2, k3, k4] = radial_coeffs;
    return eval_poly_horner<5>(
        {1.f, 3.f * k1, 5.f * k2, 7.f * k3, 9.f * k4}, theta2
    );
}

// Compute the inverse radial distortion: theta_d -> theta
template <size_t N_ITER = 20>
GSPLAT_HOST_DEVICE auto undistortion(
    float const &theta_d,
    std::array<float, 4> const &radial_coeffs,
    float const &min_theta = 0.f,
    float const &max_theta = std::numeric_limits<float>::max()
) -> std::pair<float, bool> {
    // define the residual and Jacobian of the equation
    auto const func = [&theta_d, &radial_coeffs, &min_theta, &max_theta](
                          const float &theta
                      ) -> std::pair<float, float> {
        auto const valid_flag = (theta >= min_theta) && (theta <= max_theta);
        if (!valid_flag)
            return {float{}, float{}};
        auto const J = distortion_jac(theta, radial_coeffs);
        auto const residual = distortion(theta, radial_coeffs) - theta_d;
        return {residual, J};
    };
    // solve the equation
    return solver_newton<1, N_ITER>(func, theta_d, 1e-6f);
}

// Compute the maximum theta such that [0, max_theta] is monotonicly
// increasing.
template <size_t N_ITER = 20>
GSPLAT_HOST_DEVICE auto monotonic_max_theta(
    std::array<float, 4> const &radial_coeffs, float guess = 1.57f
) -> float {
    // The distortion function is
    //   f(theta) = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 +
    //   k4*theta^8)
    // The derivative is
    //   f'(theta) = 1 + 3*k1*theta^2 + 5*k2*theta^4 + 7*k3*theta^6 +
    //   9*k4*theta^8
    // The maximum theta such that [0, max_theta] is monotonicly increasing is
    // the minimal positive root of f'(theta) = 0.

    // Setting x = theta^2, so we just need to solve this:
    //   0 = 1 + 3*k1*x + 5*k2*x^2 + 7*k3*x^3 + 9*k4*x^4
    auto const &[k1, k2, k3, k4] = radial_coeffs;
    constexpr float INF = std::numeric_limits<float>::max();
    auto const x2 = solver_poly_minimal_positive<N_ITER>(
        {1.f, 3.f * k1, 5.f * k2, 7.f * k3, 9.f * k4}, 0.f, guess, INF
    );
    return x2 == INF ? INF : std::sqrt(x2);
}

} // namespace gsplat::fisheye
