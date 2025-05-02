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

} // namespace gsplat::fisheye
