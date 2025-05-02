#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <tuple>

#include "core/macros.h" // for GSPLAT_HOST_DEVICE
#include "core/math.h"
#include "utils/solver.h" // for solver_newton

namespace gsplat::fisheye {

// Compute the radial distortion: theta -> theta_d
GSPLAT_HOST_DEVICE inline auto distortion(
    float const &theta, std::array<float, 4> const &radial_coeffs
) -> float {
    auto const theta2 = theta * theta;
    auto const &[k1, k2, k3, k4] = radial_coeffs;
    return theta * eval_poly_horner<5>({1.f, k1, k2, k3, k4}, theta2);
}

// Compute the Jacobian of the distortion: J = d(theta_d) / d(theta)
GSPLAT_HOST_DEVICE inline auto distortion_jac(
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
GSPLAT_HOST_DEVICE inline auto undistortion(
    float const &theta_d,
    std::array<float, 4> const &radial_coeffs,
    float const &max_theta = std::numeric_limits<float>::max()
) -> std::pair<float, bool> {
    // define the residual and Jacobian of the equation
    auto const func = [&theta_d, &radial_coeffs, &max_theta](const float &theta
                      ) -> std::pair<float, float> {
        auto const valid_flag = theta <= max_theta;
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
GSPLAT_HOST_DEVICE inline auto monotonic_max_theta(
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

// Project the point from camera space to image space (perfect fisheye)
// Returns projected image point.
GSPLAT_HOST_DEVICE inline auto project(
    glm::fvec3 const &camera_point,
    glm::fvec2 const &focal_length,
    glm::fvec2 const &principal_point,
    float const &min_2d_norm = 1e-6f
) -> glm::fvec2 {
    auto const xy = glm::fvec2(camera_point) / camera_point.z;
    auto const r = numerically_stable_norm2(xy[0], xy[1]);
    glm::fvec2 uv;
    if (r < min_2d_norm) {
        // For points at the image center, there is no distortion
        uv = xy;
    } else {
        auto const theta = std::atan(r);
        uv = theta / r * xy;
    }
    auto const image_point = focal_length * uv + principal_point;
    return image_point;
}

// Compute the Jacobian of the projection: J = d(image_point) / d(camera_point)
GSPLAT_HOST_DEVICE inline auto project_jac(
    glm::fvec3 const &camera_point,
    glm::fvec2 const &focal_length,
    glm::fvec2 const &principal_point,
    float const &min_2d_norm = 1e-6f
) -> glm::fmat3x2 {
    // TODO: implement this
    return glm::fmat3x2{};
}

// Project the point from camera space to image space (distorted fisheye)
// Returns the image point and a flag indicating if the projection is valid
GSPLAT_HOST_DEVICE inline auto project(
    glm::fvec3 const &camera_point,
    glm::fvec2 const &focal_length,
    glm::fvec2 const &principal_point,
    std::array<float, 4> const &radial_coeffs,
    float const &min_2d_norm = 1e-6f,
    float const &max_theta = std::numeric_limits<float>::max()
) -> std::pair<glm::fvec2, bool> {
    auto const xy = glm::fvec2(camera_point) / camera_point.z;
    auto const r = numerically_stable_norm2(xy[0], xy[1]);
    glm::fvec2 uv;
    if (r < min_2d_norm) {
        // For points at the image center, there is no distortion
        uv = xy;
    } else {
        auto const theta = std::atan(r);
        if (theta > max_theta) {
            // Theta is too large, might be in the invalid region
            return {glm::fvec2{}, false};
        }
        auto const theta_d = distortion(theta, radial_coeffs);
        uv = theta_d / r * xy;
    }
    auto const image_point = focal_length * uv + principal_point;
    return {image_point, true};
}

// Unproject the point from image space to camera space (perfect fisheye)
// Returns the normalized ray direction.
GSPLAT_HOST_DEVICE inline auto unproject(
    glm::fvec2 const &image_point,
    glm::fvec2 const &focal_length,
    glm::fvec2 const &principal_point,
    float const &min_2d_norm = 1e-6f
) -> glm::fvec3 {
    auto const uv = (image_point - principal_point) / focal_length;
    auto const theta = sqrtf(glm::dot(uv, uv));

    if (theta < min_2d_norm) {
        // For points at the image center, the ray direction is
        // simply pointing forward.
        return glm::fvec3{0.f, 0.f, 1.f};
    }

    auto const xy = std::sin(theta) / theta * uv;
    auto const dir = glm::fvec3{xy[0], xy[1], std::cos(theta)};
    return dir;
}

// Unproject the point from image space to camera space (distorted fisheye)
// Returns the normalized ray direction and a flag indicating if the
// unprojection is valid
GSPLAT_HOST_DEVICE inline auto unproject(
    glm::fvec2 const &image_point,
    glm::fvec2 const &focal_length,
    glm::fvec2 const &principal_point,
    std::array<float, 4> const &radial_coeffs,
    float const &min_2d_norm = 1e-6f,
    float const &max_theta = std::numeric_limits<float>::max()
) -> std::pair<glm::fvec3, bool> {
    auto const uv = (image_point - principal_point) / focal_length;
    auto const theta_d = sqrtf(glm::dot(uv, uv));

    if (theta_d < min_2d_norm) {
        // For points at the image center, the ray direction is
        // simply pointing forward.
        return {glm::fvec3{0.f, 0.f, 1.f}, true};
    }

    auto const &[theta, valid_flag] =
        undistortion(theta_d, radial_coeffs, max_theta);
    if (!valid_flag) {
        return {glm::fvec3{}, false};
    }

    auto const xy = std::sin(theta) / theta * uv;
    auto const dir = glm::fvec3{xy[0], xy[1], std::cos(theta)};
    return {dir, true};
}

} // namespace gsplat::fisheye
