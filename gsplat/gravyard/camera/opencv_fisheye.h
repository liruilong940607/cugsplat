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

namespace gsplat {

// Helper function to solve linear equation 1 + ax = 0
inline GSPLAT_HOST_DEVICE float _solve_linear(float a) {
    const float INF = std::numeric_limits<float>::max();
    return (a >= 0.0f) ? INF : -1.0f / a;
}

// Helper function to solve quadratic equation 1 + ax + bx^2 = 0
inline GSPLAT_HOST_DEVICE float _solve_quadratic(float a, float b) {
    const float INF = std::numeric_limits<float>::max();

    if (b == 0.0f) {
        return _solve_linear(a);
    }

    float delta = a * a - 4.0f * b;
    if (delta < 0.0f)
        return INF;

    delta = std::sqrt(delta) - a;
    return (delta > 0.0f) ? 2.0f / delta : INF;
}

// Helper function to solve cubic equation 1 + ax + bx^2 + cx^3 = 0
// Returns the minimal positive root or INF if no positive root exists
inline GSPLAT_HOST_DEVICE float _solve_cubic(float a, float b, float c) {
    const float INF = std::numeric_limits<float>::max();
    constexpr float PI = 3.14159265358979323846f;

    float boc = b / c;
    float boc2 = boc * boc;

    float t1 = (9.0f * a * boc - 2.0f * b * boc2 - 27.0f) / c;
    float t2 = 3.0f * a / c - boc2;
    float delta = t1 * t1 + 4.0f * t2 * t2 * t2;

    if (delta >= 0.0f) {
        // One real root case
        float d2 = std::sqrt(delta);
        float cube_root = std::cbrt((d2 + t1) / 2.0f);
        if (cube_root == 0.0f)
            return INF;

        float soln = (cube_root - (t2 / cube_root) - boc) / 3.0f;
        return (soln > 0.0f) ? soln : INF;
    } else {
        // Three real roots case (delta < 0)
        float theta = std::atan2(std::sqrt(-delta), t1) / 3.0f;
        constexpr float two_third_pi = 2.0f * PI / 3.0f;

        float t3 = 2.0f * std::sqrt(-t2);
        float min_soln = INF;

        for (int i : {-1, 0, 1}) {
            float angle = theta + i * two_third_pi;
            float s = (t3 * std::cos(angle) - boc) / 3.0f;
            if (s > 0.0f) {
                min_soln = std::min(min_soln, s);
            }
        }
        return min_soln;
    }
}

// Solve 1 + ax + bx^2 + cx^3 = 0 and return the minimal positive root
inline GSPLAT_HOST_DEVICE float
_poly3_minimal_postivie_root(float a, float b, float c) {
    if (c == 0.0f) {
        if (b == 0.0f) {
            return _solve_linear(a);
        }
        return _solve_quadratic(a, b);
    }
    return _solve_cubic(a, b, c);
}

struct OpencvFisheyeProjection {
    MaybeCached<glm::fvec2> focal_length;
    MaybeCached<glm::fvec2> principal_point;
    MaybeCached<std::array<float, 4>> radial_coeffs;

    bool is_perfect = false;
    float min_theta = 0.f;
    float max_theta = std::numeric_limits<float>::max();
    float min_2d_norm = 1e-6f;

    GSPLAT_HOST_DEVICE
    OpencvFisheyeProjection() {}

    GSPLAT_HOST_DEVICE
    OpencvFisheyeProjection(
        const glm::fvec2 *focal_length, const glm::fvec2 *principal_point
    )
        : focal_length(focal_length), principal_point(principal_point) {
        this->is_perfect = true;
    }

    GSPLAT_HOST_DEVICE OpencvFisheyeProjection(
        const glm::fvec2 *focal_length,
        const glm::fvec2 *principal_point,
        const std::array<float, 4> *radial_coeffs
    )
        : focal_length(focal_length), principal_point(principal_point),
          radial_coeffs(radial_coeffs) {}

    GSPLAT_HOST_DEVICE auto
    camera_point_to_image_point(const glm::fvec3 &camera_point
    ) -> std::pair<glm::fvec2, bool> {
        auto const xy = glm::fvec2(camera_point) / camera_point.z;
        auto const r = numerically_stable_norm2(xy[0], xy[1]);
        auto const theta = std::atan(r);

        const bool valid_flag = (theta >= min_theta) && (theta <= max_theta);
        if (!valid_flag)
            return {glm::fvec2{}, false};

        auto const theta_d =
            is_perfect ? theta : compute_distortion(theta, theta * theta);
        auto const scale_factor = theta_d / r;
        auto const uv = r < min_2d_norm ? xy : scale_factor * xy;

        auto const focal_length = this->focal_length.get();
        auto const principal_point = this->principal_point.get();
        auto const image_point = focal_length * uv + principal_point;
        return {image_point, true};
    }

    GSPLAT_HOST_DEVICE auto image_point_to_camera_ray(
        const glm::fvec2 &image_point, bool normalize = true
    ) -> std::tuple<glm::fvec3, glm::fvec3, bool> {
        auto const origin = glm::fvec3{0.f, 0.f, 0.f};

        auto const focal_length = this->focal_length.get();
        auto const principal_point = this->principal_point.get();
        auto const uv = (image_point - principal_point) / focal_length;

        if (fabs(uv[0]) < min_2d_norm && fabs(uv[1]) < min_2d_norm) {
            // For points at the image center, return a ray pointing straight
            // ahead
            auto const dir = glm::fvec3{0.f, 0.f, 1.f};
            return {origin, dir, true};
        }

        auto const theta_d = sqrtf(glm::dot(uv, uv));

        auto theta = float{};
        if (is_perfect) {
            theta = theta_d;
        } else {
            auto const &[theta_, valid_flag] =
                this->compute_undistortion(theta_d);
            if (!valid_flag)
                return {glm::fvec3{}, glm::fvec3{}, false};
            theta = theta_;
        }

        if (normalize) {
            auto const scale_factor = std::sin(theta) / theta_d;
            auto const xy = scale_factor * uv;
            auto const dir = glm::fvec3{xy[0], xy[1], std::cos(theta)};
            return {origin, dir, true};
        } else {
            auto const scale_factor = std::tan(theta) / theta_d;
            auto const xy = scale_factor * uv;
            auto const dir = glm::fvec3{xy[0], xy[1], 1.f};
            return {origin, dir, true};
        }
    }

    // TODO
    GSPLAT_HOST_DEVICE auto
    camera_point_to_image_point_jacobian(const glm::fvec3 &camera_point
    ) -> std::pair<glm::fmat3x2, bool> {
        return {glm::fmat3x2{}, true};
    }

    // Compute the maximum theta such that [0, max_theta] is monotonicly
    // increasing.
    GSPLAT_HOST_DEVICE auto set_max_theta() -> float {
        auto const &[k1, k2, k3, k4] = this->radial_coeffs.get();
        const float INF = std::numeric_limits<float>::max();

        if (k4 == 0.f) {
            // solve the root of f'(x) = 0 analyticly
            this->max_theta = std::sqrt(
                _poly3_minimal_postivie_root(3.f * k1, 5.f * k2, 7.f * k3)
            );
        } else {
            // solve the root of f'(x) = 0 numerically
            auto const &[root, converged] = solver_newton<1, 20>(
                [this, &k1, &k2, &k3, &k4](const float &x
                ) -> std::pair<float, float> {
                    auto const x2 = x * x;
                    auto const residual = eval_poly_horner<5>(
                        {1.f, 3.f * k1, 5.f * k2, 7.f * k3, 9.f * k4}, x2
                    );
                    auto const gradient =
                        x * eval_poly_horner<4>(
                                {6.f * k1, 20.f * k2, 56.f * k3, 72.f * k4}, x2
                            );
                    return {residual, gradient};
                },
                1.57f, // initial guess
                1e-6f
            );
            this->max_theta = converged ? (root > 0 ? root : INF) : INF;
        }

        return this->max_theta;
    }

  private:
    // Compute the distortion: theta_d from theta
    GSPLAT_HOST_DEVICE auto
    compute_distortion(const float theta, const float theta2) -> float {
        auto const &[k1, k2, k3, k4] = this->radial_coeffs.get();
        return theta * eval_poly_horner<5>({1.f, k1, k2, k3, k4}, theta2);
    }

    // Inverse distortion: Solve theta from theta_d
    GSPLAT_HOST_DEVICE auto compute_undistortion(const float &theta_d
    ) -> std::pair<float, bool> {
        auto const &result = solver_newton<1, 20>(
            [this, &theta_d](const float &theta) -> std::pair<float, float> {
                auto const valid_flag =
                    (theta >= min_theta) && (theta <= max_theta);
                if (!valid_flag)
                    return {float{}, float{}};
                auto const gradient = this->gradiant_distortion(theta);
                auto const residual =
                    this->compute_distortion(theta, theta * theta) - theta_d;
                return {residual, gradient};
            },
            theta_d,
            1e-6f
        );
        if (!result.converged)
            return {float{}, false};
        auto const theta = result.x;
        return {theta, true};
    }

    // Compute the Jacobian of the distortion: J = d(theta_d) / d(theta)
    GSPLAT_HOST_DEVICE auto gradiant_distortion(const float &theta2) -> float {
        auto const &[k1, k2, k3, k4] = this->radial_coeffs.get();
        return eval_poly_horner<5>(
            {1.f, 3.f * k1, 5.f * k2, 7.f * k3, 9.f * k4}, theta2
        );
    }
};

} // namespace gsplat