#pragma once

#include <algorithm>
#include <cstdint>

#include "utils/macros.h" // for GSPLAT_HOST_DEVICE
#include "utils/math.h"
#include "utils/solver.h" // for solver_newton
#include "utils/types.h"  // for MaybeValidRay, MaybeValidPoint2D

namespace gsplat {

// solve 1 + ax + bx^2 + cx^3 = 0
inline GSPLAT_HOST_DEVICE float
_poly3_minimal_postivie_root(float a, float b, float c) {
    const float INF = std::numeric_limits<float>::max();
    constexpr float PI = 3.14159265358979323846f;

    if (c == 0.0f) {
        if (b == 0.0f) {
            if (a >= 0.0f) {
                return INF;
            } else {
                return -1.0f / a;
            }
        }
        float delta = a * a - 4.0f * b;
        if (delta >= 0.0f) {
            delta = std::sqrt(delta) - a;
            if (delta > 0.0f) {
                return 2.0f / delta;
            }
        }
    } else {
        float boc = b / c;
        float boc2 = boc * boc;

        float t1 = (9.0f * a * boc - 2.0f * b * boc2 - 27.0f) / c;
        float t2 = 3.0f * a / c - boc2;
        float delta = t1 * t1 + 4.0f * t2 * t2 * t2;

        if (delta >= 0.0f) {
            float d2 = std::sqrt(delta);
            float cube_root = std::cbrt((d2 + t1) / 2.0f);
            if (cube_root != 0.0f) {
                float soln = (cube_root - (t2 / cube_root) - boc) / 3.0f;
                if (soln > 0.0f) {
                    return soln;
                }
            }
        } else {
            // Complex root case (delta < 0): 3 real roots
            float theta = std::atan2(std::sqrt(-delta), t1) / 3.0f;
            constexpr float two_third_pi = 2.0f * PI / 3.0f;

            float t3 = 2.0f * std::sqrt(-t2);
            float soln = INF;
            for (int i : {-1, 0, 1}) {
                float angle = theta + i * two_third_pi;
                float s = (t3 * std::cos(angle) - boc) / 3.0f;
                if (s > 0.0f) {
                    soln = std::min(soln, s);
                }
            }
            return soln;
        }
    }

    return INF;
}

struct OrthogonalProjection {
    // Intrinsic Parameters
    glm::fvec2 focal_length;
    glm::fvec2 principal_point;

    // Constructor
    GSPLAT_HOST_DEVICE
    OrthogonalProjection(glm::fvec2 focal_length, glm::fvec2 principal_point)
        : focal_length(focal_length), principal_point(principal_point) {}

    GSPLAT_HOST_DEVICE auto
    camera_point_to_image_point(const glm::fvec3 &camera_point
    ) const -> MaybeValidPoint2D {
        glm::fvec2 image_point = {
            focal_length[0] * camera_point[0] + principal_point[0],
            focal_length[1] * camera_point[1] + principal_point[1]
        };
        return {image_point, true};
    }

    GSPLAT_HOST_DEVICE auto image_point_to_camera_ray(
        const glm::fvec2 &xy, bool _unused = true
    ) const -> MaybeValidRay {
        auto const u = (xy[0] - principal_point[0]) / focal_length[0];
        auto const v = (xy[1] - principal_point[1]) / focal_length[1];
        // for orthogonal projection, the camera ray origin is
        // (u, v, 0) and the direction is (0, 0, 1)
        auto const origin = glm::fvec3{u, v, 0.f};
        auto const dir = glm::fvec3{0.f, 0.f, 1.f};
        return {origin, dir, true};
    }
};

// https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
struct OpencvFisheyeProjection {

    // Intrinsic Parameters
    glm::fvec2 focal_length;
    glm::fvec2 principal_point;
    // Distortion Coefficients
    std::array<float, 4> radial_coeffs = {0.f};
    float min_theta = 0.f;
    float max_theta = std::numeric_limits<float>::max();
    float min_2d_norm = 1e-6f;
    bool is_perfect = true;

    // Constructor
    GSPLAT_HOST_DEVICE
    OpencvFisheyeProjection(glm::fvec2 focal_length, glm::fvec2 principal_point)
        : focal_length(focal_length), principal_point(principal_point) {}

    GSPLAT_HOST_DEVICE OpencvFisheyeProjection(
        glm::fvec2 focal_length,
        glm::fvec2 principal_point,
        std::array<float, 4> radial_coeffs
    )
        : focal_length(focal_length), principal_point(principal_point),
          radial_coeffs(radial_coeffs) {
        this->is_perfect = is_all_zero(radial_coeffs);
        if (!is_perfect) {
            this->max_theta = set_max_theta();
        }
    }

    GSPLAT_HOST_DEVICE auto
    camera_point_to_image_point(const glm::fvec3 &camera_point
    ) const -> MaybeValidPoint2D {
        auto const x = camera_point[0] / camera_point[2];
        auto const y = camera_point[1] / camera_point[2];
        auto const &[k1, k2, k3, k4] = radial_coeffs;

        auto const r = numerically_stable_norm2(x, y);
        auto const theta = std::atan(r);
        auto const theta2 = theta * theta;

        const bool valid_flag = (theta > min_theta) && (theta < max_theta);
        if (!valid_flag)
            return {{0.f, 0.f}, false};

        auto const theta_d =
            is_perfect
                ? theta
                : theta * eval_poly_horner<5>({1.f, k1, k2, k3, k4}, theta2);
        auto const scale_factor = theta_d / r;

        auto const u = scale_factor * x;
        auto const v = scale_factor * y;

        auto const image_point = glm::fvec2{
            focal_length[0] * u + principal_point[0],
            focal_length[1] * v + principal_point[1]
        };
        return {image_point, true};
    }

    GSPLAT_HOST_DEVICE auto image_point_to_camera_ray(
        const glm::fvec2 &xy, bool normalize = true
    ) const -> MaybeValidRay {
        auto const origin = glm::fvec3{0.f, 0.f, 0.f};

        auto const uv_dist = glm::fvec2{
            (xy[0] - principal_point[0]) / focal_length[0],
            (xy[1] - principal_point[1]) / focal_length[1]
        };

        if (fabs(uv_dist[0]) < min_2d_norm && fabs(uv_dist[1]) < min_2d_norm) {
            // For points at the image center, return a ray pointing straight
            // ahead
            return {origin, {0.f, 0.f, 1.f}, true};
        }

        auto const theta_d =
            sqrt(uv_dist[0] * uv_dist[0] + uv_dist[1] * uv_dist[1]);

        auto theta = float{};
        if (is_perfect) {
            // For perfect pinhole camera, the undistortion is trivial
            theta = theta_d;
        } else {
            auto const &result = solver_newton<1, 20>(
                [this, &theta_d](float &theta) {
                    return this->_residual_gradient(theta, theta_d);
                },
                theta_d,
                1e-6f
            );
            theta = result.x;
            auto const valid_flag = result.converged;
            if (!valid_flag)
                return {origin, {0.f, 0.f, 1.f}, false};
        }

        if (normalize) {
            auto const scale_factor = std::sin(theta) / theta_d;
            auto const u = uv_dist[0] * scale_factor;
            auto const v = uv_dist[1] * scale_factor;
            return {origin, {u, v, std::cos(theta)}, true};
        } else {
            auto const scale_factor = std::tan(theta) / theta_d;
            auto const u = uv_dist[0] * scale_factor;
            auto const v = uv_dist[1] * scale_factor;
            return {origin, {u, v, 1.f}, true};
        }
    }

    GSPLAT_HOST_DEVICE auto _residual_gradient(
        const float &theta, const float &theta_d
    ) const -> std::pair<float, float> {
        auto const &[k1, k2, k3, k4] = radial_coeffs;

        const bool valid_flag = (theta > min_theta) && (theta < max_theta);
        if (!valid_flag)
            return {0.f, 0.f};

        auto const theta2 = theta * theta;
        auto const residual =
            theta * eval_poly_horner<5>({1.f, k1, k2, k3, k4}, theta2) -
            theta_d;
        auto const gradient = eval_poly_horner<5>(
            {1.f, 3.f * k1, 5.f * k2, 7.f * k3, 9.f * k4}, theta2
        );
        return {residual, gradient};
    }

    // Compute the maximum theta such that [0, max_theta] is monotonicly
    // increasing.
    GSPLAT_HOST_DEVICE auto set_max_theta() -> float {
        auto const &[k1, k2, k3, k4] = radial_coeffs;
        const float INF = std::numeric_limits<float>::max();

        if (k4 == 0.f) {
            // solve the root of f'(x) = 0 analyticly
            this->max_theta = std::sqrt(
                _poly3_minimal_postivie_root(3.f * k1, 5.f * k2, 7.f * k3)
            );
        } else {
            // solve the root of f'(x) = 0 numerically
            auto const &[root, converged] = solver_newton<1, 20>(
                [this](float &x) {
                    return this->_set_max_theta_residual_gradient(x);
                },
                1.57f, // initial guess
                1e-6f
            );
            this->max_theta = converged ? (root > 0 ? root : INF) : INF;
        }

        return this->max_theta;
    }

    GSPLAT_HOST_DEVICE auto _set_max_theta_residual_gradient(const float &x
    ) const -> std::pair<float, float> {
        auto const &[k1, k2, k3, k4] = radial_coeffs;
        auto const x2 = x * x;
        auto const residual = eval_poly_horner<5>(
            {1.f, 3.f * k1, 5.f * k2, 7.f * k3, 9.f * k4}, x2
        );
        auto const gradient =
            x * eval_poly_horner<4>(
                    {6.f * k1, 20.f * k2, 56.f * k3, 72.f * k4}, x2
                );
        return {residual, gradient};
    }
};

} // namespace gsplat