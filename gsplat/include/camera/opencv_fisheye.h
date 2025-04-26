#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <tuple>

#include "utils/macros.h" // for GSPLAT_HOST_DEVICE
#include "utils/math.h"
#include "utils/solver.h" // for solver_newton
#include "utils/types.h"  // for Maybe

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

template <class Derived> struct OpencvFisheyeProjectionImpl {
    bool is_perfect = false;
    float min_theta = 0.f;
    float max_theta = std::numeric_limits<float>::max();
    float min_2d_norm = 1e-6f;

    GSPLAT_HOST_DEVICE auto
    camera_point_to_image_point(const glm::fvec3 &camera_point
    ) -> std::pair<glm::fvec2, bool> {
        auto const derived = static_cast<Derived *>(this);

        auto const xy = glm::fvec2(camera_point) / camera_point.z;
        auto const r = numerically_stable_norm2(xy[0], xy[1]);
        auto const theta = std::atan(r);

        const bool valid_flag = (theta > min_theta) && (theta < max_theta);
        if (!valid_flag)
            return {glm::fvec2{}, false};

        auto const theta_d =
            is_perfect ? theta : compute_distortion(theta, theta * theta);
        auto const scale_factor = theta_d / r;
        auto const uv = scale_factor * xy;

        auto const focal_length = derived->get_focal_length();
        auto const principal_point = derived->get_principal_point();
        auto const image_point = focal_length * uv + principal_point;
        return {image_point, true};
    }

    GSPLAT_HOST_DEVICE auto image_point_to_camera_ray(
        const glm::fvec2 &image_point, bool normalize = true
    ) -> std::tuple<glm::fvec3, glm::fvec3, bool> {
        auto const derived = static_cast<Derived *>(this);

        auto const origin = glm::fvec3{0.f, 0.f, 0.f};

        auto const focal_length = derived->get_focal_length();
        auto const principal_point = derived->get_principal_point();
        auto const uv = (image_point - principal_point) / focal_length;

        if (fabs(uv[0]) < min_2d_norm && fabs(uv[1]) < min_2d_norm) {
            // For points at the image center, return a ray pointing straight
            // ahead
            auto const dir = glm::fvec3{0.f, 0.f, 1.f};
            return {origin, dir, true};
        }

        auto const theta_d = sqrt(glm::dot(uv, uv));

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
        auto const derived = static_cast<Derived *>(this);
        return {glm::fmat3x2{}, true};
    }

    // Compute the maximum theta such that [0, max_theta] is monotonicly
    // increasing.
    GSPLAT_HOST_DEVICE auto set_max_theta() -> float {
        auto const derived = static_cast<Derived *>(this);
        auto const &[k1, k2, k3, k4] = derived->get_radial_coeffs();
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
        auto const derived = static_cast<Derived *>(this);
        auto const &[k1, k2, k3, k4] = derived->get_radial_coeffs();
        return theta * eval_poly_horner<5>({1.f, k1, k2, k3, k4}, theta2);
    }

    // Inverse distortion: Solve theta from theta_d
    GSPLAT_HOST_DEVICE auto compute_undistortion(const float &theta_d
    ) -> std::pair<float, bool> {
        auto const &result = solver_newton<1, 20>(
            [this, &theta_d](const float &theta) -> std::pair<float, float> {
                auto const valid_flag =
                    (theta > min_theta) && (theta < max_theta);
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
        return eval_poly_horner<5>(
            {1.f, 3.f * k1, 5.f * k2, 7.f * k3, 9.f * k4}, theta2
        );
    }
};

struct OpencvFisheyeProjection
    : OpencvFisheyeProjectionImpl<OpencvFisheyeProjection> {
    glm::fvec2 focal_length;
    glm::fvec2 principal_point;
    std::array<float, 4> radial_coeffs;

    GSPLAT_HOST_DEVICE
    OpencvFisheyeProjection(glm::fvec2 focal_length, glm::fvec2 principal_point)
        : focal_length(focal_length), principal_point(principal_point) {
        this->is_perfect = true;
    }

    GSPLAT_HOST_DEVICE OpencvFisheyeProjection(
        glm::fvec2 focal_length,
        glm::fvec2 principal_point,
        std::array<float, 4> radial_coeffs = {}
    )
        : focal_length(focal_length), principal_point(principal_point),
          radial_coeffs(radial_coeffs) {}

    GSPLAT_HOST_DEVICE auto get_focal_length() const { return focal_length; }
    GSPLAT_HOST_DEVICE auto get_principal_point() const {
        return principal_point;
    }
    GSPLAT_HOST_DEVICE auto get_radial_coeffs() const { return radial_coeffs; }
};

struct BatchedOpencvFisheyeProjection
    : OpencvFisheyeProjectionImpl<BatchedOpencvFisheyeProjection> {
    uint32_t n{0}, idx{0};
    GSPLAT_HOST_DEVICE void set_index(uint32_t i) { idx = i; }
    GSPLAT_HOST_DEVICE int get_n() const { return n; }

    // pointer to device memory
    const glm::fvec2 *focal_length_ptr;
    const glm::fvec2 *principal_point_ptr;
    const std::array<float, 4> *radial_coeffs_ptr;

    // cache
    Maybe<glm::fvec2> focal_length;
    Maybe<glm::fvec2> principal_point;
    Maybe<std::array<float, 4>> radial_coeffs;

    GSPLAT_HOST_DEVICE BatchedOpencvFisheyeProjection(
        uint32_t n,
        const glm::fvec2 *focal_length_ptr,
        const glm::fvec2 *principal_point_ptr
    )
        : n(n), focal_length_ptr(focal_length_ptr),
          principal_point_ptr(principal_point_ptr) {
        this->is_perfect = true;
    }

    GSPLAT_HOST_DEVICE BatchedOpencvFisheyeProjection(
        uint32_t n,
        const glm::fvec2 *focal_length_ptr,
        const glm::fvec2 *principal_point_ptr,
        const std::array<float, 4> *radial_coeffs_ptr
    )
        : n(n), focal_length_ptr(focal_length_ptr),
          principal_point_ptr(principal_point_ptr),
          radial_coeffs_ptr(radial_coeffs_ptr) {}

    GSPLAT_HOST_DEVICE auto get_focal_length() {
        if (!focal_length.has_value()) {
            focal_length.set(focal_length_ptr[idx]);
        }
        return focal_length.get();
    }
    GSPLAT_HOST_DEVICE auto get_principal_point() {
        if (!principal_point.has_value()) {
            principal_point.set(principal_point_ptr[idx]);
        }
        return principal_point.get();
    }
    GSPLAT_HOST_DEVICE auto get_radial_coeffs() {
        if (!radial_coeffs.has_value()) {
            radial_coeffs.set(radial_coeffs_ptr[idx]);
        }
        return radial_coeffs.get();
    }
};

} // namespace gsplat