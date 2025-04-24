#pragma once

#include <algorithm>
#include <cstdint>

#include "utils/macros.h" // for GSPLAT_HOST_DEVICE
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

inline GSPLAT_HOST_DEVICE float numerically_stable_norm2(float x, float y) {
    // Computes 2-norm of a [x,y] vector in a numerically stable way
    auto const abs_x = std::fabs(x);
    auto const abs_y = std::fabs(y);
    auto const min = std::fmin(abs_x, abs_y);
    auto const max = std::fmax(abs_x, abs_y);

    if (max <= 0.f)
        return 0.f;

    auto const min_max_ratio = min / max;
    return max * std::sqrt(1.f + min_max_ratio * min_max_ratio);
}

template <size_t N_COEFFS>
inline GSPLAT_HOST_DEVICE float
eval_poly_horner(std::array<float, N_COEFFS> const &poly, float x) {
    // Evaluates a polynomial y=f(x) with
    //
    // f(x) = c_0*x^0 + c_1*x^1 + c_2*x^2 + c_3*x^3 + c_4*x^4 ...
    //
    // given by poly_coefficients c_i at points x using numerically stable
    // Horner scheme.
    //
    // The degree of the polynomial is N_COEFFS - 1

    auto y = float{0};
    for (auto cit = poly.rbegin(); cit != poly.rend(); ++cit)
        y = x * y + (*cit);
    return y;
}

template <size_t N>
inline GSPLAT_HOST_DEVICE bool is_all_zero(std::array<float, N> const &arr) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
        if (fabsf(arr[i]) >= std::numeric_limits<float>::epsilon())
            return false;
    }
    return true;
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

// https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
struct OpencvPinholeProjection {
    // Intrinsic Parameters
    glm::fvec2 focal_length;
    glm::fvec2 principal_point;
    // Distortion Coefficients
    std::array<float, 6> radial_coeffs = {0.f};
    std::array<float, 2> tangential_coeffs = {0.f};
    std::array<float, 4> thin_prism_coeffs = {0.f};
    float min_radial_dist = 0.8f;
    float max_radial_dist = std::numeric_limits<float>::max();
    // Perfect Camera: Zero Distortion Coefficients
    bool is_perfect = true;

    // Constructor
    GSPLAT_HOST_DEVICE
    OpencvPinholeProjection(glm::fvec2 focal_length, glm::fvec2 principal_point)
        : focal_length(focal_length), principal_point(principal_point) {}

    GSPLAT_HOST_DEVICE OpencvPinholeProjection(
        glm::fvec2 focal_length,
        glm::fvec2 principal_point,
        std::array<float, 6> radial_coeffs,
        std::array<float, 2> tangential_coeffs,
        std::array<float, 4> thin_prism_coeffs,
        float min_radial_dist = 0.8f,
        float max_radial_dist = std::numeric_limits<float>::max()
    )
        : focal_length(focal_length), principal_point(principal_point),
          radial_coeffs(radial_coeffs), tangential_coeffs(tangential_coeffs),
          thin_prism_coeffs(thin_prism_coeffs),
          min_radial_dist(min_radial_dist), max_radial_dist(max_radial_dist) {
        this->is_perfect = is_all_zero(radial_coeffs) &&
                           is_all_zero(tangential_coeffs) &&
                           is_all_zero(thin_prism_coeffs);
    }

    GSPLAT_HOST_DEVICE auto camera_gaussian_to_image_gaussian(
        const glm::fvec3 &camera_point, const glm::fmat3 camera_covar
    ) const -> MaybeValidGaussian2D {
        if (!is_perfect) {
            assert(
                false && "camera_gaussian_to_image_gaussian() is not "
                         "implemented for non-perfect cameras"
            );
        }

        auto const x = camera_point[0];
        auto const y = camera_point[1];
        auto const rz = 1.f / camera_point[2];
        auto const fxrz = focal_length[0] * rz;
        auto const fyrz = focal_length[1] * rz;

        // glm::fmat3x2 is 3 columns x 2 rows.
        auto const J =
            glm::fmat3x2{fxrz, 0.f, 0.f, fyrz, -fxrz * x * rz, -fyrz * y * rz};

        auto const image_covar = J * camera_covar * transpose(J);
        auto const image_point = glm::fvec2{
            fxrz * x + principal_point[0], fyrz * y + principal_point[1]
        };
        return {image_point, image_covar, true};
    }

    GSPLAT_HOST_DEVICE auto camera_gaussian_to_image_gaussian_vjp(
        // input
        const glm::fvec3 &camera_point,
        const glm::fmat3 &camera_covar,
        // output gradient
        const glm::fvec2 &v_image_point,
        const glm::fmat2 &v_image_covar
    ) const -> std::pair<glm::fvec3, glm::fmat3> {
        if (!is_perfect) {
            assert(
                false && "camera_gaussian_to_image_gaussian_vjp() is not "
                         "implemented for non-perfect cameras"
            );
        }
        auto const x = camera_point[0];
        auto const y = camera_point[1];
        auto const rz = 1.f / camera_point[2];
        auto const fxrz = focal_length[0] * rz;
        auto const fyrz = focal_length[1] * rz;

        // glm::fmat3x2 is 3 columns x 2 rows.
        auto const J =
            glm::fmat3x2{fxrz, 0.f, 0.f, fyrz, -fxrz * x * rz, -fyrz * y * rz};

        auto const v_J = v_image_covar * J * transpose(camera_covar) +
                         transpose(v_image_covar) * J * camera_covar;

        auto const v_camera_covar = transpose(J) * v_image_covar * J;
        auto const v_camera_point = glm::fvec3{
            fxrz * (-rz * v_J[0][2] + v_image_point[0]),
            fyrz * (-rz * v_J[1][2] + v_image_point[1]),
            fxrz * rz *
                    (-v_J[0][0] + 2.f * x * rz * v_J[0][2] -
                     x * v_image_point[0]) +
                fyrz * rz *
                    (-v_J[1][1] + 2.f * y * rz * v_J[1][2] -
                     y * v_image_point[1])
        };
        // input gradient
        return {v_camera_point, v_camera_covar};
    }

    GSPLAT_HOST_DEVICE auto
    camera_point_to_image_point(const glm::fvec3 &camera_point
    ) const -> MaybeValidPoint2D {
        auto const x = camera_point[0] / camera_point[2];
        auto const y = camera_point[1] / camera_point[2];

        auto u = float{};
        auto v = float{};

        if (is_perfect) {
            u = x;
            v = y;
        } else {
            auto const &[k1, k2, k3, k4, k5, k6] = radial_coeffs;
            auto const &[p1, p2] = tangential_coeffs;
            auto const &[s1, s2, s3, s4] = thin_prism_coeffs;

            auto const r2 = x * x + y * y;
            auto const a1 = 2.f * x * y;

            auto const icD_num = eval_poly_horner<4>({1.f, k1, k2, k3}, r2);
            auto const icD_den = eval_poly_horner<4>({1.f, k4, k5, k6}, r2);
            auto const icD = icD_num / icD_den;

            auto const valid_flag =
                (icD > min_radial_dist) && (icD < max_radial_dist);

            if (!valid_flag)
                return {{0.f, 0.f}, false};

            auto const dx =
                p1 * a1 + p2 * (r2 + 2.f * x * x) + r2 * (s1 + r2 * s2);
            auto const dy =
                p2 * a1 + p1 * (r2 + 2.f * y * y) + r2 * (s3 + r2 * s4);

            u = x * icD + dx;
            v = y * icD + dy;
        }

        return {
            {focal_length[0] * u + principal_point[0],
             focal_length[1] * v + principal_point[1]},
            true
        };
    }

    GSPLAT_HOST_DEVICE auto image_point_to_camera_ray(
        const glm::fvec2 &xy, bool normalize = true
    ) const -> MaybeValidRay {
        auto const origin = glm::fvec3{0.f, 0.f, 0.f};

        auto const uv_dist = glm::fvec2{
            (xy[0] - principal_point[0]) / focal_length[0],
            (xy[1] - principal_point[1]) / focal_length[1]
        };

        auto uv = glm::fvec2{};

        if (is_perfect) {
            uv = uv_dist;
        } else {
            auto const &result = solver_newton<2, 20>(
                [this, &uv_dist](const glm::fvec2 &uv) {
                    return this->_residual_jacobian(uv, uv_dist);
                },
                uv_dist,
                1e-6f
            );
            uv = result.x;
            auto const valid_flag = result.converged;
            if (!valid_flag)
                return {origin, {0.f, 0.f, 1.f}, false};
        }
        auto dir = glm::fvec3{uv[0], uv[1], 1.f};
        if (normalize) {
            auto const norm = uv[0] * uv[0] + uv[1] * uv[1] + 1.f;
            auto const inv_norm = 1.f / std::sqrt(norm);
            dir[0] *= inv_norm;
            dir[1] *= inv_norm;
            dir[2] *= inv_norm;
        }
        return {origin, dir, true};
    }

    GSPLAT_HOST_DEVICE auto _residual_jacobian(
        const glm::fvec2 &xy, const glm::fvec2 &xy_dist
    ) const -> std::pair<glm::fvec2, glm::fmat2> {
        auto const x_dist = xy_dist[0];
        auto const y_dist = xy_dist[1];
        auto const x = xy[0];
        auto const y = xy[1];
        auto const &[k1, k2, k3, k4, k5, k6] = radial_coeffs;
        auto const &[p1, p2] = tangential_coeffs;
        auto const &[s1, s2, s3, s4] = thin_prism_coeffs;

        // Compute the residual: f(x, y) = distortion(x, y) - xy_dist
        auto const r2 = x * x + y * y;
        auto const a1 = 2.f * x * y;

        auto const icD_num = eval_poly_horner<4>({1.f, k1, k2, k3}, r2);
        auto const icD_den = eval_poly_horner<4>({1.f, k4, k5, k6}, r2);
        auto const icD = icD_num / icD_den;

        auto const valid_flag =
            (icD > min_radial_dist) && (icD < max_radial_dist);
        if (!valid_flag)
            return {{0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

        auto const delta_x =
            p1 * a1 + p2 * (r2 + 2.f * x * x) + r2 * (s1 + r2 * s2);
        auto const delta_y =
            p2 * a1 + p1 * (r2 + 2.f * y * y) + r2 * (s3 + r2 * s4);

        auto const residual =
            glm::fvec2{x * icD + delta_x - x_dist, y * icD + delta_y - y_dist};

        // Compute the Jacobian: J = d(f(x, y))/d(x, y)
        auto const d_icD_num =
            eval_poly_horner<3>({k1, 2.f * k2, 3.f * k3}, r2);
        auto const d_icD_den =
            eval_poly_horner<3>({k4, 2.f * k5, 3.f * k6}, r2);
        auto const d_icD_dr2 =
            (d_icD_num * icD_den - icD_num * d_icD_den) / (icD_den * icD_den);
        auto const d_icD_dx = 2.f * x * d_icD_dr2;
        auto const d_icD_dy = 2.f * y * d_icD_dr2;

        auto const p1x = 2.f * p1 * x, p2x = 2.f * p2 * x;
        auto const p1y = 2.f * p1 * y, p2y = 2.f * p2 * y;
        auto const d_sx_dr2 = 2.f * (s1 + 2.f * s2 * r2);
        auto const d_sy_dr2 = 2.f * (s3 + 2.f * s4 * r2);

        auto const d_delta_x_dx = p1y + p2x * 3.f + x * d_sx_dr2;
        auto const d_delta_x_dy = p1x + p2y + y * d_sx_dr2;
        auto const d_delta_y_dx = p2y + p1x + x * d_sy_dr2;
        auto const d_delta_y_dy = p2x + p1y * 3.f + y * d_sy_dr2;

        auto const J = glm::fmat2{
            icD + x * d_icD_dx + d_delta_x_dx,
            y * d_icD_dx + d_delta_y_dx,
            x * d_icD_dy + d_delta_x_dy,
            icD + y * d_icD_dy + d_delta_y_dy
        };

        return {residual, J};
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