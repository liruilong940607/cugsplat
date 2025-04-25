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

template <class Derived> struct OpencvPinholeProjectionImpl {
    bool is_perfect = false;
    float min_radial_dist = 0.8f;
    float max_radial_dist = std::numeric_limits<float>::max();

    GSPLAT_HOST_DEVICE auto
    camera_point_to_image_point(const glm::fvec3 &camera_point
    ) -> std::pair<glm::fvec2, bool> {
        auto const derived = static_cast<Derived *>(this);

        auto const xy = glm::fvec2(camera_point) / camera_point.z;

        auto uv = glm::fvec2{};
        if (is_perfect) {
            uv = xy;
        } else {
            auto const &[uv_, valid_flag] = compute_distortion(xy);
            if (!valid_flag)
                return {glm::fvec2{}, false};
            uv = uv_;
        }

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

        auto xy = glm::fvec2{};
        if (is_perfect) {
            xy = uv;
        } else {
            auto const &[xy_, valid_flag] = this->compute_undistortion(uv);
            if (!valid_flag)
                return {glm::fvec3{}, glm::fvec3{}, false};
            xy = xy_;
        }
        auto dir = glm::fvec3{xy[0], xy[1], 1.f};
        if (normalize) {
            dir *= rsqrtf(glm::dot(dir, dir));
        }
        return {origin, dir, true};
    }

    GSPLAT_HOST_DEVICE auto
    camera_point_to_image_point_jacobian(const glm::fvec3 &camera_point
    ) -> std::pair<glm::fmat3x2, bool> {
        auto const derived = static_cast<Derived *>(this);
        auto const rz = 1.0f / camera_point.z;
        auto const xy = glm::fvec2(camera_point) * rz;

        auto J_uv_xy = glm::fmat2{};
        if (is_perfect) {
            J_uv_xy = glm::fmat2{1.f, 0.f, 0.f, 1.f};
        } else {
            auto const &[J_uv_xy_, icD, r2, valid_flag] =
                jacobian_distortion(xy);
            if (!valid_flag)
                return {glm::fmat3x2(0.f), false};
            J_uv_xy = J_uv_xy_;
        }
        auto const focal_length = derived->get_focal_length();
        auto const J_xy = glm::fmat2{
            focal_length[0] * J_uv_xy[0][0],
            focal_length[1] * J_uv_xy[0][1],
            focal_length[0] * J_uv_xy[1][0],
            focal_length[1] * J_uv_xy[1][1],
        };
        auto const J_xy_point = glm::fmat3x2{
            rz,
            0.f,
            0.f,
            rz,
            -xy[0] * rz,
            -xy[1] * rz,
        };
        auto const J = J_xy * J_xy_point;
        return {J, true};
    }

  private:
    // Compute the radial distortion factor icD = icD_num / icD_den
    // Where:
    //      icD_num = 1 + k1 * r2 + k2 * r4 + k3 * r6
    //      icD_den = 1 + k4 * r2 + k5 * r4 + k6 * r6
    GSPLAT_HOST_DEVICE auto compute_icD(const float r2
    ) -> std::pair<float, float> {
        auto const derived = static_cast<Derived *>(this);
        auto const &[k1, k2, k3, k4, k5, k6] = derived->get_radial_coeffs();
        auto const icD_num = eval_poly_horner<4>({1.f, k1, k2, k3}, r2);
        auto const icD_den = eval_poly_horner<4>({1.f, k4, k5, k6}, r2);
        return {icD_num, icD_den};
    }

    // Compute the gradient of the radial distortion factor icD w.r.t. r2
    // Where:
    //      r2 = x^2 + y^2
    GSPLAT_HOST_DEVICE auto gradient_icD(
        const float r2, const float icD_den, const float icD_num
    ) -> float {
        auto const derived = static_cast<Derived *>(this);
        auto const &[k1, k2, k3, k4, k5, k6] = derived->get_radial_coeffs();
        auto const d_icD_num =
            eval_poly_horner<3>({k1, 2.f * k2, 3.f * k3}, r2);
        auto const d_icD_den =
            eval_poly_horner<3>({k4, 2.f * k5, 3.f * k6}, r2);
        auto const d_icD_dr2 =
            (d_icD_num * icD_den - icD_num * d_icD_den) / (icD_den * icD_den);
        return d_icD_dr2; // d(icD) / d(r2)
    }

    // Compute the shifting in the distortion: delta.
    GSPLAT_HOST_DEVICE auto
    compute_delta(const glm::fvec2 xy, const float r2) -> glm::fvec2 {
        auto const derived = static_cast<Derived *>(this);
        auto const &[p1, p2] = derived->get_tangential_coeffs();
        auto const &[s1, s2, s3, s4] = derived->get_thin_prism_coeffs();
        auto const axy = 2.f * xy[0] * xy[1];
        auto const axx = 2.f * xy[0] * xy[0];
        auto const ayy = 2.f * xy[1] * xy[1];
        auto const delta_x = p1 * axy + p2 * (r2 + axx) + r2 * (s1 + r2 * s2);
        auto const delta_y = p2 * axy + p1 * (r2 + ayy) + r2 * (s3 + r2 * s4);
        return glm::fvec2{delta_x, delta_y};
    }

    // Compute the Jacobian of the shifting distortion: d(delta) / d(xy)
    GSPLAT_HOST_DEVICE auto
    jacobian_delta(const glm::fvec2 xy, const float r2) -> glm::fmat2 {
        auto const derived = static_cast<Derived *>(this);
        auto const &[p1, p2] = derived->get_tangential_coeffs();
        auto const &[s1, s2, s3, s4] = derived->get_thin_prism_coeffs();
        auto const p1x = 2.f * p1 * xy[0], p2x = 2.f * p2 * xy[0];
        auto const p1y = 2.f * p1 * xy[1], p2y = 2.f * p2 * xy[1];
        auto const d_sx_dr2 = 2.f * (s1 + 2.f * s2 * r2);
        auto const d_sy_dr2 = 2.f * (s3 + 2.f * s4 * r2);
        auto const d_delta_x_dx = p1y + p2x * 3.f + xy[0] * d_sx_dr2;
        auto const d_delta_x_dy = p1x + p2y + xy[1] * d_sx_dr2;
        auto const d_delta_y_dx = p2y + p1x + xy[0] * d_sy_dr2;
        auto const d_delta_y_dy = p2x + p1y * 3.f + xy[1] * d_sy_dr2;
        // column-major order
        return glm::fmat2{
            d_delta_x_dx, d_delta_y_dx, d_delta_x_dy, d_delta_y_dy
        };
    }

    // Compute the distortion: uv = icD * xy + delta
    GSPLAT_HOST_DEVICE auto compute_distortion(const glm::fvec2 &xy
    ) -> std::pair<glm::fvec2, bool> {
        auto const r2 = glm::dot(xy, xy);
        auto const &[icD_num, icD_den] = compute_icD(r2);
        auto const icD = icD_num / icD_den;
        auto const valid_flag =
            (icD > min_radial_dist) && (icD < max_radial_dist);
        if (!valid_flag)
            return {glm::fvec2{}, false};
        auto const delta = compute_delta(xy, r2);
        auto const uv = icD * xy + delta;
        return {uv, true};
    }

    // Inverse distortion: Solve xy such that uv = icD * xy + delta
    GSPLAT_HOST_DEVICE auto compute_undistortion(const glm::fvec2 &uv
    ) -> std::pair<glm::fvec2, bool> {
        auto const &result = solver_newton<2, 20>(
            [this,
             &uv](const glm::fvec2 &xy) -> std::pair<glm::fvec2, glm::fmat2> {
                auto const &[J, icD, r2, valid_flag] = jacobian_distortion(xy);
                if (!valid_flag)
                    return {glm::fvec2{}, glm::fmat2{}};
                auto const delta = compute_delta(xy, r2);
                auto const residual = icD * xy + delta - uv;
                return {residual, J};
            },
            uv,
            1e-6f
        );
        if (!result.converged)
            return {glm::fvec2{}, false};
        auto const xy = result.x;
        return {xy, true};
    }

    // Compute the Jacobian of the distortion: J = d(uv) / d(xy)
    GSPLAT_HOST_DEVICE auto jacobian_distortion(const glm::fvec2 &xy
    ) -> std::tuple<glm::fmat2, float, float, bool> {
        // Compute the distortion icD
        auto const r2 = glm::dot(xy, xy);
        auto const &[icD_num, icD_den] = compute_icD(r2);
        auto const icD = icD_num / icD_den;
        auto const valid_flag =
            (icD > min_radial_dist) && (icD < max_radial_dist);
        if (!valid_flag)
            return {glm::fmat2(0.f), 0.f, 0.f, false};

        // Compute the Jacobian: J = J(icD) * diag(xy) + diag(icD) + J(delta)
        auto const d_icD_dr2 = gradient_icD(r2, icD_den, icD_num);
        auto const d_icD_dxy = 2.f * d_icD_dr2 * xy;
        auto const J_delta = jacobian_delta(xy, r2);
        auto const J = glm::fmat2{
            icD + xy[0] * d_icD_dxy[0] + J_delta[0][0],
            xy[1] * d_icD_dxy[0] + J_delta[0][1],
            xy[0] * d_icD_dxy[1] + J_delta[1][0],
            icD + xy[1] * d_icD_dxy[1] + J_delta[1][1],
        };
        return {J, icD, r2, true};
    }
};

struct OpencvPinholeProjection
    : OpencvPinholeProjectionImpl<OpencvPinholeProjection> {
    glm::fvec2 focal_length;
    glm::fvec2 principal_point;
    std::array<float, 6> radial_coeffs;
    std::array<float, 2> tangential_coeffs;
    std::array<float, 4> thin_prism_coeffs;

    GSPLAT_HOST_DEVICE
    OpencvPinholeProjection(glm::fvec2 focal_length, glm::fvec2 principal_point)
        : focal_length(focal_length), principal_point(principal_point) {
        this->is_perfect = true;
    }

    GSPLAT_HOST_DEVICE OpencvPinholeProjection(
        glm::fvec2 focal_length,
        glm::fvec2 principal_point,
        std::array<float, 6> radial_coeffs = {},
        std::array<float, 2> tangential_coeffs = {},
        std::array<float, 4> thin_prism_coeffs = {}
    )
        : focal_length(focal_length), principal_point(principal_point),
          radial_coeffs(radial_coeffs), tangential_coeffs(tangential_coeffs),
          thin_prism_coeffs(thin_prism_coeffs) {}

    GSPLAT_HOST_DEVICE auto get_focal_length() const { return focal_length; }
    GSPLAT_HOST_DEVICE auto get_principal_point() const {
        return principal_point;
    }
    GSPLAT_HOST_DEVICE auto get_radial_coeffs() const { return radial_coeffs; }
    GSPLAT_HOST_DEVICE auto get_tangential_coeffs() const {
        return tangential_coeffs;
    }
    GSPLAT_HOST_DEVICE auto get_thin_prism_coeffs() const {
        return thin_prism_coeffs;
    }
};

struct BatchedOpencvPinholeProjection
    : OpencvPinholeProjectionImpl<BatchedOpencvPinholeProjection> {
    uint32_t n{0}, idx{0};
    GSPLAT_HOST_DEVICE void set_index(uint32_t i) { idx = i; }
    GSPLAT_HOST_DEVICE int get_n() const { return n; }

    // pointer to device memory
    const glm::fvec2 *focal_length_ptr;
    const glm::fvec2 *principal_point_ptr;
    const std::array<float, 6> *radial_coeffs_ptr;
    const std::array<float, 2> *tangential_coeffs_ptr;
    const std::array<float, 4> *thin_prism_coeffs_ptr;

    // cache
    Maybe<glm::fvec2> focal_length;
    Maybe<glm::fvec2> principal_point;
    Maybe<std::array<float, 6>> radial_coeffs;
    Maybe<std::array<float, 2>> tangential_coeffs;
    Maybe<std::array<float, 4>> thin_prism_coeffs;

    GSPLAT_HOST_DEVICE BatchedOpencvPinholeProjection(
        uint32_t n,
        const glm::fvec2 *focal_length_ptr,
        const glm::fvec2 *principal_point_ptr
    )
        : n(n), focal_length_ptr(focal_length_ptr),
          principal_point_ptr(principal_point_ptr) {
        this->is_perfect = true;
    }

    GSPLAT_HOST_DEVICE BatchedOpencvPinholeProjection(
        uint32_t n,
        const glm::fvec2 *focal_length_ptr,
        const glm::fvec2 *principal_point_ptr,
        const std::array<float, 6> *radial_coeffs_ptr,
        const std::array<float, 2> *tangential_coeffs_ptr,
        const std::array<float, 4> *thin_prism_coeffs_ptr
    )
        : n(n), focal_length_ptr(focal_length_ptr),
          principal_point_ptr(principal_point_ptr),
          radial_coeffs_ptr(radial_coeffs_ptr),
          tangential_coeffs_ptr(tangential_coeffs_ptr),
          thin_prism_coeffs_ptr(thin_prism_coeffs_ptr) {}

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
    GSPLAT_HOST_DEVICE auto get_tangential_coeffs() {
        if (!tangential_coeffs.has_value()) {
            tangential_coeffs.set(tangential_coeffs_ptr[idx]);
        }
        return tangential_coeffs.get();
    }
    GSPLAT_HOST_DEVICE auto get_thin_prism_coeffs() {
        if (!thin_prism_coeffs.has_value()) {
            thin_prism_coeffs.set(thin_prism_coeffs_ptr[idx]);
        }
        return thin_prism_coeffs.get();
    }
};

} // namespace gsplat