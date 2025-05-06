#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <tuple>

#include "cugsplat/core/macros.h" // for GSPLAT_HOST_DEVICE
#include "cugsplat/core/math.h"
#include "cugsplat/core/solver.h"

namespace cugsplat::pinhhole {

constexpr float DEFAULT_MIN_RADIAL_DIST = 0.8f;
constexpr float DEFAULT_MAX_RADIAL_DIST = std::numeric_limits<float>::max();

/// @private
// Compute the radial distortion factor icD = icD_num / icD_den
// Where:
//      icD_num = 1 + k1 * r2 + k2 * r4 + k3 * r6
//      icD_den = 1 + k4 * r2 + k5 * r4 + k6 * r6
GSPLAT_HOST_DEVICE inline auto compute_icD(
    const float r2, const std::array<float, 6> &radial_coeffs
) -> std::pair<float, float> {
    auto const &[k1, k2, k3, k4, k5, k6] = radial_coeffs;
    auto const icD_num = cugsplat::math::eval_poly_horner<4>({1.f, k1, k2, k3}, r2);
    auto const icD_den = cugsplat::math::eval_poly_horner<4>({1.f, k4, k5, k6}, r2);
    return {icD_num, icD_den};
}

/// @private
// Compute the gradient of the radial distortion factor icD w.r.t. r2
// Where:
//      r2 = x^2 + y^2
GSPLAT_HOST_DEVICE inline auto gradient_icD(
    const float r2,
    const float icD_den,
    const float icD_num,
    const std::array<float, 6> &radial_coeffs
) -> float {
    auto const &[k1, k2, k3, k4, k5, k6] = radial_coeffs;
    auto const d_icD_num =
        cugsplat::math::eval_poly_horner<3>({k1, 2.f * k2, 3.f * k3}, r2);
    auto const d_icD_den =
        cugsplat::math::eval_poly_horner<3>({k4, 2.f * k5, 3.f * k6}, r2);
    auto const d_icD_dr2 =
        (d_icD_num * icD_den - icD_num * d_icD_den) / (icD_den * icD_den);
    return d_icD_dr2; // d(icD) / d(r2)
}

/// @private
// Compute the shifting in the distortion: delta.
GSPLAT_HOST_DEVICE inline auto compute_delta(
    const glm::fvec2 xy,
    const float r2,
    const std::array<float, 2> &tangential_coeffs,
    const std::array<float, 4> &thin_prism_coeffs
) -> glm::fvec2 {
    auto const &[p1, p2] = tangential_coeffs;
    auto const &[s1, s2, s3, s4] = thin_prism_coeffs;
    auto const axy = 2.f * xy[0] * xy[1];
    auto const axx = 2.f * xy[0] * xy[0];
    auto const ayy = 2.f * xy[1] * xy[1];
    auto const delta_x = p1 * axy + p2 * (r2 + axx) + r2 * (s1 + r2 * s2);
    auto const delta_y = p2 * axy + p1 * (r2 + ayy) + r2 * (s3 + r2 * s4);
    return glm::fvec2{delta_x, delta_y};
}

/// @private
// Compute the Jacobian of the shifting distortion: d(delta) / d(xy)
GSPLAT_HOST_DEVICE inline auto jacobian_delta(
    const glm::fvec2 xy,
    const float r2,
    const std::array<float, 2> &tangential_coeffs,
    const std::array<float, 4> &thin_prism_coeffs
) -> glm::fmat2 {
    auto const &[p1, p2] = tangential_coeffs;
    auto const &[s1, s2, s3, s4] = thin_prism_coeffs;
    auto const p1x = 2.f * p1 * xy[0], p2x = 2.f * p2 * xy[0];
    auto const p1y = 2.f * p1 * xy[1], p2y = 2.f * p2 * xy[1];
    auto const d_sx_dr2 = 2.f * (s1 + 2.f * s2 * r2);
    auto const d_sy_dr2 = 2.f * (s3 + 2.f * s4 * r2);
    auto const d_delta_x_dx = p1y + p2x * 3.f + xy[0] * d_sx_dr2;
    auto const d_delta_x_dy = p1x + p2y + xy[1] * d_sx_dr2;
    auto const d_delta_y_dx = p2y + p1x + xy[0] * d_sy_dr2;
    auto const d_delta_y_dy = p2x + p1y * 3.f + xy[1] * d_sy_dr2;
    // column-major order
    return glm::fmat2{d_delta_x_dx, d_delta_y_dx, d_delta_x_dy, d_delta_y_dy};
}

/// \brief Compute the distortion: uv = icD * xy + delta
/// \param xy 2D point in normalized image coordinates
/// \param radial_coeffs Radial distortion coefficients (k1, k2, k3, k4, k5, k6)
/// \param tangential_coeffs Tangential distortion coefficients (p1, p2)
/// \param thin_prism_coeffs Thin prism distortion coefficients (s1, s2, s3, s4)
/// \param min_radial_dist Minimum radial distortion threshold for numerical stability.
/// Default value is 0.8f.
/// \param max_radial_dist Maximum radial distortion threshold for numerical stability.
/// Default value is max float.
/// \return Pair of distorted 2D point and validity flag
GSPLAT_HOST_DEVICE inline auto distortion(
    const glm::fvec2 &xy,
    const std::array<float, 6> &radial_coeffs,
    const std::array<float, 2> &tangential_coeffs,
    const std::array<float, 4> &thin_prism_coeffs,
    const float &min_radial_dist = DEFAULT_MIN_RADIAL_DIST,
    const float &max_radial_dist = DEFAULT_MAX_RADIAL_DIST
) -> std::pair<glm::fvec2, bool> {
    auto const r2 = glm::dot(xy, xy);
    auto const &[icD_num, icD_den] = compute_icD(r2, radial_coeffs);
    auto const icD = icD_num / icD_den;
    auto const valid_flag = (icD > min_radial_dist) && (icD < max_radial_dist);
    if (!valid_flag)
        return {glm::fvec2{}, false};
    auto const delta = compute_delta(xy, r2, tangential_coeffs, thin_prism_coeffs);
    auto const uv = icD * xy + delta;
    return {uv, true};
}

/// \brief Compute the Jacobian of the distortion: J = d(uv) / d(xy)
/// \param xy 2D point in normalized image coordinates
/// \param radial_coeffs Radial distortion coefficients (k1, k2, k3, k4, k5, k6)
/// \param tangential_coeffs Tangential distortion coefficients (p1, p2)
/// \param thin_prism_coeffs Thin prism distortion coefficients (s1, s2, s3, s4)
/// \param min_radial_dist Minimum radial distortion threshold for numerical stability.
/// Default value is 0.8f.
/// \param max_radial_dist Maximum radial distortion threshold for numerical stability.
/// Default value is max float.
/// \return Tuple containing the 2x2 Jacobian matrix, radial distortion factor icD,
/// squared radius r2, and validity flag
GSPLAT_HOST_DEVICE inline auto distortion_jac(
    const glm::fvec2 &xy,
    const std::array<float, 6> &radial_coeffs,
    const std::array<float, 2> &tangential_coeffs,
    const std::array<float, 4> &thin_prism_coeffs,
    const float &min_radial_dist = DEFAULT_MIN_RADIAL_DIST,
    const float &max_radial_dist = DEFAULT_MAX_RADIAL_DIST
) -> std::tuple<glm::fmat2, float, float, bool> {
    // Compute the distortion icD
    auto const r2 = glm::dot(xy, xy);
    auto const &[icD_num, icD_den] = compute_icD(r2, radial_coeffs);
    auto const icD = icD_num / icD_den;
    auto const valid_flag = (icD > min_radial_dist) && (icD < max_radial_dist);
    if (!valid_flag)
        return {glm::fmat2(0.f), 0.f, 0.f, false};

    // Compute the Jacobian: J = J(icD) * diag(xy) + diag(icD) + J(delta)
    auto const d_icD_dr2 = gradient_icD(r2, icD_den, icD_num, radial_coeffs);
    auto const d_icD_dxy = 2.f * d_icD_dr2 * xy;
    auto const J_delta = jacobian_delta(xy, r2, tangential_coeffs, thin_prism_coeffs);
    auto const J = glm::fmat2{
        icD + xy[0] * d_icD_dxy[0] + J_delta[0][0],
        xy[1] * d_icD_dxy[0] + J_delta[0][1],
        xy[0] * d_icD_dxy[1] + J_delta[1][0],
        icD + xy[1] * d_icD_dxy[1] + J_delta[1][1],
    };
    return {J, icD, r2, true};
}

/// \brief Inverse distortion: Solve xy such that uv = icD * xy + delta
/// \tparam N_ITER Number of iterations for Newton's method
/// \param uv 2D point in distorted image coordinates
/// \param radial_coeffs Radial distortion coefficients (k1, k2, k3, k4, k5, k6)
/// \param tangential_coeffs Tangential distortion coefficients (p1, p2)
/// \param thin_prism_coeffs Thin prism distortion coefficients (s1, s2, s3, s4)
/// \param min_radial_dist Minimum radial distortion threshold for numerical stability.
/// Default value is 0.8f.
/// \param max_radial_dist Maximum radial distortion threshold for numerical stability.
/// Default value is max float.
/// \return Pair of undistorted 2D point and convergence flag
template <size_t N_ITER = 20>
GSPLAT_HOST_DEVICE inline auto undistortion(
    const glm::fvec2 &uv,
    const std::array<float, 6> &radial_coeffs,
    const std::array<float, 2> &tangential_coeffs,
    const std::array<float, 4> &thin_prism_coeffs,
    const float &min_radial_dist = DEFAULT_MIN_RADIAL_DIST,
    const float &max_radial_dist = DEFAULT_MAX_RADIAL_DIST
) -> std::pair<glm::fvec2, bool> {
    // define the residual and Jacobian of the equation
    auto const func = [&uv,
                       &radial_coeffs,
                       &tangential_coeffs,
                       &thin_prism_coeffs,
                       &min_radial_dist,
                       &max_radial_dist](const glm::fvec2 &xy
                      ) -> std::pair<glm::fvec2, glm::fmat2> {
        auto const &[J, icD, r2, valid_flag] = distortion_jac(
            xy,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            min_radial_dist,
            max_radial_dist
        );
        if (!valid_flag)
            return {glm::fvec2{}, glm::fmat2{}};
        auto const delta = compute_delta(xy, r2, tangential_coeffs, thin_prism_coeffs);
        auto const residual = icD * xy + delta - uv;
        return {residual, J};
    };

    auto const &[xy, converged] = cugsplat::solver::newton<2, N_ITER>(func, uv, 1e-6f);
    return {xy, converged};
}

/// \brief Project a 3D point in camera space to 2D image space using pinhole projection
/// \param camera_point 3D point in camera space (x, y, z)
/// \param focal_length Focal length in pixels (fx, fy)
/// \param principal_point Principal point in pixels (cx, cy)
/// \return Projected 2D point in image space
GSPLAT_HOST_DEVICE inline auto project(
    glm::fvec3 const &camera_point,
    glm::fvec2 const &focal_length,
    glm::fvec2 const &principal_point
) -> glm::fvec2 {
    auto const xy = glm::fvec2(camera_point) / camera_point.z;
    auto const image_point = focal_length * xy + principal_point;
    return image_point;
}

/// \brief Compute the Jacobian of the pinhole projection:
/// J = d(image_point) / d(camera_point)
/// \param camera_point 3D point in camera space (x, y, z)
/// \param focal_length Focal length in pixels (fx, fy)
/// \param principal_point Principal point in pixels (cx, cy)
/// \return 3x2 Jacobian matrix
GSPLAT_HOST_DEVICE inline auto project_jac(
    glm::fvec3 const &camera_point,
    glm::fvec2 const &focal_length,
    glm::fvec2 const &principal_point
) -> glm::fmat3x2 {
    auto const rz = 1.0f / camera_point.z;
    auto const xy = glm::fvec2(camera_point) * rz;
    auto const J = glm::fmat3x2{
        focal_length[0] * rz,
        0.f,
        0.f,
        focal_length[1] * rz,
        -focal_length[0] * xy[0] * rz,
        -focal_length[1] * xy[1] * rz,
    };
    return J;
}

/// \brief Project a 3D point in camera space to 2D image space using pinhole
/// projection with lens distortion
/// \param camera_point 3D point in camera space (x, y, z)
/// \param focal_length Focal length in pixels (fx, fy)
/// \param principal_point Principal point in pixels (cx, cy)
/// \param radial_coeffs Radial distortion coefficients (k1, k2, k3, k4, k5, k6)
/// \param tangential_coeffs Tangential distortion coefficients (p1, p2)
/// \param thin_prism_coeffs Thin prism distortion coefficients (s1, s2, s3, s4)
/// \param min_radial_dist Minimum radial distortion threshold for numerical stability.
/// Default value is 0.8f.
/// \param max_radial_dist Maximum radial distortion threshold for numerical stability.
/// Default value is max float.
/// \return Pair of projected 2D point and validity flag
GSPLAT_HOST_DEVICE inline auto project(
    glm::fvec3 const &camera_point,
    glm::fvec2 const &focal_length,
    glm::fvec2 const &principal_point,
    std::array<float, 6> const &radial_coeffs,
    std::array<float, 2> const &tangential_coeffs,
    std::array<float, 4> const &thin_prism_coeffs,
    float const &min_radial_dist = DEFAULT_MIN_RADIAL_DIST,
    float const &max_radial_dist = DEFAULT_MAX_RADIAL_DIST
) -> std::pair<glm::fvec2, bool> {
    auto const xy = glm::fvec2(camera_point) / camera_point.z;
    auto const &[uv, valid_flag] = distortion(
        xy,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        min_radial_dist,
        max_radial_dist
    );
    if (!valid_flag)
        return {glm::fvec2{}, false};
    auto const image_point = focal_length * uv + principal_point;
    return {image_point, true};
}

/// \brief Compute the Jacobian of the pinhole projection with lens distortion:
/// J = d(image_point) / d(camera_point)
/// \param camera_point 3D point in camera space (x, y, z)
/// \param focal_length Focal length in pixels (fx, fy)
/// \param principal_point Principal point in pixels (cx, cy)
/// \param radial_coeffs Radial distortion coefficients (k1, k2, k3, k4, k5, k6)
/// \param tangential_coeffs Tangential distortion coefficients (p1, p2)
/// \param thin_prism_coeffs Thin prism distortion coefficients (s1, s2, s3, s4)
/// \param min_radial_dist Minimum radial distortion threshold for numerical stability.
/// Default value is 0.8f.
/// \param max_radial_dist Maximum radial distortion threshold for numerical stability.
/// Default value is max float.
/// \return Pair of 3x2 Jacobian matrix and validity flag
GSPLAT_HOST_DEVICE inline auto project_jac(
    glm::fvec3 const &camera_point,
    glm::fvec2 const &focal_length,
    glm::fvec2 const &principal_point,
    std::array<float, 6> const &radial_coeffs,
    std::array<float, 2> const &tangential_coeffs,
    std::array<float, 4> const &thin_prism_coeffs,
    float const &min_radial_dist = DEFAULT_MIN_RADIAL_DIST,
    float const &max_radial_dist = DEFAULT_MAX_RADIAL_DIST
) -> std::pair<glm::fmat3x2, bool> {
    auto const rz = 1.0f / camera_point.z;
    auto const xy = glm::fvec2(camera_point) * rz;

    auto const &[J_uv_xy, icD, r2, valid_flag] = distortion_jac(
        xy,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        min_radial_dist,
        max_radial_dist
    );
    if (!valid_flag)
        return {glm::fmat3x2{}, false};

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

/// \brief Unproject a 2D point from image space to camera space
/// using pinhole projection
/// \param image_point 2D point in image space
/// \param focal_length Focal length in pixels (fx, fy)
/// \param principal_point Principal point in pixels (cx, cy)
/// \return ray direction in camera space
GSPLAT_HOST_DEVICE inline auto unproject(
    glm::fvec2 const &image_point,
    glm::fvec2 const &focal_length,
    glm::fvec2 const &principal_point
) -> glm::fvec3 {
    auto const uv = (image_point - principal_point) / focal_length;
    auto const dir = glm::fvec3{uv[0], uv[1], 1.0f};
    // normalize
    return glm::normalize(dir);
}

/// \brief Unproject a 2D point from image space to camera space
/// using pinhole projection with lens distortion
/// \param image_point 2D point in image space
/// \param focal_length Focal length in pixels (fx, fy)
/// \param principal_point Principal point in pixels (cx, cy)
/// \param radial_coeffs Radial distortion coefficients (k1, k2, k3, k4, k5, k6)
/// \param tangential_coeffs Tangential distortion coefficients (p1, p2)
/// \param thin_prism_coeffs Thin prism distortion coefficients (s1, s2, s3, s4)
/// \param min_radial_dist Minimum radial distortion threshold for numerical stability.
/// Default value is 0.8f.
/// \param max_radial_dist Maximum radial distortion threshold for numerical stability.
/// Default value is max float.
/// \return Pair of ray direction and validity flag
GSPLAT_HOST_DEVICE inline auto unproject(
    glm::fvec2 const &image_point,
    glm::fvec2 const &focal_length,
    glm::fvec2 const &principal_point,
    std::array<float, 6> const &radial_coeffs,
    std::array<float, 2> const &tangential_coeffs,
    std::array<float, 4> const &thin_prism_coeffs,
    float const &min_radial_dist = DEFAULT_MIN_RADIAL_DIST,
    float const &max_radial_dist = DEFAULT_MAX_RADIAL_DIST
) -> std::pair<glm::fvec3, bool> {
    auto const uv = (image_point - principal_point) / focal_length;
    auto const &[xy, valid_flag] = undistortion(
        uv,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        min_radial_dist,
        max_radial_dist
    );
    if (!valid_flag)
        return {glm::fvec3{}, false};
    auto const dir = glm::fvec3{xy[0], xy[1], 1.0f};
    // normalize
    return {glm::normalize(dir), true};
}

} // namespace cugsplat::pinhhole