#pragma once

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <tuple>

#include "utils/macros.h" // for GSPLAT_HOST_DEVICE
#include "utils/types.h"  // for Maybe
#include "utils/cholesky3x3.h"

namespace gsplat {

inline GSPLAT_HOST_DEVICE float rsqrtf(const float x) {
#ifdef __CUDACC__
    return ::rsqrtf(x); // use CUDA’s fast rsqrtf()
#else
    return 1.0f / std::sqrt(x); // use standard sqrt on CPU
#endif
}

template <glm::length_t L, glm::qualifier Q = glm::defaultp>
inline GSPLAT_HOST_DEVICE glm::vec<L, float, Q>
safe_normalize(const glm::vec<L, float, Q> &x) {
    const float l2 = glm::dot(x, x);
    return (l2 > 0.0f) ? (x * rsqrtf(l2)) : x;
}

template <glm::length_t L, glm::qualifier Q = glm::defaultp>
inline GSPLAT_HOST_DEVICE glm::vec<L, float, Q> safe_normalize_vjp(
    const glm::vec<L, float, Q> &x, const glm::vec<L, float, Q> &v_out
) {
    const float l2 = glm::dot(x, x);
    if (l2 > 0.0f) {
        const float il = rsqrtf(l2);
        const float il3 = il * il * il;
        return il * v_out - il3 * glm::dot(v_out, x) * x;
    }
    return v_out;
}

// Convert a quaternion to a rotation matrix. We fused in the quaternion
// normalization step to avoid the need for a separate normalization pass.
inline GSPLAT_HOST_DEVICE auto quat_to_rotmat(const glm::fvec4 &quat
) -> glm::fmat3 {
    auto const quat_n = quat * rsqrtf(glm::dot(quat, quat));
    float w = quat_n[0], x = quat_n[1], y = quat_n[2], z = quat_n[3];
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return glm::fmat3(
        (1.f - 2.f * (y2 + z2)),
        (2.f * (xy + wz)),
        (2.f * (xz - wy)), // 1st col
        (2.f * (xy - wz)),
        (1.f - 2.f * (x2 + z2)),
        (2.f * (yz + wx)), // 2nd col
        (2.f * (xz + wy)),
        (2.f * (yz - wx)),
        (1.f - 2.f * (x2 + y2)) // 3rd col
    );
}

// Given d(o)/d(R), Return d(o)/d(quat)
inline GSPLAT_HOST_DEVICE auto
quat_to_rotmat_vjp(const glm::fvec4 quat, const glm::fmat3 v_R) -> glm::fvec4 {
    auto const inv_norm = rsqrtf(glm::dot(quat, quat));
    auto const quat_n = quat * inv_norm;
    float w = quat_n[0], x = quat_n[1], y = quat_n[2], z = quat_n[3];
    auto const v_quat_n = glm::fvec4(
        2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
               z * (v_R[0][1] - v_R[1][0])),
        2.f *
            (-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
             z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])),
        2.f * (x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
               z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])),
        2.f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
               2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0]))
    );
    auto const v_quat =
        (v_quat_n - glm::dot(v_quat_n, quat_n) * quat_n) * inv_norm;
    return v_quat;
}

// Convert {quat, scale} to RS.
inline GSPLAT_HOST_DEVICE auto
quat_scale_to_scaled_rotmat(
    glm::fvec4 const &quat, glm::fvec3 const &scale
) -> glm::fmat3 {
    auto const R = quat_to_rotmat(quat);
    auto const M = glm::fmat3(R[0] * scale[0], R[1] * scale[0], R[2] * scale[0]);
    return M;
}

inline GSPLAT_HOST_DEVICE auto
quat_scale_to_scaled_rotmat_vjp(
    // inputs
    glm::fvec4 const &quat, glm::fvec3 const &scale,
    // output gradients
    glm::fmat3 const &v_M
) -> std::pair<glm::fvec4, glm::fvec3> {
    auto const R = quat_to_rotmat(quat);    
    auto const v_R = glm::fmat3(v_M[0] * scale[0], v_M[1] * scale[0], v_M[2] * scale[0]);
    auto const v_quat = quat_to_rotmat_vjp(quat, v_R);
    auto const v_scale = glm::fvec3{
        glm::dot(v_M[0], R[0]), glm::dot(v_M[1], R[1]), glm::dot(v_M[2], R[2])
    };
    return {v_quat, v_scale};
}

// Convert {quat, scale} to a covariance matrix: RSSᵀRᵀ
inline GSPLAT_HOST_DEVICE auto
quat_scale_to_covar(
    glm::fvec4 const &quat, glm::fvec3 const &scale
) -> glm::fmat3 {
    auto const M = quat_scale_to_scaled_rotmat(quat, scale);
    return M * glm::transpose(M);
}

inline GSPLAT_HOST_DEVICE auto
quat_scale_to_covar_vjp(
    // inputs
    glm::fvec4 const &quat, glm::fvec3 const &scale, 
    // output gradients
    glm::fmat3 const &v_covar
) -> std::pair<glm::fvec4, glm::fvec3> {
    auto const R = quat_to_rotmat(quat);
    auto const M = glm::fmat3(R[0] * scale[0], R[1] * scale[0], R[2] * scale[0]);
    
    auto const v_M = (v_covar + glm::transpose(v_covar)) * M;
    auto const v_R = glm::fmat3(v_M[0] * scale[0], v_M[1] * scale[0], v_M[2] * scale[0]);

    auto const v_quat = quat_to_rotmat_vjp(quat, v_R);
    auto const v_scale = glm::fvec3{
        glm::dot(v_M[0], R[0]), glm::dot(v_M[1], R[1]), glm::dot(v_M[2], R[2])
    };
    return {v_quat, v_scale};
}


// Calculate the maximum response of a 3D Gaussian along a ray.
//
// return the sigma value defined as:
//      sigma = 1/2 * (x - mu)ᵀ * covar⁻¹ * (x - mu)
// Where x is the point on the ray closest to mu:
//      x = o + t * d, t = -((o - mu)ᵀ covar⁻¹ d) / (dᵀ covar⁻¹ d)
//
// Note for numerical stability, we take in the cholesky factorization L
// of the covariance matrix, where L is lower triangular:
//     covar = L * Lᵀ
inline GSPLAT_HOST_DEVICE auto gaussian_max_response_along_ray(
    glm::fvec3 const &mu,
    glm::fmat3 const &L,
    glm::fvec3 const &ray_o,
    glm::fvec3 const &ray_d
) -> float {
    auto const o_minus_mu = ray_o - mu;
    auto const gro = forward_substitution(L, o_minus_mu);
    auto const grd = forward_substitution(L, ray_d);
    auto const grd_n = safe_normalize(grd);
    auto const gcrod = glm::cross(grd_n, gro);
    auto const grayDist = glm::dot(gcrod, gcrod);
    auto const sigma = 0.5f * grayDist;
    return sigma;
}

// Return both the forward outputs and the backward gradients
// - sigma
// - v_mu: d(sigma)/d(mu)
// - v_covar: d(sigma)/d(covar)
inline GSPLAT_HOST_DEVICE auto gaussian_max_response_along_ray_wgrad(
    glm::fvec3 const &mu,
    glm::fmat3 const &L,
    glm::fvec3 const &ray_o,
    glm::fvec3 const &ray_d
) -> std::tuple<float, glm::fvec3, glm::fmat3> {
    // forward
    auto const o_minus_mu = ray_o - mu;
    auto const gro = cholesky_Linv_y(L, o_minus_mu);
    auto const grd = cholesky_Linv_y(L, ray_d);
    auto const grd_n = safe_normalize(grd);
    auto const gcrod = glm::cross(grd_n, gro);
    auto const grayDist = glm::dot(gcrod, gcrod);
    auto const sigma = 0.5f * grayDist;
    // backward
    auto const v_grd_n = -glm::cross(gcrod, gro);
    auto const v_gro = glm::cross(gcrod, grd_n);
    auto const v_grd = safe_normalize_vjp(grd, v_grd_n);
    auto const v_mu = -cholesky_LTinv_y(L, v_gro);
    auto const v_L =
        cholesky_Linv_y_vjp(L, grd, v_grd) + cholesky_Linv_y_vjp(L, gro, v_gro);
    auto const v_covar = cholesky_vjp(L, v_L);
    return {sigma, v_mu, v_covar};
}

// Calculate the maximum response of a 3D Gaussian along a ray.
//
// return the sigma value defined as:
//      sigma = 1/2 * (x - mu)ᵀ * covar⁻¹ * (x - mu)
// Where x is the point on the ray closest to mu:
//      x = o + t * d, t = -((o - mu)ᵀ covar⁻¹ d) / (dᵀ covar⁻¹ d)
//
// This version takes in the covariance matrix in quaternion and scale form:
//     covar = R * S * Sᵀ * Rᵀ
// Where R is the rotation matrix of the quaternion, and S is the diagonal
// matrix
//     S = diag(scale[0], scale[1], scale[2])
// Note the quaternion *does not need to be normalized* here, as we will
// normalize it in the function.
inline GSPLAT_HOST_DEVICE auto gaussian_max_response_along_ray(
    glm::fvec3 const &mu,
    glm::fvec4 const &quat,
    glm::fvec3 const &scale,
    glm::fvec3 const &ray_o,
    glm::fvec3 const &ray_d
) -> float {
    auto const R = quat_to_rotmat(quat);
    auto const Sinv = glm::fmat3(
        1.f / scale[0],
        0.f,
        0.f,
        0.f,
        1.f / scale[1],
        0.f,
        0.f,
        0.f,
        1.f / scale[2]
    );
    auto const Mt = glm::transpose(R * Sinv);
    auto const o_minus_mu = ray_o - mu;
    auto const gro = Mt * o_minus_mu;
    auto const grd = Mt * ray_d;
    auto const grd_n = safe_normalize(grd);
    auto const gcrod = glm::cross(grd_n, gro);
    auto const grayDist = glm::dot(gcrod, gcrod);
    auto const sigma = 0.5f * grayDist;
    return sigma;
}

// Return both the forward outputs and the backward gradients
// - sigma
// - v_mu: d(sigma)/d(mu)
// - v_quat: d(sigma)/d(quat)
// - v_scale: d(sigma)/d(scale)
inline GSPLAT_HOST_DEVICE auto gaussian_max_response_along_ray_wgrad(
    glm::fvec3 const &mu,
    glm::fvec4 const &quat,
    glm::fvec3 const &scale,
    glm::fvec3 const &ray_o,
    glm::fvec3 const &ray_d
) -> std::tuple<float, glm::fvec3, glm::fvec4, glm::fvec3> {
    // forward
    auto const R = quat_to_rotmat(quat);
    auto const Sinv = glm::fmat3(
        1.f / scale[0],
        0.f,
        0.f,
        0.f,
        1.f / scale[1],
        0.f,
        0.f,
        0.f,
        1.f / scale[2]
    );
    auto const Mt = glm::transpose(R * Sinv);
    auto const o_minus_mu = ray_o - mu;
    auto const gro = Mt * o_minus_mu;
    auto const grd = Mt * ray_d;
    auto const grd_n = safe_normalize(grd);
    auto const gcrod = glm::cross(grd_n, gro);
    auto const grayDist = glm::dot(gcrod, gcrod);
    auto const sigma = 0.5f * grayDist;

    // backward
    auto const v_grd_n = -glm::cross(gcrod, gro);
    auto const v_gro = glm::cross(gcrod, grd_n);
    auto const v_grd = safe_normalize_vjp(grd, v_grd_n);
    auto const v_Mt =
        glm::outerProduct(v_grd, ray_d) + glm::outerProduct(v_gro, o_minus_mu);
    auto const v_mu = -glm::transpose(Mt) * v_gro;
    auto const v_M = glm::transpose(v_Mt);
    auto const v_R = v_M * Sinv;
    auto const v_quat = quat_to_rotmat_vjp(quat, v_R);
    auto const v_scale = glm::fvec3{
        -Sinv[0][0] * Sinv[0][0] *
            (R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2]),
        -Sinv[1][1] * Sinv[1][1] *
            (R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2]),
        -Sinv[2][2] * Sinv[2][2] *
            (R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2])
    };

    return {sigma, v_mu, v_quat, v_scale};
}

// Calculate the maximum response of a 3D Gaussian along a ray.
//
// In this function we will blur the 3D Gaussian with a 2D filter perpendicular
// to the ray direction, which leads to a new covariace matrix.
inline GSPLAT_HOST_DEVICE auto gaussian_max_response_along_ray_filter2d(
    glm::fvec3 const &mu,
    glm::fmat3 const &L,
    glm::fmat3 const &L2,
    glm::fvec3 const &ray_o,
    glm::fvec3 const &ray_d
) -> float {
    auto const o_minus_mu = ray_o - mu;
    auto const gro2 = forward_substitution(L2, o_minus_mu);
    auto const grd2 = forward_substitution(L2, ray_d);
    auto const grd_n = safe_normalize(grd2);
    auto const gcrod = glm::cross(grd_n, gro2);
    auto const grayDist = glm::dot(gcrod, gcrod);
    auto const sigma = 0.5f * grayDist;

    // compute coeff
    auto const grd = forward_substitution(L, ray_d);
    auto const sqrt_det = L[0][0] * L[1][1] * L[2][2];
    auto const sqrt_det2 = L2[0][0] * L2[1][1] * L2[2][2];
    if (std::isnan(sqrt_det)) {
        return -1.f;
    }
    auto const grdgrd = glm::dot(grd, grd);
    auto const grd2grd2 = glm::dot(grd2, grd2);
    auto const coeff = sqrt_det / sqrt_det2 * sqrt(grdgrd / grd2grd2);
    if (coeff < (1.f / 255.f) || std::isnan(coeff)) {
        return -1.f;
    }
    return sigma - log(coeff);
}

// Return both the forward outputs and the backward gradients
// - o
// - v_mu: d(o)/d(mu)
// - v_covar: d(o)/d(covar)
// - v_covar2: d(o)/d(covar2)
inline GSPLAT_HOST_DEVICE auto gaussian_max_response_along_ray_filter2d_wgrad(
    glm::fvec3 const &mu,
    glm::fmat3 const &L,
    glm::fmat3 const &L2,
    glm::fvec3 const &ray_o,
    glm::fvec3 const &ray_d
) -> std::tuple<float, glm::fvec3, glm::fmat3, glm::fmat3> {
    // forward
    auto const o_minus_mu = ray_o - mu;
    auto const gro2 = forward_substitution(L2, o_minus_mu);
    auto const grd2 = forward_substitution(L2, ray_d);
    auto const grd_n = safe_normalize(grd2);
    auto const gcrod = glm::cross(grd_n, gro2);
    auto const grayDist = glm::dot(gcrod, gcrod);
    auto const sigma = 0.5f * grayDist;

    // compute coeff
    auto const grd = forward_substitution(L, ray_d);
    auto const sqrt_det = L[0][0] * L[1][1] * L[2][2];
    auto const sqrt_det2 = L2[0][0] * L2[1][1] * L2[2][2];
    if (std::isnan(sqrt_det)) {
        return {-1.f, glm::fvec3(0.f), glm::fmat3(0.f), glm::fmat3(0.f)};
    }
    auto const grdgrd = glm::dot(grd, grd);
    auto const grd2grd2 = glm::dot(grd2, grd2);
    auto const coeff = sqrt_det / sqrt_det2 * sqrt(grdgrd / grd2grd2);
    if (coeff < (1.f / 255.f) || std::isnan(coeff)) {
        return {-1.f, glm::fvec3(0.f), glm::fmat3(0.f), glm::fmat3(0.f)};
    }

    // backward: d(sigma)/d(mu), d(sigma)/d(covar2)
    auto const v_grd_n = -glm::cross(gcrod, gro2);
    auto const v_gro = glm::cross(gcrod, grd_n);
    auto const v_grd = safe_normalize_vjp(grd2, v_grd_n);
    auto const v_mu = -cholesky_LTinv_y(L2, v_gro);
    auto const v_L2 = cholesky_Linv_y_vjp(L2, grd2, v_grd) +
                      cholesky_Linv_y_vjp(L2, gro2, v_gro);
    auto v_covar2 = cholesky_vjp(L2, v_L2);

    // backward: d(-log(coeff))/d(covar), d(-log(coeff))/d(covar2)
    auto const ggrd = backward_substitution(L, grd);
    auto const ggrd2 = backward_substitution(L2, grd2);
    auto const v_covar =
        -0.5f * (cholesky_Winv(L) - glm::outerProduct(ggrd, ggrd) / grdgrd);
    v_covar2 +=
        0.5f * (cholesky_Winv(L2) - glm::outerProduct(ggrd2, ggrd2) / grd2grd2);
    return {sigma - log(coeff), v_mu, v_covar, v_covar2};
}

// Solve the tight axis-aligned bounding box radius for a Gaussian defined as
//      y = prefactor * exp(-1/2 * xᵀ * covar⁻¹ * x)
// at the given y value.
inline GSPLAT_HOST_DEVICE auto solve_tight_radius(
    glm::fmat2 covar, float prefactor, float y = 1.0f / 255.0f
) -> glm::fvec2 {
    if (prefactor < y) {
        return glm::fvec2(0.0f);
    }

    // Threshold distance squared on ellipse
    float sigma = -logf(y / prefactor);
    float Q = 2.0f * sigma;

    // Eigenvalues of covariance matrix
    float det = glm::determinant(covar);
    float half_trace = 0.5f * (covar[0][0] + covar[1][1]);
    float discrim = sqrtf(std::max(0.f, half_trace * half_trace - det));
    float lambda1 = half_trace + discrim;
    float lambda2 = half_trace - discrim;

    // Compute unit eigenvectors
    glm::fvec2 v1, v2;
    if (covar[0][1] == 0.0f) {
        // pick the axis that corresponds to the larger eigenvalue
        if (covar[0][0] >= covar[1][1]) {
            v1 = glm::fvec2(1,0);
        } else {
            v1 = glm::fvec2(0,1);
        }
    } else {
        v1 = glm::normalize(glm::fvec2(lambda1 - covar[1][1], covar[0][1]));
    }
    v2 = glm::fvec2(-v1.y, v1.x);  // perpendicular

    // Scale eigenvectors with eigenvalues
    v1 *= sqrtf(Q * lambda1);
    v2 *= sqrtf(Q * lambda2);

    // Compute max extent along world x/y axes (bounding box)
    auto const radius = glm::sqrt(v1 * v1 + v2 * v2);
    return radius;
}



} // namespace gsplat