// Cholesky-related linear algebra utilities for 3x3 matrices
#pragma once

#include <algorithm>
#include <glm/glm.hpp>

#include "core/macros.h" // for GSPLAT_HOST_DEVICE

namespace cugsplat {

// Solve Lx = y where L is lower triangular
inline GSPLAT_HOST_DEVICE glm::fvec3
forward_substitution(const glm::fmat3 &L, const glm::fvec3 &y) {
    constexpr float eps = 1e-10f;
    glm::fvec3 x;
    x[0] = y[0] / L[0][0];
    x[1] = (y[1] - L[0][1] * x[0]) / (std::abs(L[1][1]) > eps ? L[1][1] : eps);
    x[2] = (y[2] - L[0][2] * x[0] - L[1][2] * x[1]) /
           (std::abs(L[2][2]) > eps ? L[2][2] : eps);
    return x;
}
#define cholesky_Linv_y forward_substitution

// Solve Lᵀx = y where L is lower triangular
inline GSPLAT_HOST_DEVICE glm::fvec3
backward_substitution(const glm::fmat3 &L, const glm::fvec3 &y) {
    constexpr float eps = 1e-10f;
    glm::fvec3 x;
    x[2] = y[2] / (std::abs(L[2][2]) > eps ? L[2][2] : eps);
    x[1] = (y[1] - L[1][2] * x[2]) / (std::abs(L[1][1]) > eps ? L[1][1] : eps);
    x[0] = (y[0] - L[0][1] * x[1] - L[0][2] * x[2]) / L[0][0];
    return x;
}
#define cholesky_LTinv_y backward_substitution

// VJP for forward substitution: do/dL from do/dx
inline GSPLAT_HOST_DEVICE glm::fmat3 forward_substitution_vjp(
    const glm::fmat3 &L, const glm::fvec3 &x, const glm::fvec3 &v_x
) {
    glm::fvec3 w = backward_substitution(L, v_x);
    return -glm::outerProduct(w, x);
}
#define cholesky_Linv_y_vjp forward_substitution_vjp

// Solve Wx = y, with W = LLᵀ
inline GSPLAT_HOST_DEVICE glm::fvec3
cholesky_Winv_y(const glm::fmat3 &L, const glm::fvec3 &y) {
    return backward_substitution(L, forward_substitution(L, y));
}

// Compute W⁻¹ = (LLᵀ)⁻¹
inline GSPLAT_HOST_DEVICE glm::fmat3 cholesky_Winv(const glm::fmat3 &L) {
    glm::fmat3 Winv;
    for (int i = 0; i < 3; ++i) {
        glm::fvec3 ei(0.0f);
        ei[i] = 1.0f;
        Winv[i] = cholesky_Winv_y(L, ei);
    }
    return 0.5f * (Winv + glm::transpose(Winv));
}

// Compute L⁻¹ from L
inline GSPLAT_HOST_DEVICE glm::fmat3 cholesky_Linv(const glm::fmat3 &L) {
    glm::fmat3 Linv;
    for (int i = 0; i < 3; ++i) {
        glm::fvec3 ei(0.0f);
        ei[i] = 1.0f;
        Linv[i] = forward_substitution(L, ei);
    }
    return Linv;
}

// VJP for cholesky_Linv: do/dL from do/d(L⁻¹)
inline GSPLAT_HOST_DEVICE glm::fmat3
cholesky_Linv_vjp(const glm::fmat3 &L, const glm::fmat3 &v_Linv) {
    glm::fmat3 X;
    for (int c = 0; c < 3; ++c) {
        X[c] = backward_substitution(L, v_Linv[c]);
    }
    glm::fmat3 G;
    for (int r = 0; r < 3; ++r) {
        glm::fvec3 y(X[0][r], X[1][r], X[2][r]);
        glm::fvec3 solved = forward_substitution(L, y);
        G[0][r] = -solved[0];
        G[1][r] = -solved[1];
        G[2][r] = -solved[2];
    }
    return G;
}

// Compute Cholesky decomposition: A = LLᵀ
inline GSPLAT_HOST_DEVICE std::pair<glm::fmat3, bool> cholesky(const glm::fmat3 &A) {
    glm::fmat3 L(0.0f);
    constexpr double eps = 1e-10;

    double l00_sq = static_cast<double>(A[0][0]);
    if (l00_sq < eps)
        return {L, false};
    double l00 = std::sqrt(l00_sq);
    double l01 = static_cast<double>(A[1][0]) / l00;
    double l02 = static_cast<double>(A[2][0]) / l00;

    double l11_sq = static_cast<double>(A[1][1]) - l01 * l01;
    if (l11_sq < -eps)
        return {L, false};
    double l11 = std::sqrt(std::max(l11_sq, 0.0));

    double l12 = (static_cast<double>(A[2][1]) - l02 * l01) / l11;

    double l22_sq = static_cast<double>(A[2][2]) - l02 * l02 - l12 * l12;
    if (l22_sq < -eps)
        return {L, false};
    double l22 = std::sqrt(std::max(l22_sq, 0.0));

    L[0][0] = static_cast<float>(l00);
    L[0][1] = static_cast<float>(l01);
    L[0][2] = static_cast<float>(l02);
    L[1][1] = static_cast<float>(l11);
    L[1][2] = static_cast<float>(l12);
    L[2][2] = static_cast<float>(l22);
    return {L, true};
}

// VJP for cholesky: do/dA from do/dL
inline GSPLAT_HOST_DEVICE glm::fmat3
cholesky_vjp(const glm::fmat3 &L, const glm::fmat3 &v_L) {
    glm::fmat3 S = glm::transpose(L) * v_L;
    glm::fmat3 P(0.0f);
    P[0][0] = 0.5f * S[0][0];
    P[0][1] = S[0][1];
    P[0][2] = S[0][2];
    P[1][1] = 0.5f * S[1][1];
    P[1][2] = S[1][2];
    P[2][2] = 0.5f * S[2][2];

    glm::fmat3 Sym = P + glm::transpose(P);
    glm::fmat3 Linv = cholesky_Linv(L);
    glm::fmat3 v_A = 0.5f * glm::transpose(Linv) * Sym * Linv;
    return 0.5f * (v_A + glm::transpose(v_A));
}

} // namespace cugsplat
