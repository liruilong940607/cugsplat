#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>

#include "macros.h" // for GSPLAT_HOST_DEVICE

namespace gsplat {

// Return x = L⁻¹ y, where L is lower triangular (col <= row)
// forward_substitution
inline GSPLAT_HOST_DEVICE auto
forward_substitution(glm::mat3 const &L, glm::fvec3 const &y) -> glm::fvec3 {
    constexpr float eps = 1e-10f;
    glm::fvec3 x;
    x[0] = y[0] / L[0][0];
    x[1] = (y[1] - L[0][1] * x[0]) / (std::abs(L[1][1]) > eps ? L[1][1] : eps);
    x[2] = (y[2] - L[0][2] * x[0] - L[1][2] * x[1]) /
           (std::abs(L[2][2]) > eps ? L[2][2] : eps);
    return x;
}
#define cholesky_Linv_y forward_substitution

// Return x = (Lᵀ)⁻¹ y, where L is lower triangular (col <= row)
// backward_substitution
inline GSPLAT_HOST_DEVICE auto
backward_substitution(glm::mat3 const &L, glm::fvec3 const &y) -> glm::fvec3 {
    constexpr float eps = 1e-10f;
    glm::fvec3 x;
    x[2] = y[2] / (std::abs(L[2][2]) > eps ? L[2][2] : eps);
    x[1] = (y[1] - L[1][2] * x[2]) / (std::abs(L[1][1]) > eps ? L[1][1] : eps);
    x[0] = (y[0] - L[0][1] * x[1] - L[0][2] * x[2]) / L[0][0];
    return x;
}
#define cholesky_LTinv_y backward_substitution

// Given v_x = do/dx, return do/dL
inline GSPLAT_HOST_DEVICE auto forward_substitution_vjp(
    glm::mat3 const &L, glm::fvec3 const &x, glm::fvec3 const &v_x
) -> glm::mat3 {
    // Solve Lᵀ w = v_x  (upper‑triangular back substitution)
    glm::fvec3 w = backward_substitution(L, v_x);
    // v_L = − w ⊗ xᵀ
    glm::mat3 v_L = -glm::outerProduct(w, x);
    return v_L;
}
#define cholesky_Linv_y_vjp forward_substitution_vjp

// Return x = W⁻¹ y  with  W = L Lᵀ
// ⇒  x = (Lᵀ)⁻¹ (L⁻¹ y)
inline GSPLAT_HOST_DEVICE auto
cholesky_Winv_y(const glm::mat3 &L, const glm::fvec3 &y) -> glm::fvec3 {
    return backward_substitution(L, forward_substitution(L, y));
}

// Return W⁻¹, where  W = L Lᵀ, using 6 triangular solves
inline GSPLAT_HOST_DEVICE auto cholesky_Winv(const glm::mat3 &L) -> glm::mat3 {
    glm::mat3 Winv;

#pragma unroll
    for (int i = 0; i < 3; ++i) { // basis vectors e₀,e₁,e₂
        glm::fvec3 ei(0.0f);
        ei[i] = 1.0f;
        Winv[i] = cholesky_Winv_y(L, ei); // column i of W⁻¹
    }

    // enforce symmetry to kill tiny FP asymmetry
    Winv = 0.5f * (Winv + glm::transpose(Winv));
    return Winv;
}

// Return L⁻¹
inline GSPLAT_HOST_DEVICE auto cholesky_Linv(const glm::mat3 &L) -> glm::mat3 {
    glm::mat3 Linv;

#pragma unroll
    for (int i = 0; i < 3; ++i) { // basis vectors e₀,e₁,e₂
        glm::fvec3 ei(0.0f);
        ei[i] = 1.0f;
        Linv[i] = forward_substitution(L, ei); // column i of L⁻¹
    }

    return Linv;
}

// Given do/d(L⁻¹), return do/dL = – L⁻ᵀ · v_Linv · L⁻ᵀ .
inline GSPLAT_HOST_DEVICE auto
cholesky_Linv_vjp(const glm::mat3 &L, const glm::mat3 &v_Linv) -> glm::mat3 {
    // // TODO: this is stupid, should use forward_substitution
    // backward_substitution const auto Linv = cholesky_Linv(L); // L⁻¹ const
    // auto LinvT = glm::transpose(Linv); return - LinvT * v_Linv * LinvT;

    // Step 1: X ← L⁻ᵀ · v_Linv   (solve Lᵀ X = v_Linv)
    glm::mat3 X;
#pragma unroll
    for (int c = 0; c < 3; ++c) { // solve column‑by‑column
        X[c] = backward_substitution(L, v_Linv[c]);
    }

    // Step 2: G ← – X · L⁻ᵀ = - (L⁻¹ Xᵀ)ᵀ
    glm::mat3 G;
#pragma unroll
    for (int r = 0; r < 3; ++r) { // solve row‑by‑row
        glm::fvec3 y =
            forward_substitution(L, glm::fvec3(X[0][r], X[1][r], X[2][r]));
        G[0][r] = -y[0];
        G[1][r] = -y[1];
        G[2][r] = -y[2];
    }
    return G;
}

// Cholesky decomposition for 3x3 matrices
// Returns L such that A = L * L^T, where L is lower triangular (col <= row)
inline GSPLAT_HOST_DEVICE auto cholesky(const glm::mat3 &A
) -> std::pair<glm::mat3, bool> {
    // float alpha = 1e-4f * glm::compMax(scale * scale);
    // Covar += glm::mat3(alpha);

    glm::mat3 L(0.0f);

    // ---- Cholesky in double ---------------
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

    // ---- write back in float -----------------------------------
    L[0][0] = static_cast<float>(l00);
    L[0][1] = static_cast<float>(l01);
    L[1][1] = static_cast<float>(l11);
    L[0][2] = static_cast<float>(l02);
    L[1][2] = static_cast<float>(l12);
    L[2][2] = static_cast<float>(l22);
    return {L, true};
}

// Given do/dL, return do/dA
// "Differentiation of the Cholesky decomposition"
// https://homepages.inf.ed.ac.uk/imurray2/pub/16choldiff/choldiff.pdf
inline GSPLAT_HOST_DEVICE auto
cholesky_vjp(const glm::mat3 &L, const glm::mat3 &v_L) -> glm::mat3 {
    // S = Lᵀ · v_L
    glm::mat3 S = glm::transpose(L) * v_L;

    // P = tril(S) - 0.5 * diag(S)
    glm::mat3 P(0.0f);
    P[0][0] = 0.5f * S[0][0];
    P[0][1] = S[0][1];
    P[1][1] = 0.5f * S[1][1];
    P[0][2] = S[0][2];
    P[1][2] = S[1][2];
    P[2][2] = 0.5f * S[2][2];

    glm::mat3 Sym = P + glm::transpose(P); // symmetric 3×3

    // v_A = (Lᵀ)⁻¹ Sym L⁻¹
    glm::mat3 Linv = cholesky_Linv(L);                        // L⁻¹
    glm::mat3 v_A = 0.5f * glm::transpose(Linv) * Sym * Linv; // (Lᵀ)⁻¹ Sym L⁻¹

    // Optionally enforce symmetry to kill tiny FP asymmetry
    v_A = 0.5f * (v_A + glm::transpose(v_A));
    return v_A;
}

} // namespace gsplat
