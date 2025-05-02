#pragma once

#include <cstdint>
#include <glm/glm.hpp>

#include "core/macros.h"

namespace gsplat::math {

inline GSPLAT_HOST_DEVICE float rsqrtf(const float x) {
#ifdef __CUDACC__
    return ::rsqrtf(x); // use CUDA's fast rsqrtf()
#else
    return 1.0f / std::sqrt(x); // use standard sqrt on CPU
#endif
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

template <size_t SKIP_FIRST_N = 0, size_t N>
inline GSPLAT_HOST_DEVICE bool is_all_zero(std::array<float, N> const &arr) {
#pragma unroll
    for (size_t i = SKIP_FIRST_N; i < N; ++i) {
        if (fabsf(arr[i]) >= std::numeric_limits<float>::epsilon())
            return false;
    }
    return true;
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

} // namespace gsplat::math