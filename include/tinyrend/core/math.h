#pragma once

#include <cstdint>

#include "tinyrend/common/macros.h"
#include "tinyrend/common/mat.h"
#include "tinyrend/common/vec.h"

namespace tinyrend::math {

template <size_t N_COEFFS>
inline TREND_HOST_DEVICE float
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
inline TREND_HOST_DEVICE bool is_all_zero(std::array<float, N> const &arr) {
#pragma unroll
    for (size_t i = SKIP_FIRST_N; i < N; ++i) {
        if (fabsf(arr[i]) >= std::numeric_limits<float>::epsilon())
            return false;
    }
    return true;
}

template <typename T, size_t N>
inline TREND_HOST_DEVICE vec<T, N> safe_normalize(const vec<T, N> &x) {
    const T l2 = dot(x, x);
    return (l2 > 0.0f) ? (x * rsqrtf(l2)) : x;
}

template <typename T, size_t N>
inline TREND_HOST_DEVICE vec<T, N>
safe_normalize_vjp(const vec<T, N> &x, const vec<T, N> &v_out) {
    const T l2 = dot(x, x);
    if (l2 > 0.0f) {
        const T il = rsqrtf(l2);
        const T il3 = il * il * il;
        return il * v_out - il3 * dot(v_out, x) * x;
    }
    return v_out;
}

} // namespace tinyrend::math