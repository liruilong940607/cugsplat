#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>

#include "core/macros.h" // for GSPLAT_HOST_DEVICE

namespace gsplat {

template <size_t DIM> struct NewtonSolverResult;

template <> struct NewtonSolverResult<1> {
    using dtype = float;
    dtype x;
    bool converged;
};

template <> struct NewtonSolverResult<2> {
    using dtype = glm::fvec2;
    dtype x;
    bool converged;
};

template <
    size_t N_ITER,
    typename Func> // Func(float x) -> pair<float, float> = {residual, dfdx}
inline GSPLAT_HOST_DEVICE auto solver_newton_1d(
    const Func &f, const float &x0, const float epsilon = 1e-6f
) -> NewtonSolverResult<1> {
    auto x = x0;
    auto converged = false;

#pragma unroll
    for (size_t i = 0; i < N_ITER; ++i) {
        auto const &[r, dfdx] = f(x);
        auto const dx = r / dfdx;
        x -= dx;

        if (fabs(dx) < epsilon) {
            converged = true;
            break;
        }
    }

    return {x, converged};
}

template <
    size_t N_ITER,
    typename Func> // Func(xy) -> pair<residual[2], jacobian[2][2]>
inline GSPLAT_HOST_DEVICE auto solver_newton_2d(
    const Func &f, const glm::fvec2 &x0, float epsilon = 1e-6f
) -> NewtonSolverResult<2> {
    auto x = x0;
    auto converged = false;

#pragma unroll
    for (size_t i = 0; i < N_ITER; ++i) {
        auto const &[r, J] = f(x);

        auto const dx = r * glm::inverse(J);
        x -= dx;

        if (fabs(dx[0]) < epsilon && fabs(dx[1]) < epsilon) {
            converged = true;
            break;
        }
    }

    return {x, converged};
}

template <size_t DIM, size_t N_ITER, typename Func>
inline GSPLAT_HOST_DEVICE auto solver_newton(
    const Func &f,
    typename NewtonSolverResult<DIM>::dtype x0,
    const float epsilon = 1e-6f
) -> NewtonSolverResult<DIM> {
    static_assert(DIM == 1 || DIM == 2, "Only 1D and 2D supported");

    if constexpr (DIM == 1) {
        return solver_newton_1d<N_ITER>(f, x0, epsilon);
    } else {
        return solver_newton_2d<N_ITER>(f, x0, epsilon);
    }
}

// Solve a linear equation y=c_0+c_1*x and return the minimal positive root.
// If no positive root exists, return default_value.
inline GSPLAT_HOST_DEVICE float solver_linear_minimal_positive(
    std::array<float, 2> const &poly, float y, float default_value
) {
    auto const &[c0, c1] = poly;
    if (c1 == 0.f) {
        return default_value;
    }
    auto const root = (y - c0) / c1;
    return root > 0.f ? root : default_value;
}

// Solve a quadratic equation y=c_0+c_1*x+c_2*x^2 and return the minimal
// positive root. If no positive root exists, return default_value.
inline GSPLAT_HOST_DEVICE float solver_quadratic_minimal_positive(
    std::array<float, 3> const &poly, float y, float default_value
) {
    auto const &[c0, c1, c2] = poly;
    if (c2 == 0.f) {
        // reduce to y = c0 + c1*x
        return solver_linear_minimal_positive({c0, c1}, y, default_value);
    } else if (c0 == y) {
        // reduce to x(c1 + c2*x) = 0
        return solver_linear_minimal_positive({c1, c2}, 0.f, default_value);
    }

    // normalize the equation to 1 + ax + bx^2 = 0
    auto const a = c1 / (c0 - y);
    auto const b = c2 / (c0 - y);
    auto const discriminant = a * a - 4.f * b;
    if (discriminant < 0.f) {
        return default_value;
    }
    auto const root = (-a + std::sqrt(discriminant)) / 2.f;
    return root > 0.f ? root : default_value;
}

// Solve a cubic equation y=c_0+c_1*x+c_2*x^2+c_3*x^3 and return the minimal
// positive root. If no positive root exists, return default_value.
inline GSPLAT_HOST_DEVICE float solver_cubic_minimal_positive(
    std::array<float, 4> const &poly, float y, float default_value
) {
    auto const &[c0, c1, c2, c3] = poly;
    if (c3 == 0.f) {
        // reduce to y = c0 + c1*x + c2*x^2
        return solver_quadratic_minimal_positive(
            {c0, c1, c2}, y, default_value
        );
    } else if (c0 == y) {
        // reduce to x(c1 + c2*x + c3*x^2) = 0
        return solver_quadratic_minimal_positive(
            {c1, c2, c3}, 0.f, default_value
        );
    }

    // normalize the equation to 1 + ax + bx^2 + cx^3 = 0
    auto const a = c1 / (c0 - y);
    auto const b = c2 / (c0 - y);
    auto const c = c3 / (c0 - y);

    float boc = b / c;
    float boc2 = boc * boc;

    float t1 = (9.0f * a * boc - 2.0f * b * boc2 - 27.0f) / c;
    float t2 = 3.0f * a / c - boc2;
    float delta = t1 * t1 + 4.0f * t2 * t2 * t2;

    if (delta >= 0.0f) {
        // One real root case
        float d2 = std::sqrt(delta);
        float cube_root = std::cbrt((d2 + t1) / 2.0f);
        if (cube_root == 0.0f)
            return default_value;

        float soln = (cube_root - (t2 / cube_root) - boc) / 3.0f;
        return soln > 0.0f ? soln : default_value;
    } else {
        constexpr float PI = 3.14159265358979323846f;
        constexpr float two_third_pi = 2.f * PI / 3.f;

        // Three real roots case (delta < 0)
        float theta = std::atan2(std::sqrt(-delta), t1) / 3.0f;

        float t3 = 2.0f * std::sqrt(-t2);
        float min_soln = std::numeric_limits<float>::max();
        bool soln_found = false;

        for (int i : {-1, 0, 1}) {
            float angle = theta + i * two_third_pi;
            float s = (t3 * std::cos(angle) - boc) / 3.0f;
            if (s > 0.0f) {
                min_soln = std::min(min_soln, s);
                soln_found = true;
            }
        }
        return soln_found ? min_soln : default_value;
    }
}

// Solve a polynomial y=f(x) with
//
// f(x) = c_0*x^0 + c_1*x^1 + c_2*x^2 + c_3*x^3 + c_4*x^4 ...
//
// using newton method and return the minimal positive root.
// If no positive root exists or netwon does not converge, return default_value.
template <size_t N_ITER = 20, size_t N_COEFFS>
inline GSPLAT_HOST_DEVICE float solver_polyN_minimal_positive_newton(
    std::array<float, N_COEFFS> const &poly,
    float y,
    float guess,
    float default_value
) {
    // check if all coefficients from x^4 onwards are zero
    bool is_higher_order_all_zero = true;
#pragma unroll
    for (size_t i = 4; i < N_COEFFS; ++i) {
        if (poly[i] != 0.f) {
            is_higher_order_all_zero = false;
            break;
        }
    }
    if (is_higher_order_all_zero) {
        // reduce to cubic
        return solver_cubic_minimal_positive(
            {poly[0], poly[1], poly[2], poly[3]}, y, default_value
        );
    }

    // compute the derivative of the polynomial
    auto d_poly = std::array<float, N_COEFFS - 1>{};
#pragma unroll
    for (size_t i = 0; i < N_COEFFS - 1; ++i) {
        d_poly[i] = (i + 1) * poly[i + 1];
    }
    // define the residual and Jacobian of the equation
    auto const func = [&y, &poly, &d_poly](const float &x
                      ) -> std::pair<float, float> {
        auto const J = eval_poly_horner<N_COEFFS - 1>(d_poly, x);
        auto const residual = eval_poly_horner<N_COEFFS>(poly, x) - y;
        return {residual, J};
    };
    // solve the equation.
    auto const &[root, converged] =
        solver_newton<1, N_ITER>(func, guess, 1e-6f);
    return (converged && root > 0.f) ? root : default_value;
}

// Solve a polynomial y=f(x) with
//
// f(x) = c_0*x^0 + c_1*x^1 + c_2*x^2 + c_3*x^3 + c_4*x^4 ...
//
// and return the minimal positive root. Use analytically derived formula for
// up to cubic polynomials and newton method for higher order polynomials.
// If no positive root exists or newton does not converge, return default_value.
template <size_t N_ITER = 20, size_t N_COEFFS>
inline GSPLAT_HOST_DEVICE float solver_poly_minimal_positive(
    std::array<float, N_COEFFS> const &poly,
    float y,
    float guess,
    float default_value
) {
    if constexpr (N_COEFFS == 1) {
        // f(x) = c_0*x^0
        return default_value;
    } else if constexpr (N_COEFFS == 2) { // linear
        // f(x) = c_0 + c_1*x
        return solver_linear_minimal_positive(poly, y, default_value);
    } else if constexpr (N_COEFFS == 3) { // quadratic
        // f(x) = c_0 + c_1*x + c_2*x^2
        return solver_quadratic_minimal_positive(poly, y, default_value);
    } else if constexpr (N_COEFFS == 4) { // cubic
        // f(x) = c_0 + c_1*x + c_2*x^2 + c_3*x^3
        return solver_cubic_minimal_positive(poly, y, default_value);
    } else {
        // f(x) = c_0 + c_1*x + c_2*x^2 + c_3*x^3 + c_4*x^4 ...
        return solver_polyN_minimal_positive_newton<N_ITER>(
            poly, y, guess, default_value
        );
    }
}

} // namespace gsplat
