#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "macros.h" // for GSPLAT_HOST_DEVICE
#include "types.h"

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
    size_t MAX_ITERATIONS,
    typename Func> // Func(float x) -> pair<float, float> = {residual, dfdx}
inline GSPLAT_HOST_DEVICE auto solver_newton_1d(
    const Func &f, const float &x0, const float epsilon = 1e-6f
) -> NewtonSolverResult<1> {
    auto x = x0;
    auto converged = false;

#pragma unroll
    for (size_t i = 0; i < MAX_ITERATIONS; ++i) {
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
    size_t MAX_ITERATIONS,
    typename Func> // Func(xy) -> pair<residual[2], jacobian[2][2]>
inline GSPLAT_HOST_DEVICE auto solver_newton_2d(
    const Func &f, const glm::fvec2 &x0, float epsilon = 1e-6f
) -> NewtonSolverResult<2> {
    auto x = x0;
    auto converged = false;

#pragma unroll
    for (size_t i = 0; i < MAX_ITERATIONS; ++i) {
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

template <size_t DIM, size_t MAX_ITERATIONS, typename Func>
inline GSPLAT_HOST_DEVICE auto solver_newton(
    const Func &f,
    typename NewtonSolverResult<DIM>::dtype x0,
    const float epsilon = 1e-6f
) -> NewtonSolverResult<DIM> {
    static_assert(DIM == 1 || DIM == 2, "Only 1D and 2D supported");

    if constexpr (DIM == 1) {
        return solver_newton_1d<MAX_ITERATIONS>(f, x0, epsilon);
    } else {
        return solver_newton_2d<MAX_ITERATIONS>(f, x0, epsilon);
    }
}

} // namespace gsplat
