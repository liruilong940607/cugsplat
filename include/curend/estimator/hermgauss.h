// Multidimensional Gauss-Hermite quadrature points and weights.
//
// Reference:
//   numpy.polynomial.hermite.hermgauss

#pragma once

#include <array>
#include <glm/glm.hpp>

#include "curend/core/macros.h" // for GSPLAT_HOST_DEVICE

namespace curend::ghq {

// Constants for different orders
constexpr int ORDER3 = 3;
constexpr int ORDER5 = 5;

// 1D Gauss-Hermite quadrature points and weights for order=3
constexpr std::array<float, ORDER3> HERMGAUSS_POINTS_1D_ORDER3 = {
    -1.224744871391589f, // -sqrt(3/2)
    0.0f,
    1.224744871391589f // sqrt(3/2)
};

constexpr std::array<float, ORDER3> HERMGAUSS_WEIGHTS_1D_ORDER3 = {
    0.295408975150919f, // 1/(6*sqrt(pi))
    1.181635900603677f, // 2/(3*sqrt(pi))
    0.295408975150919f  // 1/(6*sqrt(pi))
};

// 1D Gauss-Hermite quadrature points and weights for order=5
constexpr std::array<float, ORDER5> HERMGAUSS_POINTS_1D_ORDER5 = {
    -2.020182870456086f, // -sqrt(5/2 + sqrt(15)/2)
    -0.958572464613818f, // -sqrt(5/2 - sqrt(15)/2)
    0.0f,
    0.958572464613818f, // sqrt(5/2 - sqrt(15)/2)
    2.020182870456086f  // sqrt(5/2 + sqrt(15)/2)
};

constexpr std::array<float, ORDER5> HERMGAUSS_WEIGHTS_1D_ORDER5 = {
    0.019953242059046f, // (30 + sqrt(15))/(60*sqrt(pi))
    0.393619323152241f, // (30 - sqrt(15))/(60*sqrt(pi))
    0.945308720482942f, // 1/(3*sqrt(pi))
    0.393619323152241f, // (30 - sqrt(15))/(60*sqrt(pi))
    0.019953242059046f  // (30 + sqrt(15))/(60*sqrt(pi))
};

// Helper function for compile-time power
template <int B, int E> struct constexpr_pow {
    static constexpr int value = B * constexpr_pow<B, E - 1>::value;
};

template <int B> struct constexpr_pow<B, 0> {
    static constexpr int value = 1;
};

// Recursive implementation for tensor weights
template <int N, int I = 0, int J = 0, int K = 0>
GSPLAT_HOST_DEVICE constexpr auto hermgauss_weights_impl(
    std::array<float, N> const &w1d, std::array<float, constexpr_pow<N, 3>::value> &w
) -> void {
    if constexpr (K < N) {
        w[K * N * N + J * N + I] = w1d[I] * w1d[J] * w1d[K];
        if constexpr (I + 1 < N) {
            hermgauss_weights_impl<N, I + 1, J, K>(w1d, w);
        } else if constexpr (J + 1 < N) {
            hermgauss_weights_impl<N, 0, J + 1, K>(w1d, w);
        } else {
            hermgauss_weights_impl<N, 0, 0, K + 1>(w1d, w);
        }
    }
}

// Recursive implementation for tensor points
template <int N, int I = 0, int J = 0, int K = 0>
GSPLAT_HOST_DEVICE constexpr auto hermgauss_points_impl(
    std::array<float, N> const &x1d,
    std::array<glm::fvec3, constexpr_pow<N, 3>::value> &p
) -> void {
    if constexpr (K < N) {
        p[K * N * N + J * N + I] = glm::fvec3(x1d[I], x1d[J], x1d[K]);
        if constexpr (I + 1 < N) {
            hermgauss_points_impl<N, I + 1, J, K>(x1d, p);
        } else if constexpr (J + 1 < N) {
            hermgauss_points_impl<N, 0, J + 1, K>(x1d, p);
        } else {
            hermgauss_points_impl<N, 0, 0, K + 1>(x1d, p);
        }
    }
}

// Main function to generate tensor weights
template <int N>
GSPLAT_HOST_DEVICE constexpr auto hermgauss_weights(std::array<float, N> const &w1d
) -> std::array<float, constexpr_pow<N, 3>::value> {
    std::array<float, constexpr_pow<N, 3>::value> w{};
    hermgauss_weights_impl<N>(w1d, w);
    return w;
}

// Main function to generate tensor points
template <int N>
GSPLAT_HOST_DEVICE constexpr auto hermgauss_points(std::array<float, N> const &x1d
) -> std::array<glm::fvec3, constexpr_pow<N, 3>::value> {
    std::array<glm::fvec3, constexpr_pow<N, 3>::value> p{};
    hermgauss_points_impl<N>(x1d, p);
    return p;
}

// Compile-time generated 3D weights and points for order 3
GSPLAT_HOST_DEVICE constexpr auto HERMGAUSS_WEIGHTS_3D_ORDER3 =
    hermgauss_weights<ORDER3>(HERMGAUSS_WEIGHTS_1D_ORDER3);
GSPLAT_HOST_DEVICE constexpr auto HERMGAUSS_POINTS_3D_ORDER3 =
    hermgauss_points<ORDER3>(HERMGAUSS_POINTS_1D_ORDER3);

// Compile-time generated 3D weights and points for order 5
GSPLAT_HOST_DEVICE constexpr auto HERMGAUSS_WEIGHTS_3D_ORDER5 =
    hermgauss_weights<ORDER5>(HERMGAUSS_WEIGHTS_1D_ORDER5);
GSPLAT_HOST_DEVICE constexpr auto HERMGAUSS_POINTS_3D_ORDER5 =
    hermgauss_points<ORDER5>(HERMGAUSS_POINTS_1D_ORDER5);

} // namespace curend::ghq