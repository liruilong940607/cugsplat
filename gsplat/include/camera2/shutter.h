#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <tuple>

#include "core/macros.h" // for GSPLAT_HOST_DEVICE
#include "core/math.h"
#include "utils/solver.h" // for solver_newton

namespace gsplat::shutter {

enum class Type {
    GLOBAL,
    ROLLING_TOP_TO_BOTTOM,
    ROLLING_LEFT_TO_RIGHT,
    ROLLING_BOTTOM_TO_TOP,
    ROLLING_RIGHT_TO_LEFT,
};

// // Example
// GSPLAT_HOST_DEVICE auto distortion(
//     float const &theta, std::array<float, 4> const &radial_coeffs
// ) -> float {
//     auto const theta2 = theta * theta;
//     auto const &[k1, k2, k3, k4] = radial_coeffs;
//     return theta * eval_poly_horner<5>({1.f, k1, k2, k3, k4}, theta2);
// }

} // namespace gsplat::shutter