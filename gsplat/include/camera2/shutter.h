#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <tuple>

#include "core/macros.h" // for GSPLAT_HOST_DEVICE
#include "core/math.h"

namespace gsplat::shutter {

enum class Type {
    GLOBAL,
    ROLLING_TOP_TO_BOTTOM,
    ROLLING_LEFT_TO_RIGHT,
    ROLLING_BOTTOM_TO_TOP,
    ROLLING_RIGHT_TO_LEFT,
};

GSPLAT_HOST_DEVICE inline auto relative_frame_time(
    const glm::fvec2 &image_point,
    const std::array<uint32_t, 2> &resolution,
    const Type &shutter_type
) -> float {
    auto t = 0.f;
    switch (shutter_type) {
    case Type::ROLLING_TOP_TO_BOTTOM:
        t = std::floor(image_point[1]) / (resolution[1] - 1);
        break;
    case Type::ROLLING_LEFT_TO_RIGHT:
        t = std::floor(image_point[0]) / (resolution[0] - 1);
        break;
    case Type::ROLLING_BOTTOM_TO_TOP:
        t = (resolution[1] - std::ceil(image_point[1])) / (resolution[1] - 1);
        break;
    case Type::ROLLING_RIGHT_TO_LEFT:
        t = (resolution[0] - std::ceil(image_point[0])) / (resolution[0] - 1);
        break;
    }
    return t;
}

template <size_t N_ITER = 10, typename RotationType>
GSPLAT_HOST_DEVICE inline auto point_world_to_image(
    const glm::fvec3 &world_point,
    const RotationType &pose_r_start,
    const glm::fvec3 &pose_t_start,
    const RotationType &pose_r_end,
    const glm::fvec3 &pose_t_end,
    const Type &shutter_type
) -> glm::fvec2 {
    static_assert(
        std::is_same_v<RotationType, glm::fmat3> ||
            std::is_same_v<RotationType, glm::fquat>,
        "RotationType must be either glm::fmat3 or glm::fquat"
    );

    auto const t = relative_frame_time(image_point, resolution, shutter_type);
}

} // namespace gsplat::shutter