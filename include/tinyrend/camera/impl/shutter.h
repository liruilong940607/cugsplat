#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

#include "tinyrend/common/macros.h" // for TREND_HOST_DEVICE
#include "tinyrend/common/mat.h"
#include "tinyrend/common/math.h"
#include "tinyrend/common/vec.h"
#include "tinyrend/util/se3.h"

namespace tinyrend::camera::impl::shutter {

/// \brief Enumeration of shutter types
/// \details Defines different types of camera shutters
enum class Type {
    /// All pixels are exposed simultaneously (No rolling shutter)
    GLOBAL,
    /// Exposure starts from top and moves to bottom
    ROLLING_TOP_TO_BOTTOM,
    /// Exposure starts from left and moves to right
    ROLLING_LEFT_TO_RIGHT,
    /// Exposure starts from bottom and moves to top
    ROLLING_BOTTOM_TO_TOP,
    /// Exposure starts from right and moves to left
    ROLLING_RIGHT_TO_LEFT,
};

/// \brief Compute the relative frame time for a given image point
/// \param image_point 2D point in image space
/// \param resolution Image resolution (width, height)
/// \param shutter_type Type of shutter being used
/// \return Relative time in [0, 1] range where 0 is start of frame and 1 is end of
/// frame
TREND_HOST_DEVICE inline auto relative_frame_time(
    const fvec2 &image_point,
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

} // namespace tinyrend::camera::impl::shutter