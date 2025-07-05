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

/// \brief Result structure for point_world_to_image function
/// \tparam RotationType Type of rotation representation (fmat3 or fquat)
template <typename RotationType> struct PointWorldToImageResult {
    fvec2 image_point;       ///< Projected 2D point in image space
    fvec3 camera_point;      ///< 3D point in camera space
    RotationType pose_r;     ///< Camera rotation at exposure time
    fvec3 pose_t;            ///< Camera translation at exposure time
    bool valid_flag = false; ///< Flag indicating if projection was successful
};

/// \brief Project a world point to an image point using rolling shutter
/// \tparam N_ITER Number of iterations for convergence
/// \tparam RotationType Type of rotation representation (fmat3 or fquat)
/// \tparam Func Type of projection function
/// \param project_fn Function to project a camera point to an image point
/// \param resolution Image resolution (width, height)
/// \param world_point 3D point in world space
/// \param pose_r_start Camera rotation at start of frame
/// \param pose_t_start Camera translation at start of frame
/// \param pose_r_end Camera rotation at end of frame
/// \param pose_t_end Camera translation at end of frame
/// \param shutter_type Type of shutter being used
/// \return PointWorldToImageResult containing the projected results
template <size_t N_ITER = 10, typename RotationType, typename Func>
TREND_HOST_DEVICE inline auto point_world_to_image(
    Func project_fn, // Function to project a camera point to an image point
    const std::array<uint32_t, 2> &resolution,
    const fvec3 &world_point,
    const RotationType &pose_r_start,
    const fvec3 &pose_t_start,
    const RotationType &pose_r_end,
    const fvec3 &pose_t_end,
    const Type &shutter_type
) -> PointWorldToImageResult<RotationType> {
    // Compile-time constraint: ensure Func has the correct signature
    static_assert(
        std::is_invocable_v<Func, fvec3>, "Func must be callable with a fvec3 argument"
    );
    static_assert(
        std::is_same_v<std::invoke_result_t<Func, fvec3>, std::pair<fvec2, bool>>,
        "Func must return std::pair<fvec2, bool> representing {image_point, valid_flag}"
    );

    static_assert(
        std::is_same_v<RotationType, fmat3> || std::is_same_v<RotationType, fquat>,
        "RotationType must be either fmat3 or fquat"
    );

    // Always perform transformation using start pose
    auto const camera_point_start =
        tinyrend::se3::transform_point(pose_r_start, pose_t_start, world_point);
    auto const &[image_point_start, valid_flag_start] = project_fn(camera_point_start);
    if (shutter_type == Type::GLOBAL) {
        if (!valid_flag_start) {
            return PointWorldToImageResult<RotationType>{};
        }
        return PointWorldToImageResult<RotationType>{
            image_point_start, camera_point_start, pose_r_start, pose_t_start, true
        };
    }

    // Initialize the image point using the start or end pose
    fvec2 init_image_point;
    if (valid_flag_start) {
        init_image_point = image_point_start;
    } else {
        auto const camera_point_end =
            tinyrend::se3::transform_point(pose_r_end, pose_t_end, world_point);
        auto const &[image_point_end, valid_flag_end] = project_fn(camera_point_end);
        if (valid_flag_end) {
            init_image_point = image_point_end;
        } else {
            return PointWorldToImageResult<RotationType>{};
        }
    }

    // Iterate to converge to the correct image point
    auto image_point_rs = init_image_point;
    fvec3 camera_point_rs;
    bool valid_flag_rs;
    RotationType pose_r_rs;
    fvec3 pose_t_rs;
#pragma unroll
    for (auto j = 0; j < N_ITER; ++j) {
        auto const t = relative_frame_time(image_point_rs, resolution, shutter_type);
        std::tie(pose_r_rs, pose_t_rs) = tinyrend::se3::interpolate(
            t, pose_r_start, pose_t_start, pose_r_end, pose_t_end
        );
        camera_point_rs =
            tinyrend::se3::transform_point(pose_r_rs, pose_t_rs, world_point);
        std::tie(image_point_rs, valid_flag_rs) = project_fn(camera_point_rs);
        if (!valid_flag_rs) {
            return PointWorldToImageResult<RotationType>{};
        }
        // TODO: add early exit and convergence check
    }
    return PointWorldToImageResult<RotationType>{
        image_point_rs, camera_point_rs, pose_r_rs, pose_t_rs, true
    };
}

} // namespace tinyrend::camera::impl::shutter