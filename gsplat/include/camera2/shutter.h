#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <tuple>

#include "core/macros.h" // for GSPLAT_HOST_DEVICE
#include "core/math.h"
#include "utils/se3.h"

namespace gsplat::shutter {

enum class Type {
    GLOBAL,
    ROLLING_TOP_TO_BOTTOM,
    ROLLING_LEFT_TO_RIGHT,
    ROLLING_BOTTOM_TO_TOP,
    ROLLING_RIGHT_TO_LEFT,
};

// Compute the relative frame time for a given image point
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

template <typename RotationType> struct PointWorldToImageResult {
    glm::fvec2 image_point;
    glm::fvec3 camera_point;
    RotationType pose_r;
    glm::fvec3 pose_t;
    bool valid_flag = false;
};

// Project a world point to an image point using the start and end poses
// and the shutter type
template <size_t N_ITER = 10, typename RotationType, typename Func>
GSPLAT_HOST_DEVICE inline auto point_world_to_image(
    Func project_fn, // Function to project a camera point to an image point
    const glm::fvec3 &world_point,
    const RotationType &pose_r_start,
    const glm::fvec3 &pose_t_start,
    const RotationType &pose_r_end,
    const glm::fvec3 &pose_t_end,
    const Type &shutter_type
) -> PointWorldToImageResult<RotationType> {
    static_assert(
        std::is_same_v<RotationType, glm::fmat3> ||
            std::is_same_v<RotationType, glm::fquat>,
        "RotationType must be either glm::fmat3 or glm::fquat"
    );

    // Always perform transformation using start pose
    auto const camera_point_start =
        se3::transform_point(pose_r_start, pose_t_start, world_point);
    auto const &[image_point_start, valid_flag_start] =
        project_fn(camera_point_start);
    if (shutter_type == Type::GLOBAL) {
        return PointWorldToImageResult{};
    }

    // Initialize the image point using the start or end pose
    glm::fvec2 init_image_point;
    if (valid_flag_start) {
        init_image_point = image_point_start;
    } else {
        auto const camera_point_end =
            se3::transform_point(pose_r_end, pose_t_end, world_point);
        auto const &[image_point_end, valid_flag_end] =
            project_fn(camera_point_end);
        if (valid_flag_end) {
            init_image_point = image_point_end;
        } else {
            return PointWorldToImageResult{};
        }
    }

    // Iterate to converge to the correct image point
    auto image_point_rs = init_image_point;
    glm::fvec3 camera_point_rs;
    bool valid_flag_rs;
    RotationType pose_r_rs;
    glm::fvec3 pose_t_rs;
#pragma unroll
    for (auto j = 0; j < N_ITER; ++j) {
        auto const t =
            relative_frame_time(image_point, resolution, shutter_type);
        std::tie(pose_r_rs, pose_t_rs) = se3::interpolate(
            t, pose_r_start, pose_t_start, pose_r_end, pose_t_end
        );
        camera_point_rs =
            se3::transform_point(pose_r_rs, pose_t_rs, world_point);
        std::tie(image_point_rs, valid_flag_rs) = project_fn(camera_point_rs);
        if (!valid_flag_rs) {
            return PointWorldToImageResult{};
        }
        // TODO: add early exit and convergence check
    }
    return PointWorldToImageResult{
        image_point_rs, camera_point_rs, pose_r_rs, pose_t_rs, true
    };
}

} // namespace gsplat::shutter