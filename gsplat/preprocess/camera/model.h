#pragma once

#include <array>
#include <cmath>
#include <limits>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp> // glm rotate

#include "camera_projection.h"
#include "macros.h" // for GSPLAT_HOST_DEVICE
#include "types.h"  // for MaybeValidRay, MaybeValidPoint2D

namespace gsplat {

inline GSPLAT_HOST_DEVICE auto interpolate_shutter_pose(
    float relative_frame_time,
    ShutterPose const &pose_start,
    ShutterPose const &pose_end
) -> ShutterPose {
    // Interpolate a pose linearly for a relative frame time
    auto const t_rs = (1.f - relative_frame_time) * pose_start.t +
                      relative_frame_time * pose_end.t;
    auto const q_rs = glm::slerp(pose_start.q, pose_end.q, relative_frame_time);
    return ShutterPose{t_rs, q_rs};
}

inline GSPLAT_HOST_DEVICE auto camera_ray_to_world_ray(
    const MaybeValidRay &camera_ray, const ShutterPose &pose
) -> MaybeValidRay {
    if (!camera_ray.valid_flag) {
        return {{0.f, 0.f, 0.f}, {0.f, 0.f, 1.f}, false};
    }
    auto const R_inv = glm::mat3_cast(glm::inverse(pose.q));
    return {R_inv * (camera_ray.o - pose.t), R_inv * camera_ray.d, true};
}

inline GSPLAT_HOST_DEVICE auto world_to_camera(
    glm::fvec3 const &world_vec, const ShutterPose &pose
) -> glm::fvec3 {
    return glm::rotate(pose.q, world_vec) + pose.t;
}

inline GSPLAT_HOST_DEVICE auto image_point_in_image_bounds_margin(
    glm::fvec2 const &image_point,
    std::array<uint32_t, 2> const &resolution,
    float margin_factor
) -> bool {
    const float MARGIN_X = resolution[0] * margin_factor;
    const float MARGIN_Y = resolution[1] * margin_factor;
    bool valid = true;
    valid &= (-MARGIN_X) <= image_point[0] &&
             image_point[0] < (resolution[0] + MARGIN_X);
    valid &= (-MARGIN_Y) <= image_point[1] &&
             image_point[1] < (resolution[1] + MARGIN_Y);
    return valid;
}

template <class CameraProjection>
inline GSPLAT_HOST_DEVICE auto camera_point_to_image_poin_with_checks(
    const CameraProjection &projector,
    const glm::fvec3 &camera_point,
    std::array<uint32_t, 2> resolution,
    float margin_factor,
    float near_plane,
    float far_plane
) -> MaybeValidPoint2D {
    if (camera_point[2] < near_plane || camera_point[2] > far_plane) {
        return {{0.f, 0.f}, false};
    }
    auto [image_point, valid_flag] =
        projector.camera_point_to_image_point(camera_point);
    if (valid_flag) {
        // Check if the image points fall within the image, set points that
        // have too large distortion or fall outside the image sensor to
        // invalid
        auto const in_bound = image_point_in_image_bounds_margin(
            image_point, resolution, margin_factor
        );
        valid_flag = in_bound;
    }
    return {image_point, valid_flag};
}

template <class CameraProjection> struct CameraModel {

    CameraProjection projector;
    std::array<uint32_t, 2> resolution;
    ShutterType shutter_type;
    float near_plane = 0.0f;
    float far_plane = std::numeric_limits<float>::max();

    GSPLAT_HOST_DEVICE CameraModel(
        const CameraProjection &projector,
        const std::array<uint32_t, 2> &resolution,
        ShutterType shutter_type = ShutterType::GLOBAL,
        float near_plane = 0.0f,
        float far_plane = std::numeric_limits<float>::max()
    )
        : projector(projector), resolution(resolution),
          shutter_type(shutter_type), near_plane(near_plane),
          far_plane(far_plane) {}

    inline GSPLAT_HOST_DEVICE auto image_point_to_world_ray(
        glm::fvec2 const &image_point,
        ShutterPose const &pose_start,
        ShutterPose const &pose_end
    ) const -> MaybeValidRay {
        // Unproject ray and transform to world using shutter pose

        auto const camera_ray =
            projector.image_point_to_camera_ray(image_point);
        if (!camera_ray.valid_flag) {
            return {{0.f, 0.f, 0.f}, {0.f, 0.f, 1.f}, false};
        }

        auto const t_rs = _shutter_relative_frame_time(image_point);
        auto const pose_rs =
            interpolate_shutter_pose(t_rs, pose_start, pose_end);
        return camera_ray_to_world_ray(camera_ray, pose_rs);
    }

    template <size_t N_ROLLING_SHUTTER_ITERATIONS = 10>
    inline GSPLAT_HOST_DEVICE auto world_point_to_image_point(
        const glm::fvec3 &world_point,
        const ShutterPose &pose_start,
        const ShutterPose &pose_end,
        float margin_factor
    ) const -> std::pair<MaybeValidPoint2D, ShutterPose> {
        // Perform rolling-shutter-based world point to image point projection /
        // optimization

        // Always perform transformation using start pose
        auto const &[image_point_start, valid_start] =
            camera_point_to_image_poin_with_checks(
                projector,
                world_to_camera(world_point, pose_start),
                resolution,
                margin_factor,
                near_plane,
                far_plane
            );

        if (shutter_type == ShutterType::GLOBAL) {
            // Exit early if we have a global shutter sensor
            return {{image_point_start, valid_start}, pose_start};
        }

        // This selection prefers points at the start-of-frame pose over
        // end-of-frame points
        glm::fvec2 init_image_point;
        if (valid_start) {
            init_image_point = image_point_start;
        } else {
            // Do initial transformations using both start and end poses to
            // determine all candidate points and take union of valid
            // projections as iteration starting points
            auto const &[image_point_end, valid_end] =
                _camera_point_to_image_point(
                    world_to_camera(world_point, pose_end), margin_factor
                );
            if (valid_end) {
                init_image_point = image_point_end;
            } else {
                // No valid projection at start or finish -> mark point as
                // invalid. Still return projection result at end of frame
                return {{image_point_end, false}, pose_end};
            }
        }

        // Compute the new timestamp and project again
        auto image_points_rs_prev = init_image_point;
        ShutterPose pose_rs;
#pragma unroll
        for (auto j = 0; j < N_ROLLING_SHUTTER_ITERATIONS; ++j) {
            pose_rs = interpolate_shutter_pose(
                _shutter_relative_frame_time(image_points_rs_prev),
                pose_start,
                pose_end
            );
            auto const [image_point_rs, valid_rs] =
                _camera_point_to_image_point(
                    world_to_camera(world_point, pose_rs), margin_factor
                );
            image_points_rs_prev = image_point_rs;
        }

        return {{image_points_rs_prev, true}, pose_rs};
    }

    inline GSPLAT_HOST_DEVICE auto _camera_point_to_image_point(
        const glm::fvec3 &camera_point, float margin_factor
    ) const -> MaybeValidPoint2D {
        if (camera_point[2] < near_plane || camera_point[2] > far_plane) {
            return {{0.f, 0.f}, false};
        }
        auto [image_point, valid_flag] =
            projector.camera_point_to_image_point(camera_point);
        if (valid_flag) {
            // Check if the image points fall within the image, set points that
            // have too large distortion or fall outside the image sensor to
            // invalid
            auto const in_bound = image_point_in_image_bounds_margin(
                image_point, resolution, margin_factor
            );
            valid_flag = in_bound;
        }
        return {image_point, valid_flag};
    }

    // Function to compute the relative frame time for a given image point based
    // on the shutter type
    inline GSPLAT_HOST_DEVICE auto
    _shutter_relative_frame_time(const glm::fvec2 image_point) const -> float {
        auto t = 0.f;
        switch (shutter_type) {
        case ShutterType::ROLLING_TOP_TO_BOTTOM:
            t = std::floor(image_point[1]) / (resolution[1] - 1);
            break;
        case ShutterType::ROLLING_LEFT_TO_RIGHT:
            t = std::floor(image_point[0]) / (resolution[0] - 1);
            break;
        case ShutterType::ROLLING_BOTTOM_TO_TOP:
            t = (resolution[1] - std::ceil(image_point[1])) /
                (resolution[1] - 1);
            break;
        case ShutterType::ROLLING_RIGHT_TO_LEFT:
            t = (resolution[0] - std::ceil(image_point[0])) /
                (resolution[0] - 1);
            break;
        }
        return t;
    }
};

using OpencvPinholeCameraModel = CameraModel<OpencvPinholeProjection>;
using OpencvFisheyeCameraModel = CameraModel<OpencvFisheyeProjection>;
using OrthogonalCameraModel = CameraModel<OrthogonalProjection>;

} // namespace gsplat