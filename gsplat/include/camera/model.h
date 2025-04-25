#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <tuple>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp> // glm rotate

#include "utils/macros.h" // for GSPLAT_HOST_DEVICE
#include "utils/math.h"
#include "utils/types.h"  // for Maybe

namespace gsplat {

// Check if the image points fall within the image with a margin
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

// Interpolate a pose linearly for a relative frame time
template <class CameraPose>
inline GSPLAT_HOST_DEVICE auto interpolate_shutter_pose(
    float relative_frame_time,
    CameraPose const &pose_start,
    CameraPose const &pose_end
) -> CameraPose {
    if constexpr (std::is_same_v<CameraPose, SE3Quat>) {
        auto const t_rs = (1.f - relative_frame_time) * pose_start.t +
                          relative_frame_time * pose_end.t;
        auto const q_rs = glm::slerp(pose_start.q, pose_end.q, relative_frame_time);
        return SE3Quat{t_rs, q_rs};
    } else if constexpr (std::is_same_v<CameraPose, SE3Mat>) {
        auto const t_rs = (1.f - relative_frame_time) * pose_start.t +
                          relative_frame_time * pose_end.t;
        auto const q_rs = glm::slerp(
            glm::quat_cast(pose_start.R), glm::quat_cast(pose_end.R), relative_frame_time);
        auto const R_rs = glm::mat3_cast(q_rs);
        return SE3Mat{t_rs, R_rs};
    } else {
        static_assert(
            std::is_same_v<CameraPose, SE3Quat> || std::is_same_v<CameraPose, SE3Mat>,
            "interpolate_shutter_pose<CameraPose>: unsupported CameraPose type"
        );
    }
}

// Transform a ray from camera space to world space
template <class CameraPose>
inline GSPLAT_HOST_DEVICE auto camera_ray_to_world_ray(
    const glm::fvec3 &ray_o, const glm::fvec3 &ray_d, const CameraPose &pose
) -> std::tuple<glm::fvec3, glm::fvec3> {
    if constexpr (std::is_same_v<CameraPose, SE3Quat>) {
        auto const R_inv = glm::mat3_cast(glm::inverse(pose.q));
        return {R_inv * (ray_o - pose.t), R_inv * ray_d};
    } else if constexpr (std::is_same_v<CameraPose, SE3Mat>) {
        auto const R_inv = glm::transpose(pose.R);
        return {R_inv * (ray_o - pose.t), R_inv * ray_d};
    } else {
        static_assert(
            std::is_same_v<CameraPose, SE3Quat> || std::is_same_v<CameraPose, SE3Mat>,
            "camera_ray_to_world_ray<CameraPose>: unsupported CameraPose type"
        );
    }
}

// Transform a ray from world space to camera space
template <class CameraPose>
inline GSPLAT_HOST_DEVICE auto world_ray_to_camera_ray(
    const glm::fvec3 &ray_o, const glm::fvec3 &ray_d, const CameraPose &pose
) -> std::tuple<glm::fvec3, glm::fvec3> {
    if constexpr (std::is_same_v<CameraPose, SE3Quat>) {
        auto const R = glm::mat3_cast(pose.q);
        return {R * ray_o + pose.t, R * ray_d};
    } else if constexpr (std::is_same_v<CameraPose, SE3Mat>) {
        return {pose.R * ray_o + pose.t, pose.R * ray_d};
    } else {
        static_assert(
            std::is_same_v<CameraPose, SE3Quat> || std::is_same_v<CameraPose, SE3Mat>,
            "world_ray_to_camera_ray<CameraPose>: unsupported CameraPose type"
        );
    }
}

// Transform a point from world space to camera space
template <class CameraPose>
inline GSPLAT_HOST_DEVICE auto world_point_to_camera_point(
    const glm::fvec3 &world_point, const CameraPose &pose
) -> glm::fvec3 {
    if constexpr (std::is_same_v<CameraPose, SE3Quat>) {
        return glm::rotate(pose.q, world_point) + pose.t;
    } else if constexpr (std::is_same_v<CameraPose, SE3Mat>) {
        return pose.R * world_point + pose.t;
    } else {
        static_assert(
            std::is_same_v<CameraPose, SE3Quat> || std::is_same_v<CameraPose, SE3Mat>,
            "world_to_camera<CameraPose>: unsupported CameraPose type"
        );
    }
}

// Transform a point from camera space to world space
template <class CameraPose>
inline GSPLAT_HOST_DEVICE auto camera_point_to_world_point(
    const glm::fvec3 &camera_point, const CameraPose &pose
) -> glm::fvec3 {
    if constexpr (std::is_same_v<CameraPose, SE3Quat>) {
        return glm::rotate(glm::inverse(pose.q), camera_point - pose.t);
    } else if constexpr (std::is_same_v<CameraPose, SE3Mat>) {
        return pose.R * (camera_point - pose.t);
    } else {
        static_assert(
            std::is_same_v<CameraPose, SE3Quat> || std::is_same_v<CameraPose, SE3Mat>,
            "camera_to_world<CameraPose>: unsupported CameraPose type"
        );
    }
}

// Transform a covariance matrix from world space to camera space
template <class CameraPose>
inline GSPLAT_HOST_DEVICE auto world_covar_to_camera_covar(
    const glm::fmat3 &world_covar, const CameraPose &pose
) -> glm::fmat3 {
    if constexpr (std::is_same_v<CameraPose, SE3Quat>) {
        auto const R = glm::mat3_cast(pose.q);
        return R * world_covar * glm::transpose(R);
    } else if constexpr (std::is_same_v<CameraPose, SE3Mat>) {
        return pose.R * world_covar * glm::transpose(pose.R);
    } else {
        static_assert(
            std::is_same_v<CameraPose, SE3Quat> || std::is_same_v<CameraPose, SE3Mat>,
            "world_covar_to_camera_covar<CameraPose>: unsupported CameraPose type"
        );
    }
}

template <class CameraProjection, class CameraPose> 
struct CameraModel {

    std::array<uint32_t, 2> resolution;
    // intrinsic
    CameraProjection projector;
    // extrinsic
    CameraPose pose_start;
    CameraPose pose_end; // for rolling shutter only

    // for early return if the point is outside of the view frustum
    const float margin_factor = 0.15f;
    const float near_plane = std::numeric_limits<float>::min();
    const float far_plane = std::numeric_limits<float>::max();
    
    ShutterType shutter_type = ShutterType::GLOBAL;

    GSPLAT_HOST_DEVICE CameraModel(
        const std::array<uint32_t, 2> &resolution,
        const CameraProjection &projector,
        const CameraPose &pose_start,
        const float margin_factor = 0.15f,
        const float near_plane = std::numeric_limits<float>::min(),
        const float far_plane = std::numeric_limits<float>::max()
    )
        : resolution(resolution),
          projector(projector),
          pose_start(pose_start),
          margin_factor(margin_factor),
            near_plane(near_plane),
            far_plane(far_plane) {}

    GSPLAT_HOST_DEVICE CameraModel(
        const std::array<uint32_t, 2> &resolution,
        const CameraProjection &projector,
        const CameraPose &pose_start,
        const CameraPose &pose_end,
        ShutterType shutter_type,
        const float margin_factor = 0.15f,
        const float near_plane = std::numeric_limits<float>::min(),
        const float far_plane = std::numeric_limits<float>::max()
    )
        : resolution(resolution),
          projector(projector),
          pose_start(pose_start),
          pose_end(pose_end),
          shutter_type(shutter_type),
            margin_factor(margin_factor),
              near_plane(near_plane),
              far_plane(far_plane) {}

    inline GSPLAT_HOST_DEVICE auto image_point_to_world_ray(
        glm::fvec2 const &image_point
    ) -> std::tuple<glm::fvec3, glm::fvec3, bool> {
        auto const &[camera_ray_o, camera_ray_d, valid_flag] =
            projector.image_point_to_camera_ray(image_point);
        if (!valid_flag) {
            return {glm::fvec3{}, glm::fvec3{}, false};
        }
        auto const pose = shutter_type == ShutterType::GLOBAL
                              ? pose_start
                              : interpolate_shutter_pose(
                                    shutter_relative_frame_time(image_point),
                                    pose_start,
                                    pose_end);
        auto const &[world_ray_o, world_ray_d] = 
            camera_ray_to_world_ray(camera_ray_o, camera_ray_d, pose);
        return {world_ray_o, world_ray_d, true};
    }

    inline GSPLAT_HOST_DEVICE auto world_point_to_image_point(
        const glm::fvec3 &world_point
    ) -> std::tuple<glm::fvec2, float, bool> {
        auto const &[camera_point, image_point, valid_flag, pose] =
            _world_point_to_image_point(world_point);
        if (!valid_flag) {
            return {glm::fvec2{}, float{}, false};
        }
        return {image_point, camera_point.z, true};
    }

    inline GSPLAT_HOST_DEVICE auto world_gaussian_to_image_gaussian(
        const glm::fvec3 &world_point, const glm::fmat3 &world_covar
    ) -> std::tuple<glm::fvec2, glm::fmat2, float, bool> {
        auto const &[camera_point, image_point, point_valid_flag, pose] =
            _world_point_to_image_point(world_point);
        if (!point_valid_flag) {
            return {glm::fvec2{}, glm::fmat2{}, float{}, false};
        }
        auto const &[image_covar, covar_valid_flag] = 
            _world_covar_to_image_covar(camera_point, world_covar, pose);
        if (!covar_valid_flag) {
            return {glm::fvec2{}, glm::fmat2{}, float{}, false};
        }
        return {image_point, image_covar, camera_point.z, true};
    }
    
    inline GSPLAT_HOST_DEVICE auto _world_covar_to_image_covar(
        const glm::fvec3 &camera_point, const glm::fmat3 &world_covar, 
        const CameraPose &pose
    ) -> std::tuple<glm::fmat2, bool> {
        auto const camera_covar = world_covar_to_camera_covar(world_covar, pose);
        auto const &[J, valid_flag] = 
            projector.camera_point_to_image_point_jacobian(camera_point);
        if (!valid_flag) {
            return {glm::fmat2{}, false};
        }
        auto const image_covar = J * camera_covar * glm::transpose(J);
        return {image_covar, true};
    }

    template <size_t N_ROLLING_SHUTTER_ITERATIONS = 10>
    inline GSPLAT_HOST_DEVICE auto _world_point_to_image_point(
        const glm::fvec3 &world_point
    ) -> std::tuple<glm::fvec3, glm::fvec2, bool, CameraPose> {
        // Perform rolling-shutter-based world point to image point projection /
        // optimization

        // Always perform transformation using start pose
        auto const &[camera_point_start, image_point_start, valid_start] = 
            _world_to_camera_and_image_and_checks(world_point, pose_start);
        if (shutter_type == ShutterType::GLOBAL) {
            // Exit early if we have a global shutter sensor
            return {camera_point_start, image_point_start, valid_start, pose_start};
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
            auto const &[camera_point_end, image_point_end, valid_end] = 
                _world_to_camera_and_image_and_checks(world_point, pose_end);
            if (valid_end) {
                init_image_point = image_point_end;
            } else {
                // No valid projection at start or finish -> mark point as
                // invalid. Still return projection result at end of frame
                return {camera_point_end, image_point_end, false, pose_end};
            }
        }

        // Compute the new timestamp and project again
        auto image_point_rs = init_image_point;
        glm::fvec3 camera_point_rs;
        CameraPose pose_rs;
#pragma unroll
        for (auto j = 0; j < N_ROLLING_SHUTTER_ITERATIONS; ++j) {
            pose_rs = interpolate_shutter_pose(
                shutter_relative_frame_time(image_point_rs),
                pose_start,
                pose_end
            );
            auto const &[camera_point_rs_, image_point_rs_, valid_rs] = 
                _world_to_camera_and_image_and_checks(world_point, pose_rs);
            image_point_rs = image_point_rs_;
            camera_point_rs = camera_point_rs_;
            if (!valid_rs) {
                return {camera_point_rs, image_point_rs, false, pose_rs};
            }
            // TODO: add early exit if the image point is not changing
        }

        return {camera_point_rs, image_point_rs, true, pose_rs};
    }

private:

    // Function to compute the relative frame time for a given image point based
    // on the shutter type
    inline GSPLAT_HOST_DEVICE auto
    shutter_relative_frame_time(const glm::fvec2 image_point) -> float {
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

    inline GSPLAT_HOST_DEVICE auto _world_to_camera_and_image_and_checks(
        const glm::fvec3 &world_point, const CameraPose &pose
    ) -> std::tuple<glm::fvec3, glm::fvec2, bool> {
        auto const camera_point = 
            world_point_to_camera_point(world_point, pose);
        if (camera_point.z < near_plane || camera_point.z > far_plane) {
            return {glm::fvec3{}, glm::fvec2{}, false};
        }

        auto const &[image_point, valid_flag] =
            projector.camera_point_to_image_point(camera_point);
        if (!valid_flag) {
            return {glm::fvec3{}, glm::fvec2{}, false};
        }

        auto const in_fov = image_point_in_image_bounds_margin(
            image_point, resolution, margin_factor
        );
        if (!in_fov) {
            return {glm::fvec3{}, glm::fvec2{}, false};
        }

        return {camera_point, image_point, true};
    }
};



} // namespace gsplat