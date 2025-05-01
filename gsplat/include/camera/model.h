#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <tuple>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp> // glm rotate

#include "core/macros.h" // for GSPLAT_HOST_DEVICE
#include "core/math.h"
#include "core/tensor.h"  // for Maybe
#include "core/types.h"   // for ShutterType
#include "utils/solver.h" // for solver_newton

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
template <class RotationType>
inline GSPLAT_HOST_DEVICE auto interpolate_shutter_pose(
    float relative_frame_time,
    const RotationType &pose_r_start,
    const glm::fvec3 &pose_t_start,
    const RotationType &pose_r_end,
    const glm::fvec3 &pose_t_end
) -> std::pair<RotationType, glm::fvec3> {
    auto const pose_t_rs = (1.f - relative_frame_time) * pose_t_start +
                           relative_frame_time * pose_t_end;
    if constexpr (std::is_same_v<RotationType, glm::fquat>) {
        auto const pose_r_rs =
            glm::slerp(pose_r_start, pose_r_end, relative_frame_time);
        return {pose_r_rs, pose_t_rs};
    } else if constexpr (std::is_same_v<RotationType, glm::fmat3>) {
        auto const pose_r_rs = glm::slerp(
            glm::quat_cast(pose_r_start),
            glm::quat_cast(pose_r_end),
            relative_frame_time
        );
        return {glm::mat3_cast(pose_r_rs), pose_t_rs};
    } else {
        static_assert(
            std::is_same_v<RotationType, glm::fquat> ||
                std::is_same_v<RotationType, glm::fmat3>,
            "interpolate_shutter_pose<RotationType>: unsupported RotationType "
            "type"
        );
    }
}

// Transform a ray from camera space to world space
template <class RotationType>
inline GSPLAT_HOST_DEVICE auto camera_ray_to_world_ray(
    const glm::fvec3 &ray_o,
    const glm::fvec3 &ray_d,
    const RotationType &pose_r,
    const glm::fvec3 &pose_t
) -> std::tuple<glm::fvec3, glm::fvec3> {
    if constexpr (std::is_same_v<RotationType, glm::fquat>) {
        auto const R_inv = glm::mat3_cast(glm::inverse(pose_r));
        return {R_inv * (ray_o - pose_t), R_inv * ray_d};
    } else if constexpr (std::is_same_v<RotationType, glm::fmat3>) {
        auto const R_inv = glm::transpose(pose_r);
        return {R_inv * (ray_o - pose_t), R_inv * ray_d};
    } else {
        static_assert(
            std::is_same_v<RotationType, glm::fquat> ||
                std::is_same_v<RotationType, glm::fmat3>,
            "camera_ray_to_world_ray<RotationType>: unsupported RotationType "
            "type"
        );
    }
}

// Transform a ray from world space to camera space
template <class RotationType>
inline GSPLAT_HOST_DEVICE auto world_ray_to_camera_ray(
    const glm::fvec3 &ray_o,
    const glm::fvec3 &ray_d,
    const RotationType &pose_r,
    const glm::fvec3 &pose_t
) -> std::tuple<glm::fvec3, glm::fvec3> {
    if constexpr (std::is_same_v<RotationType, glm::fquat>) {
        auto const R = glm::mat3_cast(pose_r);
        return {R * ray_o + pose_t, R * ray_d};
    } else if constexpr (std::is_same_v<RotationType, glm::fmat3>) {
        auto const R = pose_r;
        return {R * ray_o + pose_t, R * ray_d};
    } else {
        static_assert(
            std::is_same_v<RotationType, glm::fquat> ||
                std::is_same_v<RotationType, glm::fmat3>,
            "world_ray_to_camera_ray<RotationType>: unsupported RotationType "
            "type"
        );
    }
}

// Transform a point from world space to camera space
template <class RotationType>
inline GSPLAT_HOST_DEVICE auto world_point_to_camera_point(
    const glm::fvec3 &world_point,
    const RotationType &pose_r,
    const glm::fvec3 &pose_t
) -> glm::fvec3 {
    if constexpr (std::is_same_v<RotationType, glm::fquat>) {
        return glm::rotate(pose_r, world_point) + pose_t;
    } else if constexpr (std::is_same_v<RotationType, glm::fmat3>) {
        return pose_r * world_point + pose_t;
    } else {
        static_assert(
            std::is_same_v<RotationType, glm::fquat> ||
                std::is_same_v<RotationType, glm::fmat3>,
            "world_to_camera<RotationType>: unsupported RotationType type"
        );
    }
}

// Transform a point from camera space to world space
template <class RotationType>
inline GSPLAT_HOST_DEVICE auto camera_point_to_world_point(
    const glm::fvec3 &camera_point,
    const RotationType &pose_r,
    const glm::fvec3 &pose_t
) -> glm::fvec3 {
    if constexpr (std::is_same_v<RotationType, glm::fquat>) {
        return glm::rotate(glm::inverse(pose_r), camera_point) + pose_t;
    } else if constexpr (std::is_same_v<RotationType, glm::fmat3>) {
        return pose_r * (camera_point - pose_t);
    } else {
        static_assert(
            std::is_same_v<RotationType, glm::fquat> ||
                std::is_same_v<RotationType, glm::fmat3>,
            "camera_to_world<RotationType>: unsupported RotationType type"
        );
    }
}

// Transform a covariance matrix from world space to camera space
template <class RotationType>
inline GSPLAT_HOST_DEVICE auto world_covar_to_camera_covar(
    const glm::fmat3 &world_covar,
    const RotationType &pose_r,
    const glm::fvec3 &pose_t
) -> glm::fmat3 {
    if constexpr (std::is_same_v<RotationType, glm::fquat>) {
        auto const R = glm::mat3_cast(pose_r);
        return R * world_covar * glm::transpose(R);
    } else if constexpr (std::is_same_v<RotationType, glm::fmat3>) {
        return pose_r * world_covar * glm::transpose(pose_r);
    } else {
        static_assert(
            std::is_same_v<RotationType, glm::fquat> ||
                std::is_same_v<RotationType, glm::fmat3>,
            "world_covar_to_camera_covar<RotationType>: unsupported "
            "RotationType "
            "type"
        );
    }
}

template <class CameraProjection, class RotationType> struct CameraModel {
    // extrinsic
    RotationType pose_r_start;
    glm::fvec3 pose_t_start;
    RotationType pose_r_end;
    glm::fvec3 pose_t_end; // for rolling shutter only
    // intrinsic
    ShutterType shutter_type = ShutterType::GLOBAL;
    CameraProjection projector;
    std::array<uint32_t, 2> resolution;

    // for early return if the point is outside of the view frustum
    const float margin_factor = 0.15f;
    const float near_plane = std::numeric_limits<float>::min();
    const float far_plane = std::numeric_limits<float>::max();

    // Default constructor
    inline GSPLAT_HOST_DEVICE CameraModel() {}

    // Constructor for Global Shutter
    inline GSPLAT_HOST_DEVICE CameraModel(
        std::array<uint32_t, 2> &resolution,
        CameraProjection &projector,
        const RotationType &pose_r_start,
        const glm::fvec3 &pose_t_start,
        const float margin_factor = 0.15f,
        const float near_plane = std::numeric_limits<float>::min(),
        const float far_plane = std::numeric_limits<float>::max()
    )
        : resolution(resolution), margin_factor(margin_factor),
          near_plane(near_plane), far_plane(far_plane) {
        this->projector = projector;
        this->pose_r_start = pose_r_start;
        this->pose_t_start = pose_t_start;
    }

    // Constructor for Rolling Shutter
    inline GSPLAT_HOST_DEVICE CameraModel(
        std::array<uint32_t, 2> &resolution,
        CameraProjection &projector,
        const RotationType &pose_r_start,
        const glm::fvec3 &pose_t_start,
        const RotationType &pose_r_end,
        const glm::fvec3 &pose_t_end,
        const ShutterType &shutter_type,
        const float margin_factor = 0.15f,
        const float near_plane = std::numeric_limits<float>::min(),
        const float far_plane = std::numeric_limits<float>::max()
    )
        : resolution(resolution), margin_factor(margin_factor),
          near_plane(near_plane), far_plane(far_plane) {
        this->projector = projector;
        this->pose_r_start = pose_r_start;
        this->pose_t_start = pose_t_start;
        this->pose_r_end = pose_r_end;
        this->pose_t_end = pose_t_end;
        this->shutter_type = shutter_type;
    }

    inline GSPLAT_HOST_DEVICE auto
    image_point_to_world_ray(glm::fvec2 const &image_point
    ) -> std::tuple<glm::fvec3, glm::fvec3, bool> {
        auto const &[camera_ray_o, camera_ray_d, valid_flag] =
            projector.image_point_to_camera_ray(image_point);
        if (!valid_flag) {
            return {glm::fvec3{}, glm::fvec3{}, false};
        }
        auto const &[pose_r, pose_t] =
            shutter_type == ShutterType::GLOBAL
                ? std::make_pair(pose_r_start, pose_t_start)
                : interpolate_shutter_pose(
                      shutter_relative_frame_time(image_point),
                      pose_r_start,
                      pose_t_start,
                      pose_r_end,
                      pose_t_end
                  );
        auto const &[world_ray_o, world_ray_d] =
            camera_ray_to_world_ray(camera_ray_o, camera_ray_d, pose_r, pose_t);
        return {world_ray_o, world_ray_d, true};
    }

    inline GSPLAT_HOST_DEVICE auto
    world_point_to_image_point(const glm::fvec3 &world_point
    ) -> std::tuple<glm::fvec2, float, bool> {
        auto result = _world_point_to_image_point(world_point);
        auto const camera_point = std::get<0>(result);
        auto const image_point = std::get<1>(result);
        auto const valid_flag = std::get<2>(result);
        if (!valid_flag) {
            return {glm::fvec2{}, float{}, false};
        }
        return {image_point, camera_point.z, true};
    }

    inline GSPLAT_HOST_DEVICE auto world_gaussian_to_image_gaussian(
        const glm::fvec3 &world_point, const glm::fmat3 &world_covar
    ) -> std::tuple<glm::fvec2, glm::fmat2, float, bool> {
        auto result = _world_point_to_image_point(world_point);
        auto const camera_point = std::get<0>(result);
        auto const image_point = std::get<1>(result);
        auto const point_valid_flag = std::get<2>(result);
        auto pose = std::get<3>(result);
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
        const glm::fvec3 &camera_point,
        const glm::fmat3 &world_covar,
        const RotationType &pose_r,
        const glm::fvec3 &pose_t
    ) -> std::tuple<glm::fmat2, bool> {
        auto const camera_covar =
            world_covar_to_camera_covar(world_covar, pose_r, pose_t);
        auto const &[J, valid_flag] =
            projector.camera_point_to_image_point_jacobian(camera_point);
        if (!valid_flag) {
            return {glm::fmat2{}, false};
        }
        auto const image_covar = J * camera_covar * glm::transpose(J);
        return {image_covar, true};
    }

    template <size_t N_ROLLING_SHUTTER_ITERATIONS = 10>
    inline GSPLAT_HOST_DEVICE auto
    _world_point_to_image_point(const glm::fvec3 &world_point
    ) -> std::tuple<glm::fvec3, glm::fvec2, bool, RotationType, glm::fvec3> {
        // Perform rolling-shutter-based world point to image point projection /
        // optimization
        // Always perform transformation using start pose
        auto const &[camera_point_start, image_point_start, valid_start] =
            _world_to_camera_and_image_and_checks(
                world_point, pose_r_start, pose_t_start
            );
        if (shutter_type == ShutterType::GLOBAL) {
            // Exit early if we have a global shutter sensor
            return {
                camera_point_start,
                image_point_start,
                valid_start,
                pose_r_start,
                pose_t_start
            };
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
                _world_to_camera_and_image_and_checks(
                    world_point, pose_r_end, pose_t_end
                );
            if (valid_end) {
                init_image_point = image_point_end;
            } else {
                // No valid projection at start or finish -> mark point as
                // invalid. Still return projection result at end of frame
                return {
                    camera_point_end,
                    image_point_end,
                    false,
                    pose_r_end,
                    pose_t_end
                };
            }
        }

        // Compute the new timestamp and project again
        auto image_point_rs = init_image_point;
        glm::fvec3 camera_point_rs;
        RotationType pose_r_rs;
        glm::fvec3 pose_t_rs;
#pragma unroll
        for (auto j = 0; j < N_ROLLING_SHUTTER_ITERATIONS; ++j) {
            auto const &[pose_r_rs, pose_t_rs] = interpolate_shutter_pose(
                shutter_relative_frame_time(image_point_rs),
                pose_r_start,
                pose_t_start,
                pose_r_end,
                pose_t_end
            );
            auto const &[camera_point_rs_, image_point_rs_, valid_rs] =
                _world_to_camera_and_image_and_checks(
                    world_point, pose_r_rs, pose_t_rs
                );
            image_point_rs = image_point_rs_;
            camera_point_rs = camera_point_rs_;
            if (!valid_rs) {
                return {
                    camera_point_rs, image_point_rs, false, pose_r_rs, pose_t_rs
                };
            }
            // TODO: add early exit if the image point is not changing
        }

        return {camera_point_rs, image_point_rs, true, pose_r_rs, pose_t_rs};
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
        const glm::fvec3 &world_point,
        const RotationType &pose_r,
        const glm::fvec3 &pose_t
    ) -> std::tuple<glm::fvec3, glm::fvec2, bool> {
        auto const camera_point =
            world_point_to_camera_point(world_point, pose_r, pose_t);
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