#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

#include "tinyrend/camera/impl/fisheye.h"
#include "tinyrend/camera/impl/orthogonal.h"
#include "tinyrend/camera/impl/pinhole.h"
#include "tinyrend/camera/impl/shutter.h"
#include "tinyrend/common/macros.h" // for TREND_HOST_DEVICE
#include "tinyrend/common/vec.h"
#include "tinyrend/util/se3.h"

namespace tinyrend::camera {

TREND_HOST_DEVICE inline auto image_point_in_image_bounds_margin(
    fvec2 const &image_point,
    std::array<uint32_t, 2> const &resolution,
    float margin_factor
) {
    const float MARGIN_X = resolution[0] * margin_factor;
    const float MARGIN_Y = resolution[1] * margin_factor;
    bool valid = true;
    valid &=
        (-MARGIN_X) <= image_point[0] && image_point[0] < (resolution[0] + MARGIN_X);
    valid &=
        (-MARGIN_Y) <= image_point[1] && image_point[1] < (resolution[1] + MARGIN_Y);
    return valid;
}

struct Ray {
    fvec3 o; // origin
    fvec3 d; // direction
    bool valid_flag;
};

struct ImagePoint {
    fvec2 p; // pixel coordinates
    bool valid_flag;
};

struct ShutterPoses {
    struct Pose {
        fvec3 t; // translation
        fquat r; // rotation
    };

    Pose start;
    Pose end;
};

inline TREND_HOST_DEVICE auto interpolate_shutter_poses(
    float const relative_frame_time, ShutterPoses const &shutter_poses
) -> ShutterPoses::Pose {
    auto const t_start = shutter_poses.start.t;
    auto const t_end = shutter_poses.end.t;
    auto const r_start = shutter_poses.start.r;
    auto const r_end = shutter_poses.end.r;

    auto const t_rs =
        (1.f - relative_frame_time) * t_start + relative_frame_time * t_end;
    auto const r_rs = normalize(slerp(r_start, r_end, relative_frame_time));
    return ShutterPoses::Pose{t_rs, r_rs};
}

template <typename DerivedCameraModel> struct BaseCameraModel {
    // CRTP base class for all camera model types

    struct Parameters {
        std::array<uint32_t, 2> resolution;
        ShutterType shutter_type;
    };

    Parameters parameters;

    inline TREND_HOST_DEVICE auto _shutter_relative_frame_time(fvec2 const &image_point
    ) const -> float {
        auto const derived = static_cast<DerivedCameraModel const *>(this);
        auto const &resolution = derived->parameters.resolution;
        auto const &shutter_type = derived->parameters.shutter_type;
        return ::impl::shutter::relative_frame_time(
            image_point, resolution, shutter_type
        );
    }

    // Unproject image point to world ray (with rolling shutter)
    inline TREND_HOST_DEVICE auto image_point_to_world_ray(
        fvec2 const &image_point, ShutterPoses const &shutter_poses
    ) -> Ray {
        auto const derived = static_cast<DerivedCameraModel const *>(this);
        auto const camera_ray = derived->image_point_to_camera_ray(image_point);

        // If the camera ray in not valid, return a zero ray
        if (!camera_ray.valid_flag) {
            return Ray{fvec3(0.f), fvec3(0.f), false};
        }

        // Interpolate the shutter poses
        auto const pose_rs = interpolate_shutter_poses(
            _shutter_relative_frame_time(image_point), shutter_poses
        );

        // Transform the camera ray to the world frame
        auto const &[world_ray_o, world_ray_d] = tinyrend::se3::transform_ray(
            pose_rs.r, pose_rs.t, camera_ray.o, camera_ray.d
        );
        return Ray{world_ray_o, world_ray_d, true};
    }

    inline TREND_HOST_DEVICE auto _transform_world_to_image(
        fvec3 const &world_point, ShutterPoses::Pose const &pose
    ) const -> ImagePoint {
        auto const derived = static_cast<DerivedCameraModel const *>(this);
        auto const camera_point =
            tinyrend::se3::transform_point(pose.r, pose.t, world_point);
        auto const &[image_point, valid_flag] =
            derived->camera_point_to_image_point(camera_point);
        return ImagePoint{image_point, valid_flag};
    }

    // Project world point to image point (with rolling shutter)
    template <size_t N_ROLLING_SHUTTER_ITERATIONS = 10>
    inline TREND_HOST_DEVICE auto world_point_to_image_point(
        fvec3 const &world_point, ShutterPoses const &shutter_poses
    ) const -> ImagePoint {
        auto const derived = static_cast<DerivedCameraModel const *>(this);

        // Always perform transformation using start pose
        auto const image_point_start =
            _transform_world_to_image(world_point, shutter_poses.start);

        // Exit early if we have a global shutter sensor
        if (derived->parameters.shutter_type == ShutterType::GLOBAL) {
            return image_point_start;
        }

        // This selection prefers points at the start-of-frame pose over
        // end-of-frame points
        auto p_rs = fvec2{};
        if (image_point_start.valid_flag) {
            p_rs = image_point_start.p;
        } else {
            // Do initial transformations using both start and end poses to
            // determine all candidate points and take union of valid projections as
            // iteration starting points
            auto const image_point_end =
                _transform_world_to_image(world_point, shutter_poses.end);
            if (image_point_end.valid_flag) {
                p_rs = image_point_end.p;
            } else {
                // No valid projection at start or finish -> mark point as invalid.
                return ImagePoint{fvec2{}, false};
            }
        }

        // Iterative Rolling Shutter Optimization
#pragma unroll
        for (auto j = 0; j < N_ROLLING_SHUTTER_ITERATIONS; ++j) {
            auto const pose_rs = interpolate_shutter_poses(
                _shutter_relative_frame_time(p_rs), shutter_poses
            );
            auto const image_point_rs = _transform_world_to_image(world_point, pose_rs);
            if (image_point_rs.valid_flag) {
                p_rs = image_point_rs.p;
            } else {
                // Rolling shutter optimization failed -> mark point as invalid.
                return ImagePoint{fvec2{}, false};
            }
        }

        return ImagePoint{p_rs, true};
    }
};

struct PerfectPinholeCameraModel : BaseCameraModel<PerfectPinholeCameraModel> {
    // OpenCV-like pinhole camera model without any distortion

    using Base = BaseCameraModel<PerfectPinholeCameraModel>;

    struct Parameters : Base::Parameters {
        fvec2 principal_point;
        fvec2 focal_length;
    };

    Parameters parameters;
    float margin_factor = 0.f;

    // Constructor
    TREND_HOST_DEVICE
    PerfectPinholeCameraModel(Parameters const &parameters, float margin_factor = 0.f)
        : parameters(parameters), margin_factor(margin_factor) {}

    inline TREND_HOST_DEVICE auto camera_point_to_image_point(fvec3 const &camera_point
    ) const -> ImagePoint {
        auto image_point = fvec2{};

        // Treat all the points behind the camera plane to invalid / projecting
        // to origin
        if (camera_point[2] <= 0.f)
            return {image_point, false};

        // Project using ideal pinhole model
        auto const uv = fvec2(camera_point[0], camera_point[1]) / camera_point[2];
        image_point = uv * parameters.focal_length + parameters.principal_point;

        // Check if the image points fall within the image, set points that have
        // too large distortion or fall outside the image sensor to invalid
        auto valid = true;
        valid &= image_point_in_image_bounds_margin(
            image_point, parameters.resolution, margin_factor
        );

        return {image_point, valid};
    }

    inline TREND_HOST_DEVICE auto image_point_to_camera_ray(fvec2 const &image_point
    ) const -> Ray {
        // Transform the image point to uv coordinate
        auto const uv =
            (image_point - parameters.principal_point) / parameters.focal_length;

        // Unproject the image point to camera ray
        auto const camera_ray_d = normalize(fvec3{uv[0], uv[1], 1.f});
        auto const camera_ray_o = fvec3{};
        return {camera_ray_o, camera_ray_d, true};
    }
};

} // namespace tinyrend::camera
