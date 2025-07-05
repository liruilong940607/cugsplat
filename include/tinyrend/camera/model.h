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
) -> bool {
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

struct ImagePointAndZBuffer {
    fvec2 p; // pixel coordinates
    float z; // z-depth
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
        tinyrend::camera::impl::shutter::Type shutter_type;
        std::array<uint32_t, 2> resolution;
        float margin_factor;
    };

    Parameters parameters;

    inline TREND_HOST_DEVICE auto _shutter_relative_frame_time(fvec2 const &image_point
    ) const -> float {
        auto const derived = static_cast<DerivedCameraModel const *>(this);
        auto const &resolution = derived->parameters.resolution;
        auto const &shutter_type = derived->parameters.shutter_type;
        return tinyrend::camera::impl::shutter::relative_frame_time(
            image_point, resolution, shutter_type
        );
    }

    // Unproject image point to world ray (with rolling shutter)
    inline TREND_HOST_DEVICE auto image_point_to_world_ray(
        fvec2 const &image_point, ShutterPoses const &shutter_poses
    ) -> Ray {
        auto const derived = static_cast<DerivedCameraModel const *>(this);
        auto const camera_ray = derived->image_point_to_camera_ray_impl(image_point);

        // If the camera ray in not valid, return a zero ray
        if (!camera_ray.valid_flag) {
            return Ray{fvec3{}, fvec3{}, false};
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
    ) const -> ImagePointAndZBuffer {
        auto const derived = static_cast<DerivedCameraModel const *>(this);
        auto const camera_point =
            tinyrend::se3::transform_point(pose.r, pose.t, world_point);
        auto const z_depth = camera_point[2];

        // Treat all the points behind the camera plane to invalid
        if (z_depth <= 0.f) {
            return ImagePointAndZBuffer{fvec2{}, 0.f, false};
        }

        // Project the camera point to the image plane
        auto const &[image_point, valid_flag] =
            derived->camera_point_to_image_point_impl(camera_point);

        if (!valid_flag) {
            // If the camera point is invalid, return an invalid image point
            return ImagePointAndZBuffer{fvec2{}, 0.f, false};
        } else {
            // In case camera point is valid, check if the image point is valid
            // Check if the image points fall within the image, set points that have
            // too large distortion or fall outside the image sensor to invalid
            auto valid = true;
            valid &= image_point_in_image_bounds_margin(
                image_point,
                derived->parameters.resolution,
                derived->parameters.margin_factor
            );
            return ImagePointAndZBuffer{image_point, z_depth, valid};
        }
    }

    // Project world point to image point (with rolling shutter)
    template <size_t N_ROLLING_SHUTTER_ITERATIONS = 10>
    inline TREND_HOST_DEVICE auto world_point_to_image_point(
        fvec3 const &world_point, ShutterPoses const &shutter_poses
    ) const -> ImagePointAndZBuffer {
        auto const derived = static_cast<DerivedCameraModel const *>(this);

        // Always perform transformation using start pose
        auto const image_point_and_depth_start =
            _transform_world_to_image(world_point, shutter_poses.start);

        // Exit early if we have a global shutter sensor
        if (derived->parameters.shutter_type ==
            tinyrend::camera::impl::shutter::Type::GLOBAL) {
            return image_point_and_depth_start;
        }

        // This selection prefers points at the start-of-frame pose over
        // end-of-frame points
        auto p_rs = fvec2{};
        auto z_rs = 0.f;
        if (image_point_and_depth_start.valid_flag) {
            p_rs = image_point_and_depth_start.p;
            z_rs = image_point_and_depth_start.z;
        } else {
            // Do initial transformations using both start and end poses to
            // determine all candidate points and take union of valid projections as
            // iteration starting points
            auto const image_point_and_depth_end =
                _transform_world_to_image(world_point, shutter_poses.end);
            if (image_point_and_depth_end.valid_flag) {
                p_rs = image_point_and_depth_end.p;
                z_rs = image_point_and_depth_end.z;
            } else {
                // No valid projection at start or finish -> mark point as invalid.
                return ImagePointAndZBuffer{fvec2{}, 0.f, false};
            }
        }

        // Iterative Rolling Shutter Optimization
#pragma unroll
        for (auto j = 0; j < N_ROLLING_SHUTTER_ITERATIONS; ++j) {
            auto const pose_rs = interpolate_shutter_poses(
                _shutter_relative_frame_time(p_rs), shutter_poses
            );
            auto const image_point_and_depth_rs =
                _transform_world_to_image(world_point, pose_rs);
            if (image_point_and_depth_rs.valid_flag) {
                p_rs = image_point_and_depth_rs.p;
                z_rs = image_point_and_depth_rs.z;
            } else {
                // Rolling shutter optimization failed -> mark point as invalid.
                return ImagePointAndZBuffer{fvec2{}, 0.f, false};
            }
        }

        return ImagePointAndZBuffer{p_rs, z_rs, true};
    }
};

struct PerfectPinholeCameraModelImpl : BaseCameraModel<PerfectPinholeCameraModelImpl> {
    // OpenCV-like pinhole camera model without any distortion

    using Base = BaseCameraModel<PerfectPinholeCameraModelImpl>;

    struct Parameters : Base::Parameters {
        fvec2 principal_point;
        fvec2 focal_length;
    };

    Parameters parameters;

    TREND_HOST_DEVICE
    PerfectPinholeCameraModelImpl(Parameters const &parameters)
        : parameters(parameters) {}

    inline TREND_HOST_DEVICE auto
    camera_point_to_image_point_impl(fvec3 const &camera_point
    ) const -> std::pair<fvec2, bool> {
        auto const image_point = tinyrend::camera::impl::pinhole::project(
            camera_point, parameters.focal_length, parameters.principal_point
        );
        return {image_point, true};
    }

    inline TREND_HOST_DEVICE auto
    image_point_to_camera_ray_impl(fvec2 const &image_point) const -> Ray {
        auto const camera_ray_o = fvec3{};
        auto const camera_ray_d = tinyrend::camera::impl::pinhole::unproject(
            image_point, parameters.focal_length, parameters.principal_point
        );
        return Ray{camera_ray_o, camera_ray_d, true};
    }
};

struct OpenCVPinholeCameraModelImpl : BaseCameraModel<OpenCVPinholeCameraModelImpl> {
    // OpenCV-like pinhole camera model with distortion

    using Base = BaseCameraModel<OpenCVPinholeCameraModelImpl>;

    struct Parameters : Base::Parameters {
        fvec2 principal_point;
        fvec2 focal_length;
        std::array<float, 6> radial_coeffs;
        std::array<float, 2> tangential_coeffs;
        std::array<float, 4> thin_prism_coeffs;
        float min_radial_dist = 0.8f;
        float max_radial_dist = std::numeric_limits<float>::max();
    };

    Parameters parameters;

    TREND_HOST_DEVICE
    OpenCVPinholeCameraModelImpl(Parameters const &parameters)
        : parameters(parameters) {}

    inline TREND_HOST_DEVICE auto
    camera_point_to_image_point_impl(fvec3 const &camera_point
    ) const -> std::pair<fvec2, bool> {
        auto const &[image_point, valid_flag] =
            tinyrend::camera::impl::pinhole::project(
                camera_point,
                parameters.focal_length,
                parameters.principal_point,
                parameters.radial_coeffs,
                parameters.tangential_coeffs,
                parameters.thin_prism_coeffs,
                parameters.min_radial_dist,
                parameters.max_radial_dist
            );
        return {image_point, valid_flag};
    }

    inline TREND_HOST_DEVICE auto
    image_point_to_camera_ray_impl(fvec2 const &image_point) const -> Ray {
        auto const camera_ray_o = fvec3{};
        auto const &[camera_ray_d, valid_flag] =
            tinyrend::camera::impl::pinhole::unproject(
                image_point,
                parameters.focal_length,
                parameters.principal_point,
                parameters.radial_coeffs,
                parameters.tangential_coeffs,
                parameters.thin_prism_coeffs,
                parameters.min_radial_dist,
                parameters.max_radial_dist
            );
        return Ray{camera_ray_o, camera_ray_d, valid_flag};
    }
};

struct PerfectFisheyeCameraModelImpl : BaseCameraModel<PerfectFisheyeCameraModelImpl> {
    // Perfect fisheye camera model without any distortion

    using Base = BaseCameraModel<PerfectFisheyeCameraModelImpl>;

    struct Parameters : Base::Parameters {
        fvec2 principal_point;
        fvec2 focal_length;
        float min_2d_norm = 1e-6f;
    };

    Parameters parameters;

    TREND_HOST_DEVICE
    PerfectFisheyeCameraModelImpl(Parameters const &parameters)
        : parameters(parameters) {}

    inline TREND_HOST_DEVICE auto
    camera_point_to_image_point_impl(fvec3 const &camera_point
    ) const -> std::pair<fvec2, bool> {
        auto const image_point = tinyrend::camera::impl::fisheye::project(
            camera_point,
            parameters.focal_length,
            parameters.principal_point,
            parameters.min_2d_norm
        );
        return {image_point, true};
    }

    inline TREND_HOST_DEVICE auto
    image_point_to_camera_ray_impl(fvec2 const &image_point) const -> Ray {
        auto const camera_ray_o = fvec3{};
        auto const camera_ray_d = tinyrend::camera::impl::fisheye::unproject(
            image_point,
            parameters.focal_length,
            parameters.principal_point,
            parameters.min_2d_norm
        );
        return Ray{camera_ray_o, camera_ray_d, true};
    }
};

struct OpenCVFisheyeCameraModelImpl : BaseCameraModel<OpenCVFisheyeCameraModelImpl> {
    // OpenCV-like fisheye camera model with distortion

    using Base = BaseCameraModel<OpenCVFisheyeCameraModelImpl>;

    struct Parameters : Base::Parameters {
        fvec2 principal_point;
        fvec2 focal_length;
        std::array<float, 4> radial_coeffs;
        float min_2d_norm = 1e-6f;
        float max_theta = std::numeric_limits<float>::max();
    };

    Parameters parameters;

    TREND_HOST_DEVICE
    OpenCVFisheyeCameraModelImpl(Parameters const &parameters)
        : parameters(parameters) {}

    inline TREND_HOST_DEVICE auto
    camera_point_to_image_point_impl(fvec3 const &camera_point
    ) const -> std::pair<fvec2, bool> {
        auto const &[image_point, valid_flag] =
            tinyrend::camera::impl::fisheye::project(
                camera_point,
                parameters.focal_length,
                parameters.principal_point,
                parameters.radial_coeffs,
                parameters.min_2d_norm,
                parameters.max_theta
            );
        return {image_point, valid_flag};
    }

    inline TREND_HOST_DEVICE auto
    image_point_to_camera_ray_impl(fvec2 const &image_point) const -> Ray {
        auto const camera_ray_o = fvec3{};
        auto const &[camera_ray_d, valid_flag] =
            tinyrend::camera::impl::fisheye::unproject(
                image_point,
                parameters.focal_length,
                parameters.principal_point,
                parameters.radial_coeffs,
                parameters.min_2d_norm,
                parameters.max_theta
            );
        return Ray{camera_ray_o, camera_ray_d, valid_flag};
    }
};

struct OrthogonalCameraModelImpl : BaseCameraModel<OrthogonalCameraModelImpl> {
    // Orthogonal camera model

    using Base = BaseCameraModel<OrthogonalCameraModelImpl>;

    struct Parameters : Base::Parameters {
        fvec2 principal_point;
        fvec2 focal_length;
    };

    Parameters parameters;

    TREND_HOST_DEVICE
    OrthogonalCameraModelImpl(Parameters const &parameters) : parameters(parameters) {}

    inline TREND_HOST_DEVICE auto
    camera_point_to_image_point_impl(fvec3 const &camera_point
    ) const -> std::pair<fvec2, bool> {
        auto const image_point = tinyrend::camera::impl::orthogonal::project(
            camera_point, parameters.focal_length, parameters.principal_point
        );
        return {image_point, true};
    }

    inline TREND_HOST_DEVICE auto
    image_point_to_camera_ray_impl(fvec2 const &image_point) const -> Ray {
        auto const &[camera_ray_o, camera_ray_d] =
            tinyrend::camera::impl::orthogonal::unproject(
                image_point, parameters.focal_length, parameters.principal_point
            );
        return Ray{camera_ray_o, camera_ray_d, true};
    }
};

using PerfectPinholeCameraModel = BaseCameraModel<PerfectPinholeCameraModelImpl>;
using OpenCVPinholeCameraModel = BaseCameraModel<OpenCVPinholeCameraModelImpl>;
using PerfectFisheyeCameraModel = BaseCameraModel<PerfectFisheyeCameraModelImpl>;
using OpenCVFisheyeCameraModel = BaseCameraModel<OpenCVFisheyeCameraModelImpl>;
using OrthogonalCameraModel = BaseCameraModel<OrthogonalCameraModelImpl>;

} // namespace tinyrend::camera
