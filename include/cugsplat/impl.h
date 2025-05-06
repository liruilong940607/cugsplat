#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <tuple>

#include "cugsplat/camera/fisheye.h"
#include "cugsplat/camera/orthogonal.h"
#include "cugsplat/camera/pinhole.h"
#include "cugsplat/camera/shutter.h"
#include "cugsplat/core/macros.h" // for GSPLAT_HOST_DEVICE
#include "cugsplat/core/se3.h"
#include "cugsplat/estimator/ut.h"
#include "cugsplat/gaussian.h"

namespace cugsplat::impl {

enum struct CameraType { PERFECT_PINHOLE, PERFECT_FISHEYE, ORTHO };

template <CameraType CAMERA_TYPE>
GSPLAT_HOST_DEVICE inline auto projection_perfect_camera_shutter(
    const float *Ks, // [3, 3]
    const float near_plane,
    const float far_plane,
    const float *viewmats0, // [4, 4]
    const float *viewmats1, // [4, 4]
    const cugsplat::shutter::Type shutter_type,
    const uint32_t width,
    const uint32_t height,
    const float *means,  // [3]
    const float *quats,  // [4]
    const float *scales, // [3]
    const float margin_factor = 0.15f
) -> std::tuple<glm::fvec2, float, glm::fmat2, bool> {

    glm::fvec2 means2d;
    float depth;
    glm::fmat2 covar2d;
    bool valid_flag = false;

    do {
        // load extrinsics
        auto const world_to_camera_R0 = glm::fmat3(
            viewmats0[0],
            viewmats0[4],
            viewmats0[8],
            viewmats0[1],
            viewmats0[5],
            viewmats0[9],
            viewmats0[2],
            viewmats0[6],
            viewmats0[10]
        );
        auto const world_to_camera_t0 =
            glm::fvec3(viewmats0[3], viewmats0[7], viewmats0[11]);

        // load extrinsics (for rolling shutter)
        glm::fmat3 world_to_camera_R1;
        glm::fvec3 world_to_camera_t1;
        if (shutter_type != cugsplat::shutter::Type::GLOBAL) {
            world_to_camera_R1 = glm::fmat3(
                viewmats1[0],
                viewmats1[4],
                viewmats1[8],
                viewmats1[1],
                viewmats1[5],
                viewmats1[9],
                viewmats1[2],
                viewmats1[6],
                viewmats1[10]
            );
            world_to_camera_t1 = glm::fvec3(viewmats1[3], viewmats1[7], viewmats1[11]);
        }

        // transform world to camera
        auto const mu = glm::fvec3(means[0], means[1], means[2]);
        auto const mu_c0 =
            cugsplat::se3::transform_point(world_to_camera_R0, world_to_camera_t0, mu);
        if (shutter_type == cugsplat::shutter::Type::GLOBAL) {
            if (mu_c0.z < near_plane || mu_c0.z > far_plane) {
                break;
            }
        } else {
            auto const mu_c1 = cugsplat::se3::transform_point(
                world_to_camera_R1, world_to_camera_t1, mu
            );
            if ((mu_c0.z < near_plane || mu_c0.z > far_plane) &&
                (mu_c1.z < near_plane || mu_c1.z > far_plane)) {
                break;
            }
        }

        // load intrinsics
        auto const focal_length = glm::fvec2(Ks[0], Ks[4]);
        auto const principal_point = glm::fvec2(Ks[2], Ks[5]);

        auto const point_camera_to_image_fn =
            [&focal_length, &principal_point, &width, &height, &margin_factor](
                const glm::fvec3 &camera_point
            ) -> std::pair<glm::fvec2, bool> {
            // camera to image plane
            glm::fvec2 image_point;
            if constexpr (CAMERA_TYPE == CameraType::PERFECT_FISHEYE) {
                image_point = cugsplat::fisheye::project(
                    camera_point, focal_length, principal_point
                );
            } else if constexpr (CAMERA_TYPE == CameraType::PERFECT_PINHOLE) {
                image_point = cugsplat::pinhole::project(
                    camera_point, focal_length, principal_point
                );
            } else if constexpr (CAMERA_TYPE == CameraType::ORTHO) {
                image_point = cugsplat::orthogonal::project(
                    camera_point, focal_length, principal_point
                );
            }
            // check if in image (with a margin)
            auto const uv = image_point / glm::fvec2(width, height);
            if (uv.x < -margin_factor || uv.x > 1.f + margin_factor ||
                uv.y < -margin_factor || uv.y > 1.f + margin_factor) {
                return {glm::fvec2{}, false};
            } else {
                return {image_point, true};
            }
        };

        // world to image plane
        auto const result = cugsplat::shutter::point_world_to_image(
            point_camera_to_image_fn,
            {width, height},
            mu,
            world_to_camera_R0,
            world_to_camera_t0,
            world_to_camera_R1,
            world_to_camera_t1,
            shutter_type
        );
        if (!result.valid_flag) {
            break;
        }
        means2d = result.image_point;
        depth = result.camera_point.z;
        auto const world_to_camera_R = result.pose_r;

        // load covariance
        auto const quat = glm::fvec4(quats[0], quats[1], quats[2], quats[3]);
        auto const scale = glm::fvec3(scales[0], scales[1], scales[2]);
        auto const covar = cugsplat::gaussian::quat_scale_to_covar(quat, scale);

        // project covariance
        covar2d = cugsplat::se3::transform_covar(world_to_camera_R, covar);
        valid_flag = true;
    } while (false);

    return {means2d, depth, covar2d, valid_flag};
}

} // namespace cugsplat::impl
