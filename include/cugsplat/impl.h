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

template <size_t N, typename T>
inline GSPLAT_HOST_DEVICE std::array<T, N> make_array(const T *ptr, size_t offset = 0) {
    std::array<T, N> arr{}; // zero-initialize
    if (!ptr) {
        return arr;
    }
#pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
        arr[i] = ptr[offset + i];
    }
    return arr;
}

enum struct CameraType { PERFECT_PINHOLE, PINHOLE, PERFECT_FISHEYE, FISHEYE, ORTHO };

template <CameraType CAMERA_TYPE> struct DistortionParameters {};

template <> struct DistortionParameters<CameraType::PINHOLE> {
    float *radial_coeffs = nullptr;
    float *tangential_coeffs = nullptr;
    float *thin_prism_coeffs = nullptr;
};

template <> struct DistortionParameters<CameraType::FISHEYE> {
    float *radial_coeffs = nullptr;
};

template <CameraType CAMERA_TYPE>
GSPLAT_HOST_DEVICE inline auto projection(
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
    const float margin_factor = 0.15f,
    const DistortionParameters<CAMERA_TYPE> &dist_params = {}
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

        // define function to project a camera point to an image point
        auto const point_camera_to_image_fn = [&focal_length,
                                               &principal_point,
                                               &width,
                                               &height,
                                               &margin_factor,
                                               &dist_params](
                                                  const glm::fvec3 &camera_point
                                              ) -> std::pair<glm::fvec2, bool> {
            // camera to image plane
            glm::fvec2 image_point;
            bool valid_flag;
            if constexpr (CAMERA_TYPE == CameraType::PERFECT_FISHEYE) {
                image_point = cugsplat::fisheye::project(
                    camera_point, focal_length, principal_point
                );
                valid_flag = true;
            } else if constexpr (CAMERA_TYPE == CameraType::PERFECT_PINHOLE) {
                image_point = cugsplat::pinhole::project(
                    camera_point, focal_length, principal_point
                );
                valid_flag = true;
            } else if constexpr (CAMERA_TYPE == CameraType::ORTHO) {
                image_point = cugsplat::orthogonal::project(
                    camera_point, focal_length, principal_point
                );
                valid_flag = true;
            } else if constexpr (CAMERA_TYPE == CameraType::PINHOLE) {
                auto const &[image_point_, valid_flag_] = cugsplat::pinhole::project(
                    camera_point,
                    focal_length,
                    principal_point,
                    make_array<6>(dist_params.radial_coeffs),
                    make_array<2>(dist_params.tangential_coeffs),
                    make_array<4>(dist_params.thin_prism_coeffs)
                );
                image_point = image_point_;
                valid_flag = valid_flag_;
            } else if constexpr (CAMERA_TYPE == CameraType::FISHEYE) {
                auto const &[image_point_, valid_flag_] = cugsplat::fisheye::project(
                    camera_point,
                    focal_length,
                    principal_point,
                    make_array<4>(dist_params.radial_coeffs)
                );
                image_point = image_point_;
                valid_flag = valid_flag_;
            }
            if (!valid_flag) {
                return {glm::fvec2{}, false};
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

        // define function to project a world point to an image point
        using AuxData = std::pair<glm::fvec3, glm::fmat3>;
        auto const point_world_to_image_fn =
            [&point_camera_to_image_fn,
             &width,
             &height,
             &world_to_camera_R0,
             &world_to_camera_t0,
             &world_to_camera_R1,
             &world_to_camera_t1,
             &shutter_type](const glm::fvec3 &world_point
            ) -> std::tuple<glm::fvec2, bool, AuxData> {
            auto const result = cugsplat::shutter::point_world_to_image(
                point_camera_to_image_fn,
                {width, height},
                world_point,
                world_to_camera_R0,
                world_to_camera_t0,
                world_to_camera_R1,
                world_to_camera_t1,
                shutter_type
            );
            return {
                result.image_point,
                result.valid_flag,
                AuxData{result.camera_point, result.pose_r}
            };
        };

        // execute the function
        auto const [image_point, image_point_valid_flag, aux] =
            point_world_to_image_fn(mu);
        if (!image_point_valid_flag) {
            break;
        }
        means2d = image_point;
        auto const camera_point = std::get<0>(aux);
        depth = camera_point.z;
        auto const world_to_camera_R = std::get<1>(aux);

        // project covariance from camera space to image space
        glm::fmat3x2 J;
        if constexpr (CAMERA_TYPE == CameraType::PERFECT_FISHEYE) {
            J = cugsplat::fisheye::project_jac(camera_point, focal_length);
        } else if constexpr (CAMERA_TYPE == CameraType::PERFECT_PINHOLE) {
            J = cugsplat::pinhole::project_jac(camera_point, focal_length);
        } else if constexpr (CAMERA_TYPE == CameraType::ORTHO) {
            J = cugsplat::orthogonal::project_jac(camera_point, focal_length);
        } else if constexpr (CAMERA_TYPE == CameraType::PINHOLE) {
            auto const &[J_, valid_flag_] = cugsplat::pinhole::project_jac(
                camera_point,
                focal_length,
                make_array<6>(dist_params.radial_coeffs),
                make_array<2>(dist_params.tangential_coeffs),
                make_array<4>(dist_params.thin_prism_coeffs)
            );
            if (!valid_flag_) {
                break;
            }
            J = J_;
        } else if constexpr (CAMERA_TYPE == CameraType::FISHEYE) {
            // auto const &[J_, valid_flag_] = cugsplat::fisheye::project_jac(
            //     camera_point, focal_length,
            //     make_array<4>(dist_params.radial_coeffs)
            // );
            // if (!valid_flag_) {
            //     break;
            // }
            // J = J_;
            // TODO: implement
            // static_assert(false, "Fisheye projection Jacobian not implemented");
        }

        // load covariance
        auto const quat = glm::fvec4(quats[0], quats[1], quats[2], quats[3]);
        auto const scale = glm::fvec3(scales[0], scales[1], scales[2]);
        auto const covar = cugsplat::gaussian::quat_scale_to_covar(quat, scale);
        // transform covariance to camera space, then to image space
        auto const covar_c = cugsplat::se3::transform_covar(world_to_camera_R, covar);
        covar2d = J * covar_c * glm::transpose(J);

        // reach here means valid
        valid_flag = true;
    } while (false);

    return {means2d, depth, covar2d, valid_flag};
}

} // namespace cugsplat::impl
