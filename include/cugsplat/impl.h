#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
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

enum struct CameraType {
    PINHOLE,
    FISHEYE,
    ORTHO,
    DISTORTED_PINHOLE,
    DISTORTED_FISHEYE
};

template <CameraType CAMERA_TYPE> struct DistortionParameters {};

template <> struct DistortionParameters<CameraType::PINHOLE> {
    float *radial_coeffs = nullptr;
    float *tangential_coeffs = nullptr;
    float *thin_prism_coeffs = nullptr;
};

template <> struct DistortionParameters<CameraType::FISHEYE> {
    float *radial_coeffs = nullptr;
};

struct ProjectionForwardResult {
    glm::fvec2 means2d;
    float depth;
    glm::fmat2 covar2d;
    bool valid_flag;
};

template <CameraType CAMERA_TYPE, bool USE_UT = false>
GSPLAT_HOST_DEVICE inline auto projection_forward(
    const float *intrinsic_ptr, // [3, 3]
    const float near_plane,
    const float far_plane,
    const float *world_to_camera0_ptr, // [4, 4]
    const float *world_to_camera1_ptr, // [4, 4]
    const cugsplat::shutter::Type shutter_type,
    const uint32_t width,
    const uint32_t height,
    const float *mean_ptr,  // [3]
    const float *quat_ptr,  // [4]
    const float *scale_ptr, // [3]
    const float margin_factor = 0.15f,
    const DistortionParameters<CAMERA_TYPE> &dist_params = {}
) -> ProjectionForwardResult {

    // prepare return values
    glm::fvec2 means2d;
    float depth;
    glm::fmat2 covar2d;
    bool valid_flag = false;

    do {
        // load extrinsics
        // note glm is column-major, and we assume the input is row-major
        auto const world_to_camera0 =
            glm::transpose(glm::make_mat4(world_to_camera0_ptr));
        auto const world_to_camera_R0 = glm::fmat3(world_to_camera0);
        auto const world_to_camera_t0 = glm::fvec3(world_to_camera0[3]);

        // load the second extrinsics for rolling shutter
        glm::fmat3 world_to_camera_R1;
        glm::fvec3 world_to_camera_t1;
        if (shutter_type != cugsplat::shutter::Type::GLOBAL) {
            auto const world_to_camera1 =
                glm::transpose(glm::make_mat4(world_to_camera1_ptr));
            world_to_camera_R1 = glm::fmat3(world_to_camera1);
            world_to_camera_t1 = glm::fvec3(world_to_camera1[3]);
        }

        // transform world to camera
        auto const mu = glm::fvec3(mean_ptr[0], mean_ptr[1], mean_ptr[2]);
        auto const mu_c0 =
            cugsplat::se3::transform_point(world_to_camera_R0, world_to_camera_t0, mu);
        if (shutter_type == cugsplat::shutter::Type::GLOBAL) {
            // If point is not in the frustum, skip
            if (mu_c0.z < near_plane || mu_c0.z > far_plane) {
                break;
            }
        } else {
            // For rolling shutter, only skip if the point is not in either frustum
            auto const mu_c1 = cugsplat::se3::transform_point(
                world_to_camera_R1, world_to_camera_t1, mu
            );
            if ((mu_c0.z < near_plane || mu_c0.z > far_plane) &&
                (mu_c1.z < near_plane || mu_c1.z > far_plane)) {
                break;
            }
        }

        // load intrinsics
        auto const focal_length = glm::fvec2(intrinsic_ptr[0], intrinsic_ptr[4]);
        auto const principal_point = glm::fvec2(intrinsic_ptr[2], intrinsic_ptr[5]);

        // load distortion coefficients
        std::conditional_t<
            CAMERA_TYPE == CameraType::PINHOLE ||
                CAMERA_TYPE == CameraType::DISTORTED_PINHOLE,
            std::array<float, 6>,
            std::array<float, 4>>
            radial_coeffs;
        std::array<float, 2> tangential_coeffs;
        std::array<float, 4> thin_prism_coeffs;
        if constexpr (CAMERA_TYPE == CameraType::PINHOLE) {
            radial_coeffs = make_array<6>(dist_params.radial_coeffs);
            tangential_coeffs = make_array<2>(dist_params.tangential_coeffs);
            thin_prism_coeffs = make_array<4>(dist_params.thin_prism_coeffs);
        } else if constexpr (CAMERA_TYPE == CameraType::FISHEYE) {
            radial_coeffs = make_array<4>(dist_params.radial_coeffs);
        }

        // define function to project a camera point to an image point
        auto const point_camera_to_image_fn = [&focal_length,
                                               &principal_point,
                                               &width,
                                               &height,
                                               &margin_factor,
                                               &radial_coeffs,
                                               &tangential_coeffs,
                                               &thin_prism_coeffs](
                                                  const glm::fvec3 &camera_point
                                              ) -> std::pair<glm::fvec2, bool> {
            // camera to image plane
            glm::fvec2 image_point;
            bool valid_flag;
            if constexpr (CAMERA_TYPE == CameraType::FISHEYE) {
                image_point = cugsplat::fisheye::project(
                    camera_point, focal_length, principal_point
                );
                valid_flag = true;
            } else if constexpr (CAMERA_TYPE == CameraType::PINHOLE) {
                image_point = cugsplat::pinhole::project(
                    camera_point, focal_length, principal_point
                );
                valid_flag = true;
            } else if constexpr (CAMERA_TYPE == CameraType::ORTHO) {
                image_point = cugsplat::orthogonal::project(
                    camera_point, focal_length, principal_point
                );
                valid_flag = true;
            } else if constexpr (CAMERA_TYPE == CameraType::DISTORTED_PINHOLE) {
                auto const &[image_point_, valid_flag_] = cugsplat::pinhole::project(
                    camera_point,
                    focal_length,
                    principal_point,
                    radial_coeffs,
                    tangential_coeffs,
                    thin_prism_coeffs
                );
                image_point = image_point_;
                valid_flag = valid_flag_;
            } else if constexpr (CAMERA_TYPE == CameraType::DISTORTED_FISHEYE) {
                auto const &[image_point_, valid_flag_] = cugsplat::fisheye::project(
                    camera_point, focal_length, principal_point, radial_coeffs
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

        if constexpr (USE_UT) {
            // load covariance
            auto const quat =
                glm::fvec4(quat_ptr[0], quat_ptr[1], quat_ptr[2], quat_ptr[3]);
            auto const scale = glm::fvec3(scale_ptr[0], scale_ptr[1], scale_ptr[2]);
            auto const sqrt_covar =
                cugsplat::gaussian::quat_scale_to_scaled_rotmat(quat, scale);

            // execute the function using unscented transform
            auto const result = cugsplat::ut::transform<3, 2, AuxData>(
                point_world_to_image_fn, mu, sqrt_covar
            );
            if (!result.valid_flag) {
                break;
            }
            means2d = result.mu;
            covar2d = result.covar;
            auto const camera_point = result.aux.first;
            depth = camera_point.z;

        } else {
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
            static_assert(
                CAMERA_TYPE != CameraType::DISTORTED_FISHEYE,
                "Jacobian for distorted fisheye is not implemented"
            );
            if constexpr (CAMERA_TYPE == CameraType::FISHEYE) {
                J = cugsplat::fisheye::project_jac(camera_point, focal_length);
            } else if constexpr (CAMERA_TYPE == CameraType::PINHOLE) {
                J = cugsplat::pinhole::project_jac(camera_point, focal_length);
            } else if constexpr (CAMERA_TYPE == CameraType::ORTHO) {
                J = cugsplat::orthogonal::project_jac(camera_point, focal_length);
            } else if constexpr (CAMERA_TYPE == CameraType::DISTORTED_PINHOLE) {
                auto const &[J_, valid_flag_] = cugsplat::pinhole::project_jac(
                    camera_point,
                    focal_length,
                    radial_coeffs,
                    tangential_coeffs,
                    thin_prism_coeffs
                );
                if (!valid_flag_) {
                    break;
                }
                J = J_;
            } else if constexpr (CAMERA_TYPE == CameraType::DISTORTED_FISHEYE) {
                // TODO: implement
            }

            // load covariance
            auto const quat =
                glm::fvec4(quat_ptr[0], quat_ptr[1], quat_ptr[2], quat_ptr[3]);
            auto const scale = glm::fvec3(scale_ptr[0], scale_ptr[1], scale_ptr[2]);
            auto const covar = cugsplat::gaussian::quat_scale_to_covar(quat, scale);
            // transform covariance to camera space, then to image space
            auto const covar_c =
                cugsplat::se3::transform_covar(world_to_camera_R, covar);
            covar2d = J * covar_c * glm::transpose(J);
        }

        // reach here mean_ptr valid
        valid_flag = true;
    } while (false);

    return {means2d, depth, covar2d, valid_flag};
}

struct ProjectionBackwardResult {
    glm::fvec3 v_mean;
    glm::fvec4 v_quat;
    glm::fvec3 v_scale;
    glm::fmat4 v_world_to_camera0;
    glm::fmat4 v_world_to_camera1;
};

// template <CameraType CAMERA_TYPE, bool USE_UT = false>
// GSPLAT_HOST_DEVICE inline auto projection_backward(
//     // inputs
//     const float *intrinsic_ptr, // [3, 3]
//     const float near_plane,
//     const float far_plane,
//     const float *world_to_camera0_ptr, // [4, 4]
//     const float *world_to_camera1_ptr, // [4, 4]
//     const cugsplat::shutter::Type shutter_type,
//     const uint32_t width,
//     const uint32_t height,
//     const float *mean_ptr,  // [3]
//     const float *quat_ptr,  // [4]
//     const float *scale_ptr, // [3]
//     // outputs
//     float *mean2d_ptr,  // [2]
//     float *depth_ptr,   // [1]
//     float *covar2d_ptr, // [2, 2]
//     // output gradients
//     float *v_mean2d_ptr,  // [2]
//     float *v_depth_ptr,   // [1]
//     float *v_covar2d_ptr, // [2, 2]
//     // inputs with default values
//     const float margin_factor = 0.15f,
//     const DistortionParameters<CAMERA_TYPE> &dist_params = {}
// ) -> ProjectionBackwardResult {

//     // prepare return values
//     glm::fvec3 v_mean = {};
//     glm::fmat3 v_covar = {};
//     glm::fmat3 v_world_to_camera_R = {};
//     glm::fvec3 v_world_to_camera_t = {};

//     do {
//         // load output gradients
//         auto const v_mean2d = glm::make_vec2(v_mean2d_ptr);
//         auto const v_depth = v_depth_ptr[0];
//         auto const v_covar2d = glm::make_mat2(v_covar2d_ptr);

//         glm::fmat3 const world_to_camera_R;
//         auto const world_to_camera_Rt = glm::transpose(world_to_camera_R);

//         if constexpr (CAMERA_TYPE == CameraType::FISHEYE) {
//             auto const J = cugsplat::fisheye::project_jac(mu, focal_length);
//             auto const Jt = glm::transpose(J);

//             auto const v_mean_c = Jt * v_mean2d;
//             v_mean += world_to_camera_Rt * v_mean_c;
//             v_world_to_camera_R += glm::outerProduct(v_mean_c, mean);
//             v_world_to_camera_t += v_mean_c;

//             auto const v_covar_c = Jt * v_covar2d * J;
//             v_covar += world_to_camera_Rt * v_covar_c * world_to_camera_R;
//             v_world_to_camera_R +=
//                 v_covar_c * world_to_camera_R * glm::transpose(covar) +
//                 glm::transpose(v_covar_c) * world_to_camera_R * covar;

//             auto const v_J = v_covar2d * J * glm::transpose(covar) +
//                              glm::transpose(v_covar2d) * J * covar;
//             auto const &[H1, H2] = cugsplat::fisheye::project_hess(mu, focal_length);
//         }

//         // load extrinsics
//         // note glm is column-major, and we assume the input is row-major
//         auto const world_to_camera0 =
//             glm::transpose(glm::make_mat4(world_to_camera0_ptr));
//         auto const world_to_camera_R0 = glm::fmat3(world_to_camera0);
//         auto const world_to_camera_t0 = glm::fvec3(world_to_camera0[3]);

//         // load the second extrinsics for rolling shutter
//         glm::fmat3 world_to_camera_R1;
//         glm::fvec3 world_to_camera_t1;
//         if (shutter_type != cugsplat::shutter::Type::GLOBAL) {
//             auto const world_to_camera1 =
//                 glm::transpose(glm::make_mat4(world_to_camera1_ptr));
//             world_to_camera_R1 = glm::fmat3(world_to_camera1);
//             world_to_camera_t1 = glm::fvec3(world_to_camera1[3]);
//         }

//         // transform world to camera
//         auto const mu = glm::fvec3(mean_ptr[0], mean_ptr[1], mean_ptr[2]);
//         auto const mu_c0 =
//             cugsplat::se3::transform_point(world_to_camera_R0, world_to_camera_t0,
//             mu);
//         if (shutter_type == cugsplat::shutter::Type::GLOBAL) {
//             // If point is not in the frustum, skip
//             if (mu_c0.z < near_plane || mu_c0.z > far_plane) {
//                 break;
//             }
//         } else {
//             // For rolling shutter, only skip if the point is not in either frustum
//             auto const mu_c1 = cugsplat::se3::transform_point(
//                 world_to_camera_R1, world_to_camera_t1, mu
//             );
//             if ((mu_c0.z < near_plane || mu_c0.z > far_plane) &&
//                 (mu_c1.z < near_plane || mu_c1.z > far_plane)) {
//                 break;
//             }
//         }

//         // load intrinsics
//         auto const focal_length = glm::fvec2(intrinsic_ptr[0], intrinsic_ptr[4]);
//         auto const principal_point = glm::fvec2(intrinsic_ptr[2], intrinsic_ptr[5]);

//         // load distortion coefficients
//         std::conditional_t<
//             CAMERA_TYPE == CameraType::PINHOLE ||
//                 CAMERA_TYPE == CameraType::DISTORTED_PINHOLE,
//             std::array<float, 6>,
//             std::array<float, 4>>
//             radial_coeffs;
//         std::array<float, 2> tangential_coeffs;
//         std::array<float, 4> thin_prism_coeffs;
//         if constexpr (CAMERA_TYPE == CameraType::PINHOLE) {
//             radial_coeffs = make_array<6>(dist_params.radial_coeffs);
//             tangential_coeffs = make_array<2>(dist_params.tangential_coeffs);
//             thin_prism_coeffs = make_array<4>(dist_params.thin_prism_coeffs);
//         } else if constexpr (CAMERA_TYPE == CameraType::FISHEYE) {
//             radial_coeffs = make_array<4>(dist_params.radial_coeffs);
//         }

//         // define function to project a camera point to an image point
//         auto const point_camera_to_image_fn = [&focal_length,
//                                                &principal_point,
//                                                &width,
//                                                &height,
//                                                &margin_factor,
//                                                &radial_coeffs,
//                                                &tangential_coeffs,
//                                                &thin_prism_coeffs](
//                                                   const glm::fvec3 &camera_point
//                                               ) -> std::pair<glm::fvec2, bool> {
//             // camera to image plane
//             glm::fvec2 image_point;
//             bool valid_flag;
//             if constexpr (CAMERA_TYPE == CameraType::FISHEYE) {
//                 image_point = cugsplat::fisheye::project(
//                     camera_point, focal_length, principal_point
//                 );
//                 valid_flag = true;
//             } else if constexpr (CAMERA_TYPE == CameraType::PINHOLE) {
//                 image_point = cugsplat::pinhole::project(
//                     camera_point, focal_length, principal_point
//                 );
//                 valid_flag = true;
//             } else if constexpr (CAMERA_TYPE == CameraType::ORTHO) {
//                 image_point = cugsplat::orthogonal::project(
//                     camera_point, focal_length, principal_point
//                 );
//                 valid_flag = true;
//             } else if constexpr (CAMERA_TYPE == CameraType::DISTORTED_PINHOLE) {
//                 auto const &[image_point_, valid_flag_] = cugsplat::pinhole::project(
//                     camera_point,
//                     focal_length,
//                     principal_point,
//                     radial_coeffs,
//                     tangential_coeffs,
//                     thin_prism_coeffs
//                 );
//                 image_point = image_point_;
//                 valid_flag = valid_flag_;
//             } else if constexpr (CAMERA_TYPE == CameraType::DISTORTED_FISHEYE) {
//                 auto const &[image_point_, valid_flag_] = cugsplat::fisheye::project(
//                     camera_point, focal_length, principal_point, radial_coeffs
//                 );
//                 image_point = image_point_;
//                 valid_flag = valid_flag_;
//             }
//             if (!valid_flag) {
//                 return {glm::fvec2{}, false};
//             }
//             // check if in image (with a margin)
//             auto const uv = image_point / glm::fvec2(width, height);
//             if (uv.x < -margin_factor || uv.x > 1.f + margin_factor ||
//                 uv.y < -margin_factor || uv.y > 1.f + margin_factor) {
//                 return {glm::fvec2{}, false};
//             } else {
//                 return {image_point, true};
//             }
//         };

//         // define function to project a world point to an image point
//         using AuxData = std::pair<glm::fvec3, glm::fmat3>;
//         auto const point_world_to_image_fn =
//             [&point_camera_to_image_fn,
//              &width,
//              &height,
//              &world_to_camera_R0,
//              &world_to_camera_t0,
//              &world_to_camera_R1,
//              &world_to_camera_t1,
//              &shutter_type](const glm::fvec3 &world_point
//             ) -> std::tuple<glm::fvec2, bool, AuxData> {
//             auto const result = cugsplat::shutter::point_world_to_image(
//                 point_camera_to_image_fn,
//                 {width, height},
//                 world_point,
//                 world_to_camera_R0,
//                 world_to_camera_t0,
//                 world_to_camera_R1,
//                 world_to_camera_t1,
//                 shutter_type
//             );
//             return {
//                 result.image_point,
//                 result.valid_flag,
//                 AuxData{result.camera_point, result.pose_r}
//             };
//         };

//         if constexpr (USE_UT) {
//             // load covariance
//             auto const quat =
//                 glm::fvec4(quat_ptr[0], quat_ptr[1], quat_ptr[2], quat_ptr[3]);
//             auto const scale = glm::fvec3(scale_ptr[0], scale_ptr[1], scale_ptr[2]);
//             auto const sqrt_covar =
//                 cugsplat::gaussian::quat_scale_to_scaled_rotmat(quat, scale);

//             // execute the function using unscented transform
//             auto const result = cugsplat::ut::transform<3, 2, AuxData>(
//                 point_world_to_image_fn, mu, sqrt_covar
//             );
//             if (!result.valid_flag) {
//                 break;
//             }
//             means2d = result.mu;
//             covar2d = result.covar;
//             auto const camera_point = result.aux.first;
//             depth = camera_point.z;

//         } else {
//             // execute the function
//             auto const [image_point, image_point_valid_flag, aux] =
//                 point_world_to_image_fn(mu);
//             if (!image_point_valid_flag) {
//                 break;
//             }
//             means2d = image_point;
//             auto const camera_point = std::get<0>(aux);
//             depth = camera_point.z;
//             auto const world_to_camera_R = std::get<1>(aux);

//             // project covariance from camera space to image space
//             glm::fmat3x2 J;
//             static_assert(
//                 CAMERA_TYPE != CameraType::DISTORTED_FISHEYE,
//                 "Jacobian for distorted fisheye is not implemented"
//             );
//             if constexpr (CAMERA_TYPE == CameraType::FISHEYE) {
//                 J = cugsplat::fisheye::project_jac(camera_point, focal_length);
//             } else if constexpr (CAMERA_TYPE == CameraType::PINHOLE) {
//                 J = cugsplat::pinhole::project_jac(camera_point, focal_length);
//             } else if constexpr (CAMERA_TYPE == CameraType::ORTHO) {
//                 J = cugsplat::orthogonal::project_jac(camera_point, focal_length);
//             } else if constexpr (CAMERA_TYPE == CameraType::DISTORTED_PINHOLE) {
//                 auto const &[J_, valid_flag_] = cugsplat::pinhole::project_jac(
//                     camera_point,
//                     focal_length,
//                     radial_coeffs,
//                     tangential_coeffs,
//                     thin_prism_coeffs
//                 );
//                 if (!valid_flag_) {
//                     break;
//                 }
//                 J = J_;
//             } else if constexpr (CAMERA_TYPE == CameraType::DISTORTED_FISHEYE) {
//                 // TODO: implement
//             }

//             // load covariance
//             auto const quat =
//                 glm::fvec4(quat_ptr[0], quat_ptr[1], quat_ptr[2], quat_ptr[3]);
//             auto const scale = glm::fvec3(scale_ptr[0], scale_ptr[1], scale_ptr[2]);
//             auto const covar = cugsplat::gaussian::quat_scale_to_covar(quat, scale);
//             // transform covariance to camera space, then to image space
//             auto const covar_c =
//                 cugsplat::se3::transform_covar(world_to_camera_R, covar);
//             covar2d = J * covar_c * glm::transpose(J);
//         }

//         // reach here mean_ptr valid
//         valid_flag = true;
//     } while (false);

//     return {means2d, depth, covar2d, valid_flag};
// }

// struct ProjectionBackwardResult {
//     glm::fvec3 v_means;
//     glm::fvec4 v_quats;
//     glm::fvec3 v_scales;
//     glm::fmat4 v_viewmats0;
//     glm::fmat4 v_viewmats1;
//     glm::fvec2 v_Ks;
// };

} // namespace cugsplat::impl
