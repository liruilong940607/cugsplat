#pragma once

#include <glm/glm.hpp>
#include <stdint.h>

#include "camera/model.h"
#include "core/types.h"
#include "gaussian/utils.h"

namespace curend {

struct DevicePrimitiveIn3DGS {
    const glm::fvec3 *mean_ptr;
    const glm::fvec4 *quat_ptr;
    const glm::fvec3 *scale_ptr;
    const float *opacity_ptr;

    GSPLAT_HOST_DEVICE inline void shift_ptr(size_t index) {
        mean_ptr += index;
        quat_ptr += index;
        scale_ptr += index;
        opacity_ptr += index;
    }
};

struct PrimitiveOut3DGS {
    float opacity;
    glm::fvec2 mean;
    glm::fvec3 conic;
    float depth;
    glm::fvec2 radius;
};

struct DevicePrimitiveOut3DGS {
    float *opacity_ptr;
    glm::fvec2 *mean_ptr;
    glm::fvec3 *conic_ptr;
    float *depth_ptr;
    glm::fvec2 *radius_ptr;

    GSPLAT_HOST_DEVICE inline void shift_ptr(size_t index) {
        opacity_ptr += index;
        mean_ptr += index;
        conic_ptr += index;
        depth_ptr += index;
        radius_ptr += index;
    }

    GSPLAT_HOST_DEVICE inline void set_value(const PrimitiveOut3DGS &primitive) {
        *opacity_ptr = primitive.opacity;
        *mean_ptr = primitive.mean;
        *conic_ptr = primitive.conic;
        *depth_ptr = primitive.depth;
        *radius_ptr = primitive.radius;
    }
};

struct PreprocessOperator3DGS {
    // parameters
    const float filter_size = 0.3f;
    const float alpha_threshold = 1.0f / 255.0f;

    template <class CameraProjection, class RotationType>
    inline GSPLAT_HOST_DEVICE auto forward(
        CameraModel<CameraProjection, RotationType> &d_camera,
        const DevicePrimitiveIn3DGS &d_gaussian
    ) -> std::pair<PrimitiveOut3DGS, bool> {
        // Compute projected center.
        auto const world_point = *d_gaussian.mean_ptr;
        auto const &[camera_point, image_point, point_valid_flag, pose_r, pose_t] =
            d_camera._world_point_to_image_point(world_point);
        if (!point_valid_flag) {
            return {PrimitiveOut3DGS{}, false};
        }

        // Compute projected covariance.
        auto const quat = *d_gaussian.quat_ptr;
        auto const scale = *d_gaussian.scale_ptr;
        auto const world_covar = quat_scale_to_covar(quat, scale);
        auto [image_covar, covar_valid_flag] = d_camera._world_covar_to_image_covar(
            camera_point, world_covar, pose_r, pose_t
        );
        if (!covar_valid_flag) {
            return {PrimitiveOut3DGS{}, false};
        }

        // Fetch the opacity
        auto opacity = *d_gaussian.opacity_ptr;

        // Apply anti-aliasing filter
        float det;
        if (filter_size > 0) {
            auto const det_orig = glm::determinant(image_covar);
            image_covar += glm::fmat2(filter_size);
            det = glm::determinant(image_covar);
            opacity *= sqrtf(det_orig / det);
        } else {
            det = glm::determinant(image_covar);
        }
        if (det < 0) {
            return {PrimitiveOut3DGS{}, false};
        }

        // Compute the bounding box of this gaussian on the image plane
        auto const radius = solve_tight_radius(image_covar, opacity, alpha_threshold);

        // Check again if the gaussian is outside the image plane
        auto const &[render_width, render_height] = d_camera.resolution;
        if (image_point.x + radius.x < 0 || image_point.x - radius.x > render_width ||
            image_point.y + radius.y < 0 || image_point.y - radius.y > render_height) {
            return {PrimitiveOut3DGS{}, false};
        }

        auto const preci = glm::inverse(image_covar);
        auto const conic = glm::fvec3{preci[0][0], preci[1][1], preci[0][1]};

        auto const primitive_out =
            PrimitiveOut3DGS{opacity, image_point, conic, camera_point.z, radius};
        return {primitive_out, true};
    }
};

} // namespace curend