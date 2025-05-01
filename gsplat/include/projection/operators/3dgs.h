#pragma once

#include <glm/glm.hpp>
#include <stdint.h>

#include "camera/model.h"
#include "core/types.h"
#include "gaussian/utils.h"

namespace gsplat {

struct PreprocessOperator3DGS {
    // pointers to output buffer
    float *opacity_ptr;
    glm::fvec2 *mean_ptr;
    glm::fvec3 *conic_ptr;
    float *depth_ptr;
    glm::fvec2 *radius_ptr;

    // parameters
    const float filter_size = 0.0f;
    const float alpha_threshold = 1.0f / 255.0f;

    // cache: internal state to be written to output buffer
    float opacity;
    glm::fvec2 mean;
    glm::fvec3 conic;
    float depth;
    glm::fvec2 radius;

    template <class CameraProjection, class CameraPose, class Gaussian>
    inline GSPLAT_HOST_DEVICE bool forward(
        CameraModel<CameraProjection, CameraPose> &camera, Gaussian &gaussian
    ) {
        // Compute projected center.
        auto const world_point = gaussian.get_mean();
        auto const
            &[camera_point, image_point, point_valid_flag, pose_r, pose_t] =
                camera._world_point_to_image_point(world_point);
        if (!point_valid_flag) {
            return false;
        }

        // Compute projected covariance.
        auto const quat = gaussian.get_quat();
        auto const scale = gaussian.get_scale();
        auto const world_covar = quat_scale_to_covar(quat, scale);
        auto [image_covar, covar_valid_flag] =
            camera._world_covar_to_image_covar(
                camera_point, world_covar, pose_r, pose_t
            );
        if (!covar_valid_flag) {
            return false;
        }

        // Fetch the opacity
        auto opacity = gaussian.get_opacity();

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
            return false;
        }

        // Compute the bounding box of this gaussian on the image plane
        auto const radius =
            solve_tight_radius(image_covar, opacity, alpha_threshold);

        // Check again if the gaussian is outside the image plane
        auto const &[render_width, render_height] = camera.resolution;
        if (image_point.x + radius.x < 0 ||
            image_point.x - radius.x > render_width ||
            image_point.y + radius.y < 0 ||
            image_point.y - radius.y > render_height) {
            return false;
        }

        auto const preci = glm::inverse(image_covar);
        auto const conic = glm::fvec3{preci[0][0], preci[1][1], preci[0][1]};

        this->opacity = opacity;
        this->mean = image_point;
        this->conic = conic;
        this->depth = camera_point.z;
        this->radius = radius;
        return true;
    }

    inline GSPLAT_HOST_DEVICE void write_to_buffer(uint32_t index) {
        this->opacity_ptr[index] = this->opacity;
        this->mean_ptr[index] = this->mean;
        this->conic_ptr[index] = this->conic;
        this->depth_ptr[index] = this->depth;
        this->radius_ptr[index] = this->radius;
    }
};

} // namespace gsplat