#pragma once

#include <glm/glm.hpp>

#include "utils/types.h"
#include "camera/model.h"

namespace gsplat::device {

struct DevicePrimitiveOutImage2DGS {
    // pointers to output buffer
    float *opacities;
    glm::fvec2 *means;
    glm::fvec3 *conics;
    float *depths;
    glm::fvec2 *radius;

    // parameters
    float filter_size;

    // ctx: internal state to be written to output buffer
    float opacity;
    glm::fvec2 mean;
    glm::fvec3 conic;
    float depth;
    glm::fvec2 radii;

    template <class CameraProjection, class CameraPose, class DevicePrimitiveIn> 
    inline __device__ bool
    preprocess(
        CameraModel<CameraProjection, CameraPose> &d_camera, 
        DevicePrimitiveIn &d_gaussians_in) {
        
        // Compute projected center.
        auto const world_point = d_gaussians_in.get_mean();
        auto const &[camera_point, image_point, point_valid_flag, pose] =
            d_camera._world_to_camera_and_image_shutter(world_point);
        if (!point_valid_flag) {
            return false;
        }

        // Compute projected covariance.
        auto const world_covar = d_gaussians_in.get_covar();
        auto const &[image_covar, covar_valid_flag] = 
            d_camera._world_covar_to_image_covar(camera_point, world_covar, pose);
        if (!covar_valid_flag) {
            return false;
        }

        // Check: If the covariance matrix is not positive definite, skip it
        auto const det_orig = glm::determinant(image_covar);
        if (det_orig < 0) {
            return false;
        }

        // Fetch the opacity
        auto opacity = d_gaussians_in.get_opacity();

        // Apply anti-aliasing filter
        if (filter_size > 0) {
            image_covar += glm::fmat2(filter_size);
            auto const det_blur = glm::determinant(image_covar);
            opacity *= sqrtf(det_orig / det_blur);
        }

        // Compute the bounding box of this gaussian on the image plane
        auto const radii = solve_tight_radius(image_covar, opacity, 1.0f / 255.0f);

        // Check again if the gaussian is outside the image plane
        auto const &[render_width, render_height] = d_camera.resolution;
        if (mean.x + radii.x < 0 || mean.x - radii.x > render_width ||
            mean.y + radii.y < 0 || mean.y - radii.y > render_height) {
            return false;
        }

        this->opacity = opacity;
        this->mean = mean;
        auto const preci = glm::inverse(image_covar);
        this->conic = {preci[0][0], preci[1][1], preci[0][1]};
        this->depth = camera_point.z;
        this->radii = radii;
        return true;
    }

    inline __device__ void write_to_buffer(uint32_t index) {
        this->opacities[index] = this->opacity;
        this->means[index] = this->mean;
        this->conics[index] = this->conic;
        this->depths[index] = this->depth;
        this->radius[index] = this->radii;
    }

    inline __host__ void free() {
        cudaFree(this->opacities);
        cudaFree(this->means);
        cudaFree(this->conics);
        cudaFree(this->depths);
        cudaFree(this->radius);
    }
};

} // namespace gsplat::device
