#include <glm/glm.hpp>

#include "preprocess/util.cuh"

namespace cugsplat::preprocess {

struct DeviceGaussianOutImage2DGS {
    // pointers to output buffer
    float* __restrict__ opacities;
    fvec2* __restrict__ means;
    fvec3* __restrict__ conics;
    float* __restrict__ depths;
    fvec2* __restrict__ radius;

    // parameters
    uint32_t render_width;
    uint32_t render_height;
    float near_plane;
    float far_plane;
    float margin_factor;
    float filter_size;

    // ctx: internal state to be written to output buffer
    float opacity;
    fvec2 mean;
    fvec3 conic;
    float depth;
    fvec2 radius;

    template <class DeviceCameraModel, class DeviceGaussianIn>
    inline __device__ bool preprocess(
        const DeviceCameraModel d_camera,
        const DeviceGaussianIn d_gaussians_in,
        const Parameters& params
    ) {
        // Check: If the gaussian is outside the camera frustum, skip it
        auto const depth = d_gaussians_in.image_depth(d_camera);
        if (depth < params.near_plane || depth > params.far_plane) {
            return false;
        }

        // Compute the projected gaussian on the image plane
        auto &[mean, covar, valid_flag] = 
            d_gaussians_in.world_to_image(d_camera);
        if (!valid_flag) {
            return false;
        }

        // Check: If the gaussian is outside the image plane, skip it
        auto const min_x = - params.margin_factor * params.render_width;
        auto const min_y = - params.margin_factor * params.render_height;
        auto const max_x = (1 + params.margin_factor) * params.render_width;
        auto const max_y = (1 + params.margin_factor) * params.render_height;
        if (mean.x < min_x || mean.x > max_x || mean.y < min_y || mean.y > max_y) {
            return false;
        }

        // Check: If the covariance matrix is not positive definite, skip it
        auto const det_orig = glm::determinant(covar);
        if (det_orig < 0) {
            return false;
        }

        // Fetch the opacity
        auto opacity = d_gaussians_in.get_opacity();

        // Apply anti-aliasing filter
        if (params.filter_size > 0) {
            covar += mat2(params.filter_size);
            auto const det_blur = glm::determinant(covar);
            opacity *= sqrtf(det_orig / det_blur);
        }

        // Compute the bounding box of this gaussian on the image plane
        auto const radius = solve_tight_radius(covar, opacity, 1.0f / 255.0f);

        // Check again if the gaussian is outside the image plane
        if (mean.x - radius.x < 0 || mean.x + radius.x > params.render_width ||
            mean.y - radius.y < 0 || mean.y + radius.y > params.render_height) {
            return false;
        }

        this->opacity = opacity;
        this->mean = mean;
        auto const preci = glm::inverse(covar);
        this->conic = {preci[0][0], preci[1][1], preci[0][1]};
        this->depth = depth;
        this->radius = radius;
        return true;
    }

    inline __device__ void export(uint32_t index) {
        this->opacities[index] = this->opacity;
        this->means[index] = this->mean;
        this->conics[index] = this->conic;
        this->depths[index] = this->depth;
        this->radius[index] = this->radius;
    }
};

} // namespace cugsplat::preprocess

