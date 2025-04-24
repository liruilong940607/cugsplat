#include <glm/glm.hpp>

#include "preprocess/util.cuh"

namespace gsplat::preprocess {

struct DevicePrimitiveOutImage2DGS {
    // pointers to output buffer
    float* opacities;
    glm::fvec2* means;
    glm::fvec3* conics;
    float* depths;
    glm::fvec2* radius;

    // parameters
    uint32_t render_width;
    uint32_t render_height;
    float near_plane;
    float far_plane;
    float margin_factor;
    float filter_size;

    // ctx: internal state to be written to output buffer
    float opacity;
    glm::fvec2 mean;
    glm::fvec3 conic;
    float depth;
    glm::fvec2 radii;

    template <class DeviceCameraModel, class DevicePrimitiveIn>
    inline __device__ bool preprocess(
        DeviceCameraModel &d_camera,
        DevicePrimitiveIn &d_gaussians_in
    ) {
        // Check: If the gaussian is outside the camera frustum, skip it
        auto const depth = d_gaussians_in.image_depth(d_camera);
        if (depth < near_plane || depth > far_plane) {
            return false;
        }

        // Compute the projected gaussian on the image plane
        auto [mean, covar, valid_flag] = 
            d_gaussians_in.world_to_image(d_camera);
        if (!valid_flag) {
            return false;
        }

        // Check: If the gaussian is outside the image plane, skip it
        auto const min_x = - margin_factor * render_width;
        auto const min_y = - margin_factor * render_height;
        auto const max_x = (1 + margin_factor) * render_width;
        auto const max_y = (1 + margin_factor) * render_height;
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
        if (filter_size > 0) {
            covar += glm::mat2(filter_size);
            auto const det_blur = glm::determinant(covar);
            opacity *= sqrtf(det_orig / det_blur);
        }
        
        // Compute the bounding box of this gaussian on the image plane
        auto const radii = solve_tight_radius(covar, opacity, 1.0f / 255.0f);

        // Check again if the gaussian is outside the image plane
        if (mean.x + radii.x < 0 || mean.x - radii.x > render_width ||
            mean.y + radii.y < 0 || mean.y - radii.y > render_height) {
            return false;
        }

        this->opacity = opacity;
        this->mean = mean;
        auto const preci = glm::inverse(covar);
        this->conic = {preci[0][0], preci[1][1], preci[0][1]};
        this->depth = depth;
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

} // namespace gsplat::preprocess

