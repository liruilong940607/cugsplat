#include <stdint.h>
#include <glm/glm.hpp>

#include "kernels/preprocess.cuh"
#include "util/macros.cuh"

namespace cugsplat {

using namespace glm;

struct DeviceGaussianOutWorldDGS {
    uint32_t n;
    uint32_t index;
    float* __restrict__ opacities;
    fvec2* __restrict__ means;
    float* __restrict__ triuLs; // [6]
    float* __restrict__ depths;
    fvec2* __restrict__ radius;

    DEFINE_VALUE_SETGET(uint32_t, n)
    DEFINE_VALUE_SETGET(uint32_t, index)

    // ctx
    float opacity;
    fvec2 mean;
    float triuL[6];
    float depth;
    fvec2 radius;

    template <class DeviceCameraModel, class DeviceGaussianIn>
    inline __device__ bool preprocess(
        const DeviceCameraModel d_camera,
        const DeviceGaussianIn d_gaussians_in,
        const PreprocessParameters& params
    ) {
        // Check: If the gaussian is outside the camera frustum, skip it
        auto const depth = d_camera.distance_to_image(d_gaussians_in);
        if (depth < params.near_plane || depth > params.far_plane) {
            return false;
        }

        // Compute the projected gaussian on the image plane
        auto &[mean, covar, valid_flag] = 
            d_camera.gaussian_world_to_image(d_gaussians_in);
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
        auto const det_orig = determinant(covar);
        if (det_orig < 0) {
            return false;
        }

        // Fetch the opacity
        auto opacity = d_gaussians_in.get_opacity();

        // Apply anti-aliasing filter
        fmat3 L;
        if (params.filter_size > 0) {
            // TODO: implement anti-aliasing filter
        }

        // Compute the bounding box of this gaussian on the image plane
        auto const radius = compute_radius(opacity, covar);

        // Check again if the gaussian is outside the image plane
        if (mean.x - radius.x < 0 || mean.x + radius.x > params.render_width ||
            mean.y - radius.y < 0 || mean.y + radius.y > params.render_height) {
            return false;
        }

        this->opacity = opacity;
        this->mean = mean;
        this->triuL[0] = L[0][0];
        this->triuL[1] = L[0][1];
        this->triuL[2] = L[0][2];
        this->triuL[3] = L[1][1];
        this->triuL[4] = L[1][2];
        this->triuL[5] = L[2][2];
        this->depth = depth;
        this->radius = radius;
        return true;
    }

    inline __device__ void export() {
        this->opacities[index] = this->opacity;
        this->means[index] = this->mean;
        #pragma unroll
        for (int i = 0; i < 6; ++i)
            this->triuLs[index * 6 + i] = this->triuL[i];
        this->depths[index] = this->depth;
        this->radius[index] = this->radius;
    }
};




} // namespace cugsplat

