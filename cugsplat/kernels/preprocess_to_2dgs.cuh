#include <stdint.h>
#include <glm/glm.hpp>

#include "kernels/preprocess.cuh"
#include "util/macros.cuh"

namespace cugsplat {

using namespace glm;

inline __device__ auto compute_radius(float opacity, fmat2 covar) -> fvec2 {
    constexpr float alpha_threshold = 1.0f / 255.0f;
    if (opacity < alpha_threshold) {
        return fvec2(0.0f, 0.0f);
    }

    // Compute opacity-aware bounding box.
    // https://arxiv.org/pdf/2402.00525 Section B.2
    float extend = 3.33f;
    extend = min(extend, sqrt(2.0f * __logf(opacity / alpha_threshold)));

    // compute tight rectangular bounding box (non differentiable)
    // https://arxiv.org/pdf/2402.00525
    auto const b = 0.5f * (covar[0][0] + covar[1][1]);
    auto const det = determinant(covar);
    auto const tmp = sqrtf(max(0.01f, b * b - det));
    auto const v1 = b + tmp; // larger eigenvalue
    auto const r1 = extend * sqrtf(v1);
    auto const radius_x = ceilf(min(extend * sqrtf(covar[0][0]), r1));
    auto const radius_y = ceilf(min(extend * sqrtf(covar[1][1]), r1));
    return fvec2(radius_x, radius_y);
}

struct DeviceGaussianOut2DGS {
    uint32_t n;
    uint32_t index;
    float* __restrict__ opacities;
    fvec2* __restrict__ means;
    fvec3* __restrict__ conics;
    float* __restrict__ depth;
    fvec2* __restrict__ radius;

    DEFINE_VALUE_SETGET(uint32_t, n)
    DEFINE_VALUE_SETGET(uint32_t, index)

    // ctx
    float opacity;
    fvec2 mean;
    fvec3 conic;
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
        if (params.filter_size > 0) {
            covar += mat2(params.filter_size);
            auto const det_blur = determinant(covar);
            opacity *= sqrtf(det_orig / det_blur);
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
        auto const preci = inverse(covar);
        this->conic = {preci[0][0], preci[1][1], preci[0][1]};
        this->depth = depth;
        this->radius = radius;
        return true;
    }

    inline __device__ void export() {
        this->opacities[index] = this->opacity;
        this->means[index] = this->mean;
        this->conics[index] = this->conic;
    }
};




} // namespace cugsplat

