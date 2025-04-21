#include <stdint.h>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>

#include "kernels/preprocess.cuh"
#include "util/macros.cuh"

namespace cugsplat {

using namespace glm;

struct DeviceGaussianOutWorld2DGS {
    uint32_t n;
    uint32_t index;
    fmat3* __restrict__ transforms;
    float* __restrict__ depths;
    fvec2* __restrict__ centers;
    fvec2* __restrict__ radius;

    DEFINE_VALUE_SETGET(uint32_t, n)
    DEFINE_VALUE_SETGET(uint32_t, index)

    // ctx
    fmat3 transform;
    float depth;
    fvec2 center;
    fvec2 radius;

    template <class DeviceCameraModel, class DeviceGaussianIn>
    inline __device__ bool preprocess(
        const DeviceCameraModel d_camera,
        const DeviceGaussianIn d_gaussians_in,
        const PreprocessParameters& params
    ) {
        // Check: If the gaussian is outside the camera frustum, skip it
        auto const depth = d_gaussians_in.depth_to_image(d_camera);
        if (depth < params.near_plane || depth > params.far_plane) {
            return false;
        }

        // Compute the projected gaussian on the image plane
        // KWH is 3x2 matrix; mean is 3D vector
        auto &[mean, KWH, valid_flag] = 
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

        // Compute aabb
        auto const M = transpose(fmat3(KWH[0], KWH[1], mean));
        auto const M0 = M[0], M1 = M[1], M2 = M[2];
        auto const temp_point = fvec3(1.0f, 1.0f, -1.0f);
        auto const distance = compAdd(temp_point * M2 * M2);
        if (distance == 0.0f) {
            return false;
        }
        auto const f = (1.0f / distance) * temp_point;
        auto const center = fvec2(compAdd(f * M0 * M2), compAdd(f * M1 * M2));
        auto const temp = fvec2(compAdd(f * M0 * M0), compAdd(f * M1 * M1));
        auto const half_extend = center * center - temp;
        auto const radius = 3.33f * glm::sqrt(glm::max(fvec2(1e-4f), half_extend));
    
        // Check again if the gaussian is outside the image plane
        if (center.x - radius.x < 0 || center.x + radius.x > params.render_width ||
            center.y - radius.y < 0 || center.y + radius.y > params.render_height) {
            return false;
        }

        this->opacity = opacity;
        this->center = center;
        this->transform = M;
        this->depth = depth;
        this->radius = radius;
        return true;
    }

    inline __device__ void export() {
        this->centers[index] = this->center;
        this->transforms[index] = this->transform;
        this->depths[index] = this->depth;
        this->radius[index] = this->radius;
    }
};

} // namespace cugsplat

