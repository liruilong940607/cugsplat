#include <stdint.h>
#include <glm/glm.hpp>

#include "preprocess/util.cuh"

namespace cugsplat::preprocess {

struct DevicePrimitiveInWorld3DGS {
    uint32_t n;
    uint32_t index;

    __device__ void set_index(uint32_t index) { this->index = index; }
    __device__ int get_n() const { return this->n; }

    // pointers to input buffer
    float* __restrict__ opacities;
    glm::fvec3* __restrict__ means;
    glm::fvec4* __restrict__ quats;
    glm::fvec3* __restrict__ scales;

    // cached values to avoid repeated memory accesses
    Maybe<float> opacity;
    Maybe<glm::fvec3> mean;
    Maybe<glm::fvec4> quat;
    Maybe<glm::fvec3> scale;

    inline __device__ float get_opacity() { 
        if (!this->opacity.has_value()) {
            this->opacity.set(opacities[index]);
        }
        return this->opacity.get();
    }

    inline __device__ glm::fvec3 get_mean() { 
        if (!this->mean.has_value()) {
            this->mean.set(means[index]);
        }
        return this->mean.get();
    }

    inline __device__ glm::fvec4 get_quat() { 
        if (!this->quat.has_value()) {
            this->quat.set(quats[index]);
        }
        return this->quat.get();
    }

    inline __device__ glm::fvec3 get_scale() { 
        if (!this->scale.has_value()) {
            this->scale.set(scales[index]);
        }
        return this->scale.get();
    }

    template <class DeviceCameraModel>
    inline __device__ float image_depth(const DeviceCameraModel d_camera) {
        auto const world_point = this->get_mean();
        auto const camera_point = d_camera.point_world_to_camera(world_point);
        return camera_point.z;
    }

    template <class DeviceCameraModel>
    inline __device__ auto world_to_image(
        const DeviceCameraModel d_camera
    ) -> std::pair<glm::fvec2, glm::fmat2, bool> {
        auto const world_point = this->get_mean();
        auto const world_covar = quat_scale_to_covar(
            this->get_quat(), this->get_scale()
        );
        auto const image_point = d_camera.point_world_to_image(world_point);
        auto const J = d_camera.jacobian_world_to_image(world_point);
        auto const image_covar = J * world_covar * transpose(J);
        return {image_point, image_covar, true};
    }
};

} // namespace cugsplat::preprocess

