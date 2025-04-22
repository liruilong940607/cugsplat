#include <stdint.h>
#include <glm/glm.hpp>

#include "preprocess/util.cuh"

namespace cugsplat::preprocess {

struct DeviceSimplePinholeCameraEWA {
    uint32_t n;
    uint32_t index;

    __device__ void set_index(uint32_t index) { this->index = index; }
    __device__ int get_n() const { return this->n; }

    // pointers to input buffer
    glm::fvec2* focal_lengths;
    glm::fvec2* principal_points;
    glm::fmat3* world_to_cameras_R;
    glm::fvec3* world_to_cameras_t;

    // cached values to avoid repeated memory accesses
    Maybe<glm::fvec2> focal_length;
    Maybe<glm::fvec2> principal_point;
    Maybe<glm::fmat3> world_to_camera_R;
    Maybe<glm::fvec3> world_to_camera_t;

    // ctx
    Maybe<glm::fvec3> camera_point;

    inline __device__ glm::fvec2 get_focal_lengths() { 
        if (!this->focal_length.has_value()) {
            this->focal_length.set(this->focal_lengths[index]);
        }
        return this->focal_length.get();
    }

    inline __device__ glm::fvec2 get_principal_point() { 
        if (!this->principal_point.has_value()) {
            this->principal_point.set(this->principal_points[index]);
        }
        return this->principal_point.get();
    }

    inline __device__ glm::fmat3 get_world_to_camera_R() { 
        if (!this->world_to_camera_R.has_value()) {
            this->world_to_camera_R.set(this->world_to_cameras_R[index]);
        }
        return this->world_to_camera_R.get();
    }

    inline __device__ glm::fvec3 get_world_to_camera_t() { 
        if (!this->world_to_camera_t.has_value()) {
            this->world_to_camera_t.set(this->world_to_cameras_t[index]);
        }
        return this->world_to_camera_t.get();
    }

    inline __device__ glm::fvec3 point_world_to_camera(const glm::fvec3& world_point) {
        if (!this->camera_point.has_value()) {
            auto const R = this->get_world_to_camera_R();
            auto const t = this->get_world_to_camera_t();
            auto const camera_point = R * world_point + t;
            this->camera_point.set(camera_point);
        }
        return this->camera_point.get();
    }

    inline __device__ glm::fvec2 point_world_to_image(const glm::fvec3& world_point) {
        auto const camera_point = this->point_world_to_camera(world_point);
        auto const focal_length = this->get_focal_lengths();
        auto const principal_point = this->get_principal_point();
        return glm::fvec2(
            camera_point.x / camera_point.z * focal_length.x + principal_point.x,
            camera_point.y / camera_point.z * focal_length.y + principal_point.y
        );
    }

    inline __device__ glm::fmat2 jacobian_world_to_image(const glm::fvec3& world_point) {
        auto const camera_point = this->point_world_to_camera(world_point);
        auto const camera_J = this->jacobian_camera_to_image(camera_point);
        auto const world_to_camera_R = this->get_world_to_camera_R();
        return camera_J * world_to_camera_R;
    }

    inline __device__ glm::fmat3x2 jacobian_camera_to_image(const glm::fvec3& camera_point) {
        auto const focal_length = this->get_focal_lengths();
        auto const x = camera_point.x;
        auto const y = camera_point.y;
        auto const rz = 1.0f / camera_point.z;
        auto const rz2 = rz * rz;
        // mat3x2 is 3 columns x 2 rows.
        auto const J = glm::fmat3x2(
            focal_length.x * rz,
            0.f, // 1st column
            0.f,
            focal_length.y * rz, // 2nd column
            -focal_length.x * x * rz2,
            -focal_length.y * y * rz2 // 3rd column
        );
    }

    inline __host__ void free() {
        cudaFree(this->focal_lengths);
        cudaFree(this->principal_points);
        cudaFree(this->world_to_cameras_R);
        cudaFree(this->world_to_cameras_t);
    }
};

} // namespace cugsplat::preprocess

