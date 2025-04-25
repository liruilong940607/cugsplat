#pragma once

#include <stdint.h>
#include <glm/glm.hpp>

#include "utils/camera_projection.h"

namespace gsplat {

struct DeviceOpencvPinholeProjection : OpencvPinholeProjection {
    uint32_t n;
    uint32_t index;

    __device__ void set_index(uint32_t index) { this->index = index; }
    __device__ int get_n() const { return this->n; }

    // pointers to input buffer
    glm::fvec2* focal_length_ptr;
    // glm::fvec2* principal_point_ptr;
    // std::array<float, 6>* radial_coeffs_ptr;
    // std::array<float, 2>* tangential_coeffs_ptr;
    // std::array<float, 4>* thin_prism_coeffs_ptr;

    // cached values to avoid repeated memory accesses
    Maybe<glm::fvec2> focal_length2;
    // Maybe<glm::fvec2> principal_point;
    // Maybe<std::array<float, 6>> radial_coeff;
    // Maybe<std::array<float, 2>> tangential_coeff;
    // Maybe<std::array<float, 4>> thin_prism_coeff;

    GSPLAT_HOST_DEVICE glm::fvec2 get_focal_length2() {
        if (!this->focal_length2.has_value()) {
            this->focal_length2.set(this->focal_length_ptr[index]);
        }
        return this->focal_length2.get();
    }
        
//     // GSPLAT_HOST_DEVICE glm::fvec2 get_principal_point() const {return principal_point;}
//     // GSPLAT_HOST_DEVICE std::array<float, 6> get_radial_coeffs() const {return radial_coeffs;}
//     // GSPLAT_HOST_DEVICE std::array<float, 2> get_tangential_coeffs() const {return tangential_coeffs;}
//     // GSPLAT_HOST_DEVICE std::array<float, 4> get_thin_prism_coeffs() const {return thin_prism_coeffs;}

};










// struct DeviceOpencvPinholeCameraModel {
//     uint32_t n;
//     uint32_t index;

//     __device__ void set_index(uint32_t index) { this->index = index; }
//     __device__ int get_n() const { return this->n; }

//     std::array<uint32_t, 2> resolution;

//     // pointers to input buffer
//     glm::fvec2* focal_lengths;
//     glm::fvec2* principal_points;
//     glm::fmat3* world_to_cameras_R1;
//     glm::fvec3* world_to_cameras_t1;
//     glm::fmat3* world_to_cameras_R2;
//     glm::fvec3* world_to_cameras_t2;
    

//     // cached values to avoid repeated memory accesses
//     Maybe<glm::fvec2> focal_length;
//     Maybe<glm::fvec2> principal_point;
//     Maybe<glm::fmat3> world_to_camera_R;
//     Maybe<glm::fvec3> world_to_camera_t;

//     inline __device__ glm::fvec2 get_focal_lengths() { 
//         if (!this->focal_length.has_value()) {
//             this->focal_length.set(this->focal_lengths[index]);
//         }
//         return this->focal_length.get();
//     }

//     inline __device__ glm::fvec2 get_principal_point() { 
//         if (!this->principal_point.has_value()) {
//             this->principal_point.set(this->principal_points[index]);
//         }
//         return this->principal_point.get();
//     }

//     inline __device__ glm::fmat3 get_world_to_camera_R() { 
//         if (!this->world_to_camera_R.has_value()) {
//             this->world_to_camera_R.set(this->world_to_cameras_R[index]);
//         }
//         return this->world_to_camera_R.get();
//     }

//     inline __device__ glm::fvec3 get_world_to_camera_t() { 
//         if (!this->world_to_camera_t.has_value()) {
//             this->world_to_camera_t.set(this->world_to_cameras_t[index]);
//         }
//         return this->world_to_camera_t.get();
//     }
// }

}