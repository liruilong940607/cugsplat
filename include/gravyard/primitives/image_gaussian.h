// #pragma once

// #include <cstdint>
// #include <glm/glm.hpp>

// #include "tinyrend/core/macros.h" // for GSPLAT_HOST_DEVICE
// #include "tinyrend/rasterization/kernel.cuh"

// namespace tinyrend::rasterization {

// template <size_t FEATURE_DIM>
// struct ImageGaussians : public PrimitiveBase<ImageGaussians<FEATURE_DIM>> {
//     /*
//     A collection of Gaussian primitives defined on a image plane.
//     */

//     // Pointers to the device memory (shared across all threads)
//     glm::fvec2 *mu;     // [N, 2]
//     glm::fvec3 *conics; // [N, 3]
//     float *features;    // [N, FEATURE_DIM]

//     // Outputs
//     float accum_features[FEATURE_DIM];
//     float *buffer_features; // [n_images, image_h, image_w, FEATURE_DIM]

//     inline GSPLAT_HOST_DEVICE static auto sm_size_per_primitive_impl() -> uint32_t
//     {
//         return sizeof(glm::fvec2) + sizeof(glm::fvec3);
//     }

//     inline GSPLAT_HOST_DEVICE auto initialize_impl() -> bool {
//         // Initialize the accum_features
// #pragma unroll
//         for (uint32_t i = 0; i < FEATURE_DIM; i++) {
//             accum_features[i] = 0.0f;
//         }
//         return true;
//     }

//     inline GSPLAT_HOST_DEVICE auto
//     load_to_shared_memory_impl(uint32_t sm_id, uint32_t global_id) -> void {
//         glm::fvec2 *sm_ptr_mu = reinterpret_cast<glm::fvec2 *>(this->sm_ptr);
//         glm::fvec3 *sm_ptr_conics =
//             reinterpret_cast<glm::fvec3 *>(&sm_ptr_mu[this->threads_per_block]);
//         sm_ptr_mu[sm_id] = mu[global_id];
//         sm_ptr_conics[sm_id] = conics[global_id];
//     }

//     inline GSPLAT_HOST_DEVICE auto get_light_attenuation_impl(uint32_t sm_id
//     ) -> float {
//         glm::fvec2 *sm_ptr_mu = reinterpret_cast<glm::fvec2 *>(this->sm_ptr);
//         glm::fvec3 *sm_ptr_conics =
//             reinterpret_cast<glm::fvec3 *>(&sm_ptr_mu[this->threads_per_block]);
//         auto const mu = sm_ptr_mu[sm_id];
//         auto const conic = sm_ptr_conics[sm_id];

//         auto const dx = this->pixel_x - mu.x;
//         auto const dy = this->pixel_y - mu.y;
//         auto const sigma =
//             0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
//         return exp(-sigma);
//     }

//     inline GSPLAT_HOST_DEVICE auto
//     accumulate_impl(float T, float alpha, uint32_t primitive_id) -> void {
// #pragma unroll
//         for (uint32_t i = 0; i < FEATURE_DIM; i++) {
//             accum_features[i] += T * alpha * features[primitive_id * FEATURE_DIM +
//             i];
//         }
//     }

//     inline GSPLAT_HOST_DEVICE auto write_to_buffer_impl() -> void {
//         auto const buffer_features_offset =
//             this->image_id * this->image_h * this->image_w * FEATURE_DIM +
//             this->pixel_y * this->image_w * FEATURE_DIM + this->pixel_x *
//             FEATURE_DIM;
// #pragma unroll
//         for (uint32_t i = 0; i < FEATURE_DIM; i++) {
//             buffer_features[buffer_features_offset + i] = accum_features[i];
//         }
//     }
// };

// } // namespace tinyrend::rasterization