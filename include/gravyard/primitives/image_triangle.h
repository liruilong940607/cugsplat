// #pragma once

// #include <cstdint>
// #include <glm/glm.hpp>

// #include "tinyrend/core/macros.h" // for GSPLAT_HOST_DEVICE
// #include "tinyrend/rasterization/base.cuh"

// namespace tinyrend::rasterization {

// struct ImageTriangles : public PrimitiveBase<ImageTriangles> {
//     /*
//     A collection of triangle primitives defined on a image plane.
//     */

//     // Pointers to the device memory (shared across all threads)
//     glm::fvec2 *v0; // [N, 2]
//     glm::fvec2 *v1; // [N, 2]
//     glm::fvec2 *v2; // [N, 2]

//     // Per-thread data
//     uint32_t _image_id;
//     uint32_t _pixel_x;
//     uint32_t _pixel_y;
//     void *_sm_ptr;
//     uint32_t _sm_n_primitives;

//     inline GSPLAT_HOST_DEVICE auto initialize_impl(
//         uint32_t image_id,
//         uint32_t pixel_x,
//         uint32_t pixel_y,
//         void *sm_ptr,
//         uint32_t sm_n_primitives
//     ) -> bool {
//         _image_id = image_id;
//         _pixel_x = pixel_x;
//         _pixel_y = pixel_y;
//         _sm_ptr = sm_ptr;
//         _sm_n_primitives = sm_n_primitives;
//         return true;
//     }

//     inline GSPLAT_HOST_DEVICE static auto sm_size_per_primitive_impl() -> uint32_t
//     {
//         return sizeof(glm::fvec2) * 3;
//     }

//     inline GSPLAT_HOST_DEVICE auto
//     load_to_shared_memory_impl(uint32_t sm_id, uint32_t global_id) -> void {
//         glm::fvec2 *sm_ptr_v0 = reinterpret_cast<glm::fvec2 *>(_sm_ptr);
//         glm::fvec2 *sm_ptr_v1 =
//             reinterpret_cast<glm::fvec2 *>(&sm_ptr_v0[_sm_n_primitives]);
//         glm::fvec2 *sm_ptr_v2 =
//             reinterpret_cast<glm::fvec2 *>(&sm_ptr_v1[_sm_n_primitives]);
//         sm_ptr_v0[sm_id] = v0[global_id];
//         sm_ptr_v1[sm_id] = v1[global_id];
//         sm_ptr_v2[sm_id] = v2[global_id];
//     }

//     inline GSPLAT_HOST_DEVICE auto get_light_attenuation_impl(uint32_t sm_id
//     ) -> float {
//         glm::fvec2 *sm_ptr_v0 = reinterpret_cast<glm::fvec2 *>(_sm_ptr);
//         glm::fvec2 *sm_ptr_v1 =
//             reinterpret_cast<glm::fvec2 *>(&sm_ptr_v0[_sm_n_primitives]);
//         glm::fvec2 *sm_ptr_v2 =
//             reinterpret_cast<glm::fvec2 *>(&sm_ptr_v1[_sm_n_primitives]);
//         auto const v0 = sm_ptr_v0[sm_id];
//         auto const v1 = sm_ptr_v1[sm_id];
//         auto const v2 = sm_ptr_v2[sm_id];

//         // Calculate barycentric coordinates
//         auto const A = v0;
//         auto const B = v1;
//         auto const C = v2;
//         auto const P = glm::fvec2(_pixel_x, _pixel_y);
//         float u = ((B.y - C.y) * (P.x - C.x) + (C.x - B.x) * (P.y - C.y)) /
//                   ((B.y - C.y) * (A.x - C.x) + (C.x - B.x) * (A.y - C.y));
//         float v = ((C.y - A.y) * (P.x - C.x) + (A.x - C.x) * (P.y - C.y)) /
//                   ((B.y - C.y) * (A.x - C.x) + (C.x - B.x) * (A.y - C.y));
//         float w = 1.0f - u - v;

//         // If point is outside triangle, return 0
//         if (u < 0.0f || v < 0.0f || w < 0.0f) {
//             return 0.0f;
//         }

//         // Calculate gradient
//         float gradient = 3.0f * std::min(std::min(u, v), w);
//         return gradient;
//     }

//     inline GSPLAT_HOST_DEVICE auto accumulate_impl(float T, float alpha) -> void {
//         // Do nothing
//     }

//     inline GSPLAT_HOST_DEVICE auto write_to_buffer_impl(uint32_t offset_pixel) ->
//     void {
//         // Do nothing
//     }
// };

// } // namespace tinyrend::rasterization