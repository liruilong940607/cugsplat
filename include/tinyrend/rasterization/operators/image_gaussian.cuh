#pragma once

#include <cooperative_groups.h>
#include <cstdint>
#include <glm/glm.hpp>

#include "tinyrend/core/warp.cuh"
#include "tinyrend/rasterization/kernel.cuh"

namespace tinyrend::rasterization {

namespace cg = cooperative_groups;

inline __device__ auto evaluate_light_attenuation(
    float opacity, glm::fvec2 mean, glm::fvec3 conic, float pixel_x, float pixel_y
) -> float {
    auto const dx = pixel_x - mean.x;
    auto const dy = pixel_y - mean.y;
    auto const sigma =
        0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
    auto const alpha = opacity * __expf(-sigma);
    return alpha;
}

template <size_t FEATURE_DIM>
struct ImageGaussianRasterizeKernelForwardOperator
    : BaseRasterizeKernelOperator<
          ImageGaussianRasterizeKernelForwardOperator<FEATURE_DIM>> {

    // Inputs
    float *opacity_ptr;    // [N, 1]
    glm::fvec2 *mean_ptr;  // [N, 2]
    glm::fvec3 *conic_ptr; // [N, 3]
    float *feature_ptr; // [N, FEATURE_DIM] (e.g., 3 for RGB or 256 for neural features)

    // Outputs
    int32_t *render_last_index_ptr; // [n_images, image_height, image_width, 1]
    float *render_alpha_ptr;        // [n_images, image_height, image_width, 1]
    float *render_feature_ptr; // [n_images, image_height, image_width, FEATURE_DIM]

    // Internal variables
    float _expected_feature[FEATURE_DIM] = {0.0f};
    float _T = 1.0f;          // current transmittance
    int32_t _last_index = -1; // the index of intersections ([n_isects]) for the last
                              // one being rasterized. -1 means no intersection.

    // Configs
    const float skip_if_alpha_smaller_than = 1.0f / 255.0f;
    const float maximum_alpha = 0.999f; // For backward numerical stability.
    const float stop_if_next_trans_smaller_than =
        1e-4f; // For backward numerical stability.

    static inline __host__ auto sm_size_per_primitive_impl() -> uint32_t {
        // cache the opacity, mean, conic, and primitive_id
        return sizeof(float) + sizeof(glm::fvec2) + sizeof(glm::fvec3) +
               sizeof(uint32_t);
    }

    inline __device__ auto initialize_impl() -> bool { return true; }

    inline __device__ auto primitive_preprocess_impl(uint32_t primitive_id) -> void {
        // cache data to shared memory
        auto const sm_opacity_ptr = reinterpret_cast<float *>(this->sm_ptr);
        auto const sm_mean_ptr =
            reinterpret_cast<glm::fvec2 *>(&sm_opacity_ptr[this->n_threads_per_block]);
        auto const sm_conic_ptr =
            reinterpret_cast<glm::fvec3 *>(&sm_mean_ptr[this->n_threads_per_block]);
        auto const sm_primitive_id_ptr =
            reinterpret_cast<uint32_t *>(&sm_conic_ptr[this->n_threads_per_block]);
        sm_opacity_ptr[this->thread_rank] = this->opacity_ptr[primitive_id];
        sm_mean_ptr[this->thread_rank] = this->mean_ptr[primitive_id];
        sm_conic_ptr[this->thread_rank] = this->conic_ptr[primitive_id];
        sm_primitive_id_ptr[this->thread_rank] = primitive_id;
    }

    template <class WarpT>
    inline __device__ auto
    rasterize_impl(uint32_t batch_start, uint32_t t, WarpT &warp) -> bool {
        // load data from shared memory
        auto const sm_opacity_ptr = reinterpret_cast<float *>(this->sm_ptr);
        auto const sm_mean_ptr =
            reinterpret_cast<glm::fvec2 *>(&sm_opacity_ptr[this->n_threads_per_block]);
        auto const sm_conic_ptr =
            reinterpret_cast<glm::fvec3 *>(&sm_mean_ptr[this->n_threads_per_block]);
        auto const sm_primitive_id_ptr =
            reinterpret_cast<uint32_t *>(&sm_conic_ptr[this->n_threads_per_block]);
        auto const opacity = sm_opacity_ptr[t];
        auto const mean = sm_mean_ptr[t];
        auto const conic = sm_conic_ptr[t];

        // compute the light attenuation
        auto const alpha =
            min(this->maximum_alpha,
                evaluate_light_attenuation(
                    opacity, mean, conic, this->pixel_x, this->pixel_y
                ));
        // skip if the alpha is smaller than the threshold
        if (alpha < this->skip_if_alpha_smaller_than) {
            return false; // continue
        }

        // check if I should stop
        auto const next_T = this->_T * (1.0f - alpha);
        if (next_T < this->stop_if_next_trans_smaller_than) {
            return true; // terminate
        }

        // weights for expectation calculation
        auto const weight = alpha * this->_T;

        // accumulate the expectation of the feature
        auto const primitive_id = sm_primitive_id_ptr[t];
#pragma unroll
        for (size_t i = 0; i < FEATURE_DIM; i++) {
            this->_expected_feature[i] +=
                weight * this->feature_ptr[primitive_id * FEATURE_DIM + i];
        }

        // update the transmittance
        this->_T = next_T;

        // the global index in all intersections ([n_isects]).
        this->_last_index = batch_start + t;

        // Return whether we want to terminate the rasterization process.
        return false;
    }

    inline __device__ auto pixel_postprocess_impl() -> void {
        // write to the output buffer
        auto const offset_pixel =
            this->image_id * this->image_height * this->image_width + this->pixel_id;
        this->render_alpha_ptr[offset_pixel] = 1.0f - this->_T;

        this->render_last_index_ptr[offset_pixel] = this->_last_index;

#pragma unroll
        for (size_t i = 0; i < FEATURE_DIM; i++) {
            this->render_feature_ptr[offset_pixel * FEATURE_DIM + i] =
                this->_expected_feature[i];
        }
    }
};

// struct ImageGaussianRasterizeKernelBackwardOperator
//     : BaseRasterizeKernelOperator<ImageGaussianRasterizeKernelBackwardOperator> {

//     // Forward Inputs
//     float *opacity_ptr; // [N, 1]

//     // Forward Outputs
//     float *render_alpha_ptr; // [n_images, image_height, image_width, 1]

//     // Gradients for Forward Outputs
//     float *v_render_alpha_ptr; // [n_images, image_height, image_width, 1]

//     // Gradients for Forward Inputs
//     float *v_opacity_ptr; // [N, 1]

//     // Internal variables
//     float _T_final;        // final transmittance
//     float _T;              // current transmittance (from back to front)
//     float _v_render_alpha; // dl/d_render_alpha for this pixel

//     static inline __host__ auto sm_size_per_primitive_impl() -> uint32_t {
//         // since we will cache the opacity [float] and primitive_id [uint32_t] in the
//         // shared memory, the total shared memory size per primitive is:
//         return sizeof(float) + sizeof(uint32_t);
//     }

//     inline __device__ auto initialize_impl() -> bool {
//         // load the gradient for this pixel
//         auto const offset_pixel =
//             this->image_id * this->image_height * this->image_width + this->pixel_id;
//         this->_v_render_alpha = this->v_render_alpha_ptr[offset_pixel];

//         // load the initial transmittance as remaining transmittance
//         this->_T_final = 1.0f - this->render_alpha_ptr[offset_pixel];
//         this->_T = this->_T_final;
//         return true;
//     }

//     inline __device__ auto primitive_preprocess_impl(uint32_t primitive_id) -> void {
//         // cache data to shared memory
//         auto const sm_opacity_ptr = reinterpret_cast<float *>(this->sm_ptr);
//         auto const sm_primitive_id_ptr =
//             reinterpret_cast<uint32_t
//             *>(&sm_opacity_ptr[this->n_threads_per_block]);
//         sm_opacity_ptr[this->thread_rank] = this->opacity_ptr[primitive_id];
//         sm_primitive_id_ptr[this->thread_rank] = primitive_id;
//     }

//     template <class WarpT>
//     inline __device__ auto
//     rasterize_impl(uint32_t batch_start, uint32_t t, WarpT &warp) -> bool {
//         // load data from shared memory
//         auto const sm_opacity_ptr = reinterpret_cast<float *>(this->sm_ptr);
//         auto const sm_primitive_id_ptr =
//             reinterpret_cast<uint32_t
//             *>(&sm_opacity_ptr[this->n_threads_per_block]);
//         auto const alpha = sm_opacity_ptr[t];
//         auto const primitive_id = sm_primitive_id_ptr[t];

//         // compute the gradient
//         auto const ra = 1.0f / (1.0f - alpha);
//         this->_T *= ra;
//         auto v_alpha = this->_T_final * ra * this->_v_render_alpha;

//         // reduce the gradient over the warp [faster than atomicAdd to global memory]
//         tinyrend::warp::warpSum(v_alpha, warp);

//         // first thread in the warp writes the gradient to global memory.
//         if (warp.thread_rank() == 0) {
//             float *v_opacity_ptr = (float *)this->v_opacity_ptr;
//             atomicAdd(v_opacity_ptr + primitive_id, v_alpha);
//         }

//         // Return whether we want to terminate the rasterization process.
//         return false;
//     }

//     inline __device__ auto pixel_postprocess_impl() -> void {
//         // Do nothing
//     }
// };

} // namespace tinyrend::rasterization