// This file implements the simple Gaussian rasterization kernel operator.
// The simple Gaussian is very simple, defined by a 2D mean and a 2x2 covariance matrix.

#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>
#include <glm/glm.hpp>

#include "tinyrend/core/macros.h" // for __device__
#include "tinyrend/rasterization/kernel2.cuh"

namespace tinyrend::rasterization {

namespace cg = cooperative_groups;

template <class WarpT> inline __device__ void warpSum(float &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(glm::fvec2 &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(glm::fmat2 &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
}

struct SimplePlanerRasterizeKernelForwardOperator
    : BaseRasterizeKernelOperator<SimplePlanerRasterizeKernelForwardOperator> {

    // Inputs
    float *opacity_ptr; // [N, 1]

    // Outputs
    float *alphamap_ptr; // [n_images, image_height, image_width, 1]

    // Internal variables
    float _T = 1.0f; // current transmittance

    static inline __host__ auto smem_size_per_primitive_impl() -> uint32_t {
        return sizeof(float);
    }

    inline __device__ auto initialize_impl() -> bool { return true; }

    inline __device__ auto primitive_preprocess_impl(uint32_t primitive_id) -> void {
        // cache data to shared memory
        auto const smem_opacity_ptr = reinterpret_cast<float *>(this->smem_ptr);
        smem_opacity_ptr[this->thread_rank] = this->opacity_ptr[primitive_id];
    }

    template <class WarpT>
    inline __device__ auto
    rasterize_impl(uint32_t batch_start, uint32_t t, WarpT &warp) -> bool {
        // load data from shared memory
        auto const smem_opacity_ptr = reinterpret_cast<float *>(this->smem_ptr);
        auto const alpha = smem_opacity_ptr[t];

        // update the transmittance
        auto const next_T = this->_T * (1.0f - alpha);
        this->_T = next_T;

        return false; // Return whether we want to terminate the rasterization process.
    }

    inline __device__ auto pixel_postprocess_impl() -> void {
        if (this->alphamap_ptr != nullptr) {
            auto const offset_pixel =
                this->image_id * this->image_height * this->image_width +
                this->pixel_id;
            this->alphamap_ptr[offset_pixel] = 1.0f - this->_T;
        }
    }
};

struct SimplePlanerRasterizeKernelBackwardOperator
    : BaseRasterizeKernelOperator<SimplePlanerRasterizeKernelBackwardOperator> {

    // Forward Inputs
    float *opacity_ptr; // [N, 1]

    // Forward Outputs
    float *alphamap_ptr; // [n_images, image_height, image_width, 1]

    // Gradients for Forward Outputs
    float *v_alphamap_ptr; // [n_images, image_height, image_width, 1]

    // Gradients for Forward Inputs
    float *v_opacity_ptr; // [N, 1]

    // Internal variables
    float _T_final;    // final transmittance
    float _T;          // current transmittance (from back to front)
    float _v_alphamap; // dl/d_alphamap for this pixel

    static inline __host__ auto smem_size_per_primitive_impl() -> uint32_t {
        return sizeof(float) + sizeof(uint32_t);
    }

    inline __device__ auto initialize_impl() -> bool {
        // load the gradient for this pixel
        auto const offset_pixel =
            this->image_id * this->image_height * this->image_width + this->pixel_id;
        this->_v_alphamap = this->v_alphamap_ptr[offset_pixel];

        // load the initial transmittance as remaining transmittance
        this->_T_final = 1.0f - this->alphamap_ptr[offset_pixel];
        this->_T = this->_T_final;
        return true;
    }

    inline __device__ auto primitive_preprocess_impl(uint32_t primitive_id) -> void {
        // cache data to shared memory
        auto const smem_opacity_ptr = reinterpret_cast<float *>(this->smem_ptr);
        auto const smem_primitive_id_ptr =
            reinterpret_cast<uint32_t *>(&smem_opacity_ptr[this->n_threads_per_block]);
        smem_opacity_ptr[this->thread_rank] = this->opacity_ptr[primitive_id];
        smem_primitive_id_ptr[this->thread_rank] = primitive_id;
    }

    template <class WarpT>
    inline __device__ auto
    rasterize_impl(uint32_t batch_start, uint32_t t, WarpT &warp) -> bool {
        // load data from shared memory
        auto const smem_opacity_ptr = reinterpret_cast<float *>(this->smem_ptr);
        auto const smem_primitive_id_ptr =
            reinterpret_cast<uint32_t *>(&smem_opacity_ptr[this->n_threads_per_block]);
        auto const alpha = smem_opacity_ptr[t];
        auto const primitive_id = smem_primitive_id_ptr[t];

        // compute the gradient
        auto const ra = 1.0f / (1.0f - alpha);
        this->_T *= ra;
        auto v_alpha = this->_T_final * ra * this->_v_alphamap;

        warpSum(v_alpha, warp);

        // first thread in the block handles the gradient accumulation
        if (warp.thread_rank() == 0) {
            float *v_opacity_ptr = (float *)this->v_opacity_ptr;
            atomicAdd(v_opacity_ptr + primitive_id, v_alpha);
        }

        return false; // Return whether we want to terminate the rasterization process.
    }

    inline __device__ auto pixel_postprocess_impl() -> void {
        // Do nothing
    }
};

} // namespace tinyrend::rasterization