// This file implements the simple Gaussian rasterization kernel operator.
// The simple Gaussian is very simple, defined by a 2D mean and a 2x2 covariance matrix.

#pragma once

#include <cstdint>
#include <glm/glm.hpp>

#include "tinyrend/core/macros.h" // for GSPLAT_HOST_DEVICE
#include "tinyrend/rasterization/kernel2.cuh"

namespace tinyrend::rasterization {

struct SimpleGaussianRasterizeKernelForwardOperator
    : BaseRasterizeKernelOperator<SimpleGaussianRasterizeKernelForwardOperator> {

    // Inputs
    glm::fvec2 *in_mean_ptr;       // [N, 2]
    glm::fmat2 *in_covariance_ptr; // [N, 2, 2]

    // Outputs
    float *out_alphamap_ptr; // [n_images, image_height, image_width, 1]

    // Internal variables
    float _T = 1.0f; // current transmittance

    static inline GSPLAT_HOST_DEVICE auto smem_size_per_primitive_impl() -> uint32_t {
        return sizeof(glm::fvec2) + sizeof(glm::fmat2);
    }

    inline GSPLAT_HOST_DEVICE auto initialize_impl() -> bool { return true; }

    inline GSPLAT_HOST_DEVICE auto primitive_preprocess_impl(uint32_t primitive_id
    ) -> void {
        // cache data to shared memory
        auto const smem_mean_ptr = reinterpret_cast<glm::fvec2 *>(this->smem_ptr);
        auto const smem_covariance_ptr =
            reinterpret_cast<glm::fmat2 *>(&smem_mean_ptr[this->n_threads_per_block]);
        smem_mean_ptr[this->thread_rank] = this->in_mean_ptr[primitive_id];
        smem_covariance_ptr[this->thread_rank] = this->in_covariance_ptr[primitive_id];
    }

    inline GSPLAT_HOST_DEVICE auto
    rasterize_impl(uint32_t batch_start, uint32_t t) -> bool {
        // load data from shared memory
        auto const smem_mean_ptr = reinterpret_cast<glm::fvec2 *>(this->smem_ptr);
        auto const smem_covariance_ptr =
            reinterpret_cast<glm::fmat2 *>(&smem_mean_ptr[this->n_threads_per_block]);
        auto const mu = smem_mean_ptr[t];
        auto const covariance = smem_covariance_ptr[t];

        // compute the light attenuation
        auto const dx = this->pixel_x - mu.x;
        auto const dy = this->pixel_y - mu.y;
        auto const sigma =
            0.5f * (covariance[0][0] * dx * dx + covariance[1][1] * dy * dy) +
            covariance[0][1] * dx * dy;
        auto const alpha = exp(-sigma);

        // update the transmittance
        auto const next_T = this->_T * (1.0f - alpha);
        this->_T = next_T;

        return false; // Return whether we want to terminate the rasterization process.
    }

    inline GSPLAT_HOST_DEVICE auto pixel_postprocess_impl() -> void {
        if (this->out_alphamap_ptr != nullptr) {
            auto const offset_pixel =
                this->image_id * this->image_height * this->image_width +
                this->pixel_id;
            this->out_alphamap_ptr[offset_pixel] = 1.0f - this->_T;
        }
    }
};

struct SimpleGaussianRasterizeKernelBackwardOperator
    : BaseRasterizeKernelOperator<SimpleGaussianRasterizeKernelBackwardOperator> {
    static inline GSPLAT_HOST_DEVICE auto smem_size_per_primitive_impl() -> uint32_t {
        return 0;
    }

    inline GSPLAT_HOST_DEVICE auto initialize_impl() -> bool { return true; }

    inline GSPLAT_HOST_DEVICE auto primitive_preprocess_impl(uint32_t primitive_id
    ) -> void {
        // Do nothing
    }

    inline GSPLAT_HOST_DEVICE auto
    rasterize_impl(uint32_t batch_start, uint32_t t) -> bool {
        // Do nothing
        return false; // Return whether we want to terminate the rasterization process.
    }

    inline GSPLAT_HOST_DEVICE auto pixel_postprocess_impl() -> void {
        // Do nothing
    }
};

} // namespace tinyrend::rasterization