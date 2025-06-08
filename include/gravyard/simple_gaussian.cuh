// This file implements the simple Gaussian rasterization kernel operator.
// The simple Gaussian is very simple, defined by a 2D mean and a 2x2 covariance
matrix.

#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>
#include <glm/glm.hpp>

#include "tinyrend/core/macros.h" // for __device__
#include "tinyrend/rasterization/kernel2.cuh"

    namespace tinyrend::rasterization {

    namespace cg = cooperative_groups;

    template <class WarpT>
    inline __device__ void warpSum(glm::fvec2 & val, WarpT & warp) {
        val.x = cg::reduce(warp, val.x, cg::plus<float>());
        val.y = cg::reduce(warp, val.y, cg::plus<float>());
    }

    template <class WarpT>
    inline __device__ void warpSum(glm::fmat2 & val, WarpT & warp) {
        warpSum(val[0], warp);
        warpSum(val[1], warp);
    }

    struct SimpleGaussianRasterizeKernelForwardOperator
        : BaseRasterizeKernelOperator<SimpleGaussianRasterizeKernelForwardOperator> {

        // Inputs
        glm::fvec2 *mean_ptr;       // [N, 2]
        glm::fmat2 *covariance_ptr; // [N, 2, 2]

        // Outputs
        float *render_alpha_ptr; // [n_images, image_height, image_width, 1]

        // Internal variables
        float _T = 1.0f; // current transmittance

        static inline __host__ auto smem_size_per_primitive_impl() -> uint32_t {
            return sizeof(glm::fvec2) + sizeof(glm::fmat2);
        }

        inline __device__ auto initialize_impl() -> bool { return true; }

        inline __device__ auto primitive_preprocess_impl(uint32_t primitive_id
        ) -> void {
            // cache data to shared memory
            auto const smem_mean_ptr = reinterpret_cast<glm::fvec2 *>(this->smem_ptr);
            auto const smem_covariance_ptr =
                reinterpret_cast<glm::fmat2 *>(&smem_mean_ptr[this->n_threads_per_block]
                );
            smem_mean_ptr[this->thread_rank] = this->mean_ptr[primitive_id];
            smem_covariance_ptr[this->thread_rank] = this->covariance_ptr[primitive_id];
        }

        template <class WarpT>
        inline __device__ auto
        rasterize_impl(uint32_t batch_start, uint32_t t, WarpT &warp) -> bool {
            // load data from shared memory
            auto const smem_mean_ptr = reinterpret_cast<glm::fvec2 *>(this->smem_ptr);
            auto const smem_covariance_ptr =
                reinterpret_cast<glm::fmat2 *>(&smem_mean_ptr[this->n_threads_per_block]
                );
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

            return false; // Return whether we want to terminate the rasterization
            process.
        }

        inline __device__ auto pixel_postprocess_impl() -> void {
            if (this->render_alpha_ptr != nullptr) {
                auto const offset_pixel =
                    this->image_id * this->image_height * this->image_width +
                    this->pixel_id;
                this->render_alpha_ptr[offset_pixel] = 1.0f - this->_T;
            }
        }
    };

    struct SimpleGaussianRasterizeKernelBackwardOperator
        : BaseRasterizeKernelOperator<SimpleGaussianRasterizeKernelBackwardOperator> {

        // Forward Inputs
        glm::fvec2 *mean_ptr;       // [N, 2]
        glm::fmat2 *covariance_ptr; // [N, 2, 2]

        // Forward Outputs
        float *render_alpha_ptr; // [n_images, image_height, image_width, 1]

        // Gradients for Forward Outputs
        float *v_render_alpha_ptr; // [n_images, image_height, image_width, 1]

        // Gradients for Forward Inputs
        glm::fvec2 *v_mean_ptr;       // [N, 2]
        glm::fmat2 *v_covariance_ptr; // [N, 2, 2]

        // Internal variables
        float _T_final;        // final transmittance
        float _T;              // current transmittance (from back to front)
        float _v_render_alpha; // dl/d_render_alpha for this pixel

        static inline __host__ auto smem_size_per_primitive_impl() -> uint32_t {
            return sizeof(glm::fvec2) + sizeof(glm::fmat2) + sizeof(uint32_t);
        }

        inline __device__ auto initialize_impl() -> bool {
            // load the gradient for this pixel
            auto const offset_pixel =
                this->image_id * this->image_height * this->image_width +
                this->pixel_id;
            this->_v_render_alpha = this->v_render_alpha_ptr[offset_pixel];

            // load the initial transmittance as remaining transmittance
            this->_T_final = 1.0f - this->render_alpha_ptr[offset_pixel];
            this->_T = this->_T_final;
            return true;
        }

        inline __device__ auto primitive_preprocess_impl(uint32_t primitive_id
        ) -> void {
            // cache data to shared memory
            auto const smem_mean_ptr = reinterpret_cast<glm::fvec2 *>(this->smem_ptr);
            auto const smem_covariance_ptr =
                reinterpret_cast<glm::fmat2 *>(&smem_mean_ptr[this->n_threads_per_block]
                );
            auto const smem_primitive_id_ptr = reinterpret_cast<uint32_t *>(
                &smem_covariance_ptr[this->n_threads_per_block]
            );
            smem_mean_ptr[this->thread_rank] = this->mean_ptr[primitive_id];
            smem_covariance_ptr[this->thread_rank] = this->covariance_ptr[primitive_id];
            smem_primitive_id_ptr[this->thread_rank] = primitive_id;
        }

        template <class WarpT>
        inline __device__ auto
        rasterize_impl(uint32_t batch_start, uint32_t t, WarpT &warp) -> bool {
            // load data from shared memory
            auto const smem_mean_ptr = reinterpret_cast<glm::fvec2 *>(this->smem_ptr);
            auto const smem_covariance_ptr =
                reinterpret_cast<glm::fmat2 *>(&smem_mean_ptr[this->n_threads_per_block]
                );
            auto const smem_primitive_id_ptr = reinterpret_cast<uint32_t *>(
                &smem_covariance_ptr[this->n_threads_per_block]
            );
            auto const mu = smem_mean_ptr[t];
            auto const covariance = smem_covariance_ptr[t];
            auto const primitive_id = smem_primitive_id_ptr[t];

            // compute the light attenuation
            auto const dx = this->pixel_x - mu.x;
            auto const dy = this->pixel_y - mu.y;
            auto const sigma =
                0.5f * (covariance[0][0] * dx * dx + covariance[1][1] * dy * dy) +
                covariance[0][1] * dx * dy;
            auto const alpha = exp(-sigma);

            // compute the gradient
            auto const ra = 1.0f / (1.0f - alpha);
            this->_T *= ra;
            auto const fac = alpha * this->_T;
            auto const v_alpha = this->_T_final * ra * this->_v_render_alpha;

            auto const v_sigma = -alpha * v_alpha;

            auto v_mean = v_sigma * glm::fvec2(
                                        covariance[0][0] * dx + covariance[0][1] * dy,
                                        covariance[1][1] * dy + covariance[0][1] * dx
                                    );
            auto v_covariance =
                0.5f * v_sigma * glm::fmat2(dx * dx, dx * dy, dx * dy, dy * dy);

            warpSum(v_mean, warp);
            warpSum(v_covariance, warp);

            // first thread in the block handles the gradient accumulation
            if (t == 0) {
                float *v_mean_ptr = (float *)this->v_mean_ptr;
                atomicAdd(v_mean_ptr + primitive_id * 3, v_mean.x);
                atomicAdd(v_mean_ptr + primitive_id * 3 + 1, v_mean.y);

                float *v_covariance_ptr = (float *)this->v_covariance_ptr;
                atomicAdd(v_covariance_ptr + primitive_id * 4, v_covariance[0][0]);
                atomicAdd(v_covariance_ptr + primitive_id * 4 + 1, v_covariance[0][1]);
                atomicAdd(v_covariance_ptr + primitive_id * 4 + 2, v_covariance[1][0]);
                atomicAdd(v_covariance_ptr + primitive_id * 4 + 3, v_covariance[1][1]);
            }

            return false; // Return whether we want to terminate the rasterization
            process.
        }

        inline __device__ auto pixel_postprocess_impl() -> void {
            // Do nothing
        }
    };

} // namespace tinyrend::rasterization