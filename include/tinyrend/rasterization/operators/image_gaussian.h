#pragma once

#include <cstdint>
#include <glm/glm.hpp>

#include "tinyrend/core/macros.h" // for GSPLAT_HOST_DEVICE
#include "tinyrend/rasterization/kernel2.cuh"

namespace tinyrend::rasterization {

struct ImageGaussianRasterizeKernelOperator
    : BaseRasterizeKernelOperator<ImageGaussianRasterizeKernelOperator> {
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