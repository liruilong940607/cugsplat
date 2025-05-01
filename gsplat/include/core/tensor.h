#pragma once

#include <cstdint>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#include <type_traits>

#include "core/macros.h" // for GSPLAT_HOST_DEVICE
#include "core/types.h"  // for Maybe

namespace gsplat {

// Helper function to get number of float components
template <typename U>
GSPLAT_HOST_DEVICE static constexpr size_t get_float_components() {
    if constexpr (std::is_same_v<U, glm::fmat3>) {
        return 9; // 3x3 matrix
    } else if constexpr (std::is_same_v<U, glm::fvec3>) {
        return 3; // 3D vector
    } else if constexpr (std::is_array_v<U> &&
                         std::is_same_v<std::remove_extent_t<U>, float>) {
        return std::extent_v<U>; // float array
    } else {
        return sizeof(U) / sizeof(float); // fallback for other types
    }
}

#ifdef __CUDACC__

// Helper function for warp reduction
template <int OFFSET> __device__ static float warp_reduce(float val) {
    if constexpr (OFFSET > 0) {
        val += __shfl_down_sync(0xFFFFFFFF, val, OFFSET);
        return warp_reduce<OFFSET / 2>(val);
    }
    return val;
}

#endif

template <typename T> struct Tensor {
    // Data members
    const T *data_ptr = nullptr;    // Pointer to global memory for data
    T *grad_ptr = nullptr;          // Pointer to global memory for gradient
    Maybe<T> data_val = Maybe<T>(); // Cached data in local memory
    Maybe<T> grad_val = Maybe<T>(); // Cached gradient in local memory

    // Default constructor
    GSPLAT_HOST_DEVICE Tensor() {}

    // Constructor with data pointer and gradient pointer
    GSPLAT_HOST_DEVICE Tensor(const T *data_ptr, T *grad_ptr = nullptr)
        : data_ptr(data_ptr), grad_ptr(grad_ptr) {}

    // Shift pointer by an offset
    GSPLAT_HOST_DEVICE void shift_ptr(size_t offset) {
        if (data_ptr) {
            data_ptr += offset;
        }
        if (grad_ptr) {
            grad_ptr += offset;
        }
    }

    // Get data with caching
    GSPLAT_HOST_DEVICE T get_data() {
        if (!data_val.has_value() && data_ptr) {
            data_val.set(data_ptr[0]);
        }
        return data_val.get();
    }

    // Get grad with caching
    GSPLAT_HOST_DEVICE T get_grad() {
        if (!grad_val.has_value() && grad_ptr) {
            grad_val.set(grad_ptr[0]);
        }
        return grad_val.get();
    }

    // Set data
    GSPLAT_HOST_DEVICE void set_data(T value) { data_val.set(value); }

    // Set grad
    GSPLAT_HOST_DEVICE void set_grad(T value) { grad_val.set(value); }

#ifdef __CUDACC__

    // Warp reduction for gradient
    template <size_t WARP_SIZE> __device__ void export_grad() {
        if (!grad_ptr || !grad_val.has_value())
            return;

        // Convert to basic float type
        float *val_components = reinterpret_cast<float *>(&grad_val._value);
        float *ptr_components = reinterpret_cast<float *>(grad_ptr);

        constexpr size_t num_components = get_float_components<T>();
#pragma unroll
        for (int i = 0; i < num_components; ++i) {
            // Warp reduction with compile-time constant iterations
            float val = warp_reduce<WARP_SIZE / 2>(val_components[i]);

            // Atomic addition: only the first lane in the warp
            if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
                atomicAdd(ptr_components + i, val);
            }
        }
    }

#endif
};

} // namespace gsplat