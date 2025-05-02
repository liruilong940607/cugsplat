#pragma once

#include <cstdint>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "core/macros.h" // for GSPLAT_HOST_DEVICE

namespace gsplat {

template <typename T> struct Maybe {
    bool _has_value = false;
    T _value;

    GSPLAT_HOST_DEVICE inline T get() const {
        return this->_has_value ? this->_value : T{};
    }

    GSPLAT_HOST_DEVICE inline bool has_value() const {
        return this->_has_value;
    }

    GSPLAT_HOST_DEVICE inline void set(const T &v) {
        this->_value = v;
        this->_has_value = true;
    }
};

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

// template <typename T, bool CacheOnly = false>
// struct MaybeCached {
//     // Data members
//     std::conditional_t<CacheOnly, std::nullptr_t, const T*> data_ptr =
//     nullptr; std::conditional_t<CacheOnly, std::nullptr_t, T*> grad_ptr =
//     nullptr; Maybe<T> data_val = Maybe<T>(); Maybe<T> grad_val = Maybe<T>();
//     bool requires_grad_ = false;

//     // Default constructor
//     GSPLAT_HOST_DEVICE MaybeCached() {}

//     // Constructor for cached-only mode
//     template<bool C = CacheOnly, std::enable_if_t<C, int> = 0>
//     GSPLAT_HOST_DEVICE MaybeCached(const T& data, const T& grad = T{})
//         : data_val(data), grad_val(grad) {}

//     // Constructor for pointer mode
//     template<bool C = CacheOnly, std::enable_if_t<!C, int> = 0>
//     GSPLAT_HOST_DEVICE MaybeCached(const T* data_ptr, T* grad_ptr = nullptr)
//         : data_ptr(data_ptr), grad_ptr(grad_ptr) {
//             if (this->grad_ptr) requires_grad_ = true;
//     }

//     // Rest of the methods remain similar but with conditional compilation...
// };

template <typename T, bool Mutable = false> struct MaybeCached {
    using ptr_type = std::conditional_t<Mutable, T *, const T *>;

    ptr_type _data_ptr = nullptr;
    Maybe<T> _data = Maybe<T>();

    GSPLAT_HOST_DEVICE MaybeCached() {}

    GSPLAT_HOST_DEVICE MaybeCached(ptr_type data_ptr) : _data_ptr(data_ptr) {}

    GSPLAT_HOST_DEVICE MaybeCached(T &data) { _data.set(data); }

    GSPLAT_HOST_DEVICE inline T get() {
        if (!_data.has_value() && _data_ptr) {
            _data.set(_data_ptr[0]);
        }
        return _data.get();
    }

    // Shift pointer by an offset
    GSPLAT_HOST_DEVICE inline void shift_ptr(size_t offset) {
        if (_data_ptr) {
            _data_ptr += offset;
        }
    }

    // Set data
    GSPLAT_HOST_DEVICE inline void set(T value) { _data.set(value); }

    // Accumulate data
    GSPLAT_HOST_DEVICE inline void accum(T value) {
        if (!_data.has_value()) {
            _data.set(T{});
        }
        _data._value += value;
    }

#ifdef __CUDACC__
    template <size_t WARP_SIZE> __device__ inline void export_grad() {
        if (!_data_ptr || !_data.has_value())
            return;

        // Convert to basic float type
        float *val_components = reinterpret_cast<float *>(&_data._value);
        float *ptr_components = reinterpret_cast<float *>(_data_ptr);

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

template <typename T, bool RequiresGrad = false> struct Tensor {
    using data_type = MaybeCached<T, false>;
    using grad_type =
        std::conditional_t<RequiresGrad, MaybeCached<T, true>, std::nullptr_t>;

    data_type data;
    grad_type grad;

    GSPLAT_HOST_DEVICE Tensor() {}

    GSPLAT_HOST_DEVICE Tensor(const T *data_ptr) : data(data_ptr) {}

    GSPLAT_HOST_DEVICE Tensor(const T *data_ptr, T *grad_ptr)
        : data(data_ptr), grad(grad_ptr) {}

    GSPLAT_HOST_DEVICE static constexpr bool requires_grad() {
        return RequiresGrad;
    }

    GSPLAT_HOST_DEVICE inline void shift_ptr(size_t offset) {
        data.shift_ptr(offset);
        if constexpr (RequiresGrad) {
            grad.shift_ptr(offset);
        }
    }

    GSPLAT_HOST_DEVICE inline T get() { return data.get(); }

    GSPLAT_HOST_DEVICE inline T get_grad() {
        if constexpr (RequiresGrad) {
            return grad.get();
        } else {
            return T{};
        }
    }
};

// template <typename T, bool RequiresGrad = false> struct MaybeCached {
//     // Data members

//     // Pointer to global memory
//     const T *data_ptr = nullptr;
//     // Cached data in local memory
//     Maybe<T> data_val = Maybe<T>();
//     if constexpr (RequiresGrad) {
//         // Pointer to global memory for gradient
//         T* grad_ptr = nullptr;
//         // Cached gradient in local memory
//         Maybe<T> grad_val = Maybe<T>();
//     }

//     // Default constructor
//     GSPLAT_HOST_DEVICE MaybeCached() {}

//     // Constructor with data pointer and (optional) gradient pointer
//     if constexpr (RequiresGrad) {
//         GSPLAT_HOST_DEVICE MaybeCached(const T *data_ptr, T *grad_ptr =
//         nullptr)
//             : data_ptr(data_ptr), grad_ptr(grad_ptr) {}
//     } else {
//         GSPLAT_HOST_DEVICE MaybeCached(const T *data_ptr)
//             : data_ptr(data_ptr) {}
//     }

//     // Shift pointer by an offset
//     GSPLAT_HOST_DEVICE inline void shift_ptr(size_t offset) {
//         if (data_ptr) {
//             data_ptr += offset;
//         }
//         if constexpr (RequiresGrad) {
//             if (grad_ptr) {
//                 grad_ptr += offset;
//             }
//         }
//     }

//     // Get data with caching
//     GSPLAT_HOST_DEVICE inline T get_data() {
//         if (!data_val.has_value() && data_ptr) {
//             data_val.set(data_ptr[0]);
//         }
//         return data_val.get();
//     }

//     // Get grad with caching
//     if constexpr (RequiresGrad) {
//         GSPLAT_HOST_DEVICE inline T get_grad() {
//             if (!grad_val.has_value() && grad_ptr) {
//                 grad_val.set(grad_ptr[0]);
//             }
//             return grad_val.get();
//         }
//     }

//     // Set data
//     GSPLAT_HOST_DEVICE inline void set_data(T value) { data_val.set(value); }

//     if constexpr (RequiresGrad) {
//         // Set grad
//         GSPLAT_HOST_DEVICE inline void set_grad(T value) {
//         grad_val.set(value); }

//         // accumulate grad
//         GSPLAT_HOST_DEVICE inline void accum_grad(T value) {
//             if (!grad_val.has_value()) {
//                 grad_val.set(T{});
//             }
//             grad_val._value += value;
//         }

// #ifdef __CUDACC__

//     // Warp reduction for gradient
//     template <size_t WARP_SIZE> __device__ inline void export_grad() {
//         if (!grad_ptr || !grad_val.has_value())
//             return;

//         // Convert to basic float type
//         float *val_components = reinterpret_cast<float *>(&grad_val._value);
//         float *ptr_components = reinterpret_cast<float *>(grad_ptr);

//         constexpr size_t num_components = get_float_components<T>();
// #pragma unroll
//         for (int i = 0; i < num_components; ++i) {
//             // Warp reduction with compile-time constant iterations
//             float val = warp_reduce<WARP_SIZE / 2>(val_components[i]);

//             // Atomic addition: only the first lane in the warp
//             if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
//                 atomicAdd(ptr_components + i, val);
//             }
//         }
//     }

// #endif

//     }

// };

} // namespace gsplat