#pragma once

#include <cuda_runtime.h>

namespace curend {

// Template for generating a linear kernel launcher
template <typename Func, typename... Args>
__global__ void linear_kernel_cuda(size_t n_elements, Func func, Args... args) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements)
        return;
    func(idx, args...);
}

// Helper to launch a linear kernel
template <typename Func, typename... Args>
void launch_linear_kernel_cuda(size_t n_elements, Func func, Args... args) {
    constexpr int BLOCK_SIZE = 256;
    const int num_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    linear_kernel_cuda<<<num_blocks, BLOCK_SIZE>>>(n_elements, func, args...);
}

template <typename Func, typename... Args>
void launch_linear_kernel_cpu(size_t n_elements, Func func, Args... args) {
    for (size_t i = 0; i < n_elements; i++) {
        func(i, args...);
    }
}

template <bool USE_CUDA, typename Func, typename... Args>
void launch_linear_kernel(size_t n_elements, Func func, Args... args) {
    if constexpr (USE_CUDA) {
        launch_linear_kernel_cuda(n_elements, func, args...);
    } else {
        launch_linear_kernel_cpu(n_elements, func, args...);
    }
}

} // namespace curend
