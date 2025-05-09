#pragma once

#include <cuda_runtime.h>

namespace cugsplat {

// Template for generating a linear kernel launcher
template <typename Func, typename... Args>
__global__ void linear_kernel(size_t n_elements, Func func, Args... args) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements)
        return;
    func(idx, args...);
}

// Helper to launch a linear kernel
template <typename Func, typename... Args>
void launch_linear_kernel(size_t n_elements, Func func, Args... args) {
    constexpr int BLOCK_SIZE = 256;
    const int num_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    linear_kernel<<<num_blocks, BLOCK_SIZE>>>(n_elements, func, args...);
}

} // namespace cugsplat
