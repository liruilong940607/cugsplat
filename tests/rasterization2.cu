#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <iostream>

#include "helpers.cuh"
#include "tinyrend/rasterization/kernel2.cuh"

using namespace tinyrend::rasterization;

auto test_rasterization2() -> int {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    const int n_primitives = 2;

    NullRasterizeKernelOperator op;

    // Create isect info on GPU
    uint32_t isect_primitive_ids_host[n_primitives] = {0, 1};
    uint32_t isect_prefix_sum_per_tile_host[1] = {2};
    uint32_t *isect_primitive_ids;
    cudaMalloc(&isect_primitive_ids, sizeof(uint32_t) * n_primitives);
    cudaMemcpy(
        isect_primitive_ids,
        isect_primitive_ids_host,
        sizeof(uint32_t) * n_primitives,
        cudaMemcpyHostToDevice
    );
    uint32_t *isect_prefix_sum_per_tile;
    cudaMalloc(&isect_prefix_sum_per_tile, sizeof(uint32_t) * 1);
    cudaMemcpy(
        isect_prefix_sum_per_tile,
        isect_prefix_sum_per_tile_host,
        sizeof(uint32_t) * 1,
        cudaMemcpyHostToDevice
    );

    // image size
    const uint32_t image_height = 16;
    const uint32_t image_width = 16;

    // launch rasterization kernel
    dim3 threads(16, 16, 1);
    dim3 grid(1, 1, 1);
    size_t shmem_size =
        NullRasterizeKernelOperator::smem_size_per_primitive() * 16 * 16;
    rasterize_kernel_forward<<<grid, threads, shmem_size>>>(
        op, image_height, image_width, isect_primitive_ids, isect_prefix_sum_per_tile
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}

auto main() -> int {
    int fails = 0;
    fails += test_rasterization2();

    if (fails == 0) {
        printf("\nAll tests passed!\n");
    } else {
        printf("\n%d tests failed!\n", fails);
    }

    return fails;
}