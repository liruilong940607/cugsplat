#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <iostream>

#include "helpers.cuh"
#include "tinyrend/rasterization/kernel2.cuh"

using namespace tinyrend::rasterization;

auto test_rasterization2() -> int {

    const int n_primitives = 2;

    NullRasterizeKernelOperator op{};

    // Create isect info on GPU

    auto const isect_primitive_ids = create_device_ptr<uint32_t>({0, 1});
    auto const isect_prefix_sum_per_tile = create_device_ptr<uint32_t>({2});

    // image size
    const uint32_t image_height = 32;
    const uint32_t image_width = 32;

    // launch rasterization kernel
    dim3 threads(16, 16, 1);
    dim3 grid(1, 1, 1);
    size_t shmem_size =
        NullRasterizeKernelOperator::smem_size_per_primitive() * 16 * 16;
    rasterize_kernel_forward<<<grid, threads, shmem_size>>>(
        op, image_height, image_width, isect_primitive_ids, isect_prefix_sum_per_tile
    );

    check_cuda_error();
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