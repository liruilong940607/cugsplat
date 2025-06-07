#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <iostream>

#include "helpers.cuh"
#include "helpers.h"
#include "tinyrend/rasterization/kernel2.cuh"
#include "tinyrend/rasterization/operators/simple_gaussian.cuh"

using namespace tinyrend::rasterization;

auto test_rasterization2() -> int {

    // number of primitives
    const int n_primitives = 2;
    // image size
    const uint32_t image_height = 16;
    const uint32_t image_width = 16;

    // NullRasterizeKernelOperator op{};
    SimpleGaussianRasterizeKernelForwardOperator op{};
    using OpType = decltype(op);

    op.mean_ptr =
        create_device_ptr<glm::fvec2>({glm::fvec2(6.0f, 6.0f), glm::fvec2(10.0f, 10.0f)}
        );
    op.covariance_ptr = create_device_ptr<glm::fmat2>(
        {glm::fmat2(0.25f, 0.0f, 0.0f, 0.25f), glm::fmat2(0.30f, 0.0f, 0.0f, 0.30f)}
    );
    op.alphamap_ptr = create_device_ptr<float>(image_height * image_width);

    // Create isect info on GPU
    auto const isect_primitive_ids = create_device_ptr<uint32_t>({0, 1});
    auto const isect_prefix_sum_per_tile = create_device_ptr<uint32_t>({2});

    // launch rasterization kernel
    dim3 threads(16, 16, 1);
    dim3 grid(1, 1, 1);
    size_t shmem_size = OpType::smem_size_per_primitive() * 16 * 16;
    rasterize_kernel_forward<<<grid, threads, shmem_size>>>(
        op, image_height, image_width, isect_primitive_ids, isect_prefix_sum_per_tile
    );

    // copy data back to host
    float *alphamap_host = new float[image_height * image_width];
    cudaMemcpy(
        alphamap_host,
        op.alphamap_ptr,
        sizeof(float) * image_height * image_width,
        cudaMemcpyDeviceToHost
    );
    save_png(alphamap_host, image_width, image_height, "results/alphamap.png");

    // SimpleGaussianRasterizeKernelBackwardOperator op_backward{};

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