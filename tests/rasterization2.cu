#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <iostream>

#include "helpers.cuh"
#include "helpers.h"
#include "tinyrend/rasterization/kernel2.cuh"
#include "tinyrend/rasterization/operators/simple_planer.cuh"

using namespace tinyrend::rasterization;

auto test_rasterization2() -> int {

    // number of primitives
    const int n_primitives = 2;
    // image size
    const uint32_t image_height = 4;
    const uint32_t image_width = 2;

    // NullRasterizeKernelOperator op{};
    SimplePlanerRasterizeKernelForwardOperator op{};
    using OpType = decltype(op);

    // op.mean_ptr =
    //     create_device_ptr<glm::fvec2>({glm::fvec2(6.0f, 6.0f),
    //     glm::fvec2(10.0f, 10.0f)}
    //     );
    // op.covariance_ptr = create_device_ptr<glm::fmat2>(
    //     {glm::fmat2(0.25f, 0.0f, 0.0f, 0.25f), glm::fmat2(0.30f, 0.0f, 0.0f, 0.30f)}
    // );
    op.opacity_ptr = create_device_ptr<float>({0.5f, 0.7f});
    op.render_alpha_ptr = create_device_ptr<float>(image_height * image_width);

    // Create isect info on GPU
    auto const isect_primitive_ids = create_device_ptr<uint32_t>({0, 1});
    auto const isect_prefix_sum_per_tile = create_device_ptr<uint32_t>({2});

    // launch rasterization kernel
    dim3 threads(16, 16, 1);
    dim3 grid(1, 1, 1);
    size_t shmem_size = OpType::smem_size_per_primitive() * 16 * 16;
    rasterize_kernel<<<grid, threads, shmem_size>>>(
        op, image_height, image_width, isect_primitive_ids, isect_prefix_sum_per_tile
    );

    // copy data back to host
    float *render_alpha_host = new float[image_height * image_width];
    cudaMemcpy(
        render_alpha_host,
        op.render_alpha_ptr,
        sizeof(float) * image_height * image_width,
        cudaMemcpyDeviceToHost
    );
    float ground_truth_render_alpha = 0.5f + (1 - 0.5f) * 0.7f;
    assert(is_close(render_alpha_host[0], ground_truth_render_alpha));
    save_png(render_alpha_host, image_width, image_height, "results/render_alpha.png");

    SimplePlanerRasterizeKernelBackwardOperator op_backward{};
    op_backward.opacity_ptr = op.opacity_ptr;
    op_backward.render_alpha_ptr = op.render_alpha_ptr;
    op_backward.v_render_alpha_ptr =
        create_device_ptr_with_init<float>(image_height * image_width, 0.3f);
    op_backward.v_opacity_ptr = create_device_ptr_with_init<float>(n_primitives, 0.0f);

    rasterize_kernel<<<grid, threads, shmem_size>>>(
        op_backward,
        image_height,
        image_width,
        isect_primitive_ids,
        isect_prefix_sum_per_tile,
        true // reverse order
    );

    // copy data back to host
    float *v_opacity_host = new float[n_primitives];
    cudaMemcpy(
        v_opacity_host,
        op_backward.v_opacity_ptr,
        sizeof(float) * n_primitives,
        cudaMemcpyDeviceToHost
    );

    // o = a + (1 - a) * b
    // o = 0.5f + (1 - 0.5f) * 0.7f
    // dl/da = dl/do * do/da = 0.3f * (1 - 0.7f) = 0.09f
    // dl/db = dl/do * do/db = 0.3f * 0.5f = 0.15f
    assert(is_close(v_opacity_host[0], 0.09f * image_height * image_width));
    assert(is_close(v_opacity_host[1], 0.15f * image_height * image_width));

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