#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <iostream>

#include "helpers.cuh"
#include "helpers.h"
#include "tinyrend/rasterization/kernel.cuh"
#include "tinyrend/rasterization/operators/simple_planer.cuh"

using namespace tinyrend::rasterization;

auto test_rasterization_simple_planer() -> int {

    // Configurations
    const int n_primitives = 2;
    const uint32_t image_height = 28;
    const uint32_t image_width = 22;
    const uint32_t tile_width = 8;
    const uint32_t tile_height = 16;
    dim3 threads(tile_width, tile_height, 1);
    dim3 grid(1, 1, 1);

    // Create primitive data:
    auto const opacity_ptr = create_device_ptr<float>({0.5f, 0.7f});
    // Create isect info: all two primitives are intersected with the first tile
    auto const isect_primitive_ids = create_device_ptr<uint32_t>({0, 1});
    auto const isect_prefix_sum_per_tile = create_device_ptr<uint32_t>({2});

    // Prepare forward outputs
    auto render_alpha_ptr =
        create_device_ptr<float>(image_height * image_width); // only alloc mem, no init

    // Create forward operator
    SimplePlanerRasterizeKernelForwardOperator forward_op{};
    forward_op.opacity_ptr = opacity_ptr;
    forward_op.render_alpha_ptr = render_alpha_ptr;

    // Launch forward rasterization
    size_t forward_shmem_size =
        decltype(forward_op)::sm_size_per_primitive() * threads.x * threads.y;
    rasterize_kernel<<<grid, threads, forward_shmem_size>>>(
        forward_op,
        image_height,
        image_width,
        isect_primitive_ids,
        isect_prefix_sum_per_tile
    );

    // Copy data back to host and check the result
    auto const h_render_alpha_ptr =
        device_ptr_to_host_ptr<float>(render_alpha_ptr, image_height * image_width);
    for (int x = 0; x < tile_width; x++) {
        for (int y = 0; y < tile_height; y++) {
            int i = x + y * image_width;
            assert(is_close(h_render_alpha_ptr[i], 0.5f + (1 - 0.5f) * 0.7f));
        }
    }
    save_png(h_render_alpha_ptr, image_width, image_height, "results/render_alpha.png");

    // Prepare backward gradients
    auto const v_render_alpha_ptr =
        create_device_ptr<float>(image_height * image_width, 0.3f);
    auto v_opacity_ptr = create_device_ptr<float>(n_primitives, 0.0f); // zero init

    // Create backward operator
    SimplePlanerRasterizeKernelBackwardOperator backward_op{};
    backward_op.opacity_ptr = opacity_ptr;
    backward_op.render_alpha_ptr = render_alpha_ptr;
    backward_op.v_render_alpha_ptr = v_render_alpha_ptr;
    backward_op.v_opacity_ptr = v_opacity_ptr;

    // Launch backward rasterization
    size_t backward_shmem_size =
        decltype(backward_op)::sm_size_per_primitive() * threads.x * threads.y;
    rasterize_kernel<<<grid, threads, backward_shmem_size>>>(
        backward_op,
        image_height,
        image_width,
        isect_primitive_ids,
        isect_prefix_sum_per_tile,
        true // reverse order
    );

    // o = a + (1 - a) * b
    // o = 0.5f + (1 - 0.5f) * 0.7f
    // dl/da = dl/do * do/da = 0.3f * (1 - 0.7f) = 0.09f
    // dl/db = dl/do * do/db = 0.3f * 0.5f = 0.15f
    auto const h_v_opacity_ptr =
        device_ptr_to_host_ptr<float>(v_opacity_ptr, n_primitives);
    assert(is_close(h_v_opacity_ptr[0], 0.09f * tile_width * tile_height));
    assert(is_close(h_v_opacity_ptr[1], 0.15f * tile_width * tile_height));

    check_cuda_error();
    return 0;
}

auto main() -> int {
    int fails = 0;
    fails += test_rasterization_simple_planer();

    if (fails == 0) {
        printf("\nAll tests passed!\n");
    } else {
        printf("\n%d tests failed!\n", fails);
    }

    return fails;
}