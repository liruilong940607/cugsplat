#include "tinyrend/core/vec.h"
#include "tinyrend/rasterization/base.cuh"
#include "tinyrend/rasterization/operators/simple_planer.cuh"

namespace tinyrend::rasterization {

void launch_simple_planer_forward(
    // Primitives
    const size_t n_primitives,
    const float *__restrict__ opacities, // [n_primitives]

    // Images
    const size_t n_images,
    const size_t image_height,
    const size_t image_width,
    const size_t tile_width,
    const size_t tile_height,

    // Isect info
    const uint32_t *__restrict__ isect_primitive_ids,       // [n_isects]
    const uint32_t *__restrict__ isect_prefix_sum_per_tile, // [n_tiles]

    // Outputs
    float *__restrict__ render_alpha // [n_images, image_height, image_width, 1]
) {
    SimplePlanerRasterizeKernelForwardOperator op{};
    op.opacity_ptr = opacities;
    op.render_alpha_ptr = render_alpha;

    dim3 threads(tile_width, tile_height, 1);
    dim3 grid(1, 1, 1);
    size_t sm_size = decltype(op)::sm_size_per_primitive() * tile_width * tile_height;
    rasterize_kernel<<<grid, threads, sm_size>>>(
        op, image_height, image_width, isect_primitive_ids, isect_prefix_sum_per_tile
    );
}

void launch_simple_planer_backward(
    // Primitives
    const size_t n_primitives,
    const float *__restrict__ opacities, // [n_primitives]

    // Images
    const size_t n_images,
    const size_t image_height,
    const size_t image_width,
    const size_t tile_width,
    const size_t tile_height,

    // Isect info
    const uint32_t *__restrict__ isect_primitive_ids,       // [n_isects]
    const uint32_t *__restrict__ isect_prefix_sum_per_tile, // [n_tiles]

    // Outputs
    const float *__restrict__ render_alpha, // [n_images, image_height, image_width, 1]

    // Gradient for outputs
    const float
        *__restrict__ v_render_alpha, // [n_images, image_height, image_width, 1]

    // Gradient for inputs
    float *__restrict__ v_opacity // [n_primitives]
) {
    SimplePlanerRasterizeKernelBackwardOperator op{};
    op.opacity_ptr = opacities;
    op.render_alpha_ptr = render_alpha;
    op.v_render_alpha_ptr = v_render_alpha;
    op.v_opacity_ptr = v_opacity;

    dim3 threads(tile_width, tile_height, 1);
    dim3 grid(1, 1, 1);
    size_t sm_size = decltype(op)::sm_size_per_primitive() * tile_width * tile_height;
    rasterize_kernel<<<grid, threads, sm_size>>>(
        op,
        image_height,
        image_width,
        isect_primitive_ids,
        isect_prefix_sum_per_tile,
        true // reverse order
    );
}

} // namespace tinyrend::rasterization