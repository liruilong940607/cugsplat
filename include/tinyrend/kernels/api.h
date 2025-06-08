#pragma once

#include <glm/glm.hpp>

namespace tinyrend::camera::fisheye {

template <bool USE_CUDA>
void project_kernel_launcher(
    const size_t n_elements,
    const glm::fvec3 *__restrict__ camera_points,
    const glm::fvec2 *__restrict__ focal_lengths,
    const glm::fvec2 *__restrict__ principal_points,
    glm::fvec2 *__restrict__ image_points
);

} // namespace tinyrend::camera::fisheye

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
);

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
);

} // namespace tinyrend::rasterization
