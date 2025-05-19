#pragma once

#include <glm/glm.hpp>

namespace tinyrend::fisheye {

template <bool USE_CUDA>
void project_kernel_launcher(
    const size_t n_elements,
    const glm::fvec3 *__restrict__ camera_points,
    const glm::fvec2 *__restrict__ focal_lengths,
    const glm::fvec2 *__restrict__ principal_points,
    glm::fvec2 *__restrict__ image_points
);

} // namespace tinyrend::fisheye
