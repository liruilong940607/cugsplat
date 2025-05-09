#include "cugsplat/camera/fisheye.h"
#include "cugsplat/core/macros.h"
#include "cugsplat/kernel_launcher.cuh"

namespace cugsplat::fisheye {

template <bool USE_CUDA>
void project_kernel_launcher(
    const size_t n_elements,
    const glm::fvec3 *__restrict__ camera_points,
    const glm::fvec2 *__restrict__ focal_lengths,
    const glm::fvec2 *__restrict__ principal_points,
    glm::fvec2 *__restrict__ image_points
) {
    cugsplat::launch_linear_kernel<USE_CUDA>(
        n_elements,
        [camera_points,
         focal_lengths,
         principal_points,
         image_points] GSPLAT_HOST_DEVICE(size_t idx) {
            image_points[idx] = cugsplat::fisheye::project(
                camera_points[idx], focal_lengths[idx], principal_points[idx]
            );
        }
    );
}

template void project_kernel_launcher<true>(
    const size_t n_elements,
    const glm::fvec3 *__restrict__ camera_points,
    const glm::fvec2 *__restrict__ focal_lengths,
    const glm::fvec2 *__restrict__ principal_points,
    glm::fvec2 *__restrict__ image_points
);

template void project_kernel_launcher<false>(
    const size_t n_elements,
    const glm::fvec3 *__restrict__ camera_points,
    const glm::fvec2 *__restrict__ focal_lengths,
    const glm::fvec2 *__restrict__ principal_points,
    glm::fvec2 *__restrict__ image_points
);

} // namespace cugsplat::fisheye