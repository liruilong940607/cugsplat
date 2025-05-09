#include "cugsplat/camera/fisheye.h"
#include "cugsplat/kernel_launcher.cuh"

namespace cugsplat::fisheye {

void project_kernel_launcher(
    const size_t n_elements,
    const glm::fvec3 *__restrict__ camera_points,
    const glm::fvec2 *__restrict__ focal_lengths,
    const glm::fvec2 *__restrict__ principal_points,
    glm::fvec2 *__restrict__ image_points
) {
    cugsplat::launch_linear_kernel(
        n_elements,
        [camera_points, focal_lengths, principal_points, image_points] __device__(
            size_t idx
        ) {
            image_points[idx] = cugsplat::fisheye::project(
                camera_points[idx], focal_lengths[idx], principal_points[idx]
            );
        }
    );
}

} // namespace cugsplat::fisheye