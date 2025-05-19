#include "tinyrend/camera/fisheye.h"
#include "tinyrend/core/macros.h"
#include "tinyrend/kernel_launcher.cuh"

namespace tinyrend::fisheye {

#define FISHEYE_PROJECT_SIGNATURE                                                      \
    const size_t n_elements, const glm::fvec3 *__restrict__ camera_points,             \
        const glm::fvec2 *__restrict__ focal_lengths,                                  \
        const glm::fvec2 *__restrict__ principal_points,                               \
        glm::fvec2 *__restrict__ image_points

template <bool USE_CUDA> void project_kernel_launcher(FISHEYE_PROJECT_SIGNATURE) {
    tinyrend::launch_linear_kernel<USE_CUDA>(
        n_elements,
        [camera_points,
         focal_lengths,
         principal_points,
         image_points] GSPLAT_HOST_DEVICE(size_t idx) {
            image_points[idx] = tinyrend::fisheye::project(
                camera_points[idx], focal_lengths[idx], principal_points[idx]
            );
        }
    );
}

template void project_kernel_launcher<true>(FISHEYE_PROJECT_SIGNATURE);
template void project_kernel_launcher<false>(FISHEYE_PROJECT_SIGNATURE);

} // namespace tinyrend::fisheye