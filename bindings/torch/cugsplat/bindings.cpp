#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h> 
#include <glm/glm.hpp>

#include "cugsplat/kernels/api.h"

torch::Tensor fisheye_project(
    torch::Tensor camera_points,
    torch::Tensor focal_lengths,
    torch::Tensor principal_points
) {
    auto n_elements = camera_points.numel() / camera_points.size(-1);
    auto image_points = torch::empty_like(focal_lengths);
    
    #define LAUNCH_KERNEL(USE_CUDA) \
        cugsplat::fisheye::project_kernel_launcher<USE_CUDA>( \
            n_elements, \
            reinterpret_cast<const glm::fvec3*>(camera_points.data_ptr<float>()), \
            reinterpret_cast<const glm::fvec2*>(focal_lengths.data_ptr<float>()), \
            reinterpret_cast<const glm::fvec2*>(principal_points.data_ptr<float>()), \
            reinterpret_cast<glm::fvec2*>(image_points.data_ptr<float>()))

    if (camera_points.device().is_cuda()) {
        const at::cuda::OptionalCUDAGuard device_guard(camera_points.device());
        LAUNCH_KERNEL(true);
    } else {
        LAUNCH_KERNEL(false);
    }

    #undef LAUNCH_KERNEL
    return image_points;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fisheye_project", &fisheye_project);
}