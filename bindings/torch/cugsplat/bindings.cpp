#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h> 
#include <glm/glm.hpp>

#include "cugsplat/kernels/api.h"

torch::Tensor fisheye_project(
    torch::Tensor camera_points,
    torch::Tensor focal_lengths,
    torch::Tensor principal_points
) {
    auto n_elements = camera_points.size(0);
    auto image_points = torch::empty_like(focal_lengths);

    if (camera_points.device().is_cuda()) {
        const at::cuda::OptionalCUDAGuard device_guard(camera_points.device());
        cugsplat::fisheye::project_kernel_launcher<true>(
            n_elements,
            reinterpret_cast<const glm::fvec3*>(camera_points.data_ptr<float>()),
            reinterpret_cast<const glm::fvec2*>(focal_lengths.data_ptr<float>()),
            reinterpret_cast<const glm::fvec2*>(principal_points.data_ptr<float>()),
            reinterpret_cast<glm::fvec2*>(image_points.data_ptr<float>())
        );
    } else {
        cugsplat::fisheye::project_kernel_launcher<false>(
            n_elements,
            reinterpret_cast<const glm::fvec3*>(camera_points.data_ptr<float>()),
            reinterpret_cast<const glm::fvec2*>(focal_lengths.data_ptr<float>()),
            reinterpret_cast<const glm::fvec2*>(principal_points.data_ptr<float>()),
            reinterpret_cast<glm::fvec2*>(image_points.data_ptr<float>())
        );
    }
    return image_points;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fisheye_project", &fisheye_project);
}