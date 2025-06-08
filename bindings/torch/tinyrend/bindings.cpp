#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h> 
#include <glm/glm.hpp>

#include "tinyrend/kernels/api.h"

torch::Tensor fisheye_project(
    torch::Tensor camera_points,
    torch::Tensor focal_lengths,
    torch::Tensor principal_points
) {
    auto n_elements = camera_points.numel() / camera_points.size(-1);
    auto image_points = torch::empty_like(focal_lengths);
    
    #define LAUNCH_KERNEL(USE_CUDA) \
        tinyrend::fisheye::project_kernel_launcher<USE_CUDA>( \
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

torch::Tensor rasterize_simple_planer_forward(
    torch::Tensor opacities, // [n_primitives]
    const size_t n_images,
    const size_t image_height,
    const size_t image_width,
    const size_t tile_width,
    const size_t tile_height,
    torch::Tensor isect_primitive_ids, // [n_isects]
    torch::Tensor isect_prefix_sum_per_tile // [n_tiles]
) {
    auto n_primitives = opacities.size(0);
    auto opt = opacities.options();
    auto render_alpha = torch::empty({n_images, image_height, image_width, 1}, opt);
    
    if (opacities.device().is_cuda()) {
        const at::cuda::OptionalCUDAGuard device_guard(opacities.device());
        tinyrend::rasterization::simple_planer_forward_kernel_launcher(
            n_primitives,
            opacities.data_ptr<float>(),
            n_images,
            image_height,
            image_width,
            tile_width,
            tile_height,
            isect_primitive_ids.data_ptr<uint32_t>(),
            isect_prefix_sum_per_tile.data_ptr<uint32_t>(),
            render_alpha.data_ptr<float>()
        );
    } else {
        throw std::runtime_error("Not implemented for CPU");
    }

    return render_alpha;
}

torch::Tensor rasterize_simple_planer_backward(
    // forward inputs
    torch::Tensor opacities, // [n_primitives]
    const size_t n_images,
    const size_t image_height,
    const size_t image_width,
    const size_t tile_width,
    const size_t tile_height,
    torch::Tensor isect_primitive_ids, // [n_isects]
    torch::Tensor isect_prefix_sum_per_tile, // [n_tiles]
    // forward outputs
    torch::Tensor render_alpha, // [n_images, image_height, image_width, 1]
    // gradient for forward outputs
    torch::Tensor v_render_alpha // [n_images, image_height, image_width, 1]
) {
    auto n_primitives = opacities.size(0);
    auto opt = opacities.options();
    auto v_opacity = torch::zeros({n_primitives}, opt);

    if (opacities.device().is_cuda()) {
        const at::cuda::OptionalCUDAGuard device_guard(opacities.device());
        tinyrend::rasterization::simple_planer_backward_kernel_launcher(
            n_primitives,
            opacities.data_ptr<float>(),
            n_images,
            image_height,
            image_width,
            tile_width,
            tile_height,
            isect_primitive_ids.data_ptr<uint32_t>(),
            isect_prefix_sum_per_tile.data_ptr<uint32_t>(),
            render_alpha.data_ptr<float>(),
            v_render_alpha.data_ptr<float>(),
            v_opacity.data_ptr<float>()
        );
    } else {
        throw std::runtime_error("Not implemented for CPU");
    }

    return v_opacity;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fisheye_project", &fisheye_project);
    m.def("rasterize_simple_planer_forward", &rasterize_simple_planer_forward);
    m.def("rasterize_simple_planer_backward", &rasterize_simple_planer_backward);
}