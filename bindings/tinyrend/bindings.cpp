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
        tinyrend::camera::fisheye::project_kernel_launcher<USE_CUDA>( \
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

class RasterizeSimplePlanerFunction : public torch::autograd::Function<RasterizeSimplePlanerFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor opacities, // [n_primitives]
        const int64_t n_images,
        const int64_t image_height,
        const int64_t image_width,
        const int64_t tile_width,
        const int64_t tile_height,
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

        // Save tensors needed for backward
        ctx->save_for_backward({opacities, render_alpha, isect_primitive_ids, isect_prefix_sum_per_tile});
        ctx->saved_data["n_images"] = n_images;
        ctx->saved_data["image_height"] = image_height;
        ctx->saved_data["image_width"] = image_width;
        ctx->saved_data["tile_width"] = tile_width;
        ctx->saved_data["tile_height"] = tile_height;

        return render_alpha;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        // Get saved tensors and data
        auto saved = ctx->get_saved_variables();
        auto opacities = saved[0];
        auto render_alpha = saved[1];
        auto isect_primitive_ids = saved[2];
        auto isect_prefix_sum_per_tile = saved[3];
        auto n_images = ctx->saved_data["n_images"].toInt();
        auto image_height = ctx->saved_data["image_height"].toInt();
        auto image_width = ctx->saved_data["image_width"].toInt();
        auto tile_width = ctx->saved_data["tile_width"].toInt();
        auto tile_height = ctx->saved_data["tile_height"].toInt();

        auto n_primitives = opacities.size(0);
        auto opt = opacities.options();
        auto v_opacity = torch::zeros({n_primitives}, opt);
        auto v_render_alpha = grad_outputs[0];

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

        // Return gradients for all inputs that require grad
        return {v_opacity, torch::Tensor(), torch::Tensor(), torch::Tensor(), 
                torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

torch::Tensor rasterize_simple_planer(
    torch::Tensor opacities,
    const int64_t n_images,
    const int64_t image_height,
    const int64_t image_width,
    const int64_t tile_width,
    const int64_t tile_height,
    torch::Tensor isect_primitive_ids,
    torch::Tensor isect_prefix_sum_per_tile
) {
    return RasterizeSimplePlanerFunction::apply(
        opacities.contiguous(), 
        n_images, image_height, image_width,
        tile_width, tile_height, 
        isect_primitive_ids.contiguous(), isect_prefix_sum_per_tile.contiguous()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fisheye_project", &fisheye_project);
    m.def("rasterize_simple_planer", &rasterize_simple_planer);
}