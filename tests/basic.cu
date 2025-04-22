#include <cuda_runtime.h>
#include <iostream>

#include "preprocess/camera/simple_pinhole_ewa.cuh"
#include "preprocess/primitive_in/world3dgs.cuh"
#include "preprocess/primitive_out/image2dgs.cuh"
#include "preprocess/kernel.cuh"

int main(){
    // create camera
    auto const focal_length = glm::fvec2(800.0f, 600.0f);
    auto const principal_point = glm::fvec2(400.0f, 300.0f);
    auto const world_to_camera_R = glm::fmat3(1.0f);
    auto const world_to_camera_t = glm::fvec3(0.0f, 0.0f, 0.0f);

    cugsplat::preprocess::DeviceSimplePinholeCameraEWA d_camera;
    d_camera.n = 1;
    cudaMalloc(&d_camera.focal_lengths, 1 * sizeof(glm::fvec2));
    cudaMemcpy(d_camera.focal_lengths, &focal_length, 1 * sizeof(glm::fvec2), cudaMemcpyHostToDevice);
    cudaMalloc(&d_camera.principal_points, 1 * sizeof(glm::fvec2));
    cudaMemcpy(d_camera.principal_points, &principal_point, 1 * sizeof(glm::fvec2), cudaMemcpyHostToDevice);
    cudaMalloc(&d_camera.world_to_cameras_R, 1 * sizeof(glm::fmat3));
    cudaMemcpy(d_camera.world_to_cameras_R, &world_to_camera_R, 1 * sizeof(glm::fmat3), cudaMemcpyHostToDevice);
    cudaMalloc(&d_camera.world_to_cameras_t, 1 * sizeof(glm::fvec3));
    cudaMemcpy(d_camera.world_to_cameras_t, &world_to_camera_t, 1 * sizeof(glm::fvec3), cudaMemcpyHostToDevice);

    // create primitive input
    auto const opacity = float(0.8f);
    auto const mean = glm::fvec3(0.0f, 0.0f, 1.0f);
    auto const quat = glm::fvec4(1.0f, 0.0f, 0.0f, 0.0f);
    auto const scale = glm::fvec3(1.0f, 1.0f, 1.0f);

    cugsplat::preprocess::DevicePrimitiveInWorld3DGS d_gaussian_in;
    d_gaussian_in.n = 1;
    cudaMalloc(&d_gaussian_in.opacities, 1 * sizeof(float));
    cudaMemcpy(d_gaussian_in.opacities, &opacity, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_gaussian_in.means, 1 * sizeof(glm::fvec3));
    cudaMemcpy(d_gaussian_in.means, &mean, 1 * sizeof(glm::fvec3), cudaMemcpyHostToDevice);
    cudaMalloc(&d_gaussian_in.quats, 1 * sizeof(glm::fvec4));
    cudaMemcpy(d_gaussian_in.quats, &quat, 1 * sizeof(glm::fvec4), cudaMemcpyHostToDevice);
    cudaMalloc(&d_gaussian_in.scales, 1 * sizeof(glm::fvec3));
    cudaMemcpy(d_gaussian_in.scales, &scale, 1 * sizeof(glm::fvec3), cudaMemcpyHostToDevice);

    // create primitive output
    cugsplat::preprocess::DevicePrimitiveOutImage2DGS d_gaussian_out;
    cudaMalloc(&d_gaussian_out.opacities, 1 * sizeof(float));
    cudaMalloc(&d_gaussian_out.means, 1 * sizeof(glm::fvec2));
    cudaMalloc(&d_gaussian_out.conics, 1 * sizeof(glm::fvec3));
    cudaMalloc(&d_gaussian_out.depths, 1 * sizeof(float));
    cudaMalloc(&d_gaussian_out.radius, 1 * sizeof(glm::fvec2));
    d_gaussian_out.render_width = 800;
    d_gaussian_out.render_height = 600;
    d_gaussian_out.near_plane = 0.1f;
    d_gaussian_out.far_plane = 100.0f;
    d_gaussian_out.margin_factor = 100.0f;
    d_gaussian_out.filter_size = 0.1f;

    dim3 blockDim(1, 1, 1);
    dim3 gridDim(1, 1, 1);

    cugsplat::preprocess::PreprocessKernel<
        cugsplat::preprocess::DeviceSimplePinholeCameraEWA,
        cugsplat::preprocess::DevicePrimitiveInWorld3DGS,
        cugsplat::preprocess::DevicePrimitiveOutImage2DGS,
        false,
        1
    ><<<gridDim, blockDim>>>(
        d_camera,
        d_gaussian_in,
        d_gaussian_out,
        nullptr,
        nullptr
    );

    // free memory
    d_camera.free();
    d_gaussian_in.free();
    d_gaussian_out.free();
    return 0;
}