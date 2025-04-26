#include <cuda_runtime.h>
#include <iostream>

#include "camera/model.h"
#include "camera/opencv_pinhole.h"
#include "primitive/gaussian.h"
#include "projection/image2dgs.h"
#include "projection/kernel.cuh"
#include "utils/types.h"

using namespace gsplat;

int main() {
    // create camera
    auto const focal_length = glm::fvec2(800.0f, 600.0f);
    auto const principal_point = glm::fvec2(400.0f, 300.0f);
    auto const world_to_camera_R = glm::fmat3(1.0f);
    auto const world_to_camera_t = glm::fvec3(0.0f);
    std::array<uint32_t, 2> resolution = {800, 600};

    auto const pose = SE3Mat{world_to_camera_t, world_to_camera_R};
    auto projector =
        BatchedOpencvPinholeProjection(1, &focal_length, &principal_point);
    auto camera = CameraModel(resolution, projector, pose);

    // create input gaussian
    auto const opacity = float(0.8f);
    auto const mean = glm::fvec3(0.0f, 0.0f, 1.0f);
    auto const quat = glm::fvec4(1.0f, 0.0f, 0.0f, 0.0f);
    auto const scale = glm::fvec3(1.0f, 1.0f, 1.0f);
    auto gaussian = BatchPrimitive3DGS(1, &opacity, &mean, &quat, &scale);

    // create operator
    OutputOperatorImage2DGS op;
    op.preprocess(camera, gaussian);
    std::cout << "Opacity: " << op.opacity << std::endl;
    std::cout << "Mean: " << op.mean.x << ", " << op.mean.y << std::endl;
    std::cout << "Conic: " << op.conic.x << ", " << op.conic.y << ", "
              << op.conic.z << std::endl;
    std::cout << "Depth: " << op.depth << std::endl;
    std::cout << "Radius: " << op.radius.x << ", " << op.radius.y << std::endl;

    // gsplat::device::DeviceSimplePinholeCameraEWA d_camera;
    // d_camera.n = 1;
    // cudaMalloc(&d_camera.focal_lengths, 1 * sizeof(glm::fvec2));
    // cudaMemcpy(d_camera.focal_lengths, &focal_length, 1 *
    // sizeof(glm::fvec2), cudaMemcpyHostToDevice);
    // cudaMalloc(&d_camera.principal_points, 1 * sizeof(glm::fvec2));
    // cudaMemcpy(d_camera.principal_points, &principal_point, 1 *
    // sizeof(glm::fvec2), cudaMemcpyHostToDevice);
    // cudaMalloc(&d_camera.world_to_cameras_R, 1 * sizeof(glm::fmat3));
    // cudaMemcpy(d_camera.world_to_cameras_R, &world_to_camera_R, 1 *
    // sizeof(glm::fmat3), cudaMemcpyHostToDevice);
    // cudaMalloc(&d_camera.world_to_cameras_t, 1 * sizeof(glm::fvec3));
    // cudaMemcpy(d_camera.world_to_cameras_t, &world_to_camera_t, 1 *
    // sizeof(glm::fvec3), cudaMemcpyHostToDevice);

    // // create primitive input
    // auto const opacity = float(0.8f);
    // auto const mean = glm::fvec3(0.0f, 0.0f, 1.0f);
    // auto const quat = glm::fvec4(1.0f, 0.0f, 0.0f, 0.0f);
    // auto const scale = glm::fvec3(1.0f, 1.0f, 1.0f);

    // gsplat::device::DevicePrimitiveInWorld3DGS d_gaussian_in;
    // d_gaussian_in.n = 1;
    // cudaMalloc(&d_gaussian_in.opacity_ptr, 1 * sizeof(float));
    // cudaMemcpy(d_gaussian_in.opacity_ptr, &opacity, 1 * sizeof(float),
    // cudaMemcpyHostToDevice); cudaMalloc(&d_gaussian_in.mean_ptr, 1 *
    // sizeof(glm::fvec3)); cudaMemcpy(d_gaussian_in.mean_ptr, &mean, 1 *
    // sizeof(glm::fvec3), cudaMemcpyHostToDevice);
    // cudaMalloc(&d_gaussian_in.quat_ptr, 1 * sizeof(glm::fvec4));
    // cudaMemcpy(d_gaussian_in.quat_ptr, &quat, 1 * sizeof(glm::fvec4),
    // cudaMemcpyHostToDevice); cudaMalloc(&d_gaussian_in.scale_ptr, 1 *
    // sizeof(glm::fvec3)); cudaMemcpy(d_gaussian_in.scale_ptr, &scale, 1 *
    // sizeof(glm::fvec3), cudaMemcpyHostToDevice);

    // // create primitive output
    // gsplat::device::DevicePrimitiveOutImage2DGS d_gaussian_out;
    // cudaMalloc(&d_gaussian_out.opacities, 1 * sizeof(float));
    // cudaMalloc(&d_gaussian_out.means, 1 * sizeof(glm::fvec2));
    // cudaMalloc(&d_gaussian_out.conics, 1 * sizeof(glm::fvec3));
    // cudaMalloc(&d_gaussian_out.depths, 1 * sizeof(float));
    // cudaMalloc(&d_gaussian_out.radius, 1 * sizeof(glm::fvec2));
    // d_gaussian_out.render_width = 800;
    // d_gaussian_out.render_height = 600;
    // d_gaussian_out.near_plane = 0.1f;
    // d_gaussian_out.far_plane = 100.0f;
    // d_gaussian_out.margin_factor = 100.0f;
    // d_gaussian_out.filter_size = 0.1f;

    // dim3 blockDim(256, 1, 1);
    // dim3 gridDim(1, 1, 1);

    // gsplat::device::PreprocessKernel<
    //     gsplat::device::DeviceSimplePinholeCameraEWA,
    //     gsplat::device::DevicePrimitiveInWorld3DGS,
    //     gsplat::device::DevicePrimitiveOutImage2DGS,
    //     false,
    //     1
    // ><<<gridDim, blockDim>>>(
    //     d_camera,
    //     d_gaussian_in,
    //     d_gaussian_out,
    //     nullptr,
    //     nullptr
    // );
    // cudaDeviceSynchronize();
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err) <<
    //     std::endl;
    // }

    // // copy output data back to host
    // float opacity_out;
    // glm::fvec2 mean_out;
    // glm::fvec3 conic_out;
    // float depth_out;
    // glm::fvec2 radius_out;
    // cudaMemcpy(&opacity_out, d_gaussian_out.opacities, sizeof(float),
    // cudaMemcpyDeviceToHost); cudaMemcpy(&mean_out, d_gaussian_out.means,
    // sizeof(glm::fvec2), cudaMemcpyDeviceToHost); cudaMemcpy(&conic_out,
    // d_gaussian_out.conics, sizeof(glm::fvec3), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&depth_out, d_gaussian_out.depths, sizeof(float),
    // cudaMemcpyDeviceToHost); cudaMemcpy(&radius_out,
    // d_gaussian_out.radius, sizeof(glm::fvec2), cudaMemcpyDeviceToHost);
    // std::cout << "Opacity: " << opacity_out << std::endl;
    // std::cout << "Mean: " << mean_out.x << ", " << mean_out.y <<
    // std::endl; std::cout << "Conic: " << conic_out.x << ", " <<
    // conic_out.y << ", " << conic_out.z << std::endl; std::cout << "Depth:
    // " << depth_out << std::endl; std::cout << "Radius: " << radius_out.x
    // << ", " << radius_out.y << std::endl;

    // // free memory
    // d_camera.free();
    // d_gaussian_out.free();

    // cudaFree(d_gaussian_in.opacity_ptr);
    // cudaFree(d_gaussian_in.mean_ptr);
    // cudaFree(d_gaussian_in.quat_ptr);
    // cudaFree(d_gaussian_in.scale_ptr);
    return 0;
}